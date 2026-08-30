import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import analyses.routing_semantics.audit as audit
from analyses.routing_semantics.audit import (
    DEFAULT_MANIFEST,
    _holm_adjusted_p_values,
    _resolve_audit_output_paths,
    _route_pair_similarities,
    _route_separation,
    _sample_route_pairs,
    _shift_routes,
    audit_route_cells,
    locked_repository_state,
    load_capture_samples,
    load_manifest,
    load_or_extract_features,
    load_route_cells,
    sha256_float32_array,
    sha256_source_tree,
    validate_locked_inputs,
    write_json_atomic,
    write_text_atomic,
)


class RoutingSemanticAuditTests(unittest.TestCase):
    def test_canonical_manifest_is_strict_and_self_consistent(self):
        manifest = load_manifest(DEFAULT_MANIFEST)
        self.assertEqual(manifest["expected"]["sample_count"], 32)
        self.assertEqual(len(manifest["expected"]["sample_latents"]), 32)
        self.assertEqual(manifest["expected"]["num_routed_experts"], 12)
        self.assertEqual(
            manifest["feature_extractor"]["attention_backend"],
            "torch_no_xformers",
        )
        self.assertRegex(
            manifest["feature_extractor"]["dino_source_revision"],
            r"^[0-9a-f]{40}$",
        )
        self.assertEqual(manifest["statistics"]["control_resamples"], 4999)
        self.assertIs(
            manifest["statistics"]["image_is_the_independent_unit"], True
        )

        malformed = copy.deepcopy(manifest)
        malformed["expected"]["unexpected"] = 1
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "manifest.json"
            path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "expected keys differ"):
                load_manifest(path)

    def test_route_cells_require_exact_cells_shape_and_expert_range(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "routes.npz"
            np.savez(
                path,
                base_sigma0_5_block1=np.zeros((2, 4), dtype=np.int16),
            )
            with self.assertRaisesRegex(ValueError, "route-cell mismatch"):
                load_route_cells(
                    path,
                    prefix="base",
                    sample_count=2,
                    token_grid_size=2,
                    expected_expert_count=2,
                    expected_blocks=[1],
                    expected_sigmas=[0.5],
                )

            np.savez(
                path,
                **{"base_sigma0.5_block1": np.full((2, 4), 2, dtype=np.int16)},
            )
            with self.assertRaisesRegex(ValueError, "outside 2 experts"):
                load_route_cells(
                    path,
                    prefix="base",
                    sample_count=2,
                    token_grid_size=2,
                    expected_expert_count=2,
                    expected_blocks=[1],
                    expected_sigmas=[0.5],
                )

    def test_dino_source_tree_hash_is_stable_and_ignores_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            source = Path(temporary_dir) / "dinov2"
            (source / "dinov2").mkdir(parents=True)
            (source / "hubconf.py").write_text("MODEL = 'base'\n", encoding="utf-8")
            model_file = source / "dinov2" / "model.py"
            model_file.write_text("WIDTH = 768\n", encoding="utf-8")
            first = sha256_source_tree(source)

            bytecode = source / "dinov2" / "__pycache__" / "model.pyc"
            bytecode.parent.mkdir()
            bytecode.write_bytes(b"ignored runtime cache")
            self.assertEqual(sha256_source_tree(source), first)

            model_file.write_text("WIDTH = 769\n", encoding="utf-8")
            self.assertNotEqual(sha256_source_tree(source), first)

    def test_capture_samples_relocate_only_inside_locked_root(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "latents"
            rows = []
            for index in range(2):
                class_dir = f"n{index:08d}"
                latent = root / class_dir / f"sample-{index}.latent.npz"
                latent.parent.mkdir(parents=True)
                latent.write_bytes(f"latent-{index}".encode("ascii"))
                rows.append(
                    {
                        "label": index,
                        "class_dir": class_dir,
                        "latent_path": f"/missing/legacy/{latent.name}",
                    }
                )
            summary = Path(temporary_dir) / "summary.json"
            summary.write_text(json.dumps({"samples": rows}), encoding="utf-8")
            observed = load_capture_samples(
                summary,
                latent_root=root,
                expected_count=2,
                expected_latents=[
                    {
                        "relative_path": f"n{index:08d}/sample-{index}.latent.npz",
                        "sha256": hashlib.sha256(
                            f"latent-{index}".encode("ascii")
                        ).hexdigest(),
                    }
                    for index in range(2)
                ],
            )
            self.assertEqual([row["index"] for row in observed], [0, 1])
            self.assertTrue(all(Path(row["latent_path"]).is_file() for row in observed))

            Path(observed[0]["latent_path"]).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "latent SHA256 mismatch"):
                load_capture_samples(
                    summary,
                    latent_root=root,
                    expected_count=2,
                    expected_latents=[
                        {
                            "relative_path": f"n{index:08d}/sample-{index}.latent.npz",
                            "sha256": hashlib.sha256(
                                f"latent-{index}".encode("ascii")
                            ).hexdigest(),
                        }
                        for index in range(2)
                    ],
                )

            rows[0]["class_dir"] = rows[1]["class_dir"]
            summary.write_text(json.dumps({"samples": rows}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "locked latent path"):
                load_capture_samples(
                    summary,
                    latent_root=root,
                    expected_count=2,
                    expected_latents=[
                        {
                            "relative_path": f"n{index:08d}/sample-{index}.latent.npz",
                            "sha256": hashlib.sha256(
                                f"latent-{index}".encode("ascii")
                            ).hexdigest(),
                        }
                        for index in range(2)
                    ],
                )

    def test_spatial_shift_matches_roll_and_preserves_each_image_counts(self):
        routes = np.arange(32, dtype=np.int64).reshape(2, 16) % 4
        shifts = np.asarray([[1, 2], [3, 1]])
        shifted = _shift_routes(routes, shifts, grid_size=4)
        for image_index in range(2):
            expected = np.roll(
                routes[image_index].reshape(4, 4),
                tuple(shifts[image_index]),
                axis=(0, 1),
            ).reshape(-1)
            np.testing.assert_array_equal(shifted[image_index], expected)
            np.testing.assert_array_equal(
                np.bincount(shifted[image_index], minlength=4),
                np.bincount(routes[image_index], minlength=4),
            )

    def test_sampled_route_separation_matches_brute_force(self):
        rng = np.random.default_rng(7)
        features = rng.normal(size=(3, 9, 6)).astype(np.float32)
        features /= np.linalg.norm(features, axis=-1, keepdims=True)
        routes = np.asarray(
            [rng.permutation(np.arange(9) % 3) for _ in range(3)],
            dtype=np.int64,
        )
        left, right = _sample_route_pairs(9, pair_count=36, seed=11)
        similarities = _route_pair_similarities(features, left, right)
        observed = _route_separation(routes, left, right, similarities)
        expected = []
        for image_index in range(3):
            same = routes[image_index, left] == routes[image_index, right]
            expected.append(
                similarities[image_index, same].mean()
                - similarities[image_index, ~same].mean()
            )
        np.testing.assert_allclose(observed, expected, rtol=1e-12, atol=1e-12)

    def test_feature_cache_is_bound_to_protocol_and_locked_inputs(self):
        samples = [
            {"latent_sha256": hashlib.sha256(str(index).encode()).hexdigest()}
            for index in range(2)
        ]
        locked_inputs = {
            "route_ids": {"sha256": "1" * 64},
            "dino_state_dict": {"sha256": "2" * 64},
        }
        runtime = {
            "software_versions": {"python": "test"},
            "device": "cpu",
        }
        features = np.ones((2, 4, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as temporary_dir:
            cache = Path(temporary_dir) / "features.npz"
            victim = Path(temporary_dir) / "victim.txt"
            victim.write_text("unchanged\n", encoding="utf-8")
            legacy_temporary = cache.with_name(
                f".{cache.name}.{audit.os.getpid()}.tmp"
            )
            legacy_temporary.symlink_to(victim)
            with patch.object(audit, "extract_dino_features", return_value=features):
                first, first_cache = load_or_extract_features(
                    cache,
                    samples,
                    vae_path=Path(temporary_dir) / "vae",
                    dino_path=Path(temporary_dir) / "dino.pth",
                    dino_source_path=Path(temporary_dir) / "dinov2-source",
                    device="cpu",
                    batch_size=2,
                    latent_key="latent",
                    latent_shape=(8, 32, 32),
                    locked_inputs=locked_inputs,
                    protocol_sha256="3" * 64,
                    repository_commit="a" * 40,
                    dino_source_revision="b" * 40,
                    dino_source_tree_sha256="c" * 64,
                    runtime=runtime,
                )
            np.testing.assert_allclose(first, np.full_like(features, 1 / np.sqrt(3)))
            self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged\n")
            self.assertFalse(cache.is_symlink())
            self.assertFalse(first_cache["cache_hit"])
            self.assertEqual(first_cache["generation_runtime"], runtime)
            with patch.object(
                audit,
                "extract_dino_features",
                side_effect=AssertionError("cache should be reused"),
            ):
                second, second_cache = load_or_extract_features(
                    cache,
                    samples,
                    vae_path=Path(temporary_dir) / "vae",
                    dino_path=Path(temporary_dir) / "dino.pth",
                    dino_source_path=Path(temporary_dir) / "dinov2-source",
                    device="cpu",
                    batch_size=2,
                    latent_key="latent",
                    latent_shape=(8, 32, 32),
                    locked_inputs=locked_inputs,
                    protocol_sha256="3" * 64,
                    repository_commit="a" * 40,
                    dino_source_revision="b" * 40,
                    dino_source_tree_sha256="c" * 64,
                    runtime=runtime,
                )
            np.testing.assert_array_equal(first, second)
            self.assertEqual(sha256_float32_array(first), first_cache["feature_sha256"])
            self.assertEqual(sha256_float32_array(second), second_cache["feature_sha256"])
            self.assertTrue(second_cache["cache_hit"])
            self.assertEqual(second_cache["generation_runtime"], runtime)

            with self.assertRaisesRegex(ValueError, "cache identity"):
                load_or_extract_features(
                    cache,
                    samples,
                    vae_path=Path(temporary_dir) / "vae",
                    dino_path=Path(temporary_dir) / "dino.pth",
                    dino_source_path=Path(temporary_dir) / "dinov2-source",
                    device="cuda:7",
                    batch_size=2,
                    latent_key="latent",
                    latent_shape=(8, 32, 32),
                    locked_inputs=locked_inputs,
                    protocol_sha256="3" * 64,
                    repository_commit="a" * 40,
                    dino_source_revision="b" * 40,
                    dino_source_tree_sha256="c" * 64,
                    runtime={**runtime, "device": "cuda:7"},
                )

            with self.assertRaisesRegex(ValueError, "cache identity"):
                load_or_extract_features(
                    cache,
                    samples,
                    vae_path=Path(temporary_dir) / "vae",
                    dino_path=Path(temporary_dir) / "dino.pth",
                    dino_source_path=Path(temporary_dir) / "dinov2-source",
                    device="cpu",
                    batch_size=2,
                    latent_key="latent",
                    latent_shape=(8, 32, 32),
                    locked_inputs=locked_inputs,
                    protocol_sha256="4" * 64,
                    repository_commit="a" * 40,
                    dino_source_revision="b" * 40,
                    dino_source_tree_sha256="c" * 64,
                    runtime=runtime,
                )

            with np.load(cache, allow_pickle=False) as archive:
                tampered = {key: np.asarray(archive[key]).copy() for key in archive.files}
            tampered["features"][0, 0, 0] += 0.5
            np.savez_compressed(cache, **tampered)
            with self.assertRaisesRegex(ValueError, "content SHA256"):
                load_or_extract_features(
                    cache,
                    samples,
                    vae_path=Path(temporary_dir) / "vae",
                    dino_path=Path(temporary_dir) / "dino.pth",
                    dino_source_path=Path(temporary_dir) / "dinov2-source",
                    device="cpu",
                    batch_size=2,
                    latent_key="latent",
                    latent_shape=(8, 32, 32),
                    locked_inputs=locked_inputs,
                    protocol_sha256="3" * 64,
                    repository_commit="a" * 40,
                    dino_source_revision="b" * 40,
                    dino_source_tree_sha256="c" * 64,
                    runtime=runtime,
                )

    def test_locked_inputs_verify_every_large_artifact_digest(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths = {
                "route_ids": root / "routes.npz",
                "capture_summary": root / "capture.json",
                "dino_state_dict": root / "dino.pth",
            }
            for name, path in paths.items():
                path.write_bytes(name.encode("ascii"))
            dino_source = root / "dinov2-source"
            dino_source.mkdir()
            (dino_source / "hubconf.py").write_bytes(b"locked source")
            vae = root / "vae"
            vae.mkdir()
            (vae / "config.json").write_bytes(b"vae-config")
            (vae / "diffusion_pytorch_model.safetensors").write_bytes(b"vae-weights")
            manifest = load_manifest(DEFAULT_MANIFEST)
            expected = manifest["expected"]
            expected["route_ids_sha256"] = hashlib.sha256(
                paths["route_ids"].read_bytes()
            ).hexdigest()
            expected["capture_summary_sha256"] = hashlib.sha256(
                paths["capture_summary"].read_bytes()
            ).hexdigest()
            expected["dino_state_dict_sha256"] = hashlib.sha256(
                paths["dino_state_dict"].read_bytes()
            ).hexdigest()
            expected["dino_source_tree_sha256"] = sha256_source_tree(dino_source)
            expected["vae_config_sha256"] = hashlib.sha256(b"vae-config").hexdigest()
            expected["vae_state_dict_sha256"] = hashlib.sha256(b"vae-weights").hexdigest()
            records = validate_locked_inputs(
                manifest,
                route_ids_path=paths["route_ids"],
                capture_summary_path=paths["capture_summary"],
                dino_path=paths["dino_state_dict"],
                dino_source_path=dino_source,
                vae_path=vae,
            )
            self.assertEqual(set(records), {
                "route_ids",
                "capture_summary",
                "dino_state_dict",
                "dino_source_tree",
                "vae_config",
                "vae_state_dict",
            })
            paths["dino_state_dict"].write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "dino_state_dict SHA256 mismatch"):
                validate_locked_inputs(
                    manifest,
                    route_ids_path=paths["route_ids"],
                    capture_summary_path=paths["capture_summary"],
                    dino_path=paths["dino_state_dict"],
                    dino_source_path=dino_source,
                    vae_path=vae,
                )

    def test_output_paths_cannot_overlap_inputs_or_each_other(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            capture_dir = root / "capture"
            capture_dir.mkdir()
            capture_summary = capture_dir / "summary.json"
            capture_summary.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "collides with a locked input"):
                _resolve_audit_output_paths(
                    capture_dir,
                    None,
                    protected_paths=[capture_summary],
                    protected_roots=[],
                )

            output_dir = root / "output"
            with self.assertRaisesRegex(ValueError, "must be distinct"):
                _resolve_audit_output_paths(
                    output_dir,
                    output_dir / "summary.md",
                    protected_paths=[],
                    protected_roots=[],
                )

            with self.assertRaisesRegex(
                ValueError, "must not be ancestors or descendants"
            ):
                _resolve_audit_output_paths(
                    output_dir,
                    output_dir,
                    protected_paths=[],
                    protected_roots=[],
                )

            with self.assertRaisesRegex(
                ValueError, "must not be ancestors or descendants"
            ):
                _resolve_audit_output_paths(
                    output_dir,
                    output_dir / "summary.json" / "cache.npz",
                    protected_paths=[],
                    protected_roots=[],
                )

            blocking_file = root / "not-a-directory"
            blocking_file.write_text("blocked\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-directory existing ancestor"):
                _resolve_audit_output_paths(
                    blocking_file / "results",
                    None,
                    protected_paths=[],
                    protected_roots=[],
                )
            with self.assertRaisesRegex(ValueError, "non-directory existing ancestor"):
                _resolve_audit_output_paths(
                    output_dir,
                    blocking_file / "cache.npz",
                    protected_paths=[],
                    protected_roots=[],
                )

            source_root = root / "source"
            source_root.mkdir()
            with self.assertRaisesRegex(ValueError, "protected input/source tree"):
                _resolve_audit_output_paths(
                    source_root / "results",
                    None,
                    protected_paths=[],
                    protected_roots=[source_root],
                )

    def test_atomic_writers_ignore_legacy_predictable_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            victim = root / "victim.txt"
            victim.write_text("unchanged\n", encoding="utf-8")

            json_path = root / "summary.json"
            json_legacy = json_path.with_name(
                f".{json_path.name}.{audit.os.getpid()}.tmp"
            )
            json_legacy.symlink_to(victim)
            write_json_atomic(json_path, {"value": 1})
            self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged\n")
            self.assertEqual(
                json.loads(json_path.read_text(encoding="utf-8")),
                {"value": 1},
            )
            self.assertFalse(json_path.is_symlink())

            text_path = root / "summary.md"
            text_legacy = text_path.with_name(
                f".{text_path.name}.{audit.os.getpid()}.tmp"
            )
            text_legacy.symlink_to(victim)
            write_text_atomic(text_path, "complete\n")
            self.assertEqual(victim.read_text(encoding="utf-8"), "unchanged\n")
            self.assertEqual(text_path.read_text(encoding="utf-8"), "complete\n")
            self.assertFalse(text_path.is_symlink())

    def test_repository_gate_requires_clean_pushed_head(self):
        values = {
            ("rev-parse", "HEAD"): "a" * 40,
            ("branch", "--show-current"): "analysis/test",
            (
                "rev-parse",
                "--abbrev-ref",
                "--symbolic-full-name",
                "@{upstream}",
            ): "origin/analysis/test",
            ("rev-parse", "@{upstream}"): "a" * 40,
            (
                "config",
                "--local",
                "--get-all",
                "remote.origin.url",
            ): audit.AUTHORITATIVE_REMOTE_URL,
            ("config", "--local", "branch.analysis/test.remote"): "origin",
            (
                "config",
                "--local",
                "branch.analysis/test.merge",
            ): "refs/heads/analysis/test",
        }

        def git_output(*args):
            return values[args]

        with (
            patch.object(audit, "_git_output", side_effect=git_output),
            patch.object(audit, "reject_history_overrides") as reject_history,
            patch.object(audit, "reject_index_overrides") as reject_index,
            patch.object(audit, "fresh_worktree_status", return_value=""),
            patch.object(audit, "authoritative_remote_tip", return_value="a" * 40),
        ):
            state = locked_repository_state()
        self.assertTrue(state["status_clean"])
        reject_history.assert_called_once_with(audit.PROJECT_ROOT)
        reject_index.assert_called_once_with(audit.PROJECT_ROOT)

        with (
            patch.object(audit, "reject_history_overrides"),
            patch.object(audit, "reject_index_overrides"),
            patch.object(audit, "fresh_worktree_status", return_value="?? new.py\n"),
        ):
            with self.assertRaisesRegex(RuntimeError, "clean committed worktree"):
                locked_repository_state()

        values[("rev-parse", "@{upstream}")] = "b" * 40
        with (
            patch.object(audit, "_git_output", side_effect=git_output),
            patch.object(audit, "reject_history_overrides"),
            patch.object(audit, "reject_index_overrides"),
            patch.object(audit, "fresh_worktree_status", return_value=""),
        ):
            with self.assertRaisesRegex(RuntimeError, "not the pushed upstream"):
                locked_repository_state()

        values[("rev-parse", "@{upstream}")] = "a" * 40
        with (
            patch.object(audit, "_git_output", side_effect=git_output),
            patch.object(audit, "reject_history_overrides"),
            patch.object(audit, "reject_index_overrides"),
            patch.object(audit, "fresh_worktree_status", return_value=""),
            patch.object(audit, "authoritative_remote_tip", return_value="b" * 40),
        ):
            with self.assertRaisesRegex(RuntimeError, "live remote commit"):
                locked_repository_state()

    def test_holm_adjustment_controls_the_complete_metric_family(self):
        raw = np.asarray([0.001, 0.01, 0.03, 0.2])
        adjusted = _holm_adjusted_p_values(raw)
        np.testing.assert_allclose(adjusted, [0.004, 0.03, 0.06, 0.2])

    def test_independent_semantic_routes_pass_locked_controls(self):
        rng = np.random.default_rng(20260830)
        image_count = 8
        grid_size = 4
        expert_count = 4
        routes = []
        features = []
        labels = np.repeat(np.arange(expert_count), 4)
        for _ in range(image_count):
            image_labels = rng.permutation(labels)
            routes.append(image_labels)
            image_features = np.eye(expert_count, dtype=np.float32)[image_labels]
            image_features += rng.normal(0, 1e-4, image_features.shape)
            features.append(image_features)
        result = audit_route_cells(
            np.asarray(features),
            {(0.5, 1): np.asarray(routes, dtype=np.int64)},
            grid_size=grid_size,
            expected_expert_count=expert_count,
            cross_image_token_stride=1,
            route_separation_pair_count=100,
            control_resamples=99,
            bootstrap_resamples=500,
            seed=19,
            gates={
                "minimum_within_image_knn_delta": 0.2,
                "minimum_cross_image_knn_delta": 0.2,
                "minimum_route_separation_delta": 0.2,
                "maximum_one_sided_control_p": 0.05,
                "minimum_passing_cells": 1,
                "minimum_passing_blocks": 1,
                "minimum_passing_sigmas": 1,
            },
        )
        self.assertTrue(result["independent_semantic_structure_supported"])
        self.assertTrue(result["cells"][0]["passed"])
        self.assertEqual(result["multiple_testing"]["family_size"], 3)
        self.assertEqual(
            result["cells"][0]["cross_image_dino_knn"]["correct_minus_control"][
                "method"
            ],
            "two_way_query_gallery_image_cluster_with_control_draw",
        )


if __name__ == "__main__":
    unittest.main()
