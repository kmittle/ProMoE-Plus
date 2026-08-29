import copy
import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

import analyses.fresh_base_routing.audit as audit
from analyses.fresh_base_routing.audit import (
    CANONICAL_CONFIG_SHA256,
    CANONICAL_TRAINING_CONFIG_SHA256,
    CHECKPOINT_STEPS,
    CANONICAL_MANIFEST_SHA256,
    DEFAULT_MANIFEST,
    _expected_manifest_cases,
    _dataset_identity_from_latent_root,
    _fresh_training_log_snapshot,
    _git_contract,
    _load_recomputed_stage_summary,
    _load_protocol,
    _mkdir_secure,
    _optimizer_state_contract,
    _output_dir,
    _plumbing_summary,
    _read_archive_bytes,
    _stage_cases,
    _write_sealed_summary,
    _verify_rebuilt_protocol_contract,
    _summary_payload,
    _canonical_manifest_path,
    _trainer_state_contract,
    _validate_run_dir,
    _verify_canonical_manifest_binding,
    _verify_checkpoint_contracts,
    _verify_output_dir_contract,
    _verify_training_log,
    longitudinal_decision,
)
from train import _capture_rng_state


def _gate(routing=True, stage=True, safety=True):
    return {
        "routing_accuracy_gap_passed": routing,
        "stage_structure_passed": stage,
        "safety_passed": safety,
    }


def _training_provenance():
    source_sha256 = {
        relative: hashlib.sha256(
            (audit.PROJECT_ROOT / relative).read_bytes()
        ).hexdigest()
        for relative in audit.LOCKED_TRAINING_SOURCE_PATHS
    }
    return {
        "version": 1,
        "strict": True,
        "git": {
            "commit": "a" * 40,
            "origin_repa_commit": "a" * 40,
            "status_clean": True,
            "origin_repa_divergence": "0\t0",
        },
        "config": {
            "version": 1,
            "basename": f"{audit.CONFIG_STEM}.yaml",
            "payload_sha256": CANONICAL_TRAINING_CONFIG_SHA256,
        },
        "source_sha256": source_sha256,
        "environment": {"cuda_devices": {}},
    }


def _checkpoint_rng_state():
    state = _capture_rng_state()
    if "cuda" not in state:
        state["cuda"] = torch.get_rng_state()
    return state


class FreshBaseRoutingAuditTests(unittest.TestCase):
    def test_git_contract_rejects_dirty_temporary_index_status(self):
        completed = type('Completed', (), {'stdout': '0\t0\n'})()
        state = {
            'commit': 'a' * 40,
            'origin_repa': 'a' * 40,
            'authoritative_remote_tip': 'a' * 40,
            'status': '?? untracked.py\n',
        }
        with (
            patch('analyses.fresh_base_routing.audit.subprocess.run') as run,
            patch(
                'analyses.fresh_base_routing.audit.repository_state',
                return_value=state,
            ),
        ):
            run.return_value = completed
            with self.assertRaisesRegex(RuntimeError, 'clean committed tree'):
                _git_contract()

    def test_git_contract_rejects_unpushed_authoritative_tip(self):
        completed = type('Completed', (), {'stdout': '0\t0\n'})()
        state = {
            'commit': 'a' * 40,
            'origin_repa': 'a' * 40,
            'authoritative_remote_tip': 'b' * 40,
            'status': '',
        }
        with patch(
            'analyses.fresh_base_routing.audit.subprocess.run',
            return_value=completed,
        ), patch(
            'analyses.fresh_base_routing.audit.repository_state',
            return_value=state,
        ):
            with self.assertRaisesRegex(RuntimeError, 'pushed to origin/repa'):
                _git_contract()

    def test_longitudinal_decision_requires_primary_and_two_earlier_steps(self):
        gates = {str(step): _gate() for step in CHECKPOINT_STEPS}
        decision = longitudinal_decision(gates)
        self.assertTrue(decision["routing_gap_supported"])
        self.assertTrue(decision["phase_structure_supported"])

        gates["50000"] = _gate(routing=False, stage=False)
        gates["100000"] = _gate(routing=False, stage=False)
        decision = longitudinal_decision(gates)
        self.assertFalse(decision["routing_gap_supported"])
        self.assertFalse(decision["phase_structure_supported"])

    def test_longitudinal_decision_rejects_primary_or_safety_failure(self):
        gates = {str(step): _gate() for step in CHECKPOINT_STEPS}
        gates["200000"] = _gate(routing=False)
        self.assertFalse(longitudinal_decision(gates)["routing_gap_supported"])

        gates = {str(step): _gate() for step in CHECKPOINT_STEPS}
        gates["50000"] = _gate(safety=False)
        decision = longitudinal_decision(gates)
        self.assertFalse(decision["routing_gap_supported"])
        self.assertFalse(decision["phase_structure_supported"])

    def test_longitudinal_decision_rejects_colliding_checkpoint_keys(self):
        gates = {str(step): _gate() for step in CHECKPOINT_STEPS}
        gates[CHECKPOINT_STEPS[0]] = _gate()
        with self.assertRaisesRegex(ValueError, "collide"):
            longitudinal_decision(gates)

    def test_longitudinal_decision_requires_exact_boolean_fields(self):
        for invalid_value in (1, "true"):
            with self.subTest(invalid_value=invalid_value):
                gates = {str(step): _gate() for step in CHECKPOINT_STEPS}
                gates[str(CHECKPOINT_STEPS[0])][
                    "routing_accuracy_gap_passed"
                ] = invalid_value
                with self.assertRaisesRegex(ValueError, "exact booleans"):
                    longitudinal_decision(gates)

    def test_optimizer_state_matches_locked_parameter_contract(self):
        parameter_specs = (
            {
                "shape": (2, 3),
                "dtype": torch.float32,
                "layout": torch.strided,
            },
        )
        optimizer_state = {
            "param_groups": [{"params": [0], "amsgrad": False}],
            "state": {
                0: {
                    "step": torch.tensor(6.0),
                    "exp_avg": torch.zeros(2, 3),
                    "exp_avg_sq": torch.ones(2, 3),
                }
            },
        }
        self.assertEqual(
            _optimizer_state_contract(optimizer_state, 5, parameter_specs),
            {
                "num_parameter_groups": 1,
                "num_parameters": 1,
                "optimizer_step": 6,
            },
        )

        invalid_moments = {
            "shape": torch.zeros(3, 2),
            "dtype": torch.zeros(2, 3, dtype=torch.float64),
            "nan": torch.full((2, 3), float("nan")),
            "inf": torch.full((2, 3), float("inf")),
        }
        for name, moment in invalid_moments.items():
            with self.subTest(name=name):
                malformed = copy.deepcopy(optimizer_state)
                malformed["state"][0]["exp_avg"] = moment
                with self.assertRaisesRegex(
                    ValueError,
                    "differs from the locked model|non-finite",
                ):
                    _optimizer_state_contract(malformed, 5, parameter_specs)

    def test_manifest_selection_is_deterministic_and_split_locked(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            class_dirs = []
            for label in range(40):
                class_dir = root / f"n{label:08d}"
                class_dir.mkdir()
                (class_dir / f"n{label:08d}_1.latent.npz").touch()
                class_dirs.append(class_dir)
            selection = {
                "salt": "unit-test-salt",
                "excluded_labels": [0, 1, 2, 3],
            }
            first = _expected_manifest_cases(selection, class_dirs)
            second = _expected_manifest_cases(selection, class_dirs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)
        self.assertEqual(
            [row["split"] for row in first].count("plumbing"), 4
        )
        self.assertEqual(
            [row["split"] for row in first].count("discovery"), 8
        )
        self.assertEqual(
            [row["split"] for row in first].count("confirmatory"), 24
        )

    def test_plumbing_rejects_any_nonzero_noop(self):
        rows = {
            step: [object(), object(), object(), object()]
            for step in CHECKPOINT_STEPS
        }
        passing = {
            "safety": {
                "noop_abs_mse_change": 0.0,
                "noop_abs_output_change": 0.0,
                "forced_unforced_abs_mse_change": 0.0,
                "forced_unforced_abs_output_change": 0.0,
                "joint_native_abs_mse_change": 0.0,
                "joint_native_abs_output_change": 0.0,
            },
            "native_capacity_counts_match": True,
        }
        with patch(
            "analyses.fresh_base_routing.audit._case_metrics",
            return_value=passing,
        ):
            self.assertTrue(_plumbing_summary(rows)["passed"])

        failing = {
            **passing,
            "safety": {**passing["safety"], "noop_abs_output_change": 1e-7},
        }
        with patch(
            "analyses.fresh_base_routing.audit._case_metrics",
            return_value=failing,
        ):
            self.assertFalse(_plumbing_summary(rows)["passed"])

    def test_plumbing_rejects_nonfinite_safety_values(self):
        rows = {
            step: [object(), object(), object(), object()]
            for step in CHECKPOINT_STEPS
        }
        metrics = {
            "safety": {
                "noop_abs_mse_change": float("nan"),
                "noop_abs_output_change": 0.0,
                "forced_unforced_abs_mse_change": 0.0,
                "forced_unforced_abs_output_change": 0.0,
                "joint_native_abs_mse_change": 0.0,
                "joint_native_abs_output_change": 0.0,
            },
            "native_capacity_counts_match": True,
        }
        with patch(
            "analyses.fresh_base_routing.audit._case_metrics",
            return_value=metrics,
        ):
            with self.assertRaisesRegex(ValueError, "Non-finite"):
                _plumbing_summary(rows)

    def test_plumbing_requires_exact_checkpoint_key_set(self):
        passing = {
            "safety": {
                "noop_abs_mse_change": 0.0,
                "noop_abs_output_change": 0.0,
            },
            "native_capacity_counts_match": True,
        }
        rows = {step: [object()] for step in CHECKPOINT_STEPS}
        with patch(
            "analyses.fresh_base_routing.audit._case_metrics",
            return_value=passing,
        ):
            with self.assertRaisesRegex(ValueError, "exactly all locked"):
                _plumbing_summary({str(CHECKPOINT_STEPS[0]): rows[CHECKPOINT_STEPS[0]]})
            extra = {**rows, 999999: [object()]}
            with self.assertRaisesRegex(ValueError, "exactly all locked"):
                _plumbing_summary(extra)
            duplicate = {**rows, str(CHECKPOINT_STEPS[0]): [object()]}
            with self.assertRaisesRegex(ValueError, "collide"):
                _plumbing_summary(duplicate)

    def test_mkdir_secure_accepts_concurrent_creator_race(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            target = root / "case" / "step"
            original_mkdir = os.mkdir
            raced = {"done": False}

            def mkdir_with_race(path, mode=0o777, *, dir_fd=None):
                if path == "case" and not raced["done"]:
                    raced["done"] = True
                    original_mkdir(path, mode=mode, dir_fd=dir_fd)
                    raise FileExistsError(path)
                return original_mkdir(path, mode=mode, dir_fd=dir_fd)

            with patch(
                "analyses.fresh_base_routing.audit.os.mkdir",
                new=mkdir_with_race,
            ):
                self.assertEqual(_mkdir_secure(target, root), target)
            self.assertTrue(raced["done"])
            self.assertTrue(target.is_dir())

    def test_mkdir_secure_rejects_symlink_inserted_after_creation_attempt(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "root"
            outside = Path(temporary_dir) / "outside"
            root.mkdir()
            outside.mkdir()
            target = root / "case" / "step"
            original_mkdir = __import__("os").mkdir
            raced = {"done": False}

            def mkdir_with_symlink(path, mode=0o777, *, dir_fd=None):
                if path == "case" and not raced["done"]:
                    raced["done"] = True
                    (root / "case").symlink_to(outside, target_is_directory=True)
                    raise FileExistsError(path)
                return original_mkdir(path, mode=mode, dir_fd=dir_fd)

            with patch("analyses.fresh_base_routing.audit.os.mkdir", new=mkdir_with_symlink):
                with self.assertRaisesRegex(ValueError, "Symlink directory"):
                    _mkdir_secure(target, root)
            self.assertFalse((outside / "step").exists())

    def test_archive_read_rejects_replaced_directory_component(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive_root = Path(temporary_dir) / "archive"
            output_dir = archive_root / "audit"
            outside = Path(temporary_dir) / "outside"
            output_dir.mkdir(parents=True)
            outside.mkdir()
            payload = output_dir / "payload.json"
            payload.write_text("{}\n", encoding="utf-8")
            original_open = os.open
            replaced = {"done": False}

            def open_with_replacement(path, flags, mode=0o777, *, dir_fd=None):
                if path == "audit" and dir_fd is not None and not replaced["done"]:
                    replaced["done"] = True
                    output_dir.rename(archive_root / "audit-original")
                    output_dir.symlink_to(outside, target_is_directory=True)
                return original_open(path, flags, mode, dir_fd=dir_fd)

            with patch.object(audit, "ARCHIVE_ROOT", archive_root), patch(
                "analyses.fresh_base_routing.audit.os.open",
                new=open_with_replacement,
            ):
                with self.assertRaisesRegex(ValueError, "Symlink directory"):
                    _read_archive_bytes(payload, output_dir, "Test payload")
            self.assertTrue(replaced["done"])

    def test_protocol_reader_rejects_symlink_files(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive_root = Path(temporary_dir) / "archive"
            output_dir = archive_root / "audit"
            output_dir.mkdir(parents=True)
            target = archive_root / "outside-protocol.json"
            target.write_text("{}\n", encoding="utf-8")
            (output_dir / "protocol.json").symlink_to(target)
            (output_dir / "protocol.sha256").write_text(
                "0" * 64 + "\n",
                encoding="ascii",
            )
            with patch.object(audit, "ARCHIVE_ROOT", archive_root):
                with self.assertRaisesRegex(ValueError, "Symlink"):
                    _load_protocol(output_dir)

    def test_audit_lock_rejects_symlink_file(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive_root = Path(temporary_dir) / "archive"
            output_dir = archive_root / "audit"
            output_dir.mkdir(parents=True)
            target = archive_root / "outside-lock"
            target.touch()
            (output_dir / ".fresh-base-routing-audit.lock").symlink_to(target)
            with patch.object(audit, "ARCHIVE_ROOT", archive_root):
                with self.assertRaisesRegex(ValueError, "Symlink"):
                    with audit._audit_lock(output_dir):
                        self.fail("Symlink lock unexpectedly opened")

    def test_stage_cases_rejects_non_mapping_entries(self):
        protocol = {"manifest": {"cases": [None]}}
        with self.assertRaisesRegex(ValueError, "must all be mappings"):
            _stage_cases(protocol, "plumbing")

    def test_recomputed_summary_rejects_forged_summary_and_seal_pair(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            archive_root = Path(temporary_dir) / "archive"
            output_dir = archive_root / "audit"
            archive_root.mkdir()
            protocol_sha256 = "a" * 64
            forged = {
                "audit_version": audit.AUDIT_VERSION,
                "protocol_sha256": protocol_sha256,
                "stage": "plumbing",
                "case_ids": ["case-1"],
                "gate": {"passed": True},
            }
            recomputed = copy.deepcopy(forged)
            recomputed["gate"]["passed"] = False
            cases = [{"id": "case-1", "split": "plumbing"}]
            with patch.object(audit, "ARCHIVE_ROOT", archive_root):
                _write_sealed_summary(
                    output_dir,
                    "plumbing",
                    forged,
                    protocol_sha256,
                )
                with patch.object(
                    audit,
                    "_stage_cases",
                    return_value=cases,
                ), patch.object(
                    audit,
                    "_load_stage_results",
                    return_value={},
                ), patch.object(
                    audit,
                    "_build_stage_summary",
                    return_value=recomputed,
                ):
                    with self.assertRaisesRegex(
                        RuntimeError,
                        "differs from sealed case-result recomputation",
                    ):
                        _load_recomputed_stage_summary(
                            output_dir,
                            "plumbing",
                            {},
                            protocol_sha256,
                        )

    def test_protocol_rebuild_rejects_pairwise_tampering(self):
        protocol = {
            "run": {"path": "/run"},
            "manifest": {"latent_root": "/latent"},
            "settings": {"devices": ["cuda:0", "cuda:1"]},
            "scope": "original",
        }
        canonical_manifest = copy.deepcopy(protocol["manifest"])
        output_dir = Path("/archive/audit")
        run_dir = Path("/run")
        with patch(
            "analyses.fresh_base_routing.audit._build_protocol",
            return_value={**protocol, "scope": "tampered"},
        ) as rebuild:
            with self.assertRaisesRegex(RuntimeError, "current run/config/source"):
                _verify_rebuilt_protocol_contract(
                    protocol,
                    canonical_manifest,
                    output_dir,
                    run_dir,
                )
            args = rebuild.call_args.args[0]
            self.assertEqual(args.run_dir, run_dir)
            self.assertEqual(args.latent_root, "/latent")
            self.assertEqual(args.devices, ("cuda:0", "cuda:1"))

    def test_output_must_stay_in_project_analysis_archive(self):
        with self.assertRaisesRegex(ValueError, "must stay under"):
            _output_dir("/tmp/fresh-base-routing-audit")

    def test_only_canonical_manifest_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "canonical manifest"):
                _canonical_manifest_path(Path(temporary_dir) / "manifest.json")

    def test_protocol_manifest_is_reloaded_from_canonical_path_and_hash(self):
        canonical = {
            "path": str(DEFAULT_MANIFEST.resolve()),
            "sha256": CANONICAL_MANIFEST_SHA256,
            "latent_root": "/tmp/latent-root",
            "selection": {"locked": True},
            "cases": [],
        }
        with patch(
            "analyses.fresh_base_routing.audit._verify_file"
        ) as verify_file, patch(
            "analyses.fresh_base_routing.audit.load_manifest",
            return_value=canonical,
        ) as load_manifest:
            self.assertEqual(
                _verify_canonical_manifest_binding(canonical), canonical
            )
            verify_file.assert_called_once_with(
                DEFAULT_MANIFEST.resolve(),
                CANONICAL_MANIFEST_SHA256,
                "Canonical audit manifest",
            )
            load_manifest.assert_called_once_with(
                DEFAULT_MANIFEST.resolve(), "/tmp/latent-root"
            )

        tampered = {**canonical, "sha256": "0" * 64}
        with self.assertRaisesRegex(ValueError, "canonical manifest SHA256"):
            _verify_canonical_manifest_binding(tampered)

    def test_output_dir_must_match_sealed_protocol(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            with self.assertRaisesRegex(ValueError, "output directory"):
                _verify_output_dir_contract(
                    Path(temporary_dir),
                    {"output_dir": "/tmp/another-audit"},
                )

    def test_run_dir_accepts_registered_output_symlink(self):
        import analyses.fresh_base_routing.audit as audit

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            lexical_outputs = root / "outputs"
            external_outputs = root / "run-disk"
            target = (
                external_outputs
                / audit.MODEL_NAME
                / audit.CONFIG_STEM
            )
            target.mkdir(parents=True)
            link = lexical_outputs / audit.MODEL_NAME / audit.CONFIG_STEM
            link.parent.mkdir(parents=True)
            link.symlink_to(target, target_is_directory=True)
            with patch.object(audit, "OUTPUT_ROOT", lexical_outputs), patch.object(
                audit,
                "ALLOWED_EXTERNAL_OUTPUT_ROOTS",
                (external_outputs,),
            ):
                self.assertEqual(
                    _validate_run_dir(link, output_root=lexical_outputs),
                    link,
                )

    def test_run_dir_rejects_unregistered_output_symlink_target(self):
        import analyses.fresh_base_routing.audit as audit

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            lexical_outputs = root / "outputs"
            outside = root / "outside"
            target = outside / audit.MODEL_NAME / audit.CONFIG_STEM
            target.mkdir(parents=True)
            link = lexical_outputs / audit.MODEL_NAME / audit.CONFIG_STEM
            link.parent.mkdir(parents=True)
            link.symlink_to(target, target_is_directory=True)
            with patch.object(audit, "OUTPUT_ROOT", lexical_outputs), patch.object(
                audit,
                "ALLOWED_EXTERNAL_OUTPUT_ROOTS",
                (),
            ):
                with self.assertRaisesRegex(ValueError, "symlink target"):
                    _validate_run_dir(link, output_root=lexical_outputs)

    def test_symlinked_output_root_does_not_authorize_its_target(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            outside = root / "outside"
            target = outside / audit.MODEL_NAME / audit.CONFIG_STEM
            target.mkdir(parents=True)
            lexical_outputs = root / "outputs"
            lexical_outputs.symlink_to(outside, target_is_directory=True)
            run_dir = lexical_outputs / audit.MODEL_NAME / audit.CONFIG_STEM
            registered = root / "registered"
            registered.mkdir()
            with patch.object(audit, "OUTPUT_ROOT", lexical_outputs), patch.object(
                audit,
                "ALLOWED_EXTERNAL_OUTPUT_ROOTS",
                (registered,),
            ):
                with self.assertRaisesRegex(ValueError, "registered output roots"):
                    _validate_run_dir(run_dir, output_root=lexical_outputs)

    def test_stage_summary_print_payload_has_no_eager_default_lookup(self):
        gate = {"passed": True}
        decision = {"authorize_next_stage": True}
        self.assertIs(_summary_payload({"gate": gate}), gate)
        self.assertIs(_summary_payload({"decision": decision}), decision)

    def test_fresh_cases_are_disjoint_from_all_prior_manifests(self):
        current_path = Path(__file__).resolve().parent / "manifests" / (
            "fresh_base_routing_audit_v1.json"
        )
        current = json.loads(current_path.read_text(encoding="utf-8"))
        current_labels = {case["label"] for case in current["cases"]}
        prior_labels = set()
        analyses_root = current_path.parents[2]
        def add_labels(value):
            labels = set()
            if isinstance(value, dict):
                for key, nested in value.items():
                    if key in {"label", "class_label", "class_idx", "class_index"}:
                        if isinstance(nested, int) and 0 <= nested < 1000:
                            labels.add(nested)
                    if key in {"id", "case_id"} and isinstance(nested, str):
                        labels.update(
                            int(match.group(1))
                            for match in re.finditer(
                                r"class(?:_|-)?(\d{1,3})(?:\D|$)", nested
                            )
                        )
                    labels.update(add_labels(nested))
            elif isinstance(value, list):
                for nested in value:
                    labels.update(add_labels(nested))
            return labels

        for path in analyses_root.rglob("*.json"):
            if path == current_path:
                continue
            prior_labels.update(
                int(match.group(1))
                for match in re.finditer(
                    r"class(?:_|-)?(\d{1,3})(?:\D|$)", path.name
                )
            )
            if path.name in {
                "manifest.json",
                "protocol.json",
                "summary.json",
                "gate-summary.json",
                "diagnostic-summary.json",
                "plumbing-summary.json",
                "discovery-summary.json",
            }:
                payload = json.loads(path.read_text(encoding="utf-8"))
                prior_labels.update(add_labels(payload))
        self.assertTrue(current_labels.isdisjoint(prior_labels))

    def test_fresh_log_requires_ordered_hashed_markers(self):
        run_id = "a" * 32
        launch_sha256 = "b" * 64
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            marker = (
                f"[time-INFO]: Fresh run marker: run_id={run_id} fresh=True "
                f"config={audit.CONFIG_STEM} "
                f"output_dir={run_dir} global_seed=0 world_size=4 "
                f"launch_sha256={launch_sha256}"
            )
            lines = [
                "[time-INFO]: Training RNG seed: 0 (global_seed=0, rank=0, world_size=4)",
                (
                    f"[time-INFO]: Training provenance: run_id={run_id} "
                    f"launch_sha256={launch_sha256} git_commit={'a' * 40} "
                    f"config_sha256={CANONICAL_TRAINING_CONFIG_SHA256}"
                ),
                marker,
                "[time-ERROR]: No checkpoints found in directory: "
                f"{checkpoint_dir}",
                "[time-INFO]: Resume progress: next_step=0, data_batches_seen=0, "
                "sampler_epoch=0, sampler_batch_offset=0",
                "[time-INFO]: epoch 0-step 0 mse_loss: 1.0 total_loss: 1.0",
            ]
            checkpoints = {}
            for step in CHECKPOINT_STEPS:
                path = checkpoint_dir / f"ckpt_step_{step}.pth"
                content = f"checkpoint-{step}".encode("ascii")
                path.write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                lines.append(
                    f"[time-INFO]: ********************* Checkpoint saved at {path} "
                    f"run_id={run_id} step={step} size={len(content)} "
                    f"sha256={digest} launch_sha256={launch_sha256}"
                )
                checkpoints[str(step)] = {
                    "path": str(path),
                    "size": len(content),
                    "sha256": digest,
                    "run_id": run_id,
                    "trainer_contract": {
                        "training_provenance_sha256": launch_sha256,
                    },
                }
            (run_dir / "training.log").write_text("\n".join(lines) + "\n")
            snapshot = _fresh_training_log_snapshot(run_dir)
            _verify_training_log(snapshot, checkpoints)
            first_step = CHECKPOINT_STEPS[0]
            _verify_training_log(
                snapshot,
                {str(first_step): checkpoints[str(first_step)]},
                checkpoint_steps_to_bind=(first_step,),
            )
            original_log = (run_dir / "training.log").read_text()

            extended_steps = CHECKPOINT_STEPS + (250000, 300000)
            extended_checkpoints = copy.deepcopy(checkpoints)
            extended_lines = original_log.splitlines()
            for step in extended_steps[len(CHECKPOINT_STEPS):]:
                path = checkpoint_dir / f"ckpt_step_{step}.pth"
                content = f"checkpoint-{step}".encode("ascii")
                path.write_bytes(content)
                digest = hashlib.sha256(content).hexdigest()
                marker_line = (
                    f"[time-INFO]: ********************* Checkpoint saved at {path} "
                    f"run_id={run_id} step={step} size={len(content)} "
                    f"sha256={digest} launch_sha256={launch_sha256}"
                )
                extended_lines.append(marker_line)
                extended_checkpoints[str(step)] = {
                    "path": str(path),
                    "size": len(content),
                    "sha256": digest,
                    "run_id": run_id,
                    "trainer_contract": {
                        "training_provenance_sha256": launch_sha256,
                    },
                }
            (run_dir / "training.log").write_text(
                "\n".join(extended_lines) + "\n"
            )
            extended_snapshot = _fresh_training_log_snapshot(
                run_dir,
                checkpoint_steps=extended_steps,
            )
            self.assertEqual(
                extended_snapshot["checkpoint_steps"],
                list(extended_steps),
            )
            _verify_training_log(extended_snapshot, extended_checkpoints)
            (run_dir / "training.log").write_text(original_log)

            (run_dir / "training.log").write_text(
                (run_dir / "training.log").read_text().replace(
                    f"size={len(f'checkpoint-{CHECKPOINT_STEPS[0]}')} ",
                    "size=999 ",
                    1,
                )
            )
            with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                _verify_training_log(snapshot, checkpoints)

            marker_to_remove = next(
                marker["line"]
                for marker in snapshot["checkpoint_markers"].values()
                if "step=50000 " in marker["line"]
            )
            (run_dir / "training.log").write_text(
                "\n".join(
                    line
                    for line in original_log.splitlines()
                    if line != marker_to_remove
                )
                + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "marker"):
                _verify_training_log(snapshot, checkpoints)

            (run_dir / "training.log").write_text(
                original_log.rstrip("\n") + "\n" + marker_to_remove + "\n"
            )
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                _verify_training_log(snapshot, checkpoints)

    def test_fresh_log_resolves_relative_paths_from_training_worktree(self):
        run_id = "a" * 32
        launch_sha256 = "b" * 64
        checkpoint_step = CHECKPOINT_STEPS[0]
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            training_root = root / "training-worktree"
            analysis_root = root / "analysis-worktree"
            output_root = training_root / "outputs"
            run_dir = output_root / audit.MODEL_NAME / audit.CONFIG_STEM
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            analysis_root.mkdir()
            checkpoint_path = checkpoint_dir / f"ckpt_step_{checkpoint_step}.pth"
            checkpoint_content = b"checkpoint"
            checkpoint_path.write_bytes(checkpoint_content)
            checkpoint_sha256 = hashlib.sha256(checkpoint_content).hexdigest()
            relative_run = Path("outputs") / audit.MODEL_NAME / audit.CONFIG_STEM
            relative_checkpoint_dir = relative_run / "checkpoints"
            relative_checkpoint = (
                relative_checkpoint_dir / f"ckpt_step_{checkpoint_step}.pth"
            )
            lines = [
                "[time-INFO]: Training RNG seed: 0 "
                "(global_seed=0, rank=0, world_size=4)",
                (
                    f"[time-INFO]: Training provenance: run_id={run_id} "
                    f"launch_sha256={launch_sha256} git_commit={'a' * 40} "
                    f"config_sha256={CANONICAL_TRAINING_CONFIG_SHA256}"
                ),
                (
                    f"[time-INFO]: Fresh run marker: run_id={run_id} fresh=True "
                    f"config={audit.CONFIG_STEM} output_dir={relative_run} "
                    f"global_seed=0 world_size=4 launch_sha256={launch_sha256}"
                ),
                (
                    "[time-ERROR]: No checkpoints found in directory: "
                    f"{relative_checkpoint_dir}"
                ),
                "[time-INFO]: Resume progress: next_step=0, data_batches_seen=0, "
                "sampler_epoch=0, sampler_batch_offset=0",
                "[time-INFO]: epoch 0-step 0 mse_loss: 1.0 total_loss: 1.0",
                (
                    f"[time-INFO]: ********************* Checkpoint saved at "
                    f"{relative_checkpoint} run_id={run_id} step={checkpoint_step} "
                    f"size={len(checkpoint_content)} sha256={checkpoint_sha256} "
                    f"launch_sha256={launch_sha256}"
                ),
            ]
            (run_dir / "training.log").write_text("\n".join(lines) + "\n")
            checkpoints = {
                str(checkpoint_step): {
                    "size": len(checkpoint_content),
                    "sha256": checkpoint_sha256,
                    "run_id": run_id,
                    "trainer_contract": {
                        "training_provenance_sha256": launch_sha256,
                    },
                }
            }

            with patch.object(audit, "PROJECT_ROOT", analysis_root):
                snapshot = _fresh_training_log_snapshot(
                    run_dir,
                    checkpoint_steps=(checkpoint_step,),
                    project_root=training_root,
                )
                _verify_training_log(
                    snapshot,
                    checkpoints,
                    run_dir=run_dir,
                    output_root=output_root,
                    project_root=training_root,
                )
            self.assertEqual(snapshot["project_root"], str(training_root))

    def test_fresh_log_rejects_multiple_valid_fresh_invocations(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            run_dir = Path(temporary_dir) / "run"
            checkpoint_dir = run_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            lines = []
            for run_id in ("a" * 32, "b" * 32):
                launch_sha256 = hashlib.sha256(run_id.encode("ascii")).hexdigest()
                lines.extend(
                    [
                        (
                            f"[time-INFO]: Fresh run marker: run_id={run_id} "
                            f"fresh=True config={audit.CONFIG_STEM} "
                            f"output_dir={run_dir} global_seed=0 world_size=4 "
                            f"launch_sha256={launch_sha256}"
                        ),
                        (
                            "[time-ERROR]: No checkpoints found in directory: "
                            f"{checkpoint_dir}"
                        ),
                        (
                            "[time-INFO]: Resume progress: next_step=0, "
                            "data_batches_seen=0, sampler_epoch=0, "
                            "sampler_batch_offset=0"
                        ),
                        "[time-INFO]: epoch 0-step 0 mse_loss: 1.0 total_loss: 1.0",
                    ]
                )
                for step in CHECKPOINT_STEPS:
                    path = checkpoint_dir / f"ckpt_step_{step}.pth"
                    if not path.exists():
                        content = f"checkpoint-{step}".encode("ascii")
                        path.write_bytes(content)
                    digest = hashlib.sha256(path.read_bytes()).hexdigest()
                    lines.append(
                        f"[time-INFO]: ********************* Checkpoint saved at {path} "
                        f"run_id={run_id} step={step} size={path.stat().st_size} "
                        f"sha256={digest} launch_sha256={launch_sha256}"
                    )
            (run_dir / "training.log").write_text("\n".join(lines) + "\n")
            with self.assertRaisesRegex(ValueError, "exactly one"):
                _fresh_training_log_snapshot(run_dir)

    def test_protocol_checkpoint_contract_is_rebuilt_from_files(self):
        run_id = "a" * 32
        launch_sha256 = "b" * 64
        trajectory = {"version": 2, "sampler_contract": {"dataset": {}}}
        observed = {}
        for step in CHECKPOINT_STEPS:
            contract = {
                "trajectory": trajectory,
                "run_id": run_id,
                "training_provenance_sha256": launch_sha256,
                "progress": {"next_step": step + 1},
            }
            observed[str(step)] = {
                "path": f"/run/checkpoints/ckpt_step_{step}.pth",
                "lexical_path": f"/run/checkpoints/ckpt_step_{step}.pth",
                "size": step,
                "mtime_ns": step,
                "sha256": "a" * 64,
                "state": "ema_model_state_dict",
                "run_id": run_id,
                "trainer_contract": contract,
            }

        with patch(
            "analyses.fresh_base_routing.audit._checkpoint_records",
            return_value=(observed, Path("/config.yaml"), trajectory),
        ) as rebuild:
            _verify_checkpoint_contracts(
                Path("/run"),
                {},
                run_id,
                launch_sha256,
                copy.deepcopy(observed),
                trajectory,
            )
            rebuild.assert_called_once_with(
                Path("/run"),
                {},
                expected_run_id=run_id,
                expected_training_provenance_sha256=launch_sha256,
            )

        tampered = copy.deepcopy(observed)
        tampered["50000"]["trainer_contract"]["progress"]["next_step"] += 1
        with patch(
            "analyses.fresh_base_routing.audit._checkpoint_records",
            return_value=(observed, Path("/config.yaml"), trajectory),
        ):
            with self.assertRaisesRegex(RuntimeError, "trainer contract"):
                _verify_checkpoint_contracts(
                    Path("/run"),
                    {},
                    run_id,
                    launch_sha256,
                    tampered,
                    trajectory,
                )

    def test_trainer_state_contract_rejects_legacy_or_mismatched_progress(self):
        state = {
            "version": 2,
            "augmentation_seed_version": 1,
            "global_seed": 0,
            "world_size": 4,
            "grad_mix": 1,
            "batches_per_epoch": 100,
            "next_step": 51,
            "data_batches_seen": 51,
            "sampler_epoch": 0,
            "sampler_batch_offset": 51,
            "sampler_contract": {
                "version": 1,
                "type": "distributed",
                "global_seed": 0,
                "per_rank_batch_size": 64,
                "drop_last": False,
                "case1_prob": None,
                "dataset": {
                    "version": 1,
                    "type": "__mp_main__.LatentFolder",
                    "num_samples": 1000,
                    "ordered_samples_sha256": "0" * 64,
                },
            },
            "rank_states": [
                {
                    "rank": rank,
                    "rng_state": _checkpoint_rng_state(),
                }
                for rank in range(4)
            ],
            "training_provenance": _training_provenance(),
        }
        launch_sha256 = audit._json_sha256(state["training_provenance"])
        self.assertEqual(
            _trainer_state_contract(
                state,
                50,
                4,
                0,
                256,
                expected_training_provenance_sha256=launch_sha256,
            )["progress"]["next_step"],
            51,
        )
        state["version"] = 1
        with self.assertRaisesRegex(ValueError, "version 2"):
            _trainer_state_contract(
                state,
                50,
                4,
                0,
                256,
                expected_training_provenance_sha256=launch_sha256,
            )


if __name__ == "__main__":
    unittest.main()
