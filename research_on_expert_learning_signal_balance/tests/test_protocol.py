from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from credit_redistribution.controller import BRANCHES
from credit_redistribution.git_provenance import run_git
from credit_redistribution.heldout import canonical_json_sha256
from credit_redistribution.protocol import (
    PROJECT_ROOT,
    SEALED_GPU_IDS,
    SOURCE_PATHS,
    _sealed_gpu_device_pairs,
    _source_hashes,
    _validate_branch_configs,
    git_contract,
    load_and_verify_protocol,
    resolve_archived_artifact_path,
)
from credit_redistribution.protocol_lock import (
    V3_SHA256,
    V4_SHA256,
    load_effective_protocol,
)
from credit_redistribution.protocol import V3_PATH, V4_PATH


class ProtocolTest(unittest.TestCase):
    def test_archived_artifact_paths_use_project_relocations(self):
        cases = {
            "/home/dev/promoe-probes/credit-balance-gate-base200k-v1-preregister.json":
                PROJECT_ROOT
                / "credit_redistribution/archived_credit_redistribution/2026-08-28"
                / "dirty_diagnostics/credit-balance-gate-base200k-v1-preregister.json",
            "/home/dev/promoe-probes/base-seed0-ckpt_step_200000.pth":
                PROJECT_ROOT
                / "analyses/archvied_analyses/2026-08-28/dirty_probes"
                / "promoe-probes/base-seed0-ckpt_step_200000.pth",
            (
                "/home/dev/promoe-runs/ProMoE_TC_B_lossfree/"
                "004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k/"
                "checkpoints/ckpt_step_200000.pth"
            ):
                PROJECT_ROOT
                / "outputs/archived_outputs/2026-08-28/ProMoE_TC_B_lossfree"
                / "004_ProMoE_B_lossfree_u1e2_credit_control_s0_200k"
                / "checkpoints/ckpt_step_200000.pth",
        }
        for locked, expected in cases.items():
            with self.subTest(locked=locked):
                self.assertEqual(resolve_archived_artifact_path(locked), expected)
        with self.assertRaisesRegex(RuntimeError, "no approved relocation"):
            resolve_archived_artifact_path("/home/dev/promoe-probes/unknown")

    def test_git_contract_ignores_repository_environment_redirects(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            redirected = root / "redirected"
            authoritative = root / "authoritative.git"
            real.mkdir()
            redirected.mkdir()
            authoritative.mkdir()
            run_git(
                authoritative,
                "init",
                "--bare",
                "--quiet",
                check=True,
            )
            heads = {}
            for repository in (real, redirected):
                run_git(repository, "init", "--quiet", check=True)
                run_git(
                    repository,
                    "config",
                    "user.email",
                    "test@example.com",
                    check=True,
                )
                run_git(
                    repository,
                    "config",
                    "user.name",
                    "Test User",
                    check=True,
                )
                run_git(
                    repository,
                    "-c",
                    "commit.gpgSign=false",
                    "-c",
                    "core.hooksPath=/dev/null",
                    "-c",
                    "user.email=test@example.com",
                    "-c",
                    "user.name=Test User",
                    "commit",
                    "--allow-empty",
                    "--quiet",
                    "-m",
                    "initial",
                    check=True,
                )
                head = run_git(
                    repository, "rev-parse", "HEAD", check=True
                ).stdout.strip()
                heads[repository.name] = head
                run_git(
                    repository,
                    "update-ref",
                    "refs/remotes/origin/repa",
                    head,
                    check=True,
                )
            run_git(
                real,
                "remote",
                "add",
                "origin",
                str(authoritative),
                check=True,
            )
            run_git(
                real,
                "push",
                "--quiet",
                "origin",
                f"{heads['real']}:refs/heads/repa",
                check=True,
            )

            run_git(
                real,
                "-c",
                "commit.gpgSign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "--allow-empty",
                "--quiet",
                "-m",
                "unpushed",
                check=True,
            )
            with mock.patch(
                "credit_redistribution.protocol.PROJECT_ROOT", real
            ), mock.patch(
                "credit_redistribution.git_provenance."
                "AUTHORITATIVE_REMOTE_URL",
                str(authoritative),
            ), mock.patch.dict(
                os.environ,
                {
                    "GIT_DIR": str(redirected / ".git"),
                    "GIT_WORK_TREE": str(redirected),
                },
            ), self.assertRaisesRegex(RuntimeError, "HEAD == origin/repa"):
                git_contract(require_clean=False, require_origin=True)

            real_head = run_git(
                real, "rev-parse", "HEAD", check=True
            ).stdout.strip()
            run_git(
                real,
                "update-ref",
                "refs/remotes/origin/repa",
                real_head,
                check=True,
            )
            run_git(
                real,
                "push",
                "--quiet",
                "origin",
                f"{real_head}:refs/heads/repa",
                check=True,
            )
            (real / "dirty.txt").write_text("dirty\n", encoding="utf-8")
            with mock.patch(
                "credit_redistribution.protocol.PROJECT_ROOT", real
            ), mock.patch(
                "credit_redistribution.git_provenance."
                "AUTHORITATIVE_REMOTE_URL",
                str(authoritative),
            ), mock.patch.dict(
                os.environ,
                {
                    "GIT_DIR": str(redirected / ".git"),
                    "GIT_WORK_TREE": str(redirected),
                },
            ), self.assertRaisesRegex(RuntimeError, "clean worktree"):
                git_contract(require_clean=True, require_origin=True)

    def test_sealed_gpu_mapping_handles_remapped_visibility(self):
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "4,5,6,7"}):
            with mock.patch(
                "credit_redistribution.protocol.torch.cuda.device_count",
                return_value=4,
            ):
                self.assertEqual(
                    _sealed_gpu_device_pairs(),
                    [(physical_id, offset) for offset, physical_id in enumerate(SEALED_GPU_IDS)],
                )

    def test_sealed_gpu_mapping_handles_uuid_visibility(self):
        with mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b,GPU-c,GPU-d",
                "PROMOE_SEALED_PHYSICAL_GPU_IDS": "4,5,6,7",
            },
        ):
            with mock.patch(
                "credit_redistribution.protocol.torch.cuda.device_count",
                return_value=4,
            ):
                self.assertEqual(
                    _sealed_gpu_device_pairs(),
                    [(physical_id, offset) for offset, physical_id in enumerate(SEALED_GPU_IDS)],
                )

    def test_sealed_gpu_mapping_rejects_unprovable_numeric_remap(self):
        with mock.patch.dict(
            os.environ,
            {"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
            clear=True,
        ):
            with mock.patch(
                "credit_redistribution.protocol.torch.cuda.device_count",
                return_value=4,
            ):
                with self.assertRaisesRegex(RuntimeError, "Cannot map"):
                    _sealed_gpu_device_pairs()

    def test_sealed_gpu_mapping_rejects_contradictory_stale_declaration(self):
        with mock.patch.dict(
            os.environ,
            {
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "PROMOE_SEALED_PHYSICAL_GPU_IDS": "4,5,6,7",
            },
            clear=True,
        ):
            with mock.patch(
                "credit_redistribution.protocol.torch.cuda.device_count",
                return_value=4,
            ):
                with self.assertRaisesRegex(RuntimeError, "contradicts numeric"):
                    _sealed_gpu_device_pairs()

    def test_protocol_loader_rejects_sidecar_consistent_document_tampering(self):
        expected = {
            "status": "immutable_pre_efficacy",
            "git": {"commit": "abc"},
            "branches": [{"name": "measure_only_control"}],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protocol.json"
            path.write_text(json.dumps(expected), encoding="utf-8")
            path.with_suffix(".sha256").write_text(
                canonical_json_sha256(expected) + "\n", encoding="utf-8"
            )
            with mock.patch(
                "credit_redistribution.protocol.build_protocol",
                return_value=copy.deepcopy(expected),
            ):
                loaded, _ = load_and_verify_protocol(path, require_git=True)
                self.assertEqual(loaded, expected)

                tampered = copy.deepcopy(expected)
                tampered["branches"][0]["name"] = "tampered"
                path.write_text(json.dumps(tampered), encoding="utf-8")
                path.with_suffix(".sha256").write_text(
                    canonical_json_sha256(tampered) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "reconstructed contract"):
                    load_and_verify_protocol(path, require_git=True)

    def test_effective_preregistration_is_exact_single_field_amendment(self):
        effective = load_effective_protocol(V3_PATH, V4_PATH)
        self.assertEqual(effective["effective_protocol_hashes"]["v3"], V3_SHA256)
        self.assertEqual(effective["effective_protocol_hashes"]["v4"], V4_SHA256)
        self.assertEqual(effective["statistics"]["bootstrap_resamples"], 200_000)

    def test_three_branch_definition_is_complete_and_ordered(self):
        rows = _validate_branch_configs()
        self.assertEqual(tuple(row["name"] for row in rows), BRANCHES)
        self.assertTrue(all(row["gpu_ids"] == [4, 5, 6, 7] for row in rows))
        self.assertEqual(len({row["output_dir"] for row in rows}), 3)

    def test_source_hash_manifest_contains_every_declared_source(self):
        hashes = _source_hashes()
        self.assertTrue(set(SOURCE_PATHS).issubset(hashes))
        self.assertIn("credit_redistribution/orchestration.py", hashes)
        self.assertIn("credit_redistribution/git_provenance.py", hashes)
        self.assertIn("analyses/run_credit_redistribution_gate.py", hashes)
        self.assertIn("models/phase_metric.py", hashes)


if __name__ == "__main__":
    unittest.main()
