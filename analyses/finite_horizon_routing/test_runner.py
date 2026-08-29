"""Tests for sealed stage execution and cross-process locking."""

import json
import io
import os
import signal
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from easydict import EasyDict
from torch.torch_version import TorchVersion

from analyses.finite_horizon_routing.runner import (
    LOCKED_BRANCH,
    PROTOCOL_FILENAME,
    PROTOCOL_SEAL_FILENAME,
    SPLIT_PREREQUISITES,
    SUMMARY_SHA256_FIELD,
    _analysis_runtime_environment,
    _arm_parent_death_signal,
    _exclusive_lock,
    _git_contract,
    _main_worktree_root,
    _publish_protocol,
    _read_checkpoint_record,
    _read_sealed_json,
    _sealed_payload,
    _verify_completed_split,
    _write_sealed_json,
    verify_protocol,
)
from analyses.run_finite_horizon_routing_probe import _publish_result


class SealedResultTest(unittest.TestCase):
    def test_payload_edit_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "summary.json"
            _write_sealed_json(
                path,
                {"passed": True, "metric": 0.25},
                SUMMARY_SHA256_FIELD,
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["metric"] = 0.75
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "content changed"):
                _read_sealed_json(path, SUMMARY_SHA256_FIELD)

    def test_previous_summary_is_recomputed_from_case_results(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            summary_path = output_dir / "summaries" / "discovery.json"
            _write_sealed_json(
                summary_path,
                {"passed": True, "source": "edited"},
                SUMMARY_SHA256_FIELD,
            )
            with (
                patch(
                    "analyses.finite_horizon_routing.runner._load_split_results",
                    return_value=([{"id": "case"}], [{"result_sha256": "a" * 64}]),
                ),
                patch(
                    "analyses.finite_horizon_routing.runner._build_summary",
                    return_value={"passed": False, "source": "recomputed"},
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "recomputed"):
                    _verify_completed_split(
                        output_dir,
                        {"protocol_sha256": "b" * 64},
                        "discovery",
                    )


class CheckpointLoadCompatibilityTest(unittest.TestCase):
    def test_restricted_loader_allows_project_metadata_types(self):
        from analyses.finite_horizon_routing.runner import _torch_load_handle

        payload = {
            "cfg": EasyDict({"model_name": "ProMoE_TC_B"}),
            "torch_version": TorchVersion(str(torch.__version__)),
            "tensor": torch.ones(2),
        }
        buffer = io.BytesIO()
        torch.save(payload, buffer)
        loaded = _torch_load_handle(buffer)
        self.assertIsInstance(loaded["cfg"], dict)
        self.assertEqual(loaded["cfg"]["model_name"], "ProMoE_TC_B")
        self.assertEqual(str(loaded["torch_version"]), str(torch.__version__))
        torch.testing.assert_close(loaded["tensor"], payload["tensor"])


class ProtocolRebuildTest(unittest.TestCase):
    def test_resigning_protocol_and_sidecar_cannot_replace_cases(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "gate"
            output_dir.mkdir()
            canonical_payload = {
                "output_dir": str(output_dir.resolve()),
                "cases": [{"id": "locked", "label": 3, "seed": 17}],
                "assignments": {
                    "confirmatory": [
                        {"case_id": "locked", "device": "cuda:0"}
                    ]
                },
            }
            tampered_payload = {
                **canonical_payload,
                "cases": [{"id": "replacement", "label": 999, "seed": 1}],
                "assignments": {
                    "confirmatory": [
                        {"case_id": "replacement", "device": "cuda:3"}
                    ]
                },
            }
            tampered_protocol = _sealed_payload(
                tampered_payload,
                "protocol_sha256",
            )
            (output_dir / PROTOCOL_FILENAME).write_text(
                json.dumps(tampered_protocol),
                encoding="utf-8",
            )
            (output_dir / PROTOCOL_SEAL_FILENAME).write_text(
                tampered_protocol["protocol_sha256"] + "\n",
                encoding="ascii",
            )

            with patch(
                "analyses.finite_horizon_routing.runner._rebuild_protocol_payload",
                return_value=canonical_payload,
            ) as rebuild:
                with self.assertRaisesRegex(ValueError, "canonical current"):
                    verify_protocol(output_dir)
            rebuild.assert_called_once_with(output_dir.resolve())


class LockTest(unittest.TestCase):
    def test_second_process_contract_cannot_take_same_lock(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "run.lock"
            with _exclusive_lock(path, "test"):
                with self.assertRaisesRegex(RuntimeError, "holds"):
                    with _exclusive_lock(path, "test"):
                        pass

    def test_protocol_publication_is_locked_atomic_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "gate"
            protocol = {"protocol_sha256": "a" * 64, "value": 1}
            _publish_protocol(output_dir, protocol)
            self.assertEqual(
                json.loads((output_dir / PROTOCOL_FILENAME).read_text()),
                protocol,
            )
            self.assertEqual(
                (output_dir / PROTOCOL_SEAL_FILENAME).read_text(encoding="ascii"),
                "a" * 64 + "\n",
            )
            with self.assertRaisesRegex(FileExistsError, "never overwrite"):
                _publish_protocol(
                    output_dir,
                    {"protocol_sha256": "b" * 64, "value": 2},
                )
            self.assertFalse(list(root.glob(".gate.prepare-*")))

    def test_protocol_publication_uses_its_sibling_lock(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            output_dir = root / "gate"
            lock_path = root / ".gate.prepare.lock"
            with _exclusive_lock(lock_path, "test preparation"):
                with self.assertRaisesRegex(RuntimeError, "preparation lock"):
                    _publish_protocol(
                        output_dir,
                        {"protocol_sha256": "a" * 64},
                    )

    def test_confirmation_requires_both_earlier_stages(self):
        self.assertEqual(
            SPLIT_PREREQUISITES["confirmatory"],
            ("plumbing", "discovery"),
        )


class SingleProbePublicationTest(unittest.TestCase):
    def test_concurrent_publish_has_one_complete_winner(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir) / "probe.json"

            def publish(index):
                try:
                    _publish_result(output_path, {"winner": index})
                except FileExistsError:
                    return "exists"
                return "published"

            with ThreadPoolExecutor(max_workers=2) as executor:
                outcomes = list(executor.map(publish, (1, 2)))

            self.assertCountEqual(outcomes, ("published", "exists"))
            self.assertIn(
                json.loads(output_path.read_text(encoding="utf-8")),
                ({"winner": 1}, {"winner": 2}),
            )
            self.assertFalse(
                list(output_path.parent.glob(f".{output_path.name}.*.tmp"))
            )


class RuntimeContractTest(unittest.TestCase):
    def test_analysis_environment_seals_driver_and_cudnn(self):
        completed = SimpleNamespace(stdout="580.95.05\n580.95.05\n")
        with (
            patch(
                "analyses.finite_horizon_routing.runner._runtime_environment",
                return_value={"torch": "test"},
            ),
            patch(
                "analyses.finite_horizon_routing.runner.subprocess.run",
                return_value=completed,
            ),
            patch(
                "analyses.finite_horizon_routing.runner.inspect.getsource",
                return_value="scheduler source",
            ),
            patch(
                "analyses.finite_horizon_routing.runner.torch.backends.cudnn.version",
                return_value=90100,
            ),
        ):
            environment = _analysis_runtime_environment(("cuda:0",))
        self.assertEqual(environment["cuda_driver_version"], "580.95.05")
        self.assertEqual(environment["cudnn_runtime_version"], 90100)

    def test_gpu_worker_arms_sigkill_and_checks_parent_race(self):
        prctl = MagicMock(return_value=0)
        library = SimpleNamespace(prctl=prctl)
        with (
            patch(
                "analyses.finite_horizon_routing.runner.ctypes.CDLL",
                return_value=library,
            ),
            patch(
                "analyses.finite_horizon_routing.runner.os.getppid",
                return_value=123,
            ),
        ):
            _arm_parent_death_signal(123)
        self.assertEqual(prctl.call_args.args[:2], (1, int(signal.SIGKILL)))

        with (
            patch(
                "analyses.finite_horizon_routing.runner.ctypes.CDLL",
                return_value=library,
            ),
            patch(
                "analyses.finite_horizon_routing.runner.os.getppid",
                return_value=124,
            ),
            patch(
                "analyses.finite_horizon_routing.runner.os.getpid",
                return_value=456,
            ),
            patch("analyses.finite_horizon_routing.runner.os.kill") as kill,
        ):
            with self.assertRaisesRegex(RuntimeError, "parent exited"):
                _arm_parent_death_signal(123)
        kill.assert_called_once_with(456, signal.SIGKILL)

    def test_checkpoint_replacement_cannot_mix_payload_and_file_identity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            checkpoint_path = root / "ckpt_step_300000.pth"
            replacement_path = root / "replacement.pth"
            base = {
                "step": 300000,
                "model_state_dict": {},
                "ema_model_state_dict": {},
            }
            torch.save({**base, "trainer_state": {"source": "A"}}, checkpoint_path)
            torch.save({**base, "trainer_state": {"source": "B"}}, replacement_path)
            runtime_cfg = SimpleNamespace(
                gpu_ids=[0, 1, 2, 3],
                global_seed=0,
                total_train_batch_size=256,
            )
            training_log = {
                "run_id": "a" * 32,
                "training_provenance_sha256": "b" * 64,
            }
            trainer_contract = MagicMock(return_value={
                "run_id": training_log["run_id"],
                "training_provenance_sha256": training_log[
                    "training_provenance_sha256"
                ],
            })
            original_torch_load = torch.load
            replaced = False

            def replace_then_load(handle, **kwargs):
                nonlocal replaced
                if not replaced:
                    os.replace(replacement_path, checkpoint_path)
                    replaced = True
                return original_torch_load(handle, **kwargs)

            with (
                patch(
                    "analyses.finite_horizon_routing.runner._trainer_state_contract",
                    trainer_contract,
                ),
                patch(
                    "analyses.finite_horizon_routing.runner.torch.load",
                    side_effect=replace_then_load,
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "changed while it was open"):
                    _read_checkpoint_record(
                        checkpoint_path,
                        runtime_cfg,
                        training_log,
                    )
            self.assertEqual(trainer_contract.call_args.args[0]["source"], "A")


class WorktreeResolutionTest(unittest.TestCase):
    def test_git_contract_requires_fresh_hashes_and_live_remote_tip(self):
        commit = "a" * 40
        state = {
            "branch": LOCKED_BRANCH,
            "commit": commit,
            "authoritative_remote_url": "git@example/repo.git",
            "authoritative_remote_ref": f"refs/heads/{LOCKED_BRANCH}",
            "authoritative_remote_tip": commit,
            "status": "",
        }
        with patch(
            "analyses.finite_horizon_routing.runner.repository_state",
            return_value=state,
        ) as repository_state:
            contract = _git_contract()
        repository_state.assert_called_once_with(
            Path(__file__).resolve().parents[2],
            authoritative_remote_ref=f"refs/heads/{LOCKED_BRANCH}",
        )
        self.assertEqual(contract["authoritative_remote_tip"], commit)
        self.assertTrue(contract["status_clean"])

        dirty = {**state, "status": " M analyses/changed.py\n"}
        with patch(
            "analyses.finite_horizon_routing.runner.repository_state",
            return_value=dirty,
        ):
            with self.assertRaisesRegex(RuntimeError, "clean committed"):
                _git_contract()

        unpushed = {**state, "authoritative_remote_tip": "b" * 40}
        with patch(
            "analyses.finite_horizon_routing.runner.repository_state",
            return_value=unpushed,
        ):
            with self.assertRaisesRegex(RuntimeError, "authoritative analysis"):
                _git_contract()

    def test_main_worktree_comes_from_git_common_directory(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            analysis_root = root / "linked-worktree"
            main_root = root / "main-worktree"
            analysis_root.mkdir()
            common_dir = main_root / ".git"
            common_dir.mkdir(parents=True)
            with patch(
                "analyses.finite_horizon_routing.runner.git_output",
                return_value=str(common_dir),
            ) as git_output:
                self.assertEqual(_main_worktree_root(analysis_root), main_root)
            git_output.assert_called_once_with(
                analysis_root,
                "rev-parse",
                "--git-common-dir",
            )


if __name__ == "__main__":
    unittest.main()
