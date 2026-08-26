import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from analyses.run_timestep_utility_probe import build_parser as build_probe_parser
from analyses.timestep_utility import DEFAULT_BLOCK_INDICES
from analyses.timestep_utility.batch import BLOCK_INDICES
from analyses.run_timestep_utility_probe_batch import (
    _collect_project_source_hashes,
    _pending_result_path,
    _pending_seal_path,
    _publish_summary,
    _require_confirmatory_unlock,
    _result_seal_path,
    _run_device_cases,
    sha256_file,
)


class _FakeModel:
    def __init__(self, depth=12):
        self.blocks = [
            SimpleNamespace(
                use_moe=index % 2 == 1,
                mlp=SimpleNamespace(top_k=1, router_weight_mode="identity"),
            )
            for index in range(depth)
        ]

    @staticmethod
    def parameters():
        return ()


class TimestepUtilityBatchRunnerTests(unittest.TestCase):
    @staticmethod
    def _payload(root):
        result_path = root / "cases" / "discovery" / "01_case00.json"
        case = {
            "id": "case00",
            "latent": str(root / "case00.latent.npz"),
            "latent_sha256": "1" * 64,
            "label": 1,
            "seed": 7,
        }
        return {
            "device": "cuda:4",
            "jobs": [{"case": case, "result_path": str(result_path)}],
            "checkpoint": str(root / "checkpoint.pth"),
            "weights_checkpoint": str(root / "weights.pth"),
            "config": str(root / "config.yaml"),
            "manifest": str(root / "manifest.json"),
            "protocol": str(root / "protocol.json"),
            "checkpoint_sha256": "2" * 64,
            "weights_sha256": "2" * 64,
            "config_sha256": "3" * 64,
            "manifest_sha256": "4" * 64,
            "source_sha256": {},
            "protocol_sha256": "5" * 64,
        }

    def test_locked_blocks_match_base_depth_and_interleave(self):
        self.assertEqual(BLOCK_INDICES, DEFAULT_BLOCK_INDICES)
        self.assertEqual(BLOCK_INDICES, (1, 5, 11))
        model = _FakeModel()
        with patch(
            "analyses.run_timestep_utility_probe_batch._build_model",
            return_value=model,
        ):
            metadata, _ = _collect_project_source_hashes(object())
        self.assertEqual(metadata["block_contract"], {
            "depth": 12,
            "blocks": [
                {
                    "index": index,
                    "use_moe": True,
                    "top_k": 1,
                    "router_weight_mode": "identity",
                }
                for index in BLOCK_INDICES
            ],
        })

    def test_standalone_cli_defaults_to_locked_base_blocks(self):
        args = build_probe_parser().parse_args([
            "--ckpt", "checkpoint.pth",
            "--latent", "sample.latent.npz",
            "--label", "0",
        ])
        self.assertEqual(args.block_indices, BLOCK_INDICES)

    def test_prepare_rejects_out_of_range_block(self):
        with patch(
            "analyses.run_timestep_utility_probe_batch._build_model",
            return_value=_FakeModel(depth=11),
        ):
            with self.assertRaisesRegex(ValueError, "block_index 11 is outside"):
                _collect_project_source_hashes(object())

    def test_prepare_rejects_dense_block(self):
        model = _FakeModel()
        model.blocks[BLOCK_INDICES[1]].use_moe = False
        with patch(
            "analyses.run_timestep_utility_probe_batch._build_model",
            return_value=model,
        ):
            with self.assertRaisesRegex(ValueError, "block 5 is not an MoE block"):
                _collect_project_source_hashes(object())

    def test_prepare_rejects_wrong_top_k(self):
        model = _FakeModel()
        model.blocks[BLOCK_INDICES[1]].mlp.top_k = 2
        with patch(
            "analyses.run_timestep_utility_probe_batch._build_model",
            return_value=model,
        ):
            with self.assertRaisesRegex(ValueError, "block 5 has top_k=2"):
                _collect_project_source_hashes(object())

    def test_prepare_rejects_wrong_router_weight_mode(self):
        model = _FakeModel()
        model.blocks[BLOCK_INDICES[1]].mlp.router_weight_mode = "softmax"
        with patch(
            "analyses.run_timestep_utility_probe_batch._build_model",
            return_value=model,
        ):
            with self.assertRaisesRegex(
                ValueError,
                "block 5 has router_weight_mode='softmax'",
            ):
                _collect_project_source_hashes(object())

    def test_failed_tail_verification_cannot_publish_stale_pending(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            payload = self._payload(Path(temporary_dir))
            result_path = Path(payload["jobs"][0]["result_path"])
            pending_path = _pending_result_path(result_path, payload["device"])
            seal_path = _pending_seal_path(pending_path)

            with (
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_verify_locked_inputs",
                    side_effect=[None, RuntimeError("checkpoint changed")],
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch._verify_file"
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "run_timestep_utility_probe",
                    return_value={"marker": "stale"},
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_validate_case_result"
                ),
                patch("torch.cuda.set_device"),
                patch("torch.cuda.empty_cache"),
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                    _run_device_cases(payload)

            self.assertTrue(pending_path.is_file())
            self.assertFalse(seal_path.exists())
            self.assertFalse(result_path.exists())

            unlink_order = []
            original_unlink = Path.unlink

            def record_unlink(path, *unlink_args, **unlink_kwargs):
                unlink_order.append(path)
                return original_unlink(path, *unlink_args, **unlink_kwargs)

            with (
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_verify_locked_inputs"
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch._verify_file"
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "run_timestep_utility_probe",
                    return_value={"marker": "fresh"},
                ) as probe,
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_validate_case_result"
                ),
                patch("torch.cuda.set_device"),
                patch("torch.cuda.empty_cache"),
                patch.object(Path, "unlink", new=record_unlink),
            ):
                _run_device_cases(payload)

            probe.assert_called_once()
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["marker"], "fresh")
            self.assertTrue(_result_seal_path(result_path).is_file())
            self.assertFalse(pending_path.exists())
            self.assertFalse(seal_path.exists())
            paired_cleanup = [
                path
                for path in unlink_order
                if path in {pending_path, seal_path}
            ]
            self.assertEqual(paired_cleanup[-2:], [seal_path, pending_path])

    def test_each_published_case_gets_its_own_seal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            payload = self._payload(root)
            second_case = {
                **payload["jobs"][0]["case"],
                "id": "case01",
                "label": 2,
                "seed": 8,
                "latent_sha256": "6" * 64,
            }
            second_path = root / "cases" / "discovery" / "02_case01.json"
            payload["jobs"].append({
                "case": second_case,
                "result_path": str(second_path),
            })

            with (
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_verify_locked_inputs"
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch._verify_file"
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "run_timestep_utility_probe",
                    side_effect=[{"marker": "first"}, {"marker": "second"}],
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_validate_case_result"
                ),
                patch("torch.cuda.set_device"),
                patch("torch.cuda.empty_cache"),
            ):
                _run_device_cases(payload)

            first_path = Path(payload["jobs"][0]["result_path"])
            first_seal = json.loads(
                _result_seal_path(first_path).read_text(encoding="utf-8")
            )
            second_seal = json.loads(
                _result_seal_path(second_path).read_text(encoding="utf-8")
            )
            self.assertEqual(first_seal["case_id"], "case00")
            self.assertEqual(second_seal["case_id"], "case01")

    def test_existing_summary_cannot_be_rebased(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            summary_path = Path(temporary_dir) / "discovery-summary.json"
            existing = {"gate": {"passed": False}, "case_results": ["old"]}
            replacement = {"gate": {"passed": True}, "case_results": ["new"]}
            summary_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with patch(
                "analyses.run_timestep_utility_probe_batch._verify_locked_inputs"
            ):
                with self.assertRaisesRegex(ValueError, "differs from recomputation"):
                    _publish_summary(summary_path, replacement, {})
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")),
                existing,
            )

    def test_pending_replacement_after_tail_check_is_not_published(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            payload = self._payload(Path(temporary_dir))
            result_path = Path(payload["jobs"][0]["result_path"])
            pending_path = _pending_result_path(result_path, payload["device"])
            seal_path = _pending_seal_path(pending_path)
            verification_count = 0

            def verify_inputs(_):
                nonlocal verification_count
                verification_count += 1
                if verification_count == 2:
                    pending_path.write_text(
                        json.dumps({"marker": "replacement"}) + "\n",
                        encoding="utf-8",
                    )

            def verify_file(path, expected_sha256, description):
                if "Pending result" not in description:
                    return
                if sha256_file(path) != expected_sha256:
                    raise RuntimeError(f"{description} changed")

            with (
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_verify_locked_inputs",
                    side_effect=verify_inputs,
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch._verify_file",
                    side_effect=verify_file,
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "run_timestep_utility_probe",
                    return_value={"marker": "validated"},
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_validate_case_result"
                ),
                patch("torch.cuda.set_device"),
                patch("torch.cuda.empty_cache"),
            ):
                with self.assertRaisesRegex(RuntimeError, "Pending result.*changed"):
                    _run_device_cases(payload)

            self.assertFalse(result_path.exists())
            self.assertFalse(seal_path.exists())

    def test_confirmatory_unlock_recomputes_and_requires_discovery_pass(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            summary_path = output_dir / "discovery-summary.json"
            manifest = {"cases": [{"split": "discovery", "id": "case00"}]}
            common_payload = {"protocol_sha256": "5" * 64}
            failed_gate = {
                "safety_passed": True,
                "routing_accuracy_gap_passed": False,
                "passed": False,
            }
            failed_summary = {"gate": failed_gate}
            summary_path.write_text(
                json.dumps(failed_summary),
                encoding="utf-8",
            )
            with (
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_load_split_results",
                    return_value=[{"case": "result"}],
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "aggregate_case_results",
                    return_value=failed_gate,
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_build_split_summary",
                    return_value=failed_summary,
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_verify_locked_inputs"
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "requires discovery"):
                    _require_confirmatory_unlock(
                        output_dir=output_dir,
                        protocol_path=output_dir / "protocol.json",
                        manifest=manifest,
                        devices=("cuda:4",),
                        common_payload=common_payload,
                    )

            passed_gate = {
                "safety_passed": True,
                "routing_accuracy_gap_passed": True,
                "stage_structure_passed": False,
                "passed": True,
            }
            passed_summary = {"gate": passed_gate}
            summary_path.write_text(
                json.dumps(passed_summary),
                encoding="utf-8",
            )
            with (
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_load_split_results",
                    return_value=[{"case": "result"}],
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "aggregate_case_results",
                    return_value=passed_gate,
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_build_split_summary",
                    return_value=passed_summary,
                ),
                patch(
                    "analyses.run_timestep_utility_probe_batch."
                    "_verify_locked_inputs"
                ),
            ):
                _require_confirmatory_unlock(
                    output_dir=output_dir,
                    protocol_path=output_dir / "protocol.json",
                    manifest=manifest,
                    devices=("cuda:4",),
                    common_payload=common_payload,
                )


if __name__ == "__main__":
    unittest.main()
