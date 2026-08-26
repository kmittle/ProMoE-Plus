import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyses.run_expert_function_consistency_probe_batch import (
    _cleanup_published_result_artifacts,
    _device_pending_path,
    _pending_seal_path,
    _publish_summary,
    _run_device_cases,
    _verify_worker_inputs,
    sha256_file,
)


class ExpertFunctionBatchRunnerTests(unittest.TestCase):
    @staticmethod
    def _payload(root):
        result_path = root / "cases" / "01_case00.json"
        case = {
            "id": "case00",
            "latent": str(root / "case00.latent.npz"),
            "latent_sha256": "1" * 64,
            "latent_key": "latent",
            "label": 1,
            "seed": 7,
        }
        expected_run = {"protocol_sha256": "2" * 64}
        return {
            "device": "cuda:4",
            "total_cases": 1,
            "jobs": [{
                "index": 1,
                "case": case,
                "result_path": str(result_path),
                "expected_run": expected_run,
            }],
            "checkpoint": str(root / "checkpoint.pth"),
            "weights_checkpoint": str(root / "weights.pth"),
            "config": str(root / "config.yaml"),
            "manifest": str(root / "manifest.json"),
            "protocol": str(root / "protocol.json"),
            "num_threads": 1,
            "checkpoint_sha256": "3" * 64,
            "weights_sha256": "3" * 64,
            "config_sha256": "4" * 64,
            "manifest_sha256": "5" * 64,
            "source_sha256": {},
            "protocol_sha256": "2" * 64,
        }

    def test_worker_verifies_manifest_and_protocol(self):
        payload = self._payload(Path("/tmp/expert-function-test"))
        with patch(
            "analyses.run_expert_function_consistency_probe_batch."
            "_verify_locked_file"
        ) as verify_file:
            _verify_worker_inputs(payload)
        verified = {
            call.args[2]: (Path(call.args[0]), call.args[1])
            for call in verify_file.call_args_list
        }
        self.assertEqual(verified["Manifest"], (
            Path(payload["manifest"]),
            payload["manifest_sha256"],
        ))
        self.assertEqual(verified["Protocol"], (
            Path(payload["protocol"]),
            payload["protocol_sha256"],
        ))

    def test_summary_tail_failure_cannot_publish(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            summary_path = Path(temporary_dir) / "summary.json"
            with patch(
                "analyses.run_expert_function_consistency_probe_batch."
                "_verify_worker_inputs",
                side_effect=[None, RuntimeError("protocol changed")],
            ):
                with self.assertRaisesRegex(RuntimeError, "protocol changed"):
                    _publish_summary(
                        summary_path,
                        {"gate": {"passed": True}},
                        {},
                    )
            self.assertFalse(summary_path.exists())
            self.assertTrue(
                summary_path.with_suffix(".json.pending").is_file()
            )

    def test_existing_summary_cannot_be_rebased(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            summary_path = Path(temporary_dir) / "summary.json"
            existing = {"gate": {"passed": False}}
            summary_path.write_text(
                json.dumps(existing, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with patch(
                "analyses.run_expert_function_consistency_probe_batch."
                "_verify_worker_inputs"
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    "Existing summary differs from recomputation",
                ):
                    _publish_summary(
                        summary_path,
                        {"gate": {"passed": True}},
                        {},
                    )
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")),
                existing,
            )

    def test_failed_tail_verification_cannot_publish_stale_pending(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            payload = self._payload(Path(temporary_dir))
            result_path = Path(payload["jobs"][0]["result_path"])
            pending_path = result_path.with_suffix(
                result_path.suffix + ".pending.cuda_4"
            )
            seal_path = _pending_seal_path(pending_path)

            with (
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "_verify_worker_inputs",
                    side_effect=[None, RuntimeError("checkpoint changed")],
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "_verify_locked_file"
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "run_expert_function_consistency_probe",
                    return_value={"marker": "stale"},
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "validate_case_result"
                ),
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
                    "analyses.run_expert_function_consistency_probe_batch."
                    "_verify_worker_inputs"
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "_verify_locked_file"
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "run_expert_function_consistency_probe",
                    return_value={"marker": "fresh"},
                ) as probe,
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "validate_case_result"
                ),
                patch.object(Path, "unlink", new=record_unlink),
            ):
                _run_device_cases(payload)

            probe.assert_called_once()
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(result["marker"], "fresh")
            self.assertFalse(pending_path.exists())
            self.assertFalse(seal_path.exists())
            paired_cleanup = [
                path
                for path in unlink_order
                if path in {pending_path, seal_path}
            ]
            self.assertEqual(paired_cleanup[-2:], [seal_path, pending_path])

    def test_pending_replacement_after_validation_is_not_sealed(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            payload = self._payload(Path(temporary_dir))
            result_path = Path(payload["jobs"][0]["result_path"])
            pending_path = result_path.with_suffix(
                result_path.suffix + ".pending.cuda_4"
            )
            seal_path = _pending_seal_path(pending_path)

            verification_count = 0

            def verify_worker_inputs(_):
                nonlocal verification_count
                verification_count += 1
                if verification_count == 2:
                    pending_path.write_text(
                        json.dumps({"marker": "replacement"}) + "\n",
                        encoding="utf-8",
                    )

            def verify_pending(path, expected_sha256, description):
                if "Pending result" not in description:
                    return
                if sha256_file(path) != expected_sha256:
                    raise RuntimeError(f"{description} changed")

            with (
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "_verify_worker_inputs",
                    side_effect=verify_worker_inputs,
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "_verify_locked_file",
                    side_effect=verify_pending,
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "run_expert_function_consistency_probe",
                    return_value={"marker": "validated"},
                ),
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "validate_case_result"
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "Pending result.*changed"):
                    _run_device_cases(payload)

            self.assertFalse(result_path.exists())
            self.assertFalse(seal_path.exists())

    def test_published_result_cleans_matching_sealed_pending_in_safe_order(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            payload = self._payload(root)
            job = payload["jobs"][0]
            result_path = Path(job["result_path"])
            result_path.parent.mkdir(parents=True)
            published_result = {"marker": "published"}
            result_path.write_text(
                json.dumps(published_result) + "\n",
                encoding="utf-8",
            )
            pending_path = _device_pending_path(result_path, payload["device"])
            seal_path = _pending_seal_path(pending_path)
            pending_path.write_text("{}\n", encoding="utf-8")
            seal_path.write_text("{}\n", encoding="utf-8")

            unlink_order = []
            original_unlink = Path.unlink

            def record_unlink(path, *unlink_args, **unlink_kwargs):
                unlink_order.append(path)
                return original_unlink(path, *unlink_args, **unlink_kwargs)

            with (
                patch(
                    "analyses.run_expert_function_consistency_probe_batch."
                    "_load_sealed_pending",
                    return_value=(published_result, "a" * 64),
                ),
                patch.object(Path, "unlink", new=record_unlink),
            ):
                _cleanup_published_result_artifacts(
                    result_path,
                    payload["device"],
                    job["case"],
                    job["expected_run"],
                    published_result,
                )

            self.assertFalse(pending_path.exists())
            self.assertFalse(seal_path.exists())
            self.assertEqual(unlink_order[-2:], [seal_path, pending_path])

    def test_published_result_cleans_single_stale_artifact(self):
        for artifact in ("pending", "seal"):
            with (
                self.subTest(artifact=artifact),
                tempfile.TemporaryDirectory() as temporary_dir,
            ):
                root = Path(temporary_dir)
                payload = self._payload(root)
                job = payload["jobs"][0]
                result_path = Path(job["result_path"])
                result_path.parent.mkdir(parents=True)
                published_result = {"marker": "published"}
                pending_path = _device_pending_path(
                    result_path,
                    payload["device"],
                )
                seal_path = _pending_seal_path(pending_path)
                stale_path = pending_path if artifact == "pending" else seal_path
                stale_path.write_text("{}\n", encoding="utf-8")

                _cleanup_published_result_artifacts(
                    result_path,
                    payload["device"],
                    job["case"],
                    job["expected_run"],
                    published_result,
                )

                self.assertFalse(pending_path.exists())
                self.assertFalse(seal_path.exists())

    def test_published_result_preserves_different_sealed_pending(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            payload = self._payload(root)
            job = payload["jobs"][0]
            result_path = Path(job["result_path"])
            result_path.parent.mkdir(parents=True)
            pending_path = _device_pending_path(result_path, payload["device"])
            seal_path = _pending_seal_path(pending_path)
            pending_path.write_text("{}\n", encoding="utf-8")
            seal_path.write_text("{}\n", encoding="utf-8")

            with patch(
                "analyses.run_expert_function_consistency_probe_batch."
                "_load_sealed_pending",
                return_value=({"marker": "different"}, "a" * 64),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "Published and sealed pending results differ",
                ):
                    _cleanup_published_result_artifacts(
                        result_path,
                        payload["device"],
                        job["case"],
                        job["expected_run"],
                        {"marker": "published"},
                    )

            self.assertTrue(pending_path.is_file())
            self.assertTrue(seal_path.is_file())


if __name__ == "__main__":
    unittest.main()
