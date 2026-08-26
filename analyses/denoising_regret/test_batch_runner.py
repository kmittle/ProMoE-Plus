import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from analyses import run_denoising_regret_probe_batch as batch_runner
from analyses.denoising_regret.batch import FDRR_GATE_MODEL_NAME
from analyses.run_denoising_regret_probe_batch import (
    _json_payload_sha256,
    _load_existing_result,
    _load_published_result,
    _pending_result_path,
    _pending_seal,
    _pending_seal_path,
    _publish_summary,
    _result_seal,
    _result_seal_path,
    _verify_protocol_inputs,
    _write_or_validate_protocol,
    sha256_file,
)


class DenoisingRegretBatchRunnerTests(unittest.TestCase):
    @staticmethod
    def _main_args(root):
        checkpoint = root / "ckpt_step_10000.pth"
        weights = root / "weights.pth"
        checkpoint.write_text("checkpoint", encoding="utf-8")
        weights.write_text("weights", encoding="utf-8")
        return SimpleNamespace(
            ckpt=str(checkpoint),
            weights_ckpt=str(weights),
            manifest=str(root / "manifest.json"),
            latent_root=str(root / "latents"),
            num_threads=1,
            output_dir=str(root / "output"),
            overwrite_cases=False,
        )

    @staticmethod
    def _manifest(root):
        latent = root / "case.latent.npz"
        latent.write_text("latent", encoding="utf-8")
        manifest_path = root / "manifest.json"
        manifest_path.write_text("{}\n", encoding="utf-8")
        return {
            "name": "fdrr_gate_v1",
            "path": str(manifest_path),
            "cases": [
                {
                    "id": f"case{index:02d}",
                    "latent": str(latent),
                    "latent_key": "latent",
                    "label": index,
                    "seed": index,
                }
                for index in range(6)
            ],
        }

    def test_main_reaches_locked_model_validation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = self._main_args(root)
            parser = Mock()
            parser.parse_args.return_value = args
            with (
                patch.object(batch_runner, "build_parser", return_value=parser),
                patch.object(batch_runner, "parse_checkpoint_step", return_value=10000),
                patch.object(batch_runner, "load_manifest", return_value=self._manifest(root)),
                patch.object(
                    batch_runner,
                    "resolve_config_from_checkpoint",
                    return_value=root / "config.yaml",
                ),
                patch.object(
                    batch_runner,
                    "load_runtime_cfg",
                    return_value=SimpleNamespace(model_name="wrong_model"),
                ),
            ):
                with self.assertRaisesRegex(ValueError, FDRR_GATE_MODEL_NAME):
                    batch_runner.main()

    def test_tail_verification_failure_cannot_publish_cases(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            args = self._main_args(root)
            manifest = self._manifest(root)
            parser = Mock()
            parser.parse_args.return_value = args
            protocol_path = root / "output" / "protocol.json"
            protocol = {
                "checkpoint": {"sha256": "a" * 64},
                "weights_checkpoint": {"sha256": "b" * 64},
                "config": {"sha256": "c" * 64},
                "manifest": {"sha256": "d" * 64},
                "project_source_sha256": {},
            }

            with (
                patch.object(batch_runner, "build_parser", return_value=parser),
                patch.object(batch_runner, "parse_checkpoint_step", return_value=10000),
                patch.object(batch_runner, "load_manifest", return_value=manifest),
                patch.object(
                    batch_runner,
                    "resolve_config_from_checkpoint",
                    return_value=root / "config.yaml",
                ),
                patch.object(
                    batch_runner,
                    "load_runtime_cfg",
                    return_value=SimpleNamespace(model_name=FDRR_GATE_MODEL_NAME),
                ),
                patch.object(batch_runner, "_build_protocol", return_value=protocol),
                patch.object(
                    batch_runner,
                    "_write_or_validate_protocol",
                    return_value=(protocol_path, "e" * 64),
                ),
                patch.object(
                    batch_runner,
                    "_verify_protocol_inputs",
                    side_effect=[None, RuntimeError("checkpoint changed")],
                ),
                patch.object(
                    batch_runner,
                    "run_probe",
                    side_effect=lambda **kwargs: {"label": kwargs["label"]},
                ),
                patch.object(batch_runner, "_validate_result_contract"),
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint changed"):
                    batch_runner.main()

            cases_dir = Path(args.output_dir) / "cases"
            for index, case in enumerate(manifest["cases"], start=1):
                result_path = cases_dir / f"{index:02d}_{case['id']}.json"
                self.assertFalse(result_path.exists())
                self.assertTrue(_pending_result_path(result_path).is_file())

    def test_published_result_content_is_bound_to_seal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            result_path = root / "case.json"
            case = {
                "id": "case00",
                "latent_sha256": "a" * 64,
            }
            protocol_sha256 = "b" * 64
            result = {"marker": "locked"}
            batch_runner.write_json_atomic(result_path, result)
            batch_runner.write_json_atomic(
                _result_seal_path(result_path),
                _result_seal(
                    _json_payload_sha256(result),
                    case,
                    protocol_sha256,
                ),
            )
            self.assertEqual(
                _load_published_result(
                    result_path,
                    case,
                    {},
                    protocol_sha256,
                ),
                result,
            )

            batch_runner.write_json_atomic(result_path, {"marker": "changed"})
            with self.assertRaisesRegex(ValueError, "seal is incompatible"):
                _load_published_result(
                    result_path,
                    case,
                    {},
                    protocol_sha256,
                )

    def test_existing_summary_cannot_be_rebased(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            summary_path = Path(temporary_dir) / "summary.json"
            existing = {"gate": {"passed": False}, "case_results": ["old"]}
            replacement = {"gate": {"passed": True}, "case_results": ["new"]}
            batch_runner.write_json_atomic(summary_path, existing)
            with self.assertRaisesRegex(ValueError, "differs from recomputation"):
                _publish_summary(summary_path, replacement, False)
            self.assertEqual(
                json.loads(summary_path.read_text(encoding="utf-8")),
                existing,
            )

    def _run_overwrite_case(self, root, published_state, resume_pending):
        args = self._main_args(root)
        args.overwrite_cases = True
        manifest = self._manifest(root)
        manifest["cases"] = manifest["cases"][:1]
        case = {
            **manifest["cases"][0],
            "latent_sha256": sha256_file(manifest["cases"][0]["latent"]),
        }
        parser = Mock()
        parser.parse_args.return_value = args
        protocol_sha256 = "e" * 64
        protocol_path = Path(args.output_dir) / "protocol.json"
        protocol = {
            "checkpoint": {"sha256": "a" * 64},
            "weights_checkpoint": {"sha256": "b" * 64},
            "config": {"sha256": "c" * 64},
            "manifest": {"sha256": "d" * 64},
            "project_source_sha256": {},
        }
        requirements = {
            **batch_runner.FDRR_GATE_REQUIREMENTS,
            "min_cases": 1,
        }
        result_path = Path(args.output_dir) / "cases" / f"01_{case['id']}.json"
        result_seal_path = _result_seal_path(result_path)
        pending_path = _pending_result_path(result_path)
        pending_seal_path = _pending_seal_path(pending_path)
        old_result = [] if published_state == "non_object" else {"marker": "old"}
        new_result = {"marker": "new"}
        if published_state != "orphan_seal":
            batch_runner.write_json_atomic(result_path, old_result)
        batch_runner.write_json_atomic(
            result_seal_path,
            _result_seal(
                _json_payload_sha256(old_result),
                case,
                protocol_sha256,
            ),
        )
        if resume_pending:
            batch_runner.write_json_atomic(pending_path, new_result)
            batch_runner.write_json_atomic(
                pending_seal_path,
                _pending_seal(
                    _json_payload_sha256(new_result),
                    case,
                    protocol_sha256,
                ),
            )

        probe = Mock(return_value=new_result)
        unlink_order = []
        original_unlink = Path.unlink
        validate_result_contract = batch_runner._validate_result_contract

        def validate_type_only(result, expected, description):
            if not isinstance(result, dict):
                validate_result_contract(result, expected, description)

        def record_unlink(path, *unlink_args, **unlink_kwargs):
            unlink_order.append(path)
            return original_unlink(path, *unlink_args, **unlink_kwargs)

        with (
            patch.object(batch_runner, "build_parser", return_value=parser),
            patch.object(batch_runner, "parse_checkpoint_step", return_value=10000),
            patch.object(batch_runner, "load_manifest", return_value=manifest),
            patch.object(
                batch_runner,
                "resolve_config_from_checkpoint",
                return_value=root / "config.yaml",
            ),
            patch.object(
                batch_runner,
                "load_runtime_cfg",
                return_value=SimpleNamespace(model_name=FDRR_GATE_MODEL_NAME),
            ),
            patch.object(batch_runner, "FDRR_GATE_REQUIREMENTS", requirements),
            patch.object(batch_runner, "_build_protocol", return_value=protocol),
            patch.object(
                batch_runner,
                "_write_or_validate_protocol",
                return_value=(protocol_path, protocol_sha256),
            ),
            patch.object(batch_runner, "_verify_protocol_inputs"),
            patch.object(
                batch_runner,
                "_validate_result_contract",
                side_effect=validate_type_only,
            ),
            patch.object(batch_runner, "run_probe", probe),
            patch.object(
                batch_runner,
                "build_gate_summary",
                return_value={"checks": {}, "passed": True},
            ),
            patch.object(Path, "unlink", new=record_unlink),
        ):
            batch_runner.main()

        if resume_pending:
            probe.assert_not_called()
        else:
            probe.assert_called_once()
        self.assertEqual(
            json.loads(result_path.read_text(encoding="utf-8")),
            new_result,
        )
        self.assertTrue(result_seal_path.is_file())
        self.assertFalse(pending_path.exists())
        self.assertFalse(pending_seal_path.exists())
        paired_cleanup = [
            path
            for path in unlink_order
            if path in {pending_path, pending_seal_path}
        ]
        self.assertEqual(paired_cleanup[-2:], [pending_seal_path, pending_path])
        summary = json.loads(
            (Path(args.output_dir) / "summary.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            summary["case_results"][0]["sha256"],
            sha256_file(result_path),
        )

    def test_overwrite_resumes_matching_sealed_pending(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            self._run_overwrite_case(
                Path(temporary_dir),
                published_state="published",
                resume_pending=True,
            )

    def test_sealed_pending_replaces_orphan_published_seal(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            self._run_overwrite_case(
                Path(temporary_dir),
                published_state="orphan_seal",
                resume_pending=True,
            )

    def test_overwrite_recomputes_non_object_published_result(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            self._run_overwrite_case(
                Path(temporary_dir),
                published_state="non_object",
                resume_pending=False,
            )

    def test_cached_result_requires_content_provenance(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_path = Path(temporary_dir) / "case.json"
            result_path.write_text(
                json.dumps({"checkpoint_sha256": "a" * 64}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "checkpoint_sha256"):
                _load_existing_result(
                    result_path,
                    {"checkpoint_sha256": "b" * 64},
                )

    def test_cached_result_must_be_json_object(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            result_path = Path(temporary_dir) / "case.json"
            result_path.write_text("[]\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "must be a JSON object"):
                _load_existing_result(result_path, {})

    def test_legacy_results_require_explicit_protocol_replacement(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir)
            cases_dir = output_dir / "cases"
            cases_dir.mkdir()
            (cases_dir / "01_case.json").write_text("{}\n", encoding="utf-8")
            protocol = {"protocol_version": 1, "marker": "locked"}

            with self.assertRaisesRegex(RuntimeError, "no locked protocol"):
                _write_or_validate_protocol(output_dir, protocol, False)

            protocol_path, protocol_sha256 = _write_or_validate_protocol(
                output_dir,
                protocol,
                True,
            )
            self.assertEqual(sha256_file(protocol_path), protocol_sha256)
            self.assertEqual(
                (output_dir / "protocol.sha256").read_text(encoding="utf-8"),
                f"{protocol_sha256}  protocol.json\n",
            )

    def test_protocol_verification_detects_in_place_input_change(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            files = {}
            for name, content in (
                ("checkpoint", "checkpoint"),
                ("weights", "weights"),
                ("config", "config"),
                ("manifest", "manifest"),
                ("latent", "latent"),
            ):
                path = root / name
                path.write_text(content, encoding="utf-8")
                files[name] = path
            protocol = {
                "checkpoint": {
                    "path": str(files["checkpoint"]),
                    "sha256": sha256_file(files["checkpoint"]),
                },
                "weights_checkpoint": {
                    "path": str(files["weights"]),
                    "sha256": sha256_file(files["weights"]),
                },
                "config": {
                    "path": str(files["config"]),
                    "sha256": sha256_file(files["config"]),
                },
                "manifest": {
                    "path": str(files["manifest"]),
                    "sha256": sha256_file(files["manifest"]),
                    "cases": [{
                        "id": "case00",
                        "latent": str(files["latent"]),
                        "latent_sha256": sha256_file(files["latent"]),
                    }],
                },
                "project_source_sha256": {},
            }
            protocol_path, protocol_sha256 = _write_or_validate_protocol(
                root / "output",
                protocol,
                False,
            )
            _verify_protocol_inputs(protocol, protocol_path, protocol_sha256)

            files["config"].write_text("changed", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Config changed"):
                _verify_protocol_inputs(protocol, protocol_path, protocol_sha256)


if __name__ == "__main__":
    unittest.main()
