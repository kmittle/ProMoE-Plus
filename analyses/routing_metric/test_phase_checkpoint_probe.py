import itertools
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from analyses.denoising_regret.io import write_json_atomic
from analyses.routing_metric.phase_checkpoint_probe import (
    CANONICAL_SPEC_SHA256,
    CONFIRMATORY_REQUIREMENTS,
    DISCOVERY_REQUIREMENTS,
    FactorialRouterOverride,
    build_gate_summary,
    load_gate_spec,
    run_exact_dispatch_case,
    run_factorial_case,
    sha256_file,
)
from analyses.run_phase_metric_checkpoint_probe import (
    DEFAULT_SPEC,
    PROBE_VERSION,
    RUNNER_VERSION,
    _json_payload_sha256,
    _load_published_result,
    _merge_device_shards,
    _recover_result_publication,
    _require_confirmatory_unlock,
    _validate_result,
    _validate_matched_configs,
)
from models.models_ProMoE_TC import DiT


PROJECT_ROOT = Path(__file__).resolve().parents[2]


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _model_kwargs():
    return {
        "input_size": 8,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "learn_sigma": False,
        "MoE_config": _AttrDict(
            num_routed_experts=4,
            moe_intermediate_size=48,
            shared_expert_intermediate_size=48,
            load_balance_loss_coef=0,
            norm_topk_prob=False,
            seq_aux=False,
            use_shared_expert=True,
            interleave=True,
            init_MoeMLP=False,
            top_k=1,
            router_weight_mode="identity",
            routing_contrastive_lam=1.0,
            use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07,
            phase_metric_config={
                "enabled": True,
                "rank": 4,
                "num_fourier_bands": 2,
                "num_train_timesteps": 1000,
                "scale": 0.25,
                "shuffle_timestep": False,
                "init_seed": 1729,
            },
        ),
    }


def _candidate_model():
    torch.manual_seed(17)
    model = DiT(**_model_kwargs()).eval().requires_grad_(False)
    metric = model.blocks[1].mlp.phase_metric
    generator = torch.Generator().manual_seed(19)
    with torch.no_grad():
        metric.phase_to_rank.copy_(
            0.8 * torch.randn(
                metric.phase_to_rank.shape,
                generator=generator,
            )
        )
        metric.expert_gain.copy_(
            0.4 * torch.randn(metric.expert_gain.shape, generator=generator)
        )
    return model


def _requirements(split):
    keys = (
        DISCOVERY_REQUIREMENTS
        if split == "discovery"
        else CONFIRMATORY_REQUIREMENTS
    )
    requirements = {}
    for key in keys:
        if key == "minimum_exact_probe_count":
            requirements[key] = 2
        elif key.startswith("maximum_"):
            requirements[key] = 0.0
        else:
            requirements[key] = 0.0
    return requirements


def _summary_spec():
    return {
        "block_indices": (1,),
        "sigmas": (0.5,),
        "protocol": {
            "split_counts": {"discovery": 2, "confirmatory": 2},
            "bootstrap_resamples": 2000,
            "bootstrap_seeds": {"discovery": 31, "confirmatory": 37},
            "requirements": {
                "discovery": _requirements("discovery"),
                "confirmatory": _requirements("confirmatory"),
            },
        },
    }


def _case_result(case_id, phase_phase=0.8):
    case_number = sum(case_id.encode("utf-8"))
    records = [
        {
            "sigma": 0.5,
            "block_index": 1,
            "phase_score_preference": 0.2 + 0.1 * index,
            "exact_base_minus_phase_mse": 0.1 + 0.01 * index,
            "exact_phase_route_relative_gain": 0.1 + 0.01 * index,
        }
        for index in range(2)
    ]
    return {
        "case_id": case_id,
        "split": "discovery",
        "label": 7,
        "seed": 1000 + case_number,
        "latent_sha256": f"{case_number:064x}",
        "factorial": {
            "mode_mse": {
                "phase_phase": [phase_phase],
                "phase_base": [0.8],
                "base_phase": [1.2],
                "base_base": [1.0],
                "shuffled_phase_phase": [1.2],
            },
            "route_rows": [{
                "block_index": 1,
                "sigma": 0.5,
                "token_count": 10,
                "flip_count": 2,
            }],
            "native_override_max_abs_output_change": 0.0,
            "native_override_max_abs_mse_change": 0.0,
        },
        "exact_dispatch": {
            "records": records,
            "cells": [{
                "block_index": 1,
                "sigma": 0.5,
                "token_count": 10,
                "flip_count": 2,
            }],
            "noop_max_abs_mse_change": 0.0,
        },
        "base_checkpoint_mse": [1.1],
    }


def _published_result(cases, gate):
    return {
        "runner_version": RUNNER_VERSION,
        "probe_version": PROBE_VERSION,
        "protocol_sha256": "a" * 64,
        "split": "discovery",
        "probe": {"probe_version": PROBE_VERSION, "cases": cases},
        "gate": gate,
    }


def _result_protocol(cases):
    return {
        "cases": [
            {
                "id": case["case_id"],
                "split": case["split"],
                "label": case["label"],
                "seed": case["seed"],
                "latent_sha256": case["latent_sha256"],
            }
            for case in cases
        ]
    }


class PhaseCheckpointProbeTests(unittest.TestCase):
    def test_phase_phase_override_matches_native_router_and_model(self):
        model = _candidate_model()
        moe = model.blocks[1].mlp
        generator = torch.Generator().manual_seed(23)
        hidden = torch.randn(2, 16, 32, generator=generator)
        labels = torch.tensor([7, 1000])
        timesteps = torch.tensor([200.0, 800.0])
        native_router = moe.compute_router(hidden, labels, timesteps)
        with FactorialRouterOverride(model, "phase", "phase") as override:
            observed_router = moe.compute_router(hidden, labels, timesteps)
        for native, observed in zip(native_router, observed_router):
            if native is None:
                self.assertIsNone(observed)
            else:
                torch.testing.assert_close(observed, native, rtol=0, atol=0)
        self.assertIn(1, override.captures)

        inputs = torch.randn(2, 4, 8, 8, generator=generator)
        with torch.inference_mode():
            native_output = model(inputs, timesteps, context=labels)
            with FactorialRouterOverride(model, "phase", "phase"):
                override_output = model(inputs, timesteps, context=labels)
        torch.testing.assert_close(override_output, native_output, rtol=0, atol=0)

    def test_factorial_case_runs_mixed_sigmas_and_matches_native(self):
        model = _candidate_model()
        with tempfile.TemporaryDirectory() as directory:
            latent = Path(directory) / "toy.latent.npz"
            generator = np.random.default_rng(29)
            np.savez(
                latent,
                latent=generator.standard_normal((8, 8, 8)).astype(np.float32),
            )
            case = {
                "id": "toy",
                "seed": 31,
                "label": 7,
                "latent": str(latent),
            }
            spec = {
                "sigmas": (0.2, 0.5, 0.8),
                "block_indices": (1,),
                "protocol": {"phase_shuffle_offset": 1},
            }
            result = run_factorial_case(model, case, spec, torch.device("cpu"))
        self.assertEqual(set(result["mode_mse"]), {
            "phase_phase",
            "phase_base",
            "base_phase",
            "base_base",
            "shuffled_phase_phase",
        })
        self.assertEqual(result["native_override_max_abs_output_change"], 0.0)
        self.assertEqual(result["native_override_max_abs_mse_change"], 0.0)
        self.assertEqual(len(result["route_rows"]), 3)

    def test_exact_dispatch_keeps_native_weight_and_has_exact_noop(self):
        model = _candidate_model()
        model.blocks[1].mlp.phase_metric.scale = 4.0
        with tempfile.TemporaryDirectory() as directory:
            latent = Path(directory) / "toy.latent.npz"
            generator = np.random.default_rng(41)
            np.savez(
                latent,
                latent=generator.standard_normal((8, 8, 8)).astype(np.float32),
            )
            case = {
                "id": "toy-exact",
                "seed": 43,
                "label": 11,
                "latent": str(latent),
            }
            spec = {
                "sigmas": (0.2, 0.5, 0.8),
                "block_indices": (1,),
                "protocol": {
                    "tokens_per_cell": 2,
                    "noop_tokens_per_cell": 1,
                    "exact_batch_size": 2,
                },
            }
            result = run_exact_dispatch_case(
                model, case, spec, torch.device("cpu")
            )
        self.assertEqual(len(result["cells"]), 3)
        self.assertEqual(result["noop_max_abs_mse_change"], 0.0)
        self.assertGreater(sum(cell["flip_count"] for cell in result["cells"]), 0)
        self.assertGreater(len(result["records"]), 0)
        self.assertTrue(all(
            np.isfinite(record["exact_base_minus_phase_mse"])
            for record in result["records"]
        ))

    def test_direct_native_vs_base_route_check_catches_bad_interaction(self):
        spec = _summary_spec()
        passing_cases = [_case_result("a"), _case_result("b")]
        passing = build_gate_summary(passing_cases, spec, "discovery")
        self.assertTrue(passing["passed"])
        self.assertEqual(passing["decision"], "authorize_confirmatory")
        confirmatory = build_gate_summary(passing_cases, spec, "confirmatory")
        self.assertEqual(
            confirmatory["decision"], "authorize_continue_training"
        )
        self.assertTrue(
            passing["checks"][
                "mean_native_vs_base_base_relative_gain"
            ]["passed"]
        )

        failing_cases = [
            _case_result("a", phase_phase=1.1),
            _case_result("b", phase_phase=1.1),
        ]
        failing = build_gate_summary(failing_cases, spec, "discovery")
        self.assertFalse(failing["passed"])
        self.assertFalse(
            failing["checks"][
                "mean_native_vs_base_base_relative_gain"
            ]["passed"]
        )
        self.assertTrue(failing["checks"]["mean_selection_relative_gain"]["passed"])

    def test_gate_rejects_nonfinite_raw_values_before_aggregation(self):
        def set_noop_nan(cases):
            cases[1]["exact_dispatch"]["noop_max_abs_mse_change"] = float("nan")

        def set_exact_nan(cases):
            cases[1]["exact_dispatch"]["records"][0][
                "exact_base_minus_phase_mse"
            ] = float("nan")

        def set_invariant_nan(cases):
            cases[1]["factorial"][
                "native_override_max_abs_output_change"
            ] = float("nan")

        for name, mutate in (
            ("noop", set_noop_nan),
            ("exact", set_exact_nan),
            ("invariant", set_invariant_nan),
        ):
            with self.subTest(name=name):
                cases = [_case_result("a"), _case_result("b")]
                mutate(cases)
                with self.assertRaises(RuntimeError):
                    build_gate_summary(cases, _summary_spec(), "discovery")

    def test_real_gate_spec_has_the_exact_locked_schema(self):
        spec = load_gate_spec(DEFAULT_SPEC, PROJECT_ROOT)
        self.assertEqual(sha256_file(DEFAULT_SPEC), CANONICAL_SPEC_SHA256)
        self.assertEqual(
            set(spec["protocol"]["requirements"]["discovery"]),
            DISCOVERY_REQUIREMENTS,
        )
        self.assertEqual(
            set(spec["protocol"]["requirements"]["confirmatory"]),
            CONFIRMATORY_REQUIREMENTS,
        )
        _validate_matched_configs(
            PROJECT_ROOT / "configs" / "004_ProMoE_B_phase_metric.yaml",
            PROJECT_ROOT / "configs" / "004_ProMoE_B_phase_metric_base_s0.yaml",
            spec,
        )

    def test_gate_spec_rejects_an_alternate_path(self):
        with tempfile.TemporaryDirectory() as directory:
            copied_spec = Path(directory) / DEFAULT_SPEC.name
            copied_spec.write_bytes(DEFAULT_SPEC.read_bytes())
            with self.assertRaises(ValueError):
                load_gate_spec(copied_spec, PROJECT_ROOT)

    def test_result_cases_must_match_locked_identity_and_order(self):
        spec = _summary_spec()
        cases = [_case_result("a"), _case_result("b")]
        protocol = _result_protocol(cases)
        gate = build_gate_summary(cases, spec, "discovery")
        result = _published_result(cases, gate)
        _validate_result(result, "discovery", spec, "a" * 64, protocol)

        reordered = _published_result(list(reversed(cases)), gate)
        with self.assertRaises(ValueError):
            _validate_result(
                reordered, "discovery", spec, "a" * 64, protocol
            )

        relabeled_cases = [dict(case) for case in cases]
        relabeled_cases[0]["label"] += 1
        relabeled = _published_result(relabeled_cases, gate)
        with self.assertRaises(ValueError):
            _validate_result(
                relabeled, "discovery", spec, "a" * 64, protocol
            )

    def test_confirmatory_requires_a_valid_passing_discovery_result(self):
        spec = _summary_spec()
        cases = [_case_result("a"), _case_result("b")]
        protocol = _result_protocol(cases)
        gate = build_gate_summary(cases, spec, "discovery")
        result = _published_result(cases, gate)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            with self.assertRaises(FileNotFoundError):
                _require_confirmatory_unlock(
                    output_dir, spec, "a" * 64, protocol
                )

            result_path = output_dir / "discovery-result.json"
            write_json_atomic(result_path, result)
            result_hash = _json_payload_sha256(result)
            (output_dir / "discovery-result.sha256").write_text(
                f"{result_hash}  discovery-result.json\n",
                encoding="utf-8",
            )
            _require_confirmatory_unlock(
                output_dir, spec, "a" * 64, protocol
            )

            result["gate"]["passed"] = False
            write_json_atomic(result_path, result)
            result_hash = _json_payload_sha256(result)
            (output_dir / "discovery-result.sha256").write_text(
                f"{result_hash}  discovery-result.json\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                _require_confirmatory_unlock(
                    output_dir, spec, "a" * 64, protocol
                )

            failing_cases = [
                _case_result("a", phase_phase=1.1),
                _case_result("b", phase_phase=1.1),
            ]
            failing_gate = build_gate_summary(
                failing_cases, spec, "discovery"
            )
            failing_result = _published_result(failing_cases, failing_gate)
            write_json_atomic(result_path, failing_result)
            result_hash = _json_payload_sha256(failing_result)
            (output_dir / "discovery-result.sha256").write_text(
                f"{result_hash}  discovery-result.json\n",
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                _require_confirmatory_unlock(
                    output_dir, spec, "a" * 64, protocol
                )

    def test_result_publication_state_matrix_fails_closed(self):
        spec = _summary_spec()
        cases = [_case_result("a"), _case_result("b")]
        protocol = _result_protocol(cases)
        gate = build_gate_summary(cases, spec, "discovery")
        result = _published_result(cases, gate)
        result_hash = _json_payload_sha256(result)
        names = ("result", "seal", "pending_result", "pending_seal")
        expected = {
            frozenset(): "absent",
            frozenset({"pending_result"}): "recovered",
            frozenset({"pending_result", "pending_seal"}): "recovered",
            frozenset({"result", "pending_seal"}): "recovered",
            frozenset({"result", "seal"}): "published",
        }
        for bits in itertools.product((False, True), repeat=len(names)):
            state_names = frozenset(
                name for name, enabled in zip(names, bits) if enabled
            )
            with self.subTest(state=sorted(state_names)):
                with tempfile.TemporaryDirectory() as directory:
                    output_dir = Path(directory)
                    paths = {
                        "result": output_dir / "discovery-result.json",
                        "seal": output_dir / "discovery-result.sha256",
                        "pending_result": (
                            output_dir / "discovery-result.json.pending"
                        ),
                        "pending_seal": (
                            output_dir / "discovery-result.sha256.pending"
                        ),
                    }
                    for name in state_names:
                        if name in {"result", "pending_result"}:
                            write_json_atomic(paths[name], result)
                        else:
                            paths[name].write_text(
                                f"{result_hash}  discovery-result.json\n",
                                encoding="utf-8",
                            )
                    call = lambda: _recover_result_publication(
                        output_dir,
                        "discovery",
                        spec,
                        "a" * 64,
                        protocol,
                    )
                    if state_names not in expected:
                        with self.assertRaises(RuntimeError):
                            call()
                        continue
                    self.assertEqual(call(), expected[state_names])
                    if expected[state_names] == "absent":
                        continue
                    _, observed_gate = _load_published_result(
                        output_dir,
                        "discovery",
                        spec,
                        "a" * 64,
                        protocol,
                    )
                    self.assertEqual(observed_gate, gate)

    def test_parallel_shards_restore_manifest_order(self):
        metadata = {
            "probe_version": PROBE_VERSION,
            "candidate_config": "candidate.yaml",
            "base_config": "base.yaml",
            "candidate_contract": [{"block_index": 1}],
        }
        shards = [
            {
                "device": "cuda:1",
                "result": {**metadata, "cases": [{"case_id": "b"}]},
            },
            {
                "device": "cuda:0",
                "result": {**metadata, "cases": [{"case_id": "a"}]},
            },
        ]
        merged = _merge_device_shards(
            shards,
            [{"id": "a"}, {"id": "b"}],
            ("cuda:0", "cuda:1"),
        )
        self.assertEqual(
            [case["case_id"] for case in merged["cases"]],
            ["a", "b"],
        )
        self.assertEqual(
            [item["device"] for item in shards],
            ["cuda:1", "cuda:0"],
        )


if __name__ == "__main__":
    unittest.main()
