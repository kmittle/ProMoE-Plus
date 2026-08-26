"""Run the denoising-regret launch gate over a fixed case manifest."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from analyses.denoising_regret import run_probe
from analyses.denoising_regret.batch import (
    BATCH_VERSION,
    FDRR_GATE_BLOCK_INDEX,
    FDRR_GATE_EXACT_BATCH_SIZE,
    FDRR_GATE_MIN_CHECKPOINT_STEP,
    FDRR_GATE_MODEL_NAME,
    FDRR_GATE_REQUIREMENTS,
    FDRR_GATE_SIGMAS,
    REQUIRED_PROBE_VERSION,
    build_gate_summary,
    load_manifest,
)
from analyses.denoising_regret.io import write_json_atomic
from analyses.denoising_regret.probe import _build_model
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)


PROTOCOL_VERSION = 1
PENDING_SEAL_VERSION = 1
RESULT_SEAL_VERSION = 1
STATIC_SOURCE_PATHS = (
    "requirements.txt",
    "analyses/run_denoising_regret_probe_batch.py",
    "analyses/denoising_regret/batch.py",
    "analyses/denoising_regret/io.py",
    "analyses/denoising_regret/probe.py",
)


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _module_source_path(module):
    raw_path = getattr(module, "__file__", None)
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.suffix in {".pyc", ".pyo"}:
        try:
            path = Path(importlib.util.source_from_cache(str(path)))
        except ValueError:
            return None
    try:
        return path.resolve()
    except OSError:
        return None


def _collect_project_source_hashes(runtime_cfg):
    with torch.random.fork_rng(devices=[]):
        model = _build_model(runtime_cfg)
    del model
    gc.collect()

    project_root = PROJECT_ROOT.resolve()
    relative_paths = set(STATIC_SOURCE_PATHS)
    for module in tuple(sys.modules.values()):
        if module is None:
            continue
        source_path = _module_source_path(module)
        if source_path is None or not source_path.is_file():
            continue
        try:
            relative = source_path.relative_to(project_root)
        except ValueError:
            continue
        relative_paths.add(relative.as_posix())
    return {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in sorted(relative_paths)
    }


def _write_text_atomic(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _json_payload_sha256(payload):
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _pending_result_path(result_path):
    return Path(f"{result_path}.pending")


def _pending_seal_path(pending_path):
    return Path(f"{pending_path}.seal")


def _pending_seal(pending_sha256, case, protocol_sha256):
    return {
        "version": PENDING_SEAL_VERSION,
        "case_id": case["id"],
        "latent_sha256": case["latent_sha256"],
        "protocol_sha256": protocol_sha256,
        "pending_sha256": pending_sha256,
    }


def _result_seal_path(result_path):
    return Path(f"{result_path}.seal")


def _result_seal(result_sha256, case, protocol_sha256):
    return {
        "version": RESULT_SEAL_VERSION,
        "case_id": case["id"],
        "latent_sha256": case["latent_sha256"],
        "protocol_sha256": protocol_sha256,
        "result_sha256": result_sha256,
    }


def _build_protocol(
    checkpoint_path,
    weights_checkpoint_path,
    config_path,
    manifest,
    runtime_cfg,
    output_dir,
    num_threads,
):
    return {
        "protocol_version": PROTOCOL_VERSION,
        "batch_version": BATCH_VERSION,
        "probe_version": REQUIRED_PROBE_VERSION,
        "checkpoint": {
            "path": str(checkpoint_path),
            "sha256": sha256_file(checkpoint_path),
            "step": parse_checkpoint_step(checkpoint_path),
        },
        "weights_checkpoint": {
            "path": str(weights_checkpoint_path),
            "sha256": sha256_file(weights_checkpoint_path),
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "model_name": runtime_cfg.model_name,
        },
        "manifest": {
            "path": manifest["path"],
            "sha256": sha256_file(manifest["path"]),
            "name": manifest["name"],
            "cases": [
                {
                    "id": case["id"],
                    "latent": case["latent"],
                    "latent_sha256": case["latent_sha256"],
                }
                for case in manifest["cases"]
            ],
        },
        "run": {
            "sigmas": list(FDRR_GATE_SIGMAS),
            "block_index": FDRR_GATE_BLOCK_INDEX,
            "candidate_mode": "mixed",
            "num_token_probes": FDRR_GATE_REQUIREMENTS[
                "required_token_probes_per_sigma"
            ],
            "exact_batch_size": FDRR_GATE_EXACT_BATCH_SIZE,
            "device": "cpu",
            "num_threads": int(num_threads),
            "output_dir": str(output_dir),
        },
        "gate_requirements": dict(FDRR_GATE_REQUIREMENTS),
        "project_source_sha256": _collect_project_source_hashes(runtime_cfg),
    }


def _write_or_validate_protocol(output_dir, protocol, overwrite_cases):
    output_dir = Path(output_dir)
    protocol_path = output_dir / "protocol.json"
    hash_path = output_dir / "protocol.sha256"
    published = list((output_dir / "cases").glob("*.json"))
    published.extend(path for path in (output_dir / "summary.json",) if path.exists())
    if not protocol_path.exists() and published and not overwrite_cases:
        raise RuntimeError(
            "Existing FDRR results have no locked protocol; use a fresh output "
            "directory or --overwrite-cases"
        )
    if protocol_path.exists():
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol and not overwrite_cases:
            raise ValueError(
                "Existing FDRR protocol differs; use a fresh output directory or "
                "--overwrite-cases"
            )
    if not protocol_path.exists() or overwrite_cases:
        write_json_atomic(protocol_path, protocol)
    protocol_sha256 = sha256_file(protocol_path)
    expected_line = f"{protocol_sha256}  protocol.json\n"
    if hash_path.exists() and not overwrite_cases:
        if hash_path.read_text(encoding="utf-8") != expected_line:
            raise ValueError(f"Protocol checksum differs: {hash_path}")
    else:
        _write_text_atomic(hash_path, expected_line)
    return protocol_path, protocol_sha256


def _verify_file(path, expected_sha256, description):
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"{description} changed after protocol lock: {Path(path).resolve()}"
        )


def _verify_protocol_inputs(protocol, protocol_path, protocol_sha256):
    _verify_file(protocol_path, protocol_sha256, "Protocol")
    _verify_file(
        protocol["checkpoint"]["path"],
        protocol["checkpoint"]["sha256"],
        "Checkpoint",
    )
    _verify_file(
        protocol["weights_checkpoint"]["path"],
        protocol["weights_checkpoint"]["sha256"],
        "Weights checkpoint",
    )
    _verify_file(
        protocol["config"]["path"],
        protocol["config"]["sha256"],
        "Config",
    )
    _verify_file(
        protocol["manifest"]["path"],
        protocol["manifest"]["sha256"],
        "Manifest",
    )
    for case in protocol["manifest"]["cases"]:
        _verify_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
    for relative, expected in protocol["project_source_sha256"].items():
        _verify_file(PROJECT_ROOT / relative, expected, f"Source {relative}")


def _default_output_dir(checkpoint_path, manifest_name):
    checkpoint_path = Path(checkpoint_path).resolve()
    step = checkpoint_path.stem.removeprefix("ckpt_step_")
    return (
        checkpoint_path.parent.parent
        / "sample"
        / f"step{step}"
        / "denoising_regret_probe_batch"
        / manifest_name
    )


def _validate_result_contract(result, expected, description):
    if not isinstance(result, dict):
        raise ValueError(f"{description} must be a JSON object")
    mismatches = []
    for key, value in expected.items():
        actual = result.get(key)
        if actual != value:
            mismatches.append(f"{key}: expected {value!r}, found {actual!r}")
    if mismatches:
        raise ValueError(
            f"{description} is incompatible: " + "; ".join(mismatches)
        )


def _load_existing_result(path, expected):
    result = json.loads(Path(path).read_text(encoding="utf-8"))
    _validate_result_contract(result, expected, f"Existing case result {path}")
    return result


def _load_published_result(path, case, expected, protocol_sha256):
    path = Path(path)
    seal_path = _result_seal_path(path)
    if not seal_path.is_file():
        raise RuntimeError(f"Published result has no seal: {path}")
    result = _load_existing_result(path, expected)
    result_sha256 = _json_payload_sha256(result)
    _verify_file(path, result_sha256, f"Published result for {case['id']}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected_seal = _result_seal(result_sha256, case, protocol_sha256)
    if seal != expected_seal:
        raise ValueError(f"Published result seal is incompatible: {seal_path}")
    _verify_file(
        seal_path,
        _json_payload_sha256(expected_seal),
        f"Published result seal for {case['id']}",
    )
    return result


def _load_sealed_pending(pending_path, case, expected, protocol_sha256):
    pending_path = Path(pending_path)
    seal_path = _pending_seal_path(pending_path)
    result = json.loads(pending_path.read_text(encoding="utf-8"))
    _validate_result_contract(
        result,
        expected,
        f"Sealed pending result {pending_path}",
    )
    pending_sha256 = _json_payload_sha256(result)
    _verify_file(
        pending_path,
        pending_sha256,
        f"Sealed pending result for {case['id']}",
    )
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    expected_seal = _pending_seal(pending_sha256, case, protocol_sha256)
    if seal != expected_seal:
        raise ValueError(f"Pending seal is incompatible: {seal_path}")
    return result, pending_sha256


def _publish_summary(summary_path, payload, overwrite):
    summary_path = Path(summary_path)
    pending_path = Path(f"{summary_path}.pending")
    expected_sha256 = _json_payload_sha256(payload)
    if summary_path.exists() and not overwrite:
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise ValueError(
                f"Existing FDRR summary differs from recomputation: {summary_path}"
            )
        _verify_file(summary_path, expected_sha256, "Existing FDRR summary")
        pending_path.unlink(missing_ok=True)
        return
    write_json_atomic(pending_path, payload)
    _verify_file(pending_path, expected_sha256, "Pending FDRR summary")
    os.replace(pending_path, summary_path)
    _verify_file(summary_path, expected_sha256, "Published FDRR summary")


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run probe v4 over a fixed ImageNet latent manifest and require every "
            "case/noise cell to pass the FDRR launch thresholds."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Checkpoint path")
    parser.add_argument(
        "--weights-ckpt",
        help="Optional local checkpoint copy used only for loading weights",
    )
    parser.add_argument("--manifest", required=True, help="Version-1 case manifest")
    parser.add_argument("--latent-root", required=True, help="ImageNet latent root")
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--overwrite-cases",
        action="store_true",
        help=(
            "Start or resume recomputation and atomically replace per-case "
            "results"
        ),
    )
    return parser


def main():
    args = build_parser().parse_args()
    checkpoint_path = Path(args.ckpt).resolve()
    weights_checkpoint_path = Path(args.weights_ckpt or args.ckpt).resolve()
    if not checkpoint_path.is_file() or not weights_checkpoint_path.is_file():
        raise FileNotFoundError("Checkpoint and weights checkpoint must exist")
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    if checkpoint_step < FDRR_GATE_MIN_CHECKPOINT_STEP:
        raise ValueError(
            f"FDRR gate requires checkpoint step >= "
            f"{FDRR_GATE_MIN_CHECKPOINT_STEP}, got {checkpoint_step}"
        )
    manifest = load_manifest(args.manifest, args.latent_root)
    manifest = {
        **manifest,
        "cases": [
            {**case, "latent_sha256": sha256_file(case["latent"])}
            for case in manifest["cases"]
        ],
    }
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else _default_output_dir(checkpoint_path, manifest["name"])
    )

    requirements = dict(FDRR_GATE_REQUIREMENTS)
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    if len(manifest["cases"]) != requirements["min_cases"]:
        raise ValueError(
            f"Manifest has {len(manifest['cases'])} cases, expected exactly "
            f"{requirements['min_cases']}"
        )

    config_path = resolve_config_from_checkpoint(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != FDRR_GATE_MODEL_NAME:
        raise ValueError(f"FDRR gate model must be {FDRR_GATE_MODEL_NAME}")
    protocol = _build_protocol(
        checkpoint_path=checkpoint_path,
        weights_checkpoint_path=weights_checkpoint_path,
        config_path=config_path,
        manifest=manifest,
        runtime_cfg=runtime_cfg,
        output_dir=output_dir,
        num_threads=args.num_threads,
    )
    protocol_path, protocol_sha256 = _write_or_validate_protocol(
        output_dir,
        protocol,
        args.overwrite_cases,
    )
    _verify_protocol_inputs(protocol, protocol_path, protocol_sha256)

    provenance = {
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "weights_checkpoint_sha256": protocol["weights_checkpoint"]["sha256"],
        "config_sha256": protocol["config"]["sha256"],
        "manifest_sha256": protocol["manifest"]["sha256"],
        "protocol_sha256": protocol_sha256,
        "project_source_sha256": protocol["project_source_sha256"],
    }

    case_contracts = []
    pending_results = []
    cases_dir = output_dir / "cases"
    for index, case in enumerate(manifest["cases"], start=1):
        result_path = cases_dir / f"{index:02d}_{case['id']}.json"
        pending_path = _pending_result_path(result_path)
        seal_path = _pending_seal_path(pending_path)
        result_seal_path = _result_seal_path(result_path)
        expected = {
            "probe_version": REQUIRED_PROBE_VERSION,
            "counterfactual_route_weight": "selected",
            "checkpoint": str(checkpoint_path),
            "weights_checkpoint": str(weights_checkpoint_path),
            "checkpoint_step": checkpoint_step,
            "weights_checkpoint_step": checkpoint_step,
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "weights_checkpoint_sha256": provenance[
                "weights_checkpoint_sha256"
            ],
            "config": str(config_path),
            "config_sha256": provenance["config_sha256"],
            "model_name": FDRR_GATE_MODEL_NAME,
            "manifest_sha256": provenance["manifest_sha256"],
            "latent_sha256": case["latent_sha256"],
            "protocol_sha256": protocol_sha256,
            "project_source_sha256": provenance["project_source_sha256"],
            "latent": case["latent"],
            "latent_key": case["latent_key"],
            "label": case["label"],
            "block_index": FDRR_GATE_BLOCK_INDEX,
            "candidate_mode": "mixed",
            "sigmas": list(FDRR_GATE_SIGMAS),
            "num_token_probes_requested": requirements[
                "required_token_probes_per_sigma"
            ],
            "exact_batch_size": FDRR_GATE_EXACT_BATCH_SIZE,
            "seed": case["seed"],
            "device": "cpu",
            "num_threads": int(args.num_threads),
            "batch_case": case,
        }
        case_contracts.append((result_path, case, expected))

        pending_result = None
        pending_sha256 = None
        if pending_path.exists():
            if seal_path.exists():
                try:
                    pending_result, pending_sha256 = _load_sealed_pending(
                        pending_path,
                        case,
                        expected,
                        protocol_sha256,
                    )
                except (OSError, RuntimeError, ValueError):
                    if not args.overwrite_cases:
                        raise
                    seal_path.unlink(missing_ok=True)
                    pending_path.unlink(missing_ok=True)
            else:
                pending_path.unlink()
                print(
                    f"[{index}/{len(manifest['cases'])}] Discarding unsealed "
                    f"pending {case['id']}"
                )

        published_result = None
        published_error = None
        if result_path.exists():
            try:
                published_result = _load_published_result(
                    result_path,
                    case,
                    expected,
                    protocol_sha256,
                )
            except (OSError, RuntimeError, ValueError) as error:
                published_error = error
        elif result_seal_path.exists():
            if pending_result is not None or args.overwrite_cases:
                result_seal_path.unlink()
            else:
                raise RuntimeError(
                    f"Published result seal has no result: {result_seal_path}"
                )

        if seal_path.exists() and not pending_path.exists():
            if published_result is not None or args.overwrite_cases:
                seal_path.unlink()
            else:
                raise RuntimeError(f"Pending seal has no pending result: {seal_path}")

        if pending_result is not None:
            if published_result is not None and published_result == pending_result:
                seal_path.unlink()
                pending_path.unlink()
                print(f"[{index}/{len(manifest['cases'])}] Reusing {case['id']}")
                gc.collect()
                continue
            if published_result is not None and not args.overwrite_cases:
                raise RuntimeError(
                    f"Published and pending results differ for {case['id']}"
                )
            pending_results.append((
                pending_path,
                seal_path,
                result_path,
                case,
                expected,
                pending_result,
                pending_sha256,
            ))
            print(
                f"[{index}/{len(manifest['cases'])}] Reusing sealed pending "
                f"{case['id']}"
            )
            gc.collect()
            continue

        if published_result is not None and not args.overwrite_cases:
            print(f"[{index}/{len(manifest['cases'])}] Reusing {case['id']}")
            gc.collect()
            continue
        if published_error is not None and not args.overwrite_cases:
            raise published_error

        print(f"[{index}/{len(manifest['cases'])}] Probing {case['id']}")
        result = run_probe(
            checkpoint_path=checkpoint_path,
            weights_checkpoint_path=weights_checkpoint_path,
            latent_path=case["latent"],
            latent_key=case["latent_key"],
            label=case["label"],
            sigmas=FDRR_GATE_SIGMAS,
            block_index=FDRR_GATE_BLOCK_INDEX,
            num_token_probes=requirements[
                "required_token_probes_per_sigma"
            ],
            candidate_mode="mixed",
            exact_batch_size=FDRR_GATE_EXACT_BATCH_SIZE,
            seed=case["seed"],
            device="cpu",
            num_threads=args.num_threads,
        )
        result.update(provenance)
        result["latent_sha256"] = case["latent_sha256"]
        result["batch_case"] = case
        _verify_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        _validate_result_contract(
            result,
            expected,
            f"Fresh case result for {case['id']}",
        )
        pending_sha256 = _json_payload_sha256(result)
        write_json_atomic(pending_path, result)
        _verify_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        pending_results.append((
            pending_path,
            seal_path,
            result_path,
            case,
            expected,
            result,
            pending_sha256,
        ))
        gc.collect()

    _verify_protocol_inputs(protocol, protocol_path, protocol_sha256)

    for pending in pending_results:
        pending_path, _, _, case, expected, result, pending_sha256 = pending
        _verify_file(
            case["latent"],
            case["latent_sha256"],
            f"Latent for {case['id']}",
        )
        _verify_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        reloaded = json.loads(pending_path.read_text(encoding="utf-8"))
        _validate_result_contract(
            reloaded,
            expected,
            f"Pending result for {case['id']}",
        )
        if reloaded != result:
            raise RuntimeError(f"Pending result snapshot changed for {case['id']}")

    sealed_results = []
    for pending in pending_results:
        pending_path, seal_path, _, case, _, _, pending_sha256 = pending
        seal = _pending_seal(pending_sha256, case, protocol_sha256)
        seal_sha256 = _json_payload_sha256(seal)
        if seal_path.exists():
            existing_seal = json.loads(seal_path.read_text(encoding="utf-8"))
            if existing_seal != seal:
                raise ValueError(f"Pending seal is incompatible: {seal_path}")
        else:
            write_json_atomic(seal_path, seal)
        _verify_file(
            seal_path,
            seal_sha256,
            f"Pending seal for {case['id']}",
        )
        sealed_results.append((*pending, seal_sha256))

    _verify_protocol_inputs(protocol, protocol_path, protocol_sha256)
    for sealed in sealed_results:
        (
            pending_path,
            seal_path,
            result_path,
            case,
            expected,
            result,
            pending_sha256,
            seal_sha256,
        ) = sealed
        _verify_file(
            pending_path,
            pending_sha256,
            f"Pending result for {case['id']}",
        )
        _verify_file(
            seal_path,
            seal_sha256,
            f"Pending seal for {case['id']}",
        )
        write_json_atomic(result_path, result)
        _verify_file(
            result_path,
            pending_sha256,
            f"Published result for {case['id']}",
        )
        published = _load_existing_result(result_path, expected)
        if published != result:
            raise RuntimeError(f"Published result snapshot changed for {case['id']}")
        final_seal = _result_seal(pending_sha256, case, protocol_sha256)
        result_seal_path = _result_seal_path(result_path)
        write_json_atomic(result_seal_path, final_seal)
        _verify_file(
            result_seal_path,
            _json_payload_sha256(final_seal),
            f"Published result seal for {case['id']}",
        )
        seal_path.unlink()
        pending_path.unlink()

    _verify_protocol_inputs(protocol, protocol_path, protocol_sha256)
    case_results = []
    case_artifacts = []
    for result_path, case, expected in case_contracts:
        result = _load_published_result(
            result_path,
            case,
            expected,
            protocol_sha256,
        )
        result_seal_path = _result_seal_path(result_path)
        case_results.append(result)
        case_artifacts.append({
            "case_id": case["id"],
            "path": str(result_path),
            "sha256": sha256_file(result_path),
            "seal": str(result_seal_path),
            "seal_sha256": sha256_file(result_seal_path),
        })
    gate = build_gate_summary(case_results, requirements)
    payload = {
        "batch_version": BATCH_VERSION,
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "protocol": str(protocol_path),
        "protocol_sha256": protocol_sha256,
        "provenance": provenance,
        "manifest": manifest,
        "run": {
            "sigmas": list(FDRR_GATE_SIGMAS),
            "block_index": FDRR_GATE_BLOCK_INDEX,
            "candidate_mode": "mixed",
            "num_token_probes": requirements[
                "required_token_probes_per_sigma"
            ],
            "exact_batch_size": FDRR_GATE_EXACT_BATCH_SIZE,
            "device": "cpu",
            "num_threads": int(args.num_threads),
        },
        "case_results": case_artifacts,
        "gate": gate,
    }
    summary_path = output_dir / "summary.json"
    _verify_protocol_inputs(protocol, protocol_path, protocol_sha256)
    _publish_summary(summary_path, payload, args.overwrite_cases)
    print(json.dumps(gate["checks"], indent=2, sort_keys=True))
    print(f"FDRR regret-evidence gate: {'PASS' if gate['passed'] else 'FAIL'}")
    print(f"Saved: {summary_path}")
    if not gate["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
