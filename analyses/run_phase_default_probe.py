"""Run the locked phase-conditioned default-output diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analyses.denoising_regret.io import write_json_atomic
from analyses.phase_default import load_manifest, run_phase_default_probe
from analyses.phase_default.probe import PROBE_VERSION, sha256_file
from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    resolve_config_from_checkpoint,
)


DEFAULT_MANIFEST = (
    PROJECT_ROOT
    / "analyses"
    / "phase_default"
    / "manifests"
    / "phase_default_gate_v1.json"
)
SOURCE_FILES = (
    "analyses/phase_default/__init__.py",
    "analyses/phase_default/probe.py",
    "analyses/run_phase_default_probe.py",
)


def _json_sha256(payload):
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _validate_checkpoint(path, protocol, description):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if path.stat().st_size != protocol["checkpoint_size"]:
        raise ValueError(f"{description} size differs from the locked checkpoint")
    actual_hash = sha256_file(path)
    if actual_hash != protocol["checkpoint_sha256"]:
        raise ValueError(f"{description} SHA256 differs from the locked checkpoint")
    return path, actual_hash


def _build_protocol(args, manifest, checkpoint_path, weights_path):
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if runtime_cfg.model_name != manifest["protocol"]["model_name"]:
        raise ValueError("Checkpoint config model_name differs from the manifest")
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Locked source file is missing: {path}")
        source_hashes[relative] = sha256_file(path)
    return {
        "phase_default_probe_version": PROBE_VERSION,
        "manifest": {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "name": manifest["name"],
            "version": manifest["version"],
        },
        "locked_experiment": manifest["protocol"],
        "checkpoint": {
            "canonical_path": str(checkpoint_path),
            "weights_path": str(weights_path),
            "size": int(checkpoint_path.stat().st_size),
            "sha256": manifest["protocol"]["checkpoint_sha256"],
        },
        "config": {
            "path": str(config_path),
            "sha256": sha256_file(config_path),
            "model_name": runtime_cfg.model_name,
        },
        "source_sha256": source_hashes,
        "runtime": {
            "device": args.device,
            "num_threads": int(args.num_threads),
            "latent_root": str(Path(args.latent_root).resolve()),
        },
    }


def _write_or_validate_protocol(output_dir, protocol):
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol_path = output_dir / "protocol.json"
    hash_path = output_dir / "protocol.sha256"
    protocol_hash = _json_sha256(protocol)
    if protocol_path.exists() or hash_path.exists():
        if not protocol_path.is_file() or not hash_path.is_file():
            raise RuntimeError("Protocol lock is incomplete")
        existing = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError("Existing protocol differs from the requested run")
        if hash_path.read_text(encoding="utf-8") != protocol_hash + "\n":
            raise RuntimeError("Existing protocol hash is invalid")
    else:
        write_json_atomic(protocol_path, protocol)
        hash_path.write_text(protocol_hash + "\n", encoding="utf-8")
    return protocol_hash


def _progress(split, current, total, case_id):
    print(f"[{split}] {current}/{total}: {case_id}", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Compare zero, global, phase-conditioned, and shuffled-phase "
            "default expert outputs on a frozen Base checkpoint."
        )
    )
    parser.add_argument("--ckpt", required=True, help="Canonical Base-200K checkpoint")
    parser.add_argument(
        "--weights-ckpt",
        required=True,
        help="Local byte-identical checkpoint used to load weights",
    )
    parser.add_argument(
        "--manifest",
        default=str(DEFAULT_MANIFEST),
        help="Locked calibration/confirmation manifest",
    )
    parser.add_argument(
        "--latent-root",
        required=True,
        help="ImageNet VAE latent root containing 1000 synset directories",
    )
    parser.add_argument("--output-dir", required=True, help="Fresh result directory")
    parser.add_argument(
        "--device",
        default="cpu",
        help="Explicit torch device; no GPU is claimed by default",
    )
    parser.add_argument("--num-threads", type=int, default=8)
    return parser


def main():
    args = build_parser().parse_args()
    if args.num_threads < 1:
        raise ValueError("num_threads must be positive")
    output_dir = Path(args.output_dir).resolve()
    result_path = output_dir / "result.json"
    result_hash_path = output_dir / "result.sha256"
    if result_path.exists() or result_hash_path.exists():
        raise FileExistsError(
            f"Result already exists under {output_dir}; locked results are immutable"
        )

    manifest = load_manifest(args.manifest, args.latent_root)
    protocol_spec = manifest["protocol"]
    checkpoint_path, checkpoint_hash = _validate_checkpoint(
        args.ckpt, protocol_spec, "Canonical checkpoint"
    )
    weights_path, weights_hash = _validate_checkpoint(
        args.weights_ckpt, protocol_spec, "Local weights checkpoint"
    )
    if checkpoint_hash != weights_hash:
        raise RuntimeError("Canonical and local checkpoint hashes differ")
    protocol = _build_protocol(args, manifest, checkpoint_path, weights_path)
    protocol_hash = _write_or_validate_protocol(output_dir, protocol)
    print(f"Protocol SHA256: {protocol_hash}", flush=True)

    result = run_phase_default_probe(
        checkpoint_path=checkpoint_path,
        weights_checkpoint_path=weights_path,
        manifest=manifest,
        device=args.device,
        num_threads=args.num_threads,
        progress=_progress,
    )
    result["protocol_sha256"] = protocol_hash
    result["provenance"] = protocol
    write_json_atomic(result_path, result)
    result_hash = sha256_file(result_path)
    result_hash_path.write_text(result_hash + "\n", encoding="utf-8")
    print(json.dumps(result["gate"], indent=2, sort_keys=True), flush=True)
    print(f"Saved: {result_path}", flush=True)
    print(f"Result SHA256: {result_hash}", flush=True)


if __name__ == "__main__":
    main()
