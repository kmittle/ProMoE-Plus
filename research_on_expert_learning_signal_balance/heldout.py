"""Materialize the sealed class-disjoint held-out tensors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from .protocol_lock import load_effective_protocol
from .serialization import atomic_write_json, sha256_file, tensor_sha256


HELDOUT_MANIFEST_VERSION = 1
PARENT_PROTOCOL_SHA256 = (
    "9c25bd0144228e921be1a5491dafa32299356f5af00e0a5cc15d857a1eeef096"
)


def canonical_json_sha256(payload):
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed_mod(modulus, *parts):
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16) % int(modulus)


def _rank_key(salt, value):
    return hashlib.sha256(f"{salt}|{value}".encode("utf-8")).hexdigest()


def _load_parent_labels(parent_protocol_path):
    parent_protocol_path = Path(parent_protocol_path).resolve()
    with parent_protocol_path.open("r", encoding="utf-8") as handle:
        parent = json.load(handle)
    observed_hash = canonical_json_sha256(parent)
    if observed_hash != PARENT_PROTOCOL_SHA256:
        raise RuntimeError("Parent gate protocol canonical hash mismatch")
    sidecar = parent_protocol_path.with_suffix(".sha256")
    if sidecar.read_text(encoding="utf-8") != PARENT_PROTOCOL_SHA256 + "\n":
        raise RuntimeError("Parent gate protocol sidecar mismatch")
    cases = parent.get("manifest", {}).get("cases")
    if not isinstance(cases, list) or len(cases) != 104:
        raise ValueError("Parent protocol must contain exactly 104 held-out cases")
    labels = [case.get("label") for case in cases]
    if (
        any(isinstance(label, bool) or not isinstance(label, int) for label in labels)
        or len(set(labels)) != 104
        or any(not 0 <= label < 1000 for label in labels)
    ):
        raise ValueError("Parent protocol labels are not 104 unique ImageNet IDs")
    return set(labels), observed_hash


def _class_directories(latent_root):
    latent_root = Path(latent_root).resolve()
    directories = sorted(path for path in latent_root.iterdir() if path.is_dir())
    if len(directories) != 1000:
        raise ValueError(f"Expected 1000 ImageNet class directories, found {len(directories)}")
    names = [path.name for path in directories]
    synsets = all(
        len(name) == 9 and name.startswith("n") and name[1:].isdigit()
        for name in names
    )
    numeric = all(name.isdigit() and int(name) == index for index, name in enumerate(names))
    if not synsets and not numeric:
        raise ValueError("Latent classes must use sorted synsets or 0000..0999 IDs")
    return directories


def select_cases(latent_root, excluded_labels, salt, case_count):
    directories = _class_directories(latent_root)
    remaining = [label for label in range(1000) if label not in excluded_labels]
    labels = sorted(remaining, key=lambda label: _rank_key(salt, label))[:case_count]
    if len(labels) != case_count:
        raise RuntimeError("Insufficient class-disjoint ImageNet labels")
    root = Path(latent_root).resolve()
    cases = []
    for case_index, label in enumerate(labels):
        candidates = sorted(
            path for path in directories[label].iterdir()
            if path.is_file() and path.name.endswith(".latent.npz")
        )
        if not candidates:
            raise FileNotFoundError(f"Class {label} contains no latent files")
        selected = min(
            candidates,
            key=lambda path: _rank_key(salt, path.relative_to(root).as_posix()),
        )
        cases.append({
            "index": case_index,
            "label": label,
            "relative_path": selected.relative_to(root).as_posix(),
        })
    return cases


def _atomic_save_npy(path, tensor):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    value = tensor.detach().cpu().contiguous()
    if value.dtype != torch.float32:
        raise TypeError("Held-out tensors must use float32")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.save(handle, value.numpy(), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, 0o444)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_npy_tensor(path):
    array = np.load(path, allow_pickle=False)
    return torch.from_numpy(np.array(array, copy=True)).contiguous()


def _write_or_verify_tensor(path, tensor):
    path = Path(path)
    expected_tensor_hash = tensor_sha256(tensor)
    if path.exists():
        observed = _load_npy_tensor(path)
        if observed.dtype != tensor.dtype or tuple(observed.shape) != tuple(tensor.shape):
            raise RuntimeError(f"Held-out tensor metadata changed: {path}")
        if tensor_sha256(observed) != expected_tensor_hash:
            raise RuntimeError(f"Held-out tensor bytes changed: {path}")
    else:
        _atomic_save_npy(path, tensor)
    os.chmod(path, 0o444)
    return {
        "path": path.name,
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "tensor_sha256": expected_tensor_hash,
        "file_sha256": sha256_file(path),
    }


def _materialize_case(case, latent_root, tensor_dir, salt, noise_draws):
    source = Path(latent_root).resolve() / case["relative_path"]
    with np.load(source, allow_pickle=False) as archive:
        if "latent" not in archive.files:
            raise KeyError(f"latent key is absent from {source}")
        parameters_array = np.array(archive["latent"], dtype=np.float32, copy=True)
    if parameters_array.shape != (8, 32, 32):
        raise ValueError(f"Unexpected VAE parameter shape: {parameters_array.shape}")
    parameters = torch.from_numpy(parameters_array).contiguous()
    posterior_seed = stable_seed_mod(
        2147483647, salt, case["relative_path"], "posterior"
    )
    posterior_generator = torch.Generator(device="cpu")
    posterior_generator.manual_seed(posterior_seed)
    distribution = DiagonalGaussianDistribution(parameters.unsqueeze(0))
    z = distribution.sample(generator=posterior_generator).squeeze(0)
    z = z.to(dtype=torch.float32).mul(torch.tensor(0.18215, dtype=torch.float32))
    z = z.unsqueeze(1).contiguous()
    if z.shape != (4, 1, 32, 32):
        raise RuntimeError(f"Materialized z shape changed: {tuple(z.shape)}")

    stem = f"case-{case['index']:03d}-label-{case['label']:03d}"
    z_record = _write_or_verify_tensor(tensor_dir / f"{stem}-z.npy", z)
    noises = []
    for draw in range(noise_draws):
        noise_seed = stable_seed_mod(
            2147483647, salt, case["relative_path"], draw, "noise"
        )
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(noise_seed)
        noise = torch.randn(
            z.shape,
            generator=noise_generator,
            dtype=torch.float32,
            device="cpu",
        ).contiguous()
        record = _write_or_verify_tensor(
            tensor_dir / f"{stem}-noise-{draw:02d}.npy", noise
        )
        record["draw"] = draw
        record["seed"] = noise_seed
        noises.append(record)
    return {
        **case,
        "source_latent_sha256": sha256_file(source),
        "latent_key": "latent",
        "posterior_parameter_dtype": "float32",
        "posterior_parameter_shape": list(parameters.shape),
        "posterior_parameter_sha256": tensor_sha256(parameters),
        "posterior_seed": posterior_seed,
        "z": z_record,
        "noise": noises,
    }


def materialize_heldout(
    latent_root,
    output_dir,
    parent_protocol_path,
    preregister_v3_path,
    preregister_v4_path,
):
    protocol = load_effective_protocol(preregister_v3_path, preregister_v4_path)
    heldout = protocol["heldout_inputs"]
    salt = heldout["selection_salt"]
    case_count = int(heldout["case_count"])
    noise_draws = int(heldout["noise_draws_per_image"])
    if case_count != 128 or noise_draws != 8 or heldout["latent_key"] != "latent":
        raise ValueError("Held-out constants differ from the sealed protocol")
    excluded, parent_hash = _load_parent_labels(parent_protocol_path)
    cases = select_cases(latent_root, excluded, salt, case_count)
    output_dir = Path(output_dir).resolve()
    tensor_dir = output_dir / "tensors"
    records = [
        _materialize_case(case, latent_root, tensor_dir, salt, noise_draws)
        for case in cases
    ]
    manifest = {
        "version": HELDOUT_MANIFEST_VERSION,
        "name": "promoe_credit_rate_budget_redistribution_heldout_v4",
        "selection_salt": salt,
        "selection_rule": heldout["selection_rule"],
        "latent_root": str(Path(latent_root).resolve()),
        "parent_protocol_path": str(Path(parent_protocol_path).resolve()),
        "parent_protocol_canonical_sha256": parent_hash,
        "effective_protocol_hashes": protocol["effective_protocol_hashes"],
        "case_count": case_count,
        "noise_draws_per_image": noise_draws,
        "sigmas": list(heldout["sigmas"]),
        "cases": records,
    }
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            if json.load(handle) != manifest:
                raise RuntimeError("Existing held-out manifest differs")
    else:
        atomic_write_json(manifest_path, manifest, mode=0o444)
    os.chmod(manifest_path, 0o444)
    manifest_hash = canonical_json_sha256(manifest)
    sidecar = output_dir / "manifest.sha256"
    expected_sidecar = manifest_hash + "\n"
    if sidecar.exists():
        if sidecar.read_text(encoding="utf-8") != expected_sidecar:
            raise RuntimeError("Held-out manifest sidecar differs")
    else:
        sidecar.write_text(expected_sidecar, encoding="utf-8")
    os.chmod(sidecar, 0o444)
    complete = output_dir / "COMPLETE"
    if not complete.exists():
        complete.write_text(manifest_hash + "\n", encoding="utf-8")
    elif complete.read_text(encoding="utf-8") != manifest_hash + "\n":
        raise RuntimeError("Held-out COMPLETE marker differs")
    os.chmod(complete, 0o444)
    return manifest_path, manifest_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latent-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--parent-protocol", required=True)
    parser.add_argument("--preregister-v3", required=True)
    parser.add_argument("--preregister-v4", required=True)
    args = parser.parse_args()
    path, digest = materialize_heldout(
        latent_root=args.latent_root,
        output_dir=args.output_dir,
        parent_protocol_path=args.parent_protocol,
        preregister_v3_path=args.preregister_v3,
        preregister_v4_path=args.preregister_v4,
    )
    print(f"Held-out manifest: {path}")
    print(f"Canonical SHA256: {digest}")


if __name__ == "__main__":
    main()
