"""Deterministic rank-local support batches for the RCL mechanism gate."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from .protocol import (
    SUPPORT_BATCH_SIZE,
    SUPPORT_GROUP_COUNT,
    SUPPORT_SELECTION_SALT,
    SUPPORT_SIGMA_POLICY,
)


SUPPORT_UNCONDITIONAL_COUNT = 6


def _digest(*parts):
    payload = "|".join(str(part) for part in parts).encode("ascii")
    return hashlib.sha256(payload).digest()


def _support_sigma(case):
    policy = SUPPORT_SIGMA_POLICY
    seed = int.from_bytes(
        _digest(
            policy["seed_salt"],
            case["group_index"],
            case["label"],
            case["latent_relative"],
        )[:8],
        "big",
    ) % (2 ** 63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    logit = torch.normal(
        mean=float(policy["logit_mean"]),
        std=float(policy["logit_std"]),
        size=(),
        generator=generator,
        dtype=torch.float32,
    )
    u = torch.sigmoid(logit * float(policy["sigmoid_scale"]))
    shift = float(policy["shift"])
    sigma = shift * u / (1.0 + (shift - 1.0) * u)
    return int(seed), float(sigma.item())


def select_support_cases(
    latent_root,
    excluded_labels,
    expected_class_count=1000,
    group_count=SUPPORT_GROUP_COUNT,
    batch_size=SUPPORT_BATCH_SIZE,
    salt=SUPPORT_SELECTION_SALT,
):
    """Select one latent from each of 256 fixed, query-disjoint classes."""

    latent_root = Path(latent_root).resolve(strict=True)
    if latent_root.is_symlink() or not latent_root.is_dir():
        raise ValueError("latent_root must be a real directory")
    expected_class_count = int(expected_class_count)
    group_count = int(group_count)
    batch_size = int(batch_size)
    if expected_class_count < 1 or group_count < 1 or batch_size < 1:
        raise ValueError("Class, group, and batch counts must be positive")
    class_dirs = sorted(
        path
        for path in latent_root.iterdir()
        if path.is_dir() and not path.is_symlink()
    )
    if len(class_dirs) != expected_class_count:
        raise ValueError(
            f"Expected {expected_class_count} latent classes, found {len(class_dirs)}"
        )
    excluded = {int(label) for label in excluded_labels}
    if any(label < 0 or label >= expected_class_count for label in excluded):
        raise ValueError("An excluded label lies outside the class range")
    required = group_count * batch_size
    available = [
        (label, class_dir)
        for label, class_dir in enumerate(class_dirs)
        if label not in excluded
    ]
    if len(available) < required:
        raise ValueError("Not enough query-disjoint classes for support batches")
    ranked = sorted(
        available,
        key=lambda item: _digest(
            salt,
            "class",
            f"{item[0]:03d}",
            item[1].name,
        ),
    )[:required]

    cases = []
    for selection_rank, (label, class_dir) in enumerate(ranked):
        latent_paths = sorted(class_dir.glob("*.latent.npz"))
        if not latent_paths:
            raise FileNotFoundError(f"Support class has no latents: {class_dir}")
        if any(path.is_symlink() or not path.is_file() for path in latent_paths):
            raise ValueError(f"Support class contains a non-regular latent: {class_dir}")
        choice_digest = _digest(
            salt,
            "latent",
            f"{label:03d}",
            class_dir.name,
        )
        latent_index = int.from_bytes(choice_digest[:8], "big") % len(latent_paths)
        seed = int.from_bytes(choice_digest[8:16], "big") % 2_147_483_647
        latent_path = latent_paths[latent_index].resolve(strict=True)
        relative = latent_path.relative_to(latent_root)
        cases.append({
            "selection_rank": int(selection_rank),
            "group_index": int(selection_rank % group_count),
            "label": int(label),
            "synset": class_dir.name,
            "latent_relative": relative.as_posix(),
            "latent": str(latent_path),
            "latent_key": "latent",
            "seed": int(seed),
            "unconditional": False,
        })

    for group_index in range(group_count):
        group = [case for case in cases if case["group_index"] == group_index]
        if len(group) != batch_size:
            raise RuntimeError("Support groups are not exactly balanced")
        unconditional = sorted(
            group,
            key=lambda case: _digest(
                salt,
                "unconditional",
                group_index,
                case["latent_relative"],
            ),
        )[:SUPPORT_UNCONDITIONAL_COUNT]
        unconditional_paths = {case["latent_relative"] for case in unconditional}
        for case in group:
            case["unconditional"] = (
                case["latent_relative"] in unconditional_paths
            )
            sigma_seed, sigma = _support_sigma(case)
            case["sigma_seed"] = sigma_seed
            case["sigma"] = sigma
    return cases
