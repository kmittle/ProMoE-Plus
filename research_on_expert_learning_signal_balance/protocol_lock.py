"""Load the sealed v3 protocol plus its single-field v4 amendment."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path


V3_SHA256 = "d6cfb1c992e29536bebe05a3c2baecc5fb0b7da1d5d48dd7140905cc4e2869df"
V4_SHA256 = "86d17dbdfdaf38fccd4fc347513422d25f33c792deb7b9925f145e253ba631c2"
BOOTSTRAP_RESAMPLES = 200_000


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_effective_protocol(v3_path, v4_path):
    v3_path = Path(v3_path).resolve()
    v4_path = Path(v4_path).resolve()
    if sha256_file(v3_path) != V3_SHA256:
        raise RuntimeError("Sealed v3 preregistration hash mismatch")
    if sha256_file(v4_path) != V4_SHA256:
        raise RuntimeError("Sealed v4 preregistration hash mismatch")

    v3 = _load_json(v3_path)
    v4 = _load_json(v4_path)
    if v3.get("version") != 3 or v4.get("version") != 4:
        raise ValueError("Unsupported credit-redistribution protocol version")
    effective = v4.get("effective_protocol", {})
    if effective.get("base_sha256") != V3_SHA256:
        raise ValueError("v4 does not bind the sealed v3 protocol")
    amendments = effective.get("amendments")
    expected = [{
        "operation": "add",
        "json_pointer": "/statistics/bootstrap_resamples",
        "value": BOOTSTRAP_RESAMPLES,
    }]
    if amendments != expected:
        raise ValueError("v4 contains an unexpected scientific amendment")
    boundary = v4.get("knowledge_boundary", {})
    blinded = (
        "parent_base_discovery_visible",
        "parent_base_confirmatory_visible",
        "lossfree_cross_checkpoint_efficacy_visible",
        "three_arm_efficacy_visible",
    )
    if any(boundary.get(key) is not False for key in blinded):
        raise ValueError("v4 was not sealed under the required efficacy blind")

    merged = copy.deepcopy(v3)
    if "bootstrap_resamples" in merged.get("statistics", {}):
        raise ValueError("v3 unexpectedly already fixes bootstrap_resamples")
    merged["statistics"]["bootstrap_resamples"] = BOOTSTRAP_RESAMPLES
    merged["effective_version"] = 4
    merged["effective_protocol_hashes"] = {
        "v3": V3_SHA256,
        "v4": V4_SHA256,
    }
    return merged
