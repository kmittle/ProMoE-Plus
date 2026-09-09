"""Streaming canonical digests for deterministic training-state replay."""

from __future__ import annotations

import hashlib
import math
import struct

import numpy as np
import torch

from .serialization import canonical_value_bytes


STATE_DIGEST_VERSION = 1
STATE_SECTIONS = (
    "model_state_dict",
    "ema_model_state_dict",
    "optimizer_state_dict",
    "credit_redistribution_state",
    "trainer_state",
)


def _write_length(digest, value):
    value = int(value)
    if value < 0 or value >= 2 ** 64:
        raise ValueError("Canonical length is outside uint64 range")
    digest.update(struct.pack("<Q", value))


def _write_bytes(digest, value):
    value = bytes(value)
    _write_length(digest, len(value))
    digest.update(value)


def _tensor_array(tensor):
    tensor = tensor.detach().cpu().contiguous()
    if tensor.layout != torch.strided:
        raise TypeError("Canonical state tensors must use strided layout")
    if tensor.dtype == torch.bfloat16:
        return tensor.view(torch.uint16).numpy().astype("<u2", copy=False)
    array = tensor.numpy()
    if array.dtype.byteorder not in {"|", "<"}:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return array


def _key_sort_bytes(value):
    return canonical_value_bytes(value)


def _update(digest, value):
    if value is None:
        digest.update(b"n")
        return
    if isinstance(value, bool):
        digest.update(b"b1" if value else b"b0")
        return
    if isinstance(value, int):
        digest.update(b"i")
        _write_bytes(digest, str(value).encode("ascii"))
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical state contains a nonfinite float")
        digest.update(b"f")
        digest.update(struct.pack("<d", value))
        return
    if isinstance(value, str):
        digest.update(b"s")
        _write_bytes(digest, value.encode("utf-8"))
        return
    if isinstance(value, bytes):
        digest.update(b"y")
        _write_bytes(digest, value)
        return
    if torch.is_tensor(value):
        digest.update(b"t")
        array = _tensor_array(value)
        _write_bytes(digest, str(value.dtype).removeprefix("torch.").encode("utf-8"))
        _write_length(digest, value.ndim)
        for dimension in value.shape:
            _write_bytes(digest, str(int(dimension)).encode("ascii"))
        _write_length(digest, int(array.nbytes))
        digest.update(memoryview(array).cast("B"))
        return
    if isinstance(value, np.ndarray):
        _update(digest, torch.from_numpy(value))
        return
    if isinstance(value, list):
        digest.update(b"l")
        _write_length(digest, len(value))
        for item in value:
            _update(digest, item)
        return
    if isinstance(value, tuple):
        digest.update(b"u")
        _write_length(digest, len(value))
        for item in value:
            _update(digest, item)
        return
    if isinstance(value, dict):
        digest.update(b"d")
        _write_length(digest, len(value))
        ordered = sorted(value.items(), key=lambda item: _key_sort_bytes(item[0]))
        for key, item in ordered:
            _update(digest, key)
            _update(digest, item)
        return
    raise TypeError(f"Unsupported canonical state type: {type(value).__name__}")


def canonical_state_sha256(value):
    digest = hashlib.sha256()
    digest.update(f"credit-state-v{STATE_DIGEST_VERSION}".encode("ascii"))
    _update(digest, value)
    return digest.hexdigest()


def checkpoint_state_digests(checkpoint):
    if not isinstance(checkpoint, dict):
        raise TypeError("Checkpoint must be a mapping")
    required = set(STATE_SECTIONS) - {"credit_redistribution_state"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint lacks replay state sections: {missing}")
    sections = {
        name: canonical_state_sha256(checkpoint.get(name))
        for name in STATE_SECTIONS
    }
    sections["step"] = canonical_state_sha256(checkpoint.get("step"))
    sections["combined"] = canonical_state_sha256(sections)
    return sections
