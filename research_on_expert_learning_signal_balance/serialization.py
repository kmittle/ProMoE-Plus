"""Canonical byte encodings used by transcripts and replay digests."""

from __future__ import annotations

import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path

import numpy as np
import torch


SERIALIZATION_VERSION = 1


def _u64(value):
    value = int(value)
    if not 0 <= value < 2 ** 64:
        raise ValueError("uint64 value is out of range")
    return struct.pack("<Q", value)


def _frame(payload):
    payload = bytes(payload)
    return _u64(len(payload)) + payload


def field_frame(name, payload):
    name_bytes = str(name).encode("utf-8")
    return _frame(name_bytes) + _frame(payload)


def _tensor_dtype_name(tensor):
    name = str(tensor.dtype)
    if not name.startswith("torch."):
        raise TypeError(f"Unsupported tensor dtype name: {name}")
    return name.removeprefix("torch.")


def tensor_raw_bytes(tensor):
    if not torch.is_tensor(tensor):
        raise TypeError("Expected a tensor")
    if tensor.layout != torch.strided:
        raise TypeError("Only strided tensors have a canonical byte encoding")
    value = tensor.detach().cpu().contiguous()
    if value.dtype == torch.bfloat16:
        array = value.view(torch.uint16).numpy().astype("<u2", copy=False)
        return array.tobytes(order="C")
    array = value.numpy()
    if array.dtype.byteorder not in {"|", "<"}:
        array = array.astype(array.dtype.newbyteorder("<"), copy=False)
    return array.tobytes(order="C")


def tensor_payload(tensor):
    value = tensor.detach().cpu().contiguous()
    metadata = bytearray()
    metadata.extend(_frame(_tensor_dtype_name(value).encode("utf-8")))
    metadata.extend(_u64(value.ndim))
    for dimension in value.shape:
        metadata.extend(_frame(str(int(dimension)).encode("ascii")))
    metadata.extend(tensor_raw_bytes(value))
    return bytes(metadata)


def tensor_sha256(tensor):
    return hashlib.sha256(tensor_payload(tensor)).hexdigest()


def string_list_payload(values):
    payload = bytearray(_u64(len(values)))
    for value in values:
        payload.extend(_frame(str(value).encode("utf-8")))
    return bytes(payload)


def int64_tensor(values):
    return torch.as_tensor(values, dtype=torch.int64).detach().cpu().contiguous()


def canonical_value_bytes(value):
    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b" + (b"1" if value else b"0")
    if isinstance(value, int):
        encoded = str(value).encode("ascii")
        return b"i" + _frame(encoded)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("Canonical states cannot contain nonfinite floats")
        return b"f" + struct.pack("<d", value)
    if isinstance(value, str):
        return b"s" + _frame(value.encode("utf-8"))
    if isinstance(value, bytes):
        return b"y" + _frame(value)
    if torch.is_tensor(value):
        return b"t" + _frame(tensor_payload(value))
    if isinstance(value, np.ndarray):
        return b"a" + _frame(tensor_payload(torch.from_numpy(value)))
    if isinstance(value, list):
        return b"l" + _u64(len(value)) + b"".join(
            _frame(canonical_value_bytes(item)) for item in value
        )
    if isinstance(value, tuple):
        return b"u" + _u64(len(value)) + b"".join(
            _frame(canonical_value_bytes(item)) for item in value
        )
    if isinstance(value, dict):
        encoded_items = [
            (canonical_value_bytes(key), canonical_value_bytes(item))
            for key, item in value.items()
        ]
        encoded_items.sort(key=lambda pair: pair[0])
        return b"d" + _u64(len(encoded_items)) + b"".join(
            _frame(key) + _frame(item) for key, item in encoded_items
        )
    raise TypeError(f"Unsupported canonical state type: {type(value).__name__}")


def content_sha256(value):
    return hashlib.sha256(canonical_value_bytes(value)).hexdigest()


def atomic_write_json(path, payload, mode=0o644):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
