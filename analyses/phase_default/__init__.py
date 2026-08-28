"""Phase-conditioned default-output diagnostics for sparse MoE routing."""

from .probe import (
    DEFAULT_BLOCK_INDICES,
    PROBE_VERSION,
    build_gate_summary,
    load_manifest,
    run_phase_default_probe,
)


__all__ = [
    "DEFAULT_BLOCK_INDICES",
    "PROBE_VERSION",
    "build_gate_summary",
    "load_manifest",
    "run_phase_default_probe",
]
