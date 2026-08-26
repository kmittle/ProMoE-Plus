"""Natural-input counterfactual expert-utility diagnostics."""

from .probe import (
    DEFAULT_BLOCK_INDICES,
    PROBE_VERSION,
    run_timestep_utility_probe,
)


__all__ = [
    "DEFAULT_BLOCK_INDICES",
    "PROBE_VERSION",
    "run_timestep_utility_probe",
]
