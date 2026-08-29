"""Finite-horizon causal routing diagnostics for ProMoE."""

from .protocol import (
    BLOCK_INDICES,
    CANDIDATE_CHUNK_SIZE,
    CANDIDATE_COUNT,
    HORIZONS,
    NUM_TRAIN_TIMESTEPS,
    PROBE_VERSION,
    SAMPLE_SHIFT,
    SAMPLE_STEPS,
    SCHEDULER_SHIFT,
    START_INDICES,
    analytic_flow_state,
    euler_flow_step,
    sampling_sigmas,
    summarize_cell_records,
    validate_count_preserving_candidates,
)
from .probe import run_finite_horizon_routing_probe
from .batch import aggregate_case_results, requirements_for_split


__all__ = [
    "BLOCK_INDICES",
    "CANDIDATE_CHUNK_SIZE",
    "CANDIDATE_COUNT",
    "HORIZONS",
    "NUM_TRAIN_TIMESTEPS",
    "PROBE_VERSION",
    "SAMPLE_SHIFT",
    "SAMPLE_STEPS",
    "SCHEDULER_SHIFT",
    "START_INDICES",
    "analytic_flow_state",
    "euler_flow_step",
    "sampling_sigmas",
    "summarize_cell_records",
    "validate_count_preserving_candidates",
    "run_finite_horizon_routing_probe",
    "aggregate_case_results",
    "requirements_for_split",
]
