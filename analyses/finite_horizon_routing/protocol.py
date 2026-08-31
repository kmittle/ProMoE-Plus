"""Locked mathematical and statistical contract for finite-horizon routing."""

from __future__ import annotations

import math

import numpy as np
import torch
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from analyses.denoising_regret.probe import _correlation, _rankdata


PROBE_VERSION = 1
SAMPLE_STEPS = 250
SAMPLE_SHIFT = 1.0
SCHEDULER_SHIFT = 1.0
NUM_TRAIN_TIMESTEPS = 1000
START_INDICES = (50, 125, 200)
HORIZONS = (1, 2, 4, 8)
BLOCK_INDICES = (1, 3, 5, 7, 9, 11)
CANDIDATE_COUNT = 16
CANDIDATE_CHUNK_SIZE = 8


def sampling_sigmas(
    sample_steps=SAMPLE_STEPS,
    shift=SAMPLE_SHIFT,
    scheduler_shift=SCHEDULER_SHIFT,
    num_train_timesteps=NUM_TRAIN_TIMESTEPS,
):
    """Return the exact float32 sigma grid consumed by ``sample.py``."""

    if isinstance(sample_steps, bool) or int(sample_steps) != sample_steps:
        raise ValueError("sample_steps must be an integer")
    sample_steps = int(sample_steps)
    shift = float(shift)
    scheduler_shift = float(scheduler_shift)
    if isinstance(num_train_timesteps, bool) or int(num_train_timesteps) != num_train_timesteps:
        raise ValueError("num_train_timesteps must be an integer")
    num_train_timesteps = int(num_train_timesteps)
    if (
        sample_steps < 1
        or num_train_timesteps < 1
        or not math.isfinite(shift)
        or shift <= 0
        or not math.isfinite(scheduler_shift)
        or scheduler_shift <= 0
    ):
        raise ValueError("sampling steps and both shifts must be positive")

    # sample.py first constructs sample_shift-adjusted numpy sigmas, then
    # FlowMatchEulerDiscreteScheduler casts them to float32, applies cfg.shift,
    # and appends the terminal zero used by scheduler.step().
    base = np.linspace(1.0, 0.0, sample_steps + 1)[:sample_steps]
    requested = shift * base / (1.0 + (shift - 1.0) * base)
    scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=num_train_timesteps,
        shift=scheduler_shift,
    )
    scheduler.set_timesteps(sigmas=requested, device="cpu")
    sigmas = scheduler.sigmas.cpu().numpy()
    if (
        sigmas.shape != (sample_steps + 1,)
        or sigmas[0] != 1.0
        or sigmas[-1] != 0.0
        or not np.all(np.diff(sigmas) < 0)
    ):
        raise RuntimeError("Sampling sigma construction violated its contract")
    return sigmas


def validate_schedule_positions(
    sigmas,
    start_indices=START_INDICES,
    horizons=HORIZONS,
):
    sigmas = np.asarray(sigmas, dtype=np.float64)
    starts = tuple(int(index) for index in start_indices)
    horizons = tuple(int(horizon) for horizon in horizons)
    if sigmas.ndim != 1 or sigmas.size < 2:
        raise ValueError("sigmas must be a one-dimensional schedule")
    if not starts or len(starts) != len(set(starts)):
        raise ValueError("start_indices must be nonempty and unique")
    if not horizons or tuple(sorted(horizons)) != horizons or horizons[0] < 1:
        raise ValueError("horizons must be unique positive values in order")
    if any(index < 0 or index + horizons[-1] >= sigmas.size for index in starts):
        raise ValueError("A start index cannot support the complete horizon grid")
    return starts, horizons


def analytic_flow_state(clean_latent, noise, sigma):
    """State on the known linear flow path between one image and one noise."""

    if clean_latent.shape != noise.shape:
        raise ValueError("clean_latent and noise must have identical shapes")
    sigma = float(sigma)
    if not math.isfinite(sigma) or not 0.0 <= sigma <= 1.0:
        raise ValueError("sigma must lie in [0, 1]")
    return (1.0 - sigma) * clean_latent + sigma * noise


def euler_flow_step(state, velocity, sigma, next_sigma):
    """One update with the same float32 arithmetic as diffusers' scheduler."""

    if state.shape != velocity.shape:
        raise ValueError("state and velocity must have identical shapes")
    sigma = float(sigma)
    next_sigma = float(next_sigma)
    if not 0.0 <= next_sigma < sigma <= 1.0:
        raise ValueError("Euler sigmas must decrease inside [0, 1]")
    sigma_tensor = torch.as_tensor(sigma, device=state.device, dtype=torch.float32)
    next_sigma_tensor = torch.as_tensor(
        next_sigma,
        device=state.device,
        dtype=torch.float32,
    )
    previous = state.to(torch.float32) + (
        next_sigma_tensor - sigma_tensor
    ) * velocity
    return previous.to(velocity.dtype)


def _count_vector(values, num_experts):
    return np.bincount(
        np.asarray(values, dtype=np.int64),
        minlength=int(num_experts),
    )


def validate_count_preserving_candidates(candidates, native_routes, num_experts):
    """Require every intervention to preserve the full expert-count vector."""

    native_routes = np.asarray(native_routes, dtype=np.int64)
    num_experts = int(num_experts)
    if native_routes.ndim != 1 or native_routes.size < 2:
        raise ValueError("native_routes must contain at least two tokens")
    if num_experts < 2 or native_routes.min() < 0 or native_routes.max() >= num_experts:
        raise ValueError("native_routes name an invalid expert")
    if not candidates:
        raise ValueError("At least one candidate is required")
    native_counts = _count_vector(native_routes, num_experts)
    signatures = set()
    normalized = []
    for candidate in candidates:
        tokens = np.asarray(candidate["tokens"], dtype=np.int64)
        sources = np.asarray(candidate["source_experts"], dtype=np.int64)
        destinations = np.asarray(candidate["destination_experts"], dtype=np.int64)
        if (
            tokens.ndim != 1
            or tokens.size < 2
            or sources.shape != tokens.shape
            or destinations.shape != tokens.shape
        ):
            raise ValueError("Candidate token and expert vectors must align")
        if np.unique(tokens).size != tokens.size:
            raise ValueError("A candidate cannot reuse a token")
        if tokens.min() < 0 or tokens.max() >= native_routes.size:
            raise ValueError("Candidate token lies outside the route sequence")
        if not np.array_equal(native_routes[tokens], sources):
            raise ValueError("Candidate sources disagree with native routing")
        if (
            destinations.min() < 0
            or destinations.max() >= num_experts
            or np.any(sources == destinations)
        ):
            raise ValueError("Every candidate slot must name a different valid expert")
        candidate_routes = native_routes.copy()
        candidate_routes[tokens] = destinations
        candidate_counts = _count_vector(candidate_routes, num_experts)
        if not np.array_equal(candidate_counts, native_counts):
            raise ValueError("Candidate changes the full expert-count vector")
        signature = tuple(sorted(zip(tokens.tolist(), destinations.tolist())))
        if signature in signatures:
            raise ValueError("Candidate assignments must be unique")
        signatures.add(signature)
        normalized.append({
            **candidate,
            "tokens": tokens.tolist(),
            "source_experts": sources.tolist(),
            "destination_experts": destinations.tolist(),
            "changed_tokens": int(tokens.size),
            "full_native_count_vector": native_counts.tolist(),
            "full_candidate_count_vector": candidate_counts.tolist(),
            "full_count_match": True,
        })
    return normalized


def _finite_vector(records, key):
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError(f"Candidate metric {key!r} must be finite")
    return values


def _spearman(left, right):
    value = _correlation(_rankdata(left), _rankdata(right))
    return None if value is None else float(value)


def _top_overlap(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 4:
        raise ValueError("Top-set overlap needs at least four aligned candidates")
    count = max(1, left.size // 4)
    left_top = set(np.argsort(left, kind="stable")[-count:].tolist())
    right_top = set(np.argsort(right, kind="stable")[-count:].tolist())
    return float(len(left_top & right_top) / count)


def _sign_disagreement(left, right, epsilon):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    decisive = (np.abs(left) > epsilon) & (np.abs(right) > epsilon)
    count = int(decisive.sum())
    disagree = int(((left[decisive] * right[decisive]) < 0).sum())
    return {
        "decisive_candidates": count,
        "disagreeing_candidates": disagree,
        "rate": float(disagree / count) if count else None,
    }


def summarize_cell_records(records, numerical_epsilon):
    """Compare immediate assignment utility with each finite rollout horizon."""

    if len(records) < 4:
        raise ValueError("A cell needs at least four routing candidates")
    numerical_epsilon = float(numerical_epsilon)
    if not math.isfinite(numerical_epsilon) or numerical_epsilon <= 0:
        raise ValueError("numerical_epsilon must be positive")
    candidate_ids = [record["id"] for record in records]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("Candidate record IDs must be unique")
    immediate = _finite_vector(records, "immediate_gain_relative")
    swap_preference = -_finite_vector(records, "mean_router_margin")
    per_horizon = {}
    for horizon in HORIZONS:
        key = f"h{horizon}_gain_relative"
        future = _finite_vector(records, key)
        best_immediate = int(np.argmax(immediate))
        best_future = int(np.argmax(future))
        future_range = float(future.max() - future.min())
        regret = float(future[best_future] - future[best_immediate])
        per_horizon[str(horizon)] = {
            "immediate_future_spearman": _spearman(immediate, future),
            "swap_preference_future_spearman": _spearman(
                swap_preference,
                future,
            ),
            "top_quartile_overlap": _top_overlap(immediate, future),
            "sign_disagreement": _sign_disagreement(
                immediate,
                future,
                numerical_epsilon,
            ),
            "immediate_best_candidate_id": candidate_ids[best_immediate],
            "future_best_candidate_id": candidate_ids[best_future],
            "best_future_gain_relative": float(future[best_future]),
            "best_candidate_matches": bool(best_immediate == best_future),
            "immediate_best_future_regret": regret,
            "immediate_best_future_regret_fraction_of_range": (
                float(regret / future_range)
                if future_range > numerical_epsilon
                else None
            ),
            "future_gain_range": future_range,
            "future_beneficial_rate": float(np.mean(future > numerical_epsilon)),
            "future_harmful_rate": float(np.mean(future < -numerical_epsilon)),
        }
    return {
        "num_candidates": len(records),
        "swap_preference_immediate_spearman": _spearman(
            swap_preference,
            immediate,
        ),
        "immediate_gain_range": float(immediate.max() - immediate.min()),
        "immediate_beneficial_rate": float(
            np.mean(immediate > numerical_epsilon)
        ),
        "immediate_harmful_rate": float(
            np.mean(immediate < -numerical_epsilon)
        ),
        "per_horizon": per_horizon,
    }
