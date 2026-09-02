from __future__ import annotations

import gc
import importlib
import inspect
import time
from contextlib import contextmanager
from pathlib import Path
from types import MethodType

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution

from analyses.t_SNE.checkpoint_utils import (
    load_runtime_cfg,
    parse_checkpoint_step,
    resolve_config_from_checkpoint,
)


class RoutingProbeCapture:
    """Sever one MoE output from its prefix and retain its suffix gradient."""

    def __init__(self, moe_layer):
        self.enabled = False
        self.hidden_states = None
        self.labels = None
        self.moe_output = None
        self._handle = moe_layer.register_forward_hook(self._capture)

    def _capture(self, module, inputs, output):
        if not self.enabled:
            return None
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("Expected SparseMoeBlock to return (output, aux_loss)")
        self.hidden_states = inputs[0].detach()
        self.labels = inputs[1].detach()
        self.moe_output = output[0].detach().requires_grad_(True)
        return self.moe_output, output[1]

    def start(self):
        self.enabled = True
        self.hidden_states = None
        self.labels = None
        self.moe_output = None

    def stop(self):
        self.enabled = False

    def close(self):
        self._handle.remove()


def _build_model(runtime_cfg):
    registry_modules = (
        "train",
        "train_with_repa",
        "train_with_MoS_repa",
    )
    model_spec = None
    for module_name in registry_modules:
        model_dict = importlib.import_module(module_name).model_dict
        if runtime_cfg.model_name in model_dict:
            model_spec = model_dict[runtime_cfg.model_name]
            break
    if model_spec is None:
        raise KeyError(f"Unknown model_name: {runtime_cfg.model_name}")
    model_class, config_key = model_spec
    return model_class(**getattr(runtime_cfg, config_key))


@contextmanager
def _checkpoint_safe_globals():
    """Allow only the metadata classes stored by ProMoE checkpoints."""

    safe_globals = getattr(
        getattr(torch, "serialization", None),
        "safe_globals",
        None,
    )
    if safe_globals is None:
        yield
        return

    try:
        from easydict import EasyDict
        from torch.torch_version import TorchVersion
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "The restricted checkpoint loader cannot import its metadata types"
        ) from error

    with safe_globals([EasyDict, TorchVersion]):
        yield


def _load_checkpoint_payload(checkpoint_path):
    load_kwargs = {"map_location": "cpu"}
    try:
        supports_weights_only = (
            "weights_only" in inspect.signature(torch.load).parameters
        )
    except (TypeError, ValueError):
        supports_weights_only = True

    if not supports_weights_only:
        raise RuntimeError(
            "Secure checkpoint analysis requires a PyTorch version with "
            "weights_only loading"
        )
    load_kwargs["weights_only"] = True
    with _checkpoint_safe_globals():
        return torch.load(checkpoint_path, **load_kwargs)


def _load_checkpoint_model(runtime_cfg, checkpoint_path, device):
    model = _build_model(runtime_cfg)
    load_start = time.perf_counter()
    checkpoint = _load_checkpoint_payload(checkpoint_path)

    if "ema_model_state_dict" in checkpoint:
        state_dict = checkpoint["ema_model_state_dict"]
        state_name = "ema_model_state_dict"
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        state_name = "model_state_dict"
    else:
        raise KeyError(
            "Checkpoint has neither ema_model_state_dict nor model_state_dict"
        )
    checkpoint_step = checkpoint.get("step")
    if isinstance(checkpoint_step, bool) or not isinstance(checkpoint_step, int):
        raise ValueError("Checkpoint must contain an integer step")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"Checkpoint/model mismatch: missing={missing}, unexpected={unexpected}"
        )
    del state_dict
    del checkpoint
    gc.collect()

    model = model.to(device).eval().requires_grad_(False)
    return model, state_name, checkpoint_step, time.perf_counter() - load_start


def _load_latent(latent_path, latent_key, seed, device):
    with np.load(latent_path) as latent_file:
        if latent_key not in latent_file:
            raise KeyError(
                f"Latent key {latent_key!r} is absent; found {latent_file.files}"
            )
        parameters = np.array(latent_file[latent_key], copy=True)
    if parameters.ndim != 3 or parameters.shape[0] % 2 != 0:
        raise ValueError(
            f"Expected VAE distribution parameters shaped [2C,H,W], got {parameters.shape}"
        )

    torch.manual_seed(seed)
    posterior = DiagonalGaussianDistribution(
        torch.from_numpy(parameters).unsqueeze(0).to(device=device)
    )
    latent = posterior.sample().mul_(0.18215)
    return latent.unsqueeze(2)


def _extract_prediction(model_output, target_channels):
    prediction = model_output[0] if isinstance(model_output, tuple) else model_output
    if prediction.ndim == 5 and prediction.shape[2] == 1:
        prediction = prediction.squeeze(2)
    if prediction.ndim != 4:
        raise ValueError(f"Expected a 4D model prediction, got {prediction.shape}")
    if prediction.shape[1] != target_channels:
        if prediction.shape[1] != target_channels * 2:
            raise ValueError(
                f"Cannot split {prediction.shape[1]} output channels into {target_channels}"
            )
        prediction, _ = prediction.chunk(2, dim=1)
    return prediction


def _per_sample_mse(prediction, target):
    difference = prediction.double() - target.double()
    return difference.square().flatten(1).mean(dim=1)


def _compute_router(moe_layer, hidden_states, labels, timestep=None):
    if getattr(moe_layer, "phase_metric", None) is None:
        return moe_layer.compute_router(hidden_states, labels)
    if timestep is None:
        raise ValueError("Phase-aware router calls require timestep")
    return moe_layer.compute_router(hidden_states, labels, timestep)


def _all_router_weights(moe_layer, hidden_states, timestep=None):
    if hidden_states.ndim != 3:
        raise ValueError(
            "Router score reconstruction expects (batch, tokens, hidden), got "
            f"{tuple(hidden_states.shape)}"
        )
    normalized_input = F.normalize(hidden_states.float(), p=2, dim=-1)
    normalized_centers = F.normalize(
        moe_layer.cluster_centers.detach().float(), p=2, dim=-1
    )
    cosine = normalized_input @ normalized_centers.T
    phase_metric = getattr(moe_layer, "phase_metric", None)
    if phase_metric is not None:
        if timestep is None:
            raise ValueError(
                "Phase-aware router score reconstruction requires timestep"
            )
        batch_size, token_count, hidden_size = normalized_input.shape
        phase_timesteps = timestep.reshape(-1).to(hidden_states.device)
        if phase_timesteps.numel() != batch_size:
            raise ValueError(
                "timestep must contain one value per router batch row; got "
                f"{phase_timesteps.numel()} for batch size {batch_size}"
            )
        if (
            getattr(moe_layer, "phase_metric_shuffle_timestep", False)
            and batch_size > 1
        ):
            phase_timesteps = torch.roll(phase_timesteps, shifts=1, dims=0)
        token_timesteps = phase_timesteps.repeat_interleave(token_count)
        with torch.no_grad():
            phase_residual = phase_metric(
                normalized_input.reshape(-1, hidden_size),
                normalized_centers,
                token_timesteps,
            ).view(batch_size, token_count, -1)
        cosine = cosine + phase_residual
    if moe_layer.router_weight_mode == "softmax":
        return F.softmax(cosine, dim=-1)
    if moe_layer.router_weight_mode == "sigmoid":
        return torch.sigmoid(cosine)
    if moe_layer.router_weight_mode == "identity":
        return cosine
    raise ValueError(f"Unsupported router_weight_mode: {moe_layer.router_weight_mode}")


def _evaluate_experts(experts, hidden_states, expert_ids):
    outputs = torch.zeros_like(hidden_states, dtype=torch.float32)
    for expert_id, expert in enumerate(experts):
        selected = expert_ids == expert_id
        if selected.any():
            outputs[selected] = expert(hidden_states[selected]).float()
    return outputs


def _choose_challengers(router_weights, current_ids, mode, generator):
    if mode == "runner-up":
        challenger_ids = torch.topk(router_weights, k=2, dim=-1).indices[..., 1]
        return challenger_ids, torch.ones_like(current_ids, dtype=torch.bool)

    num_experts = router_weights.shape[-1]
    random_ids = torch.randint(
        num_experts - 1,
        current_ids.shape,
        generator=generator,
        device=current_ids.device,
    )
    random_ids = random_ids + (random_ids >= current_ids).long()
    if mode == "random":
        return random_ids, torch.zeros_like(current_ids, dtype=torch.bool)
    if mode == "mixed":
        runner_up = torch.topk(router_weights, k=2, dim=-1).indices[..., 1]
        positions = torch.arange(
            current_ids.numel(), device=current_ids.device
        ).reshape(current_ids.shape)
        use_runner_up = positions % 2 == 0
        return torch.where(use_runner_up, runner_up, random_ids), use_runner_up
    raise ValueError(f"Unknown candidate mode: {mode}")


def _configure_torch_threads(num_threads):
    requested_interop_threads = max(1, min(4, num_threads))
    intraop_request_applied = True
    interop_request_applied = True
    try:
        torch.set_num_threads(num_threads)
    except RuntimeError:
        intraop_request_applied = False
    try:
        torch.set_num_interop_threads(requested_interop_threads)
    except RuntimeError:
        # PyTorch permits this setting only before parallel work starts and only
        # once per process. Repeated library calls should keep the effective value.
        interop_request_applied = False
    return {
        "requested_intraop_threads": int(num_threads),
        "effective_intraop_threads": int(torch.get_num_threads()),
        "intraop_request_applied": intraop_request_applied,
        "requested_interop_threads": int(requested_interop_threads),
        "effective_interop_threads": int(torch.get_num_interop_threads()),
        "interop_request_applied": interop_request_applied,
    }


@contextmanager
def _forced_routes(moe_layer, token_indices, expert_indices):
    if token_indices.shape != expert_indices.shape:
        raise ValueError("Forced token and expert indices must align")
    if token_indices.ndim not in {1, 2}:
        raise ValueError("Forced route indices must be one- or two-dimensional")
    original_compute_router = moe_layer.compute_router

    def compute_router_with_override(
        this,
        hidden_states,
        labels,
        timestep=None,
    ):
        if timestep is None:
            router_result = original_compute_router(hidden_states, labels)
        else:
            router_result = original_compute_router(
                hidden_states,
                labels,
                timestep,
            )
        weights, indices, auxiliary_loss = router_result
        expected_batch = (
            token_indices.numel()
            if token_indices.ndim == 1
            else token_indices.shape[0]
        )
        if hidden_states.shape[0] != expected_batch:
            raise RuntimeError(
                "Forced route count must match the counterfactual batch size"
            )
        rows = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        if token_indices.ndim == 2:
            rows = rows.unsqueeze(1).expand_as(token_indices)
        indices[rows, token_indices, 0] = expert_indices
        return weights, indices, auxiliary_loss

    if "compute_router" in moe_layer.__dict__:
        raise RuntimeError("MoE layer already has an instance compute_router override")
    moe_layer.compute_router = MethodType(compute_router_with_override, moe_layer)
    try:
        yield
    finally:
        del moe_layer.compute_router


def _exact_counterfactual_changes(
    model,
    moe_layer,
    noised_latent,
    timestep,
    label,
    target,
    token_indices,
    challenger_ids,
    batch_size,
):
    changes = []
    target_channels = target.shape[1]
    for start in range(0, token_indices.numel(), batch_size):
        stop = min(start + batch_size, token_indices.numel())
        count = stop - start
        batch_latent = noised_latent.repeat(count, 1, 1, 1, 1)
        batch_timestep = timestep.repeat(count)
        batch_label = label.repeat(count)
        batch_target = target.repeat(count, 1, 1, 1)

        with torch.inference_mode():
            base_output = model(
                batch_latent, batch_timestep, context=batch_label
            )
            base_prediction = _extract_prediction(base_output, target_channels)
            base_losses = _per_sample_mse(base_prediction, batch_target)

            with _forced_routes(
                moe_layer,
                token_indices[start:stop],
                challenger_ids[start:stop],
            ):
                alternative_output = model(
                    batch_latent, batch_timestep, context=batch_label
                )
            alternative_prediction = _extract_prediction(
                alternative_output, target_channels
            )
            alternative_losses = _per_sample_mse(
                alternative_prediction, batch_target
            )
        changes.append((alternative_losses - base_losses).cpu())
    return torch.cat(changes)


def _rankdata(values):
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _correlation(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or left.std() == 0 or right.std() == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def summarize_records(records):
    predicted = np.array(
        [record["first_order_change"] for record in records], dtype=np.float64
    )
    exact = np.array(
        [record["exact_mse_change"] for record in records], dtype=np.float64
    )
    predicted_better = predicted < 0
    exact_better = exact < 0
    true_positive = predicted_better & exact_better

    summary = {
        "num_probes": int(len(records)),
        "pearson": _correlation(predicted, exact),
        "spearman": _correlation(_rankdata(predicted), _rankdata(exact)),
        "sign_agreement": float(np.mean(np.signbit(predicted) == np.signbit(exact))),
        "exact_better_rate": float(exact_better.mean()),
        "predicted_better_rate": float(predicted_better.mean()),
        "predicted_better_precision": (
            float(true_positive.sum() / predicted_better.sum())
            if predicted_better.any()
            else None
        ),
        "predicted_better_recall": (
            float(true_positive.sum() / exact_better.sum())
            if exact_better.any()
            else None
        ),
        "mean_exact_mse_change": float(exact.mean()),
        "median_abs_exact_mse_change": float(np.median(np.abs(exact))),
        "min_exact_mse_change": float(exact.min()),
        "max_exact_mse_change": float(exact.max()),
    }
    return summary


def _probe_sigma(
    model,
    moe_layer,
    capture,
    clean_latent,
    noise,
    label,
    sigma,
    num_train_timesteps,
    num_token_probes,
    candidate_mode,
    exact_batch_size,
    generator,
):
    sigma_tensor = torch.tensor(
        float(sigma), device=clean_latent.device, dtype=clean_latent.dtype
    )
    timestep = torch.full(
        (1,),
        float(sigma) * num_train_timesteps,
        device=clean_latent.device,
        dtype=clean_latent.dtype,
    )
    noised_latent = (1.0 - sigma_tensor) * clean_latent + sigma_tensor * noise
    target = (noise - clean_latent).squeeze(2)

    capture.start()
    model_output = model(noised_latent, timestep, context=label)
    prediction = _extract_prediction(model_output, target.shape[1])
    base_loss = _per_sample_mse(prediction, target).mean()
    if capture.moe_output is None:
        raise RuntimeError("The MoE probe hook did not capture an output")
    moe_gradient, = torch.autograd.grad(base_loss, capture.moe_output)
    capture.stop()

    hidden_states = capture.hidden_states
    router_weights = _all_router_weights(moe_layer, hidden_states, timestep)
    current_ids = router_weights.argmax(dim=-1)

    num_tokens = hidden_states.shape[1]
    probe_count = min(num_token_probes, num_tokens)
    token_indices = torch.randperm(
        num_tokens, generator=generator, device=hidden_states.device
    )[:probe_count]
    probe_hidden = hidden_states[0, token_indices]
    probe_router_weights = router_weights[0, token_indices]
    probe_current = current_ids[0, token_indices]
    probe_challenger, probe_uses_runner_up = _choose_challengers(
        probe_router_weights, probe_current, candidate_mode, generator
    )
    if candidate_mode == "mixed":
        expected_runner_up = (probe_count + 1) // 2
        if int(probe_uses_runner_up.sum().item()) != expected_runner_up:
            raise RuntimeError("Mixed challengers must alternate by sampled probe slot")

    with torch.no_grad():
        current_outputs = _evaluate_experts(
            moe_layer.experts[:moe_layer.num_routed_experts],
            probe_hidden,
            probe_current,
        )
        challenger_outputs = _evaluate_experts(
            moe_layer.experts[:moe_layer.num_routed_experts],
            probe_hidden,
            probe_challenger,
        )
        probe_slots = torch.arange(probe_count, device=hidden_states.device)
        current_weights = probe_router_weights[
            probe_slots, probe_current
        ].unsqueeze(-1)
        challenger_weights = probe_router_weights[
            probe_slots, probe_challenger
        ].unsqueeze(-1)
        output_delta = current_weights * (
            challenger_outputs - current_outputs
        )
        gradient = moe_gradient[0, token_indices].float()
        first_order_change = (gradient * output_delta).sum(dim=-1)
        normalized_change = first_order_change / (
            gradient.norm(dim=-1) * output_delta.norm(dim=-1)
        ).clamp_min(1e-12)
        router_margin = (current_weights - challenger_weights).squeeze(-1)

    exact_change = _exact_counterfactual_changes(
        model,
        moe_layer,
        noised_latent,
        timestep,
        label,
        target,
        token_indices,
        probe_challenger,
        exact_batch_size,
    )
    noop_count = min(exact_batch_size, probe_count)
    noop_change = _exact_counterfactual_changes(
        model,
        moe_layer,
        noised_latent,
        timestep,
        label,
        target,
        token_indices[:noop_count],
        probe_current[:noop_count],
        exact_batch_size,
    )

    records = []
    for index in range(probe_count):
        records.append({
            "sigma": float(sigma),
            "timestep": float(timestep.item()),
            "token_index": int(token_indices[index].item()),
            "current_expert": int(probe_current[index].item()),
            "challenger_expert": int(probe_challenger[index].item()),
            "challenger_source": (
                "runner-up" if probe_uses_runner_up[index].item() else "random"
            ),
            "selected_router_weight": float(current_weights[index].item()),
            "challenger_router_weight": float(
                challenger_weights[index].item()
            ),
            "router_margin": float(router_margin[index].item()),
            "first_order_change": float(first_order_change[index].item()),
            "normalized_first_order_change": float(
                normalized_change[index].item()
            ),
            "exact_mse_change": float(exact_change[index].item()),
        })
    diagnostics = {
        "noop_num_probes": int(noop_count),
        "noop_max_abs_mse_change": float(noop_change.abs().max().item()),
    }
    return records, float(base_loss.item()), diagnostics


def run_probe(
    checkpoint_path,
    latent_path,
    label,
    sigmas,
    block_index=3,
    num_token_probes=32,
    candidate_mode="runner-up",
    exact_batch_size=4,
    latent_key="latent",
    seed=0,
    device="cpu",
    num_threads=8,
    weights_checkpoint_path=None,
):
    checkpoint_path = Path(checkpoint_path).resolve()
    weights_checkpoint_path = Path(
        weights_checkpoint_path or checkpoint_path
    ).resolve()
    latent_path = Path(latent_path).resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not weights_checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Weights checkpoint does not exist: {weights_checkpoint_path}"
        )
    if not latent_path.is_file():
        raise FileNotFoundError(f"Latent does not exist: {latent_path}")
    if not sigmas or any(not 0 < sigma < 1 for sigma in sigmas):
        raise ValueError("Every probe sigma must be strictly between 0 and 1")
    if num_token_probes < 2:
        raise ValueError("num_token_probes must be at least 2")
    if exact_batch_size < 1:
        raise ValueError("exact_batch_size must be positive")
    if num_threads < 1:
        raise ValueError("num_threads must be positive")

    thread_config = _configure_torch_threads(num_threads)
    device = torch.device(device)
    config_path = resolve_config_from_checkpoint(checkpoint_path)
    checkpoint_step = parse_checkpoint_step(checkpoint_path)
    runtime_cfg = load_runtime_cfg(config_path)
    if not 0 <= label < runtime_cfg.num_classes:
        raise ValueError(
            f"label must be in [0, {runtime_cfg.num_classes - 1}], got {label}"
        )

    model, state_name, weights_checkpoint_step, load_seconds = _load_checkpoint_model(
        runtime_cfg, weights_checkpoint_path, device
    )
    if weights_checkpoint_step != checkpoint_step:
        raise ValueError(
            f"Loaded checkpoint step {weights_checkpoint_step} does not match "
            f"the canonical checkpoint step {checkpoint_step}"
        )
    if not 0 <= block_index < len(model.blocks):
        raise ValueError(f"block_index {block_index} is outside the model")
    block = model.blocks[block_index]
    if not block.use_moe:
        raise ValueError(f"block {block_index} is not an MoE block")
    moe_layer = block.mlp
    if moe_layer.top_k != 1:
        raise ValueError("The current probe requires top_k == 1")

    torch.manual_seed(seed)
    np.random.seed(seed % (2 ** 32))
    clean_latent = _load_latent(latent_path, latent_key, seed, device)
    torch.manual_seed(seed + 1)
    noise = torch.randn_like(clean_latent)
    label_tensor = torch.tensor([label], device=device, dtype=torch.long)
    generator = torch.Generator(device=device)
    generator.manual_seed(seed + 2)

    capture = RoutingProbeCapture(moe_layer)
    records = []
    base_losses = {}
    numerical_controls = {}
    probe_start = time.perf_counter()
    try:
        for sigma in sigmas:
            sigma_records, base_loss, diagnostics = _probe_sigma(
                model=model,
                moe_layer=moe_layer,
                capture=capture,
                clean_latent=clean_latent,
                noise=noise,
                label=label_tensor,
                sigma=sigma,
                num_train_timesteps=runtime_cfg.num_train_timesteps,
                num_token_probes=num_token_probes,
                candidate_mode=candidate_mode,
                exact_batch_size=exact_batch_size,
                generator=generator,
            )
            records.extend(sigma_records)
            base_losses[str(sigma)] = base_loss
            numerical_controls[str(sigma)] = diagnostics
    finally:
        capture.close()
    probe_seconds = time.perf_counter() - probe_start

    per_sigma = {}
    for sigma in sigmas:
        sigma_records = [record for record in records if record["sigma"] == sigma]
        per_sigma[str(sigma)] = summarize_records(sigma_records)

    return {
        "probe_version": 4,
        "counterfactual_route_weight": "selected",
        "checkpoint": str(checkpoint_path),
        "weights_checkpoint": str(weights_checkpoint_path),
        "checkpoint_step": checkpoint_step,
        "weights_checkpoint_step": weights_checkpoint_step,
        "checkpoint_state": state_name,
        "config": str(config_path),
        "model_name": runtime_cfg.model_name,
        "latent": str(latent_path),
        "latent_key": latent_key,
        "label": int(label),
        "block_index": int(block_index),
        "candidate_mode": candidate_mode,
        "sigmas": [float(sigma) for sigma in sigmas],
        "num_token_probes_requested": int(num_token_probes),
        "exact_batch_size": int(exact_batch_size),
        "seed": int(seed),
        "device": str(device),
        "num_threads": int(num_threads),
        "thread_config": thread_config,
        "model_load_seconds": load_seconds,
        "probe_seconds": probe_seconds,
        "base_mse": base_losses,
        "numerical_controls": numerical_controls,
        "summary": summarize_records(records),
        "per_sigma": per_sigma,
        "records": records,
    }
