import contextlib
import io
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from easydict import EasyDict
from torch.nn.parallel import DistributedDataParallel

from models import models_ProMoE_TC as conference
from models import models_ProMoE_TC_clean_route as clean_route
from models import models_ProMoE_TC_clean_route_corr_router as clean_corr
from models import models_ProMoE_TC_corr_router as corr_router


MODEL_KWARGS = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=48, depth=4, num_heads=4)
BATCH = 6
SEQ_LEN = 16  # (8 / 2) ** 2 patches
LABELS = torch.tensor([3, 1000, 7, 7, 1000, 42])
DROPOUT_SEED = 99
GENERATOR_SEED = 5
INTERVALS = [1, 4, 5, 10, 20, 50]


def moe_config(**overrides):
    config = EasyDict(
        num_routed_experts=4,
        moe_intermediate_size=32,
        shared_expert_intermediate_size=32,
        load_balance_loss_coef=0,
        norm_topk_prob=False,
        seq_aux=False,
        use_shared_expert=True,
        interleave=True,
        init_MoeMLP=False,
        top_k=1,
        router_weight_mode="identity",
        routing_contrastive_lam=1,
        use_top_k_for_routing_contrastive=True,
        routing_contrastive_temperature=0.07,
    )
    config.update(overrides)
    return config


def corr_config(**overrides):
    config = dict(hidden_dim=16, skip_intervals=list(INTERVALS), no_reference_prob=0.5)
    config.update(overrides)
    return config


def route_config(**overrides):
    config = dict(contrastive_source="student", teacher_sigma_ratio=0.0, align_lam=0.0, align_temperature=0.07)
    config.update(overrides)
    return config


def build_conference():
    torch.manual_seed(0)
    # The conference constructor prints its MoE layout.
    with contextlib.redirect_stdout(io.StringIO()):
        return conference.DiT(MoE_config=moe_config(), **MODEL_KWARGS)


def build_clean_route(**route_overrides):
    torch.manual_seed(0)
    return clean_route.DiT(MoE_config=moe_config(), clean_route_config=route_config(**route_overrides), **MODEL_KWARGS)


def build_corr(**corr_overrides):
    torch.manual_seed(0)
    return corr_router.DiT(MoE_config=moe_config(), corr_router_config=corr_config(**corr_overrides), **MODEL_KWARGS)


def build_clean_corr(corr_overrides=None, **route_overrides):
    torch.manual_seed(0)
    return clean_corr.DiT(
        MoE_config=moe_config(),
        corr_router_config=corr_config(**(corr_overrides or {})),
        clean_route_config=route_config(**route_overrides),
        **MODEL_KWARGS,
    )


def is_corrector(name):
    return ".corrector." in name


def perturb(model):
    """Move the conference weights off their zero-initialised gates and output layer.

    The correctors are skipped without drawing, so every model receives the
    same perturbation of the weights it shares with the conference model.
    """
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if not is_corrector(name):
                parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    return model


def perturb_correctors(model, scale=0.2):
    """Give the correctors, output layer included, nonzero weights."""
    generator = torch.Generator().manual_seed(11)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if is_corrector(name):
                parameter.add_(scale * torch.randn(parameter.shape, generator=generator))
    return model


def bias_toward(model, expert, value=5.0):
    """Correctors that add ``value`` to one expert's score for every token."""
    with torch.no_grad():
        for block in model.moe_blocks():
            block.corrector.out.weight.zero_()
            block.corrector.out.bias.zero_()
            block.corrector.out.bias[expert] = value
    return model


def batch(seed=1234, sigma_value=None):
    """Latents laid out as train.py builds them: (B, C, 1, H, W)."""
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(BATCH, 4, 1, 8, 8, generator=generator)
    noise = torch.randn(BATCH, 4, 1, 8, 8, generator=generator)
    sigma = torch.rand(BATCH, generator=generator)
    if sigma_value is not None:
        sigma = torch.full((BATCH,), float(sigma_value))
    sigma = sigma.view(BATCH, 1, 1, 1)
    x = (1.0 - sigma.squeeze()).view(BATCH, 1, 1, 1, 1) * z + sigma.squeeze().view(BATCH, 1, 1, 1, 1) * noise
    t = sigma.reshape(-1) * 1000
    return x, t, z, noise, sigma


def dropped_labels(labels=LABELS):
    """The labels the training forward sees after class dropout drawn at DROPOUT_SEED."""
    torch.manual_seed(DROPOUT_SEED)
    drop = torch.rand(BATCH) < 0.1
    return torch.where(drop, torch.full_like(labels, 1000), labels)


def model_inputs(kind, labels, z, noise, sigma, generator_seed=GENERATOR_SEED):
    inputs = {"context": labels}
    if kind in ("clean_route", "clean_corr"):
        inputs.update(clean_route_z=z, clean_route_noise=noise, clean_route_sigma=sigma)
    if kind in ("corr", "clean_corr"):
        inputs.update(
            corr_router_sigma=sigma,
            corr_router_generator=torch.Generator().manual_seed(generator_seed),
        )
    if kind == "corr":
        inputs.update(corr_router_z=z, corr_router_noise=noise)
    return inputs


def train_step(model, kind, autocast=False, labels=LABELS, sigma_value=None, z=None, noise=None):
    """One step as train.py runs it: the forward under autocast, the backward outside.

    ``z`` / ``noise`` replace the auxiliary inputs without changing x.
    """
    model.train()
    model.zero_grad(set_to_none=True)
    x, t, batch_z, batch_noise, sigma = batch(sigma_value=sigma_value)
    z = batch_z if z is None else z
    noise = batch_noise if noise is None else noise
    torch.manual_seed(DROPOUT_SEED)  # class-label dropout draws from the global RNG
    region = torch.autocast(device_type="cpu", dtype=torch.bfloat16) if autocast else contextlib.nullcontext()
    with region:
        output = model(x, t, **model_inputs(kind, labels, z, noise, sigma))
    output.float().square().mean().backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), grads


def expected_draws(no_reference_prob, sigma, labels, intervals=INTERVALS, generator_seed=GENERATOR_SEED):
    """The reference draws of one training step: who gets a reference, and its sigma_ref."""
    generator = torch.Generator().manual_seed(generator_seed)
    coin = torch.rand(BATCH, generator=generator)
    gap_index = torch.randint(len(intervals), (BATCH,), generator=generator)
    use = (labels != 1000) & (coin >= no_reference_prob)
    gaps = torch.tensor(intervals, dtype=torch.float32)[gap_index] / 1000
    return use, (sigma.reshape(-1) + gaps).clamp(max=1.0)


@contextlib.contextmanager
def captured_inputs(module):
    """Record the first positional input of every call of ``module``."""
    records = []
    handle = module.register_forward_pre_hook(lambda _module, args: records.append(args[0].detach().clone()))
    try:
        yield records
    finally:
        handle.remove()


@contextlib.contextmanager
def captured_outputs(module):
    records = []
    handle = module.register_forward_hook(lambda _module, _args, output: records.append(output.detach().clone()))
    try:
        yield records
    finally:
        handle.remove()


@contextlib.contextmanager
def captured_routing(block):
    """Record (hidden states, labels, router weights, expert indices) of every routing call."""
    records = []
    original = block.compute_router

    def wrapper(hidden_states, labels, timestep=None):
        output = original(hidden_states, labels, timestep)
        records.append((hidden_states.detach().clone(), labels.clone(), output[0].detach().clone(), output[1].clone()))
        return output

    block.compute_router = wrapper
    try:
        yield records
    finally:
        del block.compute_router


def routed_cosines(block, hidden_states, labels):
    """The float32 cosines of the prototype-routed tokens, in flat order."""
    flat = hidden_states.reshape(-1, block.hidden_size).float()
    routed = labels.view(-1, 1).expand(-1, hidden_states.shape[1]).reshape(-1) != 1000
    token_norm = F.normalize(flat[routed], p=2, dim=1)
    center_norm = F.normalize(block.cluster_centers.detach().float(), p=2, dim=1)
    return token_norm @ center_norm.T


class EquivalenceTests(unittest.TestCase):
    def test_conference_weights_and_global_rng_are_unchanged(self):
        reference = build_conference()
        reference_rng = torch.get_rng_state()
        reference_state = reference.state_dict()
        for name, builder in (("corr", build_corr), ("clean_corr", build_clean_corr)):
            with self.subTest(name):
                state = builder().state_dict()
                self.assertTrue(torch.equal(torch.get_rng_state(), reference_rng))
                self.assertEqual(set(reference_state) - set(state), set())
                extra = set(state) - set(reference_state)
                self.assertTrue(extra and all(is_corrector(key) for key in extra))
                for key, value in reference_state.items():
                    self.assertTrue(torch.equal(state[key], value), key)

    def test_a_fresh_corrector_outputs_exactly_zero(self):
        model = perturb(build_corr())
        blocks = model.moe_blocks()
        with contextlib.ExitStack() as stack:
            outputs = [stack.enter_context(captured_outputs(block.corrector)) for block in blocks]
            train_step(model, "corr")
        for records in outputs:
            self.assertTrue(records)
            for delta in records:
                self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))

    def test_training_step_matches_the_conference_model(self):
        for autocast in (False, True):
            with self.subTest(autocast=autocast):
                reference_output, reference_grads = train_step(perturb(build_conference()), "conference", autocast)
                model = perturb(build_corr())
                output, grads = train_step(model, "corr", autocast)
                trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                self.assertEqual(set(grads), trainable)
                self.assertTrue(torch.equal(output, reference_output))
                self.assertEqual(set(reference_grads), {name for name in grads if not is_corrector(name)})
                for name, grad in reference_grads.items():
                    self.assertTrue(torch.equal(grads[name], grad), name)

    def test_clean_route_variant_matches_clean_route(self):
        for autocast in (False, True):
            for align_lam in (0.0, 1.0):
                with self.subTest(autocast=autocast, align_lam=align_lam):
                    reference_output, reference_grads = train_step(
                        perturb(build_clean_route(align_lam=align_lam)), "clean_route", autocast
                    )
                    model = perturb(build_clean_corr(align_lam=align_lam))
                    output, grads = train_step(model, "clean_corr", autocast)
                    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                    self.assertEqual(set(grads), trainable)
                    self.assertTrue(torch.equal(output, reference_output))
                    for name, grad in reference_grads.items():
                        self.assertTrue(torch.equal(grads[name], grad), name)

    def test_eval_forward_matches_the_conference_model(self):
        reference = perturb(build_conference()).eval()
        model = perturb(build_corr()).eval()
        x, t, _, _, _ = batch()
        reference_x0 = torch.randn(x.shape, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            expected = reference(x, t, context=LABELS)
            self.assertTrue(torch.equal(model(x, t, context=LABELS), expected))
            self.assertTrue(torch.equal(
                model(x, t, context=LABELS, corr_reference_x0=reference_x0, corr_reference_sigma=t / 1000 + 0.01),
                expected,
            ))

    def test_training_draws_no_global_random_numbers(self):
        pairs = (
            ((build_conference, "conference"), (build_corr, "corr")),
            ((build_clean_route, "clean_route"), (build_clean_corr, "clean_corr")),
        )
        for (base_builder, base_kind), (builder, kind) in pairs:
            with self.subTest(kind):
                train_step(perturb(base_builder()), base_kind)
                reference_state = torch.get_rng_state()
                train_step(perturb_correctors(perturb(builder())), kind)
                self.assertTrue(torch.equal(torch.get_rng_state(), reference_state))


class ReferenceTests(unittest.TestCase):
    def test_the_reference_steps_back_along_the_models_own_velocity(self):
        model = perturb_correctors(perturb(build_corr(no_reference_prob=0.0)))
        x, t, _, _, sigma = batch()
        with captured_inputs(model.x_embedder) as patches, captured_inputs(model.t_embedder) as times:
            train_step(model, "corr")
        labels = dropped_labels()
        use, previous_sigma_all = expected_draws(0.0, sigma, labels)
        self.assertTrue(torch.equal(use, labels != 1000))
        index = torch.where(use)[0]
        current_sigma = sigma.reshape(-1)[index]
        previous_sigma = previous_sigma_all[index]
        shape = (-1, 1, 1, 1, 1)
        with torch.no_grad():
            y = model.y_embedder.embedding_table(labels)
            current = x[index]
            velocity = model._velocity(
                model._run(current, t[index], y[index], labels[index], model._no_reference(current)), current
            )
            previous = current + (previous_sigma - current_sigma).view(shape) * velocity
            previous_velocity = model._velocity(
                model._run(previous, previous_sigma * 1000, y[index], labels[index], model._no_reference(previous)),
                previous,
            )
            x0_hat = previous - previous_sigma.view(shape) * previous_velocity
        expected_reference = torch.zeros_like(x)
        expected_reference[index] = x0_hat
        expected_reference_sigma = torch.ones(BATCH)
        expected_reference_sigma[index] = previous_sigma

        # x_embedder: pass 1 (x_t, its empty reference), pass 2 (x_prev, its empty
        # reference), the trained pass (x_t, the reference).
        self.assertEqual(len(patches), 6)
        torch.testing.assert_close(patches[0], current.squeeze(2))
        self.assertTrue(torch.equal(patches[1], torch.zeros_like(patches[1])))
        torch.testing.assert_close(patches[2], previous.squeeze(2))
        torch.testing.assert_close(patches[4], x.squeeze(2))
        torch.testing.assert_close(patches[5], expected_reference.squeeze(2))
        # t_embedder: the same calls; an empty reference sits at sigma_ref = 1.
        self.assertEqual(len(times), 6)
        torch.testing.assert_close(times[0], t[index])
        torch.testing.assert_close(times[1], torch.full_like(times[1], 1000.0))
        torch.testing.assert_close(times[2], previous_sigma * 1000)
        torch.testing.assert_close(times[4], t)
        torch.testing.assert_close(times[5], expected_reference_sigma * 1000)

    def test_the_reference_never_sees_the_clean_latent(self):
        model = perturb_correctors(perturb(build_clean_corr(align_lam=1.0)))
        references = []
        for seed in (1, 2):
            generator = torch.Generator().manual_seed(seed)
            z = torch.randn(BATCH, 4, 1, 8, 8, generator=generator)
            noise = torch.randn(BATCH, 4, 1, 8, 8, generator=generator)
            with captured_inputs(model.x_embedder) as patches:
                train_step(model, "clean_corr", z=z, noise=noise)
            references.append(patches[-1])
        self.assertTrue(torch.equal(references[0], references[1]))
        self.assertGreater(float(references[0].abs().sum()), 0.0)

    def test_only_routed_images_draw_a_reference(self):
        labels = dropped_labels()
        _, _, _, _, sigma = batch()
        for prob in (0.0, 0.5, 0.9):
            with self.subTest(no_reference_prob=prob):
                model = perturb(build_corr(no_reference_prob=prob))
                train_step(model, "corr")
                use, previous_sigma = expected_draws(prob, sigma, labels)
                counts = model.last_corr_reference
                self.assertEqual(float(counts[0]), float(use.sum()))
                self.assertEqual(float(counts[1]), float((labels != 1000).sum()))
                gap = (previous_sigma - sigma.reshape(-1))[use].sum()
                torch.testing.assert_close(counts[2], gap)

    def test_the_reference_sigma_stops_at_one(self):
        for mode, previous_index in (("model_velocity", 2), ("same_noise", 0)):
            with self.subTest(mode):
                model = perturb(build_corr(no_reference_prob=0.0, skip_intervals=[50], previous_state_mode=mode))
                with captured_inputs(model.t_embedder) as times:
                    train_step(model, "corr", sigma_value=0.98)
                previous_time = times[previous_index]
                self.assertTrue(torch.equal(previous_time, torch.full_like(previous_time, 1000.0)))


class SameNoiseReferenceTests(unittest.TestCase):
    def build(self, kind, **overrides):
        config = dict(previous_state_mode="same_noise", **overrides)
        model = build_corr(**config) if kind == "corr" else build_clean_corr(corr_overrides=config, align_lam=1.0)
        return perturb_correctors(perturb(model))

    def test_reference_uses_the_original_pair_and_only_one_auxiliary_forward(self):
        x, t, z, noise, sigma = batch()
        labels = dropped_labels()
        use, previous_sigma_all = expected_draws(0.5, sigma, labels)
        self.assertTrue(bool(use.any()))
        self.assertTrue(bool(((labels != 1000) & ~use).any()))
        index = torch.where(use)[0]
        previous_sigma = previous_sigma_all[index]
        scale = previous_sigma.view(-1, 1, 1, 1, 1)
        previous = (1 - scale) * z[index] + scale * noise[index]
        for kind in ("corr", "clean_corr"):
            with self.subTest(kind):
                model = self.build(kind)
                with captured_inputs(model.x_embedder) as patches, captured_inputs(model.t_embedder) as times:
                    train_step(model, kind)
                # One reference pass (input + empty reference), the optional
                # clean teacher (input only), and the student (input + reference).
                self.assertEqual(len(patches), 4 if kind == "corr" else 5)
                self.assertEqual(len(times), len(patches))
                torch.testing.assert_close(patches[0], previous.squeeze(2), rtol=0, atol=0)
                torch.testing.assert_close(patches[1], torch.zeros_like(patches[1]), rtol=0, atol=0)
                torch.testing.assert_close(times[0], previous_sigma * 1000)
                torch.testing.assert_close(patches[-2], x.squeeze(2), rtol=0, atol=0)
                if kind == "clean_corr":
                    torch.testing.assert_close(patches[2], z[labels != 1000].squeeze(2), rtol=0, atol=0)
                    torch.testing.assert_close(times[2], torch.zeros_like(times[2]), rtol=0, atol=0)
                with torch.no_grad():
                    y = model.y_embedder.embedding_table(labels)
                    prediction = model._run(
                        previous, previous_sigma * 1000, y[index], labels[index], model._no_reference(previous)
                    )
                    expected = torch.zeros_like(x)
                    expected[index] = previous - scale * model._velocity(prediction, previous)
                torch.testing.assert_close(patches[-1], expected.squeeze(2))
                expected_sigma = torch.ones(BATCH)
                expected_sigma[index] = previous_sigma
                torch.testing.assert_close(times[-1], expected_sigma * 1000)

    def test_reference_is_detached_and_main_forward_keeps_gradients_in_fp32_and_bf16(self):
        for kind in ("corr", "clean_corr"):
            for autocast in (False, True):
                with self.subTest(kind=kind, autocast=autocast):
                    model = self.build(kind, no_reference_prob=0.0)
                    _, _, z, noise, _ = batch()
                    z.requires_grad_()
                    noise.requires_grad_()
                    grad_modes = []
                    handle = model.x_embedder.register_forward_pre_hook(
                        lambda _module, args: grad_modes.append(torch.is_grad_enabled())
                    )
                    try:
                        output, grads = train_step(model, kind, autocast=autocast, z=z, noise=noise)
                    finally:
                        handle.remove()
                    self.assertEqual(grad_modes, ([False] * (2 if kind == "corr" else 3)) + [True, True])
                    self.assertIsNone(z.grad)
                    self.assertIsNone(noise.grad)
                    self.assertTrue(bool(torch.isfinite(output).all()))
                    expected = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                    self.assertEqual(set(grads), expected)
                    for name, gradient in grads.items():
                        self.assertTrue(bool(torch.isfinite(gradient).all()), name)
                    self.assertGreater(sum(float(g.abs().sum()) for name, g in grads.items() if is_corrector(name)), 0)

    def test_reference_selection_and_both_random_streams_are_unchanged(self):
        for kind in ("corr", "clean_corr"):
            records = []
            for mode in ("model_velocity", "same_noise"):
                model = self.build(kind)
                model.previous_state_mode = mode
                x, t, z, noise, sigma = batch()
                inputs = model_inputs(kind, LABELS, z, noise, sigma)
                torch.manual_seed(DROPOUT_SEED)
                model(x, t, **inputs)
                records.append((
                    torch.get_rng_state(), inputs["corr_router_generator"].get_state(), model.last_corr_reference
                ))
            for before, after in zip(*records):
                self.assertTrue(torch.equal(before, after), kind)

    def test_no_auxiliary_reference_pass_when_all_images_are_uncond_or_without_reference(self):
        for kind in ("corr", "clean_corr"):
            for all_uncond in (False, True):
                with self.subTest(kind=kind, all_uncond=all_uncond):
                    model = self.build(kind, no_reference_prob=0.999999)
                    labels = torch.full_like(LABELS, 1000) if all_uncond else LABELS
                    with captured_inputs(model.x_embedder) as patches:
                        train_step(model, kind, labels=labels)
                    teacher_calls = int(kind == "clean_corr" and not all_uncond)
                    self.assertEqual(len(patches), 2 + teacher_calls)
                    self.assertEqual(float(model.last_corr_reference[0]), 0.0)
                    torch.testing.assert_close(patches[-1], torch.zeros_like(patches[-1]), rtol=0, atol=0)

    def test_four_and_five_dimensional_latents_produce_the_same_prediction(self):
        for kind in ("corr", "clean_corr"):
            outputs = []
            model = self.build(kind)
            for ndim in (4, 5):
                x, t, z, noise, sigma = batch()
                if ndim == 4:
                    x, z, noise = (value.squeeze(2) for value in (x, z, noise))
                torch.manual_seed(DROPOUT_SEED)
                outputs.append(model(x, t, **model_inputs(kind, LABELS, z, noise, sigma)).detach())
            torch.testing.assert_close(*outputs, rtol=0, atol=0)

    def test_eval_needs_no_training_pair_and_is_identical_between_modes(self):
        x, t, z, _, sigma = batch()
        for kind in ("corr", "clean_corr"):
            model = self.build(kind).eval()
            for reference in ({}, dict(corr_reference_x0=z, corr_reference_sigma=sigma)):
                outputs = []
                for mode in ("model_velocity", "same_noise"):
                    model.previous_state_mode = mode
                    with torch.no_grad():
                        outputs.append(model(x, t, context=LABELS, **reference))
                torch.testing.assert_close(*outputs, rtol=0, atol=0)


class CorrectionTests(unittest.TestCase):
    def test_selection_gate_and_contrastive_clusters_use_the_corrected_scores(self):
        model = bias_toward(perturb(build_corr()), expert=2, value=5.0)
        blocks = model.moe_blocks()
        assignments = {id(block): [] for block in blocks}
        for block in blocks:
            original = block.compute_routing_contrastive_loss

            def wrapper(token_embeddings, cluster_assignments, use_top_k=False, _block=block, _original=original):
                assignments[id(_block)].append(cluster_assignments.clone())
                return _original(token_embeddings, cluster_assignments, use_top_k=use_top_k)

            block.compute_routing_contrastive_loss = wrapper
        try:
            with contextlib.ExitStack() as stack:
                routings = [stack.enter_context(captured_routing(block)) for block in blocks]
                train_step(model, "corr")
        finally:
            for block in blocks:
                del block.compute_routing_contrastive_loss
        for block, records in zip(blocks, routings):
            hidden_states, labels, weights, indices = records[-1]
            routed = labels.view(-1, 1).expand(-1, SEQ_LEN).reshape(-1) != 1000
            flat_indices = indices.reshape(-1, block.top_k)
            flat_weights = weights.reshape(-1, block.top_k)
            self.assertTrue(torch.equal(flat_indices[routed, 0], torch.full_like(flat_indices[routed, 0], 2)))
            self.assertTrue(torch.equal(
                flat_indices[~routed, 0], torch.full_like(flat_indices[~routed, 0], block.num_experts - 1)
            ))
            cosines = routed_cosines(block, hidden_states, labels)
            torch.testing.assert_close(flat_weights[routed, 0], cosines[:, 2] + 5.0)
            self.assertTrue(torch.equal(assignments[id(block)][-1][:, 0], flat_indices[routed, 0]))

    def test_fm_trains_the_selected_experts_correction_and_the_alignment_trains_all(self):
        model = bias_toward(perturb(build_corr()), expert=2, value=5.0)
        _, grads = train_step(model, "corr")
        aligned = bias_toward(perturb(build_clean_corr(align_lam=1.0)), expert=2, value=5.0)
        _, aligned_grads = train_step(aligned, "clean_corr")
        others = [0, 1, 3]
        for index, block in enumerate(model.moe_blocks()):
            name = next(key for key, value in model.named_parameters() if value is block.corrector.out.bias)
            fm_grad = grads[name]
            self.assertGreater(float(fm_grad[2].abs()), 0.0)
            self.assertTrue(torch.equal(fm_grad[others], torch.zeros_like(fm_grad[others])), index)
            self.assertTrue(bool((aligned_grads[name][others].abs() > 0).all()), index)

    def test_the_corrector_inputs_carry_no_gradient(self):
        model = perturb_correctors(perturb(build_corr()))
        flags = []
        handles = []
        for block in model.moe_blocks():
            for layer in (
                block.corrector.reference_proj,
                block.corrector.score_proj,
                block.corrector.condition_proj,
                block.corrector.reference_sigma_proj,
            ):
                handles.append(layer.register_forward_pre_hook(
                    lambda _module, args: flags.append(args[0].requires_grad)
                ))
        try:
            train_step(model, "corr")
        finally:
            for handle in handles:
                handle.remove()
        self.assertTrue(flags)
        self.assertFalse(any(flags))

    def test_the_alignment_uses_the_corrected_student_scores(self):
        tau = 0.2
        model = perturb_correctors(perturb(build_clean_corr(align_lam=1.0, align_temperature=tau)))
        blocks = model.moe_blocks()
        for block in blocks:
            block.keep_cosines = True
        with contextlib.ExitStack() as stack:
            inputs = [stack.enter_context(captured_routing(block)) for block in blocks]
            deltas = [stack.enter_context(captured_outputs(block.corrector)) for block in blocks]
            train_step(model, "clean_corr")
        for block, routing, delta in zip(blocks, inputs, deltas):
            hidden_states, labels, _, _ = routing[-1]  # the student pass routes last
            corrected = routed_cosines(block, hidden_states, labels) + delta[-1].float()
            torch.testing.assert_close(block.last_student_cosines, corrected)
            log_q = F.log_softmax(block.last_teacher_cosines / tau, dim=1)
            log_p = F.log_softmax(corrected / tau, dim=1)
            kl = (log_q.exp() * (log_q - log_p)).sum(dim=1).mean()
            torch.testing.assert_close(block.last_kl, kl)

    def test_the_teacher_routes_without_the_correction(self):
        model = bias_toward(perturb(build_clean_corr(align_lam=1.0)), expert=2, value=5.0)
        train_step(model, "clean_corr")
        for block in model.moe_blocks():
            student_load = block.last_student_load
            teacher_load = block.last_teacher_load
            self.assertEqual(float(student_load[2]), float(student_load.sum()))
            self.assertLess(float(teacher_load[2]), float(teacher_load.sum()))

    def test_a_noisy_teacher_is_refused(self):
        with self.assertRaisesRegex(ValueError, "teacher_sigma_ratio"):
            build_clean_corr(teacher_sigma_ratio=0.5)


class DiagnosticsTests(unittest.TestCase):
    def test_statistics_cover_every_prototype_routed_token(self):
        model = perturb(build_corr())
        train_step(model, "corr")
        labels = dropped_labels()
        num_routed_tokens = float((labels != 1000).sum()) * SEQ_LEN
        reference_tokens = float(model.last_corr_reference[0]) * SEQ_LEN
        row = {name: index for index, name in enumerate(corr_router.CORR_BIN_ROWS)}
        for block in model.moe_blocks():
            bins = block.last_corr_bins
            self.assertEqual(tuple(bins.shape), (len(corr_router.CORR_BIN_ROWS), corr_router.NUM_SIGMA_BINS))
            self.assertEqual(float(bins[row["tokens"]].sum()), num_routed_tokens)
            self.assertEqual(float(bins[row["ref_tokens"]].sum()), reference_tokens)
            self.assertEqual(float(bins[row["noref_tokens"]].sum()), num_routed_tokens - reference_tokens)
            # A fresh corrector changes nothing.
            self.assertEqual(float(bins[row["delta_abs"]].sum()), 0.0)
            self.assertEqual(float(bins[row["switch_ref"]].sum() + bins[row["switch_noref"]].sum()), 0.0)

    def test_switches_count_against_the_uncorrected_choice(self):
        model = bias_toward(perturb(build_corr()), expert=2, value=5.0)
        blocks = model.moe_blocks()
        with contextlib.ExitStack() as stack:
            routings = [stack.enter_context(captured_routing(block)) for block in blocks]
            train_step(model, "corr")
        row = {name: index for index, name in enumerate(corr_router.CORR_BIN_ROWS)}
        for block, records in zip(blocks, routings):
            hidden_states, labels, _, _ = records[-1]
            uncorrected = routed_cosines(block, hidden_states, labels).argmax(dim=1)
            switched = float((uncorrected != 2).sum())
            bins = block.last_corr_bins
            self.assertEqual(float(bins[row["switch_ref"]].sum() + bins[row["switch_noref"]].sum()), switched)
            torch.testing.assert_close(bins[row["delta_abs"]].sum(), bins[row["tokens"]].sum() * 5.0 / 4)

    def test_the_collector_reports_every_statistic(self):
        import train

        model = bias_toward(perturb(build_corr()), expert=2, value=5.0)
        train_step(model, "corr")
        stats = train._collect_corr_router_stats(model)
        self.assertEqual(set(stats), set(train.CORR_ROUTER_STAT_NAMES))
        counts = model.last_corr_reference
        torch.testing.assert_close(stats["corr_router_ref_frac"], counts[0] / counts[1])
        torch.testing.assert_close(stats["corr_router_delta_abs"], torch.tensor(5.0 / 4))
        self.assertEqual(train._collect_corr_router_stats(build_conference()), {})

    def test_every_image_dropped_still_trains_every_weight(self):
        # DDP (find_unused_parameters=False) needs a gradient for every parameter
        # even on a rank whose images were all dropped to the uncond label.
        all_dropped = torch.full_like(LABELS, 1000)
        for builder, kind in ((build_corr, "corr"), (lambda: build_clean_corr(align_lam=1.0), "clean_corr")):
            with self.subTest(kind):
                model = perturb(builder())
                _, grads = train_step(model, kind, labels=all_dropped)
                trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
                self.assertEqual(set(grads), trainable)
                self.assertEqual(model.last_corr_reference.tolist(), [0.0, 0.0, 0.0])
                for block in model.moe_blocks():
                    self.assertEqual(float(block.last_corr_bins.abs().sum()), 0.0)

    def test_tied_scores_do_not_read_as_switches(self):
        # top_k = 2 with two identical prototypes: every token ties between them.
        torch.manual_seed(0)
        model = corr_router.DiT(MoE_config=moe_config(top_k=2), corr_router_config=corr_config(), **MODEL_KWARGS)
        perturb(model)
        with torch.no_grad():
            for block in model.moe_blocks():
                block.cluster_centers[1].copy_(block.cluster_centers[0])
        train_step(model, "corr")
        row = {name: index for index, name in enumerate(corr_router.CORR_BIN_ROWS)}
        for block in model.moe_blocks():
            bins = block.last_corr_bins
            self.assertGreater(float(bins[row["tokens"]].sum()), 0.0)
            self.assertEqual(float(bins[row["switch_ref"]].sum() + bins[row["switch_noref"]].sum()), 0.0)


class SamplingTests(unittest.TestCase):
    def test_sampling_reference_is_the_steps_clean_estimate(self):
        generator = torch.Generator().manual_seed(4)
        latents = torch.randn(3, 4, 1, 8, 8, generator=generator)
        prediction = torch.randn(3, 8, 8, 8, generator=generator)
        sigma = torch.tensor(0.37)
        reference = corr_router.sampling_reference(latents, prediction, sigma)
        torch.testing.assert_close(
            reference["corr_reference_x0"], latents - 0.37 * prediction[:, :4].unsqueeze(2)
        )
        torch.testing.assert_close(reference["corr_reference_sigma"], torch.full((3,), 0.37))

    def test_the_previous_euler_point_gives_the_same_estimate(self):
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

        scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)
        sigmas = torch.linspace(1, 0, 251)[:250].numpy()
        scheduler.set_timesteps(sigmas=sigmas, device="cpu")
        torch.testing.assert_close(scheduler.timesteps, scheduler.sigmas[:-1] * 1000)
        generator = torch.Generator().manual_seed(6)
        latents = torch.randn(2, 4, 1, 8, 8, generator=generator)
        velocity = torch.randn(2, 4, 8, 8, generator=generator)
        current = corr_router.sampling_reference(latents, velocity, scheduler.sigmas[0])
        stepped = scheduler.step(velocity.unsqueeze(2), scheduler.timesteps[0], latents, return_dict=False)[0]
        from_next = corr_router.sampling_reference(stepped, velocity, scheduler.sigmas[1])
        torch.testing.assert_close(current["corr_reference_x0"], from_next["corr_reference_x0"])

    def test_the_eval_forward_routes_with_the_reference(self):
        model = perturb_correctors(perturb(build_corr())).eval()
        x, t, _, _, _ = batch()
        reference_x0 = torch.randn(x.shape, generator=torch.Generator().manual_seed(3))
        with torch.no_grad():
            without = model(x, t, context=LABELS)
            empty = model(x, t, context=LABELS, corr_reference_x0=torch.zeros_like(x),
                          corr_reference_sigma=torch.ones(BATCH))
            with_reference = model(x, t, context=LABELS, corr_reference_x0=reference_x0,
                                   corr_reference_sigma=t / 1000 + 0.01)
        self.assertTrue(torch.equal(empty[:, :, :, :], without))
        self.assertFalse(torch.equal(with_reference, without))

    def test_a_mismatched_reference_is_refused(self):
        model = build_corr().eval()
        x, t, _, _, _ = batch()
        with torch.no_grad():
            with self.assertRaisesRegex(ValueError, "go together"):
                model(x, t, context=LABELS, corr_reference_x0=torch.zeros_like(x))
            with self.assertRaisesRegex(ValueError, "do not match"):
                model(x, t, context=LABELS, corr_reference_x0=torch.zeros_like(x[:2]),
                      corr_reference_sigma=torch.ones(2))


class _RecordingModel(torch.nn.Module):
    """Forwards to ``inner`` and records every call's input, labels, reference and output."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.calls = []

    def forward(self, x, timestep, **kwargs):
        output = self.inner(x, timestep, **kwargs)
        self.calls.append({
            "x": x.detach().clone(),
            "timestep": timestep.detach().clone(),
            "context": kwargs["context"].clone(),
            "reference": {
                key: kwargs[key].detach().clone()
                for key in ("corr_reference_x0", "corr_reference_sigma")
                if key in kwargs
            },
            "output": output.detach().clone(),
        })
        return output


class SamplerLoopTests(unittest.TestCase):
    STEPS = 4
    SAMPLER = EasyDict(num_train_timesteps=1000, shift=1.0, sample_steps=STEPS, sample_shift=1.0,
                       use_gradient_checkpointing=False)

    def run_loop(self, model, uses_corr_reference, guide_scale=1.5, reference_velocity=None):
        import sample

        noise = torch.randn(2, 4, 1, 8, 8, generator=torch.Generator().manual_seed(8))
        y = torch.tensor([3, 7])
        y_null = torch.full((2,), 1000)
        cfg = EasyDict(self.SAMPLER)
        if reference_velocity is not None:
            cfg.corr_reference_velocity = reference_velocity
        with torch.no_grad():
            latents = sample.denoise_latents(
                model, noise, y, y_null, cfg, guide_scale, "cpu", uses_corr_reference
            )
        return noise, y, y_null, latents

    def test_only_the_conditional_pass_routes_with_the_previous_estimate(self):
        recorder = _RecordingModel(perturb_correctors(perturb(build_corr())).eval())
        self.run_loop(recorder, uses_corr_reference=True)
        cond = [call for call in recorder.calls if bool((call["context"] != 1000).all())]
        uncond = [call for call in recorder.calls if bool((call["context"] == 1000).all())]
        self.assertEqual((len(cond), len(uncond)), (self.STEPS, self.STEPS))
        self.assertTrue(all(not call["reference"] for call in uncond))
        self.assertFalse(cond[0]["reference"])
        sigmas = torch.linspace(1, 0, self.STEPS + 1)[: self.STEPS]
        for step in range(self.STEPS):
            torch.testing.assert_close(cond[step]["timestep"], torch.full((2,), float(sigmas[step]) * 1000))
        for step in range(1, self.STEPS):
            previous = cond[step - 1]
            expected = previous["x"] - sigmas[step - 1] * previous["output"][:, :4].unsqueeze(2)
            torch.testing.assert_close(cond[step]["reference"]["corr_reference_x0"], expected)
            torch.testing.assert_close(
                cond[step]["reference"]["corr_reference_sigma"], torch.full((2,), float(sigmas[step - 1]))
            )
        # The next batch starts without a reference again.
        recorder.calls.clear()
        self.run_loop(recorder, uses_corr_reference=True)
        self.assertFalse(recorder.calls[0]["reference"])

    def test_cfg_reference_and_actual_step_use_the_same_fused_velocity(self):
        for builder in (build_corr, build_clean_corr):
            for autocast in (False, True):
                with self.subTest(model=builder.__name__, autocast=autocast):
                    recorder = _RecordingModel(perturb_correctors(perturb(builder())).eval())
                    region = (torch.autocast(device_type="cpu", dtype=torch.bfloat16)
                              if autocast else contextlib.nullcontext())
                    with region:
                        _, _, _, result = self.run_loop(recorder, True, 1.5, "cfg")
                    cond = recorder.calls[::2]
                    uncond = recorder.calls[1::2]
                    self.assertEqual(len(recorder.calls), 2 * self.STEPS)
                    self.assertFalse(cond[0]["reference"])
                    self.assertTrue(all(not call["reference"] for call in uncond))
                    for step in range(self.STEPS):
                        current = cond[step]
                        torch.testing.assert_close(current["x"], uncond[step]["x"], rtol=0, atol=0)
                        fused = uncond[step]["output"] + 1.5 * (current["output"] - uncond[step]["output"])
                        velocity = fused[:, :4].float().unsqueeze(2)
                        sigma = 1 - step / self.STEPS
                        # The Euler scheduler computes in float32, then returns
                        # the next latent in the prediction's dtype.
                        expected_next = (current["x"].float() - velocity / self.STEPS).to(fused.dtype)
                        actual_next = cond[step + 1]["x"] if step + 1 < self.STEPS else result
                        torch.testing.assert_close(actual_next, expected_next)
                        if step + 1 < self.STEPS:
                            reference = cond[step + 1]["reference"]
                            torch.testing.assert_close(reference["corr_reference_x0"],
                                                       current["x"].float() - sigma * velocity)
                            torch.testing.assert_close(reference["corr_reference_sigma"],
                                                       torch.full((2,), sigma))

    def test_reference_modes_are_identical_at_cfg_one(self):
        for builder in (build_corr, build_clean_corr):
            with self.subTest(model=builder.__name__):
                model = perturb_correctors(perturb(builder())).eval()
                conditional, fused = _RecordingModel(model), _RecordingModel(model)
                a = self.run_loop(conditional, True, 1.0, "conditional")[-1]
                b = self.run_loop(fused, True, 1.0, "cfg")[-1]
                self.assertTrue(torch.equal(a, b))
                self.assertEqual(len(fused.calls), self.STEPS)
                for left, right in zip(conditional.calls, fused.calls):
                    self.assertTrue(torch.equal(left["x"], right["x"]))
                    for key in left["reference"]:
                        self.assertTrue(torch.equal(left["reference"][key], right["reference"][key]))

    def test_explicit_conditional_preserves_the_default_and_cfg_changes_it(self):
        model = perturb_correctors(perturb(build_corr())).eval()
        default = self.run_loop(model, True)[-1]
        conditional = self.run_loop(model, True, reference_velocity="conditional")[-1]
        fused = self.run_loop(model, True, reference_velocity="cfg")[-1]
        self.assertTrue(torch.equal(default, conditional))
        self.assertFalse(torch.equal(conditional, fused))

    def test_invalid_reference_mode_or_model_is_refused(self):
        model = perturb(build_conference()).eval()
        with self.assertRaisesRegex(ValueError, "conditional or cfg"):
            self.run_loop(model, True, reference_velocity="invalid")
        with self.assertRaisesRegex(ValueError, "correction-router model"):
            self.run_loop(model, False, reference_velocity="cfg")

    def test_other_models_sample_as_before(self):
        """denoise_latents against the loop sample.py inlined before corr_router."""
        from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
        import sample

        model = perturb(build_conference()).eval()
        for guide_scale in (1.0, 1.5):
            with self.subTest(guide_scale=guide_scale):
                recorder = _RecordingModel(model)
                noise, y, y_null, latents = self.run_loop(recorder, False, guide_scale)
                self.assertTrue(all(not call["reference"] for call in recorder.calls))
                cfg = self.SAMPLER
                scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=cfg.num_train_timesteps, shift=cfg.shift)
                timesteps, _ = sample.retrieve_timesteps(
                    scheduler, device="cpu", sigmas=sample.get_sampling_sigmas(cfg.sample_steps, cfg.sample_shift)
                )
                expected = noise
                with torch.no_grad():
                    for t in timesteps:
                        timestep = torch.stack([t] * len(expected))
                        cond = model(expected, timestep, context=y, use_gradient_checkpointing=False)
                        if guide_scale > 1.0:
                            uncond = model(expected, timestep, context=y_null, use_gradient_checkpointing=False)
                            prediction = uncond + guide_scale * (cond - uncond)
                        else:
                            prediction = cond
                        if prediction.shape[1] != expected.shape[1]:
                            prediction, _ = prediction.chunk(2, dim=1)
                        expected = scheduler.step(prediction.unsqueeze(2), t, expected, return_dict=False)[0]
                self.assertTrue(torch.equal(latents, expected))


class SamplingCheckpointTests(unittest.TestCase):
    def test_inference_reads_exact_source_checkpoint_without_using_its_output_bucket(self):
        from pathlib import Path
        import sample

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".corr_sampling_") as directory:
            root = Path(directory)
            source = root / "training" / "checkpoints"
            destination = root / "inference"
            source.mkdir(parents=True)
            (destination / "checkpoints").mkdir(parents=True)
            expected = source / "ckpt_step_300000.pth"
            expected.write_bytes(b"source checkpoint placeholder")
            (source / "ckpt_step_500000.pth").write_bytes(b"later checkpoint placeholder")
            (destination / "checkpoints" / expected.name).write_bytes(b"wrong checkpoint placeholder")
            cfg = EasyDict(output_dir=str(destination), sample_checkpoint_dir=str(source),
                           step_list_for_sample=[300000])
            directory, paths, steps = sample.resolve_sampling_checkpoints(cfg)
            self.assertEqual((directory, paths, steps), (str(source), [str(expected)], [300000]))
            expected.unlink()
            with self.assertRaisesRegex(FileNotFoundError, "300000"):
                sample.resolve_sampling_checkpoints(cfg)
            expected.touch()
            with self.assertRaisesRegex(FileNotFoundError, "Empty or invalid"):
                sample.resolve_sampling_checkpoints(cfg)
            cfg.step_list_for_sample = []
            with self.assertRaisesRegex(ValueError, "explicit"):
                sample.resolve_sampling_checkpoints(cfg)

    def test_reference_modes_have_separate_sample_directories(self):
        import sample

        cfg = EasyDict(image_size=256, global_seed=0, num_fid_samples=50000, sample_batch_size=128)
        expected = "img256_cfg1.5_seed0_FID50K_bs128_ema"
        self.assertEqual(sample.sampling_folder_name(cfg, 1.5), expected)
        cfg.corr_reference_velocity = "cfg"
        self.assertEqual(sample.sampling_folder_name(cfg, 1.5), expected + "_corr_ref_cfg")
        cfg.corr_reference_velocity = "conditional"
        self.assertEqual(sample.sampling_folder_name(cfg, 1.5), expected)

    def test_separate_experiments_share_standard_internal_directory_names(self):
        import sample

        cfg = EasyDict(image_size=256, global_seed=0, num_fid_samples=50000, sample_batch_size=128,
                       corr_reference_separate_output=True)
        for scale in (1.0, 1.5):
            expected = f"img256_cfg{scale}_seed0_FID50K_bs128_ema"
            for mode in ("conditional", "cfg"):
                cfg.corr_reference_velocity = mode
                self.assertEqual(sample.sampling_folder_name(cfg, scale), expected)

    def test_standard_layout_requires_a_separate_output_and_explicit_source(self):
        from pathlib import Path
        import sample

        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix=".corr_layout_") as directory:
            root = Path(directory)
            source = root / "training" / "checkpoints"
            source.mkdir(parents=True)
            checkpoint = source / "ckpt_step_300000.pth"
            checkpoint.write_bytes(b"checkpoint placeholder")
            cfg = EasyDict(output_dir=str(root / "conditional"), corr_reference_separate_output=True,
                           step_list_for_sample=[300000])
            with self.assertRaisesRegex(ValueError, "sample_checkpoint_dir"):
                sample.resolve_sampling_checkpoints(cfg)
            cfg.sample_checkpoint_dir = str(source)
            for output in (root / "training", root / "training" / "nested", root):
                cfg.output_dir = str(output)
                with self.assertRaisesRegex(ValueError, "separate experiment"):
                    sample.resolve_sampling_checkpoints(cfg)
            for mode in ("conditional", "cfg"):
                cfg.output_dir = str(root / mode)
                cfg.corr_reference_velocity = mode
                self.assertEqual(sample.resolve_sampling_checkpoints(cfg)[1], [str(checkpoint)])
            cfg.corr_reference_separate_output = "true"
            with self.assertRaisesRegex(ValueError, "boolean"):
                sample.resolve_sampling_checkpoints(cfg)


def _ddp_worker(rank, world_size, init_file, output_dir, kind, previous_state_mode):
    """Two DDP steps as train.py runs them: bf16 forward, the stat collectives, backward, update.

    Rank 1 starts with a batch of uncond images only, so its first step trains
    every parameter through the zero-weight fallback while rank 0 runs every pass.
    """
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        import train  # the generator, stat collectors and reducer under test

        config = dict(previous_state_mode=previous_state_mode)
        model = build_corr(**config) if kind == "corr" else build_clean_corr(corr_overrides=config, align_lam=1.0)
        model = perturb_correctors(perturb(model))
        ddp_model = DistributedDataParallel(model)
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=1e-2)
        trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
        steps = []
        for step in range(2):
            ddp_model.train()
            x, t, z, noise, sigma = batch(seed=100 * rank + step)
            torch.manual_seed(1000 * rank + step)
            # Rank 1's first step has only uncond images: no reference, no teacher
            # pass and no prototype-routed token on that rank.
            labels = torch.full_like(LABELS, 1000) if (rank == 1 and step == 0) else LABELS
            inputs = {
                "context": labels,
                "corr_router_sigma": sigma,
                "corr_router_generator": train._corr_router_generator(0, step, 0, rank, x.device),
            }
            if kind == "clean_corr":
                inputs.update(clean_route_z=z, clean_route_noise=noise, clean_route_sigma=sigma)
            else:
                inputs.update(corr_router_z=z, corr_router_noise=noise)
            with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                output = ddp_model(x, t, **inputs)
            stats = train._collect_corr_router_stats(ddp_model)
            if kind == "clean_corr":
                stats.update(train._collect_clean_route_stats(ddp_model))
            output.float().square().mean().backward()
            with_grad = {name for name, parameter in model.named_parameters() if parameter.grad is not None}
            if kind == "clean_corr":
                train._reduce_clean_route_stats(stats)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            steps.append({
                "missing_grads": sorted(trainable - with_grad),
                "stats": {name: float(value) for name, value in stats.items()},
                "reference_counts": model.last_corr_reference.tolist(),
            })
        torch.save(steps, os.path.join(output_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs torch.distributed with gloo")
class DistributedTests(unittest.TestCase):
    def test_two_ddp_steps_train_every_weight_and_agree_on_the_stats(self):
        import train

        world_size = 2
        for kind, mode in (
            ("corr", "model_velocity"), ("corr", "same_noise"),
            ("clean_corr", "model_velocity"), ("clean_corr", "same_noise"),
        ):
            with self.subTest(kind=kind, mode=mode), tempfile.TemporaryDirectory() as output_dir:
                mp.spawn(
                    _ddp_worker,
                    args=(world_size, os.path.join(output_dir, "init"), output_dir, kind, mode),
                    nprocs=world_size,
                    join=True,
                )
                results = [torch.load(os.path.join(output_dir, f"rank{rank}.pt")) for rank in range(world_size)]
                for rank, steps in enumerate(results):
                    self.assertEqual(len(steps), 2)
                    for record in steps:
                        self.assertEqual(record["missing_grads"], [], f"rank {rank}")
                        self.assertTrue(set(train.CORR_ROUTER_STAT_NAMES) <= set(record["stats"]))
                # After the reduction every rank logs the same values.
                for step in range(2):
                    reference = results[0][step]["stats"]
                    self.assertEqual(set(reference), set(results[1][step]["stats"]))
                    for name, value in reference.items():
                        self.assertAlmostEqual(results[1][step]["stats"][name], value, places=5, msg=name)
                # The logged ratios are global: rank 1 had no routed image at step 0.
                self.assertEqual(results[1][0]["reference_counts"], [0.0, 0.0, 0.0])
                for step in range(2):
                    counts = [results[rank][step]["reference_counts"] for rank in range(world_size)]
                    references = sum(count[0] for count in counts)
                    routed = sum(count[1] for count in counts)
                    self.assertAlmostEqual(
                        results[0][step]["stats"]["corr_router_ref_frac"], references / routed, places=5
                    )


class TrainingIntegrationTests(unittest.TestCase):
    def test_the_models_are_registered_with_their_inputs(self):
        import train

        self.assertIs(train.model_dict["ProMoE_TC_B_corr_router"][0], corr_router.DiT)
        self.assertIs(train.model_dict["ProMoE_TC_B_clean_route_corr_router"][0], clean_corr.DiT)
        self.assertEqual(
            train.CORR_ROUTER_MODELS, {"ProMoE_TC_B_corr_router", "ProMoE_TC_B_clean_route_corr_router"}
        )
        self.assertIn("ProMoE_TC_B_clean_route_corr_router", train.CLEAN_ROUTE_MODELS)
        self.assertNotIn("ProMoE_TC_B_corr_router", train.CLEAN_ROUTE_MODELS)

    def test_the_reference_generator_depends_only_on_the_position(self):
        import train

        def draws(*position):
            return torch.rand(8, generator=train._corr_router_generator(*position, device="cpu"))

        self.assertTrue(torch.equal(draws(3, 120, 0, 1), draws(3, 120, 0, 1)))
        for other in ((3, 121, 0, 1), (3, 120, 1, 1), (3, 120, 0, 2), (4, 120, 0, 1)):
            self.assertFalse(torch.equal(draws(3, 120, 0, 1), draws(*other)), other)


class ConfigTests(unittest.TestCase):
    def test_previous_state_mode_defaults_to_the_original_velocity_method(self):
        self.assertEqual(corr_router.parse_corr_router_config(corr_config())["previous_state_mode"], "model_velocity")
        for mode in ("model_velocity", "same_noise"):
            self.assertEqual(corr_router.parse_corr_router_config(corr_config(previous_state_mode=mode))[
                "previous_state_mode"
            ], mode)

    def test_invalid_settings_are_refused(self):
        cases = {
            "missing key": {key: value for key, value in corr_config().items() if key != "hidden_dim"},
            "unknown key": dict(corr_config(), hidden=16),
            "zero hidden_dim": corr_config(hidden_dim=0),
            "float hidden_dim": corr_config(hidden_dim=16.0),
            "bool hidden_dim": corr_config(hidden_dim=True),
            "empty intervals": corr_config(skip_intervals=[]),
            "zero interval": corr_config(skip_intervals=[0, 4]),
            "interval of a thousand": corr_config(skip_intervals=[1000]),
            "repeated interval": corr_config(skip_intervals=[4, 4]),
            "float interval": corr_config(skip_intervals=[4.5]),
            "string intervals": corr_config(skip_intervals="1,4"),
            "prob of one": corr_config(no_reference_prob=1.0),
            "negative prob": corr_config(no_reference_prob=-0.1),
            "nan prob": corr_config(no_reference_prob=float("nan")),
            "unknown previous state": corr_config(previous_state_mode="typo"),
            "bool previous state": corr_config(previous_state_mode=True),
            "null previous state": corr_config(previous_state_mode=None),
        }
        for name, config in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                corr_router.parse_corr_router_config(config)
        with self.assertRaises(ValueError):
            corr_router.parse_corr_router_config(None)

    def test_unsupported_block_settings_are_refused(self):
        cases = {
            "load_balance_loss_coef": moe_config(load_balance_loss_coef=0.1),
            "router_weight_mode": moe_config(router_weight_mode="softmax"),
            "phase_metric": moe_config(phase_metric_config={"enabled": True}),
            "use_uncond_expert": moe_config(use_uncond_expert=False),
        }
        for message, config in cases.items():
            with self.subTest(message), self.assertRaisesRegex(ValueError, message):
                corr_router.DiT(MoE_config=config, corr_router_config=corr_config(), **MODEL_KWARGS)

    def test_training_needs_the_reference_inputs(self):
        x, t, z, noise, sigma = batch()
        model = build_corr().train()
        with self.assertRaisesRegex(ValueError, "corr_router_generator"):
            model(x, t, context=LABELS, corr_router_sigma=sigma)
        with self.assertRaisesRegex(ValueError, "for sampling"):
            model(x, t, **model_inputs("corr", LABELS, z, noise, sigma), corr_reference_x0=torch.zeros_like(x),
                  corr_reference_sigma=torch.ones(BATCH))
        with self.assertRaisesRegex(ValueError, "1000 \\* sigma"):
            model(x, t * 0.5, **model_inputs("corr", LABELS, z, noise, sigma))
        clean = build_clean_corr().train()
        with self.assertRaisesRegex(ValueError, "corr_router_generator"):
            clean(x, t, **model_inputs("clean_route", LABELS, z, noise, sigma))
        inputs = model_inputs("clean_corr", LABELS, z, noise, sigma)
        inputs["corr_router_sigma"] = sigma * 0.5
        with self.assertRaisesRegex(ValueError, "same noise levels"):
            clean(x, t, **inputs)

    def test_same_noise_requires_the_original_latents_and_noise_with_matching_shapes(self):
        x, t, z, noise, sigma = batch()
        for kind in ("corr", "clean_corr"):
            config = dict(previous_state_mode="same_noise")
            model = build_corr(**config) if kind == "corr" else build_clean_corr(corr_overrides=config)
            prefix = "corr_router" if kind == "corr" else "clean_route"
            for field, source in (("z", z), ("noise", noise)):
                for invalid in (None, source[:-1], source.squeeze(2), source.to("meta")):
                    with self.subTest(kind=kind, field=field, invalid_type=type(invalid).__name__):
                        inputs = model_inputs(kind, LABELS, z, noise, sigma)
                        inputs[f"{prefix}_{field}"] = invalid
                        with self.assertRaises(ValueError):
                            model(x, t, **inputs)


if __name__ == "__main__":
    unittest.main()
