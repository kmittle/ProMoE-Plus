import unittest
from unittest.mock import patch

import torch

import models.models_ProMoE_TC_repa_multi_align_denoising_regret as fdrr
from models.models_ProMoE_TC_repa_multi_align import (
    AddAuxiliaryLoss,
    suppress_auxiliary_loss_backward,
)


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _build_tiny_fdrr(num_routed_experts=4):
    torch.manual_seed(7)
    moe_config = _AttrDict(
        num_routed_experts=num_routed_experts,
        moe_intermediate_size=48,
        shared_expert_intermediate_size=48,
        load_balance_loss_coef=0,
        norm_topk_prob=False,
        seq_aux=False,
        use_shared_expert=True,
        interleave=True,
        init_MoeMLP=False,
        top_k=1,
        router_weight_mode="identity",
        routing_contrastive_lam=1.0,
        use_top_k_for_routing_contrastive=True,
        routing_contrastive_temperature=0.07,
    )
    repa_config = {
        "z_dims": [16],
        "projector_dim": 32,
        "router_hidden_dim": 32,
        "num_router_blocks": 1,
        "router_num_heads": 4,
        "align_blocks": [],
        "use_dynamic_coeff": False,
        "denoising_regret_block": 3,
        "denoising_regret_probe_interval": 1,
        "denoising_regret_token_ratio": 0.5,
        "denoising_regret_candidate_mode": "mixed",
        "denoising_regret_confidence_quantile": 0.0,
        "denoising_regret_temperature": 0.1,
        "denoising_regret_warmup_steps": 0,
        "denoising_regret_ramp_steps": 0,
        "denoising_regret_seed": 23,
    }
    model = fdrr.DiT(
        input_size=8,
        patch_size=2,
        in_channels=4,
        hidden_size=32,
        depth=6,
        num_heads=4,
        mlp_ratio=2,
        class_dropout_prob=0,
        num_classes=1000,
        learn_sigma=False,
        MoE_config=moe_config,
        repa_config=repa_config,
    ).train()

    # The production initializer starts from zero DiT gates/output. Open the
    # tiny model so its diffusion loss has a nonzero suffix gradient.
    with torch.no_grad():
        for block in model.blocks:
            hidden_size = block.adaLN_modulation[-1].bias.numel() // 6
            block.adaLN_modulation[-1].bias[5 * hidden_size:] = 1.0
        torch.nn.init.normal_(model.final_layer.linear.weight, std=0.1)
        torch.nn.init.normal_(model.final_layer.linear.bias, std=0.1)
    return model


def _set_routing_contrastive_strength(model, value):
    for block in model.blocks:
        if block.use_moe:
            block.mlp.routing_contrastive_lam = value


def _run_active_probe(model, inputs, timesteps, labels, target, strength):
    _set_routing_contrastive_strength(model, strength)
    model.zero_grad(set_to_none=True)
    captured_labels = []
    cosine_similarity = fdrr.F.cosine_similarity

    def capture_labels(*args, **kwargs):
        result = cosine_similarity(*args, **kwargs)
        captured_labels.append(result.detach().clone())
        return result

    with patch.object(
        fdrr.F,
        "cosine_similarity",
        side_effect=capture_labels,
    ):
        prediction, _, regret_loss = model(
            inputs,
            timesteps,
            labels,
            denoising_target=target,
            training_step=0,
        )
    if len(captured_labels) != 1:
        raise AssertionError(
            f"Expected one FDRR label tensor, captured {len(captured_labels)}"
        )

    (prediction.square().mean() + regret_loss).backward()
    center_gradient = (
        model.blocks[3].mlp.cluster_centers.grad.detach().clone()
    )
    return (
        prediction.detach(),
        regret_loss.detach(),
        captured_labels[0],
        center_gradient,
    )


def _check_suppression_excludes_auxiliary_gradient():
    value = torch.tensor([3.0], requires_grad=True)
    auxiliary_loss = value.square().sum()
    wrapped = AddAuxiliaryLoss.apply(value, auxiliary_loss)
    with suppress_auxiliary_loss_backward():
        isolated_gradient, = torch.autograd.grad(wrapped.square().sum(), value)
    torch.testing.assert_close(isolated_gradient, torch.tensor([6.0]))

    value = torch.tensor([3.0], requires_grad=True)
    auxiliary_loss = value.square().sum()
    wrapped = AddAuxiliaryLoss.apply(value, auxiliary_loss)
    ordinary_gradient, = torch.autograd.grad(wrapped.square().sum(), value)
    torch.testing.assert_close(ordinary_gradient, torch.tensor([12.0]))


def _check_real_fdrr_auxiliary_contract():
    model = _build_tiny_fdrr()
    inputs = torch.randn(3, 4, 8, 8)
    timesteps = torch.tensor([100.0, 300.0, 700.0])
    labels = torch.tensor([3, 5, 7])
    target = torch.randn_like(inputs)

    without_auxiliary = _run_active_probe(
        model, inputs, timesteps, labels, target, 0.0
    )
    with_auxiliary = _run_active_probe(
        model, inputs, timesteps, labels, target, 1.0
    )

    torch.testing.assert_close(
        without_auxiliary[0], with_auxiliary[0], rtol=0, atol=0
    )
    torch.testing.assert_close(
        without_auxiliary[1], with_auxiliary[1], rtol=0, atol=0
    )
    torch.testing.assert_close(
        without_auxiliary[2], with_auxiliary[2], rtol=0, atol=0
    )
    auxiliary_gradient = with_auxiliary[3] - without_auxiliary[3]
    assert auxiliary_gradient.norm().item() > 1e-4


class AuxiliaryIsolationTest(unittest.TestCase):
    def test_requires_at_least_two_routed_experts(self):
        with self.assertRaisesRegex(ValueError, "at least two routed experts"):
            _build_tiny_fdrr(num_routed_experts=1)

    def test_suppression_excludes_auxiliary_gradient(self):
        _check_suppression_excludes_auxiliary_gradient()

    def test_real_fdrr_auxiliary_contract(self):
        _check_real_fdrr_auxiliary_contract()


if __name__ == "__main__":
    unittest.main()
