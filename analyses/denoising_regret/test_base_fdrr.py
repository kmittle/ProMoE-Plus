import unittest
from unittest.mock import patch

import torch

import models.models_ProMoE_TC_denoising_regret as fdrr
from models.models_ProMoE_TC import (
    AddAuxiliaryLoss,
    DiT as BaseDiT,
    suppress_auxiliary_loss_backward,
)


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _model_kwargs():
    return {
        'input_size': 8,
        'patch_size': 2,
        'in_channels': 4,
        'hidden_size': 32,
        'depth': 6,
        'num_heads': 4,
        'mlp_ratio': 2,
        'class_dropout_prob': 0,
        'num_classes': 1000,
        'learn_sigma': False,
        'MoE_config': _AttrDict(
            num_routed_experts=4,
            moe_intermediate_size=48,
            shared_expert_intermediate_size=48,
            load_balance_loss_coef=0,
            norm_topk_prob=False,
            seq_aux=False,
            use_shared_expert=True,
            interleave=True,
            init_MoeMLP=False,
            top_k=1,
            router_weight_mode='identity',
            routing_contrastive_lam=1.0,
            use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07,
        ),
    }


def _regret_config(**overrides):
    config = {
        'denoising_regret_block': 3,
        'denoising_regret_probe_interval': 1,
        'denoising_regret_token_ratio': 0.5,
        'denoising_regret_candidate_mode': 'mixed',
        'denoising_regret_confidence_quantile': 0.0,
        'denoising_regret_temperature': 0.1,
        'denoising_regret_warmup_steps': 0,
        'denoising_regret_ramp_steps': 0,
        'denoising_regret_label_roll': 0,
        'denoising_regret_seed': 23,
        'denoising_regret_eps': 1e-6,
    }
    config.update(overrides)
    return config


def _build_fdrr(**config_overrides):
    torch.manual_seed(7)
    model = fdrr.DiT(
        **_model_kwargs(),
        denoising_regret_config=_regret_config(**config_overrides),
    ).train()

    # Zero-init is correct in production, but a nonzero suffix is needed to
    # exercise the inner diffusion-loss VJP in this tiny test.
    with torch.no_grad():
        for block in model.blocks:
            hidden_size = block.adaLN_modulation[-1].bias.numel() // 6
            block.adaLN_modulation[-1].bias[5 * hidden_size:] = 1.0
        torch.nn.init.normal_(model.final_layer.linear.weight, std=0.1)
        torch.nn.init.normal_(model.final_layer.linear.bias, std=0.1)
    return model


def _inputs():
    generator = torch.Generator().manual_seed(19)
    inputs = torch.randn(3, 4, 8, 8, generator=generator)
    timesteps = torch.tensor([100.0, 300.0, 700.0])
    labels = torch.tensor([3, 5, 7])
    target = torch.randn(3, 4, 8, 8, generator=generator)
    return inputs, timesteps, labels, target


def _set_routing_contrastive_strength(model, value):
    for block in model.blocks:
        if block.use_moe:
            block.mlp.routing_contrastive_lam = value


def _run_probe(model, strength):
    inputs, timesteps, labels, target = _inputs()
    _set_routing_contrastive_strength(model, strength)
    model.zero_grad(set_to_none=True)
    captured_labels = []
    cosine_similarity = fdrr.F.cosine_similarity

    def capture_labels(*args, **kwargs):
        result = cosine_similarity(*args, **kwargs)
        captured_labels.append(result.detach().clone())
        return result

    with patch.object(fdrr.F, 'cosine_similarity', side_effect=capture_labels):
        prediction, regret_loss = model(
            inputs,
            timesteps,
            labels,
            denoising_target=target.unsqueeze(2),
            training_step=0,
        )
    if len(captured_labels) != 1:
        raise AssertionError(
            f"Expected one FDRR label tensor, captured {len(captured_labels)}"
        )

    (prediction.square().mean() + regret_loss).backward()
    center_gradient = model.blocks[3].mlp.cluster_centers.grad.detach().clone()
    return prediction.detach(), regret_loss.detach(), captured_labels[0], center_gradient


class BaseFDRRTest(unittest.TestCase):
    def test_suppression_excludes_only_inner_auxiliary_gradient(self):
        value = torch.tensor([3.0], requires_grad=True)
        auxiliary_loss = value.square().sum()
        wrapped = AddAuxiliaryLoss.apply(value, auxiliary_loss)
        with suppress_auxiliary_loss_backward():
            isolated_gradient, = torch.autograd.grad(
                wrapped.square().sum(), value
            )
        torch.testing.assert_close(isolated_gradient, torch.tensor([6.0]))

        value = torch.tensor([3.0], requires_grad=True)
        auxiliary_loss = value.square().sum()
        wrapped = AddAuxiliaryLoss.apply(value, auxiliary_loss)
        ordinary_gradient, = torch.autograd.grad(wrapped.square().sum(), value)
        torch.testing.assert_close(ordinary_gradient, torch.tensor([12.0]))

    def test_eval_state_and_cfg_match_base(self):
        torch.manual_seed(11)
        base = BaseDiT(**_model_kwargs()).eval()
        fdrr_model = _build_fdrr().eval()
        fdrr_model.load_state_dict(base.state_dict(), strict=True)

        self.assertEqual(set(base.state_dict()), set(fdrr_model.state_dict()))
        self.assertEqual(
            sum(parameter.numel() for parameter in base.parameters()),
            sum(parameter.numel() for parameter in fdrr_model.parameters()),
        )

        inputs, timesteps, labels, _ = _inputs()
        with torch.no_grad():
            base_output = base(inputs, timesteps, labels)
            fdrr_output = fdrr_model(inputs, timesteps, labels)
        self.assertIsInstance(fdrr_output, torch.Tensor)
        torch.testing.assert_close(base_output, fdrr_output, rtol=0, atol=0)

        cfg_inputs = torch.cat([inputs[:2], inputs[:2]], dim=0)
        cfg_timesteps = torch.cat([timesteps[:2], timesteps[:2]], dim=0)
        cfg_labels = torch.tensor([3, 5, 3, 5])
        with torch.no_grad():
            cfg_output = fdrr_model.forward_with_cfg(
                cfg_inputs, cfg_timesteps, cfg_labels, cfg_scale=1.5
            )
        self.assertEqual(cfg_output.shape, cfg_inputs.shape)

    def test_labels_ignore_inner_auxiliary_but_outer_gradient_keeps_it(self):
        model = _build_fdrr()
        without_auxiliary = _run_probe(model, 0.0)
        with_auxiliary = _run_probe(model, 1.0)

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
        self.assertGreater(auxiliary_gradient.norm().item(), 1e-4)

    def test_regret_gradient_only_updates_target_centers(self):
        model = _build_fdrr()
        inputs, timesteps, labels, target = _inputs()
        model.zero_grad(set_to_none=True)
        _, regret_loss = model(
            inputs,
            timesteps,
            labels,
            denoising_target=target,
            training_step=0,
        )
        regret_loss.backward()

        nonzero_gradients = {
            name
            for name, parameter in model.named_parameters()
            if parameter.grad is not None and parameter.grad.abs().sum() > 0
        }
        self.assertEqual(
            nonzero_gradients,
            {'blocks.3.mlp.cluster_centers'},
        )

    def test_inactive_step_returns_differentiable_zero(self):
        model = _build_fdrr(denoising_regret_probe_interval=2)
        inputs, timesteps, labels, target = _inputs()
        prediction, regret_loss = model(
            inputs,
            timesteps,
            labels,
            denoising_target=target,
            training_step=1,
        )
        self.assertEqual(prediction.shape, target.shape)
        self.assertEqual(regret_loss.item(), 0.0)
        self.assertTrue(regret_loss.requires_grad)
        self.assertEqual(model.denoising_regret_stats['active'].item(), 0.0)


if __name__ == '__main__':
    unittest.main()
