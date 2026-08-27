import unittest
from unittest import mock

import torch

from models.models_ProMoE_EC_batch_choice import DiT as BaseEC
from models.models_ProMoE_EC_batch_choice_proto_t import DiT as ProtoTEC
from models.models_ProMoE_TC import DiT as BaseTC
from models.models_ProMoE_TC_proto_t import DiT as ProtoTTC


class _AttrDict(dict):
    __getattr__ = dict.__getitem__


def _model_kwargs():
    return {
        "input_size": 8,
        "patch_size": 2,
        "in_channels": 4,
        "hidden_size": 32,
        "depth": 2,
        "num_heads": 4,
        "mlp_ratio": 2,
        "class_dropout_prob": 0.1,
        "num_classes": 1000,
        "learn_sigma": False,
        "MoE_config": _AttrDict(
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
            router_weight_mode="identity",
            routing_contrastive_lam=1.0,
            use_top_k_for_routing_contrastive=True,
            routing_contrastive_temperature=0.07,
            proto_t_update_mode="residual",
            proto_t_init_seed=1729,
        ),
    }


class ProtoTControlInitializationTest(unittest.TestCase):
    @staticmethod
    def _build_pair(base_class, proto_t_class):
        torch.manual_seed(17)
        base = base_class(**_model_kwargs()).eval()
        base_rng_state = torch.get_rng_state()
        torch.manual_seed(17)
        candidate = proto_t_class(**_model_kwargs()).eval()
        candidate_rng_state = torch.get_rng_state()
        return base, candidate, base_rng_state, candidate_rng_state

    def _assert_common_state_matches(self, base_class, proto_t_class):
        base, candidate, base_rng_state, candidate_rng_state = (
            self._build_pair(base_class, proto_t_class)
        )
        torch.testing.assert_close(
            candidate_rng_state,
            base_rng_state,
            rtol=0,
            atol=0,
        )
        base_state = base.state_dict()
        candidate_state = candidate.state_dict()
        extra_keys = set(candidate_state) - set(base_state)
        self.assertTrue(extra_keys)
        self.assertTrue(
            all(".prototype_mlp." in key for key in extra_keys),
            extra_keys,
        )
        for name, value in base_state.items():
            torch.testing.assert_close(
                candidate_state[name],
                value,
                rtol=0,
                atol=0,
            )

    def test_tc_common_parameters_keep_base_initialization(self):
        self._assert_common_state_matches(BaseTC, ProtoTTC)

    def test_ec_common_parameters_keep_base_initialization(self):
        self._assert_common_state_matches(BaseEC, ProtoTEC)

    def test_model_construction_does_not_seed_cuda(self):
        for model_class in (ProtoTTC, ProtoTEC):
            with self.subTest(model_class=model_class.__name__):
                torch.random.default_generator.manual_seed(17)
                with mock.patch.object(
                    torch.cuda,
                    "manual_seed_all",
                ) as cuda_manual_seed_all:
                    model_class(**_model_kwargs())
                cuda_manual_seed_all.assert_not_called()

    def test_tc_step_zero_routing_decisions_match_base(self):
        base, candidate, _, _ = self._build_pair(BaseTC, ProtoTTC)
        generator = torch.Generator().manual_seed(29)
        hidden_states = torch.randn(3, 16, 32, generator=generator)
        labels = torch.tensor([2, 7, 1000])
        timesteps = torch.tensor([100.0, 500.0, 900.0])
        t_emb = candidate.t_embedder(timesteps)
        prototype_t = candidate.blocks[1].mlp.prototype_mlp(
            candidate.blocks[1].mlp.cluster_centers,
            t_emb,
        )

        base_router = base.blocks[1].mlp.compute_router(
            hidden_states,
            labels,
        )
        candidate_router = candidate.blocks[1].mlp.compute_router(
            hidden_states,
            labels,
            prototype_t,
        )
        torch.testing.assert_close(
            candidate_router[0],
            base_router[0],
            rtol=1e-6,
            atol=1e-7,
        )
        torch.testing.assert_close(
            candidate_router[1],
            base_router[1],
            rtol=0,
            atol=0,
        )
        self.assertIsNone(base_router[2])
        self.assertIsNone(candidate_router[2])


if __name__ == "__main__":
    unittest.main()
