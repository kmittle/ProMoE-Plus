import contextlib
import copy
import io
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from easydict import EasyDict

from models import models_ProMoE_TC as conference
from models import models_ProMoE_TC_global_center as global_center


MODEL_KWARGS = dict(input_size=8, patch_size=2, in_channels=4, hidden_size=48, depth=2, num_heads=4)
BLOCK_KWARGS = dict(
    num_routed_experts=4,
    hidden_size=16,
    moe_intermediate_size=8,
    shared_expert_intermediate_size=8,
    top_k=1,
    router_weight_mode="identity",
    routing_contrastive_lam=1,
    use_top_k_for_routing_contrastive=True,
    routing_contrastive_temperature=0.07,
)
WORLD_SIZE = 3


def moe_config():
    return EasyDict(
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


def build(module):
    torch.manual_seed(0)
    # The conference constructor prints its MoE layout.
    with contextlib.redirect_stdout(io.StringIO()):
        return module.DiT(MoE_config=copy.deepcopy(moe_config()), **MODEL_KWARGS)


def perturb(model):
    """Move every weight off its zero-initialised gates and output layer."""
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn(parameter.shape, generator=generator))
    return model


def dit_inputs():
    generator = torch.Generator().manual_seed(1234)
    x = torch.randn(6, 4, 8, 8, generator=generator)
    t = torch.rand(6, generator=generator)
    y = torch.tensor([3, 1000, 7, 7, 1000, 42])
    return x, t, y


def train_step(model):
    model.train()
    model.zero_grad(set_to_none=True)
    x, t, y = dit_inputs()
    torch.manual_seed(99)  # class-label dropout draws from the global RNG
    output = model(x, t, y)
    output.float().square().mean().backward()
    grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), grads


def rank_batches():
    """Unequal per-rank token counts; expert 3 has no token on rank 0."""
    generator = torch.Generator().manual_seed(11)
    tokens = []
    assignments = []
    for rank, size in enumerate((30, 45, 25)):
        tokens.append(torch.randn(size, 16, generator=generator))
        num_experts_on_rank = 3 if rank == 0 else 4
        assignments.append(torch.randint(0, num_experts_on_rank, (size, 1), generator=generator))
    return tokens, assignments


def block_state():
    torch.manual_seed(5)
    return global_center.SparseMoeBlock(**BLOCK_KWARGS).state_dict()


def _distributed_worker(rank, world_size, init_file, output_dir, state, tokens, assignments):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        block = global_center.SparseMoeBlock(**BLOCK_KWARGS)
        block.load_state_dict(state)
        local_tokens = tokens[rank].clone().requires_grad_(True)
        loss = block.compute_routing_contrastive_loss(local_tokens, assignments[rank], use_top_k=True)
        loss.backward()
        torch.save(
            {
                "loss": loss.detach(),
                "token_grad": local_tokens.grad,
                "center_grad": block.cluster_centers.grad,
            },
            os.path.join(output_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class SingleProcessTests(unittest.TestCase):
    def test_single_process_is_bit_identical_to_the_conference_model(self):
        reference = build(conference)
        model = build(global_center)
        reference_state = reference.state_dict()
        state = model.state_dict()
        self.assertEqual(list(state), list(reference_state))
        for name, value in reference_state.items():
            self.assertTrue(torch.equal(state[name], value), name)

        reference_output, reference_grads = train_step(perturb(reference))
        output, grads = train_step(perturb(model))
        self.assertTrue(torch.equal(output, reference_output))
        self.assertEqual(set(grads), set(reference_grads))
        for name, grad in reference_grads.items():
            self.assertTrue(torch.equal(grads[name], grad), name)

    def test_global_centers_with_one_rank_match_the_conference_loss(self):
        state = block_state()
        tokens, assignments = rank_batches()
        block = global_center.SparseMoeBlock(**BLOCK_KWARGS)
        block.load_state_dict(state)
        local_tokens = tokens[1]
        sums, counts = block._cluster_sums(local_tokens, assignments[1], use_top_k=True)
        loss = block._global_center_contrastive_loss(sums, sums.detach(), counts, world_size=1)
        expected = conference.SparseMoeBlock.compute_routing_contrastive_loss(
            block, local_tokens, assignments[1], use_top_k=True
        )
        torch.testing.assert_close(loss, expected)


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs torch.distributed with gloo")
class DistributedTests(unittest.TestCase):
    def test_ddp_average_equals_one_loss_on_the_global_batch(self):
        state = block_state()
        tokens, assignments = rank_batches()
        with tempfile.TemporaryDirectory() as output_dir:
            mp.spawn(
                _distributed_worker,
                args=(WORLD_SIZE, os.path.join(output_dir, "init"), output_dir, state, tokens, assignments),
                nprocs=WORLD_SIZE,
                join=True,
            )
            results = [torch.load(os.path.join(output_dir, f"rank{rank}.pt")) for rank in range(WORLD_SIZE)]

        # Reference: the conference loss computed once on the concatenated batch.
        reference = conference.SparseMoeBlock(**BLOCK_KWARGS)
        reference.load_state_dict(state)
        full_tokens = torch.cat(tokens).clone().requires_grad_(True)
        reference_loss = reference.compute_routing_contrastive_loss(
            full_tokens, torch.cat(assignments), use_top_k=True
        )
        reference_loss.backward()
        reference_token_grads = full_tokens.grad.split([batch.shape[0] for batch in tokens])

        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                torch.testing.assert_close(result["loss"], reference_loss.detach())
                # DDP divides every gradient by the world size.
                torch.testing.assert_close(result["token_grad"] / WORLD_SIZE, reference_token_grads[rank])
        averaged_center_grad = sum(result["center_grad"] for result in results) / WORLD_SIZE
        torch.testing.assert_close(averaged_center_grad, reference.cluster_centers.grad)


if __name__ == "__main__":
    unittest.main()
