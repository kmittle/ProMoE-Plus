import contextlib
import copy
import io
import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from easydict import EasyDict

from models import models_ProMoE_TC as conference
from models import models_ProMoE_TC_expert_contra as expert_contra
from models import models_ProMoE_TC_global_center as global_center
from models import models_ProMoE_TC_lsreg as lsreg
from models import models_ProMoE_TC_regcombo as regcombo


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
LS_STRENGTH = 0.05
LS_BETA = 0.9


def moe_config(**extra):
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
    config.update(extra)
    return config


def build(module, **extra):
    torch.manual_seed(0)
    # The conference-style constructors print their MoE layout.
    with contextlib.redirect_stdout(io.StringIO()):
        return module.DiT(MoE_config=copy.deepcopy(moe_config(**extra)), **MODEL_KWARGS)


def perturb(model):
    """Move every weight off its zero-initialised gates and output layer."""
    generator = torch.Generator().manual_seed(7)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(0.02 * torch.randn(parameter.shape, generator=generator))
    return model


def dit_inputs(seed):
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(6, 4, 8, 8, generator=generator)
    t = torch.rand(6, generator=generator)
    y = torch.tensor([3, 1000, 7, 7, 1000, 42])
    return x, t, y


def train_steps(model, num_steps=2):
    """Forward + backward on two different batches; the second sees the first's LS EMA."""
    model.train()
    outputs, grads = [], []
    for step in range(num_steps):
        model.zero_grad(set_to_none=True)
        x, t, y = dit_inputs(1234 + step)
        torch.manual_seed(99 + step)  # class-label dropout draws from the global RNG
        output = model(x, t, y)
        output.float().square().mean().backward()
        outputs.append(output.detach())
        grads.append({
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        })
    return outputs, grads


class EquivalenceTests(unittest.TestCase):
    """Each knob, alone, must reproduce the single-module model it was copied from."""

    def assert_same_run(self, reference, model):
        reference_state = reference.state_dict()
        state = model.state_dict()
        self.assertEqual(list(state), list(reference_state))
        for name, value in reference_state.items():
            self.assertTrue(torch.equal(state[name], value), name)
        reference_outputs, reference_grads = train_steps(perturb(reference))
        outputs, grads = train_steps(perturb(model))
        for step, (output, reference_output) in enumerate(zip(outputs, reference_outputs)):
            self.assertTrue(torch.equal(output, reference_output), f"output, step {step}")
        for step, (step_grads, step_reference) in enumerate(zip(grads, reference_grads)):
            self.assertEqual(set(step_grads), set(step_reference))
            for name, grad in step_reference.items():
                self.assertTrue(torch.equal(step_grads[name], grad), f"{name}, step {step}")
        return grads

    def test_all_knobs_off_is_bit_identical_to_the_conference_model(self):
        self.assert_same_run(build(conference), build(regcombo))

    def test_global_center_on_one_process_is_the_conference_model(self):
        self.assert_same_run(build(global_center), build(regcombo, global_center=True))

    def test_local_ls_reg_matches_the_lsreg_diag_branch(self):
        reference = build(lsreg, ls_apply="diag", ls_diag_sign=1.0,
                          ls_diag_strength=LS_STRENGTH, ls_ema_beta=LS_BETA)
        model = build(regcombo, ls_diag_strength=LS_STRENGTH, ls_ema_beta=LS_BETA)
        self.assert_same_run(reference, model)
        reference_blocks = [b.mlp for b in reference.blocks if b.use_moe]
        for block, reference_block in zip(model.moe_blocks(), reference_blocks):
            self.assertTrue(torch.equal(block.ls_load_ema, reference_block.ls_load_ema))
            self.assertTrue(torch.equal(block.last_mean_eps, reference_block.last_mean_eps))

    def test_param_repulsion_matches_expert_contra(self):
        knobs = dict(expert_contrastive_mode="param", expert_contrastive_lam=0.5,
                     expert_contrastive_temperature=5.0, expert_contrastive_blocks=[1])
        grads = self.assert_same_run(build(expert_contra, **knobs), build(regcombo, **knobs))
        _, plain_grads = train_steps(perturb(build(regcombo)))
        name = "blocks.1.mlp.experts.0.up_proj.weight"
        self.assertFalse(torch.equal(grads[0][name], plain_grads[0][name]), "the term must be alive here")
        block = build(regcombo, **knobs).blocks[1].mlp
        self.assertTrue(block.compute_expert_contrastive)

    def test_output_repulsion_matches_expert_contra(self):
        knobs = dict(expert_contrastive_mode="output", expert_contrastive_lam=0.5,
                     expert_contrastive_temperature=0.5, expert_contrastive_blocks=[1])
        grads = self.assert_same_run(build(expert_contra, **knobs), build(regcombo, **knobs))
        _, plain_grads = train_steps(perturb(build(regcombo)))
        name = "blocks.1.mlp.experts.0.up_proj.weight"
        self.assertFalse(torch.equal(grads[0][name], plain_grads[0][name]), "the term must be alive here")

    def test_combination_records_diagnostics(self):
        model = build(regcombo, global_center=True, ls_diag_strength=LS_STRENGTH,
                      expert_contrastive_mode="output", expert_contrastive_lam=0.5,
                      expert_contrastive_temperature=2.0, expert_contrastive_blocks=[1])
        train_steps(perturb(model), num_steps=1)
        block = model.moe_blocks()[0]
        for name in ("last_mean_eps", "last_load_hist", "last_expert_loss",
                     "last_expert_dist_min", "last_expert_dist_mean", "last_expert_norm_mean"):
            self.assertTrue(torch.is_tensor(getattr(block, name)), name)
        self.assertGreater(block.last_expert_active, 1)

    def test_train_py_stats_match_the_reduced_names(self):
        import train  # imported here: it pulls in every model family

        self.assertIn("ProMoE_TC_B_regcombo", train.model_dict)
        self.assertIn("ProMoE_TC_B_regcombo", train.REGCOMBO_MODELS)
        model = build(regcombo, global_center=True, ls_diag_strength=LS_STRENGTH,
                      expert_contrastive_mode="param", expert_contrastive_lam=0.5,
                      expert_contrastive_temperature=0.7, expert_contrastive_blocks=[1])
        train_steps(perturb(model), num_steps=1)
        stats = train._collect_regcombo_stats(model)
        # The cross-rank reduction is skipped unless every name is present.
        self.assertEqual(set(stats), set(train.REGCOMBO_STAT_NAMES))
        for name, value in stats.items():
            self.assertEqual(value.numel(), 1, name)
            self.assertTrue(torch.isfinite(value), name)
        self.assertGreater(float(stats["regcombo_ls_offset"]), 0.0)
        self.assertEqual(float(stats["regcombo_expert_active"]), 4.0)

    def test_global_output_on_one_process_matches_local_output(self):
        block = regcombo.SparseMoeBlock(**BLOCK_KWARGS, expert_contrastive_lam=0.5,
                                        expert_contrastive_temperature=4.0)
        tokens, assignments = rank_batches()
        outputs, ids = tokens[1], assignments[1][:, 0]
        local = {e: outputs[ids == e].mean(dim=0) for e in range(4) if (ids == e).any()}
        sums = {e: outputs[ids == e].float().sum(dim=0) for e in local}
        counts = {e: int((ids == e).sum()) for e in local}
        expected = block._expert_contrastive_output(local)
        got = block._expert_contrastive_output_global(sums, counts, 16, outputs.device)
        torch.testing.assert_close(got, expected)


class ConfigTests(unittest.TestCase):
    def test_misspelt_knob_is_refused(self):
        with self.assertRaisesRegex(ValueError, "unknown MoE_config keys"):
            build(regcombo, ls_diag_strenght=0.05)

    def test_weight_without_blocks_is_refused(self):
        with self.assertRaisesRegex(ValueError, "expert_contrastive_blocks"):
            build(regcombo, expert_contrastive_mode="param", expert_contrastive_lam=0.5)

    def test_blocks_without_weight_are_refused(self):
        with self.assertRaisesRegex(ValueError, "expert_contrastive_lam is 0"):
            build(regcombo, expert_contrastive_blocks=[1])

    def test_dense_block_is_refused(self):
        with self.assertRaisesRegex(ValueError, "not a MoE block"):
            build(regcombo, expert_contrastive_mode="param", expert_contrastive_lam=0.5,
                  expert_contrastive_blocks=[0])

    def test_global_counts_need_ls_reg(self):
        with self.assertRaisesRegex(ValueError, "ls_count_scope"):
            build(regcombo, ls_count_scope="global")

    def test_global_pooling_needs_output_mode(self):
        with self.assertRaisesRegex(ValueError, "expert_output_scope"):
            build(regcombo, expert_contrastive_mode="param", expert_output_scope="global")


# --------------------------------------------------------------- distributed
def rank_batches():
    """Unequal per-rank token counts; expert 3 has no token on rank 0, rank 2 sees one expert."""
    generator = torch.Generator().manual_seed(11)
    tokens = []
    assignments = []
    for rank, size in enumerate((30, 45, 25)):
        tokens.append(torch.randn(size, 16, generator=generator))
        if rank == 2:
            assignments.append(torch.full((size, 1), 1, dtype=torch.long))
        else:
            num_experts_on_rank = 3 if rank == 0 else 4
            assignments.append(torch.randint(0, num_experts_on_rank, (size, 1), generator=generator))
    return tokens, assignments


def block_state():
    torch.manual_seed(5)
    return regcombo.SparseMoeBlock(**BLOCK_KWARGS).state_dict()


def _worker(rank, world_size, init_file, output_dir, case, state, tokens, assignments):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size)
    try:
        result = {}
        local_tokens = tokens[rank].clone().requires_grad_(True)
        if case == "global_center":
            block = regcombo.SparseMoeBlock(**BLOCK_KWARGS, global_center=True)
            block.load_state_dict(state)
            reference = global_center.SparseMoeBlock(**BLOCK_KWARGS)
            reference.load_state_dict(state)
            reference_tokens = tokens[rank].clone().requires_grad_(True)
            reference_loss = reference.compute_routing_contrastive_loss(
                reference_tokens, assignments[rank], use_top_k=True)
            reference_loss.backward()
            result.update(reference_loss=reference_loss.detach(), reference_token_grad=reference_tokens.grad,
                          reference_center_grad=reference.cluster_centers.grad)
        elif case == "ls_global":
            block = regcombo.SparseMoeBlock(**BLOCK_KWARGS, ls_diag_strength=LS_STRENGTH,
                                            ls_ema_beta=LS_BETA, ls_count_scope="global")
            block.load_state_dict(state)
        elif case == "gc_ls_local":
            block = regcombo.SparseMoeBlock(**BLOCK_KWARGS, global_center=True,
                                            ls_diag_strength=LS_STRENGTH, ls_ema_beta=LS_BETA)
            block.load_state_dict(state)
        elif case == "output_global":
            block = regcombo.SparseMoeBlock(**BLOCK_KWARGS, expert_contrastive_lam=0.5,
                                            expert_contrastive_temperature=4.0,
                                            expert_output_scope="global")
            block.load_state_dict(state)
            ids = assignments[rank][:, 0]
            sums = {e: local_tokens[ids == e].float().sum(dim=0) for e in range(4) if (ids == e).any()}
            counts = {e: int((ids == e).sum()) for e in sums}
            loss = block._expert_contrastive_output_global(sums, counts, 16, local_tokens.device)
            loss.backward()
            result.update(loss=loss.detach(), token_grad=local_tokens.grad)
            torch.save(result, os.path.join(output_dir, f"rank{rank}.pt"))
            return
        block.train()
        loss = block.compute_routing_contrastive_loss(local_tokens, assignments[rank], use_top_k=True)
        if loss.requires_grad:  # a rank with fewer than two local classes returns a constant 0
            loss.backward()
        result.update(loss=loss.detach(), token_grad=local_tokens.grad,
                      center_grad=block.cluster_centers.grad, ema=block.ls_load_ema.clone())
        torch.save(result, os.path.join(output_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def run_case(case):
    state = block_state()
    tokens, assignments = rank_batches()
    with tempfile.TemporaryDirectory() as output_dir:
        mp.spawn(
            _worker,
            args=(WORLD_SIZE, os.path.join(output_dir, "init"), output_dir, case, state, tokens, assignments),
            nprocs=WORLD_SIZE,
            join=True,
        )
        results = [torch.load(os.path.join(output_dir, f"rank{rank}.pt")) for rank in range(WORLD_SIZE)]
    return state, tokens, assignments, results


def counts_of(assignment):
    return torch.stack([(assignment[:, 0] == e).sum() for e in range(4)]).float()


def diag_offset(ema_valid):
    rel = ((ema_valid - ema_valid.mean()) / (ema_valid.mean() + 1e-6)).clamp(-1.0, 1.0)
    return LS_STRENGTH * rel


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "needs torch.distributed with gloo")
class DistributedTests(unittest.TestCase):
    def test_global_center_matches_the_global_center_model(self):
        _, _, _, results = run_case("global_center")
        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                self.assertTrue(torch.equal(result["loss"], result["reference_loss"]))
                self.assertTrue(torch.equal(result["token_grad"], result["reference_token_grad"]))
                self.assertTrue(torch.equal(result["center_grad"], result["reference_center_grad"]))

    def test_global_ls_counts_use_the_all_reduced_counts(self):
        state, tokens, assignments, results = run_case("ls_global")
        global_counts = sum(counts_of(a) for a in assignments)
        expected_ema = (1.0 - LS_BETA) * global_counts
        centers = state["cluster_centers"]
        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                ids = assignments[rank][:, 0]
                valid = [e for e in range(4) if (ids == e).any()]
                if len(valid) < 2:
                    # It still joined the collective (the run did not hang) and, like
                    # lsreg, returns 0 before touching the EMA.
                    self.assertEqual(float(result["loss"]), 0.0)
                    self.assertTrue(torch.equal(result["ema"], torch.zeros(4)))
                    continue
                torch.testing.assert_close(result["ema"], expected_ema)
                local_tokens = tokens[rank].clone().requires_grad_(True)
                means = torch.stack([local_tokens[ids == e].mean(dim=0) for e in valid])
                sim = (F.normalize(centers[valid], dim=1) @ F.normalize(means, dim=1).T).clamp(-1.0, 1.0)
                sim = sim + torch.diag(diag_offset(expected_ema[valid]))
                loss = F.cross_entropy(sim / 0.07, torch.arange(len(valid)))
                loss.backward()
                torch.testing.assert_close(result["loss"], loss.detach())
                torch.testing.assert_close(result["token_grad"], local_tokens.grad)

    def test_global_center_with_local_ls_offsets(self):
        state, tokens, assignments, results = run_case("gc_ls_local")
        full = torch.cat(tokens)
        full_ids = torch.cat(assignments)[:, 0]
        valid = [e for e in range(4) if (full_ids == e).any()]
        centers = state["cluster_centers"]
        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                ema = (1.0 - LS_BETA) * counts_of(assignments[rank])
                torch.testing.assert_close(result["ema"], ema)
                means = torch.stack([full[full_ids == e].mean(dim=0) for e in valid])
                sim = (F.normalize(centers[valid], dim=1) @ F.normalize(means, dim=1).T).clamp(-1.0, 1.0)
                sim = sim + torch.diag(diag_offset(ema[valid]))
                loss = F.cross_entropy(sim / 0.07, torch.arange(len(valid)))
                torch.testing.assert_close(result["loss"], loss)

    def test_global_output_pooling_equals_one_loss_on_the_global_batch(self):
        state, tokens, assignments, results = run_case("output_global")
        reference = regcombo.SparseMoeBlock(**BLOCK_KWARGS, expert_contrastive_lam=0.5,
                                            expert_contrastive_temperature=4.0)
        reference.load_state_dict(state)
        full_tokens = torch.cat(tokens).clone().requires_grad_(True)
        full_ids = torch.cat(assignments)[:, 0]
        pools = {e: full_tokens[full_ids == e].mean(dim=0) for e in range(4) if (full_ids == e).any()}
        reference_loss = reference._expert_contrastive_output(pools)
        reference_loss.backward()
        reference_grads = full_tokens.grad.split([batch.shape[0] for batch in tokens])
        self.assertGreater(float(reference_loss), 1e-4, "the term must be alive here")
        for rank, result in enumerate(results):
            with self.subTest(rank=rank):
                torch.testing.assert_close(result["loss"], reference_loss.detach())
                # DDP divides every gradient by the world size.
                torch.testing.assert_close(result["token_grad"] / WORLD_SIZE, reference_grads[rank])


if __name__ == "__main__":
    unittest.main()
