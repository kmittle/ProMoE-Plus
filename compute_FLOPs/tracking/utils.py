"""Shared utilities for MoE block discovery and router signature detection."""

import inspect


def is_tc_compute_router(moe_module):
    """Check if the module's compute_router follows TC (Token-Choice) signature.

    TC models: compute_router(self, hidden_states, labels) -> 2 params.
    EC models: compute_router(self, cond_hidden_states) -> 1 param (incompatible).
    """
    if not hasattr(moe_module, "compute_router"):
        return False
    sig = inspect.signature(moe_module.compute_router)
    params = [p for p in sig.parameters if p != "self"]
    return len(params) == 2


def find_moe_blocks(model):
    """Find all MoE SparseMoeBlock modules and return (block_index, module) pairs.

    block_index is the position within model.blocks (DiTBlock index).
    Only DiTBlocks with use_moe=True contain a SparseMoeBlock as their .mlp attribute.
    Skips Expert-Choice (EC) blocks whose compute_router has an incompatible signature.
    """
    moe_blocks = []
    if not hasattr(model, "blocks"):
        return moe_blocks
    for i, block in enumerate(model.blocks):
        if hasattr(block, "use_moe") and block.use_moe:
            moe_module = block.mlp  # SparseMoeBlock
            if is_tc_compute_router(moe_module):
                moe_blocks.append((i, moe_module))
    return moe_blocks
