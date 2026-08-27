"""Shared utilities for MoE block discovery and router signature detection."""

import inspect


def is_tc_compute_router(moe_module):
    """Check if the module's compute_router follows TC (Token-Choice) signature.

    TC models accept ``hidden_states`` and ``labels``; phase-aware TC routers
    may additionally accept an optional ``timestep``.
    EC models: compute_router(self, cond_hidden_states) -> 1 param (incompatible).
    """
    if not hasattr(moe_module, "compute_router"):
        return False
    sig = inspect.signature(moe_module.compute_router)
    params = [
        parameter
        for name, parameter in sig.parameters.items()
        if name != "self"
    ]
    if len(params) < 2:
        return False
    if [parameter.name for parameter in params[:2]] != [
        "hidden_states",
        "labels",
    ]:
        return False
    for parameter in params[2:]:
        if parameter.kind in {
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        }:
            continue
        if (
            parameter.name == "timestep"
            and parameter.default is not inspect.Parameter.empty
        ):
            continue
        return False
    return True


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
