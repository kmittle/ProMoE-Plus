"""Count image-wise expert computation through the equivalent Linear backend."""

from models.models_ProMoE_TC_image_shared import ImageAttentionPooling, ImageWiseSharedExpertPool


def count_pool_attention(module, inputs, output):
    """QK and attention-value MACs; child Linear hooks count K/V and output projections."""
    batch, tokens, channels = inputs[0].shape
    module.total_ops += 2 * batch * tokens * channels


def count_image_router(module, inputs, output):
    """Prototype dot-product MACs; all FFN/pooling projections are counted separately."""
    batch = inputs[0].shape[0]
    module.total_ops += batch * module.hidden_size * module.num_image_experts


IMAGE_EXPERT_CUSTOM_OPS = {
    ImageAttentionPooling: count_pool_attention,
    ImageWiseSharedExpertPool: count_image_router,
}


def find_image_expert_pools(moe_module):
    """Discover replacement-shared and additive image pools by their module paths."""
    return [(name, module) for name, module in moe_module.named_modules()
            if isinstance(module, ImageWiseSharedExpertPool)]


def use_loop_backend(model):
    """Switch only image pools; return their previous backends for restoration."""
    previous = []
    for module in model.modules():
        if isinstance(module, ImageWiseSharedExpertPool):
            previous.append((module, module.expert_backend))
            module.set_expert_backend("loop")
    return previous


def restore_backends(previous):
    for module, backend in previous:
        module.set_expert_backend(backend)
