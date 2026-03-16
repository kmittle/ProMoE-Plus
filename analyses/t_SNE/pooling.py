import torch


SUPPORTED_POOL_TYPES = ("mean", "max")


def pool_block_tokens(token_features: torch.Tensor, pool_type: str = "mean") -> torch.Tensor:
    if token_features.ndim != 3:
        raise ValueError(
            f"Expected token features with shape [batch, tokens, dim], got {token_features.shape}."
        )
    if pool_type == "mean":
        return token_features.mean(dim=1)
    if pool_type == "max":
        return token_features.amax(dim=1)
    raise ValueError(
        f"Unsupported pool_type='{pool_type}'. Supported values: {SUPPORTED_POOL_TYPES}."
    )
