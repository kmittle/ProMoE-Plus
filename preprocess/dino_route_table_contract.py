"""Versioned metadata contracts for offline DINO routing tables."""


LEGACY_TABLE_VERSION = 1
LEGACY_TABLE_METHOD = (
    "class_mean_dinov2_patch_features_with_neighbor_margin_and_"
    "intra_class_variance"
)

CORRECTED_TABLE_VERSION = 2
CORRECTED_TABLE_METHOD = (
    "class_mean_dinov2_patch_features_with_neighbor_margin_and_"
    "intra_class_variance_correct_vae_decode"
)

SUPPORTED_TABLE_CONTRACTS = {
    LEGACY_TABLE_VERSION: LEGACY_TABLE_METHOD,
    CORRECTED_TABLE_VERSION: CORRECTED_TABLE_METHOD,
}
