# Analyses Overview

`analyses/` contains runnable analysis entry scripts for the ProMoE-Plus project.
Each Python entrypoint in this directory has a matching Markdown guide with the same basename.

## Entry Scripts

- [`run_tokenwise_tsne.py`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_tokenwise_tsne.py): token-wise routing t-SNE during sampling. See [`run_tokenwise_tsne.md`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_tokenwise_tsne.md).
- [`run_samplewise_pooled_tsne.py`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_samplewise_pooled_tsne.py): sample-wise t-SNE based on pooled block token features. See [`run_samplewise_pooled_tsne.md`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_samplewise_pooled_tsne.md).
- [`run_imagewise_tsne.py`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_imagewise_tsne.py): image-wise t-SNE based on pretrained image encoder features from generated PNGs. See [`run_imagewise_tsne.md`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_imagewise_tsne.md).
- [`run_repa_dyna_heatmap.py`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_repa_dyna_heatmap.py): RGB heatmap visualization of raw dynamic REPA token weights predicted by the `MLP + Sigmoid` token-weight head. See [`run_repa_dyna_heatmap.md`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_repa_dyna_heatmap.md).

## Shared Modules

Reusable helpers for these entry scripts live in [`analyses/t_SNE/`](/mnt/miah204/bycao/ProMoE-Plus/analyses/t_SNE/), including checkpoint and YAML resolution, ImageNet class utilities, feature capture, partial-result merging, encoder loading, and SVG plotting.
Dynamic REPA heatmap helpers live in [`analyses/heatmap/`](/mnt/miah204/bycao/ProMoE-Plus/analyses/heatmap/), including output-path utilities, deterministic sample specifications, token-weight capture, sampling/merging helpers, and SVG overlay plotting.
