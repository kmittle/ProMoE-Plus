# Analyses Overview

`analyses/` contains runnable analysis entry scripts for the ProMoE-Plus project.
Each Python entrypoint in this directory has a matching Markdown guide with the same basename.

## Entry Scripts

- [`run_tokenwise_tsne.py`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_tokenwise_tsne.py): token-wise routing t-SNE during sampling. See [`run_tokenwise_tsne.md`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_tokenwise_tsne.md).
- [`run_samplewise_pooled_tsne.py`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_samplewise_pooled_tsne.py): sample-wise t-SNE based on pooled block token features. See [`run_samplewise_pooled_tsne.md`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_samplewise_pooled_tsne.md).
- [`run_imagewise_tsne.py`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_imagewise_tsne.py): image-wise t-SNE based on pretrained image encoder features from generated PNGs. See [`run_imagewise_tsne.md`](/mnt/miah204/bycao/ProMoE-Plus/analyses/run_imagewise_tsne.md).

## Shared Modules

Reusable helpers for these entry scripts live in [`analyses/t_SNE/`](/mnt/miah204/bycao/ProMoE-Plus/analyses/t_SNE/), including checkpoint and YAML resolution, ImageNet class utilities, feature capture, partial-result merging, encoder loading, and SVG plotting.
