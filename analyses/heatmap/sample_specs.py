from __future__ import annotations

from dataclasses import asdict, dataclass

from analyses.t_SNE.imagenet_utils import build_class_id_signature


@dataclass(frozen=True)
class HeatmapSampleSpec:
    sample_id: int
    class_id: int
    class_name: str
    sample_index: int


def build_heatmap_sample_specs(
    class_ids: list[int],
    class_names: list[str],
    samples_per_class: int,
) -> list[HeatmapSampleSpec]:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")

    sample_specs = []
    sample_id = 0
    for class_id in class_ids:
        class_name = class_names[class_id]
        for sample_index in range(samples_per_class):
            sample_specs.append(
                HeatmapSampleSpec(
                    sample_id=sample_id,
                    class_id=class_id,
                    class_name=class_name,
                    sample_index=sample_index,
                )
            )
            sample_id += 1
    return sample_specs


def serialize_heatmap_sample_specs(sample_specs: list[HeatmapSampleSpec]) -> list[dict]:
    return [asdict(sample_spec) for sample_spec in sample_specs]


def deserialize_heatmap_sample_specs(sample_specs: list[dict]) -> list[HeatmapSampleSpec]:
    return [HeatmapSampleSpec(**sample_spec) for sample_spec in sample_specs]


def build_heatmap_output_stem(
    class_ids: list[int],
    seed: int,
    samples_per_class: int,
    analysis_every: int,
) -> str:
    class_signature = build_class_id_signature(class_ids)
    return (
        f"repa_dyna_heatmap_seed{seed}"
        f"_cls{len(class_ids)}_{class_signature}"
        f"_spc{samples_per_class}_every{analysis_every}"
    )
