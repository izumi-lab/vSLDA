from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.core.contracts import RunSpec

from .path_builders import (
    build_archive_result_dir,
    build_baseline_archive_dir,
    build_baseline_condition_id,
    build_baseline_dir,
    build_baseline_display_key,
    build_baseline_latest_dir,
    build_latest_result_dir,
    build_result_display_key,
    build_vmf_archive_dir,
    build_vmf_condition_id,
    build_vmf_display_key,
    build_vmf_experiment_dir,
    build_vmf_latest_dir,
)


@dataclass(frozen=True)
class ResultPathBuilder:
    """Centralized builder for experiment result paths."""

    root_dir: Path

    def run_dir(self, spec: RunSpec) -> Path:
        parts = [
            self.root_dir,
            spec.dataset_name,
            spec.model_name,
            f"{spec.num_topics}topic",
            f"seed{spec.seed}",
        ]
        if spec.category is not None:
            parts.append(spec.category)
        if spec.iteration is not None:
            parts.append(f"iter{spec.iteration}")
        return Path(*parts)

    def config_path(self, spec: RunSpec) -> Path:
        return self.run_dir(spec) / "config.json"

    def metadata_path(self, spec: RunSpec) -> Path:
        return self.run_dir(spec) / "metadata.json"

    def doc_topic_path(self, spec: RunSpec) -> Path:
        return self.run_dir(spec) / "doc_topic.npy"

    def sentence_topic_path(self, spec: RunSpec) -> Path:
        return self.run_dir(spec) / "sentence_topic.npy"

    def topic_word_path(self, spec: RunSpec) -> Path:
        return self.run_dir(spec) / "topic_word.npy"

    def topic_embeddings_path(self, spec: RunSpec) -> Path:
        return self.run_dir(spec) / "topic_embeddings.npy"

    def metrics_path(self, spec: RunSpec, metric_name: str) -> Path:
        return self.run_dir(spec) / "metrics" / f"{metric_name}.json"


from .path_pointers import (
    resolve_latest_result_dir,
    write_baseline_latest_pointer,
    write_latest_result_pointer,
    write_vmf_latest_pointer,
)
from .path_resolution import (
    build_baseline_doc_topic_path,
    build_vmf_doc_topic_path,
    resolve_analysis_result_dir,
    resolve_baseline_condition_dir,
    resolve_cross_model_pair_diagnostics_dir,
    resolve_sentence_topic_inspection_dir,
    resolve_topic_count_analysis_dir,
    resolve_vmf_experiment_dir,
)
from .paths_roots import (
    BASELINE_RESULTS_ROOT,
    CLASSIFICATION_RESULTS_ROOT,
    CONFIG_ROOT,
    DATA_RESOURCES_ROOT,
    DATA_ROOT,
    EXPERIMENT_RESULTS_ROOT,
    REPO_ROOT,
    RESULTS_ROOT,
    VISUALIZATION_RESULTS_ROOT,
    WIKIENTVEC_ROOT,
    resolve_project_path,
)
