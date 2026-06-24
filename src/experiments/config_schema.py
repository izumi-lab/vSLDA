from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from src.baselines.params import BaselineParams
from src.utils.random import DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    train_csv: Path
    test_csv: Path
    categories: Dict[str, Optional[Sequence[str]]]
    by_fy_root: Optional[Path] = None
    fiscal_years: Optional[List[int]] = None
    fiscal_year_mode: str = "concat_years"


@dataclass(frozen=True)
class TrainConfig:
    num_topics: List[int]
    num_iterations: int
    alpha: float | Sequence[float] | None
    kappa_default: float = 10.0
    num_components: int = 1
    gibbs_sweeps: int = 1
    num_samples: int = 1
    estimate_alpha: bool = True
    alpha_update_every: int = 1
    alpha_max_iter: int = 100
    alpha_tol: float = 1e-5
    alpha_min_value: float = 1e-3
    repair_empty_topics: bool = True
    min_topic_count_for_repair: int = 1
    avg_log_likelihood_every: int = 1
    invariant_check_every: int = 1


@dataclass(frozen=True)
class EncoderConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    device: str = "cuda"
    encode_prefix: Optional[str] = None
    backend: str = "auto"
    pooling: Optional[str] = None
    encode_prompt: Optional[str] = None
    encode_prompt_name: Optional[str] = None
    encode_batch_size: Optional[int] = None
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    tokenizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    normalize_embeddings: Optional[bool] = None
    truncate_dim: Optional[int] = None
    strip_terminal_normalize: bool = True
    embedding_variant: str = "default"
    pre_normalize_transform: str = "none"
    whitening_eps: float = 1e-5


@dataclass(frozen=True)
class PreprocessConfig:
    language: str = "english"
    delimiter: Optional[str] = " / "
    text_column: str = "data"
    target_column: Optional[str] = "target_str"
    has_labels: bool = True
    segmenter: str = "delimiter"
    tokenizer: str = "default"
    legacy_preprocessing: bool | None = None
    ja_replace_num: bool = True
    ja_stopwords_path: Optional[str] = None
    ja_dicdir: Optional[str] = None
    ja_require_unidic: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    iterations: List[int]


@dataclass(frozen=True)
class SelectionConfig:
    models: Optional[List[str]] = None
    categories: Optional[List[str]] = None
    topics: Optional[List[int]] = None
    iterations: Optional[List[int]] = None


@dataclass(frozen=True)
class PresetConfig:
    kind: str = "standard"
    purpose: str = "quantitative"


@dataclass(frozen=True)
class EvaluationConfig:
    tasks: Optional[List[str]] = None
    classifiers: Optional[List[str]] = None
    alignment_mode: str = "intersection"
    embedding_variants: Optional[List[str]] = None
    feature_resolve_mode: str = "all"


@dataclass(frozen=True)
class RuntimeConfig:
    seed_base: int | None = DEFAULT_RANDOM_SEED
    num_workers: int = 1


@dataclass(frozen=True)
class VmfInferenceConfig:
    soft_temperature: float = 1.0


@dataclass(frozen=True)
class VmfConfig:
    inference: VmfInferenceConfig = field(default_factory=VmfInferenceConfig)


@dataclass(frozen=True)
class BaselineConfig:
    name: str
    runner: str
    params: BaselineParams


@dataclass(frozen=True)
class ComparisonConfig:
    dataset: DatasetConfig
    train: TrainConfig
    encoder: EncoderConfig
    experiments: ExperimentConfig
    baselines: List[BaselineConfig]
    output_root: Path
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    selection: SelectionConfig = field(default_factory=SelectionConfig)
    preset: PresetConfig = field(default_factory=PresetConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    vmf: VmfConfig = field(default_factory=VmfConfig)
