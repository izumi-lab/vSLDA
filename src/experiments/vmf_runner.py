from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from src.core.artifacts import (
    METADATA_FILENAME,
    ExperimentAxes,
    VmfArtifactMetadata,
    save_json,
    save_vmf_metadata,
)
from src.core.paths import (
    build_vmf_archive_dir,
    build_vmf_condition_id,
    write_vmf_latest_pointer,
)
from src.core.result_identity import build_execution_id
from src.experiments.config import resolve_targets
from src.experiments.job_planning import CategoryJob, resolve_algorithm_variant
from src.models import ModelRunRequest, run_model_request
from src.utils.encoder_profiles import encoder_model_alias
from src.utils.random import DEFAULT_RANDOM_SEED


def _encoder_attr(encoder: object, name: str, default=None):
    return getattr(encoder, name, default)


def _train_attr(train: object, name: str, default=None):
    return getattr(train, name, default)


def _embedding_variant_from_encoder(encoder: object) -> str | None:
    if hasattr(encoder, "embedding_variant"):
        return str(getattr(encoder, "embedding_variant"))
    return None


@dataclass(frozen=True)
class VmfRunOptions:
    targets: list[str] | None
    train_csvs: list[str]
    test_csvs: list[str]
    num_iterations: int
    alpha: float | Sequence[float] | None
    kappa_default: float
    num_components: int
    gibbs_sweeps: int
    num_samples: int
    estimate_alpha: bool
    alpha_update_every: int
    alpha_max_iter: int
    alpha_tol: float
    alpha_min_value: float
    repair_empty_topics: bool
    min_topic_count_for_repair: int
    avg_log_likelihood_every: int
    invariant_check_every: int
    algorithm_variant: str
    encoder_name: str
    encoder_device: str
    encoder_prefix: str | None
    encoder_backend: str
    encoder_pooling: str | None
    encoder_prompt: str | None
    encoder_prompt_name: str | None
    encoder_encode_batch_size: int | None
    encoder_model_kwargs: dict[str, object]
    encoder_tokenizer_kwargs: dict[str, object]
    encoder_normalize_embeddings: bool | None
    encoder_truncate_dim: int | None
    encoder_strip_terminal_normalize: bool
    encoder_pre_normalize_transform: str
    encoder_whitening_eps: float
    output_dir: Path
    logger: object
    delimiter: str | None
    language: str
    segmenter: str
    tokenizer: str
    text_column: str
    target_column: str | None
    ja_replace_num: bool
    ja_stopwords_path: str | None
    ja_dicdir: str | None
    ja_require_unidic: bool
    soft_temperature: float
    data_run: str
    condition_id: str
    condition_fingerprint: str
    started_at: str
    execution_id: str

    def to_request_options(self) -> dict[str, object]:
        payload = asdict(self)
        payload["output_dir"] = self.output_dir
        payload["logger"] = self.logger
        return payload


@dataclass(frozen=True)
class VmfRunExecution:
    axes: ExperimentAxes
    started_at: str
    execution_id: str
    condition_id: str
    condition_fingerprint: str
    artifacts: dict[str, Path]


def build_experiment_axes(job: CategoryJob) -> ExperimentAxes:
    cfg = job.config
    return ExperimentAxes(
        dataset=cfg.dataset.name,
        model_family="vmf_sentence_lda",
        algorithm_variant=resolve_algorithm_variant(
            num_components=cfg.train.num_components,
            estimate_alpha=cfg.train.estimate_alpha,
            alpha_update_every=cfg.train.alpha_update_every,
        ),
        encoder_model=cfg.encoder.model_name,
        embedding_preprocess_variant=cfg.encoder.pre_normalize_transform,
        num_topics=int(job.num_topics),
        iteration=int(job.iteration),
        category=job.category,
        data_run=job.data_run_name,
        embedding_variant=(
            _embedding_variant_from_encoder(cfg.encoder)
            or encoder_model_alias(
                str(_encoder_attr(cfg.encoder, "model_name", "encoder"))
            )
        ),
    )


def build_vmf_condition_payload(
    job: CategoryJob,
    *,
    algorithm_variant: str,
) -> dict[str, object]:
    cfg = job.config
    return {
        "dataset": cfg.dataset.name,
        "data_run": job.data_run_name,
        "train_csvs": [str(path) for path in job.train_csvs],
        "test_csvs": [str(path) for path in job.test_csvs],
        "fiscal_years": (
            None
            if job.fiscal_years is None
            else [int(year) for year in job.fiscal_years]
        ),
        "iteration": int(job.iteration),
        "num_topics": int(job.num_topics),
        "num_components": int(cfg.train.num_components),
        "category": job.category,
        "algorithm_variant": algorithm_variant,
        "encoder_model": cfg.encoder.model_name,
        "encoder_backend": _encoder_attr(cfg.encoder, "backend", "auto"),
        "encoder_pooling": _encoder_attr(cfg.encoder, "pooling"),
        "encoder_prefix": _encoder_attr(cfg.encoder, "encode_prefix"),
        "encoder_prompt": _encoder_attr(cfg.encoder, "encode_prompt"),
        "encoder_prompt_name": _encoder_attr(cfg.encoder, "encode_prompt_name"),
        "encoder_model_kwargs": _encoder_attr(cfg.encoder, "model_kwargs", {}),
        "encoder_tokenizer_kwargs": _encoder_attr(cfg.encoder, "tokenizer_kwargs", {}),
        "encoder_normalize_embeddings": _encoder_attr(
            cfg.encoder, "normalize_embeddings"
        ),
        "encoder_truncate_dim": _encoder_attr(cfg.encoder, "truncate_dim"),
        "encoder_strip_terminal_normalize": _encoder_attr(
            cfg.encoder, "strip_terminal_normalize", True
        ),
        "embedding_preprocess_variant": cfg.encoder.pre_normalize_transform,
        "language": cfg.preprocess.language,
        "delimiter": cfg.preprocess.delimiter,
        "segmenter": cfg.preprocess.segmenter,
        "tokenizer": cfg.preprocess.tokenizer,
        "text_column": cfg.preprocess.text_column,
        "target_column": cfg.preprocess.target_column,
        "ja_replace_num": cfg.preprocess.ja_replace_num,
        "ja_stopwords_path": cfg.preprocess.ja_stopwords_path,
        "ja_dicdir": cfg.preprocess.ja_dicdir,
        "ja_require_unidic": cfg.preprocess.ja_require_unidic,
        "soft_temperature": float(job.vmf_soft_temp),
        "alpha_min_value": float(_train_attr(cfg.train, "alpha_min_value", 1e-3)),
        "repair_empty_topics": bool(
            _train_attr(cfg.train, "repair_empty_topics", True)
        ),
        "min_topic_count_for_repair": int(
            _train_attr(cfg.train, "min_topic_count_for_repair", 1)
        ),
    }


def _encoder_config_payload(job: CategoryJob) -> dict[str, object]:
    encoder = job.config.encoder
    return {
        "model_name": _encoder_attr(encoder, "model_name"),
        "device": _encoder_attr(encoder, "device", "cuda"),
        "backend": _encoder_attr(encoder, "backend", "auto"),
        "pooling": _encoder_attr(encoder, "pooling"),
        "encode_prefix": _encoder_attr(encoder, "encode_prefix"),
        "encode_prompt": _encoder_attr(encoder, "encode_prompt"),
        "encode_prompt_name": _encoder_attr(encoder, "encode_prompt_name"),
        "encode_batch_size": _encoder_attr(encoder, "encode_batch_size"),
        "model_kwargs": _encoder_attr(encoder, "model_kwargs", {}),
        "tokenizer_kwargs": _encoder_attr(encoder, "tokenizer_kwargs", {}),
        "normalize_embeddings": _encoder_attr(encoder, "normalize_embeddings"),
        "truncate_dim": _encoder_attr(encoder, "truncate_dim"),
        "strip_terminal_normalize": _encoder_attr(
            encoder, "strip_terminal_normalize", True
        ),
        "embedding_variant": (
            _embedding_variant_from_encoder(encoder)
            or encoder_model_alias(str(_encoder_attr(encoder, "model_name", "encoder")))
        ),
        "pre_normalize_transform": _encoder_attr(
            encoder, "pre_normalize_transform", "none"
        ),
        "whitening_eps": _encoder_attr(encoder, "whitening_eps", 1e-5),
    }


def build_vmf_run_options(
    *,
    job: CategoryJob,
    cfg=None,
    axes: ExperimentAxes,
    logger: object,
    vmf_out_dir: Path,
    resolved_targets: Sequence[str] | None,
    vmf_condition_id: str,
    vmf_condition_fingerprint: str,
    started_at: str,
    execution_id: str,
) -> VmfRunOptions:
    cfg = job.config if cfg is None else cfg
    return VmfRunOptions(
        targets=None if resolved_targets is None else list(resolved_targets),
        train_csvs=[str(path) for path in job.train_csvs],
        test_csvs=[str(path) for path in job.test_csvs],
        num_iterations=cfg.train.num_iterations,
        alpha=cfg.train.alpha,
        kappa_default=cfg.train.kappa_default,
        num_components=cfg.train.num_components,
        gibbs_sweeps=cfg.train.gibbs_sweeps,
        num_samples=cfg.train.num_samples,
        estimate_alpha=cfg.train.estimate_alpha,
        alpha_update_every=cfg.train.alpha_update_every,
        alpha_max_iter=cfg.train.alpha_max_iter,
        alpha_tol=cfg.train.alpha_tol,
        alpha_min_value=_train_attr(cfg.train, "alpha_min_value", 1e-3),
        repair_empty_topics=_train_attr(cfg.train, "repair_empty_topics", True),
        min_topic_count_for_repair=_train_attr(
            cfg.train, "min_topic_count_for_repair", 1
        ),
        avg_log_likelihood_every=cfg.train.avg_log_likelihood_every,
        invariant_check_every=cfg.train.invariant_check_every,
        algorithm_variant=axes.algorithm_variant,
        encoder_name=cfg.encoder.model_name,
        encoder_device=cfg.encoder.device,
        encoder_prefix=_encoder_attr(cfg.encoder, "encode_prefix"),
        encoder_backend=_encoder_attr(cfg.encoder, "backend", "auto"),
        encoder_pooling=_encoder_attr(cfg.encoder, "pooling"),
        encoder_prompt=_encoder_attr(cfg.encoder, "encode_prompt"),
        encoder_prompt_name=_encoder_attr(cfg.encoder, "encode_prompt_name"),
        encoder_encode_batch_size=_encoder_attr(cfg.encoder, "encode_batch_size"),
        encoder_model_kwargs=_encoder_attr(cfg.encoder, "model_kwargs", {}),
        encoder_tokenizer_kwargs=_encoder_attr(cfg.encoder, "tokenizer_kwargs", {}),
        encoder_normalize_embeddings=_encoder_attr(cfg.encoder, "normalize_embeddings"),
        encoder_truncate_dim=_encoder_attr(cfg.encoder, "truncate_dim"),
        encoder_strip_terminal_normalize=_encoder_attr(
            cfg.encoder, "strip_terminal_normalize", True
        ),
        encoder_pre_normalize_transform=cfg.encoder.pre_normalize_transform,
        encoder_whitening_eps=cfg.encoder.whitening_eps,
        output_dir=vmf_out_dir,
        logger=logger,
        delimiter=cfg.preprocess.delimiter,
        language=cfg.preprocess.language,
        segmenter=cfg.preprocess.segmenter,
        tokenizer=cfg.preprocess.tokenizer,
        text_column=cfg.preprocess.text_column,
        target_column=cfg.preprocess.target_column,
        ja_replace_num=cfg.preprocess.ja_replace_num,
        ja_stopwords_path=cfg.preprocess.ja_stopwords_path,
        ja_dicdir=cfg.preprocess.ja_dicdir,
        ja_require_unidic=cfg.preprocess.ja_require_unidic,
        soft_temperature=job.vmf_soft_temp,
        data_run=job.data_run_name,
        condition_id=vmf_condition_id,
        condition_fingerprint=vmf_condition_fingerprint,
        started_at=started_at,
        execution_id=execution_id,
    )


def run_vmf_job(*, job: CategoryJob, logger: object) -> VmfRunExecution:
    cfg = job.config
    axes = build_experiment_axes(job)
    vmf_condition_payload = build_vmf_condition_payload(
        job,
        algorithm_variant=axes.algorithm_variant,
    )
    vmf_condition_id, vmf_condition_fingerprint = build_vmf_condition_id(
        iteration=job.iteration,
        num_topics=job.num_topics,
        category=job.category,
        fingerprint_payload=vmf_condition_payload,
    )
    started_at = datetime.now(UTC).isoformat()
    execution_id = build_execution_id(prefix="vmf", started_at=started_at)
    resolved_targets = resolve_targets(
        cfg.dataset,
        cfg.preprocess,
        job.category,
        job.targets,
    )
    vmf_out_dir = build_vmf_archive_dir(
        iteration=job.iteration,
        num_topics=job.num_topics,
        num_components=cfg.train.num_components,
        embedding_variant=_embedding_variant_from_encoder(cfg.encoder),
        category=job.category,
        run_name=job.data_run_name,
        started_at=started_at,
        execution_id=execution_id,
        dataset_root=cfg.output_root,
    )
    vmf_options = build_vmf_run_options(
        job=job,
        axes=axes,
        logger=logger,
        vmf_out_dir=vmf_out_dir,
        resolved_targets=resolved_targets,
        vmf_condition_id=vmf_condition_id,
        vmf_condition_fingerprint=vmf_condition_fingerprint,
        started_at=started_at,
        execution_id=execution_id,
    )
    vmf_request = ModelRunRequest(
        name="vmf_sentence_lda",
        category=job.category,
        dataset=cfg.dataset.name,
        num_topics=job.num_topics,
        iteration=job.iteration,
        options=vmf_options.to_request_options(),
    )
    artifacts = run_model_request(vmf_request).as_dict()
    metadata = VmfArtifactMetadata(
        axes=axes,
        condition_id=vmf_condition_id,
        condition_fingerprint=vmf_condition_fingerprint,
        started_at=started_at,
        execution_id=execution_id,
        language=vmf_options.language,
        delimiter=vmf_options.delimiter,
        segmenter=vmf_options.segmenter,
        tokenizer=vmf_options.tokenizer,
        text_column=vmf_options.text_column,
        target_column=vmf_options.target_column,
        has_labels=cfg.preprocess.has_labels,
        ja_replace_num=vmf_options.ja_replace_num,
        ja_stopwords_path=cfg.preprocess.ja_stopwords_path,
        ja_dicdir=vmf_options.ja_dicdir,
        ja_require_unidic=vmf_options.ja_require_unidic,
        train_csvs=tuple(vmf_options.train_csvs),
        test_csvs=tuple(vmf_options.test_csvs),
        fiscal_years=job.fiscal_years,
        num_components=int(cfg.train.num_components),
        encoder_config=_encoder_config_payload(job),
    )
    metadata_path = vmf_out_dir / METADATA_FILENAME
    save_json(
        {
            **vmf_condition_payload,
            "model_name": "vmf_sentence_lda",
            "seed": (
                int(job.seed)
                if job.seed is not None
                else (
                    int(DEFAULT_RANDOM_SEED)
                    if job.seed_base is None
                    else int(job.seed_base)
                )
                + int(job.iteration)
            ),
            "condition_id": vmf_condition_id,
            "condition_fingerprint": vmf_condition_fingerprint,
            "started_at": started_at,
            "execution_id": execution_id,
        },
        vmf_out_dir / "config.json",
    )
    save_vmf_metadata(metadata, metadata_path)
    artifacts["metadata"] = metadata_path
    serialized_artifacts: dict[str, str] = {}
    for name, path in sorted(artifacts.items()):
        try:
            serialized_artifacts[name] = path.relative_to(vmf_out_dir).as_posix()
        except ValueError:
            serialized_artifacts[name] = str(path)
    write_vmf_latest_pointer(
        dataset=cfg.dataset.name,
        data_run=job.data_run_name,
        category=job.category,
        iteration=job.iteration,
        num_topics=job.num_topics,
        num_components=cfg.train.num_components,
        archive_dir=vmf_out_dir,
        started_at=started_at,
        execution_id=execution_id,
        condition_fingerprint=vmf_condition_fingerprint,
        artifacts=serialized_artifacts,
        embedding_variant=_embedding_variant_from_encoder(cfg.encoder),
        encoder_config=_encoder_config_payload(job),
        dataset_root=cfg.output_root,
    )
    return VmfRunExecution(
        axes=axes,
        started_at=started_at,
        execution_id=execution_id,
        condition_id=vmf_condition_id,
        condition_fingerprint=vmf_condition_fingerprint,
        artifacts=artifacts,
    )
