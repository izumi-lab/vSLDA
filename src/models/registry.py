from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from src.baselines import BaselineRunRequest, run_baseline_request
from src.baselines.dataset_adapters import load_preprocessed_documents_with_indices
from src.baselines.runners import RUNNERS as BASELINE_RUNNERS
from src.core.artifacts import PREPROCESSING_SELECTION_FILENAME, save_json
from src.data.preprocessing import SelectedCorpus, select_modelable_documents
from src.models.contracts import ModelArtifacts, ModelRunnerSpec, ModelRunRequest
from src.models.vmf_artifacts import build_vmf_run_output_payload, save_vmf_run_outputs
from src.models.vmf_sentence_lda import VMFLDATrainer
from src.utils.encoder import SentenceEncoder
from src.utils.encoder_inputs import (
    fit_encoder_on_sentences,
    sentence_corpus_for_encoder,
)
from src.utils.evaluation import evaluate_model


@dataclass(frozen=True)
class VmfRequestOptions:
    train_csvs: Sequence[str | Path]
    test_csvs: Sequence[str | Path]
    targets: Sequence[str] | None
    delimiter: str | None
    language: str
    segmenter: str
    tokenizer: str
    ja_replace_num: bool
    ja_dicdir: str | None
    ja_require_unidic: bool
    text_column: str
    target_column: str | None
    output_dir: Path
    logger: Any
    encoder_name: str
    encoder_device: str
    encoder_prefix: str | None
    encoder_backend: str
    encoder_pooling: str | None
    encoder_prompt: str | None
    encoder_prompt_name: str | None
    encoder_encode_batch_size: int | None
    encoder_model_kwargs: dict[str, Any]
    encoder_tokenizer_kwargs: dict[str, Any]
    encoder_normalize_embeddings: bool | None
    encoder_truncate_dim: int | None
    encoder_strip_terminal_normalize: bool
    alpha: float | Sequence[float] | None
    kappa_default: float
    num_components: int
    encoder_pre_normalize_transform: str
    encoder_whitening_eps: float
    algorithm_variant: str | None
    num_iterations: int
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
    soft_temperature: float

    @classmethod
    def from_request_options(cls, options: dict[str, Any]) -> "VmfRequestOptions":
        return cls(
            train_csvs=options["train_csvs"],
            test_csvs=options["test_csvs"],
            targets=options.get("targets"),
            delimiter=options.get("delimiter", " / "),
            language=options.get("language", "english"),
            segmenter=options.get("segmenter", "delimiter"),
            tokenizer=options.get("tokenizer", "default"),
            ja_replace_num=bool(options.get("ja_replace_num", True)),
            ja_dicdir=options.get("ja_dicdir"),
            ja_require_unidic=bool(options.get("ja_require_unidic", True)),
            text_column=options.get("text_column", "data"),
            target_column=options.get("target_column", "target_str"),
            output_dir=Path(options["output_dir"]),
            logger=options["logger"],
            encoder_name=options["encoder_name"],
            encoder_device=options["encoder_device"],
            encoder_prefix=options.get("encoder_prefix"),
            encoder_backend=str(options.get("encoder_backend", "auto")),
            encoder_pooling=options.get("encoder_pooling"),
            encoder_prompt=options.get("encoder_prompt"),
            encoder_prompt_name=options.get("encoder_prompt_name"),
            encoder_encode_batch_size=options.get("encoder_encode_batch_size"),
            encoder_model_kwargs=dict(options.get("encoder_model_kwargs") or {}),
            encoder_tokenizer_kwargs=dict(
                options.get("encoder_tokenizer_kwargs") or {}
            ),
            encoder_normalize_embeddings=options.get("encoder_normalize_embeddings"),
            encoder_truncate_dim=options.get("encoder_truncate_dim"),
            encoder_strip_terminal_normalize=bool(
                options.get("encoder_strip_terminal_normalize", True)
            ),
            alpha=options.get("alpha"),
            kappa_default=options["kappa_default"],
            num_components=options.get("num_components", 1),
            encoder_pre_normalize_transform=options["encoder_pre_normalize_transform"],
            encoder_whitening_eps=options["encoder_whitening_eps"],
            algorithm_variant=options.get("algorithm_variant"),
            num_iterations=options["num_iterations"],
            gibbs_sweeps=options["gibbs_sweeps"],
            num_samples=options["num_samples"],
            estimate_alpha=options["estimate_alpha"],
            alpha_update_every=options["alpha_update_every"],
            alpha_max_iter=options["alpha_max_iter"],
            alpha_tol=options["alpha_tol"],
            alpha_min_value=float(options.get("alpha_min_value", 1e-3)),
            repair_empty_topics=bool(options.get("repair_empty_topics", True)),
            min_topic_count_for_repair=int(
                options.get("min_topic_count_for_repair", 1)
            ),
            avg_log_likelihood_every=options.get("avg_log_likelihood_every", 1),
            invariant_check_every=options.get("invariant_check_every", 1),
            soft_temperature=float(options.get("soft_temperature", 1.0)),
        )


def _load_preprocessed_corpus_many(
    csv_paths: Sequence[str | Path],
    *,
    target_filter: Sequence[str] | None,
    delimiter: str | None,
    language: str,
    segmenter: str,
    tokenizer: str,
    ja_replace_num: bool,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
    text_column: str,
    target_column: str | None,
) -> SelectedCorpus:
    documents, raw_indices = load_preprocessed_documents_with_indices(
        csv_paths=[str(path) for path in csv_paths],
        text_column=text_column,
        target_column=target_column,
        targets=target_filter,
        delimiter=delimiter,
        language=language,
        segmenter=segmenter,
        tokenizer=tokenizer,
        ja_replace_num=ja_replace_num,
        ja_stopwords_path=None,
        ja_dicdir=ja_dicdir,
        ja_require_unidic=ja_require_unidic,
    )
    return select_modelable_documents(documents, raw_doc_indices=raw_indices)


def _run_vmf_request(request: ModelRunRequest) -> ModelArtifacts:
    options = VmfRequestOptions.from_request_options(dict(request.options))

    train_selection = _load_preprocessed_corpus_many(
        options.train_csvs,
        target_filter=options.targets,
        delimiter=options.delimiter,
        language=options.language,
        segmenter=options.segmenter,
        tokenizer=options.tokenizer,
        ja_replace_num=options.ja_replace_num,
        ja_dicdir=options.ja_dicdir,
        ja_require_unidic=options.ja_require_unidic,
        text_column=options.text_column,
        target_column=options.target_column,
    )
    train_preprocessed = train_selection.documents
    test_selection = _load_preprocessed_corpus_many(
        options.test_csvs,
        target_filter=options.targets,
        delimiter=options.delimiter,
        language=options.language,
        segmenter=options.segmenter,
        tokenizer=options.tokenizer,
        ja_replace_num=options.ja_replace_num,
        ja_dicdir=options.ja_dicdir,
        ja_require_unidic=options.ja_require_unidic,
        text_column=options.text_column,
        target_column=options.target_column,
    )
    test_preprocessed = test_selection.documents
    encoder = SentenceEncoder(
        options.encoder_name,
        device=options.encoder_device,
        encode_prefix=options.encoder_prefix,
        backend=options.encoder_backend,
        pooling=options.encoder_pooling,
        encode_prompt=options.encoder_prompt,
        encode_prompt_name=options.encoder_prompt_name,
        encode_batch_size=options.encoder_encode_batch_size,
        model_kwargs=options.encoder_model_kwargs,
        tokenizer_kwargs=options.encoder_tokenizer_kwargs,
        normalize_embeddings=options.encoder_normalize_embeddings,
        truncate_dim=options.encoder_truncate_dim,
        strip_terminal_normalize=options.encoder_strip_terminal_normalize,
    )
    fit_encoder_on_sentences(encoder, train_preprocessed)
    corpus = sentence_corpus_for_encoder(train_preprocessed, encoder)
    test_corpus = sentence_corpus_for_encoder(test_preprocessed, encoder)
    trainer = VMFLDATrainer(
        corpus=corpus,
        encoder=encoder,
        num_topics=request.num_topics,
        alpha=options.alpha,
        kappa=options.kappa_default,
        num_components=options.num_components,
        pre_normalize_transform=options.encoder_pre_normalize_transform,
        whitening_eps=options.encoder_whitening_eps,
        algorithm_variant=options.algorithm_variant,
        save_path=options.output_dir,
        log=options.logger,
    )

    start = time.time()
    trainer.sample(
        options.num_iterations,
        num_sweeps=options.gibbs_sweeps,
        num_samples=options.num_samples,
        estimate_alpha=options.estimate_alpha,
        alpha_update_every=options.alpha_update_every,
        alpha_max_iter=options.alpha_max_iter,
        alpha_tol=options.alpha_tol,
        alpha_min_value=options.alpha_min_value,
        repair_empty_topics=options.repair_empty_topics,
        min_topic_count_for_repair=options.min_topic_count_for_repair,
        avg_log_likelihood_every=options.avg_log_likelihood_every,
        invariant_check_every=options.invariant_check_every,
    )
    elapsed = time.time() - start
    embedding_cache = trainer.build_embedding_cache_report()

    metrics = evaluate_model(trainer)
    theta_train = trainer.get_document_topic_distribution()
    counts_train = (
        trainer.topic_counts_per_doc.T
        if hasattr(trainer, "topic_counts_per_doc")
        else None
    )

    counts_test_start = time.perf_counter()
    options.logger.info("Running test corpus inference")
    test_inference = trainer.infer_corpus_topic_outputs(
        test_corpus,
        temperature=options.soft_temperature,
        include_counts=True,
        include_sentence_posteriors=True,
        include_document_posteriors=True,
    )
    counts_test_inference_sec = time.perf_counter() - counts_test_start
    counts_test = test_inference.counts
    if counts_test is None:
        raise RuntimeError("counts_test was not produced by infer_corpus_topic_outputs")
    theta_test = test_inference.document_posteriors
    if theta_test is None:
        raise RuntimeError("theta_test was not produced by infer_corpus_topic_outputs")

    sentence_topic_train_soft_start = time.perf_counter()
    options.logger.info("Running train corpus soft inference from cached embeddings")
    train_inference = trainer.infer_encoded_corpus_topic_outputs(
        trainer.encoded_corpus,
        temperature=options.soft_temperature,
        include_sentence_posteriors=True,
        include_document_posteriors=True,
    )
    sentence_topic_train_soft_sec = (
        time.perf_counter() - sentence_topic_train_soft_start
    )
    sentence_topic_train_soft = train_inference.sentence_posteriors
    if sentence_topic_train_soft is None:
        raise RuntimeError(
            "sentence_topic_train_soft was not produced by infer_corpus_topic_outputs"
        )
    theta_train_soft = train_inference.document_posteriors
    if theta_train_soft is None:
        raise RuntimeError(
            "theta_train_soft was not produced by infer_corpus_topic_outputs"
        )

    sentence_topic_test_soft = test_inference.sentence_posteriors
    if sentence_topic_test_soft is None:
        raise RuntimeError(
            "sentence_topic_test_soft was not produced by infer_corpus_topic_outputs"
        )
    sentence_topic_test_soft_sec = counts_test_inference_sec
    theta_test_soft = theta_test

    saved_outputs = save_vmf_run_outputs(
        build_vmf_run_output_payload(
            theta_train=theta_train,
            theta_test=theta_test,
            theta_train_soft=theta_train_soft,
            theta_test_soft=theta_test_soft,
            sentence_topic_train_soft=sentence_topic_train_soft,
            sentence_topic_test_soft=sentence_topic_test_soft,
            train_preprocessed=train_preprocessed,
            test_preprocessed=test_preprocessed,
            counts_train=counts_train,
            embedding_cache={
                "strategy": embedding_cache.strategy,
                "num_documents": embedding_cache.num_documents,
                "total_sentences": embedding_cache.total_sentences,
                "embedding_size": embedding_cache.embedding_size,
                "pre_normalize_transform": embedding_cache.pre_normalize_transform,
                "reused_for_training_iterations": embedding_cache.reused_for_training_iterations,
                "reused_for_avg_log_likelihood": embedding_cache.reused_for_avg_log_likelihood,
            },
            metrics={
                "category": request.category,
                "num_topics": request.num_topics,
                "algorithm_variant": options.algorithm_variant,
                "num_iterations": options.num_iterations,
                "gibbs_sweeps": options.gibbs_sweeps,
                "num_samples": options.num_samples,
                "alpha_min_value": options.alpha_min_value,
                "repair_empty_topics": options.repair_empty_topics,
                "min_topic_count_for_repair": options.min_topic_count_for_repair,
                "avg_log_likelihood_every": options.avg_log_likelihood_every,
                "invariant_check_every": options.invariant_check_every,
                "elapsed_sec": elapsed,
                "training_corpus_encoding_sec": trainer.training_corpus_encoding_sec,
                "encoder_encode_batch_size": options.encoder_encode_batch_size,
                "embedding_storage_dtype": np.dtype(
                    trainer.EMBEDDING_STORAGE_DTYPE
                ).name,
                "e_step_kernel_backend": trainer.e_step_kernel_backend,
                "m_step_statistics_kernel_backend": (
                    trainer.m_step_statistics_kernel_backend
                ),
                "avg_ll_kernel_backend": trainer.avg_ll_kernel_backend,
                "test_joint_inference_sec": counts_test_inference_sec,
                "train_joint_inference_sec": sentence_topic_train_soft_sec,
                "counts_test_inference_sec": counts_test_inference_sec,
                "train_soft_inference_sec": sentence_topic_train_soft_sec,
                "test_soft_inference_sec": sentence_topic_test_soft_sec,
                "iteration_diagnostics": [
                    asdict(item) for item in trainer.iteration_diagnostics
                ],
                "avg_log_likelihood": (
                    None
                    if metrics.avg_log_likelihood is None
                    else float(metrics.avg_log_likelihood)
                ),
                "perplexity": (
                    None if metrics.perplexity is None else float(metrics.perplexity)
                ),
            },
        ),
        options.output_dir,
    )
    selection_path = options.output_dir / PREPROCESSING_SELECTION_FILENAME
    save_json(
        {
            "train": train_selection.to_json_dict(),
            "test": test_selection.to_json_dict(),
        },
        selection_path,
    )

    extras = {
        "metrics_path": saved_outputs["metrics_path"],
        "train_doc_topic_soft": saved_outputs["doc_topic_train_soft"],
        "test_doc_topic_soft": saved_outputs["doc_topic_test_soft"],
        "train_sentence_topic_soft": saved_outputs["sentence_topic_train_soft"],
        "test_sentence_topic_soft": saved_outputs["sentence_topic_test_soft"],
        "train_preprocessed": saved_outputs["train_preprocessed"],
        "test_preprocessed": saved_outputs["test_preprocessed"],
        "preprocessing_selection": selection_path,
    }
    if counts_train is not None:
        extras["counts"] = saved_outputs["table_counts_per_doc"]

    return ModelArtifacts(
        train_path=saved_outputs["doc_topic_train"],
        infer_path=saved_outputs["doc_topic_test"],
        extras=extras,
    )


VMF_RUNNER = ModelRunnerSpec(
    key="vmf_sentence_lda",
    display_name="vMF Sentence LDA",
    family="vmf_sentence_lda",
    runner=_run_vmf_request,
)


def get_model_runner_spec(name: str) -> ModelRunnerSpec:
    if name == VMF_RUNNER.key:
        return VMF_RUNNER
    if name in BASELINE_RUNNERS:
        spec = BASELINE_RUNNERS[name]
        return ModelRunnerSpec(
            key=spec.key,
            display_name=spec.display_name,
            family=spec.family,
            runner=lambda request: _run_baseline_model_request(spec.key, request),
            method_kind=spec.method_kind,
        )
    raise ValueError(f"Unknown model runner: {name}")


def _run_baseline_model_request(name: str, request: ModelRunRequest) -> ModelArtifacts:
    artifacts = run_baseline_request(
        BaselineRunRequest(
            name=name,
            category=request.category,
            dataset=request.dataset,
            num_topics=request.num_topics,
            iteration=request.iteration,
            options=dict(request.options),
        )
    )
    return ModelArtifacts(
        train_path=artifacts.train_path,
        infer_path=artifacts.infer_path,
        extras=dict(artifacts.extras),
    )


def run_model_request(request: ModelRunRequest) -> ModelArtifacts:
    spec = get_model_runner_spec(request.name)
    return spec.runner(request)
