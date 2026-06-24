from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from src.core.paths import CLASSIFICATION_RESULTS_ROOT
from src.evaluation.classification.config import (
    ALIGNMENT_MODES,
    DEFAULT_ALIGNMENT_MODE,
    DEFAULT_FEATURE_RESOLVE_MODE,
    FEATURE_RESOLVE_MODES,
)
from src.utils.random import DEFAULT_RANDOM_SEED


def register_evaluation_commands(evaluation_app: typer.Typer) -> None:
    @evaluation_app.command(
        "classify",
        help=(
            "Run classification and write outputs under "
            "results/classification/archive/<date>/<dataset>/<data_run>/all/<display_key>/<execution_id>/ "
            "and update results/classification/latest/<dataset>/<data_run>/all/<display_key>/CURRENT.json."
        ),
    )
    def classify(
        dataset: list[str] = typer.Option(["20newsgroup", "nyt"], "--dataset"),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        category: list[str] = typer.Option([], "--category"),
        topic: list[int] = typer.Option([10, 20], "--topic"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        classifier: list[str] = typer.Option(["svm"], "--classifier"),
        vmf_assignment: str = typer.Option("hard"),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        seed: Optional[int] = typer.Option(42),
        alignment_mode: str = typer.Option(DEFAULT_ALIGNMENT_MODE, "--alignment-mode"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        model: list[str] = typer.Option([], "--model"),
        feature_resolve_mode: str = typer.Option(
            DEFAULT_FEATURE_RESOLVE_MODE,
            "--feature-resolve-mode",
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        if alignment_mode not in ALIGNMENT_MODES:
            raise typer.BadParameter(
                f"alignment mode must be one of {', '.join(ALIGNMENT_MODES)}"
            )
        if feature_resolve_mode not in FEATURE_RESOLVE_MODES:
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        register_builtin_tasks()
        run_task(
            "classification",
            iterations=iteration,
            datasets=dataset,
            data_runs=data_run,
            categories=list(category) or None,
            topics=topic,
            classifiers=classifier,
            vmf_assignment=vmf_assignment,
            result_root=result_root,
            target_column=target_column,
            label_schema=label_schema,
            seed=seed,
            alignment_mode=alignment_mode,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=list(model) or None,
        )

    @evaluation_app.command(
        "classify-limited",
        help=(
            "Run limited-data classification and write outputs under "
            "results/classification/archive/<date>/<dataset>/<data_run>/all/<display_key>/<execution_id>/ "
            "and update results/classification/latest/<dataset>/<data_run>/all/<display_key>/CURRENT.json."
        ),
    )
    def classify_limited(
        dataset: list[str] = typer.Option(["20newsgroup", "nyt"], "--dataset"),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        category: list[str] = typer.Option([], "--category"),
        topic: list[int] = typer.Option([10, 20], "--topic"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        classifier: list[str] = typer.Option(["svm"], "--classifier"),
        ratio: Optional[float] = typer.Option(None, min=0.0, max=1.0),
        count: Optional[int] = typer.Option(None, min=0),
        vmf_assignment: str = typer.Option("hard"),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        stratified: bool = typer.Option(True),
        seed: Optional[int] = typer.Option(42),
        sampling_repeat: list[int] = typer.Option(
            [],
            "--sampling-repeat",
            min=0,
            help=(
                "Limited-data sampling repeat index. Repeat this option to run "
                "multiple stratified train subsets per topic-model iteration."
            ),
        ),
        sampling_seed_stride: int = typer.Option(
            1000,
            "--sampling-seed-stride",
            min=1,
            help="Seed offset between limited-data sampling repeats.",
        ),
        sampling_max_attempts: int = typer.Option(
            1,
            "--sampling-max-attempts",
            min=1,
            help=(
                "Maximum seed attempts for a limited-data sample. Attempts after "
                "the first are used only when classifier training fails because "
                "feature alignment leaves fewer than two classes."
            ),
        ),
        sampling_retry_seed_stride: int = typer.Option(
            100000,
            "--sampling-retry-seed-stride",
            min=1,
            help="Seed offset between adaptive limited-data retry attempts.",
        ),
        alignment_mode: str = typer.Option(DEFAULT_ALIGNMENT_MODE, "--alignment-mode"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        model: list[str] = typer.Option([], "--model"),
        feature_resolve_mode: str = typer.Option(
            DEFAULT_FEATURE_RESOLVE_MODE,
            "--feature-resolve-mode",
        ),
    ) -> None:
        from src.cli.workflows import resolve_limited_classification_setting
        from src.evaluation.registry import register_builtin_tasks, run_task

        try:
            mode, value = resolve_limited_classification_setting(
                ratio=ratio, count=count
            )
        except ValueError as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=2) from exc
        if alignment_mode not in ALIGNMENT_MODES:
            raise typer.BadParameter(
                f"alignment mode must be one of {', '.join(ALIGNMENT_MODES)}"
            )
        if feature_resolve_mode not in FEATURE_RESOLVE_MODES:
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        register_builtin_tasks()
        run_task(
            "classification_limited",
            mode=mode,
            value=value,
            result_root=result_root,
            iterations=iteration,
            datasets=dataset,
            data_runs=data_run,
            categories=list(category) or None,
            topics=topic,
            classifiers=classifier,
            vmf_assignment=vmf_assignment,
            target_column=target_column,
            label_schema=label_schema,
            stratified=stratified,
            seed=seed,
            sampling_repeats=list(sampling_repeat) or None,
            sampling_seed_stride=sampling_seed_stride,
            sampling_max_attempts=sampling_max_attempts,
            sampling_retry_seed_stride=sampling_retry_seed_stride,
            alignment_mode=alignment_mode,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=list(model) or None,
        )

    @evaluation_app.command(
        "summarize-classification",
        help=(
            "Read classification outputs from "
            "results/classification/latest/<dataset>/<data_run>/all/<display_key>/CURRENT.json "
            "when available, otherwise fall back to legacy category-first directories. "
            "Use --resolve-mode strict to fail on ambiguous matches instead of picking the newest."
        ),
    )
    def summarize_classification(
        metric: str = typer.Option("acc"),
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        topic: int = typer.Option(...),
        iteration: list[int] = typer.Option([0], "--iteration"),
        classifier: list[str] = typer.Option([], "--classifier"),
        vmf_assignment: str = typer.Option("hard"),
        alignment_mode: str = typer.Option(DEFAULT_ALIGNMENT_MODE, "--alignment-mode"),
        resolve_mode: str = typer.Option("latest", "--resolve-mode"),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        model: list[str] = typer.Option([], "--model"),
        exclude_category: list[str] = typer.Option([], "--exclude-category"),
        include_all_category: bool = typer.Option(False, "--include-all-category"),
        feature_resolve_mode: str = typer.Option(
            DEFAULT_FEATURE_RESOLVE_MODE,
            "--feature-resolve-mode",
        ),
        output_path: Optional[Path] = typer.Option(
            None,
            "--output-path",
            dir_okay=False,
            help="Write the rendered LaTeX table to this .tex file.",
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        if resolve_mode not in {"latest", "strict"}:
            raise typer.BadParameter("resolve mode must be 'latest' or 'strict'")
        if alignment_mode not in ALIGNMENT_MODES:
            raise typer.BadParameter(
                f"alignment mode must be one of {', '.join(ALIGNMENT_MODES)}"
            )
        if feature_resolve_mode not in FEATURE_RESOLVE_MODES:
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        register_builtin_tasks()
        run_task(
            "classification_summary",
            metric=metric,
            dataset=dataset,
            data_run=data_run,
            topics=topic,
            iterations=iteration,
            classifiers=list(classifier) or None,
            vmf_assignment=vmf_assignment,
            alignment_mode=alignment_mode,
            resolve_mode=resolve_mode,
            result_root=result_root,
            target_column=target_column,
            label_schema=label_schema,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
            selected_models=list(model) or None,
            excluded_categories=list(exclude_category) or None,
            include_all_category=include_all_category,
            output_path=output_path,
        )

    @evaluation_app.command("list-tasks")
    def list_evaluation_tasks() -> None:
        from src.evaluation.registry import list_tasks, register_builtin_tasks

        register_builtin_tasks()
        typer.echo("task\toutput\trun_from_config\tdescription")
        for task in list_tasks():
            typer.echo(
                f"{task.name}\t{task.output_kind}\t"
                f"{'yes' if task.run_from_config_supported else 'no'}\t"
                f"{task.description}"
            )

    @evaluation_app.command(
        "run-from-config",
        help=(
            "Run supported evaluation task(s) from a comparison config. "
            "When --task is omitted, evaluation.tasks is used. "
            "Generated outputs follow each task's current layout; migrated tasks use "
            "latest/archive pointers."
        ),
    )
    def run_evaluation_from_config(
        config: Path = typer.Option(..., exists=True, dir_okay=False),
        task: Optional[str] = typer.Option(None),
        classifier: list[str] = typer.Option([], "--classifier"),
        vmf_assignment: str = typer.Option("hard"),
        result_root: Path = typer.Option(CLASSIFICATION_RESULTS_ROOT, file_okay=False),
        target_column: Optional[str] = typer.Option(None),
        label_schema: str = typer.Option("identity"),
        embedding_variant: list[str] = typer.Option([], "--embedding-variant"),
        feature_resolve_mode: Optional[str] = typer.Option(
            None,
            "--feature-resolve-mode",
        ),
    ) -> None:
        from src.cli.workflows import run_evaluation_from_config_workflow

        if (
            feature_resolve_mode is not None
            and feature_resolve_mode not in FEATURE_RESOLVE_MODES
        ):
            raise typer.BadParameter(
                "feature resolve mode must be one of "
                f"{', '.join(FEATURE_RESOLVE_MODES)}"
            )
        run_evaluation_from_config_workflow(
            config=config,
            task=task,
            classifiers=classifier,
            vmf_assignment=vmf_assignment,
            result_root=result_root,
            target_column=target_column,
            label_schema=label_schema,
            embedding_variants=list(embedding_variant) or None,
            feature_resolve_mode=feature_resolve_mode,
        )

    @evaluation_app.command(
        "geometry-based-metrics",
        help=(
            "Write geometry metrics under "
            "results/topic_analysis/geometry_based/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/topic_analysis/geometry_based/latest/.../CURRENT.json "
            "by default, or under --out-root when provided."
        ),
    )
    def geometry_based_metrics(
        model: list[str] = typer.Option(["vmf"], "--model"),
        dataset: str = typer.Option(...),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: int = typer.Option(...),
        category: list[str] = typer.Option(["all"], "--category"),
        dup_threshold: float = typer.Option(0.90),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        encoder_model: Optional[str] = typer.Option(
            None,
            "--encoder-model",
            "--encoder_model",
            help=(
                "Resolve the embedding variant from an encoder model name, e.g. "
                "sentence-transformers/all-mpnet-base-v2 -> mpnet."
            ),
        ),
        out_root: Path = typer.Option(
            Path("results/topic_analysis/geometry_based"), file_okay=False
        ),
        save_per_iter_artifacts: bool = typer.Option(False),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "geometry_based_metrics",
            models=model,
            dataset=dataset,
            data_runs=data_run,
            iterations=iteration,
            num_topics=topic,
            categories=category,
            dup_threshold=dup_threshold,
            embedding_variant=embedding_variant,
            encoder_model=encoder_model,
            out_root=out_root,
            save_per_iter_artifacts=save_per_iter_artifacts,
        )

    @evaluation_app.command(
        "sentence-topic-inspection",
        help=(
            "Write inspection payloads under "
            "results/visualization/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/visualization/latest/.../CURRENT.json."
        ),
    )
    def sentence_topic_inspection(
        model: str = typer.Option("vmf_sentence_lda", "--model"),
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        category: list[str] = typer.Option(["all"], "--category"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: list[int] = typer.Option(..., "--topic"),
        condition_id: Optional[str] = typer.Option(None, "--condition-id"),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        num_components: Optional[int] = typer.Option(None, "--num-components"),
        gaussian_condition_id: Optional[str] = typer.Option(
            None, "--gaussian-condition-id"
        ),
        gaussian_embedding_variant: Optional[str] = typer.Option(
            None, "--gaussian-embedding-variant"
        ),
        gaussian_num_components: Optional[int] = typer.Option(
            None, "--gaussian-num-components"
        ),
        topk: int = typer.Option(5),
        encoder: Optional[str] = typer.Option(None),
        split: str = typer.Option("train"),
        data_column: str = typer.Option("data"),
        target_column: str = typer.Option("target_str"),
        delimiter: str = typer.Option(" / "),
        language: str = typer.Option("english"),
        segmenter: str = typer.Option("delimiter"),
        seed: int = typer.Option(DEFAULT_RANDOM_SEED),
        gaussian_topk: bool = typer.Option(False),
        device: Optional[str] = typer.Option(None),
        encode_batch_size: int = typer.Option(64),
        max_points: int = typer.Option(2000),
        results_root: Path = typer.Option(Path("results"), file_okay=False),
        out_root: Path = typer.Option(Path("results/visualization"), file_okay=False),
        no_progress: bool = typer.Option(False),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "sentence_topic_inspection",
            model=model,
            dataset=dataset,
            data_run=data_run,
            categories=category,
            iterations=iteration,
            num_topics_list=topic,
            source_condition_id=condition_id,
            embedding_variant=embedding_variant,
            num_components=num_components,
            gaussian_condition_id=gaussian_condition_id,
            gaussian_embedding_variant=gaussian_embedding_variant,
            gaussian_num_components=gaussian_num_components,
            top_k=topk,
            encoder_model=encoder,
            split=split,
            data_column=data_column,
            target_column=target_column,
            delimiter=delimiter,
            language=language,
            segmenter=segmenter,
            seed=seed,
            gaussian_topk=gaussian_topk,
            device=device,
            encode_batch_size=encode_batch_size,
            show_progress=not no_progress,
            max_points=max_points,
            results_root=results_root,
            out_root=out_root,
        )

    @evaluation_app.command(
        "word-based-label-profile",
        help=(
            "Write label-profile outputs under "
            "results/topic_analysis/label_profile/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/topic_analysis/label_profile/latest/.../CURRENT.json."
        ),
    )
    def word_based_label_profile(
        model: str = typer.Option(...),
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        category: str = typer.Option("all"),
        split: str = typer.Option("train"),
        iteration: int = typer.Option(...),
        topic: int = typer.Option(...),
        top_n: int = typer.Option(5),
        sort_by: str = typer.Option("ratio"),
        pmi_eps: float = typer.Option(1e-12),
        min_docs_per_label: int = typer.Option(1),
        vmf_assignment: str = typer.Option("soft"),
        results_root: Path = typer.Option(Path("results"), file_okay=False),
        data_column: str = typer.Option("data"),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        delimiter: str = typer.Option(" / "),
        out_json: Optional[Path] = typer.Option(None),
        out_csv: Optional[Path] = typer.Option(None),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "word_based_label_profile",
            model=model,
            dataset=dataset,
            data_run=data_run,
            category=category,
            split=split,
            iteration=iteration,
            num_topics=topic,
            top_n=top_n,
            sort_by=sort_by,
            pmi_eps=pmi_eps,
            min_docs_per_label=min_docs_per_label,
            vmf_assignment=vmf_assignment,
            results_root=results_root,
            data_column=data_column,
            target_column=target_column,
            label_schema=label_schema,
            delimiter=delimiter,
            out_json=out_json,
            out_csv=out_csv,
        )

    @evaluation_app.command("word-based-topic-word-table")
    def word_based_topic_word_table(
        profile_json: Path = typer.Option(...),
        topic_words_json: Optional[Path] = typer.Option(None),
        iteration: Optional[int] = typer.Option(None),
        labels: Optional[list[str]] = typer.Option(None),
        max_topics_per_group: Optional[int] = typer.Option(None),
        topic_source: str = typer.Option("labels"),
        words_per_topic: int = typer.Option(10),
        language: Optional[str] = typer.Option(None),
        data_column: str = typer.Option("data"),
        target_column: str = typer.Option("target_str"),
        label_schema: str = typer.Option("identity"),
        delimiter: str = typer.Option(" / "),
        min_token_len: int = typer.Option(2),
        ja_replace_num: bool = typer.Option(False),
        ja_dicdir: Optional[str] = typer.Option(None),
        ja_require_unidic: bool = typer.Option(False),
        representative_words_method: str = typer.Option("weighted_tf"),
        include_score: bool = typer.Option(False),
        layout: str = typer.Option("horizontal"),
        table_width_scale: float = typer.Option(0.95),
        out_tex: Optional[Path] = typer.Option(None),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        result = run_task(
            "word_based_topic_word_table",
            profile_json=profile_json,
            topic_words_json=topic_words_json,
            iteration=iteration,
            labels=labels,
            max_topics_per_group=max_topics_per_group,
            topic_source=topic_source,
            words_per_topic=words_per_topic,
            language=language,
            data_column=data_column,
            target_column=target_column,
            label_schema=label_schema,
            delimiter=delimiter,
            min_token_len=min_token_len,
            ja_replace_num=ja_replace_num,
            ja_dicdir=ja_dicdir,
            ja_require_unidic=ja_require_unidic,
            representative_words_method=representative_words_method,
            include_score=include_score,
            layout=layout,
            table_width_scale=table_width_scale,
            out_tex=out_tex,
        )
        if isinstance(result, Path):
            typer.echo(str(result))
        else:
            typer.echo(str(result), nl=False)

    @evaluation_app.command(
        "word-based-metrics",
        help=(
            "Write topic-word metrics under "
            "results/topic_analysis/coherence/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/topic_analysis/coherence/latest/.../CURRENT.json "
            "by default."
        ),
    )
    def word_based_metrics(
        model: list[str] = typer.Option(["vmf"], "--model"),
        dataset: str = typer.Option(...),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        topic: list[int] = typer.Option(..., "--topic"),
        category: list[str] = typer.Option(["all"], "--category"),
        embedding_variant: Optional[str] = typer.Option("mpnet", "--embedding-variant"),
        out_root: Path = typer.Option(
            Path("results/topic_analysis/coherence"), file_okay=False
        ),
        coherence: list[str] = typer.Option(["c_v"], "--coherence"),
        coherence_topn: int = typer.Option(10),
        coherence_window_size: Optional[int] = typer.Option(None),
        coherence_min_window_count: Optional[int] = typer.Option(None),
        diversity_topn: int = typer.Option(25),
        gaussian_word2vec: str = typer.Option("glove-wiki-gigaword-100"),
        coherence_split: str = typer.Option("train"),
        coherence_min_token_len: int = typer.Option(2),
        dict_no_below: int = typer.Option(3),
        dict_no_above: float = typer.Option(0.7),
        dict_exclude_single_alpha: bool = typer.Option(False),
        dict_exclude_with_digit: bool = typer.Option(False),
        dict_exclude_hiragana_only: bool = typer.Option(False),
        proxy_npmi_mode: str = typer.Option("sentence"),
        proxy_word_score_mode: str = typer.Option("word_npmi"),
        coherence_reference: str = typer.Option("dataset"),
        coherence_reference_path: Optional[Path] = typer.Option(None),
        coherence_reference_format: str = typer.Option("tokenized_jsonl"),
        coherence_reference_max_docs: Optional[int] = typer.Option(None),
        coherence_reference_min_doc_tokens: int = typer.Option(1),
        coherence_reference_streaming: bool = typer.Option(False),
        coherence_count_backend: str = typer.Option(
            "numba", "--coherence-count-backend"
        ),
        coherence_count_workers: int = typer.Option(8, "--coherence-count-workers"),
        coherence_count_chunk_size: int = typer.Option(
            25000, "--coherence-count-chunk-size"
        ),
        coherence_topic_word_workers: int = typer.Option(
            1, "--coherence-topic-word-workers"
        ),
        coherence_score_workers: int = typer.Option(1, "--coherence-score-workers"),
        skip_existing: bool = typer.Option(False),
        language: str = typer.Option("english"),
        delimiter: str = typer.Option(" / "),
        ja_replace_num: bool = typer.Option(True),
        ja_dicdir: Optional[str] = typer.Option(None),
        ja_require_unidic: bool = typer.Option(True),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "word_based_metrics",
            models=model,
            dataset=dataset,
            data_runs=data_run,
            iterations=iteration,
            num_topics=topic[0] if len(topic) == 1 else topic,
            categories=category,
            embedding_variant=embedding_variant,
            out_root=out_root,
            coherence=coherence[0] if len(coherence) == 1 else coherence,
            coherence_topn=coherence_topn,
            coherence_window_size=coherence_window_size,
            coherence_min_window_count=coherence_min_window_count,
            diversity_topn=diversity_topn,
            gaussian_word2vec=gaussian_word2vec,
            coherence_split=coherence_split,
            coherence_min_token_len=coherence_min_token_len,
            dict_no_below=dict_no_below,
            dict_no_above=dict_no_above,
            dict_exclude_single_alpha=dict_exclude_single_alpha,
            dict_exclude_with_digit=dict_exclude_with_digit,
            dict_exclude_hiragana_only=dict_exclude_hiragana_only,
            proxy_npmi_mode=proxy_npmi_mode,
            proxy_word_score_mode=proxy_word_score_mode,
            coherence_reference=coherence_reference,
            coherence_reference_path=coherence_reference_path,
            coherence_reference_format=coherence_reference_format,
            coherence_reference_max_docs=coherence_reference_max_docs,
            coherence_reference_min_doc_tokens=coherence_reference_min_doc_tokens,
            coherence_reference_streaming=coherence_reference_streaming,
            coherence_count_backend=coherence_count_backend,
            coherence_count_workers=coherence_count_workers,
            coherence_count_chunk_size=coherence_count_chunk_size,
            coherence_topic_word_workers=coherence_topic_word_workers,
            coherence_score_workers=coherence_score_workers,
            skip_existing=skip_existing,
            language=language,
            delimiter=delimiter,
            ja_replace_num=ja_replace_num,
            ja_dicdir=ja_dicdir,
            ja_require_unidic=ja_require_unidic,
        )

    @evaluation_app.command(
        "topic-count-diagnostics",
        help=(
            "Write diagnostics under "
            "results/topic_count_analysis/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/topic_count_analysis/latest/.../CURRENT.json."
        ),
    )
    def topic_count_diagnostics(
        dataset: str = typer.Option(...),
        topic: list[int] = typer.Option(..., "--topic"),
        iteration: list[int] = typer.Option([0], "--iteration"),
        category: list[str] = typer.Option(["all"], "--category"),
        data_run: list[str] = typer.Option(["default"], "--data-run"),
        condition_id: Optional[str] = typer.Option(None, "--condition-id"),
        embedding_variant: Optional[str] = typer.Option(None, "--embedding-variant"),
        num_components: Optional[int] = typer.Option(None, "--num-components"),
        split: str = typer.Option("test"),
        eval_mode: str = typer.Option("predictive-soft-theta", "--eval-mode"),
        strict: bool = typer.Option(True),
        encoder: Optional[str] = typer.Option(None),
        device: Optional[str] = typer.Option(None),
        encode_batch_size: int = typer.Option(64),
        data_column: str = typer.Option("data"),
        target_column: str = typer.Option("target_str"),
        delimiter: str = typer.Option(" / "),
        language: str = typer.Option("english"),
        segmenter: str = typer.Option("delimiter"),
        no_progress: bool = typer.Option(False),
        results_root: Path = typer.Option(Path("results/experiments"), file_okay=False),
        out_root: Path = typer.Option(
            Path("results/topic_count_analysis"), file_okay=False
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "topic_count_diagnostics",
            dataset=dataset,
            iterations=iteration,
            topics=topic,
            categories=category,
            data_runs=data_run,
            source_condition_id=condition_id,
            embedding_variant=embedding_variant,
            num_components=num_components,
            split=split,
            eval_mode=eval_mode,
            strict=strict,
            encoder_model=encoder,
            device=device,
            encode_batch_size=encode_batch_size,
            show_progress=not no_progress,
            data_column=data_column,
            target_column=target_column,
            delimiter=delimiter,
            language=language,
            segmenter=segmenter,
            results_root=results_root,
            out_root=out_root,
        )

    @evaluation_app.command(
        "cross-model-pair-diagnostics",
        help=(
            "Write pair diagnostics under "
            "results/analysis/vmf_vs_baseline/archive/<date>/<dataset>/<data_run>/<category>/<display_key>/exec_<timestamp>/ "
            "and update results/analysis/vmf_vs_baseline/latest/.../CURRENT.json."
        ),
    )
    def cross_model_pair_diagnostics(
        dataset: str = typer.Option(...),
        data_run: str = typer.Option("default", "--data-run"),
        category: str = typer.Option("all"),
        iteration: int = typer.Option(0),
        topic: int = typer.Option(...),
        split: str = typer.Option("train"),
        baseline: str = typer.Option("bleilda"),
        vmf_assignment: str = typer.Option("hard"),
        vmf_condition_id: Optional[str] = typer.Option(None, "--vmf-condition-id"),
        vmf_embedding_variant: Optional[str] = typer.Option(
            None, "--vmf-embedding-variant"
        ),
        vmf_num_components: Optional[int] = typer.Option(None, "--vmf-num-components"),
        baseline_condition_id: Optional[str] = typer.Option(
            None, "--baseline-condition-id"
        ),
        baseline_embedding_variant: Optional[str] = typer.Option(
            None, "--baseline-embedding-variant"
        ),
        baseline_num_components: Optional[int] = typer.Option(
            None, "--baseline-num-components"
        ),
        k_neighbors: int = typer.Option(30),
        baseline_max: float = typer.Option(0.05),
        vmf_min: float = typer.Option(0.6),
        topn: int = typer.Option(10),
        unique_docs: bool = typer.Option(False),
        no_row_normalize: bool = typer.Option(False),
        dump_vectors: bool = typer.Option(False),
        seed: int = typer.Option(DEFAULT_RANDOM_SEED),
        results_root: Path = typer.Option(Path("results"), file_okay=False),
        out_root: Path = typer.Option(
            Path("results/analysis/vmf_vs_baseline"), file_okay=False
        ),
    ) -> None:
        from src.evaluation.registry import register_builtin_tasks, run_task

        register_builtin_tasks()
        run_task(
            "cross_model_pair_diagnostics",
            dataset=dataset,
            data_run=data_run,
            category=category,
            iteration=iteration,
            num_topics=topic,
            split=split,
            baseline=baseline,
            k_neighbors=k_neighbors,
            baseline_max=baseline_max,
            vmf_min=vmf_min,
            topn=topn,
            unique_docs=unique_docs,
            row_normalize=not no_row_normalize,
            dump_vectors=dump_vectors,
            seed=seed,
            vmf_assignment=vmf_assignment,
            vmf_condition_id=vmf_condition_id,
            vmf_embedding_variant=vmf_embedding_variant,
            vmf_num_components=vmf_num_components,
            baseline_condition_id=baseline_condition_id,
            baseline_embedding_variant=baseline_embedding_variant,
            baseline_num_components=baseline_num_components,
            results_root=results_root,
            out_root=out_root,
        )
