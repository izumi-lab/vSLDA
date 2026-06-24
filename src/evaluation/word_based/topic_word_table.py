from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.utils import simple_preprocess

from src.core.artifacts import load_artifact_pickle
from src.core.paths import (
    RESULTS_ROOT,
    build_baseline_doc_topic_path,
    build_vmf_doc_topic_path,
)
from src.data.catalog import get_dataset_targets, resolve_dataset_dir
from src.evaluation.model_provenance import load_model_provenance_for_artifact
from src.evaluation.reporting import read_evaluation_json, write_evaluation_json
from src.evaluation.schema import build_evaluation_meta
from src.utils.japanese_tokenizer import (
    is_japanese_language,
    tokenize_japanese_documents,
)


def _escape_latex(text: str) -> str:
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = text
    for k, v in repl.items():
        out = out.replace(k, v)
    return out


def _load_doc_topics(path: Path) -> np.ndarray:
    arr = np.asarray(load_artifact_pickle(path), dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    row_sums = arr.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0.0] = 1.0
    return arr / row_sums


def _load_filtered_documents(
    *,
    dataset: str,
    category: str,
    split: str,
    data_column: str,
    target_column: str,
    label_schema: str,
    delimiter: str,
) -> list[str]:
    dataset_dir = resolve_dataset_dir(dataset)
    if dataset_dir is None:
        raise ValueError(
            f"Could not resolve dataset directory for '{dataset}' under data/."
        )
    csv_path = dataset_dir / f"{split}.csv"
    df = pd.read_csv(csv_path)
    if data_column not in df.columns:
        raise ValueError(f"data_column '{data_column}' not found in {csv_path}")
    if target_column not in df.columns:
        raise ValueError(f"target_column '{target_column}' not found in {csv_path}")

    dataset_targets = (
        get_dataset_targets(
            dataset,
            target_column=target_column,
            label_schema=label_schema,
        )
        or {}
    )
    allowed = dataset_targets.get(category)

    docs: list[str] = []
    for _, row in df.iterrows():
        text = str(row[data_column])
        if not text:
            continue
        segments = [s.strip() for s in text.split(delimiter) if s.strip()]
        if not segments:
            continue

        label = str(row[target_column]).strip()
        if allowed is not None and label not in allowed:
            continue
        docs.append(text)
    return docs


def _tokenize_documents(
    *,
    documents: list[str],
    language: str,
    delimiter: str,
    min_token_len: int,
    ja_replace_num: bool,
    ja_dicdir: str | None,
    ja_require_unidic: bool,
) -> list[list[str]]:
    if is_japanese_language(language):
        return tokenize_japanese_documents(
            documents,
            delimiter=delimiter,
            replace_num=ja_replace_num,
            stopwords=None,
            dicdir=ja_dicdir,
            require_unidic=ja_require_unidic,
        )

    toks: list[list[str]] = []
    for doc in documents:
        normalized = doc.replace(delimiter, " ")
        toks.append(simple_preprocess(normalized, deacc=True, min_len=min_token_len))
    return toks


def _extract_topic_words_by_id(
    payload: dict[str, Any],
    iteration: int | None,
) -> dict[int, list[str]]:
    per_iter = payload.get("per_iteration")
    if per_iter is None and isinstance(payload.get("topic_words_topk"), dict):
        per_iter = payload["topic_words_topk"].get("per_iteration")
    if not isinstance(per_iter, list) or not per_iter:
        raise ValueError("No per_iteration topic words found in topic-words JSON.")

    picked = per_iter[0]
    if iteration is not None:
        matched = [x for x in per_iter if int(x.get("iteration", -1)) == int(iteration)]
        if not matched:
            available = [x.get("iteration") for x in per_iter]
            raise ValueError(
                f"Requested iteration={iteration} not found in topic-words JSON. Available: {available}"
            )
        picked = matched[0]

    topics = picked.get("topics")
    if not isinstance(topics, list):
        raise ValueError("Invalid topic words format: topics is missing or not a list.")

    out: dict[int, list[str]] = {}
    for t in topics:
        topic_id = int(t.get("topic_id"))
        out[topic_id] = [
            str(word.get("word", "")).strip()
            for word in t.get("words", [])
            if str(word.get("word", "")).strip()
        ]
    return out


def _limit_topics(
    topics: list[dict[str, Any]],
    max_topics_per_group: int | None,
) -> list[dict[str, Any]]:
    if max_topics_per_group is None:
        return topics
    return topics[:max_topics_per_group]


def _iter_topic_groups(
    profile_results: dict[str, Any],
    labels_filter: set[str] | None,
    max_topics_per_group: int | None,
    topic_source: str,
):
    if topic_source in {"global", "both"}:
        global_topics = profile_results.get("global_top_topics", [])
        if isinstance(global_topics, list) and global_topics:
            yield "global", _limit_topics(global_topics, max_topics_per_group)

    if topic_source in {"labels", "both"}:
        labels = profile_results.get("labels", [])
        for item in labels:
            label = str(item.get("label", ""))
            if labels_filter is not None and label not in labels_filter:
                continue
            topics = item.get("top_topics", [])
            yield label, _limit_topics(topics, max_topics_per_group)


def _extract_selected_topic_ids(
    profile_results: dict[str, Any],
    labels_filter: set[str] | None,
    max_topics_per_group: int | None,
    topic_source: str,
) -> list[int]:
    topic_ids: list[int] = []
    for _, topics in _iter_topic_groups(
        profile_results=profile_results,
        labels_filter=labels_filter,
        max_topics_per_group=max_topics_per_group,
        topic_source=topic_source,
    ):
        for row in topics:
            topic_ids.append(int(row.get("topic_id")))
    return sorted(set(topic_ids))


def _compute_weighted_tf_topic_words(
    *,
    doc_topics: np.ndarray,
    tokenized_docs: list[list[str]],
    selected_topic_ids: list[int],
    words_per_topic: int,
) -> dict[int, list[str]]:
    if doc_topics.shape[0] != len(tokenized_docs):
        raise ValueError(
            f"Length mismatch: doc_topics rows={doc_topics.shape[0]} vs tokenized_docs={len(tokenized_docs)}"
        )
    if not selected_topic_ids:
        return {}

    counts_by_doc = [Counter(tokens) for tokens in tokenized_docs]
    score_by_topic: dict[int, Counter[str]] = {k: Counter() for k in selected_topic_ids}
    for d, bow in enumerate(counts_by_doc):
        if not bow:
            continue
        for k in selected_topic_ids:
            if k < 0 or k >= doc_topics.shape[1]:
                continue
            weight = float(doc_topics[d, k])
            if weight <= 0.0:
                continue
            for token, cnt in bow.items():
                score_by_topic[k][token] += weight * float(cnt)

    return {
        k: [
            word
            for word, _ in score_by_topic.get(k, Counter()).most_common(words_per_topic)
        ]
        for k in selected_topic_ids
    }


def _compute_doc_topic_npmi_topic_words(
    *,
    doc_topics: np.ndarray,
    tokenized_docs: list[list[str]],
    selected_topic_ids: list[int],
    words_per_topic: int,
) -> tuple[dict[int, list[str]], str]:
    if doc_topics.shape[0] != len(tokenized_docs):
        raise ValueError(
            f"Length mismatch: doc_topics rows={doc_topics.shape[0]} vs tokenized_docs={len(tokenized_docs)}"
        )
    if not selected_topic_ids:
        return {}, "document_topic_proxy_npmi"

    dictionary = Dictionary(tokenized_docs)
    if len(dictionary) == 0:
        return {k: [] for k in selected_topic_ids}, "document_topic_proxy_npmi"
    corpus_bow = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]
    num_topics = doc_topics.shape[1]
    vocab_size = len(dictionary)
    joint_counts = np.zeros((num_topics, vocab_size), dtype=np.float64)
    word_counts = np.zeros(vocab_size, dtype=np.float64)
    topic_counts = np.zeros(num_topics, dtype=np.float64)
    total_tokens = 0.0

    for doc_idx, bow in enumerate(corpus_bow):
        if not bow:
            continue
        theta = doc_topics[doc_idx]
        doc_len = 0.0
        for word_id, count in bow:
            weight = float(count)
            doc_len += weight
            word_counts[word_id] += weight
            joint_counts[:, word_id] += theta * weight
        topic_counts += theta * doc_len
        total_tokens += doc_len

    if total_tokens == 0.0:
        return {k: [] for k in selected_topic_ids}, "document_topic_proxy_npmi"

    p_wk = joint_counts / total_tokens
    p_w = word_counts / total_tokens
    p_k = topic_counts / total_tokens
    denom = np.outer(p_k, p_w)
    p_wk_safe = np.maximum(p_wk, 1e-12)
    denom_safe = np.maximum(denom, 1e-12)
    pmi = np.log(p_wk_safe / denom_safe)
    npmi = pmi / -np.log(p_wk_safe)
    npmi[p_wk == 0.0] = -1.0

    topic_words_by_id: dict[int, list[str]] = {}
    for topic_id in selected_topic_ids:
        if not (0 <= topic_id < npmi.shape[0]):
            topic_words_by_id[topic_id] = []
            continue
        row = npmi[topic_id]
        top_ids = np.argsort(-row)[: int(words_per_topic)]
        topic_words_by_id[topic_id] = [dictionary[word_id] for word_id in top_ids]
    return topic_words_by_id, "document_topic_proxy_npmi"


def _format_words_cell(words: list[str]) -> str:
    if not words:
        return _escape_latex("(N/A)")
    escaped = [_escape_latex(w) for w in words]
    if len(escaped) == 1:
        return escaped[0]
    return r"\begin{tabular}[c]{@{}l@{}}" + r" \\ ".join(escaped) + r"\end{tabular}"


def build_tex(
    *,
    profile_results: dict[str, Any],
    topic_words_by_id: dict[int, list[str]],
    words_per_topic: int,
    labels_filter: set[str] | None,
    max_topics_per_group: int | None,
    topic_source: str,
    include_score: bool,
    layout: str,
    table_width_scale: float,
) -> str:
    chunks: list[str] = []
    for _, topics in _iter_topic_groups(
        profile_results=profile_results,
        labels_filter=labels_filter,
        max_topics_per_group=max_topics_per_group,
        topic_source=topic_source,
    ):
        chunks.append(r"\begin{table}[t]")
        chunks.append(r"\centering")
        if layout == "horizontal":
            n_cols = max(1, len(topics))
            col_width = rf"\dimexpr{table_width_scale:.4f}\linewidth/{n_cols}\relax"
            colspec = " ".join([rf"p{{{col_width}}}"] * n_cols)
            chunks.append(rf"\begin{{tabular}}{{{colspec.strip()}}}")
            chunks.append(r"\hline")
            chunks.append(
                " & ".join(f"Topic{int(row.get('topic_id', -1))}" for row in topics)
                + r" \\"
            )
            chunks.append(r"\hline")
            if include_score:
                score_cells: list[str] = []
                for row in topics:
                    score_val = (
                        row.get("score_value")
                        if row.get("score_value") is not None
                        else row.get("mean_weight")
                    )
                    score_cells.append(
                        "" if score_val is None else f"{float(score_val):.4f}"
                    )
                chunks.append(" & ".join(score_cells) + r" \\")
            chunks.append(
                " & ".join(
                    _format_words_cell(
                        topic_words_by_id.get(int(row.get("topic_id", -1)), [])[
                            :words_per_topic
                        ]
                    )
                    for row in topics
                )
                + r" \\"
            )
            chunks.append(r"\hline")
        else:
            if include_score:
                chunks.append(r"\begin{tabular}{r r l l}")
                chunks.append(r"\hline")
                chunks.append(r"Rank & Topic & Score & Representative words \\")
                chunks.append(r"\hline")
            else:
                chunks.append(r"\begin{tabular}{r r l}")
                chunks.append(r"\hline")
                chunks.append(r"Rank & Topic & Representative words \\")
                chunks.append(r"\hline")
            for row in topics:
                rank = int(row.get("rank", 0))
                topic_id = int(row.get("topic_id", -1))
                words_text = _format_words_cell(
                    topic_words_by_id.get(topic_id, [])[:words_per_topic]
                )
                if include_score:
                    score_val = (
                        row.get("score_value")
                        if row.get("score_value") is not None
                        else row.get("mean_weight")
                    )
                    score_text = "" if score_val is None else f"{float(score_val):.4f}"
                    chunks.append(
                        f"{rank} & {topic_id} & {score_text} & {words_text} \\\\"
                    )
                else:
                    chunks.append(f"{rank} & {topic_id} & {words_text} \\\\")
            chunks.append(r"\hline")
        chunks.append(r"\end{tabular}")
        chunks.append(r"\end{table}")
        chunks.append("")
    return "\n".join(chunks).rstrip() + "\n"


def _resolve_doc_topic_path(
    *,
    profile_meta: dict[str, Any],
) -> Path:
    doc_topic_path_raw = profile_meta.get("doc_topic_path")
    if doc_topic_path_raw:
        return Path(str(doc_topic_path_raw))

    dataset = str(profile_meta.get("dataset", ""))
    category = str(profile_meta.get("category", "all"))
    split = str(profile_meta.get("split", "train"))
    model = str(profile_meta.get("model", ""))
    num_topics = int(profile_meta.get("num_topics", 0))
    iteration = int(profile_meta.get("iteration", 0))
    vmf_assignment = str(profile_meta.get("vmf_assignment", "soft"))
    results_root = Path(str(profile_meta.get("results_root", RESULTS_ROOT)))

    if model == "vmf_sentence_lda":
        return build_vmf_doc_topic_path(
            dataset=dataset,
            iteration=iteration,
            num_topics=num_topics,
            category=category,
            split=split,
            assignment=vmf_assignment,
            dataset_root=results_root / "experiments" / dataset,
        )

    resolved_path = build_baseline_doc_topic_path(
        model=model,
        dataset=dataset,
        iteration=iteration,
        num_topics=num_topics,
        category=category,
        split=split,
        baseline_root=results_root / "baselines",
    )
    if resolved_path is None:
        raise ValueError(
            f"Unsupported split '{split}' for automatic resolution with model '{model}'."
        )
    return resolved_path


def run_topic_table_tex(
    *,
    profile_json: Path,
    topic_words_json: Path | None = None,
    iteration: int | None = None,
    labels: list[str] | None = None,
    max_topics_per_group: int | None = None,
    topic_source: str = "labels",
    words_per_topic: int = 10,
    language: str | None = None,
    data_column: str = "data",
    target_column: str = "target_str",
    label_schema: str = "identity",
    delimiter: str = " / ",
    min_token_len: int = 2,
    ja_replace_num: bool = False,
    ja_dicdir: str | None = None,
    ja_require_unidic: bool = False,
    representative_words_method: str = "weighted_tf",
    include_score: bool = False,
    layout: str = "horizontal",
    table_width_scale: float = 0.95,
    out_tex: Path | None = None,
) -> Path | str:
    profile_meta, profile_results = read_evaluation_json(profile_json)
    if not profile_meta and isinstance(profile_results, dict):
        profile_meta = {
            key: profile_results.get(key)
            for key in [
                "dataset",
                "category",
                "split",
                "iteration",
                "num_topics",
                "model",
                "vmf_assignment",
                "results_root",
                "doc_topic_path",
                "model_provenance",
            ]
            if key in profile_results
        }
        if "labels" in profile_results or "global_top_topics" in profile_results:
            profile_results = {
                "global_top_topics": profile_results.get("global_top_topics", []),
                "labels": profile_results.get("labels", []),
            }
    if not isinstance(profile_results, dict):
        raise ValueError(
            "profile_json must contain a dict-like evaluation result payload."
        )

    labels_filter = set(labels) if labels else None
    model_provenance: dict[str, Any] = {}
    source_meta: dict[str, Any] = {
        "profile_json": str(profile_json),
        "profile_model_provenance": profile_meta.get("model_provenance"),
    }
    if profile_meta.get("model_provenance") is not None:
        model_provenance["profile"] = profile_meta["model_provenance"]

    if topic_words_json is not None:
        topic_words_meta, topic_words_results = read_evaluation_json(topic_words_json)
        if not isinstance(topic_words_results, dict):
            raise ValueError(
                "topic_words_json must contain dict-like topic words results."
            )
        topic_words_by_id = _extract_topic_words_by_id(topic_words_results, iteration)
        source_meta["topic_words_json"] = str(topic_words_json)
        source_meta["topic_words_model_provenance"] = topic_words_meta.get(
            "model_provenance"
        )
        if topic_words_meta.get("model_provenance") is not None:
            model_provenance["topic_words"] = topic_words_meta["model_provenance"]
        representative_words_source = "topic_words_json"
    else:
        dataset = str(profile_meta.get("dataset", ""))
        category = str(profile_meta.get("category", "all"))
        split = str(profile_meta.get("split", "train"))
        resolved_doc_topic_path = _resolve_doc_topic_path(profile_meta=profile_meta)
        if not resolved_doc_topic_path.exists():
            raise FileNotFoundError(
                f"Doc-topic file not found: {resolved_doc_topic_path}"
            )
        doc_topics = _load_doc_topics(resolved_doc_topic_path)
        docs = _load_filtered_documents(
            dataset=dataset,
            category=category,
            split=split,
            data_column=data_column,
            target_column=target_column,
            label_schema=label_schema,
            delimiter=delimiter,
        )
        tokenized_docs = _tokenize_documents(
            documents=docs,
            language=language or "english",
            delimiter=delimiter,
            min_token_len=int(min_token_len),
            ja_replace_num=bool(ja_replace_num),
            ja_dicdir=ja_dicdir,
            ja_require_unidic=bool(ja_require_unidic),
        )
        selected_topic_ids = _extract_selected_topic_ids(
            profile_results,
            labels_filter=labels_filter,
            max_topics_per_group=max_topics_per_group,
            topic_source=topic_source,
        )
        if representative_words_method == "weighted_tf":
            topic_words_by_id = _compute_weighted_tf_topic_words(
                doc_topics=doc_topics,
                tokenized_docs=tokenized_docs,
                selected_topic_ids=selected_topic_ids,
                words_per_topic=int(words_per_topic),
            )
            representative_words_source = "weighted_tf_from_doc_topic"
        elif representative_words_method == "npmi":
            topic_words_by_id, representative_words_source = (
                _compute_doc_topic_npmi_topic_words(
                    doc_topics=doc_topics,
                    tokenized_docs=tokenized_docs,
                    selected_topic_ids=selected_topic_ids,
                    words_per_topic=int(words_per_topic),
                )
            )
        else:
            raise ValueError(
                "representative_words_method must be one of: "
                f"'weighted_tf', 'npmi' (got {representative_words_method!r})"
            )
        source_meta["doc_topic_path"] = str(resolved_doc_topic_path)
        source_meta["doc_topic_model_provenance"] = load_model_provenance_for_artifact(
            resolved_doc_topic_path,
            model_key=str(profile_meta.get("model", "unknown")),
        )
        source_meta["representative_words_method"] = representative_words_method
        model_provenance["doc_topic"] = source_meta["doc_topic_model_provenance"]

    tex = build_tex(
        profile_results=profile_results,
        topic_words_by_id=topic_words_by_id,
        words_per_topic=int(words_per_topic),
        labels_filter=labels_filter,
        max_topics_per_group=max_topics_per_group,
        topic_source=str(topic_source),
        include_score=bool(include_score),
        layout=str(layout),
        table_width_scale=float(table_width_scale),
    )

    if out_tex is None:
        return tex

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex, encoding="utf-8")
    sidecar_path = out_tex.with_suffix(".json")
    write_evaluation_json(
        meta=build_evaluation_meta(
            task="word_based_topic_word_table",
            output_kind="payload",
            tex_path=str(out_tex),
            representative_words_source=representative_words_source,
            representative_words_method=str(representative_words_method),
            words_per_topic=int(words_per_topic),
            topic_source=str(topic_source),
            include_score=bool(include_score),
            layout=str(layout),
            table_width_scale=float(table_width_scale),
            model_provenance=model_provenance,
            source_meta=source_meta,
        ),
        results={
            "selected_topic_ids": _extract_selected_topic_ids(
                profile_results,
                labels_filter=labels_filter,
                max_topics_per_group=max_topics_per_group,
                topic_source=topic_source,
            ),
            "labels_filter": (
                sorted(labels_filter) if labels_filter is not None else None
            ),
            "max_topics_per_group": max_topics_per_group,
        },
        path=sidecar_path,
    )
    return out_tex


run_word_based_topic_word_table = run_topic_table_tex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables of representative words for selected topics."
    )
    parser.add_argument("--profile-json", type=Path, required=True)
    parser.add_argument("--topic-words-json", type=Path, default=None)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--max-topics-per-group", type=int, default=None)
    parser.add_argument(
        "--topic-source", choices=["labels", "global", "both"], default="labels"
    )
    parser.add_argument("--words-per-topic", type=int, default=10)
    parser.add_argument("--language", default=None)
    parser.add_argument("--data-column", default="data")
    parser.add_argument("--target-column", default="target_str")
    parser.add_argument("--label-schema", default="identity")
    parser.add_argument("--delimiter", default=" / ")
    parser.add_argument("--min-token-len", type=int, default=2)
    parser.add_argument("--ja-replace-num", action="store_true")
    parser.add_argument("--ja-dicdir", default=None)
    parser.add_argument("--ja-require-unidic", action="store_true")
    parser.add_argument(
        "--representative-words-method",
        choices=["weighted_tf", "npmi"],
        default="weighted_tf",
    )
    parser.add_argument("--include-score", action="store_true")
    parser.add_argument(
        "--layout", choices=["vertical", "horizontal"], default="horizontal"
    )
    parser.add_argument("--table-width-scale", type=float, default=0.95)
    parser.add_argument("--out-tex", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = run_topic_table_tex(
        profile_json=args.profile_json,
        topic_words_json=args.topic_words_json,
        iteration=args.iteration,
        labels=args.labels,
        max_topics_per_group=args.max_topics_per_group,
        topic_source=args.topic_source,
        words_per_topic=args.words_per_topic,
        language=args.language,
        data_column=args.data_column,
        target_column=args.target_column,
        label_schema=args.label_schema,
        delimiter=args.delimiter,
        min_token_len=args.min_token_len,
        ja_replace_num=args.ja_replace_num,
        ja_dicdir=args.ja_dicdir,
        ja_require_unidic=args.ja_require_unidic,
        representative_words_method=args.representative_words_method,
        include_score=args.include_score,
        layout=args.layout,
        table_width_scale=args.table_width_scale,
        out_tex=args.out_tex,
    )
    if isinstance(output, Path):
        print(f"[info] wrote tex: {output}")
    else:
        print(output, end="")


if __name__ == "__main__":
    main()
