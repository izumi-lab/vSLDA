from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.evaluation.reporting import read_evaluation_json, write_evaluation_json
from src.evaluation.schema import build_evaluation_meta


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


def _extract_topic_words_by_id(
    payload: dict[str, Any],
    iteration: int | None,
) -> dict[int, list[str]]:
    if isinstance(payload.get("topics"), list):
        per_iter = [
            {"iteration": payload.get("iteration", 0), "topics": payload["topics"]}
        ]
    else:
        per_iter = payload.get("per_iteration")
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


def run_topic_table_tex(
    *,
    profile_json: Path,
    topic_words_json: Path,
    iteration: int | None = None,
    labels: list[str] | None = None,
    max_topics_per_group: int | None = None,
    topic_source: str = "labels",
    words_per_topic: int = 10,
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

    topic_words_meta, topic_words_results = read_evaluation_json(topic_words_json)
    if not isinstance(topic_words_results, dict):
        raise ValueError("topic_words_json must contain dict-like topic words results.")
    contract = {**topic_words_results, **topic_words_meta}
    if contract.get("topic_word_role") != "display":
        raise ValueError("topic_words_json must have topic_word_role='display'")
    for required_field in (
        "model",
        "split",
        "condition_fingerprint",
        "evaluation_vocabulary_fingerprint",
    ):
        if contract.get(required_field) in {None, ""}:
            raise ValueError(
                f"topic_words_json is missing required field {required_field!r}"
            )
    model = str(contract.get("model", profile_meta.get("model", "")))
    score_mode = str(contract.get("score_mode", ""))
    allowed_score_modes = (
        {"word_topic_npmi", "decoder_topic_word_probability"}
        if model == "ctm"
        else {"word_topic_npmi"}
    )
    if score_mode not in allowed_score_modes:
        raise ValueError(
            f"display score_mode for model {model!r} must be "
            f"one of {sorted(allowed_score_modes)!r}, got {score_mode!r}"
        )
    topic_words_by_id = _extract_topic_words_by_id(topic_words_results, iteration)
    source_meta["topic_words_json"] = str(topic_words_json)
    source_meta["topic_word_role"] = "display"
    source_meta["score_mode"] = score_mode
    source_meta["topic_words_model_provenance"] = topic_words_meta.get(
        "model_provenance"
    )
    if topic_words_meta.get("model_provenance") is not None:
        model_provenance["topic_words"] = topic_words_meta["model_provenance"]
    representative_words_source = str(contract.get("source", "display_artifact"))

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
            topic_word_role="display",
            score_mode=score_mode,
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
    parser.add_argument("--topic-words-json", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=None)
    parser.add_argument("--labels", nargs="+", default=None)
    parser.add_argument("--max-topics-per-group", type=int, default=None)
    parser.add_argument(
        "--topic-source", choices=["labels", "global", "both"], default="labels"
    )
    parser.add_argument("--words-per-topic", type=int, default=10)
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
