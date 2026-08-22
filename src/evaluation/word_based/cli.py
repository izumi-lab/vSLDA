from __future__ import annotations

import argparse
from pathlib import Path

from .model_inputs import DEFAULT_EMBEDDING_VARIANT, MODEL_CHOICES
from .reference_counts import (
    DEFAULT_REFERENCE_COUNT_CHUNK_SIZE,
    DEFAULT_REFERENCE_COUNT_WORKERS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze provenance-aware topic-word metrics for fitted models."
    )
    parser.add_argument("--model", nargs="+", choices=MODEL_CHOICES, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--data_run", nargs="+", default=["default"])
    parser.add_argument("--iteration", type=int, nargs="+", required=True)
    parser.add_argument("--num_topics", type=int, nargs="+", required=True)
    parser.add_argument("--category", nargs="+", default=["all"])
    parser.add_argument(
        "--embedding_variant",
        "--embedding-variant",
        type=str,
        default=DEFAULT_EMBEDDING_VARIANT,
    )
    parser.add_argument("--prior-scale", type=float, default=None)
    parser.add_argument(
        "--out_root", type=Path, default=Path("results/topic_analysis/coherence")
    )
    parser.add_argument(
        "--coherence",
        nargs="+",
        choices=["c_v", "c_npmi", "c_uci", "u_mass", "doc_npmi"],
        default=["c_v"],
    )
    parser.add_argument("--coherence_topn", type=int, default=10)
    parser.add_argument("--coherence_window_size", type=int, default=None)
    parser.add_argument("--coherence_min_window_count", type=int, default=None)
    parser.add_argument("--diversity_topn", type=int, default=25)
    parser.add_argument(
        "--topic-word-score-mode",
        choices=["word_topic_npmi", "topic_word_probability"],
        default="word_topic_npmi",
        help=("Representative-word ranking consumed by both coherence and diversity."),
    )
    parser.add_argument(
        "--gaussian_word2vec", type=str, default="word2vec-google-news-300"
    )
    parser.add_argument("--coherence_split", choices=["train", "test"], default="train")
    parser.add_argument("--coherence_min_token_len", type=int, default=2)
    parser.add_argument("--dict_no_below", type=int, default=3)
    parser.add_argument("--dict_no_above", type=float, default=0.7)
    parser.add_argument(
        "--dict_exclude_tokens",
        "--dict-exclude-tokens",
        type=lambda value: frozenset(
            token.strip() for token in value.split(",") if token.strip()
        ),
        default=frozenset(),
        help="Comma-separated exact tokens to exclude from evaluation dictionaries.",
    )
    parser.add_argument("--dict_exclude_single_alpha", action="store_true")
    parser.add_argument("--dict_exclude_with_digit", action="store_true")
    parser.add_argument("--dict_exclude_hiragana_only", action="store_true")
    parser.add_argument("--posterior-num-chains", type=int, default=1)
    parser.add_argument("--posterior-burn-in-sweeps", type=int, default=20)
    parser.add_argument("--posterior-retained-samples", type=int, default=20)
    parser.add_argument("--posterior-thinning", type=int, default=1)
    parser.add_argument("--posterior-seed", type=int, default=0)
    parser.add_argument(
        "--posterior-backend", choices=["python", "numba"], default="numba"
    )
    parser.add_argument("--etm-theta-samples", type=int, default=100)
    parser.add_argument("--etm-posterior-seed", type=int, default=0)
    parser.add_argument("--npmi-min-expected-count", type=float, default=None)
    parser.add_argument(
        "--coherence_reference",
        choices=["dataset", "wikipedia"],
        default="wikipedia",
    )
    parser.add_argument("--coherence_reference_path", type=Path, default=None)
    parser.add_argument(
        "--coherence_reference_format",
        choices=["tokenized_jsonl"],
        default="tokenized_jsonl",
    )
    parser.add_argument("--coherence_reference_max_docs", type=int, default=None)
    parser.add_argument("--coherence_reference_min_doc_tokens", type=int, default=1)
    parser.add_argument("--coherence_reference_streaming", action="store_true")
    parser.add_argument(
        "--coherence_count_backend",
        "--coherence-count-backend",
        choices=["python", "numba", "numba_interval"],
        default="numba",
    )
    parser.add_argument(
        "--coherence_count_workers",
        "--coherence-count-workers",
        type=int,
        default=DEFAULT_REFERENCE_COUNT_WORKERS,
    )
    parser.add_argument(
        "--coherence_count_chunk_size",
        "--coherence-count-chunk-size",
        type=int,
        default=DEFAULT_REFERENCE_COUNT_CHUNK_SIZE,
    )
    parser.add_argument(
        "--reference-count-max-pending",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--coherence_topic_word_workers",
        "--coherence-topic-word-workers",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--topic-word-encoder-device",
        default="auto",
        help="Encoder device for vMF/GSLDA topic words: auto, cpu, cuda, or cuda:N.",
    )
    parser.add_argument(
        "--topic-word-encode-batch-size",
        type=int,
        default=None,
        help=(
            "Optional override for the fitted encoder batch size. When omitted, "
            "use the saved training value or its stable backend default."
        ),
    )
    parser.add_argument(
        "--coherence_score_workers",
        "--coherence-score-workers",
        type=int,
        default=1,
    )
    parser.add_argument("--skip_existing", "--skip-existing", action="store_true")
    parser.add_argument(
        "--checkpoint-mode",
        choices=["auto", "off", "refresh"],
        default="auto",
    )
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    parser.add_argument(
        "--reference-count-cache-mode",
        choices=["auto", "off", "refresh"],
        default="auto",
    )
    parser.add_argument(
        "--reference-index-mode",
        choices=["off", "auto", "build", "refresh"],
        default="off",
    )
    parser.add_argument("--reference-index-root", type=Path, default=None)
    parser.add_argument(
        "--condition-failure-policy",
        choices=[
            "fail-fast",
            "exclude-condition",
            "isolate",
            "continue-and-fail",
        ],
        default="exclude-condition",
    )
    parser.add_argument("--language", type=str, default="english")
    parser.add_argument("--delimiter", type=str, default=" / ")
    parser.add_argument("--ja_replace_num", action="store_true")
    parser.add_argument("--ja_keep_num", action="store_false", dest="ja_replace_num")
    parser.set_defaults(ja_replace_num=True)
    parser.add_argument("--ja_dicdir", type=str, default=None)
    parser.add_argument("--ja_require_unidic", action="store_true")
    parser.add_argument(
        "--ja_allow_missing_unidic", action="store_false", dest="ja_require_unidic"
    )
    parser.set_defaults(ja_require_unidic=True)
    return parser.parse_args()
