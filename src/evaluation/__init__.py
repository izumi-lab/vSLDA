from __future__ import annotations

from .classification import (
    run_classification_suite,
    run_limited_classification_suite,
    write_classification_summary,
    write_summary,
)
from .diagnostics.cross_model_pair_diagnostics import run_cross_model_pair_diagnostics
from .diagnostics.sentence_topic_inspection import run_sentence_topic_inspection
from .diagnostics.topic_count_diagnostics import run_topic_count_diagnostics
from .geometry_based.metrics import run_geometry_based_metrics
from .registry import (
    get_task,
    list_tasks,
    register_builtin_tasks,
    register_task,
    register_task_alias,
    run_task,
)
from .word_based.label_profile import run_word_based_label_profile
from .word_based.metrics import run_word_based_metrics
from .word_based.topic_word_table import run_word_based_topic_word_table

run_topic_overlap_analysis = run_geometry_based_metrics
run_topic_coherence_analysis = run_word_based_metrics
run_topic_count_perplexity_analysis = run_topic_count_diagnostics
run_label_topic_profile = run_word_based_label_profile
run_topic_table_tex = run_word_based_topic_word_table
run_vmf_vs_baseline_pair_analysis = run_cross_model_pair_diagnostics
run_inspect_slda = run_sentence_topic_inspection

__all__ = [
    "run_classification_suite",
    "run_limited_classification_suite",
    "write_classification_summary",
    "write_summary",
    "run_geometry_based_metrics",
    "run_word_based_metrics",
    "run_topic_count_diagnostics",
    "run_word_based_label_profile",
    "run_word_based_topic_word_table",
    "run_cross_model_pair_diagnostics",
    "run_sentence_topic_inspection",
    "run_topic_overlap_analysis",
    "run_topic_coherence_analysis",
    "run_topic_count_perplexity_analysis",
    "run_label_topic_profile",
    "run_topic_table_tex",
    "run_vmf_vs_baseline_pair_analysis",
    "run_inspect_slda",
    "register_builtin_tasks",
    "register_task",
    "register_task_alias",
    "list_tasks",
    "get_task",
    "run_task",
]
