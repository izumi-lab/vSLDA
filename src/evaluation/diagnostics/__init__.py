from __future__ import annotations

from ..word_based.label_profile import main as label_profile_main
from ..word_based.label_profile import run_word_based_label_profile
from ..word_based.topic_word_table import main as topic_word_table_main
from ..word_based.topic_word_table import run_word_based_topic_word_table
from .cross_model_pair_diagnostics import main as cross_model_pair_diagnostics_main
from .cross_model_pair_diagnostics import run_cross_model_pair_diagnostics
from .sentence_topic_inspection import main as sentence_topic_inspection_main
from .sentence_topic_inspection import run_sentence_topic_inspection
from .topic_count_diagnostics import run_topic_count_diagnostics

run_label_topic_profile = run_word_based_label_profile
run_topic_table_tex = run_word_based_topic_word_table

__all__ = [
    "run_cross_model_pair_diagnostics",
    "run_label_topic_profile",
    "run_topic_table_tex",
    "run_word_based_label_profile",
    "run_word_based_topic_word_table",
    "run_sentence_topic_inspection",
    "run_topic_count_diagnostics",
    "cross_model_pair_diagnostics_main",
    "label_profile_main",
    "sentence_topic_inspection_main",
    "topic_word_table_main",
]
