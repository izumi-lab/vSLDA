from __future__ import annotations

import re
from typing import Final

from src.data.text_processing import get_segmenter

DEFAULT_LANGUAGE: Final[str] = "en"


def segment_text(
    text: str,
    *,
    delimiter: str = " / ",
    language: str = DEFAULT_LANGUAGE,
) -> str:
    normalized = re.sub(r"\s+", " ", text)
    segmenter = get_segmenter(language)
    sentences = [sentence.strip() for sentence in segmenter.segment(normalized)]
    return delimiter.join(sentence for sentence in sentences if sentence)
