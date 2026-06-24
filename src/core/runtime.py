from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class CorpusSelection:
    train_csvs: tuple[Path, ...]
    test_csvs: tuple[Path, ...]
    targets: tuple[str, ...] | None = None

    def to_model_options(self) -> dict[str, Any]:
        return {
            "train_csvs": [str(path) for path in self.train_csvs],
            "test_csvs": [str(path) for path in self.test_csvs],
            "targets": None if self.targets is None else list(self.targets),
        }


@dataclass(frozen=True)
class PreprocessRuntime:
    text_column: str
    target_column: str | None
    delimiter: str | None
    language: str
    segmenter: str
    tokenizer: str
    legacy_preprocessing: bool | None
    ja_replace_num: bool
    ja_stopwords_path: str | None
    ja_dicdir: str | None
    ja_require_unidic: bool

    def to_model_options(self) -> dict[str, Any]:
        return {
            "text_column": self.text_column,
            "target_column": self.target_column,
            "delimiter": self.delimiter,
            "language": self.language,
            "segmenter": self.segmenter,
            "tokenizer": self.tokenizer,
            "legacy_preprocessing": self.legacy_preprocessing,
            "ja_replace_num": self.ja_replace_num,
            "ja_stopwords_path": self.ja_stopwords_path,
            "ja_dicdir": self.ja_dicdir,
            "ja_require_unidic": self.ja_require_unidic,
        }


@dataclass(frozen=True)
class BaselineRuntimeContext:
    corpus: CorpusSelection
    preprocess: PreprocessRuntime
    encoder_device: str
    runtime_num_workers: int

    def to_model_options(self) -> dict[str, Any]:
        options = self.corpus.to_model_options()
        options.update(self.preprocess.to_model_options())
        options.update(
            {
                "encoder_device": self.encoder_device,
                "runtime_num_workers": int(self.runtime_num_workers),
            }
        )
        return options
