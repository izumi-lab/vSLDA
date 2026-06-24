from __future__ import annotations

from pathlib import Path

import pytest

from src.core.errors import MissingDatasetError
from src.data.corpus import load_corpus
from src.data.preprocessing import preprocess_documents
from src.data.text_processing import (
    normalize_segmenter_name,
    normalize_tokenizer_name,
    split_sentences,
    tokenize_documents,
)
from src.utils import english_tokenizer as english_tokenizer_module


def test_split_sentences_supports_configured_delimiter() -> None:
    sentences = split_sentences(
        "alpha / beta / gamma",
        delimiter=" / ",
        segmenter="delimiter",
    )

    assert sentences == ["alpha", "beta", "gamma"]


def test_tokenize_documents_uses_simple_tokenizer_for_english() -> None:
    documents = tokenize_documents(
        ["Graph-based methods improve topic coherence."],
        language="english",
        tokenizer="default",
    )

    assert documents == [["graph", "base", "method", "improve", "topic", "coherence"]]


def test_tokenize_documents_lemmatizes_english_inflections_and_normalizes_numbers() -> (
    None
):
    documents = tokenize_documents(
        ["Running runners ran 123 tests."],
        language="english",
        tokenizer="default",
    )

    assert documents == [["run", "runner", "run", "<NUM>", "test"]]


def test_preprocess_documents_builds_shared_raw_and_token_views() -> None:
    corpus = preprocess_documents(
        ["Alpha beta. Gamma delta."],
        language="english",
        segmenter="pysbd",
        tokenizer="simple",
    )

    assert len(corpus.documents) == 1
    document = corpus.documents[0]
    assert document.sentences_raw == ["Alpha beta.", "Gamma delta."]
    assert document.sentences_tokenized == [["alpha", "beta"], ["gamma", "delta"]]
    assert document.sentences_joined == ["alpha beta", "gamma delta"]
    assert document.document_tokens == ["alpha", "beta", "gamma", "delta"]


def test_load_corpus_respects_delimiter_segmenter(tmp_path: Path) -> None:
    csv_path = tmp_path / "train.csv"
    csv_path.write_text(
        "data,target_str\nalpha / beta,science\nskip me,sports\n",
        encoding="utf-8",
    )

    corpus = load_corpus(
        csv_path,
        delimiter=" / ",
        segmenter="delimiter",
        target_filter=["science"],
    )

    assert corpus == [["alpha", "beta"]]


def test_tokenize_documents_keeps_japanese_particles_and_auxiliaries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Node:
        def __init__(
            self, surface: str, feature: str, next_node: "_Node | None"
        ) -> None:
            self.surface = surface
            self.feature = feature
            self.next = next_node

    class _Tagger:
        def parse(self, _text: str) -> str:
            return ""

        def parseToNode(self, _text: str) -> _Node:
            eos = _Node("", "", None)
            node4 = _Node("。", "記号,句点,*,*,*,*,。,。,。", eos)
            node3 = _Node("た", "助動詞,*,*,*,特殊・タ,基本形,た,タ,タ", node4)
            node2 = _Node(
                "東京", "名詞,固有名詞,一般,*,*,*,東京,トウキョウ,トーキョー", node3
            )
            node1 = _Node("へ", "助詞,格助詞,一般,*,*,*,へ,ヘ,エ", node2)
            return _Node(
                "行った",
                "動詞,自立,*,*,五段・ラ行,連用タ接続,行く,イッタ,イッタ",
                node1,
            )

    monkeypatch.setattr(
        "src.utils.japanese_tokenizer._create_mecab_tagger",
        lambda _dicdir, _require_unidic: _Tagger(),
    )

    documents = tokenize_documents(
        ["行ったへ東京た。"],
        language="ja",
        tokenizer="default",
        ja_require_unidic=False,
    )

    assert documents == [["行く", "へ", "東京", "た"]]


def test_load_corpus_raises_missing_dataset_error_for_unknown_path(
    tmp_path: Path,
) -> None:
    missing_csv = tmp_path / "missing.csv"

    with pytest.raises(MissingDatasetError) as exc_info:
        load_corpus(missing_csv)

    assert str(missing_csv) in str(exc_info.value)


def test_normalize_tokenizer_name_rejects_simple_for_japanese() -> None:
    with pytest.raises(ValueError):
        normalize_tokenizer_name("ja", "simple")


def test_tokenize_documents_raises_explicit_error_without_wordnet(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    english_tokenizer_module._get_wordnet_lemmatizer.cache_clear()
    monkeypatch.setattr(
        "src.utils.english_tokenizer.wordnet.ensure_loaded",
        lambda: (_ for _ in ()).throw(LookupError("missing wordnet")),
    )

    with pytest.raises(RuntimeError, match="wordnet corpus is required"):
        tokenize_documents(
            ["running"],
            language="english",
            tokenizer="default",
        )
    english_tokenizer_module._get_wordnet_lemmatizer.cache_clear()


def test_normalize_segmenter_name_rejects_unknown_value() -> None:
    with pytest.raises(ValueError):
        normalize_segmenter_name("unknown")
