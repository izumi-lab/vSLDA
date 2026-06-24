from __future__ import annotations

from src.data.preprocessing import (
    PreprocessedDocument,
    filter_selected_corpus_by_vocabulary,
    select_modelable_documents,
)


def test_select_modelable_documents_drops_empty_token_sentences_and_docs() -> None:
    documents = [
        PreprocessedDocument(
            raw_text="Alpha beta / --",
            sentences_raw=["Alpha beta", "--"],
            sentences_tokenized=[["alpha", "beta"], []],
            sentences_joined=["alpha beta", ""],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="...",
            sentences_raw=["..."],
            sentences_tokenized=[[]],
            sentences_joined=[""],
            document_tokens=[],
        ),
    ]

    selection = select_modelable_documents(documents, raw_doc_indices=[10, 11])

    assert selection.raw_doc_indices == [10]
    assert selection.sentence_indices_by_doc == [[0]]
    assert selection.dropped_doc_indices == [11]
    assert selection.drop_reasons == {11: "no_tokenized_sentences"}
    assert selection.documents[0].sentences_raw == ["Alpha beta"]
    assert selection.documents[0].sentences_tokenized == [["alpha", "beta"]]
    assert selection.documents[0].document_tokens == ["alpha", "beta"]


def test_filter_selected_corpus_by_vocabulary_preserves_original_mapping() -> None:
    documents = [
        PreprocessedDocument(
            raw_text="Alpha beta / Gamma delta",
            sentences_raw=["Alpha beta", "Gamma delta"],
            sentences_tokenized=[["alpha", "beta"], ["gamma", "delta"]],
            sentences_joined=["alpha beta", "gamma delta"],
            document_tokens=["alpha", "beta", "gamma", "delta"],
        ),
    ]
    selection = select_modelable_documents(documents, raw_doc_indices=[5])

    filtered = filter_selected_corpus_by_vocabulary(selection, {"gamma"})

    assert filtered.raw_doc_indices == [5]
    assert filtered.sentence_indices_by_doc == [[1]]
    assert filtered.documents[0].sentences_raw == ["Gamma delta"]
    assert filtered.documents[0].sentences_tokenized == [["gamma"]]
    assert filtered.documents[0].document_tokens == ["gamma"]
