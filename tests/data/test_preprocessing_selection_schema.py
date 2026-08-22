from __future__ import annotations

from pathlib import Path

import pytest

from src.core.artifacts import load_artifact_json, save_json
from src.data.preprocessing import PreprocessedDocument, select_modelable_documents
from src.data.preprocessing_selection import (
    parse_raw_doc_indices,
    resolve_preprocessing_selection,
)

SELECTION_PATH = Path("/tmp/preprocessing_selection.json")


def _flat_payload() -> dict[str, object]:
    return {
        "raw_doc_indices": [3, 1, 2],
        "sentence_indices_by_doc": [[0], [0, 1], [2]],
        "dropped_doc_indices": [],
        "drop_reasons": {},
    }


def _combined_payload() -> dict[str, object]:
    return {
        "train": {"raw_doc_indices": [3, 1, 2]},
        "test": {"raw_doc_indices": [7, 5]},
    }


def test_resolve_flat_payload_returns_payload_itself() -> None:
    payload = _flat_payload()
    resolved = resolve_preprocessing_selection(
        payload, split="train", selection_path=SELECTION_PATH
    )
    assert resolved is payload


@pytest.mark.parametrize(
    ("split", "expected"),
    [("train", [3, 1, 2]), ("test", [7, 5])],
)
def test_resolve_combined_payload_extracts_requested_split(
    split: str, expected: list[int]
) -> None:
    resolved = resolve_preprocessing_selection(
        _combined_payload(), split=split, selection_path=SELECTION_PATH
    )
    assert resolved["raw_doc_indices"] == expected


def test_resolve_rejects_unsupported_split() -> None:
    with pytest.raises(ValueError, match="Unsupported preprocessing selection split"):
        resolve_preprocessing_selection(
            _flat_payload(), split="validation", selection_path=SELECTION_PATH
        )


def test_resolve_rejects_non_mapping_payload() -> None:
    with pytest.raises(ValueError, match="expected a JSON object"):
        resolve_preprocessing_selection(
            [1, 2, 3], split="train", selection_path=SELECTION_PATH
        )


def test_resolve_combined_payload_without_requested_split_fails() -> None:
    payload = {"train": {"raw_doc_indices": [1]}}
    with pytest.raises(ValueError, match="no 'test' split"):
        resolve_preprocessing_selection(
            payload, split="test", selection_path=SELECTION_PATH
        )


def test_resolve_rejects_ambiguous_payload_with_both_markers() -> None:
    payload = {"raw_doc_indices": [1], "train": {"raw_doc_indices": [2]}}
    with pytest.raises(ValueError, match="Ambiguous preprocessing selection"):
        resolve_preprocessing_selection(
            payload, split="train", selection_path=SELECTION_PATH
        )


def test_resolve_rejects_payload_without_any_marker() -> None:
    with pytest.raises(ValueError, match="Unrecognized preprocessing selection"):
        resolve_preprocessing_selection(
            {"other": 1}, split="train", selection_path=SELECTION_PATH
        )


def test_resolve_rejects_split_section_missing_raw_doc_indices() -> None:
    payload = {"train": {"sentence_indices_by_doc": []}}
    with pytest.raises(ValueError, match="missing 'raw_doc_indices'"):
        resolve_preprocessing_selection(
            payload, split="train", selection_path=SELECTION_PATH
        )


def test_parse_preserves_stored_order() -> None:
    raw_ids = parse_raw_doc_indices(
        _flat_payload(), selection_path=SELECTION_PATH, split="train"
    )
    assert raw_ids == [3, 1, 2]


def test_parse_rejects_non_list_raw_doc_indices() -> None:
    with pytest.raises(ValueError, match="must be a list"):
        parse_raw_doc_indices(
            {"raw_doc_indices": {"0": 1}},
            selection_path=SELECTION_PATH,
            split="train",
        )


@pytest.mark.parametrize("bad_value", ["x", None, True, 1.5, [1]])
def test_parse_rejects_values_not_convertible_to_int(bad_value: object) -> None:
    with pytest.raises(ValueError, match="Invalid raw document ID"):
        parse_raw_doc_indices(
            {"raw_doc_indices": [0, bad_value]},
            selection_path=SELECTION_PATH,
            split="train",
        )


def test_parse_accepts_integral_floats_and_numeric_strings() -> None:
    raw_ids = parse_raw_doc_indices(
        {"raw_doc_indices": [2.0, "5", 9]},
        selection_path=SELECTION_PATH,
        split="train",
    )
    assert raw_ids == [2, 5, 9]


def test_parse_rejects_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="Duplicate raw document ID 4 at position 2"):
        parse_raw_doc_indices(
            {"raw_doc_indices": [4, 1, 4]},
            selection_path=SELECTION_PATH,
            split="train",
        )


def test_parse_rejects_sentence_indices_count_mismatch() -> None:
    payload = {"raw_doc_indices": [1, 2], "sentence_indices_by_doc": [[0]]}
    with pytest.raises(ValueError, match="1 'sentence_indices_by_doc' entries"):
        parse_raw_doc_indices(payload, selection_path=SELECTION_PATH, split="train")


def _document(text: str) -> PreprocessedDocument:
    return PreprocessedDocument(
        raw_text=text,
        sentences_raw=[text],
        sentences_tokenized=[text.split()],
        sentences_joined=[text],
        document_tokens=text.split(),
    )


def test_contract_vmf_combined_save_form_round_trips(tmp_path: Path) -> None:
    # Mirror the vMF save path in src/models/registry.py: one combined JSON
    # built from SelectedCorpus.to_json_dict() per split.
    train_selection = select_modelable_documents(
        [_document("alpha beta"), _document("gamma delta")],
        raw_doc_indices=[10, 11],
    )
    test_selection = select_modelable_documents(
        [_document("epsilon zeta")], raw_doc_indices=[20]
    )
    selection_path = tmp_path / "preprocessing_selection.json"
    save_json(
        {
            "train": train_selection.to_json_dict(),
            "test": test_selection.to_json_dict(),
        },
        selection_path,
    )

    payload = load_artifact_json(selection_path)
    for split, expected in (("train", [10, 11]), ("test", [20])):
        resolved = resolve_preprocessing_selection(
            payload, split=split, selection_path=selection_path
        )
        raw_ids = parse_raw_doc_indices(
            resolved, selection_path=selection_path, split=split
        )
        assert raw_ids == expected


def test_contract_baseline_flat_save_form_round_trips(tmp_path: Path) -> None:
    # Mirror the baseline save path (for example src/baselines/models/bleilda.py):
    # one flat SelectedCorpus.to_json_dict() JSON per split directory.
    selection = select_modelable_documents(
        [_document("alpha beta"), _document("gamma delta")],
        raw_doc_indices=[4, 2],
    )
    selection_path = tmp_path / "preprocessing_selection.json"
    save_json(selection.to_json_dict(), selection_path)

    payload = load_artifact_json(selection_path)
    for split in ("train", "test"):
        resolved = resolve_preprocessing_selection(
            payload, split=split, selection_path=selection_path
        )
        raw_ids = parse_raw_doc_indices(
            resolved, selection_path=selection_path, split=split
        )
        assert raw_ids == [4, 2]
