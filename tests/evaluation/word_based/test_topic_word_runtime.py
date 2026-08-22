import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from gensim.corpora import Dictionary

from src.baselines.models.etm import EtmModel
from src.core.artifacts import load_artifact_json, save_json, save_pickle
from src.data.preprocessing import PreprocessedDocument
from src.evaluation.word_based import topic_word_runtime as runtime
from src.evaluation.word_based.metrics import _persist_runtime_topic_word_artifacts
from src.evaluation.word_based.topic_assignment import (
    CollapsedFoldInConfig,
    DegenerateTopicError,
)
from src.evaluation.word_based.topic_words import TopicWordsResult


def _signature_checked(target, result, *, record: list | None = None):
    """Fake ``target`` while still rejecting keywords it does not accept.

    A ``lambda **_kwargs`` stub swallows misspelled or removed keyword
    arguments, which hides real ``TypeError``s from the callers under test.
    Binding against the genuine signature keeps the stub honest.
    """

    signature = inspect.signature(target)

    def _fake(**kwargs):
        signature.bind(**kwargs)
        if record is not None:
            record.append(kwargs)
        return result

    return _fake


@pytest.mark.parametrize(
    ("model", "loader_name"),
    [
        ("bleilda", "_load_bleilda"),
        ("vmf", "_load_vmf"),
        ("sentlda", "_load_sentlda"),
        ("sentence_gaussianlda", "_load_sentence_gaussian"),
        ("mvtm", "_word_vector_condition"),
        ("gaussianlda", "_word_vector_condition"),
        ("etm", "_load_etm"),
        ("ctm", "_load_ctm"),
    ],
)
def test_runtime_resolver_routes_all_supported_models(
    monkeypatch, tmp_path: Path, model: str, loader_name: str
) -> None:
    sentinel = object()
    calls: list[dict[str, object]] = []
    resolver_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime,
        "resolve_vmf_experiment_dir",
        _signature_checked(
            runtime.resolve_vmf_experiment_dir, tmp_path, record=resolver_calls
        ),
    )
    monkeypatch.setattr(
        runtime,
        "resolve_baseline_condition_dir",
        _signature_checked(
            runtime.resolve_baseline_condition_dir, tmp_path, record=resolver_calls
        ),
    )
    monkeypatch.setattr(
        runtime,
        "audit_model_artifacts",
        _signature_checked(
            runtime.audit_model_artifacts,
            SimpleNamespace(ok=True, missing_groups=()),
        ),
    )
    monkeypatch.setattr(
        runtime,
        loader_name,
        lambda **kwargs: calls.append(kwargs) or sentinel,
    )

    result = runtime.resolve_runtime_topic_words(
        model=model,
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="all",
        split="train",
        dictionary=Dictionary([["alpha"]]),
        topn=5,
        embedding_variant="mpnet",
        posterior_config=CollapsedFoldInConfig(),
        etm_theta_samples=3,
        etm_posterior_seed=7,
    )

    assert result is sentinel
    assert calls and calls[0]["condition_dir"] == tmp_path
    # vMF experiment dirs are not keyed on the prior-scale variant; baseline
    # dirs are, and must be told so explicitly even when there is no override.
    assert resolver_calls
    if model == "vmf":
        assert "parameter_variant" not in resolver_calls[0]
    else:
        assert resolver_calls[0]["parameter_variant"] is None


@pytest.mark.parametrize(
    ("model", "expected_variant"),
    [
        ("gaussianlda", "psi0-3"),
        ("sentence_gaussianlda", "psi0-3"),
        ("mvtm", None),
        ("bleilda", None),
    ],
)
def test_runtime_resolver_forwards_prior_scale_variant(
    monkeypatch, tmp_path: Path, model: str, expected_variant: str | None
) -> None:
    """--prior-scale must reach the baseline path resolver, not be dropped."""

    resolver_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        runtime,
        "resolve_baseline_condition_dir",
        _signature_checked(
            runtime.resolve_baseline_condition_dir, tmp_path, record=resolver_calls
        ),
    )
    monkeypatch.setattr(
        runtime,
        "audit_model_artifacts",
        _signature_checked(
            runtime.audit_model_artifacts,
            SimpleNamespace(ok=True, missing_groups=()),
        ),
    )
    monkeypatch.setattr(runtime, "_load_bleilda", lambda **_kwargs: object())
    monkeypatch.setattr(runtime, "_load_sentence_gaussian", lambda **_kwargs: object())
    monkeypatch.setattr(runtime, "_word_vector_condition", lambda **_kwargs: object())

    runtime.resolve_runtime_topic_words(
        model=model,
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="all",
        split="train",
        dictionary=Dictionary([["alpha"]]),
        topn=5,
        embedding_variant="mpnet",
        posterior_config=CollapsedFoldInConfig(),
        etm_theta_samples=3,
        etm_posterior_seed=7,
        prior_scale=3.0,
    )

    assert resolver_calls[0]["parameter_variant"] == expected_variant


def test_runtime_resolver_stops_on_failed_artifact_audit(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        runtime,
        "resolve_baseline_condition_dir",
        _signature_checked(runtime.resolve_baseline_condition_dir, tmp_path),
    )
    monkeypatch.setattr(
        runtime,
        "audit_model_artifacts",
        _signature_checked(
            runtime.audit_model_artifacts,
            SimpleNamespace(ok=False, missing_groups=(("params/model.gensim",),)),
        ),
    )

    with pytest.raises(FileNotFoundError, match="params/model.gensim"):
        runtime.resolve_runtime_topic_words(
            model="bleilda",
            dataset="dummy",
            data_run="default",
            iteration=0,
            num_topics=2,
            category="all",
            split="train",
            dictionary=Dictionary([["alpha"]]),
            topn=5,
            embedding_variant=None,
            posterior_config=CollapsedFoldInConfig(),
            etm_theta_samples=1,
            etm_posterior_seed=0,
        )


def test_restore_etm_model_materializes_meta_parameters_from_checkpoint() -> None:
    embeddings = np.arange(12, dtype=np.float32).reshape(4, 3) / 10.0
    source = EtmModel(
        embeddings=embeddings,
        num_topics=2,
        hidden_size=3,
        theta_act="relu",
        enc_drop=0.0,
    )
    with torch.no_grad():
        for index, parameter in enumerate(source.parameters(), start=1):
            parameter.fill_(index / 10.0)
    state = {
        name: tensor.detach().clone() for name, tensor in source.state_dict().items()
    }

    with torch.device("meta"):
        target = EtmModel(
            embeddings=embeddings,
            num_topics=2,
            hidden_size=3,
            theta_act="relu",
            enc_drop=0.0,
        )

    restored = runtime._restore_etm_model(target, state)

    assert restored.training is False
    assert all(not tensor.is_meta for tensor in restored.parameters())
    assert all(not tensor.is_meta for tensor in restored.buffers())
    assert all(tensor.device.type == "cpu" for tensor in restored.parameters())
    assert all(tensor.device.type == "cpu" for tensor in restored.buffers())
    for name, tensor in restored.state_dict().items():
        torch.testing.assert_close(tensor, state[name])

    bows = torch.tensor(
        [[0.25, 0.25, 0.25, 0.25], [0.1, 0.2, 0.3, 0.4]],
        dtype=torch.float32,
    )
    with torch.no_grad():
        source_mu, source_logsigma = source.encode(bows)
        restored_mu, restored_logsigma = restored.encode(bows)
        torch.testing.assert_close(restored_mu, source_mu)
        torch.testing.assert_close(restored_logsigma, source_logsigma)
        torch.testing.assert_close(restored.get_beta(), source.get_beta())


def test_validate_etm_checkpoint_beta_allows_cuda_cpu_numerical_drift() -> None:
    saved = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)
    checkpoint = saved + np.array([[3e-5, -1.5e-5, -1.5e-5]], dtype=np.float64)

    runtime._validate_etm_checkpoint_beta(
        saved_beta=saved,
        checkpoint_beta=checkpoint,
    )


def test_validate_etm_checkpoint_beta_rejects_material_difference() -> None:
    saved = np.array([[0.7, 0.2, 0.1]], dtype=np.float64)
    checkpoint = np.array([[0.6, 0.3, 0.1]], dtype=np.float64)

    with pytest.raises(ValueError, match="checkpoint beta does not match"):
        runtime._validate_etm_checkpoint_beta(
            saved_beta=saved,
            checkpoint_beta=checkpoint,
        )


def test_degenerate_topic_precheck_reports_structured_diagnostics() -> None:
    counts = np.array([[2.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    with pytest.raises(DegenerateTopicError) as caught:
        runtime._raise_for_degenerate_topics(
            counts,
            vocabulary=["<NUM>", "alpha", "beta"],
            required_topn=2,
        )

    error = caught.value
    assert error.topic_id == 0
    assert error.eligible_word_count == 1
    assert error.required_topn == 2
    assert error.eligible_words == ["<NUM>"]
    assert error.reason == "insufficient_positive_expected_counts"
    assert error.eligible_word_counts.tolist() == [1, 3]

    runtime._raise_for_degenerate_topics(
        counts,
        vocabulary=["<NUM>", "alpha", "beta"],
        required_topn=2,
        allow_zero_joint_counts=True,
    )


def _artifact_runtime(
    *, protocol: str, with_posterior: bool
) -> runtime.RuntimeTopicWords:
    words = [[("alpha", 0.8)], [("beta", 0.7)]]
    return runtime.RuntimeTopicWords(
        evaluation=TopicWordsResult(
            topic_words=words,
            topic_word_source="evaluation-source",
            score_mode="topic_word_probability",
            score_definition="test",
        ),
        display_topic_words=words,
        display_source="display-source",
        display_score_mode="word_topic_npmi",
        protocol=protocol,
        condition_dir=Path("/tmp/source-condition"),
        source_condition_fingerprint="source-condition",
        vocabulary_fingerprint="vocabulary",
        corpus_fingerprint="corpus",
        coverage={"token_coverage": 1.0},
        posterior_mean_by_doc=(
            [np.asarray([[0.75, 0.25]])] if with_posterior else None
        ),
        posterior_metadata={"samples": 1} if with_posterior else None,
        expected_counts=np.ones((2, 2)) if with_posterior else None,
    )


@pytest.mark.parametrize(
    ("model", "posterior_name"),
    [
        ("bleilda", "topic_assignment_train_posterior_mean"),
        ("etm", "etm_token_topic_train_posterior_mean"),
    ],
)
def test_runtime_artifacts_use_protocol_specific_iteration_names(
    tmp_path: Path, model: str, posterior_name: str
) -> None:
    result = _artifact_runtime(protocol="test", with_posterior=True)
    evaluation_path, display_path, iteration_artifacts, metadata = (
        _persist_runtime_topic_word_artifacts(
            out_dir=tmp_path,
            model=model,
            split="train",
            runtimes_by_iteration=[(3, result)],
            common_meta={"condition_fingerprint": "evaluation-condition"},
        )
    )

    iteration_dir = tmp_path / "iterations" / "iteration_3"
    assert evaluation_path.name == "topic_words_evaluation_topk.json"
    assert display_path.name == "topic_words_display_topk.json"
    assert (iteration_dir / f"{posterior_name}.pkl").exists()
    assert (iteration_dir / f"{posterior_name}.json").exists()
    assert (iteration_dir / "topic_word_train_expected_counts.pkl").exists()
    assert (iteration_dir / "topic_word_train_coverage.json").exists()
    assert iteration_artifacts["3"]["posterior_mean_pickle"].endswith(".pkl")
    assert metadata["evaluation_vocabulary_fingerprint"] == "vocabulary"
    display_meta = load_artifact_json(display_path)["_meta"]
    assert display_meta["topic_word_role"] == "display"
    assert display_meta["source"] == "display-source"
    evaluation_meta = load_artifact_json(evaluation_path)["_meta"]
    assert evaluation_meta["topic_word_role"] == "evaluation"
    assert evaluation_meta["source"] == "evaluation-source"

    # The JSON file is a metadata-only sidecar; arrays live in the pickle.
    sidecar = load_artifact_json(iteration_dir / f"{posterior_name}.json")
    assert "posterior_mean_by_doc" not in sidecar
    assert sidecar["iteration"] == 3
    assert sidecar["samples"] == 1


def test_runtime_artifacts_promote_npmi_words_to_evaluation(tmp_path: Path) -> None:
    base = _artifact_runtime(protocol="test", with_posterior=False)
    result = runtime.RuntimeTopicWords(
        evaluation=TopicWordsResult(
            topic_words=[[("probability", 0.9)], [("mass", 0.8)]],
            topic_word_source="probability-source",
            score_mode="topic_word_probability",
            score_definition="probability",
        ),
        display_topic_words=[[("exclusive", 0.7)], [("specific", 0.6)]],
        display_source="npmi-source",
        display_score_mode="word_topic_npmi",
        protocol=base.protocol,
        condition_dir=base.condition_dir,
        source_condition_fingerprint=base.source_condition_fingerprint,
        vocabulary_fingerprint=base.vocabulary_fingerprint,
        corpus_fingerprint=base.corpus_fingerprint,
        coverage=base.coverage,
    )

    evaluation_path, _display_path, _artifacts, metadata = (
        _persist_runtime_topic_word_artifacts(
            out_dir=tmp_path,
            model="vmf",
            split="train",
            runtimes_by_iteration=[(0, result)],
            common_meta={"condition_fingerprint": "evaluation-condition"},
            evaluation_score_mode="word_topic_npmi",
        )
    )

    evaluation = load_artifact_json(evaluation_path)
    assert evaluation["_meta"]["score_mode"] == "word_topic_npmi"
    assert (
        evaluation["results"]["per_iteration"][0]["topics"][0]["words"][0]["word"]
        == "exclusive"
    )
    probability = load_artifact_json(
        tmp_path / metadata["topic_words_probability_topk"]
    )
    assert probability["_meta"]["score_mode"] == "topic_word_probability"
    assert (
        probability["results"]["per_iteration"][0]["topics"][0]["words"][0]["word"]
        == "probability"
    )


def test_vmf_preprocessor_rejects_missing_whitening_matrix(tmp_path: Path) -> None:
    save_pickle(np.zeros(3), tmp_path / "embedding_transform_mean.pkl")
    with pytest.raises(FileNotFoundError, match="whitening"):
        runtime._vmf_embedding_preprocessor(
            tmp_path, {"pre_normalize_transform": "whitening"}
        )


def test_vmf_preprocessor_rejects_unexpected_artifacts_for_none(
    tmp_path: Path,
) -> None:
    save_pickle(np.zeros(3), tmp_path / "embedding_transform_mean.pkl")
    with pytest.raises(ValueError, match="transform artifact"):
        runtime._vmf_embedding_preprocessor(
            tmp_path, {"pre_normalize_transform": "none"}
        )


def test_vmf_preprocessor_applies_mean_center_transform(tmp_path: Path) -> None:
    save_pickle(np.array([1.0, -1.0]), tmp_path / "embedding_transform_mean.pkl")
    preprocessor = runtime._vmf_embedding_preprocessor(
        tmp_path, {"pre_normalize_transform": "mean_center"}
    )
    np.testing.assert_allclose(
        preprocessor.transform(np.array([[2.0, 2.0]])), [[1.0, 3.0]]
    )


def test_sentence_alignment_mismatch_is_reported_with_document_index() -> None:
    document = SimpleNamespace(
        sentences_joined=["a b", "c d"],
        sentences_tokenized=[["a", "b"]],
    )
    with pytest.raises(ValueError, match="document 0 has 2 joined sentences"):
        runtime._validate_sentence_alignment([document], model="vmf")


def test_ctm_loader_reports_partial_coverage_without_inf_words(
    monkeypatch, tmp_path: Path
) -> None:
    documents = [
        SimpleNamespace(document_tokens=["alpha", "beta", "gamma", "alpha"]),
    ]
    monkeypatch.setattr(
        runtime,
        "_load_documents_and_ids",
        lambda **_kwargs: (documents, [0]),
    )
    monkeypatch.setattr(
        runtime,
        "load_ctm_decoder_scores",
        lambda **_kwargs: (
            np.array([[0.6, 0.4], [0.3, 0.7]]),
            ["alpha", "beta"],
        ),
    )
    monkeypatch.setattr(runtime, "_condition_fingerprint", lambda _dir: "cond")
    monkeypatch.setattr(
        runtime,
        "load_artifact_pickle",
        lambda _path: np.array([[0.5, 0.5]]),
    )

    result = runtime._load_ctm(
        condition_dir=tmp_path,
        dataset="dummy",
        data_run="default",
        iteration=0,
        num_topics=2,
        category="all",
        dictionary=Dictionary([["alpha"], ["beta"], ["gamma"]]),
        topn=2,
        embedding_variant=None,
        split="train",
    )

    # "gamma" is outside the CTM vocabulary: coverage must reflect that and
    # uncovered words must never appear in the rankings.
    assert result.coverage["type_coverage"] == pytest.approx(2 / 3)
    assert result.coverage["token_coverage"] == pytest.approx(3 / 4)
    ranked_words = {
        word for topic in result.evaluation.topic_words for word, _ in topic
    }
    assert ranked_words == {"alpha", "beta"}
    assert all(
        np.isfinite(score)
        for topic in result.evaluation.topic_words
        for _word, score in topic
    )
    assert result.display_score_mode == "word_topic_npmi"
    assert result.expected_counts is not None
    np.testing.assert_allclose(
        result.expected_counts,
        np.array(
            [
                [4.0 / 3.0, 4.0 / 11.0, 0.0],
                [2.0 / 3.0, 7.0 / 11.0, 0.0],
            ]
        ),
    )


def _preprocessed_document(text: str) -> PreprocessedDocument:
    tokens = text.split()
    return PreprocessedDocument(
        raw_text=text,
        sentences_raw=[text],
        sentences_tokenized=[tokens],
        sentences_joined=[text],
        document_tokens=tokens,
    )


def _write_vmf_split_artifacts(
    condition_dir: Path,
    *,
    train_texts: list[str],
    test_texts: list[str],
    selection_payload: dict[str, object],
) -> None:
    save_pickle(
        [_preprocessed_document(text) for text in train_texts],
        condition_dir / "train_preprocessed.pkl",
    )
    save_pickle(
        [_preprocessed_document(text) for text in test_texts],
        condition_dir / "test_preprocessed.pkl",
    )
    save_json(selection_payload, condition_dir / "preprocessing_selection.json")


def _write_baseline_split_artifacts(
    condition_dir: Path,
    *,
    split_dir: str,
    texts: list[str],
    selection_payload: dict[str, object],
) -> None:
    target = condition_dir / split_dir
    save_pickle(
        [_preprocessed_document(text) for text in texts],
        target / "preprocessed_corpus.pkl",
    )
    save_json(selection_payload, target / "preprocessing_selection.json")


@pytest.mark.parametrize(
    ("split", "expected_ids"),
    [("train", [10, 11, 12]), ("test", [20, 21])],
)
def test_load_documents_and_ids_reads_vmf_combined_selection(
    tmp_path: Path, split: str, expected_ids: list[int]
) -> None:
    _write_vmf_split_artifacts(
        tmp_path,
        train_texts=["alpha beta", "gamma delta", "epsilon zeta"],
        test_texts=["eta theta", "iota kappa"],
        selection_payload={
            "train": {"raw_doc_indices": [10, 11, 12]},
            "test": {"raw_doc_indices": [20, 21]},
        },
    )

    documents, raw_ids = runtime._load_documents_and_ids(
        model="vmf", condition_dir=tmp_path, split=split
    )

    assert raw_ids == expected_ids
    assert len(documents) == len(expected_ids)


def test_load_documents_and_ids_reads_baseline_flat_selection(
    tmp_path: Path,
) -> None:
    _write_baseline_split_artifacts(
        tmp_path,
        split_dir="params",
        texts=["alpha beta", "gamma delta"],
        selection_payload={"raw_doc_indices": [5, 3]},
    )

    documents, raw_ids = runtime._load_documents_and_ids(
        model="bleilda", condition_dir=tmp_path, split="train"
    )

    assert raw_ids == [5, 3]
    assert len(documents) == 2


def test_load_documents_and_ids_fails_when_requested_split_is_absent(
    tmp_path: Path,
) -> None:
    _write_vmf_split_artifacts(
        tmp_path,
        train_texts=["alpha beta"],
        test_texts=["gamma delta"],
        selection_payload={"train": {"raw_doc_indices": [0]}},
    )

    with pytest.raises(ValueError, match="no 'test' split"):
        runtime._load_documents_and_ids(
            model="vmf", condition_dir=tmp_path, split="test"
        )


def test_load_documents_and_ids_fails_on_missing_raw_doc_indices(
    tmp_path: Path,
) -> None:
    _write_baseline_split_artifacts(
        tmp_path,
        split_dir="params",
        texts=["alpha beta"],
        selection_payload={"dropped_doc_indices": []},
    )

    with pytest.raises(ValueError, match="Unrecognized preprocessing selection"):
        runtime._load_documents_and_ids(
            model="bleilda", condition_dir=tmp_path, split="train"
        )


def test_load_documents_and_ids_fails_on_non_list_raw_doc_indices(
    tmp_path: Path,
) -> None:
    _write_baseline_split_artifacts(
        tmp_path,
        split_dir="params",
        texts=["alpha beta"],
        selection_payload={"raw_doc_indices": 7},
    )

    with pytest.raises(ValueError, match="must be a list"):
        runtime._load_documents_and_ids(
            model="bleilda", condition_dir=tmp_path, split="train"
        )


def test_load_documents_and_ids_fails_on_non_integer_ids(tmp_path: Path) -> None:
    _write_baseline_split_artifacts(
        tmp_path,
        split_dir="params",
        texts=["alpha beta", "gamma delta"],
        selection_payload={"raw_doc_indices": [0, "not-a-number"]},
    )

    with pytest.raises(ValueError, match="Invalid raw document ID"):
        runtime._load_documents_and_ids(
            model="bleilda", condition_dir=tmp_path, split="train"
        )


def test_load_documents_and_ids_fails_on_duplicate_ids(tmp_path: Path) -> None:
    _write_baseline_split_artifacts(
        tmp_path,
        split_dir="params",
        texts=["alpha beta", "gamma delta"],
        selection_payload={"raw_doc_indices": [6, 6]},
    )

    with pytest.raises(ValueError, match="Duplicate raw document ID"):
        runtime._load_documents_and_ids(
            model="bleilda", condition_dir=tmp_path, split="train"
        )


def test_load_documents_and_ids_fails_on_document_count_mismatch(
    tmp_path: Path,
) -> None:
    _write_vmf_split_artifacts(
        tmp_path,
        train_texts=["alpha beta", "gamma delta"],
        test_texts=["eta theta"],
        selection_payload={
            "train": {"raw_doc_indices": [0]},
            "test": {"raw_doc_indices": [1]},
        },
    )

    with pytest.raises(
        ValueError, match=r"model=vmf split=train .*IDs=1 .*documents=2"
    ):
        runtime._load_documents_and_ids(
            model="vmf", condition_dir=tmp_path, split="train"
        )


def test_load_documents_and_ids_fails_on_sentence_indices_mismatch(
    tmp_path: Path,
) -> None:
    _write_baseline_split_artifacts(
        tmp_path,
        split_dir="params",
        texts=["alpha beta", "gamma delta"],
        selection_payload={
            "raw_doc_indices": [0, 1],
            "sentence_indices_by_doc": [[0]],
        },
    )

    with pytest.raises(ValueError, match="'sentence_indices_by_doc' entries"):
        runtime._load_documents_and_ids(
            model="bleilda", condition_dir=tmp_path, split="train"
        )


def test_ctm_runtime_artifacts_persist_counts_without_token_posterior(
    tmp_path: Path,
) -> None:
    result = replace(
        _artifact_runtime(protocol="test", with_posterior=False),
        protocol="native_ctm_decoder_with_token_responsibilities",
        expected_counts=np.ones((2, 2)),
    )
    _evaluation, _display, iteration_artifacts, _metadata = (
        _persist_runtime_topic_word_artifacts(
            out_dir=tmp_path,
            model="ctm",
            split="test",
            runtimes_by_iteration=[(0, result)],
            common_meta={"condition_fingerprint": "evaluation-condition"},
        )
    )

    assert set(iteration_artifacts["0"]) == {"coverage", "expected_counts"}
    assert not list((tmp_path / "iterations").rglob("*posterior*"))
    assert list((tmp_path / "iterations").rglob("*expected_counts*"))


class _BatchRuntimeEncoder:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return 2

    def encode(self, sentences, **kwargs):
        materialized = list(sentences)
        self.calls.append((materialized, dict(kwargs)))
        return np.arange(len(materialized) * 2, dtype=np.float32).reshape(-1, 2) + 1


def _batch_documents() -> list[PreprocessedDocument]:
    return [
        PreprocessedDocument(
            raw_text="alpha beta",
            sentences_raw=["alpha", "beta"],
            sentences_tokenized=[["alpha"], ["beta"]],
            sentences_joined=["alpha", "beta"],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="gamma",
            sentences_raw=["gamma"],
            sentences_tokenized=[["gamma"]],
            sentences_joined=["gamma"],
            document_tokens=["gamma"],
        ),
    ]


def _patch_collapsed_result(monkeypatch: pytest.MonkeyPatch, captured: dict) -> None:
    def collapsed(**kwargs):
        captured.update(kwargs)
        metadata = dict(kwargs["execution_metadata"])
        metadata.update(
            collapsed_foldin_sec=0.01,
            topic_word_ranking_sec=0.01,
            topic_words_total_sec=0.02,
        )
        return SimpleNamespace(execution_metadata=metadata)

    monkeypatch.setattr(runtime, "_collapsed_result", collapsed)


def test_vmf_runtime_encodes_all_documents_once_with_training_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    documents = _batch_documents()
    encoder = _BatchRuntimeEncoder()
    captured: dict[str, object] = {}
    encoder_config = {
        "model_name": "cl-nagoya/ruri-v3-130m",
        "backend": "sentence_transformers",
        "encode_batch_size": None,
    }
    monkeypatch.setattr(
        runtime, "_load_documents_and_ids", lambda **_kwargs: (documents, [4, 9])
    )
    monkeypatch.setattr(
        runtime, "_condition_metadata", lambda _path: {"encoder_config": encoder_config}
    )
    monkeypatch.setattr(
        runtime,
        "_encoder_from_metadata",
        lambda *_args, **_kwargs: (
            encoder,
            runtime.resolve_topic_word_encode_batch_size(
                encoder_config=encoder_config, override=None
            ),
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_vmf_embedding_preprocessor",
        lambda *_args: SimpleNamespace(transform=lambda array: array),
    )
    monkeypatch.setattr(
        runtime, "load_artifact_json", lambda _path: {"alpha": [0.1, 0.1]}
    )

    def load_pickle(path):
        name = Path(path).name
        if name == "mixture_weights.pkl":
            return np.ones((2, 1))
        if name == "component_means.pkl":
            return np.ones((2, 1, 2))
        if name == "kappa_per_topic.pkl":
            return np.ones(2)
        raise AssertionError(name)

    monkeypatch.setattr(runtime, "load_artifact_pickle", load_pickle)
    monkeypatch.setattr(
        runtime,
        "vmf_mixture_log_likelihood",
        lambda encoded, **_kwargs: np.ones((encoded.shape[0], 2)),
    )
    _patch_collapsed_result(monkeypatch, captured)

    runtime._load_vmf(
        condition_dir=tmp_path,
        split="train",
        dictionary=Dictionary([["alpha", "beta", "gamma"]]),
        config=CollapsedFoldInConfig(),
        topn=1,
        npmi_min_expected_count=None,
        encoder_device="cpu",
        encoder_device_requested="auto",
        encoder_batch_size_override=None,
    )

    assert len(encoder.calls) == 1
    assert encoder.calls[0][0] == ["alpha", "beta", "gamma"]
    assert encoder.calls[0][1]["batch_size"] == 32
    assert [rows.shape[0] for rows in captured["log_likelihood_by_doc"]] == [2, 1]


def test_sentence_gaussian_runtime_encodes_all_documents_once_with_training_batch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    documents = _batch_documents()
    encoder = _BatchRuntimeEncoder()
    captured: dict[str, object] = {}
    encoder_config = {
        "model_name": "cl-nagoya/ruri-v3-130m",
        "backend": "sentence_transformers",
        "encode_batch_size": 128,
    }
    monkeypatch.setattr(
        runtime, "_load_documents_and_ids", lambda **_kwargs: (documents, [4, 9])
    )
    monkeypatch.setattr(
        runtime, "_condition_metadata", lambda _path: {"encoder_config": encoder_config}
    )
    monkeypatch.setattr(
        runtime, "build_sentence_gaussian_encoder", lambda *_args, **_kwargs: encoder
    )
    model = SimpleNamespace(
        num_tables=2,
        alpha=np.asarray([0.1, 0.1]),
        log_multivariate_tdensity_tables=lambda row: np.asarray(
            [float(row[0]), float(row[1])]
        ),
    )
    monkeypatch.setattr(
        runtime,
        "load_sentence_gaussianlda_model",
        lambda **_kwargs: SimpleNamespace(model=model),
    )
    _patch_collapsed_result(monkeypatch, captured)

    runtime._load_sentence_gaussian(
        condition_dir=tmp_path,
        split="train",
        dictionary=Dictionary([["alpha", "beta", "gamma"]]),
        config=CollapsedFoldInConfig(),
        topn=1,
        npmi_min_expected_count=None,
        encoder_device="cpu",
        encoder_device_requested="auto",
        encoder_batch_size_override=None,
    )

    assert len(encoder.calls) == 1
    assert encoder.calls[0][0] == ["alpha", "beta", "gamma"]
    assert encoder.calls[0][1]["batch_size"] == 128
    assert [rows.shape[0] for rows in captured["log_likelihood_by_doc"]] == [2, 1]
