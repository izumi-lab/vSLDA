from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.baselines.models import gaussian_trainer, gaussianlda
from src.baselines.models.gaussian_state import GaussianTrainerState
from src.baselines.models.gaussianlda import (
    infer_gaussianlda,
    persist_gaussianlda_run,
    train_gaussianlda,
)
from src.baselines.params import GaussianLdaParams, parse_gaussianlda_params
from src.core.artifacts import load_pickle
from src.data.preprocessing import PreprocessedDocument


class _RoundTripGaussianScorer:
    def __init__(self, **_kwargs) -> None:
        pass

    def sample(self, doc, num_iterations: int):
        _ = num_iterations
        return [int(token) % 2 for token in doc]


def test_parse_gaussianlda_params_defaults_num_iterations_to_20() -> None:
    params = parse_gaussianlda_params({})

    assert params.num_iterations == 20


def test_train_gaussianlda_uses_reference_alpha_inverse_num_topics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        gaussianlda,
        "_prepare_docs",
        lambda **_kwargs: (
            [[0, 1], [1]],
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            ["alpha", "beta"],
            None,
            [],
        ),
    )

    class _FakeGaussianLDATrainer:
        def __init__(
            self,
            corpus,
            vocab_embeddings,
            vocab,
            num_tables,
            alpha,
            *,
            save_path,
        ) -> None:
            captured["corpus"] = corpus
            captured["vocab_embeddings"] = vocab_embeddings
            captured["vocab"] = vocab
            captured["num_tables"] = num_tables
            captured["alpha"] = alpha
            captured["save_path"] = save_path

        def sample(self, num_iterations: int) -> None:
            captured["num_iterations"] = num_iterations

    monkeypatch.setattr(
        gaussian_trainer,
        "GaussianLDATrainer",
        _FakeGaussianLDATrainer,
    )
    monkeypatch.setattr(
        gaussianlda,
        "snapshot_gaussian_trainer",
        lambda _trainer: SimpleNamespace(
            num_tables=5,
            alpha=captured["alpha"],
            prior_kappa=0.1,
            table_counts=np.ones(5, dtype=np.int32),
            table_means=np.zeros((5, 2), dtype=np.float64),
            log_determinants=np.zeros(5, dtype=np.float64),
            table_cholesky_ltriangular_mat=np.tile(
                np.eye(2, dtype=np.float64),
                (5, 1, 1),
            ),
            table_counts_per_doc=np.zeros((5, 2), dtype=np.int32),
        ),
    )

    result = train_gaussianlda(
        train_csvs=["train.csv"],
        targets=None,
        text_column="data",
        target_column=None,
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=5,
        params=GaussianLdaParams(num_iterations=13),
        train_dir=tmp_path,
        use_legacy=False,
    )

    assert captured["num_tables"] == 5
    assert captured["alpha"] == pytest.approx(0.2)
    assert captured["num_iterations"] == 13
    assert result.trainer_state.alpha == pytest.approx(0.2)


@pytest.mark.slow
def test_train_infer_and_persist_gaussianlda(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_docs = [
        PreprocessedDocument(
            raw_text="alpha beta",
            sentences_raw=["alpha beta"],
            sentences_tokenized=[["alpha", "beta"]],
            sentences_joined=["alpha beta"],
            document_tokens=["alpha", "beta"],
        ),
        PreprocessedDocument(
            raw_text="gamma alpha",
            sentences_raw=["gamma alpha"],
            sentences_tokenized=[["gamma", "alpha"]],
            sentences_joined=["gamma alpha"],
            document_tokens=["gamma", "alpha"],
        ),
    ]
    test_docs = [
        PreprocessedDocument(
            raw_text="alpha gamma",
            sentences_raw=["alpha gamma"],
            sentences_tokenized=[["alpha", "gamma"]],
            sentences_joined=["alpha gamma"],
            document_tokens=["alpha", "gamma"],
        ),
        PreprocessedDocument(
            raw_text="beta",
            sentences_raw=["beta"],
            sentences_tokenized=[["beta"]],
            sentences_joined=["beta"],
            document_tokens=["beta"],
        ),
    ]

    def _fake_prepare_docs(*, csv_paths, **_kwargs):
        docs = train_docs if str(csv_paths[0]) == "train.csv" else test_docs
        corpus = [[0, 1], [1, 0]] if docs is train_docs else [[0, 1], [1]]
        return (
            corpus,
            np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
            ["alpha", "beta"],
            None,
            docs,
        )

    class _FakeGaussianLDATrainer:
        def __init__(self, *_args, **_kwargs) -> None:
            self.sample_calls: list[int] = []

        def sample(self, num_iterations: int) -> None:
            self.sample_calls.append(num_iterations)

    state = GaussianTrainerState(
        average_ll=(-1.0,),
        alpha=0.5,
        num_tables=2,
        prior_kappa=0.1,
        table_counts=np.asarray([2, 2], dtype=np.int32),
        table_means=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        table_inverse_covariances=np.tile(np.eye(2), (2, 1, 1)),
        log_determinants=np.zeros(2, dtype=np.float64),
        sum_table_customers=np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        sum_squared_table_customers=np.tile(np.eye(2), (2, 1, 1)),
        table_cholesky_ltriangular_mat=np.tile(np.eye(2), (2, 1, 1)),
        table_counts_per_doc=np.asarray([[1, 1], [1, 1]], dtype=np.int32),
    )

    monkeypatch.setattr(gaussianlda, "_prepare_docs", _fake_prepare_docs)
    monkeypatch.setattr(
        gaussian_trainer,
        "GaussianLDATrainer",
        _FakeGaussianLDATrainer,
    )
    monkeypatch.setattr(
        gaussianlda, "snapshot_gaussian_trainer", lambda _trainer: state
    )
    monkeypatch.setattr(gaussianlda, "GaussianLdaScorer", _RoundTripGaussianScorer)

    params = GaussianLdaParams(word2vec="local", num_iterations=3)
    train_result = train_gaussianlda(
        train_csvs=["train.csv"],
        targets=None,
        text_column="data",
        target_column=None,
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=2,
        params=params,
        train_dir=tmp_path / "train",
        use_legacy=False,
    )
    infer_result = infer_gaussianlda(
        train_result=train_result,
        test_csvs=["test.csv"],
        targets=None,
        text_column="data",
        target_column=None,
        delimiter=" / ",
        language="english",
        segmenter="delimiter",
        tokenizer="default",
        ja_replace_num=True,
        ja_stopwords_path=None,
        ja_dicdir=None,
        ja_require_unidic=True,
        num_topics=2,
        params=params,
        use_legacy=False,
    )
    artifacts = persist_gaussianlda_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "train",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    assert train_result.train_doc_topic.shape == (2, 2)
    assert infer_result.test_doc_topic.shape == (2, 2)
    assert np.all(infer_result.test_doc_topic >= 0)
    assert artifacts.train_path.name == "table_counts_per_doc.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["params_json"].name == "params.json"
    assert load_pickle(tmp_path / "train" / "table_counts_per_doc.pkl").shape == (2, 2)
