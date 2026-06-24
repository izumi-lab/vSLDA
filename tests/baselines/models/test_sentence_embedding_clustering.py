from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.baselines.models.sentence_embedding_clustering import (
    SentenceEmbeddingClusteringTrainResult,
    infer_gaussian_kmeans,
    infer_gaussian_mixture,
    infer_movmf,
    infer_spherical_kmeans,
    persist_sentence_embedding_clustering_run,
    train_gaussian_kmeans,
    train_gaussian_mixture,
    train_movmf,
    train_spherical_kmeans,
)
from src.baselines.params import (
    GaussianKMeansParams,
    GaussianMixtureParams,
    MovMFParams,
    SphericalKMeansParams,
)


class FakeSentenceEncoder:
    init_kwargs: list[dict[str, object]] = []

    def __init__(self, *args, **kwargs) -> None:
        self.init_kwargs.append(dict(kwargs))
        self._vectors = {
            "alpha": np.asarray([1.0, 0.0], dtype=np.float64),
            "alpha two": np.asarray([0.95, 0.05], dtype=np.float64),
            "beta": np.asarray([0.0, 1.0], dtype=np.float64),
            "beta two": np.asarray([0.05, 0.95], dtype=np.float64),
            "alpha test": np.asarray([0.9, 0.1], dtype=np.float64),
            "beta test": np.asarray([0.1, 0.9], dtype=np.float64),
        }

    def encode(self, sentences, **kwargs):
        _ = kwargs
        return np.asarray([self._vectors[str(sentence)] for sentence in sentences])


@pytest.fixture()
def csv_paths(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    train.write_text(
        "data,target_str\n" "alpha / alpha two,science\n" "beta / beta two,sports\n",
        encoding="utf-8",
    )
    test.write_text(
        "data,target_str\n" "alpha test,science\n" "beta test,sports\n",
        encoding="utf-8",
    )
    return train, test


def _common_kwargs(csv_paths: tuple[Path, Path], params):
    train, _test = csv_paths
    return {
        "train_csvs": [str(train)],
        "targets": None,
        "text_column": "data",
        "target_column": "target_str",
        "delimiter": " / ",
        "language": "english",
        "segmenter": "delimiter",
        "tokenizer": "default",
        "ja_replace_num": True,
        "ja_stopwords_path": None,
        "ja_dicdir": None,
        "ja_require_unidic": True,
        "num_topics": 2,
        "encoder_device": "cpu",
        "params": params,
        "train_dir": Path("unused"),
        "use_legacy": False,
    }


def _infer_kwargs(csv_paths: tuple[Path, Path], train_result, params):
    _train, test = csv_paths
    return {
        "train_result": train_result,
        "test_csvs": [str(test)],
        "targets": None,
        "text_column": "data",
        "target_column": "target_str",
        "delimiter": " / ",
        "language": "english",
        "segmenter": "delimiter",
        "tokenizer": "default",
        "ja_replace_num": True,
        "ja_stopwords_path": None,
        "ja_dicdir": None,
        "ja_require_unidic": True,
        "num_topics": 2,
        "params": params,
        "use_legacy": False,
    }


def _assert_distribution(arr: np.ndarray, shape: tuple[int, int]) -> None:
    assert arr.shape == shape
    np.testing.assert_allclose(arr.sum(axis=1), np.ones(shape[0]))
    assert np.all(np.isfinite(arr))


@pytest.mark.parametrize(
    ("train_fn", "infer_fn", "params"),
    [
        (
            train_spherical_kmeans,
            infer_spherical_kmeans,
            SphericalKMeansParams(random_state=0, n_init=2, max_iter=20),
        ),
        (
            train_gaussian_kmeans,
            infer_gaussian_kmeans,
            GaussianKMeansParams(random_state=0, n_init=2, max_iter=20),
        ),
        (
            train_movmf,
            infer_movmf,
            MovMFParams(random_state=0, n_init=2, max_iter=20),
        ),
        (
            train_gaussian_mixture,
            infer_gaussian_mixture,
            GaussianMixtureParams(random_state=0, n_init=2, max_iter=20),
        ),
    ],
)
def test_sentence_embedding_clustering_methods_produce_doc_topics(
    monkeypatch: pytest.MonkeyPatch,
    csv_paths: tuple[Path, Path],
    train_fn,
    infer_fn,
    params,
) -> None:
    monkeypatch.setattr(
        "src.baselines.models.sentence_embedding_clustering.SentenceEncoder",
        FakeSentenceEncoder,
    )

    train_result = train_fn(**_common_kwargs(csv_paths, params))
    infer_result = infer_fn(**_infer_kwargs(csv_paths, train_result, params))

    assert isinstance(train_result, SentenceEmbeddingClusteringTrainResult)
    _assert_distribution(train_result.train_doc_topic, (2, 2))
    _assert_distribution(infer_result.test_doc_topic, (2, 2))
    assert len(train_result.train_sentence_topic_soft) == 2
    assert len(infer_result.test_sentence_topic_soft) == 2


@pytest.mark.parametrize(
    ("train_fn", "params"),
    [
        (
            train_gaussian_kmeans,
            GaussianKMeansParams(
                random_state=0,
                n_init=2,
                max_iter=20,
                strip_terminal_normalize=False,
            ),
        ),
        (
            train_gaussian_mixture,
            GaussianMixtureParams(
                random_state=0,
                n_init=2,
                max_iter=20,
                strip_terminal_normalize=False,
            ),
        ),
    ],
)
def test_gaussian_clustering_passes_terminal_normalize_setting_to_encoder(
    monkeypatch: pytest.MonkeyPatch,
    csv_paths: tuple[Path, Path],
    train_fn,
    params,
) -> None:
    FakeSentenceEncoder.init_kwargs = []
    monkeypatch.setattr(
        "src.baselines.models.sentence_embedding_clustering.SentenceEncoder",
        FakeSentenceEncoder,
    )

    train_fn(**_common_kwargs(csv_paths, params))

    assert FakeSentenceEncoder.init_kwargs[0]["strip_terminal_normalize"] is False


def test_spherical_kmeans_persists_expected_artifacts(
    monkeypatch: pytest.MonkeyPatch,
    csv_paths: tuple[Path, Path],
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "src.baselines.models.sentence_embedding_clustering.SentenceEncoder",
        FakeSentenceEncoder,
    )
    params = SphericalKMeansParams(random_state=0, n_init=2, max_iter=20)
    train_result = train_spherical_kmeans(**_common_kwargs(csv_paths, params))
    infer_result = infer_spherical_kmeans(
        **_infer_kwargs(csv_paths, train_result, params)
    )

    artifacts = persist_sentence_embedding_clustering_run(
        train_result=train_result,
        infer_result=infer_result,
        train_dir=tmp_path / "params",
        infer_dir=tmp_path / "infer",
        category="all",
    )

    assert artifacts.train_path.name == "all.pkl"
    assert artifacts.infer_path.name == "all.pkl"
    assert artifacts.extras["model_state"].name == "model_state.pkl"
    assert artifacts.extras["train_sentence_topic_soft"].exists()
    assert artifacts.extras["test_sentence_topic_assignments"].exists()
