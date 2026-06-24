from __future__ import annotations

import numpy as np

from src.evaluation.classification.classifier_registry import (
    CLASSIFIER_REGISTRY,
    fit_classifiers,
    get_classifier_specs,
)
from src.evaluation.classification.metrics import (
    build_feature_importance_payload,
    build_metrics_payload,
)


class _DummyClassifier:
    def __init__(self, predictions: np.ndarray, coef: np.ndarray | None = None) -> None:
        self._predictions = predictions
        self.coef_ = coef

    def fit(self, _train_x: np.ndarray, _train_y: list[int]) -> "_DummyClassifier":
        return self

    def predict(self, _test_x: np.ndarray) -> np.ndarray:
        return self._predictions


def test_get_classifier_specs_returns_requested_items() -> None:
    specs = get_classifier_specs(["svm", "logreg"])
    assert [spec.key for spec in specs] == ["logreg", "svm"]


def test_classifier_registry_exposes_builtin_keys() -> None:
    assert set(CLASSIFIER_REGISTRY) == {"logreg", "svm"}


def test_fit_classifiers_builds_named_prediction_map() -> None:
    feature_sets = [("FeatureA", np.asarray([[0.0], [1.0]]), np.asarray([[0.5]]))]
    specs = [
        type(
            "Spec",
            (),
            {
                "display_name": "Dummy",
                "builder": lambda self=None: _DummyClassifier(
                    predictions=np.asarray([1]),
                    coef=np.asarray([[0.1, 0.9]]),
                ),
            },
        )()
    ]

    classifiers, predictions = fit_classifiers(
        feature_sets,
        train_y=[0, 1],
        classifier_specs=specs,
    )

    assert list(classifiers.keys()) == ["FeatureA [Dummy]"]
    assert predictions["FeatureA [Dummy]"].tolist() == [1]


def test_build_metrics_payload_returns_acc_and_f1() -> None:
    acc_result, f1_result = build_metrics_payload(
        test_y=[0, 1, 1],
        predictions={"ModelA": np.asarray([0, 1, 0])},
    )

    assert acc_result["ModelA"] == 66.67
    assert "macro" in f1_result
    assert "micro" in f1_result


def test_build_feature_importance_payload_handles_binary_case() -> None:
    classifiers = {
        "ModelA": _DummyClassifier(
            predictions=np.asarray([0]),
            coef=np.asarray([[0.1, 0.5, 0.4], [0.6, 0.2, 0.2]]),
        )
    }

    payload = build_feature_importance_payload(
        classifiers=classifiers,
        label_map={"cat_a": 0, "cat_b": 1},
        category="sports",
        category_labels=["cat_a", "cat_b"],
    )

    assert "ModelA" in payload
    assert payload["ModelA"]["cat_a"] == [0, 2, 1]
    assert payload["ModelA"]["cat_b"] == [1, 2, 0]


def test_build_feature_importance_payload_skips_missing_coef() -> None:
    classifiers = {
        "ModelA": _DummyClassifier(
            predictions=np.asarray([0]),
            coef=None,
        )
    }

    payload = build_feature_importance_payload(
        classifiers=classifiers,
        label_map={"cat_a": 0},
        category="all",
        category_labels=["cat_a"],
    )

    assert payload == {"ModelA": {}}
