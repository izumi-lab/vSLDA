from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ClassifierBuilder = Callable[[], Any]


@dataclass(frozen=True)
class ClassifierSpec:
    key: str
    display_name: str
    builder: ClassifierBuilder


def _build_logreg() -> Any:
    return LogisticRegression(
        random_state=42,
        multi_class="multinomial",
    )


def _build_svm() -> Any:
    return SVC(random_state=42)


CLASSIFIER_REGISTRY: dict[str, ClassifierSpec] = {
    "logreg": ClassifierSpec(
        key="logreg",
        display_name="LogReg",
        builder=_build_logreg,
    ),
    "svm": ClassifierSpec(
        key="svm",
        display_name="SVM",
        builder=_build_svm,
    ),
}


def get_classifier_specs(classifiers_to_use: Sequence[str]) -> list[ClassifierSpec]:
    return [
        CLASSIFIER_REGISTRY[key]
        for key in CLASSIFIER_REGISTRY
        if key in set(classifiers_to_use)
    ]


def fit_classifiers(
    feature_sets: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    train_y: Sequence[int],
    classifier_specs: Sequence[ClassifierSpec],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    classes = np.unique(train_y)
    if len(classes) < 2:
        raise ValueError(
            "Training data must contain at least 2 classes after alignment; "
            f"got {len(classes)} class"
        )

    classifiers: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for feature_name, train_x, test_x in feature_sets:
        for spec in classifier_specs:
            classifier = spec.builder().fit(train_x, train_y)
            result_name = f"{feature_name} [{spec.display_name}]"
            classifiers[result_name] = classifier
            predictions[result_name] = classifier.predict(test_x)
    return classifiers, predictions
