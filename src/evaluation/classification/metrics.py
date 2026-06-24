from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def build_metrics_payload(
    *,
    test_y: Sequence[int],
    predictions: dict[str, np.ndarray],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    acc_result = {
        name: round(accuracy_score(test_y, preds) * 100, 2)
        for name, preds in predictions.items()
    }
    f1_result = {
        "macro": {
            name: round(f1_score(test_y, preds, average="macro") * 100, 2)
            for name, preds in predictions.items()
        },
        "micro": {
            name: round(f1_score(test_y, preds, average="micro") * 100, 2)
            for name, preds in predictions.items()
        },
    }
    return acc_result, f1_result


def build_feature_importance_payload(
    *,
    classifiers: dict[str, Any],
    label_map: dict[str, int],
    category: str,
    category_labels: Sequence[str],
) -> dict[str, Any]:
    feature_importance: dict[str, Any] = {}
    for name, classifier in classifiers.items():
        feature_importance[name] = {}
        coef = getattr(classifier, "coef_", None)
        if coef is None:
            continue
        if category != "all" and len(category_labels) == 2:
            feature_importance[name][
                list(label_map.keys())[0]
            ] = coef.argsort().tolist()[0]
            feature_importance[name][list(label_map.keys())[1]] = coef.argsort()[
                :, ::-1
            ].tolist()[0]
        else:
            for line, label in zip(coef.argsort()[:, ::-1], label_map.keys()):
                feature_importance[name][label] = line.tolist()
    return feature_importance


def build_coverage_payload(
    *,
    raw_train_docs: int,
    raw_test_docs: int,
    label_filtered_train_docs: int,
    label_filtered_test_docs: int,
    selected_train_docs: int,
    selected_test_docs: int,
    common_train_docs: int,
    common_test_docs: int,
    available_train_docs: dict[str, int],
    available_test_docs: dict[str, int],
) -> dict[str, Any]:
    return {
        "raw_train_docs": int(raw_train_docs),
        "raw_test_docs": int(raw_test_docs),
        "label_filtered_train_docs": int(label_filtered_train_docs),
        "label_filtered_test_docs": int(label_filtered_test_docs),
        "selected_train_docs": int(selected_train_docs),
        "selected_test_docs": int(selected_test_docs),
        "common_train_docs": int(common_train_docs),
        "common_test_docs": int(common_test_docs),
        "available_train_docs": {
            str(name): int(count) for name, count in available_train_docs.items()
        },
        "available_test_docs": {
            str(name): int(count) for name, count in available_test_docs.items()
        },
    }
