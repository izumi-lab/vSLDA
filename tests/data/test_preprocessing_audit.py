from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.data.preprocessing_audit import audit_preprocessed_csv


def test_audit_preprocessed_csv_writes_review_sample_and_summary(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "train.csv"
    output_path = tmp_path / "audit.csv"
    pd.DataFrame(
        {
            "data": [
                "Alpha beta discusses quality. / ... / Beta follows.",
                ": We need printer advice. / N.",
            ],
            "target_str": ["science", "graphics"],
        }
    ).to_csv(input_path, index=False, encoding="utf-8")

    summary = audit_preprocessed_csv(
        input_path=input_path,
        output_path=output_path,
        sample_size=0,
    )

    review = pd.read_csv(output_path)
    summary_payload = json.loads(
        output_path.with_suffix(".summary.json").read_text(encoding="utf-8")
    )

    assert summary.review_rows == 5
    assert summary_payload["sentences_seen"] == 5
    assert "review_decision" in review.columns
    assert "review_comment" in review.columns
    assert review["decision"].value_counts().to_dict() == {"drop": 3, "keep": 2}
    assert set(review["sample_group"]) >= {"drop_candidate", "random"}
