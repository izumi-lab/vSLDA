from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.evaluation.topic_pairs import metrics as metrics_module
from src.evaluation.topic_pairs import summary as summary_module
from src.evaluation.topic_pairs.metrics import run_topic_pair_metrics
from src.evaluation.topic_pairs.summary import write_topic_pair_summary
from src.evaluation.topic_pairs.words import (
    ReferenceWordsError,
    reference_words_sidecar_path,
    resolve_display_words,
    write_reference_words,
)


def _write_coherence_run(
    root: Path, *, name: str, meta: dict, words: dict[int, list[str]] | None
) -> None:
    latest = root / "latest" / "dummy" / "default" / "a" / name
    archive = root / "archive" / name
    archive.mkdir(parents=True, exist_ok=True)
    latest.mkdir(parents=True, exist_ok=True)
    (latest / "CURRENT.json").write_text(
        json.dumps({"archive_dir": str(archive)}), encoding="utf-8"
    )
    (archive / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    if words is not None:
        payload = {
            "results": {
                "per_iteration": [
                    {
                        "iteration": meta["iterations"][0],
                        "topics": [
                            {
                                "topic_id": topic,
                                "words": [{"word": w, "score": 0.1} for w in listed],
                            }
                            for topic, listed in words.items()
                        ],
                    }
                ]
            }
        }
        (archive / "topic_words_display_topk.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )


BASE = dict(
    model="vmf",
    num_topics=3,
    iterations=[0],
    effective_embedding_variant="mpnet",
    prior_scale=None,
    condition_id="it0__k3__vmf__mpnet__aaaa",
    condition_fingerprint="aaaa",
    topic_word_score_mode="word_topic_npmi",
)


def test_resolve_display_words_matches_one_run(tmp_path: Path) -> None:
    root = tmp_path / "coh"
    _write_coherence_run(
        root,
        name="it0__k3__vmf__mpnet__aaaa",
        meta=BASE,
        words={0: ["a", "b", "c"], 2: ["x", "y"]},
    )
    # decoys: other iteration, other encoder, a prior-scale variant of GSLDA, the requested-but-not-read variant of SentLDA
    _write_coherence_run(
        root,
        name="it1__k3__vmf__mpnet__bbbb",
        meta={**BASE, "iterations": [1]},
        words={0: ["no"]},
    )
    _write_coherence_run(
        root,
        name="it0__k3__vmf__bge__cccc",
        meta={**BASE, "effective_embedding_variant": "bge"},
        words={0: ["no"]},
    )
    _write_coherence_run(
        root,
        name="it0__k3__sentence-gaussianl__mpnet-raw__psi0-1__dddd",
        meta=dict(
            model="sentence_gaussianlda",
            num_topics=3,
            iterations=[0],
            effective_embedding_variant="mpnet_raw",
            prior_scale=1.0,
        ),
        words={0: ["no"]},
    )
    _write_coherence_run(
        root,
        name="it0__k3__sentlda__eeee",
        meta=dict(
            model="sentlda",
            num_topics=3,
            iterations=[0],
            embedding_variant="mpnet",
            effective_embedding_variant=None,
            prior_scale=None,
        ),
        words={1: ["s1", "s2"]},
    )
    common = dict(
        dataset="dummy",
        data_run="default",
        category="a",
        iteration=0,
        num_topics=3,
        embedding_variant="mpnet",
        top_n=2,
    )
    resolved = resolve_display_words(root, model="vmf", **common)
    assert resolved.words == [["a", "b"], [], ["x", "y"]]
    assert resolved.condition_id == "it0__k3__vmf__mpnet__aaaa"
    assert resolved.score_mode == "word_topic_npmi"
    assert resolved.effective_embedding_variant == "mpnet"
    sentlda = resolve_display_words(root, model="sentlda", **common)
    assert sentlda.words == [[], ["s1", "s2"], []]
    assert sentlda.effective_embedding_variant is None
    with pytest.raises(ReferenceWordsError, match="no coherence run"):
        resolve_display_words(root, model="sentence_gaussianlda", **common)
    _write_coherence_run(
        root, name="it0__k3__vmf__mpnet__ffff", meta=BASE, words={0: ["dup"]}
    )
    with pytest.raises(ReferenceWordsError, match="2 coherence runs"):
        resolve_display_words(root, model="vmf", **common)


def test_write_reference_words_sidecar(tmp_path: Path) -> None:
    root = tmp_path / "coh"
    _write_coherence_run(
        root,
        name="it0__k3__vmf__mpnet__aaaa",
        meta=BASE,
        words={0: ["a", "b", "c", "d"], 1: ["e"], 2: ["x"]},
    )
    _write_coherence_run(
        root,
        name="it0__k3__sentlda__eeee",
        meta=dict(
            model="sentlda",
            num_topics=3,
            iterations=[0],
            effective_embedding_variant=None,
            prior_scale=None,
            condition_id="it0__k3__sentlda__eeee",
            topic_word_score_mode="word_topic_npmi",
        ),
        words={0: ["s"]},
    )
    written = write_reference_words(
        output_dir=tmp_path / "summaries",
        coherence_root=root,
        runs=(("dummy", "a", 0),),
        num_topics=3,
        encoder_variant="mpnet",
        models=("vmf", "sentlda"),
        top_n=3,
    )
    path = reference_words_sidecar_path(
        tmp_path / "summaries",
        dataset="dummy",
        data_run="default",
        encoder="mpnet",
        category="a",
        iteration=0,
        num_topics=3,
    )
    assert written == [path]
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["task"] == "topic_pair_reference_words"
    assert (
        payload["topics"] == 3 and payload["iteration"] == 0 and payload["top_n"] == 3
    )
    assert payload["words"]["vmf"] == [["a", "b", "c"], ["e"], ["x"]]
    assert payload["words"]["sentlda"] == [["s"], [], []]
    assert payload["provenance"]["vmf"]["condition_id"] == "it0__k3__vmf__mpnet__aaaa"
    assert payload["provenance"]["vmf"]["encoder_model"] == "mpnet"
    assert payload["provenance"]["sentlda"]["encoder_model"] is None
    assert payload["provenance"]["vmf"]["topic_word_score_mode"] == "word_topic_npmi"


def test_paper_summary_writes_the_reference_words(
    toy_runner, monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        metrics_module, "_uses_default_output_layout", lambda _root: True
    )
    run_topic_pair_metrics(
        models=["vmf", "sentlda", "sentence_gaussianlda"],
        dataset="dummy",
        iterations=[0, 1],
        num_topics=3,
        categories=["a"],
        out_root=tmp_path,
        embedding_variant="mpnet",
    )
    calls: list[dict] = []

    def _fake_write(**kwargs):
        calls.append(kwargs)
        return [tmp_path / "words.json"]

    monkeypatch.setattr(summary_module, "write_reference_words", _fake_write)
    monkeypatch.setattr(summary_module, "check_paper_grid", lambda records: None)
    monkeypatch.setattr(summary_module, "select_paper_records", lambda records: records)
    write_topic_pair_summary(
        out_root=tmp_path, paper=True, coherence_root=tmp_path / "coh"
    )
    assert len(calls) == 1
    assert calls[0]["output_dir"] == tmp_path / "summaries"
    assert calls[0]["coherence_root"] == tmp_path / "coh"


def test_resolve_display_words_skips_sweep_variants_and_keeps_the_newest_run(
    tmp_path: Path,
) -> None:
    """The hyperparameter sweep writes pointers beside the main run, and a main
    condition evaluated twice keeps the newer execution (as the summary does)."""

    root = tmp_path / "coh"
    _write_coherence_run(
        root,
        name="it0__k3__vmf__mpnet__aaaa",
        meta={**BASE, "started_at": "2026-08-24T01:00:00+00:00"},
        words={0: ["old"]},
    )
    _write_coherence_run(
        root,
        name="it0__k3__vmf__mpnet__bbbb",
        meta={
            **BASE,
            "condition_id": "it0__k3__vmf__mpnet__bbbb",
            "started_at": "2026-09-02T17:00:00+00:00",
        },
        words={0: ["new"]},
    )
    _write_coherence_run(
        root,
        name="it0__k3__vmf__mpnet__t-1__cccc",
        meta={
            **BASE,
            "parameter_variant": "t-1",
            "started_at": "2026-09-03T00:00:00+00:00",
        },
        words={0: ["sweep"]},
    )
    common = dict(
        dataset="dummy",
        data_run="default",
        category="a",
        iteration=0,
        num_topics=3,
        embedding_variant="mpnet",
        top_n=2,
    )
    resolved = resolve_display_words(root, model="vmf", **common)
    assert resolved.words[0] == ["new"]
    assert resolved.condition_id == "it0__k3__vmf__mpnet__bbbb"

    _write_coherence_run(
        root,
        name="it0__k3__vmf__mpnet__dddd",
        meta={**BASE, "started_at": "2026-09-02T17:00:00+00:00"},
        words={0: ["tie"]},
    )
    with pytest.raises(ReferenceWordsError, match="same time"):
        resolve_display_words(root, model="vmf", **common)
