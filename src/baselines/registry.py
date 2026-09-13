from __future__ import annotations

from dataclasses import dataclass

from src.baselines.adapters import (
    run_bertopic_kmeans,
    run_bleilda,
    run_ctm,
    run_etm,
    run_gaussian_kmeans,
    run_gaussian_mixture,
    run_gaussianlda,
    run_movmf,
    run_mvtm,
    run_sam,
    run_sam_tf,
    run_senclu,
    run_sentence_gaussianlda,
    run_sentlda,
    run_spherical_kmeans,
)
from src.baselines.contracts import BaselineRunnerCallable, BaselineRunnerSpec
from src.baselines.model_kinds import baseline_method_kind


def _runner_spec(
    *,
    key: str,
    display_name: str,
    family: str,
    runner: BaselineRunnerCallable,
) -> BaselineRunnerSpec:
    return BaselineRunnerSpec(
        key=key,
        display_name=display_name,
        family=family,
        runner=runner,
        method_kind=baseline_method_kind(key),
    )


RUNNERS: dict[str, BaselineRunnerSpec] = {
    "ctm": _runner_spec(
        key="ctm",
        display_name="Contextual TM",
        family="ctm",
        runner=run_ctm,
    ),
    "sam": _runner_spec(
        key="sam",
        display_name="SAM (tf-idf)",
        family="sam",
        runner=run_sam,
    ),
    "sam_tf": _runner_spec(
        key="sam_tf",
        display_name="SAM",
        family="sam",
        runner=run_sam_tf,
    ),
    "bleilda": _runner_spec(
        key="bleilda",
        display_name="Blei LDA",
        family="bleilda",
        runner=run_bleilda,
    ),
    "bertopic_kmeans": _runner_spec(
        key="bertopic_kmeans",
        display_name="BERTopic (UMAP + k-means)",
        family="bertopic_kmeans",
        runner=run_bertopic_kmeans,
    ),
    "gaussianlda": _runner_spec(
        key="gaussianlda",
        display_name="Gaussian LDA",
        family="gaussianlda",
        runner=run_gaussianlda,
    ),
    "etm": _runner_spec(
        key="etm",
        display_name="ETM",
        family="etm",
        runner=run_etm,
    ),
    "mvtm": _runner_spec(
        key="mvtm",
        display_name="MvTM",
        family="mvtm",
        runner=run_mvtm,
    ),
    "spherical_kmeans": _runner_spec(
        key="spherical_kmeans",
        display_name="Spherical k-means",
        family="spherical_kmeans",
        runner=run_spherical_kmeans,
    ),
    "gaussian_kmeans": _runner_spec(
        key="gaussian_kmeans",
        display_name="Gaussian k-means",
        family="gaussian_kmeans",
        runner=run_gaussian_kmeans,
    ),
    "movmf": _runner_spec(
        key="movmf",
        display_name="movMF",
        family="movmf",
        runner=run_movmf,
    ),
    "gaussian_mixture": _runner_spec(
        key="gaussian_mixture",
        display_name="Gaussian mixture",
        family="gaussian_mixture",
        runner=run_gaussian_mixture,
    ),
    "senclu": _runner_spec(
        key="senclu",
        display_name="SenClu",
        family="senclu",
        runner=run_senclu,
    ),
    "sentlda": _runner_spec(
        key="sentlda",
        display_name="sentLDA",
        family="sentlda",
        runner=run_sentlda,
    ),
    "sentence_gaussianlda": _runner_spec(
        key="sentence_gaussianlda",
        display_name="Sentence LDA",
        family="sentence_gaussianlda",
        runner=run_sentence_gaussianlda,
    ),
}


# ``SAM`` means the tf condition, so a config that writes ``runner: sam`` gets the
# tf runner.  The keys themselves are not renamed: they are baked into artifact
# paths (``results/baselines/<ds>/default/sam_tf``) and into the evaluation
# condition fingerprints, which include the model name, so swapping them would
# invalidate every coherence, entropy and classification condition already
# computed.  The alias keeps configs readable without that cost; ``sam_tfidf``
# spells out the variant for anyone who wants it explicitly.
RUNNER_ALIASES: dict[str, str] = {
    "sam": "sam_tf",
    "sam_tfidf": "sam",
    "sam_tf_idf": "sam",
}


def resolve_runner_name(name: str) -> str:
    """Map a config-facing runner name onto its registry key."""

    return RUNNER_ALIASES.get(str(name).strip(), str(name).strip())


def get_runner_spec(name: str) -> BaselineRunnerSpec:
    resolved = resolve_runner_name(name)
    if resolved not in RUNNERS:
        raise ValueError(f"Unknown baseline runner: {name}")
    return RUNNERS[resolved]


@dataclass(frozen=True)
class RunnerCompanion:
    """A runner that is always scheduled alongside another one.

    ``inherit_params_except`` lists the parent's parameter keys that must *not*
    carry over, because they are exactly what makes the companion a different
    condition.
    """

    key: str
    inherit_params_except: tuple[str, ...] = ()


# The SAM paper reports l2-normalized tf and tf-idf document representations as
# two conditions, so a config asking for SAM always wants both trees.  Everything
# except ``feature_scheme`` is shared, so the companion inherits any tuning applied
# to the entry that is spelled out and the two stay comparable.
#
# ``sam_tf`` is the parent because tf is the reported condition: it is the input
# the other bag-of-words baselines get (LDA and sentLDA are fitted on raw counts),
# whereas idf down-weights exactly the frequent words that NPMI also penalizes.
# A config therefore names the tf runner and gets tf-idf as the variant.
COMPANION_RUNNERS: dict[str, tuple[RunnerCompanion, ...]] = {
    "sam_tf": (RunnerCompanion(key="sam", inherit_params_except=("feature_scheme",)),),
}


def companion_runners(name: str) -> tuple[RunnerCompanion, ...]:
    return COMPANION_RUNNERS.get(resolve_runner_name(name), ())
