# Spherical Topic Models with Sentence Embeddings

[![python](https://img.shields.io/badge/python-3.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![CI](https://github.com/izumi-lab/vSLDA/actions/workflows/ci.yml/badge.svg)](https://github.com/izumi-lab/vSLDA/actions/workflows/ci.yml)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Spherical Topic Models with Sentence Embeddings are a family of sentence-level
topic models that assign topics to sentences, with each sentence represented as
a unit-normalized embedding on the hypersphere. Each topic is modeled by a
topic-specific von Mises–Fisher (vMF) distribution over sentence embeddings,
while the document–topic proportions follow a Dirichlet prior, preserving the
hierarchical structure of LDA.

This repository provides a CLI-first research codebase for running reproducible
experiments with this model family. The implemented model is referred to as
vMF Sentence LDA (vSLDA), a compact name for comparison with baselines such as
Gaussian LDA (GLDA) and Gaussian Sentence LDA (GSLDA).

## At a Glance

- Implements vMF Sentence LDA for sentence-level topic modeling
- Supports comparisons with topic-model baselines
- Provides CLI workflows for dataset preparation, experiments, and evaluation
- Persists reproducible artifacts under `results/`
- Main example dataset: 20 Newsgroups
- Runs custom CSV datasets from local YAML configs

The workflow is intentionally artifact-based:

1. prepare datasets as canonical train/test CSV files
2. run configured model comparisons and persist artifacts
3. evaluate saved outputs with explicit downstream tasks

For the full command guide, see [`docs/user_guide.md`](docs/user_guide.md).

## Requirements

- Python `>=3.12,<3.15`
- [uv](https://docs.astral.sh/uv/) (recommended). Poetry 2.x is also supported;
  see [Using Poetry instead of uv](#using-poetry-instead-of-uv).

## Installation

Install development and ML dependencies for the Quick Start, then install the
NLTK WordNet corpus. WordNet is used for English lemmatization in some
baselines and vocabulary-based workflows:

```bash
uv sync --group ml
uv run setup-nltk
```

`uv sync` always installs the `dev` group. For lightweight development that
does not run embedding-based experiments, the ML dependency group can be
omitted:

```bash
uv sync
```

uv picks the Python version from `.python-version` when it exists. If that file
names something uv does not understand, such as a pyenv virtualenv, pass the
version explicitly with `uv sync --python 3.12` or set `UV_PYTHON=3.12`.

### Using Poetry instead of uv

The project is also installable with Poetry 2.x. `poetry.lock` is kept in sync
with `uv.lock`, so both tools install the same versions:

```bash
poetry install --with dev,ml
poetry run setup-nltk
```

Any command in this README written as `uv run ...` also works as
`poetry run ...`.

Poetry automatically uses an in-project `.venv` when one exists, so after
`uv sync` both tools share the same environment. `uv sync` removes packages
that are not in the selected groups, so pass the same groups to both tools,
for example `uv sync --group ml` together with `poetry install --with ml`.

The committed example config uses `encoder.device: cuda`. On CPU-only machines,
copy the preset to a local config and set `encoder.device: cpu` before running
experiments.

Use the installed console script for normal runs:

```bash
uv run spherical-sentence-topics --help
```

The equivalent module entry point is:

```bash
uv run python -m src.cli --help
```

The CLI has three main command groups:

- `data`: prepare and audit datasets
- `experiments`: run artifact-generating comparison presets
- `evaluation`: run classification, topic-word, geometry, and diagnostic tasks

## Quick Start

### 1. Prepare Data

Prepare the 20 Newsgroups split used by the committed example preset:

```bash
uv run spherical-sentence-topics data prepare-20newsgroup --output-dir data/20newsgroup
```

This creates:

- `data/20newsgroup/train.csv`
- `data/20newsgroup/test.csv`

NYT data can be prepared from a raw fine-label pickle:

```bash
uv run spherical-sentence-topics data prepare-nyt \
  --raw-path data/nyt/raw/df_fine.pkl \
  --output-dir data/nyt
```

This writes `data/nyt/train.csv` and `data/nyt/test.csv` with the same
`test_size=0.4`, `random_state=42`, and stratified split policy as the
20 Newsgroups preparation path. See [`docs/preprocessing.md`](docs/preprocessing.md)
for the shared preprocessing policy.

#### Use Your Own CSV Dataset

You can also skip the built-in preparation commands and place a local dataset
under `data/`:

- `data/my_corpus/train.csv`
- `data/my_corpus/test.csv`

Each row is one document, and the example config below expects the text column
to be named `data`. For sentence-level topic modeling, store sentences joined
with `" / "`:

```csv
data
"First sentence. / Second sentence."
"Another document. / More text."
```

If you start from raw prose, call `src.data.segmentation.segment_text` from a
local preprocessing script and write the joined text to `data`.

Use the committed example config
[`configs/experiments/my_corpus.example.yaml`](configs/experiments/my_corpus.example.yaml)
as a starting point. It is configured for an unsupervised `my_corpus` run with
MinILM embeddings on CPU.

For English text, use `language: english` in the config and
`segment_text(..., language="en")` in preprocessing.

Run vMF Sentence LDA on the full corpus:

```bash
uv run spherical-sentence-topics experiments run \
  --config configs/experiments/my_corpus.example.yaml
```

Artifacts are written under `results/experiments/my_corpus/`.

If you prefer to store unsegmented prose in `data`, set
`preprocess.segmenter: pysbd` and `preprocess.delimiter: null`.

### 2. Choose an Encoder Device

On CPU-only machines, copy the preset to a local config such as
`configs/experiments/20newsgroup.local.yaml` and set:

```yaml
encoder:
  device: cpu
```

### 3. Run Experiments

Run vMF Sentence LDA from the 20 Newsgroups example preset:

```bash
uv run spherical-sentence-topics experiments run \
  --config configs/experiments/20newsgroup.example.yaml \
  --models vmf_sentence_lda \
  --iteration 0
```

This keeps the quick start lightweight by skipping the configured baseline
runners and running a single seed/iteration. The run writes artifacts under
`results/experiments/20newsgroup/`.

It also writes `results/experiments/20newsgroup/summary.json`, which records the
executed model, category, topic-count, iteration, and runtime metadata.

### 4. Run Evaluation

Run classification for one category from the same single iteration:

```bash
uv run spherical-sentence-topics evaluation classify \
  --dataset 20newsgroup \
  --category computer \
  --topic 20 \
  --iteration 0 \
  --classifier logreg \
  --model vmf_sentence_lda \
  --embedding-variant minilm
```

This trains and evaluates a logistic regression classifier using the saved vMF
document-topic features. Classification outputs are written under
`results/classification/`.

Run topic-count diagnostics for the same run:

```bash
uv run spherical-sentence-topics evaluation topic-count-diagnostics \
  --dataset 20newsgroup \
  --category computer \
  --topic 20 \
  --iteration 0 \
  --embedding-variant minilm
```

Diagnostic outputs are written under `results/topic_count_analysis/`.

Run word-based topic metrics for the same run:

```bash
uv run spherical-sentence-topics evaluation word-based-metrics \
  --dataset 20newsgroup \
  --category computer \
  --topic 20 \
  --iteration 0 \
  --model vmf \
  --embedding-variant minilm \
  --coherence-split test
```

Word-based outputs are written under `results/topic_analysis/coherence/`.
This quick-start command computes coherence against the selected dataset's test
split. For paper-grade evaluation, prefer an external reference corpus such as
tokenized Wikipedia via `--coherence-reference external` and
`--coherence-reference-path`.

## Artifact Conventions

Experiment, baseline, classification, and analysis outputs use a stable
`latest/.../CURRENT.json` pointer plus immutable archived run directories under
`archive/YYYY-MM-DD/...`.

Main generated roots:

- `results/experiments/`, `results/baselines/`: model runs
- `results/classification/`, `results/topic_analysis/`, `results/timing/`: evaluation runs
  and their `summaries/` (the `*.scores.json` sidecars are the interface to the manuscript)
- `results/analysis/`, `results/tables/`, `results/diagnostics/`, `results/visualization/`:
  analysis scripts and inspection outputs

For the complete output contract, see [`docs/artifacts.md`](docs/artifacts.md).

## Repository Layout

- `configs/experiments/`: canonical experiment presets
- `src/`: repo-owned implementation and CLI
- `tests/`: contract and smoke tests
- `docs/`: user and developer-facing documentation
- `results/`: generated outputs; writers use `latest/` pointers plus archived
  executions

Key internal entry points:

- `src/cli/app.py`: assembles the CLI from `experiment_commands.py`, `data_commands.py`,
  and `evaluation_commands.py`
- `src/experiments/config.py`: compatibility surface over config schema, loader, and parsers
- `src/experiments/execution.py`: orchestration layer over `vmf_runner.py`,
  `baseline_runner.py`, and `summary_builder.py`
- `src/core/paths.py`: compatibility surface over path builders, resolution, and latest
  pointer helpers

## Documentation

- `docs/user_guide.md`: command-oriented usage guide
- `docs/artifacts.md`: output layout and metadata conventions
- `docs/preprocessing.md`: shared preprocessing policy and model input views
- `docs/architecture.md`: internal module boundaries and responsibilities

## Development

Code quality is checked in a fixed order: isort sorts imports, black
formats, ruff lints, and pytest runs the tests. isort runs before black so
that black has the final say on layout, and ruff runs on the formatted code.

Before committing, apply the fixers and then run the checks in this order:

```bash
uv run isort .
uv run black .
uv run ruff check .
uv run pytest -q -m "not slow and not integration"
```

CI runs the same steps in the same order in check-only mode:

```bash
uv run isort --check-only .
uv run black --check .
uv run ruff check .
uv run pytest -q -m "not slow and not integration"
```

After changing dependencies in `pyproject.toml`, refresh both lock files so
uv and Poetry users stay on the same versions:

```bash
uv lock
poetry lock
```

Run ML-dependent tests after installing the `ml` dependency group:

```bash
uv run pytest -q -m "slow or integration"
```

## Citation

A paper describing this work is in preparation. Citation metadata will be added
when the paper is available.
