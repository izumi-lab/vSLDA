# User Guide

This guide summarizes the standard CLI workflow for preparing data, running
experiments, and evaluating results. For artifact details, see
`docs/artifacts.md`; for preprocessing policy, see `docs/preprocessing.md`.

## CLI Entry Points

Use the installed console script for normal runs:

```bash
poetry run spherical-sentence-topics --help
```

The equivalent module entry point is `poetry run python -m src.cli --help`.

The CLI is grouped by workflow:

- `data`: prepare and audit datasets
- `experiments`: run configured model comparisons
- `evaluation`: run downstream metrics, diagnostics, and summaries

Set up the Poetry environment with development and ML dependencies before
running the experiment workflows, then install the NLTK `wordnet` corpus. Some
baselines and vocabulary-based workflows use WordNet for English lemmatization:

```bash
poetry install --with dev,ml
poetry run setup-nltk
```

For lightweight development that does not run embedding-based experiments, the
ML dependency group can be omitted:

```bash
poetry install --with dev
```

Poetry also installs the maintained `jlc-choldate` distribution, which provides
the `choldate` import without leaving a local `choldate/` checkout in the
repository.

The committed example preset is `configs/experiments/20newsgroup.example.yaml`.
Use local copies such as `*.local.yaml` for machine-specific experiments.

## 1. Prepare Data

The standard 20 Newsgroups input is generated as CSV files:

```bash
poetry run spherical-sentence-topics data prepare-20newsgroup --output-dir data/20newsgroup
```

This writes:

- `data/20newsgroup/train.csv`
- `data/20newsgroup/test.csv`

NYT uses `data prepare-nyt` with a fine-label pickle as input and writes
`data/nyt/train.csv` and `data/nyt/test.csv` with a 60/40 stratified split
(`test_size=0.4`, `random_state=42`).

For 20 Newsgroups and NYT, prepare commands apply the shared English sentence
quality policy documented in `docs/preprocessing.md`. To inspect local samples
before regenerating CSVs, run `data audit-preprocessing` on a prepared split:

```bash
poetry run spherical-sentence-topics data audit-preprocessing \
  --input-path data/20newsgroup/test.csv \
  --sample-size 50
```

Audit outputs are written to `scripts/audit_review/` by default. They are local
review artifacts and do not change model inputs until the dataset is regenerated.

## 2. Run Experiments

For a lightweight first run, execute only vMF Sentence LDA for one iteration:

```bash
poetry run spherical-sentence-topics experiments run \
  --config configs/experiments/20newsgroup.example.yaml \
  --models vmf_sentence_lda \
  --iteration 0
```

To run the full canonical comparison preset, omit the model and iteration
filters:

```bash
poetry run spherical-sentence-topics experiments run \
  --config configs/experiments/20newsgroup.example.yaml
```

The preset compares `vmf_sentence_lda` with the configured baselines and writes
artifacts under:

- `results/experiments/20newsgroup/`
- `results/baselines/20newsgroup/`

The run also writes `results/experiments/20newsgroup/summary.json`. Use that
summary to check which category, topic count, iteration, and model combinations
were executed. For individual model executions, resolve the latest successful
archived run through `latest/.../CURRENT.json`.

Common experiment options:

- `--models vmf_sentence_lda,ctm` limits execution to selected models
- `--category`, `--topic`, and repeated `--iteration` override selection axes
- `experiments run-all` runs canonical presets in bulk

Model kind is tracked explicitly in the runner registry and persisted baseline
metadata as `method_kind`.

- `method_kind: topic_model`: `vmf_sentence_lda`, `ctm`, `bleilda`,
  `gaussianlda`, `etm`, `mvtm`, `senclu`, `sentence_gaussianlda`, `sentlda`
- `method_kind: clustering`: `bertopic_kmeans`, `spherical_kmeans`,
  `gaussian_kmeans`, `movmf`, `gaussian_mixture`

Notes:

- experiment commands write artifacts only; they do not trigger evaluation
  automatically
- baseline-specific hyperparameters are configured in `baselines[].params`
  inside the YAML preset
- `summary.json` records execution settings, runtime measurements, and
  comparison metadata

### Encoder Configuration

`encoder` controls the shared sentence embedding model used by
`vmf_sentence_lda` and embedding-aware baselines. The default example uses
`sentence-transformers/all-minilm-l6-v2`:

```yaml
encoder:
  model_name: sentence-transformers/all-minilm-l6-v2
  device: cuda
```

On CPU-only machines, copy the preset to a local config and set `device: cpu`
before running experiments.

Supported encoder profiles include:

- `sentence-transformers/all-minilm-l6-v2` -> `minilm`
- `sentence-transformers/all-mpnet-base-v2` -> `mpnet`
- `baai/bge-base-en-v1.5` -> `bge`
- `cl-nagoya/ruri-v3-130m` -> `ruri`
- `usif` -> `usif`

## 3. Run Evaluation

For the lightweight vMF-only run above, run classification on one category:

```bash
poetry run spherical-sentence-topics evaluation classify \
  --dataset 20newsgroup \
  --category computer \
  --topic 20 \
  --iteration 0 \
  --classifier logreg \
  --model vmf_sentence_lda \
  --embedding-variant minilm
```

This trains and evaluates a logistic regression classifier using saved vMF
document-topic features. Classification outputs are written under
`results/classification/`.

Topic-count diagnostics for the same run are available with:

```bash
poetry run spherical-sentence-topics evaluation topic-count-diagnostics \
  --dataset 20newsgroup \
  --category computer \
  --topic 20 \
  --iteration 0 \
  --embedding-variant minilm
```

Word-based topic metrics for the same run are available with:

```bash
poetry run spherical-sentence-topics evaluation word-based-metrics \
  --dataset 20newsgroup \
  --category computer \
  --topic 20 \
  --iteration 0 \
  --model vmf \
  --embedding-variant minilm \
  --coherence-split test
```

This quick evaluation computes coherence against the selected dataset's test
split. For paper-grade evaluation, prefer an external reference corpus such as
tokenized Wikipedia via `--coherence-reference external` and
`--coherence-reference-path`.

To run the tasks declared by a comparison config, use:

```bash
poetry run spherical-sentence-topics evaluation run-from-config \
  --config configs/experiments/20newsgroup.example.yaml
```

The committed example preset declares `evaluation.tasks: [classification]`.
Pass `--task classification` to select a task explicitly.

Direct classification is available through `evaluation classify` when you need to
target a specific dataset, topic count, iteration, classifier, or embedding
variant. Omit `--embedding-variant` to evaluate every matching embedding-aware
feature variant found through model `latest/CURRENT.json` pointers. Use
`--feature-resolve-mode strict` when invalid latest pointers should fail instead
of being skipped.

To summarize classification metrics, use
`evaluation summarize-classification`. The most common controls are `--metric`,
`--dataset`, `--topic`, `--iteration`, and `--resolve-mode`.

Other evaluation and diagnostic commands include:

- `evaluation list-tasks`
- `evaluation geometry-based-metrics`
- `evaluation word-based-metrics`
- `evaluation word-based-label-profile`
- `evaluation word-based-topic-word-table`
- `evaluation topic-count-diagnostics`
- `evaluation sentence-topic-inspection`
- `evaluation cross-model-pair-diagnostics`

For embedding-aware experiment results, pass the short embedding variant when more
than one latest pointer matches the requested dataset/category/topic/iteration.
For example, use `--embedding-variant mpnet` for `vmf_sentence_lda` or `ctm`
artifacts built from the MPNet encoder. `--encoder` can usually be omitted when
the source artifact metadata has `encoder_config`, or when `--embedding-variant`
is one of the built-in variants such as `minilm`, `mpnet`, `bge`, `ruri`, or
`usif`.

## Output Layout

Main generated roots:

- `results/experiments/`
- `results/baselines/`
- `results/classification/`

Conventions:

- experiment and baseline artifacts write full executions to
  `archive/YYYY-MM-DD/...`
- the latest successful execution for a short display key is recorded in
  `latest/.../CURRENT.json`
- embedding-aware topic model outputs append the short encoder identifier to the
  display key, for example `k10_it3_c1_mpnet`, `k10_it3_minilm`, or `k10_it3_bge`
- diagnostic outputs use the same `latest/archive` pattern under their respective
  roots
- classification feature readers prefer model `latest/CURRENT.json` pointers,
  including embedding-aware variants, and still accept older category-first
  directories when no current pointer matches
- `evaluation summarize-classification --resolve-mode strict` is available when
  ambiguous matches should fail instead of auto-selecting the newest result

Experiment and evaluation runs write `CURRENT.json` as part of the normal artifact
flow. Legacy directories are backward-compatible read targets; normal workflows
should read through `latest/.../CURRENT.json` when it exists.
