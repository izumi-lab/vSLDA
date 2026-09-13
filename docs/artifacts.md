# Artifacts

This repository uses persisted artifacts as a comparison contract between experiment runs
and downstream evaluation tasks.

## Result Roots

Model outputs (the writers of `experiments run`):

- `results/experiments/<dataset>/` for `vmf_sentence_lda` runs, plus one `summary.json` per dataset
- `results/baselines/<dataset>/<data_run>/<model>/` for the baseline runners

Evaluation and analysis roots that the current workflows populate:

- `results/classification/` for `evaluation classify` (`latest/`, `archive/`), plus
  `summaries/` (`summarize-classification`) and `figures/` (`plot_limited`)
- `results/topic_analysis/coherence/` for `word_based_metrics`, plus `summaries/`
  (`summarize-coherence`); `.cache/`, `.checkpoints/` and `partial/` hold resumable state
- `results/topic_analysis/entropy_based/` for `entropy_based_metrics`, plus `summaries/`
  and `summaries_<assignment>/` (`entropy-based-summary`, see "Summary Sidecars")
- `results/topic_analysis/topic_pairs/` for `topic_pair_metrics`, plus `summaries/`
  (`topic-pair-summary`) and `.cache/sentence_embeddings/`
- `results/topic_analysis/geometry_based/` for `geometry_based_metrics`
- `results/topic_analysis/foldin/summaries/` for `evaluation vmf-foldin-theta` (one CSV)
- `results/timing/summaries/` for `summarize-timing`
- `results/analysis/topic_sweep/<dataset>/` for `evaluation topic-sweep` (csv/json/tex plus `figures/`)
- `results/analysis/fine_category_{lexical,embedding}_similarity{,_sensitivity}/run_<timestamp>/`
  for `scripts/analyze_fine_category_*.py` (one directory per invocation, no `latest/` pointer)
- `results/tables/topic_interpretation/` for `scripts/render_topic_interpretation_tex.py` and
  `scripts/render_compared_tex.py`
- `results/diagnostics/` for `scripts/compute_gaussian_covariance_conditions.py`
- `results/visualization/` for `sentence_topic_inspection`

Roots that CLI commands define but that no current workflow populates (they appear only
when the corresponding command is run):

- `results/topic_analysis/label_profile/` for `word_based_label_profile`
- `results/topic_count_analysis/` for `topic_count_diagnostics`
- `results/analysis/vmf_vs_baseline/` for `cross_model_pair_diagnostics`

Classification readers resolve feature inputs from model `latest/CURRENT.json`
pointers first. When multiple embedding-aware model outputs match the same
topic/iteration/category, classification treats them as separate feature sets and can
filter them with `--embedding-variant`.

## vMF Sentence LDA Run Layout

vMF Sentence LDA artifact layout:

```text
results/experiments/<dataset>/<data_run>/vmf_sentence_lda/latest/<category>/<display_key>/CURRENT.json
results/experiments/<dataset>/<data_run>/vmf_sentence_lda/archive/YYYY-MM-DD/<category>/<display_key>/vmf_YYYYMMDDTHHMMSSZ/
```

`display_key` is a short human-readable identifier such as `k20_it0`.
Embedding-aware vMF runs append the short encoder identifier after any component
suffix, for example `k20_it0_c1_mpnet` or `k20_it0_c1_bge`.
The authoritative run identity remains in `metadata.json` as:

- `condition_id`
- `condition_fingerprint`
- `execution_id`
- `started_at`

`CURRENT.json` is a pointer that records which archived execution should be treated as
the latest successful result for that key.

Common files:

- `metadata.json`: stable run metadata
- `metrics.json`: run metrics and diagnostics
- `params.json`: persisted model parameters
- `doc_topic_*.pkl`: document-topic outputs
- `sentence_topic_*.pkl`: sentence-topic outputs when produced
- `*_preprocessed.pkl`: persisted shared preprocessing artifacts

## Baseline Run Layout

Baseline runners expose a shared artifact contract:

```text
results/baselines/<dataset>/<data_run>/<model>/latest/<category>/<display_key>/CURRENT.json
results/baselines/<dataset>/<data_run>/<model>/archive/YYYY-MM-DD/<category>/<display_key>/baseline_YYYYMMDDTHHMMSSZ/
results/baselines/<dataset>/<data_run>/<model>/archive/YYYY-MM-DD/<category>/<display_key>/baseline_YYYYMMDDTHHMMSSZ/params/
results/baselines/<dataset>/<data_run>/<model>/archive/YYYY-MM-DD/<category>/<display_key>/baseline_YYYYMMDDTHHMMSSZ/infer/
```

- `train_path`: primary train artifact
- `infer_path`: primary infer or test artifact
- `extras.metadata`: baseline `metadata.json`
- `extras.train_dir` and `extras.infer_dir`: model-specific artifact directories

The goal is to keep evaluation readers independent from model-specific path conventions.

Embedding-aware baseline runners use the same embedding suffix convention in
`display_key`. Sentence-embedding-aware baselines are:

- `ctm`
- `senclu`
- `sentence_gaussianlda`
- `bertopic_kmeans`
- `spherical_kmeans`
- `gaussian_kmeans`
- `movmf`
- `gaussian_mixture`

Word-embedding-aware baselines also append a short word-vector suffix:

- `gaussianlda`
- `etm`
- `mvtm`

Examples:

- `glove-wiki-gigaword-100` -> `glove100`
- `glove-wiki-gigaword-50` -> `glove50`
- `wikientvec:20190520:jawiki.word_vectors.100d.txt.bz2` -> `wikient100`

Baselines that do not use sentence or word embeddings, such as `bleilda`,
`sam` and `sentlda`, keep the plain `k<num_topics>_it<iteration>` display key.

ETM writes document-topic distributions to `params/etm.pkl` and
`infer/<category>.pkl`, a soft-preferred copy to
`infer/<category>_doc_topic_soft.pkl`, and learned topic-word probabilities to
`params/topic_word_scores.pkl` with `params/vocabulary.json`. Word-based metrics
read this learned ETM beta distribution; classification uses the document-topic
artifacts.

SAM uses the same layout with `params/sam.pkl` as the train artifact, plus
`params/idf.pkl` (needed to reproduce held-out features). Note that its
`params/topic_word_scores.pkl` holds signed*unit vectors rather than the
probability distribution ETM stores under the same filename.

CTM stores the train document-topic matrix in `params/ctm.pkl`, the vocabulary in
`params/tp.pkl`, and the fitted network in
`params/contextualized_topic_model_<hyperparameters>/epoch_<N>.pth`. The topic-word
distribution exists only inside that checkpoint: word-based metrics reload it through
`load_ctm_decoder_scores`, so the `.pth` files (about 186 MB each) must be kept as long
as CTM coherence or topic words may be recomputed.

## Evaluation Output Layout

Evaluation roots use the same `latest/archive` contract where practical, while readers
keep backward-compatible fallbacks for older category-first trees.

- `geometry_based_metrics` uses:

```text
results/topic_analysis/geometry_based/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/topic_analysis/geometry_based/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- `entropy_based_metrics` uses:

```text
results/topic_analysis/entropy_based/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/topic_analysis/entropy_based/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- `topic_pair_metrics` uses the same layout under `results/topic_analysis/topic_pairs/`
  (one condition per model x K x category, plus a `cross_model` condition per K x category
  holding the between-model topic overlap), caches the raw sentence embeddings of each
  category under `results/topic_analysis/topic_pairs/.cache/sentence_embeddings/v1/<key>/`,
  and `topic_pair_summary` writes
  `results/topic_analysis/topic_pairs/summaries/<dataset>/<data_run>/<encoder>/topic_pairs_<dataset>_<data_run>_<encoder>_<K>topic.scores.json`
  (raw per-run values: per-topic arrays, K x K pair matrices, cross-model overlap,
  provenance; nothing aggregated).

- `word_based_metrics` uses:

```text
results/topic_analysis/coherence/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/topic_analysis/coherence/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- `word_based_label_profile` uses:

```text
results/topic_analysis/label_profile/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/topic_analysis/label_profile/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- `topic_count_diagnostics` uses:

```text
results/topic_count_analysis/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/topic_count_analysis/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- `cross_model_pair_diagnostics` uses:

```text
results/analysis/vmf_vs_baseline/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/analysis/vmf_vs_baseline/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- `sentence_topic_inspection` stores per-condition payloads under:

```text
results/visualization/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/visualization/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- classification writers use:

```text
results/classification/latest/<dataset>/<data_run>/all/<display_key>/CURRENT.json
results/classification/archive/YYYY-MM-DD/<dataset>/<data_run>/all/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

- classification readers still accept older category-first trees such as:

```text
results/classification/<dataset>/<data_run>/all/<condition_id>/
```

- classification summary readers prefer `latest/.../CURRENT.json` when such pointers are
  present and otherwise fall back to legacy category-first directories
- classification feature readers prefer model latest pointers from `results/baselines/`
  and `results/experiments/`; embedding suffixes such as `_mpnet`, `_bge`, and
  `_glove100` become separate feature names
- `evaluation classify --embedding-variant mpnet` limits embedding-aware feature inputs
  to matching variants while still keeping non-embedding legacy models
- `evaluation classify --feature-resolve-mode strict` raises on invalid latest pointers;
  the default `all` mode skips invalid pointers and falls back to legacy paths when no
  current pointer matches
- `evaluation summarize-classification --resolve-mode strict` disables auto-selection of
  the newest match and raises when more than one candidate fits
- several analysis and reporting readers resolve model artifacts through the
  latest-aware path helpers

## Summary Sidecars

The `summaries/` trees are the interface between this repository and the manuscript
repository: the paper's `make sync` copies only the `*.scores.json` sidecars and aggregates
them itself. Everything else in `summaries/` (`.runs.csv`, `.runs.json`, `.tex`,
`run_coverage*.csv`) is a review artifact of this repository and is not consumed downstream.

Layout and stems:

```text
results/classification/summaries/<dataset>/<data_run>/<classifier>/<encoder>/<metric>_<dataset>_<data_run>_<classifier>_<encoder>_<assignment>_<K>topic.scores.json
results/topic_analysis/coherence/summaries/<dataset>/<data_run>/<encoder>/coherence_<dataset>_<data_run>_<measure>_<encoder>_<K>topic.scores.json
results/topic_analysis/entropy_based/summaries/<dataset>/<data_run>/<encoder>/entropy_<dataset>_<data_run>_<encoder>_<K>topic.scores.json
results/topic_analysis/topic_pairs/summaries/<dataset>/<data_run>/<encoder>/topic_pairs_<dataset>_<data_run>_<encoder>_<K>topic.scores.json
results/timing/summaries/<dataset>/<encoder>/timing_<dataset>_<data_run>_<encoder>_<K>topic.scores.json
```

- `<encoder>` is the short encoder identifier (`minilm`, `mpnet`, `bge`, ...), never the
  `_norm`-suffixed run-directory form.
- `<assignment>` names the document-topic estimator of the vMF family (vSLDA and MvTM:
  `hard`, `soft`, `foldin`, `foldincounts`). `entropy-based-summary --output-dir` writes
  non-default estimators to a sibling tree `summaries_<assignment>/` with the same stems.
- A sidecar holds raw per-run values and the provenance of every cell (`metric`, `dataset`,
  `data_run`, `topics`, `iterations`, `classifiers`, `embedding_variants`, `vmf_assignment`,
  `models`, `categories`, `scores`, `provenance`); nothing is aggregated.
- `summarize-classification` writes its sidecars next to `--output-path`, so a summary
  produced with a temporary output path leaves no trace here. The manuscript's variant
  sidecars (Gaussian prior-scale and covariance variants of the sentence Gaussian LDA
  baseline, the vSLDA hyperparameter sweep) are produced that way by the paper
  repository's sync script and exist only there.
- The timing summary spells the dataset with a hyphen (`20newsgroup-timing`) while the run
  directories use `results/experiments/20newsgroup_timing/`.

## Latest Pointers

`CURRENT.json` exists to decouple stable human-readable directory names from strict run
identity.

It typically records:

- `schema` and `schema_version`
- `task`
- `display_key`
- `dataset`, `data_run`, `category`
- `archive_dir`
- `started_at`
- `execution_id`
- `condition_fingerprint`
- `embedding_variant` and `encoder_config` for embedding-aware model outputs
- artifact filenames relative to the archived execution directory

Experiment, baseline, classification, and analysis writers create `CURRENT.json` during
normal execution. Older layouts remain a reader-side compatibility concern; normal
workflows should use the latest pointer when it exists.

## Display Key Convention

Display keys stay short and human-readable.

- run roots such as vMF and baseline use the shared suffix `k<num_topics>_it<iteration>`
- embedding-aware topic models append the short embedding identifier after the component
  suffix when present, for example `k10_it3_c1_mpnet`, `k10_it3_bge`, or
  `k20_it0_glove100`
- known encoder identifiers are `minilm`, `mpnet`, `bge`, `ruri`, and `usif`;
  unknown model names fall back to a slugified model-name tail
- known word-vector identifiers include `glove100`, `glove50`, and `wikient100`;
  unknown word-vector names fall back to a `wordvec_<slug>` label
- parameter variants follow the encoder, decimals written with `p`:
  `sentence_gaussianlda` appends `_norm` (BoW-normalized embeddings, the default since
  2026-08-25; the earlier `_raw` runs were removed), `_psi0-<scale>` (Gaussian prior
  scale) and, for reduced covariances, `_cov-diag` / `_cov-iso`, e.g.
  `k20_it2_minilm_norm_psi0-0p1_cov-iso`; vSLDA hyperparameter-sweep runs append their
  label, e.g. `k20_it0_c1_minilm_alpha0-0p1`, `..._kappa0-100`, `..._b-20`, `..._t-30`,
  `..._zeta-8`
- evaluation outputs may prepend only the axes that are not already encoded by the
  directory tree, for example `bleilda_train_k20_it0`
- strict run identity stays in `metadata.json` and `CURRENT.json` through
  `condition_fingerprint`, `execution_id`, and `started_at`
- path names do not encode long prompts, model kwargs, tokenizer kwargs, or pooling
  details; those are recorded in `metadata.json`, `CURRENT.json`, and the condition
  fingerprint

## Resolution Policy

- `CURRENT.json` is the canonical latest pointer; symlinks are not part of the documented
  contract
- readers should prefer `latest/.../CURRENT.json` and fall back to legacy directories
  only for backward compatibility
- result directories without embedding suffixes for embedding-aware runners should be
  treated as legacy or incomplete unless their metadata identifies the condition

## Metadata Files

Both vMF and baseline outputs persist `metadata.json` with:

- a schema name
- a schema version
- dataset, topic, iteration, and category axes
- preprocessing settings used for the run
- model or runner identity
- `embedding_variant` and `encoder_config` when the run depends on a sentence embedding
  model or word-vector embedding source

Baseline metadata additionally records comparison fields such as:

- `runner_key`
- `runner_family`
- `parameter_variant`
- `preprocessing_variant`
- `baseline_params`

These fields allow evaluation outputs to distinguish baseline variants without inferring
them from directory names alone.

## Summary Files

Experiment roots persist `summary.json` using the shared shape:

```json
{
  "_meta": {},
  "results": {}
}
```

Each run record is designed to be filterable by:

- data selection
- run axes
- execution policy
- runtime measurements
- artifact paths
- baseline comparison metadata when present

Evaluation JSON outputs follow the same top-level shape so reporting code can treat them
consistently.

## Preprocessing Artifacts

When shared preprocessing is persisted, the stored objects capture the document-level
views needed by multiple models and analysis tasks:

- raw text
- raw sentence strings
- tokenized sentence views
- joined sentence text
- document tokens

This keeps downstream comparison tied to the actual preprocessing configuration used at
run time.

Every run writes its own copy (`train_preprocessed.pkl` / `test_preprocessed.pkl` for
vSLDA, `preprocessed_corpus.pkl` under both `params/` and `infer/` for baselines), yet the
content depends only on dataset, category and preprocessing settings, so the copies are
byte-identical across models, K and seeds. They are never modified after writing, so
`hardlink -c` over `results/baselines results/experiments` may replace them with hard
links to reclaim space without changing any reader.

## Versioning

- experiment summaries, evaluation payloads, and artifact metadata are versioned separately
- schema changes should increment the corresponding `schema_version`
- backward compatibility is handled by explicit reader fallbacks; new writers should use
  the documented `latest/archive` layouts
