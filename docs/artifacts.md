# Artifacts

This repository uses persisted artifacts as a comparison contract between experiment runs
and downstream evaluation tasks.

## Result Roots

- `results/experiments/`
- `results/baselines/`
- `results/classification/`

Additional analysis tasks create sibling roots for topic analysis, diagnostics, and
visualization outputs.

Classification readers resolve feature inputs from model `latest/CURRENT.json`
pointers first. When multiple embedding-aware model outputs match the same
topic/iteration/category, classification treats them as separate feature sets and can
filter them with `--embedding-variant`.

Repo-owned analysis roots include:

- `results/topic_analysis/coherence/` for `word_based_metrics`
- `results/topic_analysis/geometry_based/` for `geometry_based_metrics`
- `results/topic_analysis/label_profile/` for `word_based_label_profile`
- `results/topic_count_analysis/` for `topic_count_diagnostics`
- `results/analysis/vmf_vs_baseline/` for `cross_model_pair_diagnostics`
- `results/visualization/` for `sentence_topic_inspection`

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

Baselines that do not use sentence or word embeddings, such as `bleilda` and
`sentlda`, keep the plain `k<num_topics>_it<iteration>` display key.

ETM writes document-topic distributions to `params/etm.pkl` and
`infer/<category>.pkl`, a soft-preferred copy to
`infer/<category>_doc_topic_soft.pkl`, and learned topic-word probabilities to
`params/topic_word_scores.pkl` with `params/vocabulary.json`. Word-based metrics
read this learned ETM beta distribution; classification uses the document-topic
artifacts.

## Evaluation Output Layout

Evaluation roots use the same `latest/archive` contract where practical, while readers
keep backward-compatible fallbacks for older category-first trees.

- `geometry_based_metrics` uses:

```text
results/topic_analysis/geometry_based/latest/<dataset>/<data_run>/<category>/<display_key>/CURRENT.json
results/topic_analysis/geometry_based/archive/YYYY-MM-DD/<dataset>/<data_run>/<category>/<display_key>/exec_YYYYMMDDTHHMMSSZ/
```

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

## Versioning

- experiment summaries, evaluation payloads, and artifact metadata are versioned separately
- schema changes should increment the corresponding `schema_version`
- backward compatibility is handled by explicit reader fallbacks; new writers should use
  the documented `latest/archive` layouts
