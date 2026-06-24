# Architecture

This repository is organized around a CLI-first workflow, shared preprocessing, and
persisted artifacts that can be reused by evaluation and analysis tasks.

## Main Modules

- `src/cli`
  Typer application and command groups exposed as `spherical-sentence-topics`.
  `app.py` assembles `experiment_commands.py`, `data_commands.py`, and
  `evaluation_commands.py`.
- `src/data`
  Dataset preparation, split loading, dataset catalog helpers, and shared preprocessing.
  Dataset-specific preparation logic is kept in focused modules behind the CLI
  data commands.
- `src/experiments`
  Preset loading, CLI override resolution, job planning, and experiment execution.
  Config loading is split across `config_schema.py`, `config_loader.py`, and
  `config_parsers.py`; top-level comparison orchestration lives in
  `comparison_runner.py`, job expansion and data-run resolution live in
  `job_planning.py`, and per-category execution is split across `execution.py`,
  `vmf_runner.py`, `baseline_runner.py`, and `summary_builder.py`.
- `src/baselines`
  Baseline registries, typed params, adapters, and repo-owned train/infer/persistence
  layers. Shared adapter flow lives in `adapter_specs.py` and `adapter_runtime.py`,
  while `adapters.py` remains the compatibility surface.
- `src/models`
  vMF model implementations and model lookup.
- `src/evaluation`
  Evaluation task registry, reporting helpers, classification workflows, and analysis
  tasks. Classification logic lives under the `classification/` package, while
  diagnostics, geometry-based analysis, and word-based analysis are grouped under
  focused subpackages. `word_based/metrics.py` is the main task entry point and
  shares corpus loading, model input resolution, CLI parsing, and reporting helpers
  with neighboring modules.
- `src/core`
  Shared path resolution and metadata/schema helpers used across the repository.
  Path logic is decomposed into roots, builders, resolution, and latest-pointer helpers,
  with `paths.py` kept as the compatibility import surface.

## Design Boundaries

- the CLI and persisted artifacts are the intended public contracts
- modules under `src/` are internal implementation details unless documented otherwise
- reusable logic belongs under `src/`; ad-hoc local scripts should not become public contracts
- evaluation tasks should read stable artifacts rather than re-implement training-time logic
- compatibility modules such as `src/core/paths.py`, `src/experiments/config.py`, and
  `src/baselines/adapters.py` may re-export helpers, but new code should prefer the
  narrower submodules when it owns that area

## Output Contract

- experiment and evaluation JSON outputs prefer the shared shape `{"_meta": ..., "results": ...}`
- experiment runs persist `summary.json` at the dataset root
- model runs persist `metadata.json` so downstream comparisons can recover the run axes without relying on path parsing alone

## Baseline Boundary

- baseline models should conform to the shared runner and artifact contract
- repo-owned orchestration, persistence, and metadata handling stay under `src/baselines`
- low-level Gaussian-family helpers remain isolated from the rest of the repository behind repo-owned wrappers

## Development Rules

- the CLI and persisted artifacts are the primary public contracts
- most modules under `src/` are internal implementation details, so prefer existing
  local helpers and compatibility surfaces before adding new public APIs
- tests should protect repository contracts and high-risk integration boundaries:
  config loading, path and artifact layout, metadata schemas, baseline adapters,
  CLI smoke paths, and evaluation output shape
- tracked source and generated outputs should stay separate; version `configs/`,
  `src/`, `tests/`, and `docs/`, and write generated data and results under
  `data/`, `results/`, or `tmp/`
- if a change adds a generated path, update `.gitignore` in the same change

## Extension Points

- add dataset logic under `src/data` and register aliases in `src/data/catalog.py`
  when needed
- add baseline orchestration under `src/baselines/models/`, typed params in
  `src/baselines/params.py`, and registry entries in `src/baselines/registry.py`
- add evaluation tasks under `src/evaluation`, reuse shared schema/reporting
  helpers, and register tasks in `src/evaluation/registry.py`
- record new encoder or algorithm variants in persisted metadata so comparisons
  remain explicit

See also:

- `docs/user_guide.md`
- `docs/artifacts.md`
- `docs/preprocessing.md`
