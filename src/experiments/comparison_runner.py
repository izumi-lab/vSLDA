from __future__ import annotations

import multiprocessing
import warnings
from pathlib import Path
from typing import List, Sequence

from src.core.artifacts import save_json
from src.experiments.config import (
    load_config,
    resolve_model_selection,
    resolve_run_selection,
)
from src.experiments.execution import process_category
from src.experiments.job_planning import (
    build_jobs,
    resolve_data_runs,
    resolve_parallelism_plan,
)
from src.experiments.summary_schema import SummaryRecord, build_summary_payload
from src.utils.logging import get_logger
from src.utils.random import DEFAULT_RANDOM_SEED, set_global_seed

# Avoid fork() warnings in multithreaded contexts by using spawn and silencing the deprecation notice.
try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError:
    pass
warnings.filterwarnings(
    "ignore",
    category=DeprecationWarning,
    module=r"multiprocessing\\.popen_fork",
)


def run_comparison(
    *,
    config_path: str | Path,
    models: str | None = None,
    seed: int | None = None,
    seed_base: int | None = None,
    num_workers: int | None = None,
    vmf_soft_temp: float | None = None,
    encoder_model: str | None = None,
    strip_terminal_normalize: bool | None = None,
    categories: Sequence[str] | None = None,
    num_topics: Sequence[int] | None = None,
    iterations: Sequence[int] | None = None,
) -> Path:
    if seed is not None and seed_base is not None:
        raise ValueError("Use either --seed or --seed_base, not both.")

    cfg = load_config(
        config_path,
        encoder_model=encoder_model,
        strip_terminal_normalize=strip_terminal_normalize,
    )
    runtime_cfg = getattr(cfg, "runtime", None)
    vmf_cfg = getattr(cfg, "vmf", None)
    vmf_inference_cfg = None if vmf_cfg is None else getattr(vmf_cfg, "inference", None)
    cfg_seed_base = getattr(runtime_cfg, "seed_base", DEFAULT_RANDOM_SEED)
    cfg_num_workers = getattr(runtime_cfg, "num_workers", 1)
    cfg_vmf_soft_temp = getattr(vmf_inference_cfg, "soft_temperature", 1.0)
    resolved_seed_base = cfg_seed_base if seed_base is None else seed_base
    resolved_num_workers = cfg_num_workers if num_workers is None else num_workers
    resolved_vmf_soft_temp = (
        cfg_vmf_soft_temp if vmf_soft_temp is None else vmf_soft_temp
    )

    if seed is None and resolved_seed_base is None:
        resolved_seed_base = DEFAULT_RANDOM_SEED

    if seed is not None:
        set_global_seed(seed, deterministic_torch=False)

    logger = get_logger("comparison")

    selected_models = resolve_model_selection(cfg, models=models)
    parallelism = resolve_parallelism_plan(
        requested_num_workers=resolved_num_workers,
        encoder_device=cfg.encoder.device,
        selected_models=selected_models,
    )
    selected_categories, num_topics_list, run_iterations = resolve_run_selection(
        cfg,
        categories=categories,
        num_topics=num_topics,
        iterations=iterations,
    )
    data_runs = resolve_data_runs(cfg)

    summary_records: List[SummaryRecord] = []
    for iteration in run_iterations:
        if seed is None and resolved_seed_base is not None:
            seed_for_iter = int(resolved_seed_base) + int(iteration)
            set_global_seed(seed_for_iter, deterministic_torch=False)
            logger.info(f"Seed set to {seed_for_iter} for iteration {iteration}")

        jobs = build_jobs(
            cfg=cfg,
            data_runs=data_runs,
            iterations=[iteration],
            num_topics_list=num_topics_list,
            categories=selected_categories,
            selected_models=selected_models,
            seed=seed,
            seed_base=resolved_seed_base,
            parallelism=parallelism,
            vmf_soft_temp=resolved_vmf_soft_temp,
        )

        if parallelism.category_num_workers > 1:
            with multiprocessing.Pool(
                processes=parallelism.category_num_workers
            ) as pool:
                records = pool.map(process_category, jobs)
                summary_records.extend(records)
        else:
            for job in jobs:
                record = process_category(job)
                summary_records.append(record)

    summary_path = cfg.output_root / "summary.json"
    save_json(
        build_summary_payload(
            dataset=cfg.dataset.name,
            summary_path=summary_path,
            records=summary_records,
        ),
        summary_path,
    )
    logger.info(f"Finished. Summary written to {summary_path}")
    return summary_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run vMF and/or borrowed baselines on the same CSV corpus."
    )
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--models", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--seed_base", type=int, default=None)
    parser.add_argument(
        "--num-workers",
        "--num_workers",
        dest="num_workers",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--vmf-soft-temp",
        "--vmf_soft_temp",
        dest="vmf_soft_temp",
        type=float,
        default=None,
    )
    parser.add_argument("--encoder-model", "--encoder_model", default=None)
    parser.add_argument(
        "--strip-terminal-normalize",
        dest="strip_terminal_normalize",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--keep-terminal-normalize",
        dest="strip_terminal_normalize",
        action="store_false",
    )
    parser.add_argument("--category", action="append", default=None)
    parser.add_argument("--topic", type=int, action="append", default=None)
    parser.add_argument("--iteration", type=int, action="append", default=None)
    arguments = parser.parse_args()
    run_comparison(
        config_path=arguments.config,
        models=arguments.models,
        seed=arguments.seed,
        seed_base=arguments.seed_base,
        num_workers=arguments.num_workers,
        vmf_soft_temp=arguments.vmf_soft_temp,
        encoder_model=arguments.encoder_model,
        strip_terminal_normalize=arguments.strip_terminal_normalize,
        categories=arguments.category,
        num_topics=arguments.topic,
        iterations=arguments.iteration,
    )
