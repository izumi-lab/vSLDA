from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import typer

AUDIT_REVIEW_OUTPUT_DIR = Path("scripts/audit_review")

if TYPE_CHECKING:
    from src.data.sentence_quality import SentenceQualityConfig


def _build_sentence_quality_config(
    *,
    min_word_tokens: int,
    min_alpha_chars: int,
    max_word_tokens: int,
    min_bad_boundary_word_tokens: int,
    max_parenthetical_only_word_tokens: int,
    max_punctuation_ratio: float,
    max_upper_noise_ratio: float,
) -> SentenceQualityConfig:
    from src.data.sentence_quality import SentenceQualityConfig

    return SentenceQualityConfig(
        min_word_tokens=min_word_tokens,
        min_alpha_chars=min_alpha_chars,
        max_word_tokens=max_word_tokens,
        min_bad_boundary_word_tokens=min_bad_boundary_word_tokens,
        max_parenthetical_only_word_tokens=max_parenthetical_only_word_tokens,
        max_punctuation_ratio=max_punctuation_ratio,
        max_upper_noise_ratio=max_upper_noise_ratio,
    )


def _default_audit_review_output_path(input_path: Path, output_dir: Path) -> Path:
    dataset_name = input_path.parent.name or "dataset"
    split_name = input_path.stem or "split"
    return output_dir / f"{dataset_name}_{split_name}_audit_review.csv"


def register_data_commands(data_app: typer.Typer) -> None:
    @data_app.command("prepare-20newsgroup")
    def prepare_20newsgroup(
        output_dir: Path = typer.Option(Path("data/20newsgroup"), file_okay=False),
        min_word_tokens: int = typer.Option(4, min=1),
        min_alpha_chars: int = typer.Option(4, min=0),
        max_word_tokens: int = typer.Option(120, min=1),
        min_bad_boundary_word_tokens: int = typer.Option(50, min=1),
        max_parenthetical_only_word_tokens: int = typer.Option(8, min=0),
        max_punctuation_ratio: float = typer.Option(0.45, min=0.0, max=1.0),
        max_upper_noise_ratio: float = typer.Option(0.65, min=0.0, max=1.0),
    ) -> None:
        from src.data.newsgroups import prepare_20newsgroups

        prepare_20newsgroups(
            output_dir,
            quality_config=_build_sentence_quality_config(
                min_word_tokens=min_word_tokens,
                min_alpha_chars=min_alpha_chars,
                max_word_tokens=max_word_tokens,
                min_bad_boundary_word_tokens=min_bad_boundary_word_tokens,
                max_parenthetical_only_word_tokens=(max_parenthetical_only_word_tokens),
                max_punctuation_ratio=max_punctuation_ratio,
                max_upper_noise_ratio=max_upper_noise_ratio,
            ),
        )

    @data_app.command("prepare-nyt")
    def prepare_nyt_dataset(
        raw_path: Path = typer.Option(Path("data/nyt/raw/df_fine.pkl"), exists=True),
        output_dir: Path = typer.Option(Path("data/nyt"), file_okay=False),
        test_size: float = typer.Option(0.4, min=0.0, max=1.0),
        random_state: int = typer.Option(42),
        min_word_tokens: int = typer.Option(4, min=1),
        min_alpha_chars: int = typer.Option(4, min=0),
        max_word_tokens: int = typer.Option(120, min=1),
        min_bad_boundary_word_tokens: int = typer.Option(50, min=1),
        max_parenthetical_only_word_tokens: int = typer.Option(8, min=0),
        max_punctuation_ratio: float = typer.Option(0.45, min=0.0, max=1.0),
        max_upper_noise_ratio: float = typer.Option(0.65, min=0.0, max=1.0),
    ) -> None:
        from src.data.nyt import prepare_nyt

        prepare_nyt(
            raw_path=raw_path,
            output_dir=output_dir,
            test_size=test_size,
            random_state=random_state,
            quality_config=_build_sentence_quality_config(
                min_word_tokens=min_word_tokens,
                min_alpha_chars=min_alpha_chars,
                max_word_tokens=max_word_tokens,
                min_bad_boundary_word_tokens=min_bad_boundary_word_tokens,
                max_parenthetical_only_word_tokens=(max_parenthetical_only_word_tokens),
                max_punctuation_ratio=max_punctuation_ratio,
                max_upper_noise_ratio=max_upper_noise_ratio,
            ),
        )

    @data_app.command("audit-preprocessing")
    def audit_preprocessing(
        input_path: Path = typer.Option(..., exists=True, dir_okay=False),
        output_path: Path | None = typer.Option(None, dir_okay=False),
        output_dir: Path = typer.Option(AUDIT_REVIEW_OUTPUT_DIR, file_okay=False),
        summary_path: Path | None = typer.Option(None, dir_okay=False),
        text_column: str = typer.Option("data"),
        target_column: str | None = typer.Option("target_str"),
        delimiter: str = typer.Option(" / "),
        sample_size: int = typer.Option(100, min=0),
        seed: int = typer.Option(42),
        short_token_threshold: int = typer.Option(3, min=1),
        min_word_tokens: int = typer.Option(4, min=1),
        min_alpha_chars: int = typer.Option(4, min=0),
        max_word_tokens: int = typer.Option(120, min=1),
        min_bad_boundary_word_tokens: int = typer.Option(50, min=1),
        max_parenthetical_only_word_tokens: int = typer.Option(8, min=0),
        max_punctuation_ratio: float = typer.Option(0.45, min=0.0, max=1.0),
        max_upper_noise_ratio: float = typer.Option(0.65, min=0.0, max=1.0),
    ) -> None:
        from src.data.preprocessing_audit import audit_preprocessed_csv

        resolved_output_path = output_path or _default_audit_review_output_path(
            input_path,
            output_dir,
        )
        summary = audit_preprocessed_csv(
            input_path=input_path,
            output_path=resolved_output_path,
            summary_path=summary_path,
            text_column=text_column,
            target_column=target_column,
            delimiter=delimiter,
            sample_size=sample_size,
            seed=seed,
            short_token_threshold=short_token_threshold,
            config=_build_sentence_quality_config(
                min_word_tokens=min_word_tokens,
                min_alpha_chars=min_alpha_chars,
                max_word_tokens=max_word_tokens,
                min_bad_boundary_word_tokens=min_bad_boundary_word_tokens,
                max_parenthetical_only_word_tokens=(max_parenthetical_only_word_tokens),
                max_punctuation_ratio=max_punctuation_ratio,
                max_upper_noise_ratio=max_upper_noise_ratio,
            ),
        )
        typer.echo(f"Wrote review sample: {summary.output_path}")
        typer.echo(f"Wrote audit summary: {summary.summary_path}")

    @data_app.command("subset-20newsgroup")
    def subset_20newsgroup(
        src_dir: Path = typer.Option(Path("data/20newsgroup"), file_okay=False),
        dst_dir: Path = typer.Option(Path("data/20newsgroup_subset"), file_okay=False),
        per_label: int = typer.Option(20, min=1),
        seed: int = typer.Option(42),
    ) -> None:
        from src.data.newsgroups_subset import create_20newsgroups_subset

        create_20newsgroups_subset(
            src_dir=src_dir,
            dst_dir=dst_dir,
            per_label=per_label,
            seed=seed,
        )
