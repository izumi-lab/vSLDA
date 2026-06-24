from __future__ import annotations

import typer

from src.cli.data_commands import register_data_commands
from src.cli.evaluation_commands import register_evaluation_commands
from src.cli.experiment_commands import register_experiment_commands

app = typer.Typer(help="CLI for dataset preparation, experiments, and evaluation.")
data_app = typer.Typer(help="Dataset preparation commands.")
experiments_app = typer.Typer(
    help="Experiment artifact generation commands. These do not run evaluation."
)
evaluation_app = typer.Typer(help="Explicit evaluation commands.")

app.add_typer(data_app, name="data")
app.add_typer(experiments_app, name="experiments")
app.add_typer(evaluation_app, name="evaluation")

register_data_commands(data_app)
register_experiment_commands(experiments_app)
register_evaluation_commands(evaluation_app)


def main() -> None:
    app()
