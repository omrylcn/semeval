"""Main CLI application entry point."""

import typer
from rich.console import Console

from .commands import compare, eval_cmd, init, report, validate

# Create Typer app
app = typer.Typer(
    name="semeval",
    help="🚀 SemEval - Semantic Embedding Evaluation Framework",
    add_completion=True,
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Create console for output
console = Console()

# Register commands
app.command(name="eval")(eval_cmd.eval)
app.command(name="validate")(validate.validate)
app.command(name="compare")(compare.compare)
app.command(name="report")(report.report)
app.command(name="init")(init.init)


@app.command()
def version():
    """Show SemEval version information."""
    from semeval import __version__

    console.print(f"[bold green]SemEval[/bold green] version [bold]{__version__}[/bold]")
    console.print("\nSemantic Embedding Evaluation Framework")
    console.print("https://github.com/omrylcn/semeval")


@app.command()
def info():
    """Show system and environment information."""
    import platform
    import sys

    import torch

    console.print("\n[bold]System Information[/bold]")
    console.print(f"Python: {sys.version.split()[0]}")
    console.print(f"Platform: {platform.system()} {platform.release()}")
    console.print(f"PyTorch: {torch.__version__}")
    console.print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        console.print(f"CUDA version: {torch.version.cuda}")
        console.print(f"GPU count: {torch.cuda.device_count()}")
    console.print(f"MPS available: {torch.backends.mps.is_available()}")


if __name__ == "__main__":
    app()
