"""Compare command - compare multiple models."""

from pathlib import Path

import typer

from ..utils.output import console, error, warning


def compare(
    models: str = typer.Option(
        ..., "--models", "-m", help="Comma-separated list of model names"
    ),
    data: Path = typer.Option(
        ..., "--data", "-d", help="Path to test data JSON file", exists=True
    ),
    output: Path = typer.Option(
        "comparison.html", "--output", "-o", help="Output HTML file for comparison"
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device to use: auto, cpu, cuda, mps"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
):
    """Compare multiple models on the same test data.

    Examples:
        # Compare two models
        semeval compare --models "model1,model2" --data test.json

        # Save to custom location
        semeval compare -m "model1,model2,model3" -d test.json -o results/compare.html
    """
    try:
        model_list = [m.strip() for m in models.split(",")]

        console.print(f"\n🔬 Comparing {len(model_list)} models:")
        for i, model in enumerate(model_list, 1):
            console.print(f"  {i}. [cyan]{model}[/cyan]")

        console.print(f"\n📊 Test data: [yellow]{data}[/yellow]")
        console.print(f"💾 Output: [yellow]{output}[/yellow]\n")

        # TODO: Implement model comparison
        warning(
            "Model comparison not yet implemented. This will evaluate each model and generate a comparison report."
        )

        console.print("\n[bold]Planned features:[/bold]")
        console.print("  • Side-by-side metric comparison")
        console.print("  • Statistical significance tests")
        console.print("  • Performance charts")
        console.print("  • Speed/accuracy trade-off analysis")

        console.print(
            "\n💡 [dim]For now, run 'semeval eval' separately for each model[/dim]"
        )

    except Exception as e:
        error(f"{type(e).__name__}: {e}")
        raise typer.Exit(code=1)
