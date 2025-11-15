"""Evaluation command - run model evaluation."""

import time
from pathlib import Path
from typing import Optional

import typer

from ..utils.output import console, create_progress, error, print_banner, success


def eval(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Model name or path (HuggingFace model or SentenceTransformer)",
    ),
    data: Path = typer.Option(
        ..., "--data", "-d", help="Path to test data JSON file", exists=True
    ),
    output: Path = typer.Option(
        "results/", "--output", "-o", help="Output directory for results"
    ),
    tasks: Optional[str] = typer.Option(
        None,
        "--tasks",
        "-t",
        help="Comma-separated list of tasks (e.g., 'ir,similarity'). Default: all tasks",
    ),
    device: str = typer.Option(
        "auto", "--device", help="Device to use: auto, cpu, cuda, mps"
    ),
    encoder_type: str = typer.Option(
        "sentence-transformer",
        "--encoder",
        "-e",
        help="Encoder type: sentence-transformer or huggingface",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to YAML/JSON config file (overrides other options)",
        exists=True,
    ),
):
    """Run evaluation on a model with test data.

    Examples:
        # Basic evaluation
        semeval eval --model "model-name" --data test.json

        # Specific tasks only
        semeval eval -m "model-name" -d test.json --tasks "ir,similarity"

        # Use config file
        semeval eval --config config.yaml
    """
    try:
        # Print banner
        if not verbose:
            print_banner()

        # Load config if provided
        if config:
            console.print(f"📄 Loading config from: [cyan]{config}[/cyan]")
            # TODO: Implement config-based initialization
            error("Config-based evaluation not yet implemented. Use CLI options.")
            raise typer.Exit(code=1)

        # Create output directory
        output.mkdir(parents=True, exist_ok=True)

        # Lazy import heavy dependencies only when needed
        from semeval.core.encoders import HuggingFaceEncoder, SentenceTransformerEncoder
        from semeval.core.runner import TaskRunner

        # Initialize encoder
        console.print(f"\n📦 Loading model: [bold cyan]{model}[/bold cyan]")

        with console.status("[bold green]Loading model..."):
            if encoder_type == "sentence-transformer":
                encoder = SentenceTransformerEncoder(model, device=device)
            elif encoder_type == "huggingface":
                encoder = HuggingFaceEncoder(model, device=device)
            else:
                error(f"Unknown encoder type: {encoder_type}")
                raise typer.Exit(code=1)

            embedding_dim = encoder.get_embedding_dim()

        success(f"Model loaded successfully ({embedding_dim} dimensions)")

        # Parse tasks if specified
        task_list = None
        if tasks:
            task_list = [t.strip() for t in tasks.split(",")]
            console.print(f"📊 Running tasks: [yellow]{', '.join(task_list)}[/yellow]")
        else:
            console.print("📊 Running all available tasks")

        # Create runner
        runner = TaskRunner(encoder=encoder, verbose=verbose)

        # Run evaluation with progress bar
        console.print("\n🚀 Starting evaluation...\n")

        start_time = time.time()

        with create_progress() as progress:
            task_id = progress.add_task(
                "[cyan]Running evaluation...", total=None  # Indeterminate
            )

            # Run evaluation
            result = runner.run(str(data))

            progress.update(task_id, completed=True)

        elapsed = time.time() - start_time

        # Print results
        console.print(f"\n✅ [bold green]Evaluation complete![/bold green]")
        console.print(f"⏱️  Total time: [cyan]{elapsed:.2f}s[/cyan]\n")

        # Print summary
        summary = result.get_summary()
        console.print("[bold]📊 Results Summary:[/bold]")

        for task_name, metrics in summary.items():
            if task_name in ["total_runtime", "timestamp"]:
                continue

            console.print(f"\n[bold cyan]{task_name.replace('_', ' ').title()}[/bold cyan]:")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, (int, float)):
                        console.print(f"  {metric}: [green]{value:.4f}[/green]")
                    else:
                        console.print(f"  {metric}: [yellow]{value}[/yellow]")

        # Save results
        output_file = output / f"{model.replace('/', '_')}_{int(time.time())}.json"
        result.save(str(output_file), format="json")

        console.print(f"\n💾 Results saved to: [cyan]{output_file}[/cyan]")

        # Tip
        console.print(
            "\n💡 [dim]Tip: Use 'semeval report' to generate HTML report from results[/dim]"
        )

    except KeyboardInterrupt:
        console.print("\n⚠️  [yellow]Evaluation interrupted by user[/yellow]")
        raise typer.Exit(code=130)
    except Exception as e:
        error(f"{type(e).__name__}: {e}")
        if verbose:
            console.print_exception()
        raise typer.Exit(code=1)
