"""Validate command - validate test data format."""

from pathlib import Path
from typing import Optional

import typer

from ..utils.output import console, error, success, warning


def validate(
    data: Path = typer.Argument(..., help="Path to test data JSON file", exists=True),
    strict: bool = typer.Option(
        False, "--strict", help="Enable strict validation (fail on warnings)"
    ),
    report: Optional[Path] = typer.Option(
        None, "--report", "-r", help="Save validation report to HTML file"
    ),
):
    """Validate test data format and schema.

    Examples:
        # Basic validation
        semeval validate test_data.json

        # Strict mode (fail on warnings)
        semeval validate test_data.json --strict

        # Generate HTML report
        semeval validate test_data.json --report validation_report.html
    """
    try:
        # Lazy import to avoid loading heavy dependencies
        from semeval.core.base_loader import DataValidationError
        from semeval.core.loaders import JSONDataLoader

        console.print(f"\n🔍 Validating: [cyan]{data}[/cyan]\n")

        # Load and validate using existing JSONDataLoader
        loader = JSONDataLoader()

        with console.status("[bold green]Validating data..."):
            test_data = loader.load(str(data))

        # Validation passed
        success("Schema validation passed!")

        # Print statistics
        console.print("\n[bold]📊 Data Statistics:[/bold]")

        # Check which tasks are enabled
        enabled_tasks = test_data.get_enabled_tasks()
        console.print(f"✓ Enabled tasks: [green]{', '.join(enabled_tasks)}[/green]")

        # Task-specific stats
        if test_data.tasks.information_retrieval:
            ir = test_data.tasks.information_retrieval
            console.print(f"  • IR: {len(ir.corpus)} docs, {len(ir.queries)} queries")

        if test_data.tasks.semantic_similarity:
            sim = test_data.tasks.semantic_similarity
            console.print(f"  • Similarity: {len(sim.triplets)} triplets")

        if test_data.tasks.linguistic_robustness:
            rob = test_data.tasks.linguistic_robustness
            total_pairs = 0
            if rob.morphology:
                total_pairs += len(rob.morphology)
            if rob.typo:
                total_pairs += len(rob.typo)
            if rob.negation:
                total_pairs += len(rob.negation)
            console.print(f"  • Robustness: {total_pairs} test pairs")

        if test_data.tasks.vector_arithmetic:
            arith = test_data.tasks.vector_arithmetic
            console.print(f"  • Arithmetic: {len(arith.analogies)} analogies")

        # Metadata info
        console.print("\n[bold]ℹ️  Metadata:[/bold]")
        console.print(f"  Version: [cyan]{test_data.metadata.version}[/cyan]")
        console.print(f"  Language: [cyan]{test_data.metadata.language}[/cyan]")
        if test_data.metadata.domain:
            console.print(f"  Domain: [cyan]{test_data.metadata.domain}[/cyan]")

        # Warnings (data quality checks)
        warnings_found = []

        # Check for very small datasets
        if test_data.tasks.semantic_similarity:
            if len(test_data.tasks.semantic_similarity.triplets) < 50:
                warnings_found.append(
                    "Semantic similarity has < 50 triplets (recommended: 100+)"
                )

        if test_data.tasks.information_retrieval:
            if len(test_data.tasks.information_retrieval.queries) < 10:
                warnings_found.append(
                    "Information retrieval has < 10 queries (recommended: 20+)"
                )

        # Print warnings
        if warnings_found:
            console.print("\n[bold yellow]⚠️  Warnings:[/bold yellow]")
            for w in warnings_found:
                warning(w)

            if strict:
                error("Validation failed due to warnings (strict mode enabled)")
                raise typer.Exit(code=1)

        # Generate report if requested
        if report:
            console.print(f"\n📄 Generating validation report: [cyan]{report}[/cyan]")
            # TODO: Implement HTML report generation
            warning("HTML report generation not yet implemented")

        # Final message
        console.print(
            f"\n✅ [bold green]Validation {'passed' if not warnings_found else 'passed with warnings'}[/bold green]"
        )

        if warnings_found and not strict:
            console.print("💡 [dim]Use --strict to treat warnings as errors[/dim]")

    except DataValidationError as e:
        error(f"Validation failed: {e.message}")
        if e.errors:
            console.print("\n[bold red]Validation Errors:[/bold red]")
            for err in e.errors:
                console.print(f"  • {err}")
        raise typer.Exit(code=1)
    except FileNotFoundError as e:
        error(f"File not found: {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        error(f"{type(e).__name__}: {e}")
        console.print_exception()
        raise typer.Exit(code=1)
