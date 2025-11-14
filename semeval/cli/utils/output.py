"""Rich console output utilities for beautiful CLI."""

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

# Global console instance
console = Console()


def success(message: str):
    """Print success message."""
    console.print(f"✅ [bold green]{message}[/bold green]")


def error(message: str):
    """Print error message."""
    console.print(f"❌ [bold red]Error:[/bold red] {message}")


def warning(message: str):
    """Print warning message."""
    console.print(f"⚠️  [bold yellow]Warning:[/bold yellow] {message}")


def info(message: str):
    """Print info message."""
    console.print(f"ℹ️  [bold blue]{message}[/bold blue]")


def create_progress():
    """Create a progress bar for long-running tasks."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )


def create_results_table(results: dict, title: str = "Results") -> Table:
    """Create a formatted results table.

    Parameters
    ----------
    results : dict
        Dictionary of metric name -> value
    title : str
        Table title

    Returns
    -------
    Table
        Rich table with formatted results
    """
    table = Table(title=title, show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", justify="right", style="green")

    for metric, value in results.items():
        # Format value based on type
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        elif isinstance(value, dict):
            # For nested dicts, show count
            formatted_value = f"{len(value)} items"
        else:
            formatted_value = str(value)

        table.add_row(metric, formatted_value)

    return table


def print_banner():
    """Print SemEval banner."""
    banner = """
[bold cyan]
╔═══════════════════════════════════════════════╗
║                                               ║
║   🚀  SemEval - Semantic Evaluation v0.1.1    ║
║                                               ║
╚═══════════════════════════════════════════════╝
[/bold cyan]
    """
    console.print(banner)


def print_summary(summary: dict):
    """Print evaluation summary in a nice panel.

    Parameters
    ----------
    summary : dict
        Evaluation summary with metrics
    """
    content = f"""
[bold]Evaluation Summary[/bold]

Total Runtime: [cyan]{summary.get('total_runtime', 0):.2f}s[/cyan]
Tasks Completed: [green]{summary.get('tasks_completed', 0)}[/green]
Total Samples: [yellow]{summary.get('total_samples', 0)}[/yellow]
    """

    panel = Panel(
        content.strip(),
        title="📊 Results",
        border_style="green",
        padding=(1, 2),
    )
    console.print(panel)
