"""Init command - create template test data files."""

import json
from pathlib import Path

import typer

from ..utils.output import console, error, success, warning

# Template data structures with English examples
TEMPLATES = {
    "basic": {
        "metadata": {
            "version": "1.0.0",
            "language": "en",
            "domain": "general",
            "description": "Basic example test data",
        },
        "tasks": {
            "semantic_similarity": {
                "name": "semantic_similarity",
                "enabled": True,
                "triplets": [
                    {
                        "id": "triplet_1",
                        "anchor": "This is a test sentence",
                        "positive": "This is a sample sentence",
                        "negative": "The weather is nice today",
                        "expected_similarity": {"anchor_positive": 0.8, "anchor_negative": 0.2},
                    },
                    {
                        "id": "triplet_2",
                        "anchor": "Machine learning is fascinating",
                        "positive": "Artificial intelligence is interesting",
                        "negative": "I like pizza",
                        "expected_similarity": {"anchor_positive": 0.75, "anchor_negative": 0.1},
                    },
                ]
            }
        },
    },
    "ir": {
        "metadata": {
            "version": "1.0.0",
            "language": "en",
            "domain": "general",
            "description": "Information retrieval example",
        },
        "tasks": {
            "information_retrieval": {
                "name": "information_retrieval",
                "enabled": True,
                "corpus": {
                    "doc1": "Istanbul is Turkey's most populous city and economic center",
                    "doc2": "Ankara is the capital of Turkey",
                    "doc3": "Mount Ararat is the highest mountain in Turkey",
                    "doc4": "Turkish coffee is a traditional beverage"
                },
                "queries": {
                    "q1": "What is Turkey's capital?",
                    "q2": "Tell me about Istanbul"
                },
                "relevant_docs": {
                    "q1": {"doc2": 2},
                    "q2": {"doc1": 2}
                }
            }
        },
    },
    "similarity": {
        "metadata": {
            "version": "1.0.0",
            "language": "en",
            "domain": "general",
            "description": "Comprehensive semantic similarity examples",
        },
        "tasks": {
            "semantic_similarity": {
                "name": "semantic_similarity",
                "enabled": True,
                "triplets": [
                    {
                        "id": "sim_1",
                        "anchor": "The cat is sitting on the mat",
                        "positive": "A feline is resting on the rug",
                        "negative": "Dogs are playing in the park",
                        "expected_similarity": {"anchor_positive": 0.85, "anchor_negative": 0.15},
                    },
                    {
                        "id": "sim_2",
                        "anchor": "I love programming in Python",
                        "positive": "Python is my favorite programming language",
                        "negative": "I enjoy cooking Italian food",
                        "expected_similarity": {"anchor_positive": 0.8, "anchor_negative": 0.1},
                    },
                    {
                        "id": "sim_3",
                        "anchor": "The stock market crashed today",
                        "positive": "Today's market saw a significant downturn",
                        "negative": "The weather is beautiful this morning",
                        "expected_similarity": {"anchor_positive": 0.75, "anchor_negative": 0.05},
                    },
                ]
            }
        },
    },
    "robustness": {
        "metadata": {
            "version": "1.0.0",
            "language": "en",
            "domain": "general",
            "description": "Linguistic robustness test examples",
        },
        "tasks": {
            "linguistic_robustness": {
                "name": "linguistic_robustness",
                "enabled": True,
                "morphology": [
                    {
                        "id": "morph_1",
                        "original": "The dog runs quickly",
                        "modified": "The dogs run quickly",
                        "category": "plural",
                        "expected_similarity": "high",
                    },
                    {
                        "id": "morph_2",
                        "original": "She is writing a book",
                        "modified": "She wrote a book",
                        "category": "tense",
                        "expected_similarity": "high",
                    },
                ],
                "typo": [
                    {
                        "id": "typo_1",
                        "correct": "This is a sentence",
                        "typo": "This is a sentense",
                        "type": "spelling",
                        "expected_similarity": "high",
                    },
                    {
                        "id": "typo_2",
                        "correct": "Machine learning",
                        "typo": "Machne learning",
                        "type": "missing_letter",
                        "expected_similarity": "high",
                    },
                ],
                "negation": [
                    {
                        "id": "neg_1",
                        "sentence1": "The movie was good",
                        "sentence2": "The movie was not good",
                        "type": "negation_word",
                        "expected_similarity": "low",
                    },
                    {
                        "id": "neg_2",
                        "sentence1": "I like coffee",
                        "sentence2": "I don't like coffee",
                        "type": "negation_word",
                        "expected_similarity": "low",
                    },
                ],
            }
        },
    },
}


def init(
    template: str = typer.Option(
        "basic",
        "--template",
        "-t",
        help="Template type: basic, ir, similarity, robustness",
    ),
    output: Path = typer.Option(
        "test_data.json", "--output", "-o", help="Output JSON file path"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Overwrite if file exists"
    ),
):
    """Create a template test data file.

    Examples:
        # Create basic template
        semeval init

        # Create IR template
        semeval init --template ir -o ir_test.json

        # Overwrite existing file
        semeval init --template similarity -o test.json --force
    """
    try:
        console.print(f"\n📝 Creating test data template: [cyan]{template}[/cyan]\n")

        # Check if template exists
        if template not in TEMPLATES:
            error(f"Unknown template: {template}")
            console.print("\n[bold]Available templates:[/bold]")
            for name, data in TEMPLATES.items():
                console.print(f"  • [cyan]{name}[/cyan]: {data['metadata']['description']}")
            raise typer.Exit(code=1)

        # Check if file exists
        if output.exists() and not force:
            error(f"File already exists: {output}")
            console.print("💡 [dim]Use --force to overwrite[/dim]")
            raise typer.Exit(code=1)

        # Get template data
        template_data = TEMPLATES[template]

        # Write to file
        with open(output, "w", encoding="utf-8") as f:
            json.dump(template_data, f, indent=2, ensure_ascii=False)

        success(f"Template created: {output}")

        # Print info about the template
        console.print("\n[bold]📊 Template Contents:[/bold]")
        metadata = template_data["metadata"]
        console.print(f"  Version: [cyan]{metadata['version']}[/cyan]")
        console.print(f"  Language: [cyan]{metadata['language']}[/cyan]")
        console.print(f"  Domain: [cyan]{metadata['domain']}[/cyan]")

        # Task info
        tasks = template_data.get("tasks", {})
        console.print(f"\n[bold]Included Tasks:[/bold]")
        for task_name in tasks.keys():
            console.print(f"  ✓ [green]{task_name.replace('_', ' ').title()}[/green]")

        # Next steps
        console.print("\n[bold]📝 Next Steps:[/bold]")
        console.print(f"  1. Edit [cyan]{output}[/cyan] to customize the data")
        console.print(f"  2. Validate: [yellow]semeval validate {output}[/yellow]")
        console.print(
            f"  3. Run evaluation: [yellow]semeval eval --model <model-name> --data {output}[/yellow]"
        )

        console.print(
            "\n💡 [dim]Tip: You can combine multiple tasks in a single file[/dim]"
        )

    except Exception as e:
        error(f"{type(e).__name__}: {e}")
        raise typer.Exit(code=1)
