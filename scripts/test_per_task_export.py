"""Test per-task export functionality.

This script demonstrates exporting individual files for each task.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semeval.core import TaskRunner
from semeval.core.encoders import SentenceTransformerEncoder
from semeval.postprocess import ResultsExporter


def main():
    print("=" * 70)
    print("Testing Per-Task Export")
    print("=" * 70)
    print()

    # Load model
    print("📦 Loading model...")
    encoder = SentenceTransformerEncoder(
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    )
    print("✅ Model loaded")
    print()

    # Run evaluation
    print("🔄 Running evaluation...")
    runner = TaskRunner(encoder=encoder)
    result = runner.run("data/test_data.json")
    print(f"✅ Evaluation complete: {result.total_runtime:.2f}s")
    print()

    # Setup output
    output_dir = Path("output/per_task_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    print()

    # Test per-task export
    print("=" * 70)
    print("Testing Per-Task Export")
    print("=" * 70)

    exporter = ResultsExporter()

    # Export per-task files
    task_paths = exporter.export_per_task(
        result, str(output_dir), export_formats=["json", "markdown"]
    )

    print()
    print("=" * 70)
    print("✅ Per-Task Export Complete!")
    print("=" * 70)
    print()

    # Show what was created
    print("📂 Generated files:")
    for task_name, paths in task_paths.items():
        print(f"\n   {task_name}:")
        for format_type, file_path in paths.items():
            file_size = Path(file_path).stat().st_size
            print(f"      - {Path(file_path).name} ({file_size} bytes)")

    print()
    print("💡 Check files in:", output_dir)
    print()


if __name__ == "__main__":
    main()
