"""Test post-processing with all 4 tasks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from semeval import TaskRunner, SentenceTransformerEncoder
from semeval.postprocess import ResultsExporter, ReportGenerator

def main():
    """Test post-processing features with all 4 tasks."""
    print("="*70)
    print("Testing Post-Processing with All 4 Tasks")
    print("="*70)

    # Create encoder
    print("\n📦 Loading model...")
    encoder = SentenceTransformerEncoder(
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    )
    print(f"✅ Model loaded: {encoder.model_name}")

    # Run evaluation
    runner = TaskRunner(encoder=encoder, verbose=False)
    data_path = Path(__file__).parent.parent / "data" / "test_data.json"

    print(f"\n🔄 Running evaluation...")
    result = runner.run(str(data_path))
    print(f"✅ Evaluation complete: {result.total_runtime:.2f}s")

    # Create output directory
    output_dir = Path(__file__).parent.parent / "output" / "all_tasks_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n📁 Output directory: {output_dir}")
    print("")

    # Test ResultsExporter
    print("="*70)
    print("Testing ResultsExporter")
    print("="*70)
    print("")

    exporter = ResultsExporter()

    # Export CSV
    print("1️⃣ Exporting to CSV...")
    df = exporter.export_csv(result, str(output_dir / "results.csv"))
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {list(df.columns)}")
    print("")

    # Export JSON
    print("2️⃣ Exporting to JSON...")
    exporter.export_json(result, str(output_dir / "results.json"))
    print("")

    # Export Markdown
    print("3️⃣ Exporting to Markdown...")
    exporter.export_markdown(result, str(output_dir / "results.md"))
    print("")

    # Test ReportGenerator
    print("="*70)
    print("Testing ReportGenerator")
    print("="*70)
    print("")

    generator = ReportGenerator()

    print("4️⃣ Generating comprehensive report...")
    generator.generate_report(
        result,
        str(output_dir / "comprehensive_report.md"),
        model_name="BERT Turkish (All 4 Tasks)",
        include_recommendations=True
    )
    print("")

    # Test Per-Task Export
    print("="*70)
    print("Testing Per-Task Export")
    print("="*70)

    print("\n5️⃣ Exporting per-task files...")
    task_paths = exporter.export_per_task(result, str(output_dir))

    # Summary
    print("\n="*70)
    print("✅ All Post-Processing Tests Complete!")
    print("="*70)
    print(f"\n📂 Check output files in: {output_dir}")
    print("\nGenerated files:")
    for file in sorted(output_dir.glob("*")):
        size = file.stat().st_size
        print(f"  - {file.name} ({size:,} bytes)")

if __name__ == "__main__":
    main()
