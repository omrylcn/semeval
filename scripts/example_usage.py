"""Example usage of the SemEval package.

This script demonstrates how to use the main API for evaluating
semantic embedding models with all 4 tasks and configuration system.
"""

from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import semeval
from semeval import (
    TaskRunner,
    SentenceTransformerEncoder,
    load_settings
)

print("="*70)
print(f"SemEval Package v{semeval.__version__}")
print(f"By {semeval.__author__}")
print("="*70)

# Data path
data_path = Path(__file__).parent.parent / "data" / "test_data.json"

print("\n📦 Example 1: Basic Usage with All 4 Tasks")
print("-"*70)

# Create encoder
print("Creating encoder...")
encoder = SentenceTransformerEncoder(
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
)
print(f"✅ Model: {encoder._model_name}")
print(f"✅ Embedding dim: {encoder.get_embedding_dim()}D")

# Create runner
print("\nCreating TaskRunner...")
runner = TaskRunner(encoder=encoder, verbose=False)
print("✅ Runner created")

# Run all 4 tasks
print("\nRunning all 4 evaluation tasks...")
print("  - Information Retrieval")
print("  - Semantic Similarity")
print("  - Linguistic Robustness")
print("  - Vector Arithmetic")

result = runner.run(str(data_path))

# Display summary
summary = result.get_summary()
print(f"\n✅ Evaluation completed in {summary['total_runtime']:.2f}s")

print("\nResults Summary:")
for task_name, task_info in summary['tasks'].items():
    status_icon = "✅" if task_info['status'] == 'success' else "❌"
    print(f"\n{status_icon} {task_name.upper()}")
    print(f"   Status: {task_info['status']}")
    print(f"   Runtime: {task_info['runtime']:.2f}s")

    if task_info['status'] == 'success':
        # Show 2-3 key metrics per task
        metrics = task_info['metrics']

        if task_name == 'information_retrieval':
            print(f"   NDCG@10: {metrics.get('cosine-NDCG@10', 0):.4f}")
            print(f"   MRR@10: {metrics.get('cosine-MRR@10', 0):.4f}")

        elif task_name == 'semantic_similarity':
            print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")
            print(f"   Avg Margin: {metrics.get('avg_margin', 0):.3f}")

        elif task_name == 'linguistic_robustness':
            print(f"   Overall Robustness: {metrics.get('overall_robustness', 0):.2%}")

        elif task_name == 'vector_arithmetic':
            print(f"   Accuracy: {metrics.get('accuracy', 0):.2%}")

print("\n" + "="*70)
print("📦 Example 2: Using Configuration System")
print("-"*70)

# Load settings from config
print("Loading config...")
settings = load_settings()  # Loads config.yaml
print(f"✅ Config loaded")
print(f"   Model: {settings.model.name}")
print(f"   Device: {settings.model.device}")
print(f"   Output dir: {settings.output.base_dir}")
print(f"   Verbose: {settings.logging.verbose}")

# Create encoder using config
encoder_from_config = SentenceTransformerEncoder(
    settings.model.name,
    device=settings.model.device
)

# Create runner with settings
runner_with_config = TaskRunner(
    encoder=encoder_from_config,
    settings=settings
)

print("\nRunning with config...")
result_config = runner_with_config.run(str(data_path))
print(f"✅ Completed in {result_config.total_runtime:.2f}s")

print("\n" + "="*70)
print("📦 Example 3: Run Specific Task")
print("-"*70)

# Run only Semantic Similarity task
print("Running Semantic Similarity task only...")
ss_result = runner.run_task("semantic_similarity", str(data_path))

print(f"✅ Task: {ss_result.task_name}")
print(f"✅ Status: {ss_result.status}")
print(f"✅ Runtime: {ss_result.runtime_seconds:.2f}s")

# Get specific metrics
accuracy = ss_result.get_metric('accuracy')
avg_margin = ss_result.get_metric('avg_margin')

print(f"\nKey Metrics:")
print(f"  Triplet Accuracy: {accuracy:.2%}")
print(f"  Average Margin: {avg_margin:.3f}")

print("\n" + "="*70)
print("📦 Example 4: Export Results")
print("-"*70)

from semeval.postprocess import ResultsExporter

print("Creating exporter...")
exporter = ResultsExporter()

# Create output directory
output_dir = Path(__file__).parent.parent / "output" / "examples"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Exporting results to: {output_dir}")

# Export to different formats
print("\nExporting formats:")
print("  - CSV...")
exporter.export_csv(result, str(output_dir / "results.csv"))
print("  - JSON...")
exporter.export_json(result, str(output_dir / "results.json"))
print("  - Markdown...")
exporter.export_markdown(result, str(output_dir / "results.md"))

# Per-task export
print("  - Per-task files...")
task_paths = exporter.export_per_task(result, str(output_dir))

print(f"\n✅ Exported {len(task_paths)} tasks")
for task_name, paths in task_paths.items():
    print(f"   {task_name}: {len(paths)} files")

print("\n" + "="*70)
print("📦 Example 5: Comprehensive Report")
print("-"*70)

from semeval.postprocess import ReportGenerator

print("Generating comprehensive report...")
generator = ReportGenerator()

generator.generate_report(
    result,
    str(output_dir / "comprehensive_report.md"),
    model_name="BERT Turkish (Example)",
    include_recommendations=True
)

print(f"✅ Report saved to: {output_dir / 'comprehensive_report.md'}")

print("\n" + "="*70)
print("📦 Example 6: Environment-Specific Configs")
print("-"*70)

# Load dev config
print("Loading dev config...")
dev_settings = load_settings(env="dev")
print(f"✅ Dev config:")
print(f"   Verbose: {dev_settings.logging.verbose}")
print(f"   Log level: {dev_settings.logging.level}")

# Load prod config
print("\nLoading prod config...")
prod_settings = load_settings(env="prod")
print(f"✅ Prod config:")
print(f"   Verbose: {prod_settings.logging.verbose}")
print(f"   Log level: {prod_settings.logging.level}")

print("\n" + "="*70)
print("✅ All examples completed successfully!")
print("="*70)
print("\n💡 Next steps:")
print("  - Check USAGE.md for detailed documentation")
print("  - Explore config.yaml for configuration options")
print("  - See output/examples/ for exported results")
print("  - Run test_all_tasks.py to test all tasks")
print("="*70)
