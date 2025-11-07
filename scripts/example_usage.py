"""Example usage of the semeval package.

This script demonstrates how to use the main API for evaluating
semantic search models.
"""

from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import from main package API
import semeval
from semeval import TaskRunner, SentenceTransformerEncoder

print("="*70)
print(f"SemEval Package v{semeval.__version__}")
print(f"By {semeval.__author__}")
print("="*70)

print("\n📦 Example 1: Basic Usage with TaskRunner")
print("-"*70)

# Create encoder using main API
print("Creating encoder...")
encoder = SentenceTransformerEncoder("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")
print(f"✅ Loaded: {encoder.model_name} ({encoder.get_embedding_dim()}D)")

# Create runner
print("\nCreating TaskRunner...")
runner = TaskRunner(encoder=encoder, verbose=False)
print("✅ Runner created")

# Run all tasks
print("\nRunning evaluation...")
data_path = Path(__file__).parent.parent / "data" / "test_data.json"
result = runner.run(str(data_path))

# Display summary
summary = result.get_summary()
print(f"✅ Evaluation completed in {summary['total_runtime']:.2f}s")

print("\nResults:")
for task_name, task_info in summary['tasks'].items():
    print(f"\n  {task_name.upper()}")
    if task_info['status'] == 'success':
        # Show key metrics
        for metric, value in sorted(task_info['metrics'].items())[:5]:
            print(f"    {metric:30s}: {value:.4f}")

print("\n" + "="*70)
print("📦 Example 2: Run Single Task")
print("-"*70)

# Run only IR task
print("Running Information Retrieval task only...")
ir_result = runner.run_task("information_retrieval", str(data_path))

print(f"✅ Task: {ir_result.task_name}")
print(f"✅ Status: {ir_result.status}")
print(f"✅ Runtime: {ir_result.runtime_seconds:.2f}s")

# Get specific metrics
ndcg = ir_result.get_metric('cosine-NDCG@10')
mrr = ir_result.get_metric('cosine-MRR@10')
map_score = ir_result.get_metric('cosine-MAP@10')

print(f"\nKey Metrics:")
print(f"  NDCG@10: {ndcg:.4f}")
print(f"  MRR@10:  {mrr:.4f}")
print(f"  MAP@10:  {map_score:.4f}")

print("\n" + "="*70)
print("📦 Example 3: Using HuggingFaceEncoder")
print("-"*70)

from semeval import HuggingFaceEncoder

print("Creating HuggingFace encoder...")
hf_encoder = HuggingFaceEncoder("dbmdz/bert-base-turkish-cased")
print(f"✅ Loaded: {hf_encoder.model_name} ({hf_encoder.get_embedding_dim()}D)")

print("\nRunning with HuggingFace encoder...")
hf_runner = TaskRunner(encoder=hf_encoder, verbose=False)
hf_result = hf_runner.run_task("information_retrieval", str(data_path))

print(f"✅ Status: {hf_result.status}")
print(f"✅ Runtime: {hf_result.runtime_seconds:.2f}s")

hf_ndcg = hf_result.get_metric('cosine-NDCG@10')
if hf_ndcg:
    print(f"✅ NDCG@10: {hf_ndcg:.4f}")

print("\n" + "="*70)
print("📦 Example 4: Direct Task Usage")
print("-"*70)

from semeval import InformationRetrieval, JSONDataLoader

print("Loading data directly...")
loader = JSONDataLoader()
test_data = loader.load(str(data_path))
ir_data = test_data.tasks.information_retrieval

print(f"✅ Loaded: {len(ir_data.corpus)} docs, {len(ir_data.queries)} queries")

print("\nRunning task directly...")
task = InformationRetrieval(
    encoder=encoder,
    task_data=ir_data,
    verbose=False
)
direct_result = task.run()

print(f"✅ Status: {direct_result.status}")
print(f"✅ Metrics: {len(direct_result.metrics)} metrics computed")

print("\n" + "="*70)
print("✅ All examples completed successfully!")
print("="*70)
