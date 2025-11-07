"""Test the TaskRunner end-to-end."""

from pathlib import Path
import sys
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semeval.core.encoders import SentenceTransformerEncoder
from semeval.core.runner import TaskRunner

print("="*70)
print("Testing TaskRunner (End-to-End)")
print("="*70)

# Create encoder
print("\n1️⃣ Loading model...")
model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
print(f"   Model: {model_name}")
encoder = SentenceTransformerEncoder(model_name)
print(f"✅ Model loaded: {encoder.get_embedding_dim()}D embeddings")

# Create runner
print("\n2️⃣ Creating TaskRunner...")
runner = TaskRunner(encoder=encoder, verbose=True)
print("✅ Runner created")

# Run all tasks
print("\n3️⃣ Running all tasks...")
data_path = Path(__file__).parent.parent / "data" / "test_data.json"
result = runner.run(str(data_path))

# Display results
print("\n" + "="*70)
print("EVALUATION SUMMARY")
print("="*70)

summary = result.get_summary()

print(f"\n📊 Metadata:")
print(f"   Description: {summary['metadata']['description']}")
print(f"   Language: {summary['metadata']['language']}")
print(f"   Domain: {summary['metadata']['domain']}")
print(f"   Version: {summary['metadata']['version']}")

print(f"\n⏱️  Total Runtime: {summary['total_runtime']:.2f}s")

print(f"\n📈 Task Results:")
for task_name, task_info in summary['tasks'].items():
    print(f"\n   {task_name.upper()}")
    print(f"   Status: {task_info['status']}")
    print(f"   Runtime: {task_info['runtime']:.2f}s")

    if task_info['status'] == 'success':
        print(f"   Key Metrics:")
        key_metrics = [
            'cosine-NDCG@10',
            'cosine-MRR@10',
            'cosine-MAP@10'
        ]
        for metric in key_metrics:
            value = task_info['metrics'].get(metric)
            if value is not None:
                print(f"      {metric}: {value:.4f}")
    else:
        print(f"   Error: {task_info['error']}")

# Test get_task_result
print(f"\n📋 Testing get_task_result()...")
ir_result = result.get_task_result("information_retrieval")
if ir_result:
    print(f"   ✅ Retrieved IR result: {ir_result.task_name}")
    print(f"   Status: {ir_result.status}")
    print(f"   Metrics count: {len(ir_result.metrics)}")

# Test run_task (single task)
print(f"\n📋 Testing run_task() for single task...")
single_result = runner.run_task("information_retrieval", str(data_path))
print(f"   ✅ Single task result: {single_result.task_name}")
print(f"   Status: {single_result.status}")
print(f"   NDCG@10: {single_result.get_metric('cosine-NDCG@10'):.4f}")

print("\n" + "="*70)
print("✅ Runner test completed successfully!")
print("="*70)
