"""Test Information Retrieval task end-to-end."""

from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semeval.core.loaders import JSONDataLoader
from semeval.core.encoders import SentenceTransformerEncoder
from semeval.tasks import InformationRetrieval

print("="*70)
print("Testing Information Retrieval Task (End-to-End)")
print("="*70)

# Load test data
print("\n1️⃣ Loading test data...")
loader = JSONDataLoader()
data_path = Path(__file__).parent.parent / "data" / "test_data.json"
test_data = loader.load(str(data_path))
ir_data = test_data.tasks.information_retrieval
print(f"✅ Loaded: {len(ir_data.corpus)} docs, {len(ir_data.queries)} queries")

# Create encoder
print("\n2️⃣ Loading model...")
model_name = "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
print(f"   Model: {model_name}")
encoder = SentenceTransformerEncoder(model_name)
print(f"✅ Model loaded: {encoder.get_embedding_dim()}D embeddings")

# Create and run task
print("\n3️⃣ Running Information Retrieval task...")
task = InformationRetrieval(
    encoder=encoder,
    task_data=ir_data,
    verbose=True
)

result = task.run()

# Display results
print("\n" + "="*70)
print("RESULTS")
print("="*70)

print(f"\n📊 Status: {result.status}")
print(f"⏱️  Runtime: {result.runtime_seconds:.2f}s")

if result.status == "success":
    print(f"\n📈 Metrics:")

    # Key metrics
    key_metrics = [
        'cosine-NDCG@10',
        'cosine-MRR@10',
        'cosine-MAP@10',
        'cosine-Precision@10',
        'cosine-Recall@10'
    ]

    for metric_name in key_metrics:
        value = result.get_metric(metric_name)
        if value is not None:
            print(f"   {metric_name:25s}: {value:.4f}")

    # All metrics
    print(f"\n📋 All Metrics:")
    for metric_name, value in sorted(result.metrics.items()):
        if isinstance(value, (int, float)):
            print(f"   {metric_name:30s}: {value:.4f}")

    # Metadata
    print(f"\n📝 Metadata:")
    for key, value in result.metadata.items():
        if key != 'config':
            print(f"   {key}: {value}")

    print("\n" + "="*70)
    print("✅ Test completed successfully!")
    print("="*70)

else:
    print(f"\n❌ Task failed: {result.error_message}")
    sys.exit(1)
