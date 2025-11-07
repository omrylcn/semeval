"""Test model comparison features.

This script demonstrates how to compare multiple models across all 4 tasks.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from semeval import TaskRunner, SentenceTransformerEncoder

def main():
    """Test model comparison."""
    print("="*70)
    print("Model Comparison Test")
    print("="*70)

    data_path = Path(__file__).parent.parent / "data" / "test_data.json"

    # Models to compare
    models = [
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ]

    results = []

    for i, model_name in enumerate(models, 1):
        print(f"\n{'='*70}")
        print(f"Model {i}/{len(models)}: {model_name}")
        print(f"{'='*70}")

        # Load model
        print(f"📦 Loading model...")
        encoder = SentenceTransformerEncoder(model_name)
        print(f"✅ Model loaded: {encoder.get_embedding_dim()}D embeddings")

        # Run evaluation
        print(f"🔄 Running all 4 tasks...")
        runner = TaskRunner(encoder=encoder, verbose=False)
        result = runner.run(str(data_path))

        summary = result.get_summary()
        print(f"✅ Evaluation complete: {summary['total_runtime']:.2f}s")

        # Extract metrics
        model_results = {
            'model': model_name.split('/')[-1][:30],  # Shorten name
            'total_runtime': summary['total_runtime']
        }

        # Information Retrieval
        ir_metrics = summary['tasks']['information_retrieval']['metrics']
        model_results['IR_NDCG@10'] = ir_metrics.get('cosine-NDCG@10', 0)
        model_results['IR_MRR@10'] = ir_metrics.get('cosine-MRR@10', 0)
        model_results['IR_MAP@10'] = ir_metrics.get('cosine-MAP@10', 0)

        # Semantic Similarity
        ss_metrics = summary['tasks']['semantic_similarity']['metrics']
        model_results['SS_Accuracy'] = ss_metrics.get('accuracy', 0)
        model_results['SS_Avg_Margin'] = ss_metrics.get('avg_margin', 0)

        # Linguistic Robustness
        lr_metrics = summary['tasks']['linguistic_robustness']['metrics']
        model_results['LR_Overall'] = lr_metrics.get('overall_robustness', 0)
        model_results['LR_Morphology'] = lr_metrics.get('morphology_robustness', 0)

        # Vector Arithmetic
        va_metrics = summary['tasks']['vector_arithmetic']['metrics']
        model_results['VA_Accuracy'] = va_metrics.get('accuracy', 0)

        results.append(model_results)

    # Create comparison DataFrame
    print(f"\n{'='*70}")
    print("Comparison Results")
    print(f"{'='*70}\n")

    df = pd.DataFrame(results)

    # Display main metrics
    print("📊 Information Retrieval:")
    ir_cols = ['model', 'IR_NDCG@10', 'IR_MRR@10', 'IR_MAP@10']
    print(df[ir_cols].to_string(index=False))

    print("\n📊 Semantic Similarity:")
    ss_cols = ['model', 'SS_Accuracy', 'SS_Avg_Margin']
    print(df[ss_cols].to_string(index=False))

    print("\n📊 Linguistic Robustness:")
    lr_cols = ['model', 'LR_Overall', 'LR_Morphology']
    print(df[lr_cols].to_string(index=False))

    print("\n📊 Vector Arithmetic:")
    va_cols = ['model', 'VA_Accuracy']
    print(df[va_cols].to_string(index=False))

    print("\n⏱️  Runtime:")
    runtime_cols = ['model', 'total_runtime']
    print(df[runtime_cols].to_string(index=False))

    # Find winners
    print(f"\n{'='*70}")
    print("Winners by Task")
    print(f"{'='*70}\n")

    winners = {}

    # IR - NDCG@10
    best_idx = df['IR_NDCG@10'].idxmax()
    winners['Information Retrieval (NDCG@10)'] = df.loc[best_idx, 'model']

    # Semantic Similarity - Accuracy
    best_idx = df['SS_Accuracy'].idxmax()
    winners['Semantic Similarity (Accuracy)'] = df.loc[best_idx, 'model']

    # Linguistic Robustness - Overall
    best_idx = df['LR_Overall'].idxmax()
    winners['Linguistic Robustness (Overall)'] = df.loc[best_idx, 'model']

    # Vector Arithmetic - Accuracy
    best_idx = df['VA_Accuracy'].idxmax()
    winners['Vector Arithmetic (Accuracy)'] = df.loc[best_idx, 'model']

    # Fastest
    best_idx = df['total_runtime'].idxmin()
    winners['Fastest Runtime'] = df.loc[best_idx, 'model']

    for task, winner in winners.items():
        print(f"🏆 {task:45s}: {winner}")

    # Save comparison
    output_dir = Path(__file__).parent.parent / "output" / "comparison_test"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "model_comparison.csv"
    df.to_csv(csv_path, index=False)

    print(f"\n{'='*70}")
    print("✅ Model Comparison Complete!")
    print(f"{'='*70}")
    print(f"\n📂 Results saved to: {csv_path}")
    print(f"\n💡 Tip: Open {csv_path} in Excel or Google Sheets for detailed analysis")

if __name__ == "__main__":
    main()
