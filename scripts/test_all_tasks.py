"""Test all tasks together."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semeval import TaskRunner, SentenceTransformerEncoder


def main():
    """Test all tasks."""
    print("=" * 70)
    print("Testing All Tasks Together")
    print("=" * 70)

    # Create encoder
    print("\n📦 Loading model...")
    encoder = SentenceTransformerEncoder(
        "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
    )
    print(f"✅ Model loaded: {encoder.model_name}")

    # Create runner
    runner = TaskRunner(encoder=encoder, verbose=True)

    # Test data path
    data_path = Path(__file__).parent.parent / "data" / "test_data.json"

    # Run all tasks
    print(f"\n📂 Test data: {data_path}\n")

    result = runner.run(str(data_path))

    # Print summary
    summary = result.get_summary()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\nTotal Runtime: {summary['total_runtime']:.2f}s")
    print(f"Tasks Completed: {len(summary['tasks'])}")

    for task_name, task_info in summary["tasks"].items():
        print(f"\n📋 {task_name.replace('_', ' ').title()}")
        print(f"  Status: {task_info['status']}")
        print(f"  Runtime: {task_info['runtime']:.2f}s")

        if task_info["status"] == "success":
            if task_name == "information_retrieval":
                metrics = task_info["metrics"]
                print(f"  NDCG@10: {metrics.get('cosine-NDCG@10', 0):.4f}")
                print(f"  MRR@10: {metrics.get('cosine-MRR@10', 0):.4f}")
                print(f"  MAP@10: {metrics.get('cosine-MAP@10', 0):.4f}")

            elif task_name == "semantic_similarity":
                metrics = task_info["metrics"]
                print(f"  Accuracy: {metrics.get('accuracy', 0):.2%}")
                print(f"  Avg Margin: {metrics.get('avg_margin', 0):+.3f}")
                print(f"  Margin > 0.2: {metrics.get('margin_gt_02', 0):.2%}")

            elif task_name == "linguistic_robustness":
                metrics = task_info["metrics"]
                print(
                    f"  Overall Success: {metrics.get('overall_success_rate', 0):.1%}"
                )
                print(f"  Morphology: {metrics['morphology']['success_rate']:.1%}")
                print(f"  Typo: {metrics['typo']['success_rate']:.1%}")
                print(f"  Negation: {metrics['negation']['success_rate']:.1%}")

            elif task_name == "vector_arithmetic":
                metrics = task_info["metrics"]
                top_k = metrics.get("top_k_accuracy", {})
                print(f"  Top-1 Accuracy: {top_k.get(1, 0):.1%}")
                print(f"  Top-5 Accuracy: {top_k.get(5, 0):.1%}")
                print(f"  Mean Rank: {metrics.get('mean_rank', 0):.2f}")
        else:
            print(f"  ❌ Error: {task_info['error']}")

    print("\n" + "=" * 70)
    print("✅ All Tests Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
