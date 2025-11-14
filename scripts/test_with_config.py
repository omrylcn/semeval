"""Test running evaluation with config system.

This script demonstrates using the config system with TaskRunner.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semeval.core import load_settings, TaskRunner
from semeval.core.encoders import SentenceTransformerEncoder


def main():
    print("=" * 70)
    print("Testing SemEval with Config System")
    print("=" * 70)
    print()

    # Load settings from config.yaml
    print("1️⃣  Loading configuration...")
    settings = load_settings()
    print(f"   Config loaded: {settings.app_name} v{settings.app_version}")
    print(f"   Model: {settings.model.name}")
    print(f"   Batch size: {settings.model.batch_size}")
    print(f"   Device: {settings.model.device}")
    print(f"   Verbose: {settings.logging.verbose}")
    print(f"   Output dir: {settings.output.base_dir}")
    print()

    # Create encoder with config
    print("2️⃣  Loading encoder...")
    encoder = SentenceTransformerEncoder(
        model_name_or_path=settings.model.name, device=settings.model.device
    )
    print(f"   Encoder ready: {encoder._model_name}")
    print()

    # Create runner with settings
    print("3️⃣  Creating runner with settings...")
    runner = TaskRunner(encoder=encoder, settings=settings)  # Pass settings to runner
    print(f"   Runner configured:")
    print(f"   - Device: {runner.device}")
    print(f"   - Verbose: {runner.verbose}")
    print()

    # Run evaluation
    print("4️⃣  Running evaluation...")
    test_data_path = "data/test_data.json"
    result = runner.run(test_data_path)
    print()

    # Show results
    print("5️⃣  Results:")
    summary = result.get_summary()
    print(f"   Total runtime: {summary['total_runtime']:.2f}s")
    print(f"   Tasks completed: {len(summary['tasks'])}")
    print()

    for task_name, task_info in summary["tasks"].items():
        status_icon = "✅" if task_info["status"] == "success" else "❌"
        print(f"   {status_icon} {task_name}: {task_info['status']}")

    print()
    print("=" * 70)
    print("✅ Config-based evaluation complete!")
    print("=" * 70)
    print()
    print("💡 Try different configs:")
    print("   CONFIG_PATH=config.dev.yaml python scripts/test_with_config.py")
    print("   CONFIG_PATH=config.prod.yaml python scripts/test_with_config.py")
    print()


if __name__ == "__main__":
    main()
