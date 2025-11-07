"""Test configuration system.

This script demonstrates the configuration system with YAML, env vars, and defaults.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from semeval.core import load_settings


def main():
    print("=" * 70)
    print("Testing SemEval Configuration System")
    print("=" * 70)
    print()

    # Test 1: Load default config
    print("1️⃣  Loading default configuration (config.yaml)...")
    settings = load_settings()

    print(f"   App: {settings.app_name} v{settings.app_version}")
    print(f"   Model: {settings.model.name}")
    print(f"   Device: {settings.model.device}")
    print(f"   Batch Size: {settings.model.batch_size}")
    print(f"   Output Dir: {settings.output.base_dir}")
    print(f"   Export Formats: {settings.output.export_formats}")
    print(f"   Verbose: {settings.logging.verbose}")
    print()

    # Test 2: Task configurations
    print("2️⃣  Task Configurations:")
    print(f"   IR Enabled: {settings.tasks.information_retrieval.enabled}")
    print(f"   IR NDCG@K: {settings.tasks.information_retrieval.ndcg_at_k}")
    print(f"   Semantic Similarity Enabled: {settings.tasks.semantic_similarity.enabled}")
    print(f"   Linguistic Robustness Enabled: {settings.tasks.linguistic_robustness.enabled}")
    print(f"   - Morphology Threshold: {settings.tasks.linguistic_robustness.morphology_threshold}")
    print(f"   - Typo Threshold: {settings.tasks.linguistic_robustness.typo_threshold}")
    print(f"   Vector Arithmetic Enabled: {settings.tasks.vector_arithmetic.enabled}")
    print(f"   - Top-K: {settings.tasks.vector_arithmetic.top_k}")
    print()

    # Test 3: Environment variable override
    print("3️⃣  Testing environment variable override...")
    os.environ["SEMEVAL_MODEL__BATCH_SIZE"] = "64"
    os.environ["SEMEVAL_LOGGING__VERBOSE"] = "true"
    os.environ["SEMEVAL_OUTPUT__BASE_DIR"] = "./custom_output"

    settings2 = load_settings()
    print(f"   Batch Size (overridden): {settings2.model.batch_size}")
    print(f"   Verbose (overridden): {settings2.logging.verbose}")
    print(f"   Output Dir (overridden): {settings2.output.base_dir}")
    print()

    # Clean up env vars
    del os.environ["SEMEVAL_MODEL__BATCH_SIZE"]
    del os.environ["SEMEVAL_LOGGING__VERBOSE"]
    del os.environ["SEMEVAL_OUTPUT__BASE_DIR"]

    # Test 4: Configuration validation
    print("4️⃣  Configuration validation:")
    try:
        # Morphology threshold should be between 0 and 1
        assert 0.0 <= settings.tasks.linguistic_robustness.morphology_threshold <= 1.0
        print("   ✅ Morphology threshold valid")

        # Batch size should be positive
        assert settings.model.batch_size > 0
        print("   ✅ Batch size valid")

        # Export formats should be list
        assert isinstance(settings.output.export_formats, list)
        print("   ✅ Export formats valid")

        print()
        print("✅ All validation checks passed!")

    except AssertionError as e:
        print(f"   ❌ Validation failed: {e}")

    print()
    print("=" * 70)
    print("Configuration System Test Complete!")
    print("=" * 70)
    print()
    print("💡 Tips:")
    print("   - Override settings with env vars: SEMEVAL_MODEL__NAME=custom-model")
    print("   - Use custom config: CONFIG_PATH=config.prod.yaml")
    print("   - Create .env file for local overrides (gitignored)")
    print()


if __name__ == "__main__":
    main()
