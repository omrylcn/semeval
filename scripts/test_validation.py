"""Test JSON validation with Pydantic schemas."""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from semeval.core.loaders import JSONDataLoader
from semeval.core import DataValidationError

# Test data path
data_path = Path(__file__).parent.parent / "data" / "test_data.json"

print("="*70)
print("Testing JSON Data Validation")
print("="*70)

# Test 1: Load and validate
print("\n1️⃣ Testing JSONDataLoader...")
try:
    loader = JSONDataLoader()
    test_data = loader.load(str(data_path))
    print("✅ JSON loaded and validated successfully!")

    # Show metadata
    print(f"\n📋 Metadata:")
    print(f"   Version: {test_data.metadata.version}")
    print(f"   Description: {test_data.metadata.description}")
    print(f"   Language: {test_data.metadata.language}")
    print(f"   Domain: {test_data.metadata.domain}")

    # Show enabled tasks
    print(f"\n📊 Enabled Tasks:")
    for task_name in test_data.get_enabled_tasks():
        print(f"   ✓ {task_name}")

    # IR task stats
    if test_data.tasks.information_retrieval:
        ir = test_data.tasks.information_retrieval
        print(f"\n📈 Information Retrieval Stats:")
        print(f"   Corpus size: {len(ir.corpus)}")
        print(f"   Query count: {len(ir.queries)}")
        print(f"   Total relevance judgments: {sum(len(docs) for docs in ir.relevant_docs.values())}")
        print(f"   Config: {ir.config.model_dump()}")

except DataValidationError as e:
    print(f"❌ Validation failed:\n{e}")
    sys.exit(1)
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Access specific data
print("\n2️⃣ Testing data access...")
try:
    # Access first query
    first_query_id = list(test_data.tasks.information_retrieval.queries.keys())[0]
    first_query = test_data.tasks.information_retrieval.queries[first_query_id]
    print(f"   Query {first_query_id}: '{first_query[:50]}...'")

    # Access first document
    first_doc_id = list(test_data.tasks.information_retrieval.corpus.keys())[0]
    first_doc = test_data.tasks.information_retrieval.corpus[first_doc_id]
    print(f"   Doc {first_doc_id}: '{first_doc[:50]}...'")

    # Access relevance judgment
    relevant = test_data.tasks.information_retrieval.relevant_docs[first_query_id]
    print(f"   Relevant docs for {first_query_id}: {dict(list(relevant.items())[:3])}")

    print("✅ Data access works correctly!")

except Exception as e:
    print(f"❌ Data access failed: {e}")
    sys.exit(1)

# Test 3: Pydantic model conversion
print("\n3️⃣ Testing Pydantic model features...")
try:
    # Test .model_dump() method
    ir_dict = test_data.tasks.information_retrieval.model_dump()
    print(f"   ✓ .model_dump() works - keys: {list(ir_dict.keys())}")

    # Test field access
    print(f"   ✓ Field access - name: '{ir_dict['name']}'")
    print(f"   ✓ Field access - enabled: {ir_dict['enabled']}")

    # Test validation on bad data
    print("\n4️⃣ Testing validation on invalid data...")
    import json
    bad_data = {
        "metadata": {
            "version": "1.0"
            # Missing 'description' - should fail
        }
    }

    try:
        from semeval.core.schemas import TestDataModel
        TestDataModel(**bad_data)
        print("❌ Validation should have failed!")
    except Exception:
        print("✅ Validation correctly rejected invalid data!")

except Exception as e:
    print(f"❌ Pydantic feature test failed: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✅ All validation tests passed!")
print("="*70)
