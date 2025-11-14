"""Tests for core schemas."""

from semeval.core.schemas import SimilarityConfig, TestMetadata, TripletData


def test_test_metadata_creation():
    """Test TestMetadata creation with valid data."""
    metadata = TestMetadata(
        version="1.0",
        description="Test dataset",
        language="tr",
        domain="general",
        created_by="test",
    )
    assert metadata.version == "1.0"
    assert metadata.language == "tr"


def test_semantic_similarity_config_defaults():
    """Test SimilarityConfig has correct defaults."""
    config = SimilarityConfig()
    assert config.success_threshold == 0.85
    assert config.margin_threshold == 0.2


def test_triplet_data_creation():
    """Test TripletData creation with minimal required fields."""
    triplet = TripletData(
        id="test-1",
        anchor="Bu bir test",
        positive="Bu bir deneme",
        negative="Tamamen farklı",
    )
    assert triplet.id == "test-1"
    assert triplet.difficulty is None
    assert triplet.category is None


def test_triplet_data_with_optional_fields():
    """Test TripletData creation with all fields."""
    triplet = TripletData(
        id="test-2",
        anchor="Test anchor",
        positive="Test positive",
        negative="Test negative",
        difficulty="kolay",
        category="semantic",
        subcategory="synonyms",
    )
    assert triplet.difficulty == "kolay"
    assert triplet.category == "semantic"
    assert triplet.subcategory == "synonyms"
