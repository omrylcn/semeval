"""Integration tests for all evaluation tasks."""

import pytest

from semeval.core.encoders import SentenceTransformerEncoder
from semeval.core.schemas import (
    InformationRetrievalData,
    LinguisticRobustnessData,
    SemanticSimilarityData,
    VectorArithmeticData,
)
from semeval.tasks import (
    InformationRetrieval,
    LinguisticRobustness,
    SemanticSimilarity,
    VectorArithmetic,
)


@pytest.fixture
def small_encoder():
    """Create a small encoder for fast testing."""
    return SentenceTransformerEncoder(
        "sentence-transformers/paraphrase-albert-small-v2", device="cpu"
    )


class TestSemanticSimilarityIntegration:
    """Integration tests for Semantic Similarity task."""

    @pytest.fixture
    def sample_similarity_data(self):
        """Create sample semantic similarity data."""
        return SemanticSimilarityData(
            name="Test Semantic Similarity",
            enabled=True,
            triplets=[
                {
                    "id": "triplet_1",
                    "anchor": "The cat sits on the mat",
                    "positive": "A cat is sitting on a mat",
                    "negative": "Dogs are playing in the park",
                    "category": "animals",
                },
                {
                    "id": "triplet_2",
                    "anchor": "Machine learning is fascinating",
                    "positive": "AI and ML are very interesting",
                    "negative": "Cooking recipes for beginners",
                    "category": "technology",
                },
            ],
        )

    def test_semantic_similarity_task_run(self, small_encoder, sample_similarity_data):
        """Test semantic similarity task execution."""
        task = SemanticSimilarity(
            encoder=small_encoder, task_data=sample_similarity_data, verbose=False
        )

        result = task.run()

        assert result.task_name == "semantic_similarity"
        assert result.status in ["success", "failed", "completed"]
        assert result.metrics is not None
        assert "accuracy" in result.metrics

        # Check accuracy is between 0 and 1
        if result.status == "completed":
            assert 0 <= result.metrics["accuracy"] <= 1


class TestInformationRetrievalIntegration:
    """Integration tests for Information Retrieval task."""

    @pytest.fixture
    def sample_ir_data(self):
        """Create sample IR data."""
        return InformationRetrievalData(
            name="Test IR",
            enabled=True,
            corpus={
                "doc_1": "Machine learning is a subset of artificial intelligence",
                "doc_2": "Deep learning uses neural networks with multiple layers",
                "doc_3": "Natural language processing deals with text analysis",
                "doc_4": "Computer vision focuses on image understanding",
                "doc_5": "Cooking involves preparing food with various techniques",
            },
            queries={
                "q_1": "What is machine learning?",
                "q_2": "Tell me about deep learning",
            },
            relevant_docs={
                "q_1": {"doc_1": 2, "doc_2": 1},
                "q_2": {"doc_2": 2, "doc_1": 1},
            },
        )

    def test_information_retrieval_task_run(self, small_encoder, sample_ir_data):
        """Test information retrieval task execution."""
        task = InformationRetrieval(
            encoder=small_encoder, task_data=sample_ir_data, verbose=False
        )

        result = task.run()

        assert result.task_name == "information_retrieval"
        assert result.status in ["success", "failed", "completed"]
        assert result.metrics is not None

        # Check key metrics exist
        if result.status == "completed":
            assert any("NDCG" in key for key in result.metrics.keys())
            assert any("MRR" in key for key in result.metrics.keys())


class TestLinguisticRobustnessIntegration:
    """Integration tests for Linguistic Robustness task."""

    @pytest.fixture
    def sample_robustness_data(self):
        """Create sample robustness data."""
        return LinguisticRobustnessData(
            name="Test Robustness",
            enabled=True,
            morphology=[
                {
                    "id": "morph_1",
                    "original": "running",
                    "modified": "run",
                    "category": "verb_form",
                }
            ],
            typo=[
                {
                    "id": "typo_1",
                    "correct": "hello",
                    "typo": "helo",
                    "type": "deletion",
                }
            ],
            negation=[
                {
                    "id": "neg_1",
                    "sentence1": "The movie was good",
                    "sentence2": "The movie was not good",
                    "type": "negation",
                }
            ],
        )

    def test_linguistic_robustness_task_run(
        self, small_encoder, sample_robustness_data
    ):
        """Test linguistic robustness task execution."""
        task = LinguisticRobustness(
            encoder=small_encoder, task_data=sample_robustness_data, verbose=False
        )

        result = task.run()

        assert result.task_name == "linguistic_robustness"
        assert result.status in ["success", "failed", "completed"]
        assert result.metrics is not None

        # Check robustness metrics exist
        if result.status == "completed":
            assert "overall_robustness_score" in result.metrics


class TestVectorArithmeticIntegration:
    """Integration tests for Vector Arithmetic task."""

    @pytest.fixture
    def sample_arithmetic_data(self):
        """Create sample vector arithmetic data."""
        return VectorArithmeticData(
            name="Test Vector Arithmetic",
            enabled=True,
            analogies=[
                {
                    "id": "analogy_1",
                    "a": "king",
                    "b": "queen",
                    "c": "man",
                    "d": "woman",
                    "category": "gender",
                    "subcategory": "genel_turkce",
                },
                {
                    "id": "analogy_2",
                    "a": "paris",
                    "b": "france",
                    "c": "london",
                    "d": "england",
                    "category": "geography",
                    "subcategory": "finansal",
                },
            ],
        )

    def test_vector_arithmetic_task_run(self, small_encoder, sample_arithmetic_data):
        """Test vector arithmetic task execution."""
        task = VectorArithmetic(
            encoder=small_encoder, task_data=sample_arithmetic_data, verbose=False
        )

        result = task.run()

        assert result.task_name == "vector_arithmetic"
        assert result.status in ["success", "failed", "completed"]
        assert result.metrics is not None

        # Check arithmetic metrics exist
        if result.status == "completed":
            assert "accuracy" in result.metrics or "mean_rank" in result.metrics


class TestTaskBase:
    """Tests for base task functionality."""

    def test_task_result_creation(self):
        """Test TaskResult can be created."""
        from semeval.tasks.base import TaskResult

        result = TaskResult(
            task_name="test_task",
            status="completed",
            metrics={"accuracy": 0.85},
            runtime_seconds=1.5,
        )

        assert result.task_name == "test_task"
        assert result.status == "completed"
        assert result.metrics["accuracy"] == 0.85
        assert result.runtime_seconds == 1.5

    def test_task_result_get_metric(self):
        """Test getting specific metric from TaskResult."""
        from semeval.tasks.base import TaskResult

        result = TaskResult(
            task_name="test_task",
            status="completed",
            metrics={"accuracy": 0.85, "f1": 0.80},
            runtime_seconds=1.5,
        )

        assert result.get_metric("accuracy") == 0.85
        assert result.get_metric("f1") == 0.80
        assert result.get_metric("nonexistent") is None
        assert result.get_metric("nonexistent", default=0.0) == 0.0

    def test_task_result_to_dict(self):
        """Test TaskResult to_dict conversion."""
        from semeval.tasks.base import TaskResult

        result = TaskResult(
            task_name="test_task",
            status="completed",
            metrics={"accuracy": 0.85, "f1": 0.80, "precision": 0.82},
            runtime_seconds=1.5,
        )

        result_dict = result.model_dump()

        assert result_dict["task_name"] == "test_task"
        assert result_dict["status"] == "completed"
        assert result_dict["metrics"]["accuracy"] == 0.85
        assert result_dict["runtime_seconds"] == 1.5
