"""Tests for robustness metrics (unit tests only, no model performance)."""

import pytest

from semeval.metrics.robustness_metrics import (
    compute_morphology_metrics,
    compute_negation_metrics,
    compute_robustness_summary,
    compute_typo_metrics,
)


class TestMorphologyMetrics:
    """Tests for morphology metric calculations."""

    def test_morphology_metrics_basic_calculation(self):
        """Test basic metric calculations."""
        similarities = [0.92, 0.88, 0.91]
        categories = ["cat1", "cat2", "cat1"]

        metrics = compute_morphology_metrics(similarities, categories, threshold=0.85)

        assert metrics["total_pairs"] == 3
        assert abs(metrics["avg_similarity"] - 0.903) < 0.01
        assert metrics["success_rate"] == 1.0  # All above 0.85

    def test_morphology_metrics_empty_input(self):
        """Test handling of empty input."""
        metrics = compute_morphology_metrics([], [], threshold=0.85)

        assert metrics["avg_similarity"] == 0.0
        assert metrics["success_rate"] == 0.0
        assert metrics["total_pairs"] == 0
        assert metrics["category_breakdown"] == {}

    def test_morphology_metrics_threshold_logic(self):
        """Test success rate calculation with threshold."""
        similarities = [0.90, 0.80, 0.85, 0.95]  # 3 >= 0.85
        categories = ["test"] * 4

        metrics = compute_morphology_metrics(similarities, categories, threshold=0.85)

        assert metrics["success_rate"] == 0.75  # 3 out of 4

    def test_morphology_category_breakdown(self):
        """Test category-wise breakdown."""
        similarities = [0.95, 0.90, 0.80]
        categories = ["cat1", "cat1", "cat2"]

        metrics = compute_morphology_metrics(similarities, categories, threshold=0.85)

        assert "cat1" in metrics["category_breakdown"]
        assert "cat2" in metrics["category_breakdown"]
        assert metrics["category_breakdown"]["cat1"]["count"] == 2
        assert metrics["category_breakdown"]["cat2"]["count"] == 1


class TestTypoMetrics:
    """Tests for typo metric calculations."""

    def test_typo_metrics_basic_calculation(self):
        """Test basic metric calculations."""
        similarities = [0.85, 0.80, 0.90]
        types = ["sub", "del", "ins"]

        metrics = compute_typo_metrics(similarities, types, threshold=0.75)

        assert metrics["total_pairs"] == 3
        assert abs(metrics["avg_similarity"] - 0.85) < 0.01
        assert metrics["success_rate"] == 1.0  # All above 0.75

    def test_typo_metrics_empty_input(self):
        """Test handling of empty input."""
        metrics = compute_typo_metrics([], [], threshold=0.75)

        assert metrics["avg_similarity"] == 0.0
        assert metrics["success_rate"] == 0.0
        assert metrics["total_pairs"] == 0

    def test_typo_type_breakdown(self):
        """Test type-wise breakdown."""
        similarities = [0.85, 0.90, 0.60]
        types = ["substitution", "substitution", "deletion"]

        metrics = compute_typo_metrics(similarities, types, threshold=0.75)

        assert "substitution" in metrics["type_breakdown"]
        assert metrics["type_breakdown"]["substitution"]["count"] == 2


class TestNegationMetrics:
    """Tests for negation metric calculations."""

    def test_negation_metrics_basic_calculation(self):
        """Test basic metric calculations for negation (low similarity expected)."""
        similarities = [0.20, 0.15, 0.25]  # All below 0.30
        types = ["explicit", "implicit", "semantic"]

        metrics = compute_negation_metrics(similarities, types, threshold=0.30)

        assert metrics["total_pairs"] == 3
        assert abs(metrics["avg_similarity"] - 0.20) < 0.01
        assert metrics["success_rate"] == 1.0  # All correctly low

    def test_negation_metrics_threshold_logic(self):
        """Test that negation uses < threshold (opposite of morphology/typo)."""
        similarities = [0.29, 0.30, 0.31, 0.10]
        types = ["test"] * 4

        metrics = compute_negation_metrics(similarities, types, threshold=0.30)

        # Only 0.29 and 0.10 should pass (< 0.30)
        assert metrics["success_rate"] == 0.5  # 2 out of 4

    def test_negation_metrics_empty_input(self):
        """Test handling of empty input."""
        metrics = compute_negation_metrics([], [], threshold=0.30)

        assert metrics["avg_similarity"] == 0.0
        assert metrics["success_rate"] == 0.0
        assert metrics["total_pairs"] == 0


class TestRobustnessSummary:
    """Tests for robustness summary aggregation."""

    def test_summary_basic_aggregation(self):
        """Test basic summary aggregation."""
        morph = {"success_rate": 0.90, "total_pairs": 10, "avg_similarity": 0.88}
        typo = {"success_rate": 0.80, "total_pairs": 8, "avg_similarity": 0.82}
        negation = {"success_rate": 0.95, "total_pairs": 12, "avg_similarity": 0.18}

        summary = compute_robustness_summary(morph, typo, negation)

        assert summary["total_pairs"] == 30
        # Weighted average: (0.9*10 + 0.8*8 + 0.95*12) / 30
        expected = (9.0 + 6.4 + 11.4) / 30
        assert abs(summary["overall_success_rate"] - expected) < 0.01

    def test_summary_component_rates(self):
        """Test component success rates are preserved."""
        morph = {"success_rate": 0.85, "total_pairs": 10, "avg_similarity": 0.90}
        typo = {"success_rate": 0.75, "total_pairs": 8, "avg_similarity": 0.80}
        negation = {"success_rate": 0.92, "total_pairs": 12, "avg_similarity": 0.20}

        summary = compute_robustness_summary(morph, typo, negation)

        assert summary["morphology_success"] == 0.85
        assert summary["typo_success"] == 0.75
        assert summary["negation_success"] == 0.92

    def test_summary_with_zero_pairs(self):
        """Test summary handles components with zero pairs."""
        morph = {"success_rate": 0.90, "total_pairs": 10, "avg_similarity": 0.88}
        typo = {"success_rate": 0.0, "total_pairs": 0, "avg_similarity": 0.0}
        negation = {"success_rate": 0.85, "total_pairs": 8, "avg_similarity": 0.22}

        summary = compute_robustness_summary(morph, typo, negation)

        assert summary["total_pairs"] == 18
        # Should still compute overall rate (handles zeros gracefully)
        assert "overall_success_rate" in summary
