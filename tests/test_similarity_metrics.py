"""Tests for similarity metrics (unit tests only, no model performance)."""

import pytest

from semeval.metrics.similarity_metrics import (
    _empty_result,
    compute_category_breakdown,
    compute_triplet_metrics,
)


class TestTripletMetrics:
    """Tests for triplet metric calculations."""

    def test_triplet_metrics_perfect_accuracy(self):
        """Test metrics with perfect triplet accuracy."""
        positive_sims = [0.9, 0.85, 0.95]
        negative_sims = [0.3, 0.4, 0.2]

        metrics = compute_triplet_metrics(positive_sims, negative_sims)

        # All triplets correct (positive > negative)
        assert metrics["accuracy"] == 1.0
        assert abs(metrics["avg_positive_sim"] - 0.9) < 0.01
        assert abs(metrics["avg_negative_sim"] - 0.3) < 0.01
        # Margins: [0.6, 0.45, 0.75]
        assert abs(metrics["avg_margin"] - 0.6) < 0.01
        assert abs(metrics["min_margin"] - 0.45) < 0.01
        assert abs(metrics["max_margin"] - 0.75) < 0.01

    def test_triplet_metrics_zero_accuracy(self):
        """Test metrics with zero accuracy (all wrong)."""
        positive_sims = [0.3, 0.2, 0.4]
        negative_sims = [0.9, 0.85, 0.95]

        metrics = compute_triplet_metrics(positive_sims, negative_sims)

        # All triplets wrong (positive < negative)
        assert metrics["accuracy"] == 0.0
        assert abs(metrics["avg_positive_sim"] - 0.3) < 0.01
        assert abs(metrics["avg_negative_sim"] - 0.9) < 0.01
        # Margins are negative
        assert metrics["avg_margin"] < 0.0
        assert metrics["min_margin"] < 0.0

    def test_triplet_metrics_mixed_accuracy(self):
        """Test metrics with mixed accuracy."""
        positive_sims = [0.8, 0.3, 0.7, 0.4]
        negative_sims = [0.2, 0.6, 0.5, 0.1]

        metrics = compute_triplet_metrics(positive_sims, negative_sims)

        # Triplets: correct, wrong, correct, correct = 3/4 = 0.75
        assert metrics["accuracy"] == 0.75
        assert abs(metrics["avg_positive_sim"] - 0.55) < 0.01  # (0.8+0.3+0.7+0.4)/4
        assert abs(metrics["avg_negative_sim"] - 0.35) < 0.01  # (0.2+0.6+0.5+0.1)/4

    def test_triplet_metrics_empty_input(self):
        """Test handling of empty input."""
        metrics = compute_triplet_metrics([], [])

        assert metrics["accuracy"] == 0.0
        assert metrics["avg_positive_sim"] == 0.0
        assert metrics["avg_negative_sim"] == 0.0
        assert metrics["avg_margin"] == 0.0

    def test_triplet_metrics_margin_thresholds(self):
        """Test margin threshold calculations."""
        # Margins: [0.3, 0.15, 0.25, 0.05]
        positive_sims = [0.8, 0.65, 0.75, 0.55]
        negative_sims = [0.5, 0.5, 0.5, 0.5]

        metrics = compute_triplet_metrics(positive_sims, negative_sims)

        # margin_gt_01: how many > 0.1? [0.3, 0.15, 0.25] = 3/4 = 0.75
        assert metrics["margin_gt_01"] == 0.75
        # margin_gt_02: how many > 0.2? [0.3, 0.25] = 2/4 = 0.5
        assert metrics["margin_gt_02"] == 0.5

    def test_triplet_metrics_single_triplet(self):
        """Test with single triplet."""
        positive_sims = [0.9]
        negative_sims = [0.3]

        metrics = compute_triplet_metrics(positive_sims, negative_sims)

        assert metrics["accuracy"] == 1.0
        assert abs(metrics["avg_positive_sim"] - 0.9) < 0.01
        assert abs(metrics["avg_negative_sim"] - 0.3) < 0.01
        assert abs(metrics["avg_margin"] - 0.6) < 0.01
        assert abs(metrics["min_margin"] - 0.6) < 0.01
        assert abs(metrics["max_margin"] - 0.6) < 0.01

    def test_triplet_metrics_zero_margin(self):
        """Test when positive equals negative (margin = 0)."""
        positive_sims = [0.5, 0.6]
        negative_sims = [0.5, 0.6]

        metrics = compute_triplet_metrics(positive_sims, negative_sims)

        assert metrics["accuracy"] == 0.0  # margins not > 0
        assert metrics["avg_margin"] == 0.0
        assert metrics["min_margin"] == 0.0
        assert metrics["max_margin"] == 0.0


class TestCategoryBreakdown:
    """Tests for category-wise breakdown."""

    def test_category_breakdown_single_category(self):
        """Test breakdown with single category."""
        positive_sims = [0.9, 0.85]
        negative_sims = [0.3, 0.4]
        categories = ["cat1", "cat1"]

        breakdown = compute_category_breakdown(positive_sims, negative_sims, categories)

        assert "cat1" in breakdown
        assert breakdown["cat1"]["count"] == 2
        assert breakdown["cat1"]["accuracy"] == 1.0

    def test_category_breakdown_multiple_categories(self):
        """Test breakdown with multiple categories."""
        positive_sims = [0.9, 0.3, 0.8, 0.2]
        negative_sims = [0.2, 0.6, 0.3, 0.7]
        categories = ["geo", "geo", "gender", "gender"]

        breakdown = compute_category_breakdown(positive_sims, negative_sims, categories)

        assert "geo" in breakdown
        assert "gender" in breakdown

        # geo: [0.9>0.2=correct, 0.3<0.6=wrong] = 1/2 = 0.5
        assert breakdown["geo"]["count"] == 2
        assert breakdown["geo"]["accuracy"] == 0.5

        # gender: [0.8>0.3=correct, 0.2<0.7=wrong] = 1/2 = 0.5
        assert breakdown["gender"]["count"] == 2
        assert breakdown["gender"]["accuracy"] == 0.5

    def test_category_breakdown_empty_input(self):
        """Test handling of empty input."""
        breakdown = compute_category_breakdown([], [], [])
        assert breakdown == {}

    def test_category_breakdown_preserves_metrics(self):
        """Test that category breakdown includes all metrics."""
        positive_sims = [0.9, 0.85]
        negative_sims = [0.3, 0.4]
        categories = ["test", "test"]

        breakdown = compute_category_breakdown(positive_sims, negative_sims, categories)

        # Should have all metric keys
        assert "accuracy" in breakdown["test"]
        assert "avg_positive_sim" in breakdown["test"]
        assert "avg_negative_sim" in breakdown["test"]
        assert "avg_margin" in breakdown["test"]
        assert "min_margin" in breakdown["test"]
        assert "max_margin" in breakdown["test"]
        assert "margin_gt_01" in breakdown["test"]
        assert "margin_gt_02" in breakdown["test"]
        assert "count" in breakdown["test"]


class TestEmptyResult:
    """Tests for empty result helper."""

    def test_empty_result_structure(self):
        """Test empty result has all required keys."""
        result = _empty_result()

        assert "accuracy" in result
        assert "avg_positive_sim" in result
        assert "avg_negative_sim" in result
        assert "avg_margin" in result
        assert "min_margin" in result
        assert "max_margin" in result
        assert "margin_gt_01" in result
        assert "margin_gt_02" in result

    def test_empty_result_all_zeros(self):
        """Test empty result has all zero values."""
        result = _empty_result()

        assert result["accuracy"] == 0.0
        assert result["avg_positive_sim"] == 0.0
        assert result["avg_negative_sim"] == 0.0
        assert result["avg_margin"] == 0.0
        assert result["min_margin"] == 0.0
        assert result["max_margin"] == 0.0
        assert result["margin_gt_01"] == 0.0
        assert result["margin_gt_02"] == 0.0
