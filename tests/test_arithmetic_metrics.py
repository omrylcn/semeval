"""Tests for vector arithmetic metrics (unit tests only, no model performance)."""

import pytest

from semeval.metrics.arithmetic_metrics import compute_analogy_metrics, get_failed_analogies


class TestAnalogyMetrics:
    """Tests for analogy metric calculations."""

    def test_analogy_metrics_perfect_ranking(self):
        """Test metrics when all analogies are ranked first."""
        ranks = [1, 1, 1, 1]
        categories = ["geo", "geo", "gender", "gender"]
        subcategories = ["general"] * 4

        metrics = compute_analogy_metrics(ranks, categories, subcategories)

        assert metrics["top_k_accuracy"][1] == 1.0
        assert metrics["top_k_accuracy"][5] == 1.0
        assert metrics["mean_rank"] == 1.0
        assert metrics["median_rank"] == 1.0
        assert metrics["mrr"] == 1.0
        assert metrics["total_analogies"] == 4

    def test_analogy_metrics_mixed_ranking(self):
        """Test metrics with mixed rankings."""
        ranks = [1, 2, 5, 10, 3]
        categories = ["cat1"] * 5
        subcategories = ["subcat1"] * 5

        metrics = compute_analogy_metrics(ranks, categories, subcategories, top_k=[1, 5, 10])

        # Top-1: only first one
        assert metrics["top_k_accuracy"][1] == 0.2  # 1 out of 5
        # Top-5: first 4 (ranks 1,2,5,3)
        assert metrics["top_k_accuracy"][5] == 0.8  # 4 out of 5
        # Top-10: all 5
        assert metrics["top_k_accuracy"][10] == 1.0

        # Mean: (1+2+5+10+3)/5 = 21/5 = 4.2
        assert abs(metrics["mean_rank"] - 4.2) < 0.01

    def test_analogy_metrics_empty_input(self):
        """Test handling of empty input."""
        metrics = compute_analogy_metrics([], [], [])

        assert metrics["top_k_accuracy"][1] == 0.0
        assert metrics["mean_rank"] == 0.0
        assert metrics["mrr"] == 0.0
        assert metrics["total_analogies"] == 0

    def test_analogy_metrics_mrr_calculation(self):
        """Test Mean Reciprocal Rank calculation."""
        ranks = [1, 2, 4, 5]
        categories = ["test"] * 4
        subcategories = ["test"] * 4

        metrics = compute_analogy_metrics(ranks, categories, subcategories)

        # MRR = (1/1 + 1/2 + 1/4 + 1/5) / 4 = (1.0 + 0.5 + 0.25 + 0.2) / 4
        expected_mrr = (1.0 + 0.5 + 0.25 + 0.2) / 4
        assert abs(metrics["mrr"] - expected_mrr) < 0.01

    def test_analogy_metrics_median_rank(self):
        """Test median rank calculation."""
        # Odd number of ranks
        ranks = [1, 3, 5, 7, 9]
        categories = ["test"] * 5
        subcategories = ["test"] * 5

        metrics = compute_analogy_metrics(ranks, categories, subcategories)
        assert metrics["median_rank"] == 5.0

        # Even number of ranks
        ranks = [2, 4, 6, 8]
        categories = ["test"] * 4
        subcategories = ["test"] * 4

        metrics = compute_analogy_metrics(ranks, categories, subcategories)
        assert metrics["median_rank"] == 5.0  # (4+6)/2

    def test_analogy_metrics_category_breakdown(self):
        """Test category-wise breakdown."""
        ranks = [1, 2, 3, 4]
        categories = ["geo", "geo", "gender", "gender"]
        subcategories = ["general"] * 4

        metrics = compute_analogy_metrics(ranks, categories, subcategories)

        assert "geo" in metrics["category_breakdown"]
        assert "gender" in metrics["category_breakdown"]
        assert metrics["category_breakdown"]["geo"]["count"] == 2
        assert metrics["category_breakdown"]["gender"]["count"] == 2

    def test_analogy_metrics_subcategory_breakdown(self):
        """Test subcategory-wise breakdown."""
        ranks = [1, 2, 3]
        categories = ["cat1"] * 3
        subcategories = ["sub1", "sub1", "sub2"]

        metrics = compute_analogy_metrics(ranks, categories, subcategories)

        assert "sub1" in metrics["subcategory_breakdown"]
        assert "sub2" in metrics["subcategory_breakdown"]
        assert metrics["subcategory_breakdown"]["sub1"]["count"] == 2
        assert metrics["subcategory_breakdown"]["sub2"]["count"] == 1

    def test_analogy_metrics_custom_top_k(self):
        """Test with custom top_k values."""
        ranks = [1, 2, 3, 4, 5, 10, 15]
        categories = ["test"] * 7
        subcategories = ["test"] * 7

        metrics = compute_analogy_metrics(ranks, categories, subcategories, top_k=[1, 3, 10])

        assert 1 in metrics["top_k_accuracy"]
        assert 3 in metrics["top_k_accuracy"]
        assert 10 in metrics["top_k_accuracy"]
        assert 5 not in metrics["top_k_accuracy"]  # Not requested

        assert metrics["top_k_accuracy"][1] == 1 / 7
        assert metrics["top_k_accuracy"][3] == 3 / 7
        assert metrics["top_k_accuracy"][10] == 6 / 7


class TestFailedAnalogies:
    """Tests for failed analogies extraction."""

    def test_get_failed_analogies_basic(self):
        """Test basic failed analogies extraction."""
        analogy_tuples = [
            ("a1", "b1", "c1", "d1", "cat1", "sub1"),
            ("a2", "b2", "c2", "d2", "cat2", "sub2"),
            ("a3", "b3", "c3", "d3", "cat3", "sub3"),
        ]
        ranks = [10, 1, 5]  # First failed (rank > threshold)
        predictions = [
            [("wrong1", 0.9), ("d1", 0.8)],
            [("d2", 0.95), ("other", 0.85)],
            [("wrong3", 0.88), ("d3", 0.82)],
        ]

        failed = get_failed_analogies(analogy_tuples, ranks, predictions, threshold=5)

        # Should return only first (rank 10 > threshold 5, rank 5 is not > 5)
        assert len(failed) == 1
        assert failed[0]["rank"] == 10
        assert failed[0]["expected"] == "d1"
        assert failed[0]["category"] == "cat1"
        assert "a1" in failed[0]["formula"]

    def test_get_failed_analogies_empty_input(self):
        """Test handling of empty input."""
        failed = get_failed_analogies([], [], [], threshold=5)
        assert failed == []

    def test_get_failed_analogies_all_successful(self):
        """Test when all analogies succeed."""
        analogy_tuples = [
            ("a1", "b1", "c1", "d1", "cat1", "sub1"),
            ("a2", "b2", "c2", "d2", "cat2", "sub2"),
        ]
        ranks = [1, 2]  # All good
        predictions = [[("d1", 0.9)], [("d2", 0.95)]]

        failed = get_failed_analogies(analogy_tuples, ranks, predictions, threshold=5)
        assert failed == []

    def test_get_failed_analogies_multiple_failures(self):
        """Test multiple failed analogies are collected."""
        analogy_tuples = [
            ("a1", "b1", "c1", "d1", "cat", "sub"),
            ("a2", "b2", "c2", "d2", "cat", "sub"),
            ("a3", "b3", "c3", "d3", "cat", "sub"),
        ]
        ranks = [10, 50, 20]  # All > threshold
        predictions = [[], [], []]

        failed = get_failed_analogies(analogy_tuples, ranks, predictions, threshold=5)

        # All should be in failed list (not necessarily sorted)
        assert len(failed) == 3
        ranks_in_failed = {f["rank"] for f in failed}
        assert ranks_in_failed == {10, 20, 50}

    def test_get_failed_analogies_predictions_included(self):
        """Test that top predictions are included in output."""
        analogy_tuples = [("a", "b", "c", "d", "cat", "sub")]
        ranks = [10]
        predictions = [[("pred1", 0.9), ("pred2", 0.8), ("pred3", 0.7)]]

        failed = get_failed_analogies(analogy_tuples, ranks, predictions, threshold=5)

        assert len(failed) == 1
        assert "top_predictions" in failed[0]
        assert len(failed[0]["top_predictions"]) > 0
