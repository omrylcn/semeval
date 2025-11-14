"""Tests for metrics modules."""

from semeval.metrics.similarity_metrics import compute_triplet_metrics


def test_compute_triplet_metrics_perfect_case():
    """Test triplet metrics with perfect separation."""
    positive_sims = [0.9, 0.85, 0.95]
    negative_sims = [0.1, 0.2, 0.15]

    metrics = compute_triplet_metrics(positive_sims, negative_sims)

    assert metrics["accuracy"] == 1.0  # All correct
    assert metrics["avg_margin"] > 0.6  # Large margin
    assert metrics["avg_positive_sim"] > 0.8
    assert metrics["avg_negative_sim"] < 0.3


def test_compute_triplet_metrics_failed_case():
    """Test triplet metrics with some failures."""
    positive_sims = [0.7, 0.4, 0.8]  # Middle one fails
    negative_sims = [0.3, 0.6, 0.2]  # Middle one higher

    metrics = compute_triplet_metrics(positive_sims, negative_sims)

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert metrics["accuracy"] < 1.0  # Not perfect
    assert "avg_margin" in metrics
    assert "margin_gt_02" in metrics


def test_compute_triplet_metrics_empty_input():
    """Test triplet metrics with empty lists."""
    metrics = compute_triplet_metrics([], [])

    assert metrics["accuracy"] == 0.0
    assert metrics["avg_margin"] == 0.0
    assert metrics["avg_positive_sim"] == 0.0
    assert metrics["avg_negative_sim"] == 0.0


def test_compute_triplet_metrics_margin_threshold():
    """Test margin > 0.2 metric."""
    positive_sims = [0.9, 0.7, 0.85]
    negative_sims = [0.6, 0.4, 0.5]  # Margins: 0.3, 0.3, 0.35

    metrics = compute_triplet_metrics(positive_sims, negative_sims)

    assert metrics["margin_gt_02"] == 1.0  # All margins > 0.2
