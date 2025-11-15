"""Tests for Information Retrieval metrics."""

import pytest

from semeval.metrics.ir_metrics import (
    _average_precision,
    _dcg_at_k,
    _mrr_at_k,
    _ndcg_at_k,
    _precision_at_k,
    _recall_at_k,
    aggregate_metrics,
    compute_ranking_metrics,
)


class TestDCGMetrics:
    """Tests for DCG and NDCG metrics."""

    def test_dcg_perfect_ranking(self):
        """Test DCG with perfect ranking."""
        relevances = [3, 2, 1, 0, 0]
        dcg = _dcg_at_k(relevances, k=5)
        # DCG = 3/log2(2) + 2/log2(3) + 1/log2(4) + 0 + 0
        expected = 3.0 / 1.0 + 2.0 / 1.585 + 1.0 / 2.0
        assert abs(dcg - expected) < 0.01

    def test_dcg_empty_relevances(self):
        """Test DCG with empty relevances."""
        assert _dcg_at_k([], k=5) == 0.0

    def test_dcg_at_k_limits(self):
        """Test DCG respects k limit."""
        relevances = [2, 2, 2, 2, 2]
        dcg_3 = _dcg_at_k(relevances, k=3)
        dcg_5 = _dcg_at_k(relevances, k=5)
        # DCG@3 should be less than DCG@5
        assert dcg_3 < dcg_5

    def test_ndcg_perfect_ranking(self):
        """Test NDCG with perfect ranking (should be 1.0)."""
        relevances = [3, 2, 1, 0, 0]
        ndcg = _ndcg_at_k(relevances, k=5)
        assert abs(ndcg - 1.0) < 0.01

    def test_ndcg_worst_ranking(self):
        """Test NDCG with worst ranking."""
        relevances = [0, 0, 0, 3, 2]  # Relevant docs at the end
        ndcg = _ndcg_at_k(relevances, k=5)
        # Should be less than perfect ranking
        assert ndcg < 1.0
        assert ndcg > 0.0  # But not zero

    def test_ndcg_no_relevant_docs(self):
        """Test NDCG with no relevant documents."""
        relevances = [0, 0, 0, 0, 0]
        ndcg = _ndcg_at_k(relevances, k=5)
        assert ndcg == 0.0

    def test_ndcg_at_different_k_values(self):
        """Test NDCG at different k values."""
        relevances = [2, 0, 1, 0, 0]
        ndcg_1 = _ndcg_at_k(relevances, k=1)
        ndcg_3 = _ndcg_at_k(relevances, k=3)

        # NDCG@1 should be higher because best doc is first
        assert ndcg_1 > 0.0
        assert ndcg_3 > 0.0


class TestMRRMetric:
    """Tests for Mean Reciprocal Rank."""

    def test_mrr_first_position(self):
        """Test MRR when first result is relevant."""
        relevances = [1, 0, 0, 0]
        mrr = _mrr_at_k(relevances, k=4)
        assert mrr == 1.0

    def test_mrr_second_position(self):
        """Test MRR when second result is relevant."""
        relevances = [0, 1, 0, 0]
        mrr = _mrr_at_k(relevances, k=4)
        assert mrr == 0.5

    def test_mrr_third_position(self):
        """Test MRR when third result is relevant."""
        relevances = [0, 0, 2, 0]  # Score 2 is also relevant
        mrr = _mrr_at_k(relevances, k=4)
        assert abs(mrr - 1.0 / 3) < 0.01

    def test_mrr_no_relevant(self):
        """Test MRR with no relevant documents."""
        relevances = [0, 0, 0, 0]
        mrr = _mrr_at_k(relevances, k=4)
        assert mrr == 0.0

    def test_mrr_respects_k(self):
        """Test MRR respects k limit."""
        relevances = [0, 0, 0, 1, 0]
        mrr = _mrr_at_k(relevances, k=3)
        # Relevant doc is at position 4, but k=3
        assert mrr == 0.0


class TestPrecisionRecall:
    """Tests for Precision and Recall metrics."""

    def test_precision_perfect(self):
        """Test precision with all relevant results."""
        relevances = [1, 1, 1, 1, 1]
        precision = _precision_at_k(relevances, k=5)
        assert precision == 1.0

    def test_precision_half(self):
        """Test precision with half relevant results."""
        relevances = [1, 0, 1, 0, 1]
        precision = _precision_at_k(relevances, k=5)
        assert precision == 0.6  # 3/5

    def test_precision_empty(self):
        """Test precision with empty relevances."""
        assert _precision_at_k([], k=5) == 0.0

    def test_precision_at_different_k(self):
        """Test precision at different k values."""
        relevances = [1, 1, 0, 0, 0]
        p1 = _precision_at_k(relevances, k=2)
        p5 = _precision_at_k(relevances, k=5)
        assert p1 == 1.0  # 2/2
        assert p5 == 0.4  # 2/5

    def test_recall_perfect(self):
        """Test recall when all relevant docs retrieved."""
        relevances = [1, 1, 1]
        recall = _recall_at_k(relevances, total_relevant=3, k=3)
        assert recall == 1.0

    def test_recall_partial(self):
        """Test recall when some relevant docs retrieved."""
        relevances = [1, 0, 1, 0, 0]
        recall = _recall_at_k(relevances, total_relevant=4, k=5)
        assert recall == 0.5  # Found 2 out of 4

    def test_recall_no_relevant_total(self):
        """Test recall when total_relevant is 0."""
        relevances = [0, 0, 0]
        recall = _recall_at_k(relevances, total_relevant=0, k=3)
        assert recall == 0.0

    def test_recall_respects_k(self):
        """Test recall respects k limit."""
        relevances = [1, 0, 0, 1, 1]
        r3 = _recall_at_k(relevances, total_relevant=3, k=3)
        r5 = _recall_at_k(relevances, total_relevant=3, k=5)
        assert r3 < r5  # Should find more with higher k


class TestAveragePrecision:
    """Tests for Average Precision (MAP)."""

    def test_ap_perfect_ranking(self):
        """Test AP with perfect ranking."""
        relevances = [1, 1, 1, 0, 0]
        ap = _average_precision(relevances, total_relevant=3)
        # AP = (1/1 + 2/2 + 3/3) / 3 = 3/3 = 1.0
        assert ap == 1.0

    def test_ap_mixed_ranking(self):
        """Test AP with mixed ranking."""
        relevances = [1, 0, 1, 0, 0]
        ap = _average_precision(relevances, total_relevant=2)
        # AP = (1/1 + 2/3) / 2 = (1.0 + 0.667) / 2 = 0.833
        expected = (1.0 + 2.0 / 3.0) / 2.0
        assert abs(ap - expected) < 0.01

    def test_ap_no_relevant(self):
        """Test AP with no relevant documents."""
        relevances = [0, 0, 0, 0]
        ap = _average_precision(relevances, total_relevant=0)
        assert ap == 0.0

    def test_ap_single_relevant_first(self):
        """Test AP with single relevant doc at first position."""
        relevances = [1, 0, 0, 0]
        ap = _average_precision(relevances, total_relevant=1)
        assert ap == 1.0

    def test_ap_single_relevant_last(self):
        """Test AP with single relevant doc at last position."""
        relevances = [0, 0, 0, 1]
        ap = _average_precision(relevances, total_relevant=1)
        assert ap == 0.25  # 1/4


class TestComputeRankingMetrics:
    """Tests for compute_ranking_metrics function."""

    def test_compute_ranking_metrics_basic(self):
        """Test basic ranking metrics computation."""
        retrieved_doc_ids = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        relevant_doc_ids = {"doc1": 2, "doc3": 1}  # doc1 highly relevant, doc3 relevant
        k_values = [1, 3, 5]

        metrics = compute_ranking_metrics(retrieved_doc_ids, relevant_doc_ids, k_values)

        # Check that all expected metrics are present
        assert "ndcg@1" in metrics
        assert "ndcg@3" in metrics
        assert "ndcg@5" in metrics
        assert "mrr@1" in metrics
        assert "precision@1" in metrics
        assert "recall@1" in metrics
        assert "map@5" in metrics

        # Verify some values
        assert metrics["mrr@1"] == 1.0  # First doc is relevant
        assert metrics["precision@1"] == 1.0  # First doc is relevant
        assert 0.0 <= metrics["ndcg@5"] <= 1.0

    def test_compute_ranking_metrics_no_relevant(self):
        """Test ranking metrics with no relevant documents."""
        retrieved_doc_ids = ["doc1", "doc2", "doc3"]
        relevant_doc_ids = {}
        k_values = [1, 3]

        metrics = compute_ranking_metrics(retrieved_doc_ids, relevant_doc_ids, k_values)

        # All metrics should be 0
        assert metrics["ndcg@1"] == 0.0
        assert metrics["mrr@1"] == 0.0
        assert metrics["precision@1"] == 0.0

    def test_compute_ranking_metrics_perfect(self):
        """Test ranking metrics with perfect ranking."""
        retrieved_doc_ids = ["doc1", "doc2", "doc3"]
        relevant_doc_ids = {"doc1": 2, "doc2": 1, "doc3": 1}
        k_values = [3]

        metrics = compute_ranking_metrics(retrieved_doc_ids, relevant_doc_ids, k_values)

        # Should have high scores
        assert metrics["ndcg@3"] == 1.0
        assert metrics["precision@3"] == 1.0
        assert metrics["recall@3"] == 1.0


class TestAggregateMetrics:
    """Tests for aggregate_metrics function."""

    def test_aggregate_metrics_single_query(self):
        """Test aggregating metrics from single query."""
        all_metrics = [{"ndcg@10": 0.8, "mrr@10": 0.9}]
        aggregated = aggregate_metrics(all_metrics)

        assert aggregated["ndcg@10"] == 0.8
        assert aggregated["mrr@10"] == 0.9

    def test_aggregate_metrics_multiple_queries(self):
        """Test aggregating metrics from multiple queries."""
        all_metrics = [
            {"ndcg@10": 0.8, "mrr@10": 1.0},
            {"ndcg@10": 0.6, "mrr@10": 0.5},
            {"ndcg@10": 0.7, "mrr@10": 0.75},
        ]
        aggregated = aggregate_metrics(all_metrics)

        assert abs(aggregated["ndcg@10"] - 0.7) < 0.01  # (0.8+0.6+0.7)/3
        assert abs(aggregated["mrr@10"] - 0.75) < 0.01  # (1.0+0.5+0.75)/3

    def test_aggregate_metrics_empty(self):
        """Test aggregating empty metrics."""
        aggregated = aggregate_metrics([])
        assert aggregated == {}

    def test_aggregate_metrics_missing_values(self):
        """Test aggregating metrics with missing values."""
        all_metrics = [
            {"ndcg@10": 0.8, "mrr@10": 1.0},
            {"ndcg@10": 0.6},  # Missing mrr@10
        ]
        aggregated = aggregate_metrics(all_metrics)

        assert abs(aggregated["ndcg@10"] - 0.7) < 0.01
        # Missing value treated as 0
        assert abs(aggregated["mrr@10"] - 0.5) < 0.01  # (1.0 + 0.0) / 2
