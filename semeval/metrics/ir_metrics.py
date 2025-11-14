"""Information Retrieval metrics.

Simplified IR metrics for when we can't use SentenceTransformers evaluator.
"""

from typing import Dict, List

import numpy as np


def compute_ranking_metrics(
    retrieved_doc_ids: List[str],
    relevant_doc_ids: Dict[str, int],
    k_values: List[int]
) -> Dict[str, float]:
    """Compute ranking metrics for a single query.

    Parameters
    ----------
    retrieved_doc_ids : list of str
        Retrieved document IDs in ranked order
    relevant_doc_ids : dict
        Mapping of relevant doc IDs to scores (0, 1, 2)
    k_values : list of int
        K values for metrics

    Returns
    -------
    dict
        Metrics dictionary
    """
    # Convert to relevance list
    relevances = [relevant_doc_ids.get(doc_id, 0) for doc_id in retrieved_doc_ids]
    total_relevant = len([s for s in relevant_doc_ids.values() if s > 0])

    metrics = {}

    for k in k_values:
        # NDCG@k
        metrics[f'ndcg@{k}'] = _ndcg_at_k(relevances, k)
        # MRR@k
        metrics[f'mrr@{k}'] = _mrr_at_k(relevances, k)
        # Precision@k
        metrics[f'precision@{k}'] = _precision_at_k(relevances, k)
        # Recall@k
        metrics[f'recall@{k}'] = _recall_at_k(relevances, total_relevant, k)

    # MAP
    max_k = max(k_values)
    metrics[f'map@{max_k}'] = _average_precision(relevances[:max_k], total_relevant)

    return metrics


def aggregate_metrics(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate metrics across queries."""
    if not all_metrics:
        return {}

    metric_names = set()
    for m in all_metrics:
        metric_names.update(m.keys())

    aggregated = {}
    for name in sorted(metric_names):
        values = [m.get(name, 0.0) for m in all_metrics]
        aggregated[name] = float(np.mean(values))

    return aggregated


def _dcg_at_k(relevances: List[int], k: int) -> float:
    """DCG@k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevances))


def _ndcg_at_k(relevances: List[int], k: int) -> float:
    """NDCG@k."""
    dcg = _dcg_at_k(relevances, k)
    ideal = sorted(relevances, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def _mrr_at_k(relevances: List[int], k: int) -> float:
    """MRR@k."""
    for idx, rel in enumerate(relevances[:k]):
        if rel > 0:
            return 1.0 / (idx + 1)
    return 0.0


def _precision_at_k(relevances: List[int], k: int) -> float:
    """Precision@k."""
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    return sum(1 for r in relevances if r > 0) / len(relevances)


def _recall_at_k(relevances: List[int], total_relevant: int, k: int) -> float:
    """Recall@k."""
    if total_relevant == 0:
        return 0.0
    found = sum(1 for r in relevances[:k] if r > 0)
    return found / total_relevant


def _average_precision(relevances: List[int], total_relevant: int) -> float:
    """Average Precision."""
    if total_relevant == 0:
        return 0.0

    precision_sum = 0.0
    num_relevant_found = 0

    for idx, rel in enumerate(relevances):
        if rel > 0:
            num_relevant_found += 1
            precision_sum += num_relevant_found / (idx + 1)

    return precision_sum / total_relevant
