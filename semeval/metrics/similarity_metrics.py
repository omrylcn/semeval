"""Semantic Similarity metrics.

Simple triplet accuracy metrics following the style of ir_metrics.py
"""

from typing import Dict, List

import numpy as np


def compute_triplet_metrics(
    positive_sims: List[float], negative_sims: List[float]
) -> Dict[str, float]:
    """Compute triplet accuracy metrics for a batch of triplets.

    Parameters
    ----------
    positive_sims : list of float
        Similarity scores between anchor and positive examples
    negative_sims : list of float
        Similarity scores between anchor and negative examples

    Returns
    -------
    dict
        Metrics dictionary with keys:
        - accuracy: Triplet accuracy (positive > negative)
        - avg_positive_sim: Average similarity to positives
        - avg_negative_sim: Average similarity to negatives
        - avg_margin: Average margin (positive - negative)
        - min_margin: Minimum margin
        - max_margin: Maximum margin

    Examples
    --------
    >>> pos = [0.9, 0.85, 0.7]
    >>> neg = [0.3, 0.4, 0.8]
    >>> metrics = compute_triplet_metrics(pos, neg)
    >>> print(f"Accuracy: {metrics['accuracy']:.2%}")
    Accuracy: 66.67%
    """
    if not positive_sims or not negative_sims:
        return _empty_result()

    pos_arr = np.array(positive_sims)
    neg_arr = np.array(negative_sims)

    margins = pos_arr - neg_arr
    correct = np.sum(margins > 0)

    return {
        "accuracy": float(correct / len(positive_sims)),
        "avg_positive_sim": float(np.mean(pos_arr)),
        "avg_negative_sim": float(np.mean(neg_arr)),
        "avg_margin": float(np.mean(margins)),
        "min_margin": float(np.min(margins)),
        "max_margin": float(np.max(margins)),
        "margin_gt_01": float(np.mean(margins > 0.1)),
        "margin_gt_02": float(np.mean(margins > 0.2)),
    }


def compute_category_breakdown(
    positive_sims: List[float], negative_sims: List[float], categories: List[str]
) -> Dict[str, Dict[str, float]]:
    """Compute metrics broken down by category.

    Parameters
    ----------
    positive_sims : list of float
        Similarity scores to positives
    negative_sims : list of float
        Similarity scores to negatives
    categories : list of str
        Category for each triplet

    Returns
    -------
    dict
        Category name -> metrics dict
    """
    if not categories:
        return {}

    # Group by category
    category_data = {}
    for pos, neg, cat in zip(positive_sims, negative_sims, categories):
        if cat not in category_data:
            category_data[cat] = {"pos": [], "neg": []}
        category_data[cat]["pos"].append(pos)
        category_data[cat]["neg"].append(neg)

    # Compute metrics per category
    result = {}
    for cat, data in category_data.items():
        result[cat] = compute_triplet_metrics(data["pos"], data["neg"])
        result[cat]["count"] = len(data["pos"])

    return result


def _empty_result() -> Dict[str, float]:
    """Return empty result for edge cases."""
    return {
        "accuracy": 0.0,
        "avg_positive_sim": 0.0,
        "avg_negative_sim": 0.0,
        "avg_margin": 0.0,
        "min_margin": 0.0,
        "max_margin": 0.0,
        "margin_gt_01": 0.0,
        "margin_gt_02": 0.0,
    }
