"""Metrics for vector arithmetic evaluation.

Provides metrics for analogy-based vector arithmetic tests.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def compute_analogy_metrics(
    ranks: List[int],
    categories: List[str],
    subcategories: List[str],
    top_k: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute metrics for analogy evaluation.

    Parameters
    ----------
    ranks : list of int
        Rank of correct answer for each analogy (1-indexed)
        Use -1 or large number if answer not found
    categories : list of str
        Category for each analogy (e.g., "coğrafya", "cinsiyet")
    subcategories : list of str
        Subcategory for each analogy (e.g., "genel_turkce", "finansal")
    top_k : list of int, optional
        K values for top-k accuracy (default: [1, 5, 10])

    Returns
    -------
    dict
        Dictionary containing:
        - top_k_accuracy: Dict of accuracy@k for each k
        - mean_rank: Average rank of correct answer
        - median_rank: Median rank of correct answer
        - mrr: Mean Reciprocal Rank
        - total_analogies: Total number of analogies
        - category_breakdown: Per-category statistics
        - subcategory_breakdown: Per-subcategory statistics

    Examples
    --------
    >>> ranks = [1, 2, 5, 1, 3]
    >>> categories = ["geo", "geo", "gender", "gender", "time"]
    >>> subcategories = ["general", "general", "general", "general", "general"]
    >>> metrics = compute_analogy_metrics(ranks, categories, subcategories)
    >>> metrics['top_k_accuracy'][1]
    0.4
    >>> metrics['top_k_accuracy'][5]
    1.0
    """
    if top_k is None:
        top_k = [1, 5, 10]

    if not ranks:
        return {
            "top_k_accuracy": dict.fromkeys(top_k, 0.0),
            "mean_rank": 0.0,
            "median_rank": 0.0,
            "mrr": 0.0,
            "total_analogies": 0,
            "category_breakdown": {},
            "subcategory_breakdown": {},
        }

    ranks_arr = np.array(ranks)

    # Top-k accuracy
    top_k_accuracy = {}
    for k in top_k:
        top_k_accuracy[k] = float((ranks_arr <= k).mean())

    # Mean/Median rank
    mean_rank = float(np.mean(ranks_arr))
    median_rank = float(np.median(ranks_arr))

    # MRR (Mean Reciprocal Rank)
    reciprocal_ranks = np.where(ranks_arr > 0, 1.0 / ranks_arr, 0.0)
    mrr = float(np.mean(reciprocal_ranks))

    # Category breakdown
    category_stats = {}
    for cat, rank in zip(categories, ranks):
        if cat not in category_stats:
            category_stats[cat] = {"ranks": [], "count": 0, "success_top5": 0}
        category_stats[cat]["ranks"].append(rank)
        category_stats[cat]["count"] += 1
        if rank <= 5:
            category_stats[cat]["success_top5"] += 1

    category_breakdown = {}
    for cat, stats in category_stats.items():
        category_breakdown[cat] = {
            "mean_rank": float(np.mean(stats["ranks"])),
            "top5_accuracy": (
                stats["success_top5"] / stats["count"] if stats["count"] > 0 else 0.0
            ),
            "count": stats["count"],
        }

    # Subcategory breakdown
    subcategory_stats = {}
    for subcat, rank in zip(subcategories, ranks):
        if subcat not in subcategory_stats:
            subcategory_stats[subcat] = {"ranks": [], "count": 0, "success_top5": 0}
        subcategory_stats[subcat]["ranks"].append(rank)
        subcategory_stats[subcat]["count"] += 1
        if rank <= 5:
            subcategory_stats[subcat]["success_top5"] += 1

    subcategory_breakdown = {}
    for subcat, stats in subcategory_stats.items():
        subcategory_breakdown[subcat] = {
            "mean_rank": float(np.mean(stats["ranks"])),
            "top5_accuracy": (
                stats["success_top5"] / stats["count"] if stats["count"] > 0 else 0.0
            ),
            "count": stats["count"],
        }

    return {
        "top_k_accuracy": top_k_accuracy,
        "mean_rank": mean_rank,
        "median_rank": median_rank,
        "mrr": mrr,
        "total_analogies": len(ranks),
        "category_breakdown": category_breakdown,
        "subcategory_breakdown": subcategory_breakdown,
    }


def format_analogy_formula(a: str, b: str, c: str, d: str) -> str:
    """Format analogy as readable formula.

    Parameters
    ----------
    a, b, c, d : str
        Analogy terms

    Returns
    -------
    str
        Formatted formula

    Examples
    --------
    >>> format_analogy_formula("Ankara", "Türkiye", "Almanya", "Berlin")
    'Ankara - Türkiye + Almanya ≈ Berlin'
    """
    return f"{a} - {b} + {c} ≈ {d}"


def get_failed_analogies(
    analogies: List[tuple],
    ranks: List[int],
    predictions: List[List[tuple]],
    threshold: int = 5,
) -> List[Dict[str, Any]]:
    """Get list of failed analogies for error analysis.

    Parameters
    ----------
    analogies : list of tuple
        List of (a, b, c, d, category, subcategory) tuples
    ranks : list of int
        Rank of correct answer for each analogy
    predictions : list of list of tuple
        Top predictions for each analogy [(word, score), ...]
    threshold : int, optional
        Rank threshold for failure (default: 5)

    Returns
    -------
    list of dict
        Failed analogies with predictions

    Examples
    --------
    >>> failed = get_failed_analogies(analogies, ranks, predictions, threshold=5)
    >>> len(failed)
    3
    """
    failed = []

    for analogy, rank, preds in zip(analogies, ranks, predictions):
        if rank > threshold:
            a, b, c, d, category, subcategory = analogy
            failed.append(
                {
                    "formula": format_analogy_formula(a, b, c, d),
                    "expected": d,
                    "rank": rank,
                    "category": category,
                    "subcategory": subcategory,
                    "top_predictions": preds[:5],  # Top 5 predictions
                }
            )

    return failed
