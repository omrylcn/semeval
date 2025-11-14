"""Metrics for linguistic robustness evaluation.

Provides metrics for morphology, typo, and negation robustness tests.
"""

from typing import Any, Dict, List

import numpy as np


def compute_morphology_metrics(
    similarities: List[float],
    categories: List[str],
    threshold: float = 0.85
) -> Dict[str, Any]:
    """Compute metrics for morphology robustness.

    Morphology pairs should have HIGH similarity (>threshold).

    Parameters
    ----------
    similarities : list of float
        Cosine similarities for each pair
    categories : list of str
        Category for each pair (e.g., "iyelik", "çoğul")
    threshold : float, optional
        Success threshold (default: 0.85)

    Returns
    -------
    dict
        Dictionary containing:
        - avg_similarity: Average similarity score
        - min_similarity: Minimum similarity score
        - max_similarity: Maximum similarity score
        - std_similarity: Standard deviation of scores
        - success_rate: Percentage above threshold
        - total_pairs: Total number of pairs
        - category_breakdown: Per-category statistics

    Examples
    --------
    >>> similarities = [0.92, 0.88, 0.91]
    >>> categories = ["iyelik", "çoğul", "iyelik"]
    >>> metrics = compute_morphology_metrics(similarities, categories)
    >>> metrics['success_rate']
    1.0
    """
    if not similarities:
        return {
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'std_similarity': 0.0,
            'success_rate': 0.0,
            'total_pairs': 0,
            'category_breakdown': {}
        }

    sims = np.array(similarities)

    # Category breakdown
    category_stats = {}
    for cat, sim in zip(categories, similarities):
        if cat not in category_stats:
            category_stats[cat] = {'scores': [], 'count': 0, 'success': 0}
        category_stats[cat]['scores'].append(sim)
        category_stats[cat]['count'] += 1
        if sim >= threshold:
            category_stats[cat]['success'] += 1

    # Aggregate category stats
    category_breakdown = {}
    for cat, stats in category_stats.items():
        category_breakdown[cat] = {
            'avg_similarity': float(np.mean(stats['scores'])),
            'success_rate': stats['success'] / stats['count'] if stats['count'] > 0 else 0.0,
            'count': stats['count']
        }

    return {
        'avg_similarity': float(np.mean(sims)),
        'min_similarity': float(np.min(sims)),
        'max_similarity': float(np.max(sims)),
        'std_similarity': float(np.std(sims)),
        'success_rate': float((sims >= threshold).mean()),
        'total_pairs': len(similarities),
        'category_breakdown': category_breakdown
    }


def compute_typo_metrics(
    similarities: List[float],
    types: List[str],
    threshold: float = 0.75
) -> Dict[str, Any]:
    """Compute metrics for typo robustness.

    Typo pairs should have HIGH similarity (>threshold).

    Parameters
    ----------
    similarities : list of float
        Cosine similarities for each pair
    types : list of str
        Type of typo for each pair (e.g., "türkçe_ses", "klavye")
    threshold : float, optional
        Success threshold (default: 0.75)

    Returns
    -------
    dict
        Dictionary containing:
        - avg_similarity: Average similarity score
        - min_similarity: Minimum similarity score
        - max_similarity: Maximum similarity score
        - std_similarity: Standard deviation of scores
        - success_rate: Percentage above threshold
        - total_pairs: Total number of pairs
        - type_breakdown: Per-type statistics

    Examples
    --------
    >>> similarities = [0.82, 0.76, 0.79]
    >>> types = ["klavye", "türkçe_ses", "klavye"]
    >>> metrics = compute_typo_metrics(similarities, types)
    >>> metrics['success_rate']
    1.0
    """
    if not similarities:
        return {
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'std_similarity': 0.0,
            'success_rate': 0.0,
            'total_pairs': 0,
            'type_breakdown': {}
        }

    sims = np.array(similarities)

    # Type breakdown
    type_stats = {}
    for typ, sim in zip(types, similarities):
        if typ not in type_stats:
            type_stats[typ] = {'scores': [], 'count': 0, 'success': 0}
        type_stats[typ]['scores'].append(sim)
        type_stats[typ]['count'] += 1
        if sim >= threshold:
            type_stats[typ]['success'] += 1

    # Aggregate type stats
    type_breakdown = {}
    for typ, stats in type_stats.items():
        type_breakdown[typ] = {
            'avg_similarity': float(np.mean(stats['scores'])),
            'success_rate': stats['success'] / stats['count'] if stats['count'] > 0 else 0.0,
            'count': stats['count']
        }

    return {
        'avg_similarity': float(np.mean(sims)),
        'min_similarity': float(np.min(sims)),
        'max_similarity': float(np.max(sims)),
        'std_similarity': float(np.std(sims)),
        'success_rate': float((sims >= threshold).mean()),
        'total_pairs': len(similarities),
        'type_breakdown': type_breakdown
    }


def compute_negation_metrics(
    similarities: List[float],
    types: List[str],
    threshold: float = 0.50
) -> Dict[str, Any]:
    """Compute metrics for negation/opposition detection.

    Negation pairs should have LOW similarity (<threshold).

    Parameters
    ----------
    similarities : list of float
        Cosine similarities for each pair
    types : list of str
        Type of negation for each pair (e.g., "olumsuzluk_eki", "zıt_kelime")
    threshold : float, optional
        Success threshold (default: 0.50)

    Returns
    -------
    dict
        Dictionary containing:
        - avg_similarity: Average similarity score (should be low)
        - min_similarity: Minimum similarity score
        - max_similarity: Maximum similarity score
        - std_similarity: Standard deviation of scores
        - success_rate: Percentage below threshold
        - total_pairs: Total number of pairs
        - type_breakdown: Per-type statistics

    Examples
    --------
    >>> similarities = [0.32, 0.41, 0.28]
    >>> types = ["olumsuzluk_eki", "zıt_kelime", "olumsuzluk_eki"]
    >>> metrics = compute_negation_metrics(similarities, types)
    >>> metrics['success_rate']
    1.0
    """
    if not similarities:
        return {
            'avg_similarity': 0.0,
            'min_similarity': 0.0,
            'max_similarity': 0.0,
            'std_similarity': 0.0,
            'success_rate': 0.0,
            'total_pairs': 0,
            'type_breakdown': {}
        }

    sims = np.array(similarities)

    # Type breakdown
    type_stats = {}
    for typ, sim in zip(types, similarities):
        if typ not in type_stats:
            type_stats[typ] = {'scores': [], 'count': 0, 'success': 0}
        type_stats[typ]['scores'].append(sim)
        type_stats[typ]['count'] += 1
        if sim < threshold:
            type_stats[typ]['success'] += 1

    # Aggregate type stats
    type_breakdown = {}
    for typ, stats in type_stats.items():
        type_breakdown[typ] = {
            'avg_similarity': float(np.mean(stats['scores'])),
            'success_rate': stats['success'] / stats['count'] if stats['count'] > 0 else 0.0,
            'count': stats['count']
        }

    return {
        'avg_similarity': float(np.mean(sims)),
        'min_similarity': float(np.min(sims)),
        'max_similarity': float(np.max(sims)),
        'std_similarity': float(np.std(sims)),
        'success_rate': float((sims < threshold).mean()),  # Note: < instead of >=
        'total_pairs': len(similarities),
        'type_breakdown': type_breakdown
    }


def compute_robustness_summary(
    morphology_metrics: Dict[str, Any],
    typo_metrics: Dict[str, Any],
    negation_metrics: Dict[str, Any]
) -> Dict[str, Any]:
    """Compute overall robustness summary.

    Parameters
    ----------
    morphology_metrics : dict
        Metrics from morphology test
    typo_metrics : dict
        Metrics from typo test
    negation_metrics : dict
        Metrics from negation test

    Returns
    -------
    dict
        Overall summary with aggregate statistics

    Examples
    --------
    >>> summary = compute_robustness_summary(morph_metrics, typo_metrics, neg_metrics)
    >>> summary['overall_success_rate']
    0.875
    """
    # Calculate weighted average success rate
    total_pairs = (
        morphology_metrics['total_pairs'] +
        typo_metrics['total_pairs'] +
        negation_metrics['total_pairs']
    )

    if total_pairs == 0:
        overall_success = 0.0
    else:
        overall_success = (
            morphology_metrics['success_rate'] * morphology_metrics['total_pairs'] +
            typo_metrics['success_rate'] * typo_metrics['total_pairs'] +
            negation_metrics['success_rate'] * negation_metrics['total_pairs']
        ) / total_pairs

    return {
        'overall_success_rate': overall_success,
        'total_pairs': total_pairs,
        'morphology_success': morphology_metrics['success_rate'],
        'typo_success': typo_metrics['success_rate'],
        'negation_success': negation_metrics['success_rate'],
        'subtask_breakdown': {
            'morphology': {
                'success_rate': morphology_metrics['success_rate'],
                'avg_similarity': morphology_metrics['avg_similarity'],
                'count': morphology_metrics['total_pairs']
            },
            'typo': {
                'success_rate': typo_metrics['success_rate'],
                'avg_similarity': typo_metrics['avg_similarity'],
                'count': typo_metrics['total_pairs']
            },
            'negation': {
                'success_rate': negation_metrics['success_rate'],
                'avg_similarity': negation_metrics['avg_similarity'],
                'count': negation_metrics['total_pairs']
            }
        }
    }
