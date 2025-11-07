"""Evaluation metrics.

This package provides metric calculation functions for various tasks.
"""

from .ir_metrics import compute_ranking_metrics, aggregate_metrics

__all__ = ['compute_ranking_metrics', 'aggregate_metrics']
