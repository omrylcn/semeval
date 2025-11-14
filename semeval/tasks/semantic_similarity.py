"""Semantic Similarity evaluation task implementation.

This module implements the Semantic Similarity task which evaluates
a model's ability to distinguish semantically similar from dissimilar texts
using triplet tests.
"""

import time
from typing import Any, Dict, List, Optional

from sentence_transformers import util

from ..core.schemas import SemanticSimilarityData
from ..metrics.similarity_metrics import (
    compute_category_breakdown,
    compute_triplet_metrics,
)
from .base import BaseTask, TaskResult


class SemanticSimilarity(BaseTask):
    """Semantic Similarity evaluation task using triplet tests.

    This task evaluates how well a model can distinguish between semantically
    similar and dissimilar texts. For each triplet (anchor, positive, negative),
    the model should score sim(anchor, positive) > sim(anchor, negative).

    Parameters
    ----------
    encoder : BaseEncoder
        Text encoder to use for embeddings
    task_data : SemanticSimilarityData
        Semantic Similarity task configuration and test data
    device : str, optional
        Device for computation
    verbose : bool, optional
        If True, print progress information

    Attributes
    ----------
    task_data : SemanticSimilarityData
        Task configuration and data

    Examples
    --------
    >>> from semeval.core.encoders import SentenceTransformerEncoder
    >>> from semeval.core.loaders import JSONDataLoader
    >>>
    >>> # Load data
    >>> loader = JSONDataLoader()
    >>> test_data = loader.load("data/test_data.json")
    >>> ss_data = test_data.tasks.semantic_similarity
    >>>
    >>> # Create encoder
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>>
    >>> # Run task
    >>> task = SemanticSimilarity(encoder=encoder, task_data=ss_data, verbose=True)
    >>> result = task.run()
    >>> print(f"Accuracy: {result.metrics['accuracy']:.2%}")
    """

    def __init__(
        self,
        encoder,
        task_data: SemanticSimilarityData,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize Semantic Similarity task."""
        super().__init__(encoder, task_data, device, verbose)

    def get_task_name(self) -> str:
        """Return task name.

        Returns
        -------
        str
            Task name
        """
        return "semantic_similarity"

    def _evaluate_triplets(self) -> Dict[str, Any]:
        """Evaluate all triplets and compute metrics.

        Returns
        -------
        dict
            Evaluation metrics including accuracy, margins, and breakdowns
        """
        triplets = self.task_data.triplets
        total_triplets = len(triplets)

        self._log(f"Evaluating {total_triplets} triplets...")

        # Collect all texts to encode in batches
        anchor_texts = [t.anchor for t in triplets]
        positive_texts = [t.positive for t in triplets]
        negative_texts = [t.negative for t in triplets]

        # Encode all texts
        self._log("Encoding anchor texts...")
        anchor_embeddings = self.encoder.encode(
            anchor_texts,
            convert_to_tensor=True,
            show_progress_bar=self.verbose,
            batch_size=32
        )

        self._log("Encoding positive texts...")
        positive_embeddings = self.encoder.encode(
            positive_texts,
            convert_to_tensor=True,
            show_progress_bar=self.verbose,
            batch_size=32
        )

        self._log("Encoding negative texts...")
        negative_embeddings = self.encoder.encode(
            negative_texts,
            convert_to_tensor=True,
            show_progress_bar=self.verbose,
            batch_size=32
        )

        # Compute similarities
        self._log("Computing similarities...")
        positive_sims = []
        negative_sims = []

        for anchor_emb, pos_emb, neg_emb in zip(
            anchor_embeddings, positive_embeddings, negative_embeddings
        ):
            pos_sim = util.cos_sim(anchor_emb, pos_emb).item()
            neg_sim = util.cos_sim(anchor_emb, neg_emb).item()

            positive_sims.append(pos_sim)
            negative_sims.append(neg_sim)

        # Compute overall metrics
        self._log("Computing metrics...")
        metrics = compute_triplet_metrics(positive_sims, negative_sims)

        # Add count to metrics
        metrics['triplet_count'] = total_triplets

        # Compute difficulty breakdown (if available)
        if any(t.difficulty is not None for t in triplets):
            # Filter triplets with difficulty info
            filtered_data = [
                (pos, neg, t.difficulty)
                for pos, neg, t in zip(positive_sims, negative_sims, triplets)
                if t.difficulty is not None
            ]
            if filtered_data:
                filt_pos, filt_neg, filt_diff = zip(*filtered_data)
                difficulty_stats = self._compute_difficulty_breakdown(
                    list(filt_pos), list(filt_neg), list(filt_diff)
                )
                metrics['difficulty_breakdown'] = difficulty_stats

        # Compute subcategory breakdown (if available)
        if any(t.subcategory is not None for t in triplets):
            filtered_data = [
                (pos, neg, t.subcategory)
                for pos, neg, t in zip(positive_sims, negative_sims, triplets)
                if t.subcategory is not None
            ]
            if filtered_data:
                filt_pos, filt_neg, filt_subcat = zip(*filtered_data)
                subcategory_stats = self._compute_subcategory_breakdown(
                    list(filt_pos), list(filt_neg), list(filt_subcat)
                )
                metrics['subcategory_breakdown'] = subcategory_stats

        # Compute category breakdown (if available)
        if any(t.category is not None for t in triplets):
            filtered_data = [
                (pos, neg, t.category)
                for pos, neg, t in zip(positive_sims, negative_sims, triplets)
                if t.category is not None
            ]
            if filtered_data:
                filt_pos, filt_neg, filt_cat = zip(*filtered_data)
                category_stats = compute_category_breakdown(
                    list(filt_pos), list(filt_neg), list(filt_cat)
                )
                metrics['category_breakdown'] = category_stats

        # Identify failed triplets
        failed = self._identify_failed_triplets(
            triplets, positive_sims, negative_sims, limit=5
        )
        metrics['failed_triplets_sample'] = failed

        return metrics

    def _compute_difficulty_breakdown(
        self,
        positive_sims: List[float],
        negative_sims: List[float],
        difficulties: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by difficulty level.

        Parameters
        ----------
        positive_sims : list of float
            Similarity scores to positives
        negative_sims : list of float
            Similarity scores to negatives
        difficulties : list of str
            Difficulty level for each triplet

        Returns
        -------
        dict
            Difficulty level -> metrics dict
        """
        # Group by difficulty
        diff_groups = {}
        for pos, neg, diff in zip(positive_sims, negative_sims, difficulties):
            if diff not in diff_groups:
                diff_groups[diff] = {'pos': [], 'neg': []}
            diff_groups[diff]['pos'].append(pos)
            diff_groups[diff]['neg'].append(neg)

        # Compute metrics per difficulty
        result = {}
        for diff, data in diff_groups.items():
            result[diff] = compute_triplet_metrics(data['pos'], data['neg'])
            result[diff]['count'] = len(data['pos'])

        return result

    def _compute_subcategory_breakdown(
        self,
        positive_sims: List[float],
        negative_sims: List[float],
        subcategories: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """Compute metrics broken down by subcategory.

        Parameters
        ----------
        positive_sims : list of float
            Similarity scores to positives
        negative_sims : list of float
            Similarity scores to negatives
        subcategories : list of str
            Subcategory for each triplet

        Returns
        -------
        dict
            Subcategory -> metrics dict
        """
        # Group by subcategory
        subcat_groups = {}
        for pos, neg, subcat in zip(positive_sims, negative_sims, subcategories):
            if subcat not in subcat_groups:
                subcat_groups[subcat] = {'pos': [], 'neg': []}
            subcat_groups[subcat]['pos'].append(pos)
            subcat_groups[subcat]['neg'].append(neg)

        # Compute metrics per subcategory
        result = {}
        for subcat, data in subcat_groups.items():
            result[subcat] = compute_triplet_metrics(data['pos'], data['neg'])
            result[subcat]['count'] = len(data['pos'])

        return result

    def _identify_failed_triplets(
        self,
        triplets,
        positive_sims: List[float],
        negative_sims: List[float],
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Identify failed triplets for error analysis.

        Parameters
        ----------
        triplets : list of TripletData
            All triplets
        positive_sims : list of float
            Similarity scores to positives
        negative_sims : list of float
            Similarity scores to negatives
        limit : int, optional
            Maximum number of failures to return (default: 5)

        Returns
        -------
        list of dict
            Failed triplet information
        """
        failed = []

        for triplet, pos_sim, neg_sim in zip(triplets, positive_sims, negative_sims):
            margin = pos_sim - neg_sim
            if margin <= 0:  # Failed
                failed_info = {
                    'id': triplet.id,
                    'anchor': triplet.anchor[:50] + "..." if len(triplet.anchor) > 50 else triplet.anchor,
                    'positive_sim': float(pos_sim),
                    'negative_sim': float(neg_sim),
                    'margin': float(margin),
                }

                # Add optional metadata if available
                if triplet.difficulty is not None:
                    failed_info['difficulty'] = triplet.difficulty
                if triplet.category is not None:
                    failed_info['category'] = triplet.category
                if triplet.subcategory is not None:
                    failed_info['subcategory'] = triplet.subcategory

                failed.append(failed_info)

        # Sort by margin (most negative first)
        failed.sort(key=lambda x: x['margin'])

        return failed[:limit]

    def run(self) -> TaskResult:
        """Execute the Semantic Similarity evaluation task.

        Returns
        -------
        TaskResult
            Task execution results including metrics

        Raises
        ------
        RuntimeError
            If task execution fails

        Examples
        --------
        >>> task = SemanticSimilarity(encoder=encoder, task_data=data)
        >>> result = task.run()
        >>> print(f"Status: {result.status}")
        success
        >>> print(f"Accuracy: {result.metrics['accuracy']:.2%}")
        87.50%
        """
        self._log(f"Starting {self.task_data.name}")
        self._log(f"Triplets: {len(self.task_data.triplets)}")

        start_time = time.time()

        try:
            # Evaluate all triplets
            metrics = self._evaluate_triplets()

            runtime = time.time() - start_time

            self._log(f"Evaluation completed in {runtime:.2f}s")
            self._log(f"Triplet Accuracy: {metrics['accuracy']:.2%}")
            self._log(f"Average Margin: {metrics['avg_margin']:+.3f}")

            # Create result
            result = TaskResult(
                task_name=self.get_task_name(),
                status="success",
                metrics=metrics,
                runtime_seconds=runtime,
                metadata={
                    "triplet_count": len(self.task_data.triplets),
                    "config": self.task_data.config.model_dump()
                }
            )

            return result

        except Exception as e:
            runtime = time.time() - start_time

            self._log(f"Evaluation failed: {str(e)}", level="ERROR")

            return TaskResult(
                task_name=self.get_task_name(),
                status="failed",
                metrics={},
                runtime_seconds=runtime,
                error_message=str(e),
                metadata={}
            )

    @staticmethod
    def get_export_columns(result: TaskResult) -> Dict[str, Any]:
        """Get Semantic Similarity-specific export columns."""
        metrics = result.metrics
        return {
            'accuracy': round(metrics.get('accuracy', 0), 4),
            'avg_margin': round(metrics.get('avg_margin', 0), 4),
            'avg_positive_sim': round(metrics.get('avg_positive_sim', 0), 4),
            'avg_negative_sim': round(metrics.get('avg_negative_sim', 0), 4),
            'margin_gt_02': round(metrics.get('margin_gt_02', 0), 4)
        }

    @staticmethod
    def format_markdown_report(result: TaskResult) -> List[str]:
        """Format Semantic Similarity results as markdown."""
        import pandas as pd

        metrics = result.metrics
        lines = []

        lines.append("#### Core Metrics")
        lines.append("")

        # Core metrics table
        core_data = {
            'Metric': [
                'Triplet Accuracy',
                'Average Margin',
                'Avg Positive Similarity',
                'Avg Negative Similarity',
                'Margin > 0.2'
            ],
            'Value': [
                f"{metrics.get('accuracy', 0):.2%}",
                f"{metrics.get('avg_margin', 0):+.3f}",
                f"{metrics.get('avg_positive_sim', 0):.3f}",
                f"{metrics.get('avg_negative_sim', 0):.3f}",
                f"{metrics.get('margin_gt_02', 0):.2%}"
            ]
        }
        df_core = pd.DataFrame(core_data)
        lines.append(df_core.to_markdown(index=False))

        # Difficulty breakdown
        if 'difficulty_breakdown' in metrics:
            lines.append("")
            lines.append("#### Performance by Difficulty")
            lines.append("")
            diff_data = []
            for diff, diff_metrics in metrics['difficulty_breakdown'].items():
                diff_data.append({
                    'Difficulty': diff,
                    'Accuracy': f"{diff_metrics.get('accuracy', 0):.2%}",
                    'Count': diff_metrics.get('count', 0)
                })
            df_diff = pd.DataFrame(diff_data)
            lines.append(df_diff.to_markdown(index=False))

        # Subcategory breakdown
        if 'subcategory_breakdown' in metrics:
            lines.append("")
            lines.append("#### Performance by Subcategory")
            lines.append("")
            subcat_data = []
            for subcat, subcat_metrics in metrics['subcategory_breakdown'].items():
                subcat_data.append({
                    'Subcategory': subcat,
                    'Accuracy': f"{subcat_metrics.get('accuracy', 0):.2%}",
                    'Count': subcat_metrics.get('count', 0)
                })
            df_subcat = pd.DataFrame(subcat_data)
            lines.append(df_subcat.to_markdown(index=False))

        # Analysis
        lines.append("")
        lines.append("#### Analysis")
        lines.append("")

        # Performance interpretation
        accuracy = metrics.get('accuracy', 0)
        if accuracy >= 0.9:
            lines.append("- ✅ **Excellent** semantic discrimination (accuracy ≥ 90%)")
        elif accuracy >= 0.8:
            lines.append("- 🟢 **Good** semantic discrimination (accuracy ≥ 80%)")
        elif accuracy >= 0.7:
            lines.append("- 🟡 **Moderate** semantic discrimination (accuracy ≥ 70%)")
        else:
            lines.append("- 🔴 **Poor** semantic discrimination (accuracy < 70%)")

        # Margin analysis
        avg_margin = metrics.get('avg_margin', 0)
        if avg_margin >= 0.3:
            lines.append("- ✅ **Strong** separation between positive and negative examples")
        elif avg_margin >= 0.2:
            lines.append("- 🟢 **Good** separation between positive and negative examples")
        elif avg_margin >= 0.1:
            lines.append("- 🟡 **Weak** separation between positive and negative examples")
        else:
            lines.append("- 🔴 **Very weak** separation between positive and negative examples")

        # Failed triplets
        if 'failed_triplets_sample' in metrics and metrics['failed_triplets_sample']:
            failed_count = len(metrics['failed_triplets_sample'])
            lines.append("")
            lines.append(f"**Failed Triplets:** {failed_count} examples shown")
            lines.append("")

        return lines
