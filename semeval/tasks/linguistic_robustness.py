"""Linguistic Robustness evaluation task implementation.

This module implements the Linguistic Robustness task which evaluates
a model's robustness to morphology, typos, and negation.
"""

import time
from typing import Any, Dict, List, Optional

from sentence_transformers import util

from ..core.schemas import LinguisticRobustnessData
from ..metrics.robustness_metrics import (
    compute_morphology_metrics,
    compute_negation_metrics,
    compute_robustness_summary,
    compute_typo_metrics,
)
from .base import BaseTask, TaskResult


class LinguisticRobustness(BaseTask):
    """Linguistic Robustness evaluation task.

    This task evaluates model's robustness to:
    1. Morphological variations (Turkish suffixes, inflections)
    2. Typos and spelling errors
    3. Negation and semantic opposition

    Parameters
    ----------
    encoder : BaseEncoder
        Text encoder to use for embeddings
    task_data : LinguisticRobustnessData
        Linguistic Robustness task configuration and test data
    device : str, optional
        Device for computation
    verbose : bool, optional
        If True, print progress information

    Attributes
    ----------
    task_data : LinguisticRobustnessData
        Task configuration and data

    Examples
    --------
    >>> from semeval.core.encoders import SentenceTransformerEncoder
    >>> from semeval.core.loaders import JSONDataLoader
    >>>
    >>> # Load data
    >>> loader = JSONDataLoader()
    >>> test_data = loader.load("data/test_data.json")
    >>> lr_data = test_data.tasks.linguistic_robustness
    >>>
    >>> # Create encoder
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>>
    >>> # Run task
    >>> task = LinguisticRobustness(encoder=encoder, task_data=lr_data, verbose=True)
    >>> result = task.run()
    >>> print(f"Overall Success: {result.metrics['overall_success_rate']:.1%}")
    """

    def __init__(
        self,
        encoder,
        task_data: LinguisticRobustnessData,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize Linguistic Robustness task."""
        super().__init__(encoder, task_data, device, verbose)

    def get_task_name(self) -> str:
        """Return task name.

        Returns
        -------
        str
            Task name
        """
        return "linguistic_robustness"

    def _evaluate_morphology(self) -> Dict[str, Any]:
        """Evaluate morphology robustness.

        Returns
        -------
        dict
            Morphology metrics
        """
        if not self.task_data.morphology:
            self._log("No morphology pairs to evaluate", level="WARNING")
            return {
                'avg_similarity': 0.0,
                'success_rate': 0.0,
                'total_pairs': 0,
                'category_breakdown': {}
            }

        self._log(f"Evaluating morphology ({len(self.task_data.morphology)} pairs)...")

        # Extract texts and categories
        original_texts = [p.original for p in self.task_data.morphology]
        modified_texts = [p.modified for p in self.task_data.morphology]
        categories = [p.category for p in self.task_data.morphology]

        # Encode in batches
        self._log("Encoding original texts...")
        original_embeddings = self.encoder.encode(
            original_texts,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True
        )

        self._log("Encoding modified texts...")
        modified_embeddings = self.encoder.encode(
            modified_texts,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True
        )

        # Compute similarities
        self._log("Computing similarities...")
        similarities = []
        for orig_emb, mod_emb in zip(original_embeddings, modified_embeddings):
            sim = util.cos_sim(orig_emb, mod_emb).item()
            similarities.append(sim)

        # Compute metrics
        metrics = compute_morphology_metrics(
            similarities,
            categories,
            threshold=self.task_data.config.morphology_threshold
        )

        self._log(f"Morphology: avg_sim={metrics['avg_similarity']:.3f}, " +
                  f"success={metrics['success_rate']:.1%}")

        return metrics

    def _evaluate_typo(self) -> Dict[str, Any]:
        """Evaluate typo robustness.

        Returns
        -------
        dict
            Typo metrics
        """
        if not self.task_data.typo:
            self._log("No typo pairs to evaluate", level="WARNING")
            return {
                'avg_similarity': 0.0,
                'success_rate': 0.0,
                'total_pairs': 0,
                'type_breakdown': {}
            }

        self._log(f"Evaluating typo ({len(self.task_data.typo)} pairs)...")

        # Extract texts and types
        correct_texts = [p.correct for p in self.task_data.typo]
        typo_texts = [p.typo for p in self.task_data.typo]
        types = [p.type for p in self.task_data.typo]

        # Encode in batches
        self._log("Encoding correct texts...")
        correct_embeddings = self.encoder.encode(
            correct_texts,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True
        )

        self._log("Encoding typo texts...")
        typo_embeddings = self.encoder.encode(
            typo_texts,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True
        )

        # Compute similarities
        self._log("Computing similarities...")
        similarities = []
        for correct_emb, typo_emb in zip(correct_embeddings, typo_embeddings):
            sim = util.cos_sim(correct_emb, typo_emb).item()
            similarities.append(sim)

        # Compute metrics
        metrics = compute_typo_metrics(
            similarities,
            types,
            threshold=self.task_data.config.typo_threshold
        )

        self._log(f"Typo: avg_sim={metrics['avg_similarity']:.3f}, " +
                  f"success={metrics['success_rate']:.1%}")

        return metrics

    def _evaluate_negation(self) -> Dict[str, Any]:
        """Evaluate negation/opposition detection.

        Returns
        -------
        dict
            Negation metrics
        """
        if not self.task_data.negation:
            self._log("No negation pairs to evaluate", level="WARNING")
            return {
                'avg_similarity': 0.0,
                'success_rate': 0.0,
                'total_pairs': 0,
                'type_breakdown': {}
            }

        self._log(f"Evaluating negation ({len(self.task_data.negation)} pairs)...")

        # Extract texts and types
        sentence1_texts = [p.sentence1 for p in self.task_data.negation]
        sentence2_texts = [p.sentence2 for p in self.task_data.negation]
        types = [p.type for p in self.task_data.negation]

        # Encode in batches
        self._log("Encoding sentence1 texts...")
        sentence1_embeddings = self.encoder.encode(
            sentence1_texts,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True
        )

        self._log("Encoding sentence2 texts...")
        sentence2_embeddings = self.encoder.encode(
            sentence2_texts,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True
        )

        # Compute similarities
        self._log("Computing similarities...")
        similarities = []
        for sent1_emb, sent2_emb in zip(sentence1_embeddings, sentence2_embeddings):
            sim = util.cos_sim(sent1_emb, sent2_emb).item()
            similarities.append(sim)

        # Compute metrics
        metrics = compute_negation_metrics(
            similarities,
            types,
            threshold=self.task_data.config.negation_threshold
        )

        self._log(f"Negation: avg_sim={metrics['avg_similarity']:.3f} (should be low), " +
                  f"success={metrics['success_rate']:.1%}")

        return metrics

    def run(self) -> TaskResult:
        """Execute the Linguistic Robustness evaluation task.

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
        >>> task = LinguisticRobustness(encoder=encoder, task_data=data)
        >>> result = task.run()
        >>> print(f"Status: {result.status}")
        success
        >>> print(f"Overall Success: {result.metrics['overall_success_rate']:.1%}")
        85.0%
        """
        self._log(f"Starting {self.task_data.name}")
        self._log(f"Morphology pairs: {len(self.task_data.morphology)}")
        self._log(f"Typo pairs: {len(self.task_data.typo)}")
        self._log(f"Negation pairs: {len(self.task_data.negation)}")

        start_time = time.time()

        try:
            # Run subtasks
            morphology_result = self._evaluate_morphology()
            typo_result = self._evaluate_typo()
            negation_result = self._evaluate_negation()

            # Compute overall summary
            summary = compute_robustness_summary(
                morphology_result,
                typo_result,
                negation_result
            )

            # Combine all metrics
            metrics = {
                **summary,
                'morphology': morphology_result,
                'typo': typo_result,
                'negation': negation_result
            }

            runtime = time.time() - start_time

            self._log(f"Evaluation completed in {runtime:.2f}s")
            self._log(f"Overall Success Rate: {metrics['overall_success_rate']:.1%}")

            # Create result
            result = TaskResult(
                task_name=self.get_task_name(),
                status="success",
                metrics=metrics,
                runtime_seconds=runtime,
                metadata={
                    "morphology_count": len(self.task_data.morphology),
                    "typo_count": len(self.task_data.typo),
                    "negation_count": len(self.task_data.negation),
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
        """Get Linguistic Robustness-specific export columns."""
        metrics = result.metrics
        return {
            'overall_success_rate': round(metrics.get('overall_success_rate', 0), 4),
            'morphology_success': round(metrics.get('morphology', {}).get('success_rate', 0), 4),
            'typo_success': round(metrics.get('typo', {}).get('success_rate', 0), 4),
            'negation_success': round(metrics.get('negation', {}).get('success_rate', 0), 4)
        }

    @staticmethod
    def format_markdown_report(result: TaskResult) -> List[str]:
        """Format Linguistic Robustness results as markdown."""
        import pandas as pd

        metrics = result.metrics
        lines = []

        lines.append("#### Key Metrics")
        lines.append("")

        morph_metrics = metrics.get('morphology', {})
        typo_metrics = metrics.get('typo', {})
        neg_metrics = metrics.get('negation', {})

        # Main metrics table
        main_data = {
            'Metric': [
                'Overall Success Rate',
                'Morphology Success',
                'Typo Success',
                'Negation Success'
            ],
            'Value': [
                f"{metrics.get('overall_success_rate', 0):.2%}",
                f"{morph_metrics.get('success_rate', 0):.2%}",
                f"{typo_metrics.get('success_rate', 0):.2%}",
                f"{neg_metrics.get('success_rate', 0):.2%}"
            ]
        }
        df_main = pd.DataFrame(main_data)
        lines.append(df_main.to_markdown(index=False))

        # Morphology category breakdown
        if 'category_breakdown' in morph_metrics and morph_metrics['category_breakdown']:
            lines.append("")
            lines.append("**Morphology by Category:**")
            lines.append("")
            morph_data = []
            for cat, cat_metrics in morph_metrics['category_breakdown'].items():
                morph_data.append({
                    'Category': cat,
                    'Avg Similarity': f"{cat_metrics.get('avg_similarity', 0):.3f}",
                    'Success': f"{cat_metrics.get('success_rate', 0):.1%}",
                    'Count': cat_metrics.get('count', 0)
                })
            df_morph = pd.DataFrame(morph_data)
            lines.append(df_morph.to_markdown(index=False))

        # Typo type breakdown
        if 'type_breakdown' in typo_metrics and typo_metrics['type_breakdown']:
            lines.append("")
            lines.append("**Typo by Type:**")
            lines.append("")
            typo_data = []
            for typ, typ_metrics in typo_metrics['type_breakdown'].items():
                typo_data.append({
                    'Type': typ,
                    'Avg Similarity': f"{typ_metrics.get('avg_similarity', 0):.3f}",
                    'Success': f"{typ_metrics.get('success_rate', 0):.1%}",
                    'Count': typ_metrics.get('count', 0)
                })
            df_typo = pd.DataFrame(typo_data)
            lines.append(df_typo.to_markdown(index=False))

        # Negation type breakdown
        if 'type_breakdown' in neg_metrics and neg_metrics['type_breakdown']:
            lines.append("")
            lines.append("**Negation by Type:**")
            lines.append("")
            neg_data = []
            for typ, typ_metrics in neg_metrics['type_breakdown'].items():
                neg_data.append({
                    'Type': typ,
                    'Avg Similarity': f"{typ_metrics.get('avg_similarity', 0):.3f}",
                    'Success': f"{typ_metrics.get('success_rate', 0):.1%}",
                    'Count': typ_metrics.get('count', 0)
                })
            df_neg = pd.DataFrame(neg_data)
            lines.append(df_neg.to_markdown(index=False))

        return lines
