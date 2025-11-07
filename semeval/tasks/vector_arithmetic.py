"""Vector Arithmetic evaluation task implementation.

This module implements the Vector Arithmetic task which evaluates
a model's ability to solve analogies using vector arithmetic.
"""

import time
from typing import Dict, Any, List, Tuple, Optional
from sentence_transformers import util

from .base import BaseTask, TaskResult
from ..core.schemas import VectorArithmeticData
from ..metrics.arithmetic_metrics import (
    compute_analogy_metrics,
    get_failed_analogies
)


class VectorArithmetic(BaseTask):
    """Vector Arithmetic evaluation task using analogies.

    This task evaluates how well a model can solve analogies using
    vector arithmetic. Format: a - b + c ≈ d

    Example: "Ankara" - "Türkiye" + "Almanya" ≈ "Berlin"

    Parameters
    ----------
    encoder : BaseEncoder
        Text encoder to use for embeddings
    task_data : VectorArithmeticData
        Vector Arithmetic task configuration and test data
    device : str, optional
        Device for computation
    verbose : bool, optional
        If True, print progress information

    Attributes
    ----------
    task_data : VectorArithmeticData
        Task configuration and data
    vocabulary : list of str
        Vocabulary for analogy search
    vocabulary_embeddings : Tensor
        Encoded vocabulary embeddings

    Examples
    --------
    >>> from semeval.core.encoders import SentenceTransformerEncoder
    >>> from semeval.core.loaders import JSONDataLoader
    >>>
    >>> # Load data
    >>> loader = JSONDataLoader()
    >>> test_data = loader.load("data/test_data.json")
    >>> va_data = test_data.tasks.vector_arithmetic
    >>>
    >>> # Create encoder
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>>
    >>> # Run task
    >>> task = VectorArithmetic(encoder=encoder, task_data=va_data, verbose=True)
    >>> result = task.run()
    >>> print(f"Top-5 Accuracy: {result.metrics['top_k_accuracy'][5]:.1%}")
    """

    def __init__(
        self,
        encoder,
        task_data: VectorArithmeticData,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """Initialize Vector Arithmetic task."""
        super().__init__(encoder, task_data, device, verbose)
        self.vocabulary = None
        self.vocabulary_embeddings = None

    def get_task_name(self) -> str:
        """Return task name.

        Returns
        -------
        str
            Task name
        """
        return "vector_arithmetic"

    def _build_vocabulary(self) -> None:
        """Build vocabulary from analogies and encode it."""
        self._log(f"Building vocabulary from {len(self.task_data.analogies)} analogies...")

        # Extract all unique words from analogies
        vocabulary_set = set()
        for analogy in self.task_data.analogies:
            vocabulary_set.update([analogy.a, analogy.b, analogy.c, analogy.d])

        # Convert to sorted list
        self.vocabulary = sorted(list(vocabulary_set))

        self._log(f"Vocabulary size: {len(self.vocabulary)} unique words")
        self._log("Encoding vocabulary...")

        # Encode vocabulary
        self.vocabulary_embeddings = self.encoder.encode(
            self.vocabulary,
            batch_size=32,
            show_progress_bar=self.verbose,
            convert_to_tensor=True
        )

        self._log(f"Vocabulary encoded: {self.vocabulary_embeddings.shape}")

    def _evaluate_analogies(self) -> Dict[str, Any]:
        """Evaluate all analogies.

        Returns
        -------
        dict
            Analogy evaluation metrics
        """
        total = len(self.task_data.analogies)
        self._log(f"Evaluating {total} analogies...")

        ranks = []
        categories = []
        subcategories = []
        all_predictions = []
        analogy_tuples = []

        for idx, analogy in enumerate(self.task_data.analogies):
            # Encode individual terms
            emb_a = self.encoder.encode(analogy.a, convert_to_tensor=True)
            emb_b = self.encoder.encode(analogy.b, convert_to_tensor=True)
            emb_c = self.encoder.encode(analogy.c, convert_to_tensor=True)

            # Vector arithmetic: a - b + c ≈ d
            target_embedding = emb_a - emb_b + emb_c

            # Find most similar words in vocabulary
            similarities = util.cos_sim(target_embedding, self.vocabulary_embeddings)[0]

            # Get top predictions (excluding input words a, b, c)
            exclude_words = {analogy.a, analogy.b, analogy.c}
            predictions = []

            for word_idx in similarities.argsort(descending=True):
                word = self.vocabulary[word_idx]
                if word not in exclude_words:
                    score = similarities[word_idx].item()
                    predictions.append((word, score))

                if len(predictions) >= 10:
                    break

            # Find rank of correct answer
            rank = -1
            for pred_idx, (word, _) in enumerate(predictions, 1):
                if word == analogy.d:
                    rank = pred_idx
                    break

            # If not found in top 10, set high rank
            if rank == -1:
                rank = 999

            ranks.append(rank)
            categories.append(analogy.category)
            subcategories.append(analogy.subcategory)
            all_predictions.append(predictions)
            analogy_tuples.append((
                analogy.a, analogy.b, analogy.c, analogy.d,
                analogy.category, analogy.subcategory
            ))

            # Log progress periodically
            if (idx + 1) % 10 == 0:
                self._log(f"Processed {idx + 1}/{total} analogies...")

        # Compute metrics
        metrics = compute_analogy_metrics(
            ranks,
            categories,
            subcategories,
            top_k=self.task_data.config.top_k
        )

        # Add failed analogies for error analysis
        failed = get_failed_analogies(
            analogy_tuples,
            ranks,
            all_predictions,
            threshold=5
        )
        metrics['failed_analogies_sample'] = failed[:10]  # Store top 10 failures

        return metrics

    def run(self) -> TaskResult:
        """Execute the Vector Arithmetic evaluation task.

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
        >>> task = VectorArithmetic(encoder=encoder, task_data=data)
        >>> result = task.run()
        >>> print(f"Status: {result.status}")
        success
        >>> print(f"Top-5 Accuracy: {result.metrics['top_k_accuracy'][5]:.1%}")
        75.0%
        """
        self._log(f"Starting {self.task_data.name}")
        self._log(f"Analogies: {len(self.task_data.analogies)}")

        start_time = time.time()

        try:
            # Build vocabulary
            self._build_vocabulary()

            # Evaluate analogies
            metrics = self._evaluate_analogies()

            runtime = time.time() - start_time

            self._log(f"Evaluation completed in {runtime:.2f}s")
            self._log(f"Top-5 Accuracy: {metrics['top_k_accuracy'].get(5, 0.0):.1%}")
            self._log(f"Mean Rank: {metrics['mean_rank']:.2f}")

            # Create result
            result = TaskResult(
                task_name=self.get_task_name(),
                status="success",
                metrics=metrics,
                runtime_seconds=runtime,
                metadata={
                    "analogy_count": len(self.task_data.analogies),
                    "vocabulary_size": len(self.vocabulary),
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
        """Get Vector Arithmetic-specific export columns."""
        metrics = result.metrics
        top_k = metrics.get('top_k_accuracy', {})
        return {
            'top1_accuracy': round(top_k.get(1, 0), 4),
            'top5_accuracy': round(top_k.get(5, 0), 4),
            'top10_accuracy': round(top_k.get(10, 0), 4),
            'mean_rank': round(metrics.get('mean_rank', 0), 4),
            'mrr': round(metrics.get('mrr', 0), 4)
        }

    @staticmethod
    def format_markdown_report(result: TaskResult) -> List[str]:
        """Format Vector Arithmetic results as markdown."""
        import pandas as pd

        metrics = result.metrics
        lines = []

        lines.append("#### Key Metrics")
        lines.append("")

        top_k = metrics.get('top_k_accuracy', {})

        # Main metrics table
        main_data = {
            'Metric': [
                'Top-1 Accuracy',
                'Top-5 Accuracy',
                'Top-10 Accuracy',
                'Mean Rank',
                'MRR'
            ],
            'Value': [
                f"{top_k.get(1, 0):.2%}",
                f"{top_k.get(5, 0):.2%}",
                f"{top_k.get(10, 0):.2%}",
                f"{metrics.get('mean_rank', 0):.2f}",
                f"{metrics.get('mrr', 0):.4f}"
            ]
        }
        df_main = pd.DataFrame(main_data)
        lines.append(df_main.to_markdown(index=False))

        # Subcategory breakdown
        if 'subcategory_breakdown' in metrics and metrics['subcategory_breakdown']:
            lines.append("")
            lines.append("**Performance by Subcategory:**")
            lines.append("")
            subcat_data = []
            for subcat, subcat_metrics in metrics['subcategory_breakdown'].items():
                subcat_data.append({
                    'Subcategory': subcat,
                    'Top-5 Acc': f"{subcat_metrics.get('top_k_accuracy', {}).get(5, 0):.1%}",
                    'Mean Rank': f"{subcat_metrics.get('mean_rank', 0):.1f}",
                    'Count': subcat_metrics.get('count', 0)
                })
            df_subcat = pd.DataFrame(subcat_data)
            lines.append(df_subcat.to_markdown(index=False))

        # Category breakdown (if available)
        if 'category_breakdown' in metrics and metrics['category_breakdown']:
            lines.append("")
            lines.append("**Performance by Category:**")
            lines.append("")
            cat_data = []
            for cat, cat_metrics in metrics['category_breakdown'].items():
                cat_data.append({
                    'Category': cat,
                    'Top-5 Acc': f"{cat_metrics.get('top_k_accuracy', {}).get(5, 0):.1%}",
                    'Mean Rank': f"{cat_metrics.get('mean_rank', 0):.1f}",
                    'Count': cat_metrics.get('count', 0)
                })
            df_cat = pd.DataFrame(cat_data)
            lines.append(df_cat.to_markdown(index=False))

        return lines
