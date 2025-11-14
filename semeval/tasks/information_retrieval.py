"""Information Retrieval evaluation task implementation.

This module implements the Information Retrieval task which evaluates
a model's ability to rank relevant documents higher than irrelevant ones.
"""

import time
from typing import Any, Dict, List, Optional

from sentence_transformers import util
from sentence_transformers.evaluation import InformationRetrievalEvaluator

from ..core.schemas import InformationRetrievalData
from .base import BaseTask, TaskResult


class InformationRetrieval(BaseTask):
    """Information Retrieval evaluation task.

    This task evaluates how well a model can retrieve and rank relevant
    documents given a query.

    Parameters
    ----------
    encoder : BaseEncoder
        Text encoder to use for embeddings
    task_data : InformationRetrievalData
        IR task configuration and test data
    device : str, optional
        Device for computation
    verbose : bool, optional
        If True, print progress information

    Attributes
    ----------
    task_data : InformationRetrievalData
        Task configuration and data
    evaluator : InformationRetrievalEvaluator, optional
        Sentence Transformers evaluator (only for SentenceTransformer models)

    Examples
    --------
    >>> from semeval.core.encoders import SentenceTransformerEncoder
    >>> from semeval.core.loaders import JSONDataLoader
    >>>
    >>> # Load data
    >>> loader = JSONDataLoader()
    >>> test_data = loader.load("data/test_data.json")
    >>> ir_data = test_data.tasks.information_retrieval
    >>>
    >>> # Create encoder
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>>
    >>> # Run task
    >>> task = InformationRetrieval(encoder=encoder, task_data=ir_data, verbose=True)
    >>> result = task.run()
    >>> print(f"NDCG@10: {result.metrics['ndcg@10']:.4f}")
    """

    def __init__(
        self,
        encoder,
        task_data: InformationRetrievalData,
        device: Optional[str] = None,
        verbose: bool = False,
    ):
        """Initialize Information Retrieval task."""
        super().__init__(encoder, task_data, device, verbose)
        self.evaluator = None

    def get_task_name(self) -> str:
        """Return task name.

        Returns
        -------
        str
            Task name
        """
        return "information_retrieval"

    def _use_sentence_transformers_evaluator(self) -> bool:
        """Check if we can use SentenceTransformers evaluator.

        Returns
        -------
        bool
            True if encoder has a 'model' attribute (SentenceTransformer)
        """
        return hasattr(self.encoder, "model") and hasattr(self.encoder.model, "encode")

    def _prepare_evaluator_data(self) -> tuple:
        """Prepare data in format expected by InformationRetrievalEvaluator.

        Returns
        -------
        tuple
            (queries, corpus, relevant_docs) in evaluator format
        """
        # Convert relevant_docs from {qid: {did: score}} to {qid: set(did)}
        # Only include docs with score > 0
        relevant_docs = {}
        for query_id, docs in self.task_data.relevant_docs.items():
            relevant_docs[query_id] = {
                doc_id for doc_id, score in docs.items() if score > 0
            }

        return (self.task_data.queries, self.task_data.corpus, relevant_docs)

    def _run_with_sentence_transformers(self) -> Dict[str, Any]:
        """Run evaluation using SentenceTransformers InformationRetrievalEvaluator.

        Returns
        -------
        dict
            Evaluation metrics
        """
        self._log("Using SentenceTransformers InformationRetrievalEvaluator")

        queries, corpus, relevant_docs = self._prepare_evaluator_data()

        # Create evaluator
        # Only pass supported parameters
        evaluator_params = {}
        if self.task_data.config.ndcg_at_k:
            evaluator_params["ndcg_at_k"] = self.task_data.config.ndcg_at_k
        if self.task_data.config.map_at_k:
            evaluator_params["map_at_k"] = self.task_data.config.map_at_k
        if self.task_data.config.mrr_at_k:
            evaluator_params["mrr_at_k"] = self.task_data.config.mrr_at_k

        evaluator = InformationRetrievalEvaluator(
            queries=queries,
            corpus=corpus,
            relevant_docs=relevant_docs,
            name="ir_eval",
            show_progress_bar=self.verbose,
            **evaluator_params,
        )

        # Run evaluation
        # Note: evaluator returns accuracy (1.0) and writes results to CSV
        # We need to call it and then read back the metrics
        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            # Run evaluator (it will write to CSV)
            _ = evaluator(self.encoder.model, output_path=tmpdir)

            # Read back the CSV to get all metrics
            csv_path = os.path.join(
                tmpdir, "Information-Retrieval_evaluation_ir_eval_results.csv"
            )

            if os.path.exists(csv_path):
                import csv

                with open(csv_path, "r") as f:
                    reader = csv.DictReader(f)
                    rows = list(reader)
                    if rows:
                        # Last row contains the metrics
                        metrics = {
                            k: float(v)
                            for k, v in rows[-1].items()
                            if k not in ["epoch", "steps"]
                        }
                        return metrics

        # Fallback: return empty metrics
        return {}

    def _run_with_manual_search(self) -> Dict[str, Any]:
        """Run evaluation using manual semantic search.

        This is used when encoder is not a SentenceTransformer model.

        Returns
        -------
        dict
            Evaluation metrics
        """
        self._log("Using manual semantic search")

        # Encode corpus
        self._log(f"Encoding corpus ({len(self.task_data.corpus)} documents)...")
        corpus_ids = list(self.task_data.corpus.keys())
        corpus_texts = [self.task_data.corpus[doc_id] for doc_id in corpus_ids]

        corpus_embeddings = self.encoder.encode(
            corpus_texts,
            convert_to_tensor=True,
            show_progress_bar=self.verbose,
            batch_size=32,
        )

        # Encode queries
        self._log(f"Encoding queries ({len(self.task_data.queries)} queries)...")
        query_ids = list(self.task_data.queries.keys())
        query_texts = [self.task_data.queries[qid] for qid in query_ids]

        query_embeddings = self.encoder.encode(
            query_texts,
            convert_to_tensor=True,
            show_progress_bar=self.verbose,
            batch_size=32,
        )

        # Perform semantic search
        self._log("Performing semantic search...")
        all_hits = util.semantic_search(
            query_embeddings,
            corpus_embeddings,
            top_k=max(self.task_data.config.ndcg_at_k),
        )

        # Calculate metrics for each query
        from ..metrics.ir_metrics import aggregate_metrics, compute_ranking_metrics

        all_query_metrics = []

        for query_id, hits in zip(query_ids, all_hits):
            # Get retrieved doc IDs in order
            retrieved_doc_ids = [corpus_ids[hit["corpus_id"]] for hit in hits]

            # Get relevant docs for this query
            relevant_docs = self.task_data.relevant_docs.get(query_id, {})

            # Compute metrics
            metrics = compute_ranking_metrics(
                retrieved_doc_ids=retrieved_doc_ids,
                relevant_doc_ids=relevant_docs,
                k_values=self.task_data.config.ndcg_at_k,
            )

            all_query_metrics.append(metrics)

        # Aggregate across all queries
        aggregated_metrics = aggregate_metrics(all_query_metrics)

        # Rename to match sentence_transformers format
        renamed_metrics = {}
        for metric_name, value in aggregated_metrics.items():
            # Convert "ndcg@10" to "cosine-NDCG@10" for compatibility
            renamed_metrics[f"cosine-{metric_name.upper().replace('@', '@')}"] = value

        return renamed_metrics

    def run(self) -> TaskResult:
        """Execute the Information Retrieval evaluation task.

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
        >>> task = InformationRetrieval(encoder=encoder, task_data=data)
        >>> result = task.run()
        >>> print(f"Status: {result.status}")
        success
        >>> print(f"NDCG@10: {result.metrics.get('cosine-NDCG@10', 0):.4f}")
        0.9872
        """
        self._log(f"Starting {self.task_data.name}")
        self._log(f"Corpus: {len(self.task_data.corpus)} documents")
        self._log(f"Queries: {len(self.task_data.queries)} queries")

        start_time = time.time()

        try:
            # Choose evaluation method based on encoder type
            if self._use_sentence_transformers_evaluator():
                metrics = self._run_with_sentence_transformers()
            else:
                metrics = self._run_with_manual_search()

            runtime = time.time() - start_time

            self._log(f"Evaluation completed in {runtime:.2f}s")

            # Create result
            result = TaskResult(
                task_name=self.get_task_name(),
                status="success",
                metrics=metrics,
                runtime_seconds=runtime,
                metadata={
                    "corpus_size": len(self.task_data.corpus),
                    "query_count": len(self.task_data.queries),
                    "total_relevance_judgments": sum(
                        len(docs) for docs in self.task_data.relevant_docs.values()
                    ),
                    "config": self.task_data.config.model_dump(),
                },
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
                metadata={},
            )

    @staticmethod
    def get_export_columns(result: TaskResult) -> Dict[str, Any]:
        """Get IR-specific export columns."""
        metrics = result.metrics
        return {
            "ndcg@10": round(metrics.get("cosine-NDCG@10", 0), 4),
            "mrr@10": round(metrics.get("cosine-MRR@10", 0), 4),
            "map@10": round(metrics.get("cosine-MAP@10", 0), 4),
            "precision@10": round(metrics.get("cosine-PRECISION@10", 0), 4),
            "recall@10": round(metrics.get("cosine-RECALL@10", 0), 4),
        }

    @staticmethod
    def format_markdown_report(result: TaskResult) -> List[str]:
        """Format IR results as markdown."""
        import pandas as pd

        metrics = result.metrics
        lines = []

        lines.append("#### Performance Metrics")
        lines.append("")

        # Create DataFrame for metrics table
        data = {
            "Metric": ["NDCG", "MRR", "MAP"],
            "@1": [
                metrics.get("cosine-NDCG@1", 0),
                metrics.get("cosine-MRR@1", 0),
                metrics.get("cosine-MAP@1", 0),
            ],
            "@3": [
                metrics.get("cosine-NDCG@3", 0),
                metrics.get("cosine-MRR@3", 0),
                metrics.get("cosine-MAP@3", 0),
            ],
            "@5": [
                metrics.get("cosine-NDCG@5", 0),
                metrics.get("cosine-MRR@5", 0),
                metrics.get("cosine-MAP@5", 0),
            ],
            "@10": [
                metrics.get("cosine-NDCG@10", 0),
                metrics.get("cosine-MRR@10", 0),
                metrics.get("cosine-MAP@10", 0),
            ],
        }
        df = pd.DataFrame(data)

        # Format numbers to 4 decimals
        for col in ["@1", "@3", "@5", "@10"]:
            df[col] = df[col].apply(lambda x: f"{x:.4f}")

        lines.append(df.to_markdown(index=False))
        lines.append("")
        lines.append("#### Analysis")
        lines.append("")

        # Performance interpretation
        ndcg_10 = metrics.get("cosine-NDCG@10", 0)
        if ndcg_10 >= 0.8:
            lines.append("- 🟢 **Excellent** retrieval performance (NDCG@10 ≥ 0.8)")
        elif ndcg_10 >= 0.7:
            lines.append("- 🟢 **Good** retrieval performance (NDCG@10 ≥ 0.7)")
        elif ndcg_10 >= 0.5:
            lines.append("- 🟡 **Moderate** retrieval performance (NDCG@10 ≥ 0.5)")
        else:
            lines.append("- 🔴 **Poor** retrieval performance (NDCG@10 < 0.5)")

        return lines
