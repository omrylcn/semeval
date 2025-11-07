"""Results exporter for different output formats.

Export evaluation results to JSON, CSV, and Markdown formats.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

from ..tasks import get_task_class


class ResultsExporter:
    """Export evaluation results to various formats.

    Supports JSON, CSV, and Markdown output formats for easy sharing
    and analysis of evaluation results. Uses pandas for structured data handling.

    Examples
    --------
    >>> from semeval import TaskRunner, SentenceTransformerEncoder
    >>> from semeval.postprocess import ResultsExporter
    >>>
    >>> # Run evaluation
    >>> encoder = SentenceTransformerEncoder("model-name")
    >>> runner = TaskRunner(encoder=encoder)
    >>> result = runner.run("data/test_data.json")
    >>>
    >>> # Export results
    >>> exporter = ResultsExporter()
    >>> exporter.export_json(result, "results/output.json")
    >>> exporter.export_markdown(result, "results/report.md")
    >>> df = exporter.export_csv(result, "results/metrics.csv")
    >>>
    >>> # Or get DataFrame directly
    >>> df = exporter.to_dataframe(result)
    >>> print(df.describe())
    """

    def to_dataframe(
        self,
        result,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """Convert evaluation results to pandas DataFrame.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to convert
        include_metadata : bool, optional
            Whether to include test metadata (default: True)

        Returns
        -------
        pd.DataFrame
            DataFrame containing evaluation results

        Examples
        --------
        >>> exporter = ResultsExporter()
        >>> df = exporter.to_dataframe(result)
        >>> print(df[df['task'] == 'semantic_similarity'])
        """
        summary = result.get_summary()
        rows = []

        for task_name, task_info in summary['tasks'].items():
            row = {
                'task': task_name,
                'status': task_info['status'],
                'runtime_seconds': round(task_info['runtime'], 3)
            }

            if include_metadata:
                row['test_version'] = summary['metadata'].get('version', '')
                row['language'] = summary['metadata'].get('language', '')
                row['domain'] = summary['metadata'].get('domain', '')

            # Add task-specific metrics using task's export method
            if task_info['status'] == 'success':
                # Get task class from registry
                task_class = get_task_class(task_name)

                if task_class:
                    # Use task's export method to get columns
                    # Need to create a minimal TaskResult-like object from task_info
                    from ..tasks import TaskResult
                    task_result = TaskResult(
                        task_name=task_name,
                        status=task_info['status'],
                        metrics=task_info['metrics'],
                        runtime_seconds=task_info['runtime'],
                        metadata={}
                    )
                    export_columns = task_class.get_export_columns(task_result)
                    row.update(export_columns)
                else:
                    # Fallback: use all metrics
                    row.update(task_info['metrics'])

            else:
                row['error'] = task_info.get('error', '')

            rows.append(row)

        return pd.DataFrame(rows)

    def export_json(
        self,
        result,
        output_path: str,
        indent: int = 2,
        ensure_ascii: bool = False
    ) -> None:
        """Export results to JSON format.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to export
        output_path : str
            Path to output JSON file
        indent : int, optional
            JSON indentation level (default: 2)
        ensure_ascii : bool, optional
            Whether to escape non-ASCII characters (default: False)

        Examples
        --------
        >>> exporter = ResultsExporter()
        >>> exporter.export_json(result, "results.json")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = result.get_summary()

        # Add export metadata
        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "export_format": "json",
            **summary
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=indent, ensure_ascii=ensure_ascii)

        print(f"✅ JSON exported to: {output_path}")

    def export_csv(
        self,
        result,
        output_path: str,
        include_metadata: bool = True
    ) -> pd.DataFrame:
        """Export results to CSV format using pandas.

        Creates a structured DataFrame with one row per task, including key metrics.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to export
        output_path : str
            Path to output CSV file
        include_metadata : bool, optional
            Whether to include test metadata (default: True)

        Returns
        -------
        pd.DataFrame
            DataFrame containing the exported data

        Examples
        --------
        >>> exporter = ResultsExporter()
        >>> df = exporter.export_csv(result, "metrics.csv")
        >>> print(df.head())
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use to_dataframe for consistency
        df = self.to_dataframe(result, include_metadata=include_metadata)

        # Save to CSV
        df.to_csv(output_path, index=False, encoding='utf-8')

        print(f"✅ CSV exported to: {output_path}")
        return df

    def export_markdown(
        self,
        result,
        output_path: str,
        include_details: bool = True
    ) -> None:
        """Export results to Markdown format.

        Creates a human-readable Markdown report.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to export
        output_path : str
            Path to output Markdown file
        include_details : bool, optional
            Whether to include detailed metrics (default: True)

        Examples
        --------
        >>> exporter = ResultsExporter()
        >>> exporter.export_markdown(result, "report.md")
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        summary = result.get_summary()
        lines = []

        # Header
        lines.append("# Evaluation Results")
        lines.append("")
        lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Metadata
        lines.append("## Test Metadata")
        lines.append("")
        metadata = summary['metadata']
        lines.append(f"- **Version:** {metadata.get('version', 'N/A')}")
        lines.append(f"- **Description:** {metadata.get('description', 'N/A')}")
        lines.append(f"- **Language:** {metadata.get('language', 'N/A')}")
        if metadata.get('domain'):
            lines.append(f"- **Domain:** {metadata['domain']}")
        lines.append(f"- **Total Runtime:** {summary['total_runtime']:.2f}s")
        lines.append("")

        # Task results
        lines.append("## Task Results")
        lines.append("")

        for task_name, task_info in summary['tasks'].items():
            lines.append(f"### {task_name.replace('_', ' ').title()}")
            lines.append("")
            lines.append(f"- **Status:** {task_info['status']}")
            lines.append(f"- **Runtime:** {task_info['runtime']:.2f}s")
            lines.append("")

            if task_info['status'] == 'success' and include_details:
                # Get task class from registry
                task_class = get_task_class(task_name)

                if task_class:
                    # Use task's markdown report method
                    from ..tasks import TaskResult
                    task_result = TaskResult(
                        task_name=task_name,
                        status=task_info['status'],
                        metrics=task_info['metrics'],
                        runtime_seconds=task_info['runtime'],
                        metadata={}
                    )
                    task_lines = task_class.format_markdown_report(task_result)
                    lines.extend(task_lines)
                else:
                    # Fallback: basic metrics table
                    lines.append("#### Key Metrics")
                    lines.append("")
                    lines.append("| Metric | Value |")
                    lines.append("|--------|-------|")
                    for key, value in task_info['metrics'].items():
                        if isinstance(value, float):
                            lines.append(f"| {key} | {value:.4f} |")
                        else:
                            lines.append(f"| {key} | {value} |")

                lines.append("")

            elif task_info['status'] == 'failed':
                lines.append(f"**Error:** {task_info.get('error', 'Unknown error')}")
                lines.append("")

        # Write file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        print(f"✅ Markdown exported to: {output_path}")

    def export_all(
        self,
        result,
        output_dir: str,
        base_name: str = "results"
    ) -> Dict[str, str]:
        """Export results to all supported formats.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to export
        output_dir : str
            Directory for output files
        base_name : str, optional
            Base name for output files (default: "results")

        Returns
        -------
        dict
            Dictionary mapping format to output path

        Examples
        --------
        >>> exporter = ResultsExporter()
        >>> paths = exporter.export_all(result, "output/")
        >>> print(paths['json'])
        output/results.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = {}

        # Export to each format
        paths['json'] = str(output_dir / f"{base_name}.json")
        self.export_json(result, paths['json'])

        paths['csv'] = str(output_dir / f"{base_name}.csv")
        self.export_csv(result, paths['csv'])

        paths['markdown'] = str(output_dir / f"{base_name}.md")
        self.export_markdown(result, paths['markdown'])

        print(f"\n✅ All formats exported to: {output_dir}/")
        return paths

    def export_per_task(
        self,
        result,
        output_dir: str,
        export_formats: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """Export each task to separate files.

        Creates individual JSON and Markdown files for each task.

        Parameters
        ----------
        result : EvaluationResult
            Evaluation result to export
        output_dir : str
            Directory for output files
        export_formats : list of str, optional
            Formats to export ('json', 'markdown'). Default: both

        Returns
        -------
        dict
            Nested dictionary: {task_name: {format: path}}

        Examples
        --------
        >>> exporter = ResultsExporter()
        >>> paths = exporter.export_per_task(result, "output/")
        >>> print(paths['information_retrieval']['json'])
        output/information_retrieval_result.json
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if export_formats is None:
            export_formats = ['json', 'markdown']

        summary = result.get_summary()
        task_paths = {}

        from ..tasks import get_task_class, TaskResult

        print("\n📁 Exporting per-task files...")

        for task_name, task_info in summary['tasks'].items():
            task_paths[task_name] = {}

            # Create TaskResult object
            task_result = TaskResult(
                task_name=task_name,
                status=task_info['status'],
                metrics=task_info['metrics'],
                runtime_seconds=task_info['runtime'],
                metadata={}
            )

            # Get task class for formatting
            task_class = get_task_class(task_name)

            # Export JSON
            if 'json' in export_formats:
                json_path = output_dir / f"{task_name}_result.json"
                task_data = {
                    'task_name': task_name,
                    'status': task_info['status'],
                    'runtime_seconds': task_info['runtime'],
                    'metrics': task_info['metrics'],
                    'timestamp': task_result.timestamp
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(task_data, f, indent=2, ensure_ascii=False)

                task_paths[task_name]['json'] = str(json_path)
                print(f"   ✅ {task_name}.json")

            # Export Markdown
            if 'markdown' in export_formats:
                md_path = output_dir / f"{task_name}_result.md"
                lines = []

                # Header
                lines.append(f"# {task_name.replace('_', ' ').title()}")
                lines.append("")
                lines.append(f"**Status:** {task_info['status']}")
                lines.append(f"**Runtime:** {task_info['runtime']:.2f}s")
                lines.append("")

                # Task-specific content using task's formatter
                if task_class and task_info['status'] == 'success':
                    task_lines = task_class.format_markdown_report(task_result)
                    lines.extend(task_lines)
                elif task_info['status'] == 'failed':
                    lines.append("## Error")
                    lines.append("")
                    lines.append(f"```\n{task_info.get('error', 'Unknown error')}\n```")

                with open(md_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))

                task_paths[task_name]['markdown'] = str(md_path)
                print(f"   ✅ {task_name}.md")

        print(f"\n✅ Per-task files exported to: {output_dir}/")
        return task_paths
