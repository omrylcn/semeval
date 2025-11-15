# SemEval - Comprehensive Usage Guide

This guide provides detailed instructions on how to use the SemEval framework for evaluating semantic embeddings and NLP models, with a focus on Turkish language models.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)
4. [Configuration System](#configuration-system)
5. [Evaluation Tasks](#evaluation-tasks)
6. [Data Format](#data-format)
7. [Running Evaluations](#running-evaluations)
8. [Export & Reporting](#export--reporting)
9. [Advanced Topics](#advanced-topics)
10. [API Reference](#api-reference)
11. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is SemEval?

SemEval is a comprehensive evaluation framework for semantic embeddings and NLP models. It provides:

- **4 evaluation tasks** covering different aspects of semantic understanding
- **Flexible configuration** via YAML files and environment variables
- **Rich export options** including JSON, CSV, Markdown, and per-task reports
- **Type-safe** data structures using Pydantic V2
- **Extensible architecture** for adding custom tasks and encoders

### When to Use SemEval?

Use SemEval when you need to:

- Evaluate semantic embedding quality for Turkish (or other languages)
- Compare multiple models on standardized benchmarks
- Test model robustness to linguistic variations
- Assess information retrieval performance
- Validate semantic similarity understanding
- Test compositional semantic reasoning (vector arithmetic)

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip or uv package manager
- CUDA-capable GPU (optional, for faster processing)

### Quick Install

```bash
# Clone the repository
git clone <repository-url>
cd semeval

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### Verify Installation

```python
from semeval import TaskRunner, SentenceTransformerEncoder
print("SemEval installed successfully!")
```

---

## Core Concepts

### 1. Encoders

Encoders convert text into vector embeddings. SemEval supports multiple encoder types:

**Sentence Transformer Encoder:**
```python
from semeval import SentenceTransformerEncoder

encoder = SentenceTransformerEncoder(
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    device="auto"  # or "cuda", "mps", "cpu"
)
```

**HuggingFace Encoder:**
```python
from semeval.core.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    "dbmdz/bert-base-turkish-cased",
    device="cuda",
    max_length=512
)
```

### 2. Task Runner

The `TaskRunner` orchestrates evaluation across all tasks:

```python
from semeval import TaskRunner

runner = TaskRunner(
    encoder=encoder,
    verbose=True,
    settings=None  # Optional: pass SemEvalSettings
)
```

### 3. Evaluation Result

Results are returned in a structured format:

```python
result = runner.run("data/test_data.json")

# Access summary
summary = result.get_summary()

# Access task-specific results
for task_name, task_info in summary['tasks'].items():
    print(f"{task_name}: {task_info['status']}")
    print(f"Metrics: {task_info['metrics']}")
```

### 4. Data Loaders

Data loaders parse test data files:

```python
from semeval.core.loaders import JSONDataLoader

loader = JSONDataLoader()
test_data = loader.load("data/test_data.json")

# Inspect loaded data
print(f"Tasks: {len(test_data.tasks.__dict__)}")
```

---

## Configuration System

### Overview

SemEval uses a hierarchical configuration system with the following priority:

1. **Environment Variables** (highest priority)
2. **`.env` file**
3. **Environment-specific YAML** (`config.dev.yaml`, `config.prod.yaml`)
4. **Base YAML** (`config.yaml`)
5. **Default values** (lowest priority)

### Configuration Files

#### config.yaml (Base Configuration)

```yaml
model:
  name: "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
  device: "auto"  # auto, cuda, mps, cpu
  batch_size: 32

output:
  base_dir: "output"
  export_formats:
    - json
    - csv
    - markdown
  save_comprehensive_report: true

tasks:
  information_retrieval:
    enabled: true
    ndcg_at_k: [1, 3, 5, 10]
    map_at_k: [1, 3, 5, 10]
    mrr_at_k: [1, 3, 5, 10]
    precision_at_k: [1, 3, 5, 10]
    recall_at_k: [1, 3, 5, 10]

  semantic_similarity:
    enabled: true
    report_failed_triplets: 5

  linguistic_robustness:
    enabled: true
    similarity_threshold: 0.8
    morphology_enabled: true
    typo_enabled: true
    negation_enabled: true

  vector_arithmetic:
    enabled: true
    top_k: 1

logging:
  verbose: false
  level: "INFO"
  log_dir: "logs"
```

#### config.dev.yaml (Development)

```yaml
model:
  name: "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
  device: "auto"
  batch_size: 16

tasks:
  information_retrieval:
    enabled: true
    ndcg_at_k: [1, 5, 10]  # Fewer metrics for faster dev
    map_at_k: [1, 5, 10]
    mrr_at_k: [10]

logging:
  verbose: true  # More output during development
  level: "DEBUG"
```

#### config.prod.yaml (Production)

```yaml
model:
  name: "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
  device: "cuda"
  batch_size: 64

tasks:
  information_retrieval:
    enabled: true
    ndcg_at_k: [1, 3, 5, 10, 20]  # Extended metrics
    map_at_k: [1, 3, 5, 10, 20]
    mrr_at_k: [1, 3, 5, 10, 20]

logging:
  verbose: false
  level: "INFO"
```

### Loading Configuration

```python
from semeval import load_settings

# Load default config.yaml
settings = load_settings()

# Load dev configuration
settings = load_settings(env="dev")

# Load prod configuration
settings = load_settings(env="prod")

# Access configuration
print(f"Model: {settings.model.name}")
print(f"Device: {settings.model.device}")
print(f"Output directory: {settings.output.base_dir}")
```

### Environment Variable Overrides

Create a `.env` file or set environment variables:

```bash
# .env file
SEMEVAL_MODEL__NAME=Alibaba-NLP/gte-multilingual-base
SEMEVAL_MODEL__DEVICE=cuda
SEMEVAL_MODEL__BATCH_SIZE=64
SEMEVAL_OUTPUT__BASE_DIR=custom_output
SEMEVAL_LOGGING__VERBOSE=true
SEMEVAL_LOGGING__LEVEL=DEBUG
```

Or export directly:

```bash
export SEMEVAL_MODEL__NAME="Alibaba-NLP/gte-multilingual-base"
export SEMEVAL_LOGGING__VERBOSE="true"
```

Then load:

```python
from semeval import load_settings

# Automatically picks up environment variables
settings = load_settings()
```

### Using Configuration in Code

```python
from semeval import load_settings, TaskRunner, SentenceTransformerEncoder

# Load settings
settings = load_settings(env="dev")

# Create encoder using config
encoder = SentenceTransformerEncoder(
    settings.model.name,
    device=settings.model.device
)

# Create runner with settings
runner = TaskRunner(
    encoder=encoder,
    settings=settings
)

# Run evaluation
result = runner.run("data/test_data.json")
```

---

## Evaluation Tasks

### 1. Information Retrieval (IR)

#### Purpose

Evaluates how well the model retrieves relevant documents for given queries.

#### Metrics Explained

- **NDCG@k** (Normalized Discounted Cumulative Gain): Measures ranking quality with graded relevance. Higher is better.
- **MRR@k** (Mean Reciprocal Rank): Average of 1/rank of first relevant doc. Higher is better.
- **MAP@k** (Mean Average Precision): Mean of precision values at each relevant position. Higher is better.
- **Precision@k**: Fraction of top-k results that are relevant.
- **Recall@k**: Fraction of all relevant docs found in top-k.
- **Accuracy@k**: Whether at least one relevant doc appears in top-k.

#### Usage Example

```python
from semeval import TaskRunner, SentenceTransformerEncoder

encoder = SentenceTransformerEncoder("model-name")
runner = TaskRunner(encoder=encoder)

# Run IR task
result = runner.run_task("information_retrieval", "data/test_data.json")

# Access metrics
print(f"NDCG@10: {result.metrics['cosine-NDCG@10']:.4f}")
print(f"MRR@10: {result.metrics['cosine-MRR@10']:.4f}")
print(f"MAP@10: {result.metrics['cosine-MAP@10']:.4f}")
```

#### Interpreting Results

| NDCG@10 | Quality |
|---------|---------|
| 0.9-1.0 | Excellent |
| 0.7-0.9 | Good |
| 0.5-0.7 | Fair |
| 0.0-0.5 | Poor |

### 2. Semantic Similarity

#### Purpose

Tests the model's ability to distinguish between semantically similar and dissimilar text pairs using triplet evaluation.

#### How It Works

For each triplet (anchor, positive, negative):
- Anchor and positive should be similar (high cosine similarity)
- Anchor and negative should be dissimilar (low cosine similarity)
- Success = sim(anchor, positive) > sim(anchor, negative)

#### Metrics Explained

- **Triplet Accuracy**: Percentage of correctly ordered triplets
- **Average Margin**: Mean of (positive_sim - negative_sim). Higher = better separation.
- **Margin > 0.1/0.2**: Percentage with strong separation

#### Usage Example

```python
result = runner.run_task("semantic_similarity", "data/test_data.json")

print(f"Accuracy: {result.metrics['accuracy']:.2%}")
print(f"Avg Margin: {result.metrics['avg_margin']:.3f}")
print(f"Strong Margin (>0.2): {result.metrics['margin_gt_02']:.2%}")

# Check difficulty breakdown
difficulty = result.metrics.get('difficulty_breakdown', {})
for level, metrics in difficulty.items():
    print(f"{level}: {metrics['accuracy']:.2%}")
```

#### Interpreting Results

| Accuracy | Quality |
|----------|---------|
| 0.95-1.0 | Excellent |
| 0.85-0.95 | Good |
| 0.70-0.85 | Fair |
| 0.0-0.70 | Poor |

### 3. Linguistic Robustness

#### Purpose

Evaluates model stability under linguistic variations like typos, morphological changes, and negations.

#### Variation Types

- **Morphology**: Case, number, tense variations (e.g., "araba" → "arabalar")
- **Typo**: Spelling errors (e.g., "bilgisayar" → "bilgisyar")
- **Negation**: Adding/removing negation (e.g., "iyi" → "iyi değil")

#### Metrics Explained

- **Overall Robustness**: Average stability across all variation types
- **Morphology/Typo/Negation Robustness**: Specific stability scores
- **Similarity Threshold**: Variations with cosine_sim > threshold considered stable

#### Usage Example

```python
result = runner.run_task("linguistic_robustness", "data/test_data.json")

print(f"Overall Robustness: {result.metrics['overall_robustness']:.2%}")
print(f"Morphology: {result.metrics['morphology_robustness']:.2%}")
print(f"Typo: {result.metrics['typo_robustness']:.2%}")
print(f"Negation: {result.metrics['negation_robustness']:.2%}")
```

#### Interpreting Results

| Robustness | Quality |
|------------|---------|
| 0.90-1.0 | Excellent (very stable) |
| 0.75-0.90 | Good |
| 0.60-0.75 | Fair |
| 0.0-0.60 | Poor (unstable) |

### 4. Vector Arithmetic

#### Purpose

Tests compositional semantic understanding through analogy tasks.

#### How It Works

Given (a, b, c), find d such that: `a - b + c ≈ d`

Example: "Ankara" - "Türkiye" + "Fransa" ≈ "Paris"

#### Metrics Explained

- **Analogy Accuracy**: Percentage of correctly solved analogies
- **Avg Cosine Similarity**: Average similarity to expected results
- **Top-k Accuracy**: Whether expected result is in top-k candidates

#### Usage Example

```python
result = runner.run_task("vector_arithmetic", "data/test_data.json")

print(f"Accuracy: {result.metrics['accuracy']:.2%}")
print(f"Avg Cosine Similarity: {result.metrics['avg_cosine_similarity']:.3f}")

# Check category performance
for category, metrics in result.metrics.get('category_breakdown', {}).items():
    print(f"{category}: {metrics['accuracy']:.2%}")
```

#### Interpreting Results

| Accuracy | Quality |
|----------|---------|
| 0.80-1.0 | Excellent |
| 0.60-0.80 | Good |
| 0.40-0.60 | Fair |
| 0.0-0.40 | Poor |

---

## Data Format

### Overview

SemEval uses JSON files for test data with Pydantic validation. Each file contains metadata and task-specific data.

### File Structure

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Turkish Semantic Evaluation Suite",
    "language": "tr",
    "domain": "general",
    "created_date": "2025-11-07",
    "total_tasks": 4
  },
  "tasks": {
    "information_retrieval": { ... },
    "semantic_similarity": { ... },
    "linguistic_robustness": { ... },
    "vector_arithmetic": { ... }
  }
}
```

### Task 1: Information Retrieval

```json
{
  "information_retrieval": {
    "corpus": [
      {
        "id": "doc_0",
        "text": "TCMB politika faizini %30'a yükseltti.",
        "metadata": {
          "category": "monetary_policy",
          "source": "news"
        }
      }
    ],
    "queries": [
      {
        "id": "q_0",
        "text": "TCMB faiz kararı nedir?"
      }
    ],
    "relevance_judgments": [
      {
        "query_id": "q_0",
        "doc_id": "doc_0",
        "relevance_score": 2
      }
    ],
    "config": {
      "ndcg_at_k": [1, 3, 5, 10],
      "map_at_k": [1, 3, 5, 10],
      "mrr_at_k": [1, 3, 5, 10]
    }
  }
}
```

**Relevance Scores:**
- `0`: Not relevant
- `1`: Partially relevant
- `2`: Highly relevant

### Task 2: Semantic Similarity

```json
{
  "semantic_similarity": {
    "triplets": [
      {
        "id": "t1",
        "anchor": "Hisse senedi fiyatları yükseldi.",
        "positive": "Borsa değerleri arttı.",
        "negative": "Hava durumu güneşli.",
        "difficulty": "kolay",
        "category": "borsa",
        "subcategory": "finansal"
      }
    ],
    "config": {
      "report_failed_triplets": 5
    }
  }
}
```

**Difficulty Levels:**
- `kolay` (easy)
- `orta` (medium)
- `zor` (hard)

### Task 3: Linguistic Robustness

```json
{
  "linguistic_robustness": {
    "test_cases": [
      {
        "id": "lr1",
        "original": "Bilgisayar çok hızlı.",
        "variations": [
          {
            "text": "Bilgisayarlar çok hızlı.",
            "type": "morphology",
            "description": "Plural form"
          },
          {
            "text": "Bilgisyar çok hızlı.",
            "type": "typo",
            "description": "Typo in 'bilgisayar'"
          }
        ],
        "category": "teknoloji"
      }
    ],
    "config": {
      "similarity_threshold": 0.8
    }
  }
}
```

**Variation Types:**
- `morphology`: Morphological changes (case, number, tense)
- `typo`: Spelling errors
- `negation`: Negation addition/removal

### Task 4: Vector Arithmetic

```json
{
  "vector_arithmetic": {
    "analogies": [
      {
        "id": "va1",
        "a": "Ankara",
        "b": "Türkiye",
        "c": "Paris",
        "expected_d": "Fransa",
        "category": "coğrafya",
        "subcategory": "başkentler"
      }
    ],
    "config": {
      "top_k": 1
    }
  }
}
```

### Creating Custom Test Data

```python
from semeval.core.loaders import JSONDataLoader

# Validate your JSON
loader = JSONDataLoader()
try:
    data = loader.load("my_test_data.json")
    print("✅ Data is valid!")
    print(f"Tasks found: {len(data.tasks.__dict__)}")
except Exception as e:
    print(f"❌ Validation error: {e}")
```

---

## Running Evaluations

### Basic Evaluation

```python
from semeval import TaskRunner, SentenceTransformerEncoder

# Create encoder
encoder = SentenceTransformerEncoder(
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
)

# Create runner
runner = TaskRunner(encoder=encoder, verbose=True)

# Run all tasks
result = runner.run("data/test_data.json")

# Get summary
summary = result.get_summary()
print(f"Total runtime: {summary['total_runtime']:.2f}s")
```

### Run Specific Task

```python
# Run only one task
ir_result = runner.run_task("information_retrieval", "data/test_data.json")
ss_result = runner.run_task("semantic_similarity", "data/test_data.json")
lr_result = runner.run_task("linguistic_robustness", "data/test_data.json")
va_result = runner.run_task("vector_arithmetic", "data/test_data.json")
```

### With Configuration

```python
from semeval import load_settings, TaskRunner, SentenceTransformerEncoder

# Load settings
settings = load_settings(env="dev")

# Create encoder from config
encoder = SentenceTransformerEncoder(
    settings.model.name,
    device=settings.model.device
)

# Run with settings
runner = TaskRunner(encoder=encoder, settings=settings)
result = runner.run("data/test_data.json")
```

### Batch Evaluation

```python
from semeval import TaskRunner, SentenceTransformerEncoder

models = [
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "Alibaba-NLP/gte-multilingual-base"
]

results = []
for model_name in models:
    print(f"\n{'='*70}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*70}")

    encoder = SentenceTransformerEncoder(model_name)
    runner = TaskRunner(encoder=encoder, verbose=False)
    result = runner.run("data/test_data.json")

    summary = result.get_summary()
    results.append({
        'model': model_name,
        'ndcg@10': summary['tasks']['information_retrieval']['metrics'].get('cosine-NDCG@10', 0),
        'triplet_acc': summary['tasks']['semantic_similarity']['metrics'].get('accuracy', 0),
        'robustness': summary['tasks']['linguistic_robustness']['metrics'].get('overall_robustness', 0),
        'analogy_acc': summary['tasks']['vector_arithmetic']['metrics'].get('accuracy', 0),
        'runtime': summary['total_runtime']
    })

# Create comparison table
import pandas as pd
df = pd.DataFrame(results)
print("\n" + "="*70)
print("MODEL COMPARISON")
print("="*70)
print(df.to_string(index=False))
```

---

## Export & Reporting

### Available Export Formats

SemEval supports multiple export formats:

1. **CSV**: All metrics in tabular format
2. **JSON**: Complete structured results
3. **Markdown**: Human-readable tables
4. **Per-task files**: Individual JSON/MD files for each task

### Basic Export

```python
from semeval import TaskRunner, SentenceTransformerEncoder
from semeval.postprocess import ResultsExporter

# Run evaluation
encoder = SentenceTransformerEncoder("model-name")
runner = TaskRunner(encoder=encoder)
result = runner.run("data/test_data.json")

# Create exporter
exporter = ResultsExporter()

# Export to CSV
df = exporter.export_csv(result, "output/results.csv")
print(f"CSV saved: {df.shape} rows")

# Export to JSON
exporter.export_json(result, "output/results.json")

# Export to Markdown
exporter.export_markdown(result, "output/results.md")
```

### Per-Task Export

```python
from semeval.postprocess import ResultsExporter

exporter = ResultsExporter()

# Export each task to separate files
task_paths = exporter.export_per_task(
    result,
    "output",
    export_formats=['json', 'markdown']
)

# Check what was created
for task_name, paths in task_paths.items():
    print(f"\n{task_name}:")
    for format_type, file_path in paths.items():
        print(f"  - {format_type}: {file_path}")
```

**Generated files:**
```
output/
├── information_retrieval_result.json
├── information_retrieval_result.md
├── semantic_similarity_result.json
├── semantic_similarity_result.md
├── linguistic_robustness_result.json
├── linguistic_robustness_result.md
├── vector_arithmetic_result.json
└── vector_arithmetic_result.md
```

### Comprehensive Report

```python
from semeval.postprocess import ReportGenerator

generator = ReportGenerator()

# Generate detailed markdown report
generator.generate_report(
    result,
    "output/comprehensive_report.md",
    model_name="BERT Turkish (emrecan/bert-base-turkish-cased-mean-nli-stsb-tr)",
    include_recommendations=True
)
```

The comprehensive report includes:
- Executive summary with overall scores
- Detailed metrics for each task
- Performance breakdowns (by difficulty, category, etc.)
- Recommendations for improvement
- Failed examples for debugging

### Custom Export Pipeline

```python
from semeval import TaskRunner, SentenceTransformerEncoder
from semeval.postprocess import ResultsExporter, ReportGenerator
from pathlib import Path

# Run evaluation
encoder = SentenceTransformerEncoder("model-name")
runner = TaskRunner(encoder=encoder)
result = runner.run("data/test_data.json")

# Setup output directory
output_dir = Path("output/my_evaluation")
output_dir.mkdir(parents=True, exist_ok=True)

# Export all formats
exporter = ResultsExporter()
exporter.export_csv(result, output_dir / "results.csv")
exporter.export_json(result, output_dir / "results.json")
exporter.export_markdown(result, output_dir / "results.md")

# Per-task exports
task_paths = exporter.export_per_task(result, str(output_dir))

# Comprehensive report
generator = ReportGenerator()
generator.generate_report(
    result,
    output_dir / "comprehensive_report.md",
    model_name="My Model",
    include_recommendations=True
)

print(f"\n✅ All exports saved to: {output_dir}")
```

---

## Advanced Topics

### Custom Encoder

Create a custom encoder by extending `BaseEncoder`:

```python
from semeval.core.base_encoder import BaseEncoder
import numpy as np

class MyCustomEncoder(BaseEncoder):
    def __init__(self, model_name: str):
        self._model_name = model_name
        # Initialize your model here
        self.model = load_your_model(model_name)

    def encode(self, texts, convert_to_tensor=False, **kwargs):
        """Encode texts to embeddings."""
        embeddings = self.model.encode(texts)

        if convert_to_tensor:
            import torch
            embeddings = torch.tensor(embeddings)

        return embeddings

    def get_embedding_dim(self) -> int:
        """Return embedding dimension."""
        return 768

    @property
    def model_name(self) -> str:
        """Return model name."""
        return self._model_name

# Use it
encoder = MyCustomEncoder("my-model")
runner = TaskRunner(encoder=encoder)
```

### Custom Data Loader

Create a custom data loader:

```python
from semeval.core.base_loader import BaseDataLoader
from semeval.core.schemas import TestDataModel

class MyDataLoader(BaseDataLoader):
    def load(self, source: str) -> TestDataModel:
        """Load data from your custom source."""
        # Your loading logic
        data = load_from_database(source)

        # Convert to TestDataModel
        test_data = TestDataModel(**data)

        return test_data

# Use it
loader = MyDataLoader()
runner = TaskRunner(encoder=encoder, data_loader=loader)
result = runner.run("my_data_source")
```

### Direct Task Usage

Run tasks directly without TaskRunner:

```python
from semeval.tasks import InformationRetrieval
from semeval.core.loaders import JSONDataLoader

# Load data
loader = JSONDataLoader()
test_data = loader.load("data/test_data.json")

# Create and run task
task = InformationRetrieval(
    encoder=encoder,
    task_data=test_data.tasks.information_retrieval,
    device="cuda",
    verbose=True
)
result = task.run()

# Access results
print(f"Task: {result.task_name}")
print(f"Status: {result.status}")
print(f"Runtime: {result.runtime_seconds:.2f}s")
print(f"Metrics: {result.metrics}")
```

### Programmatic Configuration

Create settings programmatically:

```python
from semeval.core.config import (
    SemEvalSettings,
    ModelConfig,
    OutputConfig,
    LoggingConfig
)

# Create custom settings
settings = SemEvalSettings(
    model=ModelConfig(
        name="Alibaba-NLP/gte-multilingual-base",
        device="cuda",
        batch_size=64
    ),
    output=OutputConfig(
        base_dir="custom_output",
        export_formats=["json", "markdown"],
        save_comprehensive_report=True
    ),
    logging=LoggingConfig(
        verbose=True,
        level="DEBUG"
    )
)

# Use custom settings
encoder = SentenceTransformerEncoder(settings.model.name)
runner = TaskRunner(encoder=encoder, settings=settings)
```

---

## API Reference

### TaskRunner

```python
class TaskRunner:
    def __init__(
        self,
        encoder: BaseEncoder,
        data_loader: Optional[BaseDataLoader] = None,
        device: Optional[str] = None,
        verbose: bool = False,
        settings: Optional[SemEvalSettings] = None
    ):
        """Initialize task runner."""

    def run(self, data_source: Union[str, TestDataModel]) -> EvaluationResult:
        """Run all enabled tasks."""

    def run_task(
        self,
        task_name: str,
        data_source: Union[str, TestDataModel]
    ) -> TaskResult:
        """Run a specific task."""
```

### ResultsExporter

```python
class ResultsExporter:
    def export_csv(
        self,
        result: EvaluationResult,
        output_path: str
    ) -> pd.DataFrame:
        """Export results to CSV."""

    def export_json(
        self,
        result: EvaluationResult,
        output_path: str
    ) -> None:
        """Export results to JSON."""

    def export_markdown(
        self,
        result: EvaluationResult,
        output_path: str
    ) -> None:
        """Export results to Markdown."""

    def export_per_task(
        self,
        result: EvaluationResult,
        output_dir: str,
        export_formats: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, str]]:
        """Export each task to separate files."""
```

### ReportGenerator

```python
class ReportGenerator:
    def generate_report(
        self,
        result: EvaluationResult,
        output_path: str,
        model_name: Optional[str] = None,
        include_recommendations: bool = True
    ) -> None:
        """Generate comprehensive markdown report."""
```

### Configuration

```python
def load_settings(env: Optional[str] = None) -> SemEvalSettings:
    """Load settings from YAML and environment variables.

    Args:
        env: Environment name ('dev', 'prod'). If None, loads config.yaml

    Returns:
        SemEvalSettings instance
    """
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

**Error:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```python
# Reduce batch size in config
settings.model.batch_size = 16  # or 8

# Or use CPU
settings.model.device = "cpu"
```

#### 2. Model Not Found

**Error:**
```
OSError: Model 'model-name' not found
```

**Solution:**
```python
# Check model name on HuggingFace
# Or download manually first
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("model-name")  # Downloads first
```

#### 3. Invalid Data Format

**Error:**
```
ValidationError: X validation errors for TestDataModel
```

**Solution:**
```python
# Validate your JSON structure
from semeval.core.loaders import JSONDataLoader

loader = JSONDataLoader()
try:
    data = loader.load("data.json")
except Exception as e:
    print(f"Validation error: {e}")
    # Fix your JSON based on error message
```

#### 4. Missing Dependencies

**Error:**
```
ModuleNotFoundError: No module named 'tabulate'
```

**Solution:**
```bash
# Install missing dependencies
uv sync
# or
pip install -e .
```

#### 5. Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'output/results.csv'
```

**Solution:**
```python
# Make sure output directory exists and is writable
from pathlib import Path
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
```

### Getting Help

If you encounter issues not covered here:

1. Check the [Architecture.md](Architecture.md) for technical details
2. Review example scripts in `scripts/` directory
3. Open an issue on GitHub with:
   - Error message
   - Minimal code to reproduce
   - Your Python version and dependencies
   - Your configuration (if relevant)

---

## Best Practices

### 1. Configuration Management

- Use `config.dev.yaml` for development (verbose, quick metrics)
- Use `config.prod.yaml` for production (optimized, full metrics)
- Store sensitive info (API keys) in `.env`, not YAML
- Use environment variables for deployment-specific overrides

### 2. Data Organization

- Keep test data versioned and documented
- Use meaningful IDs for documents, queries, triplets, etc.
- Add metadata to track data provenance
- Validate data before running large evaluations

### 3. Performance Optimization

- Use GPU (`device="cuda"`) for large datasets
- Increase batch size if you have memory
- Use smaller metric sets (`ndcg_at_k: [10]`) during development
- Cache model downloads to avoid re-downloading

### 4. Result Management

- Use per-task export for detailed analysis
- Generate comprehensive reports for stakeholders
- Keep JSON exports for programmatic analysis
- Use consistent output directory structure

### 5. Evaluation Workflow

```python
# 1. Start with dev config for quick iteration
settings = load_settings(env="dev")

# 2. Test with small data first
result = runner.run("data/small_test.json")

# 3. Validate results look reasonable
summary = result.get_summary()
for task, info in summary['tasks'].items():
    assert info['status'] == 'success'

# 4. Run full evaluation with prod config
settings = load_settings(env="prod")
result = runner.run("data/full_test.json")

# 5. Export all results
exporter.export_per_task(result, "output")
generator.generate_report(result, "output/report.md")
```

---

**For more information, see:**
- [README.md](README.md) - Quick start and overview
- [Architecture.md](Architecture.md) - Technical architecture
- Example scripts in `scripts/` directory

**Made with ❤️ for Turkish NLP**
