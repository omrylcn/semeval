# SemEval - Semantic Evaluation Package

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-orange.svg)](semeval/__init__.py)

> A modular toolkit for evaluating semantic embeddings and NLP models.

**Note:** This toolkit is language-agnostic. Examples use Turkish data, but any language is supported.

## 🎯 Overview

**SemEval** is a efficient, extensible evaluation toolkit designed to assess the quality of semantic embeddings and NLP models. Built with modularity and ease of use in mind, it provides:

- 🔌 **Pluggable Architecture**: Easily swap encoders, data loaders, and evaluation tasks
- ✅ **Type-Safe**: Full Pydantic V2 validation for all data structures
- 📊 **Comprehensive Metrics**: NDCG, MRR, MAP, Triplet Accuracy, Robustness Scores, and more
- 🚀 **Performance Optimized**: Automatic GPU/CPU detection and batch processing
- ⚙️ **Flexible Configuration**: YAML-based config with environment variable overrides
- 📝 **Rich Export Options**: JSON, CSV, Markdown, and per-task reports
- 🧪 **4 Evaluation Tasks**: Information Retrieval, Semantic Similarity, Linguistic Robustness, Vector Arithmetic

### Key Features

| Feature | Description |
|---------|-------------|
| **Multiple Encoders** | Support for Sentence Transformers, HuggingFace, and custom encoders |
| **Flexible Data Loading** | JSON-based test data with automatic validation |
| **4 Evaluation Tasks** | IR, Semantic Similarity, Linguistic Robustness, Vector Arithmetic |
| **YAML Configuration** | Environment-based configs (dev, prod) with type-safe validation |
| **Extensible Tasks** | Easy to add new evaluation tasks |
| **Batch Processing** | Efficient processing of large datasets |
| **Device Agnostic** | Automatic CUDA/MPS/CPU detection |
| **Rich Reporting** | Export to JSON, CSV, Markdown with per-task files |

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd semeval

# Install dependencies (using uv)
uv sync

# Or using pip
pip install -e .
```

### Basic Usage

```python
from semeval import TaskRunner, SentenceTransformerEncoder

# 1. Create an encoder
encoder = SentenceTransformerEncoder(
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr"
)

# 2. Create a runner
runner = TaskRunner(encoder=encoder, verbose=True)

# 3. Run all 4 evaluation tasks
result = runner.run("data/test_data.json")

# 4. Get results
summary = result.get_summary()
print(f"Total runtime: {summary['total_runtime']:.2f}s")

# 5. Access task-specific metrics
for task_name, task_info in summary['tasks'].items():
    print(f"\n{task_name}: {task_info['status']}")
```

### Using Configuration Files

```python
from semeval import TaskRunner, SentenceTransformerEncoder, load_settings

# Load settings from config.yaml (or config.dev.yaml, config.prod.yaml)
settings = load_settings()

# Create encoder using config
encoder = SentenceTransformerEncoder(
    settings.model.name,
    device=settings.model.device
)

# Run with settings
runner = TaskRunner(encoder=encoder, settings=settings)
result = runner.run("data/test_data.json")
```

**Output:**
```
[INFO] Starting Evaluation
[INFO] Loading test data from: data/test_data.json
[INFO] Model: emrecan/bert-base-turkish-cased-mean-nli-stsb-tr
[INFO] Running Information Retrieval Task
[INFO] Running Semantic Similarity Task
[INFO] Running Linguistic Robustness Task
[INFO] Running Vector Arithmetic Task
✅ Evaluation complete: 2.75s
```

---

## 📚 Table of Contents

1. [Installation](#installation)
2. [Usage Examples](#usage-examples)
3. [Evaluation Tasks](#evaluation-tasks)
4. [Configuration](#configuration)
5. [Supported Encoders](#supported-encoders)
6. [Data Format](#data-format)
7. [Export & Reporting](#export--reporting)
8. [Metrics Reference](#metrics-reference)
9. [Advanced Usage](#advanced-usage)
10. [Contributing](#contributing)

---

## 📦 Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9+
- sentence-transformers
- transformers
- pydantic>=2.0
- pydantic-settings
- pyyaml
- tabulate

### Standard Installation

```bash
# Using uv (recommended)
uv sync

# Using pip
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

---

## 💡 Usage Examples

### Example 1: Run All Tasks with Config

```python
from semeval import TaskRunner, SentenceTransformerEncoder, load_settings

# Load config
settings = load_settings(env="dev")  # or "prod"

# Create encoder
encoder = SentenceTransformerEncoder(
    settings.model.name,
    device=settings.model.device
)

# Run evaluation
runner = TaskRunner(encoder=encoder, settings=settings)
result = runner.run("data/test_data.json")

# Export results
from semeval.postprocess import ResultsExporter, ReportGenerator

exporter = ResultsExporter()
output_dir = settings.output.base_dir

# Export all formats
exporter.export_csv(result, f"{output_dir}/results.csv")
exporter.export_json(result, f"{output_dir}/results.json")
exporter.export_markdown(result, f"{output_dir}/results.md")

# Export per-task files
task_paths = exporter.export_per_task(result, output_dir)
```

### Example 2: Run Specific Task

```python
from semeval import TaskRunner, SentenceTransformerEncoder

encoder = SentenceTransformerEncoder("model-name")
runner = TaskRunner(encoder=encoder)

# Run only Semantic Similarity task
result = runner.run_task("semantic_similarity", "data/test_data.json")

print(f"Triplet Accuracy: {result.metrics['accuracy']:.2%}")
print(f"Average Margin: {result.metrics['avg_margin']:.3f}")
```

### Example 3: Environment Variable Override

```bash
# Set environment variables
export SEMEVAL_MODEL__NAME="Alibaba-NLP/gte-multilingual-base"
export SEMEVAL_MODEL__DEVICE="cuda"
export SEMEVAL_LOGGING__VERBOSE="true"
```

```python
from semeval import load_settings, TaskRunner, SentenceTransformerEncoder

# Settings automatically load from env vars
settings = load_settings()

encoder = SentenceTransformerEncoder(
    settings.model.name,  # Uses env var
    device=settings.model.device
)

runner = TaskRunner(encoder=encoder, settings=settings)
result = runner.run("data/test_data.json")
```

### Example 4: Generate Comprehensive Report

```python
from semeval import TaskRunner, SentenceTransformerEncoder
from semeval.postprocess import ReportGenerator

# Run evaluation
encoder = SentenceTransformerEncoder("model-name")
runner = TaskRunner(encoder=encoder)
result = runner.run("data/test_data.json")

# Generate comprehensive markdown report
generator = ReportGenerator()
generator.generate_report(
    result,
    "output/comprehensive_report.md",
    model_name="BERT Turkish",
    include_recommendations=True
)
```

### Example 5: Batch Evaluate Multiple Models

```python
from semeval import TaskRunner, SentenceTransformerEncoder
import pandas as pd

models = [
    "emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    "Alibaba-NLP/gte-multilingual-base"
]

results = []
for model_name in models:
    encoder = SentenceTransformerEncoder(model_name)
    runner = TaskRunner(encoder=encoder, verbose=False)
    result = runner.run("data/test_data.json")

    summary = result.get_summary()
    results.append({
        'model': model_name,
        'ndcg@10': summary['tasks']['information_retrieval']['metrics'].get('cosine-NDCG@10', 0),
        'triplet_acc': summary['tasks']['semantic_similarity']['metrics'].get('accuracy', 0),
        'runtime': summary['total_runtime']
    })

df = pd.DataFrame(results)
print(df)
```

---

## 🎯 Evaluation Tasks

SemEval includes 4 comprehensive evaluation tasks:

### 1. Information Retrieval

Evaluates the model's ability to retrieve relevant documents for queries.

**Metrics:**
- NDCG@k (Normalized Discounted Cumulative Gain)
- MRR@k (Mean Reciprocal Rank)
- MAP@k (Mean Average Precision)
- Precision@k, Recall@k, Accuracy@k

**Usage:**
```python
result = runner.run_task("information_retrieval", "data/test_data.json")
print(f"NDCG@10: {result.metrics['cosine-NDCG@10']:.4f}")
```

**Data Requirements:**
- Corpus of documents
- Query set
- Relevance judgments (query-doc pairs with scores 0-2)

### 2. Semantic Similarity

Tests the model's ability to distinguish between semantically similar and dissimilar text pairs using triplet evaluation.

**Metrics:**
- Triplet Accuracy
- Average Margin (positive_sim - negative_sim)
- Margin Distribution (> 0.1, > 0.2)
- Performance by difficulty level
- Performance by subcategory

**Usage:**
```python
result = runner.run_task("semantic_similarity", "data/test_data.json")
print(f"Accuracy: {result.metrics['accuracy']:.2%}")
print(f"Avg Margin: {result.metrics['avg_margin']:.3f}")
```

**Data Requirements:**
- Triplets: anchor, positive, negative texts
- Optional: difficulty labels, categories

### 3. Linguistic Robustness

Evaluates model stability under linguistic variations (typos, morphological changes, negations).

**Metrics:**
- Overall robustness score
- Morphology robustness (case, number, tense variations)
- Typo robustness (spelling errors)
- Negation robustness (handling of negation)
- Embedding stability metrics

**Usage:**
```python
result = runner.run_task("linguistic_robustness", "data/test_data.json")
print(f"Overall Robustness: {result.metrics['overall_robustness']:.2%}")
```

**Data Requirements:**
- Original texts with linguistic variations
- Variation types (morphology, typo, negation)

### 4. Vector Arithmetic

Tests compositional semantic understanding through analogy and vector operations.

**Metrics:**
- Analogy accuracy
- Category-specific performance
- Subcategory breakdown
- Average cosine similarity to expected results

**Usage:**
```python
result = runner.run_task("vector_arithmetic", "data/test_data.json")
print(f"Analogy Accuracy: {result.metrics['accuracy']:.2%}")
```

**Data Requirements:**
- Analogy pairs: (a, b, c, expected_d)
- Categories and subcategories

---

## ⚙️ Configuration

SemEval uses a powerful YAML-based configuration system with environment variable overrides.

### Configuration Files

- **`config.yaml`**: Base configuration
- **`config.dev.yaml`**: Development settings (verbose, quick metrics)
- **`config.prod.yaml`**: Production settings (optimized, extended metrics)

### Configuration Structure

```yaml
# config.yaml
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

  semantic_similarity:
    enabled: true
    report_failed_triplets: 5

  linguistic_robustness:
    enabled: true
    similarity_threshold: 0.8

  vector_arithmetic:
    enabled: true
    top_k: 1

logging:
  verbose: false
  level: "INFO"
```

### Environment Variable Overrides

Settings can be overridden using environment variables with the prefix `SEMEVAL_`:

```bash
# Model settings
export SEMEVAL_MODEL__NAME="Alibaba-NLP/gte-multilingual-base"
export SEMEVAL_MODEL__DEVICE="cuda"
export SEMEVAL_MODEL__BATCH_SIZE="64"

# Output settings
export SEMEVAL_OUTPUT__BASE_DIR="custom_output"

# Logging
export SEMEVAL_LOGGING__VERBOSE="true"
export SEMEVAL_LOGGING__LEVEL="DEBUG"
```

### Loading Settings

```python
from semeval import load_settings

# Load default config.yaml
settings = load_settings()

# Load environment-specific config
settings = load_settings(env="dev")    # loads config.dev.yaml
settings = load_settings(env="prod")   # loads config.prod.yaml

# Access settings
print(f"Model: {settings.model.name}")
print(f"Device: {settings.model.device}")
print(f"Output: {settings.output.base_dir}")
```

### Priority Order

Settings are loaded with the following priority (highest to lowest):

1. Environment variables (`SEMEVAL_*`)
2. `.env` file
3. Environment-specific YAML (`config.{env}.yaml`)
4. Base YAML (`config.yaml`)
5. Default values in code

---

## 🤖 Supported Encoders

### 1. Sentence Transformer Encoder

```python
from semeval.core.encoders import SentenceTransformerEncoder

encoder = SentenceTransformerEncoder(
    model_name="emrecan/bert-base-turkish-cased-mean-nli-stsb-tr",
    device="auto"  # auto, cuda, mps, cpu
)
```

**Example Models:**
- `emrecan/bert-base-turkish-cased-mean-nli-stsb-tr` (Turkish)
- `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Multilingual)
- `Alibaba-NLP/gte-multilingual-base` (Multilingual)
- `sentence-transformers/all-MiniLM-L6-v2` (English)

### 2. HuggingFace Encoder

```python
from semeval.core.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    model_name="dbmdz/bert-base-turkish-cased",
    device="cuda",
    max_length=512
)
```

### 3. Custom Encoder

```python
from semeval.core.base_encoder import BaseEncoder

class MyEncoder(BaseEncoder):
    def encode(self, texts, **kwargs):
        # Your encoding logic
        return embeddings

    def get_embedding_dim(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return "my-model"
```

---

## 📋 Data Format

### JSON Structure

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Semantic Evaluation Suite",
    "language": "tr",
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

See [USAGE.md](USAGE.md) for detailed data format specifications.

---

## 📊 Export & Reporting

### Available Export Formats

```python
from semeval.postprocess import ResultsExporter, ReportGenerator

exporter = ResultsExporter()
generator = ReportGenerator()

# Export to different formats
df = exporter.export_csv(result, "output/results.csv")
exporter.export_json(result, "output/results.json")
exporter.export_markdown(result, "output/results.md")

# Generate comprehensive report
generator.generate_report(
    result,
    "output/comprehensive_report.md",
    model_name="BERT Turkish",
    include_recommendations=True
)
```

### Per-Task Export

Export individual files for each task:

```python
# Export each task to separate JSON and Markdown files
task_paths = exporter.export_per_task(
    result,
    "output",
    export_formats=['json', 'markdown']
)

# Generated files:
# - information_retrieval_result.json
# - information_retrieval_result.md
# - semantic_similarity_result.json
# - semantic_similarity_result.md
# - linguistic_robustness_result.json
# - linguistic_robustness_result.md
# - vector_arithmetic_result.json
# - vector_arithmetic_result.md
```

### Output Directory Structure

```
output/
├── results.csv                          # All metrics in CSV
├── results.json                         # Complete results in JSON
├── results.md                           # Summary markdown
├── comprehensive_report.md              # Detailed report with recommendations
├── information_retrieval_result.json    # Per-task exports
├── information_retrieval_result.md
├── semantic_similarity_result.json
├── semantic_similarity_result.md
├── linguistic_robustness_result.json
├── linguistic_robustness_result.md
├── vector_arithmetic_result.json
└── vector_arithmetic_result.md
```

---

## 📈 Metrics Reference

### Information Retrieval

| Metric | Range | Interpretation |
|--------|-------|----------------|
| NDCG@k | [0, 1] | Ranking quality with graded relevance |
| MRR@k | [0, 1] | Reciprocal rank of first relevant doc |
| MAP@k | [0, 1] | Mean average precision |

### Semantic Similarity

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Triplet Accuracy | [0, 1] | Fraction of correctly ordered triplets |
| Average Margin | [-1, 1] | Mean difference (pos_sim - neg_sim) |

### Linguistic Robustness

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Overall Robustness | [0, 1] | Average stability across variations |
| Morphology Robustness | [0, 1] | Stability under morphological changes |
| Typo Robustness | [0, 1] | Stability under typos |

### Vector Arithmetic

| Metric | Range | Interpretation |
|--------|-------|----------------|
| Analogy Accuracy | [0, 1] | Fraction of correct analogies |
| Avg Cosine Similarity | [-1, 1] | Average similarity to expected |

---

## 🔧 Advanced Usage

### Custom Evaluation Pipeline

```python
from semeval import TaskRunner
from semeval.core.encoders import SentenceTransformerEncoder
from semeval.core.loaders import JSONDataLoader

# Load and inspect data
loader = JSONDataLoader()
data = loader.load("data/test_data.json")

# Create encoder
encoder = SentenceTransformerEncoder("model-name")

# Run specific tasks
runner = TaskRunner(encoder=encoder)
ir_result = runner.run_task("information_retrieval", data)
ss_result = runner.run_task("semantic_similarity", data)
```

### Testing Scripts

```bash
# Test configuration
uv run python scripts/test_config.py

# Test with config
uv run python scripts/test_with_config.py

# Test per-task export
uv run python scripts/test_per_task_export.py

# Test all post-processing
uv run python scripts/test_postprocessing_all.py
```

---
## 📈 Changelog

### Version 0.1.0 (Current)

**Features:**
- ✅ Information Retrieval task with comprehensive IR metrics
- ✅ Semantic Similarity task with triplet evaluation
- ✅ Linguistic Robustness task for variation testing
- ✅ Vector Arithmetic task for analogy evaluation
- ✅ YAML-based configuration with env var overrides
- ✅ Rich export options (CSV, JSON, Markdown)
- ✅ Per-task file exports
- ✅ Comprehensive report generation
- ✅ Type-safe pydantic-settings integration
- ✅ Auto device detection (cuda/mps/cpu)

**Planned:**
- 🚧 CLI interface (v0.2.0)
- 🚧 Web UI (v2.0.0)
- 🚧 Additional encoder support
- 🚧 More evaluation tasks

---

**Made with ❤️ for Semantic Embeddings**
