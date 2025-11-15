# SemEval

A modular toolkit for evaluating semantic embeddings and NLP models.

---

## 📌 Project Status

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![Python](https://img.shields.io/badge/python-3.8+-green)
![License](https://img.shields.io/badge/license-MIT-blue)
![Development](https://img.shields.io/badge/status-beta-yellow)

**Current Version**: v0.1.0  
**Next Release**: v0.1.1 (Stabilization)  
**Stability**: Beta - Active Development

> ⚠️ **v0.1.1 Focus**: Testing infrastructure, CLI interface, error handling, and CI/CD pipeline for production readiness.

📋 [View Full Roadmap →](ROADMAP.md)

---

## 🎯 Why SemEval?

### When You Need Flexibility

Existing benchmarks are excellent for standardized comparisons, but sometimes you need:

- ✅ **Quick prototyping** with your own data format
- ✅ **Domain-specific testing** without extensive setup
- ✅ **Small-scale validation** (100s of samples vs 1000s)
- ✅ **Custom evaluation tasks** tailored to your use case
- ✅ **Embedding space insights** beyond task performance

### What SemEval Offers

- 📦 **Simple JSON format** → Drop your data and go
- 🔬 **Modular design** → Use only what you need
- 🌐 **Language-agnostic** → Any language, any model
- 🔌 **Extensible** → Add your own tasks and metrics
- 🚀 **Fast iteration** → Minutes from data to insights

### Great For

- Rapid prototyping and experimentation
- Domain-specific embedding evaluation
- Testing models on proprietary data
- RAG system optimization
- Research on new embedding architectures
- Educational projects and learning

> 💡 SemEval complements existing benchmarks by focusing on flexibility and ease of use for custom evaluation scenarios.

---

## ✨ Core Features

### 🧪 Evaluation Tasks

**4 Core Tasks** (13+ planned - [see roadmap](ROADMAP.md))

- **Information Retrieval** - NDCG, MRR, MAP metrics
- **Semantic Similarity** - Triplet evaluation with margin analysis
- **Linguistic Robustness** - Test stability under variations (typos, morphology, negation)
- **Vector Arithmetic** - Analogy and compositional semantics

### 🔬 Advanced Capabilities

- **Embedding Quality Metrics** *(coming in v0.2.0)*
  - Isotropy & Uniformity Analysis
  - Representation Health Checks
  - Label-free Quality Assessment

### 🛠️ Production-Ready Features *(v0.1.1+)*

- **✅ CLI Interface** - Fast, user-friendly command-line tools (`eval`, `validate`, `init`, `version`, `info`)
- **✅ Instant Startup** - ~0.2s CLI performance with lazy loading optimization
- **✅ Comprehensive Testing** - 121 tests with >90% coverage
- **✅ Type Safety** - Full Pydantic V2 validation
- Robust error handling and recovery
- CI/CD integration ready

### 🌐 Flexible Architecture

- **Multiple Encoders**: Sentence Transformers, HuggingFace, custom encoders
- **Type-Safe**: Full Pydantic V2 validation
- **Performance Optimized**: Automatic GPU/CPU detection, batch processing
- **Rich Exports**: JSON, CSV, Markdown reports

---

## 📚 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
  - [CLI Usage](#️-cli-usage-recommended)
  - [Python API](#-python-api)
- [CLI Commands](#-cli-commands)
- [Usage Examples](#usage-examples)
- [Evaluation Tasks](#evaluation-tasks)
- [Configuration](#configuration)
- [Supported Encoders](#supported-encoders)
- [Data Format](#data-format)
- [Export & Reporting](#export--reporting)

---

## 🚀 Installation

### Requirements

- Python 3.8 or higher
- PyTorch 1.9+
- sentence-transformers
- transformers
- pydantic>=2.0
- pydantic-settings
- pyyaml

### Install

```bash
# Using uv (recommended - fast and modern)
uv pip install -e .

# Using pip
pip install -e .

# With development dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Check if CLI is working
semeval version

# Get system info
semeval info

# Create a test template
semeval init -o test.json
```

---

## ⚡ Quick Start

### 🖥️ CLI Usage (Recommended)

The fastest way to get started with SemEval:

```bash
# Create a template test file
semeval init --template basic -o my_test.json

# Validate your test data
semeval validate my_test.json

# Run evaluation
semeval eval --model "sentence-transformers/all-MiniLM-L6-v2" --data my_test.json

# Check version and system info
semeval version
semeval info
```

**Available Templates:**

- `basic` - Semantic similarity examples
- `ir` - Information retrieval examples
- `similarity` - Comprehensive similarity tests
- `robustness` - Linguistic robustness tests

**CLI Performance:** Instant startup (~0.2s) with lazy loading of ML dependencies.

### 🐍 Python API

```python
from semeval import TaskRunner, SentenceTransformerEncoder

# 1. Create an encoder with any sentence-transformers model
encoder = SentenceTransformerEncoder(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Create a runner
runner = TaskRunner(encoder=encoder, verbose=True)

# 3. Run all evaluation tasks
result = runner.run("data/test_data.json")

# 4. Get results
summary = result.get_summary()
print(f"Total runtime: {summary['total_runtime']:.2f}s")

# 5. Access task-specific metrics
for task_name, task_info in summary['tasks'].items():
    print(f"\n{task_name}: {task_info['status']}")
```

### Model Examples

```python
# English
encoder = SentenceTransformerEncoder("sentence-transformers/all-MiniLM-L6-v2")

# Multilingual
encoder = SentenceTransformerEncoder("Alibaba-NLP/gte-multilingual-base")

# Domain-specific (Turkish example)
encoder = SentenceTransformerEncoder("emrecan/bert-base-turkish-cased-mean-nli-stsb-tr")

# Your custom model
encoder = SentenceTransformerEncoder("your-organization/your-model")
```

---

## 🖥️ CLI Commands

SemEval provides a powerful command-line interface for quick evaluations and automation.

### `semeval init` - Create Test Data Templates

Generate template test data files with proper schema:

```bash
# Create basic semantic similarity template
semeval init

# Create information retrieval template
semeval init --template ir -o ir_test.json

# Create robustness testing template
semeval init --template robustness -o robustness_test.json

# Overwrite existing file
semeval init --template similarity -o test.json --force
```

**Available Templates:**

- `basic` - Semantic similarity with 2 triplet examples
- `ir` - Information retrieval with sample corpus and queries
- `similarity` - Comprehensive semantic similarity tests
- `robustness` - Linguistic robustness (morphology, typos, negation)

### `semeval validate` - Validate Test Data

Check your test data for errors before running evaluation:

```bash
# Basic validation
semeval validate test_data.json

# Strict mode (fail on warnings)
semeval validate test_data.json --strict

# Generate HTML validation report (planned)
semeval validate test_data.json --report validation.html
```

**Validation Features:**

- ✅ Schema validation with detailed error messages
- ✅ Data statistics (task counts, sample sizes)
- ✅ Quality warnings (small dataset warnings)
- ✅ Metadata verification

### `semeval eval` - Run Evaluation

Evaluate models with your test data:

```bash
# Basic evaluation
semeval eval --model "sentence-transformers/all-MiniLM-L6-v2" --data test.json

# Run specific tasks only
semeval eval -m "model-name" -d test.json --tasks "ir,similarity"

# Use different encoder type
semeval eval -m "bert-base-uncased" -d test.json --encoder huggingface

# Specify device and output directory
semeval eval -m "model" -d test.json --device cuda --output results/

# Verbose mode
semeval eval -m "model" -d test.json --verbose

# Use config file (planned)
semeval eval --config config.yaml
```

**Options:**

- `--model, -m` - Model name or path (HuggingFace/SentenceTransformers)
- `--data, -d` - Path to test data JSON file
- `--output, -o` - Output directory for results (default: `results/`)
- `--tasks, -t` - Comma-separated task list (default: all tasks)
- `--device` - Device to use: `auto`, `cpu`, `cuda`, `mps` (default: `auto`)
- `--encoder, -e` - Encoder type: `sentence-transformer`, `huggingface` (default: `sentence-transformer`)
- `--verbose, -v` - Verbose output
- `--config, -c` - Config file path (planned)

### `semeval compare` - Compare Models

Compare multiple models side-by-side (planned):

```bash
semeval compare --models "model1,model2,model3" --data test.json
```

### `semeval report` - Generate Reports

Generate formatted reports from evaluation results (planned):

```bash
# Generate HTML report
semeval report results.json

# Generate Markdown report
semeval report results.json --format markdown

# Custom output path
semeval report results.json -o my_report.html
```

### `semeval version` - Version Information

Show SemEval version:

```bash
semeval version
```

### `semeval info` - System Information

Display system and environment information:

```bash
semeval info
```

**Shows:**

- Python version
- Platform information
- PyTorch version
- CUDA availability
- MPS (Apple Silicon) availability
- GPU count

### CLI Performance

SemEval CLI is optimized for instant startup with lazy loading:

- **Help/Version/Info**: ~0.2s (no ML dependencies loaded)
- **Init/Validate**: ~0.2s (lightweight operations)
- **Eval**: Model loading time + evaluation time (ML dependencies loaded only when needed)

This is achieved through:

- Lazy module imports using Python's `__getattr__`
- Function-level imports for heavy dependencies
- No unnecessary torch/transformers loading for simple commands

---

## 📖 Usage Examples

### With Configuration

```python
from semeval import TaskRunner, SentenceTransformerEncoder, load_settings

# Load settings from config.yaml
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
[INFO] Model: sentence-transformers/all-MiniLM-L6-v2
[INFO] Running Information Retrieval Task
[INFO] Running Semantic Similarity Task
[INFO] Running Linguistic Robustness Task
[INFO] Running Vector Arithmetic Task
✅ Evaluation complete: 2.75s
```

### Export Results

```python
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

### Run Specific Task

```python
from semeval import TaskRunner, SentenceTransformerEncoder

encoder = SentenceTransformerEncoder("model-name")
runner = TaskRunner(encoder=encoder)

# Run only Semantic Similarity task
result = runner.run_task("semantic_similarity", "data/test_data.json")
print(f"Triplet Accuracy: {result.metrics['accuracy']:.2%}")
print(f"Average Margin: {result.metrics['avg_margin']:.3f}")
```

### Environment Variables

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

### Generate Comprehensive Report

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
    model_name="My Model",
    include_recommendations=True
)
```

### Compare Multiple Models

```python
from semeval import TaskRunner, SentenceTransformerEncoder
import pandas as pd

models = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "Alibaba-NLP/gte-multilingual-base",
    "your-custom-model"
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

## 🧪 Evaluation Tasks

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

- `config.yaml`: Base configuration
- `config.dev.yaml`: Development settings (verbose, quick metrics)
- `config.prod.yaml`: Production settings (optimized, extended metrics)

### Configuration Example

```yaml
# config.yaml
model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
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
settings = load_settings(env="dev")  # loads config.dev.yaml
settings = load_settings(env="prod") # loads config.prod.yaml

# Access settings
print(f"Model: {settings.model.name}")
print(f"Device: {settings.model.device}")
print(f"Output: {settings.output.base_dir}")
```

### Settings Priority

Settings are loaded with the following priority (highest to lowest):

1. Environment variables (`SEMEVAL_*`)
2. `.env` file
3. Environment-specific YAML (`config.{env}.yaml`)
4. Base YAML (`config.yaml`)
5. Default values in code

---

## 🔌 Supported Encoders

### Sentence Transformers

```python
from semeval.core.encoders import SentenceTransformerEncoder

encoder = SentenceTransformerEncoder(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    device="auto"  # auto, cuda, mps, cpu
)
```

### HuggingFace Models

```python
from semeval.core.encoders import HuggingFaceEncoder

encoder = HuggingFaceEncoder(
    model_name="bert-base-uncased",
    device="cuda",
    max_length=512
)
```

### Custom Encoder

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

## 📝 Data Format

Test data is provided in JSON format:

```json
{
  "metadata": {
    "version": "1.0",
    "description": "Semantic Evaluation Test Suite",
    "language": "en",
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

### Export Formats

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
    model_name="My Model",
    include_recommendations=True
)
```

### Per-Task Exports

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

### Output Structure

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

## 🗺️ Roadmap & Upcoming Features

### 🚨 v0.1.1 - Stabilization Release

**Focus**: Production readiness foundations

- ✅ **Testing Infrastructure** - 121 tests with >90% coverage
- ✅ **CLI Interface** - Fast command-line tools (`eval`, `validate`, `init`, `version`, `info`) with ~0.2s startup
- ✅ **Data Validation** - Schema validation with helpful error messages
- ✅ **Performance Optimization** - Lazy loading for instant CLI startup (20x speedup)
- 🔴 **Error Handling** - Robust error recovery (in progress)
- 🔴 **CI/CD Pipeline** - Automated testing and deployment (planned)

> **Why this matters**: Solid foundations ensure reliability and great developer experience before expanding features.

### 🎯 v0.2.0 - Core Expansion

**Focus**: Advanced metrics and new tasks

- ⭐ **Isotropy & Uniformity Metrics** - Label-free embedding quality analysis
- ⭐ **Semantic Textual Similarity (STS)** - Continuous similarity scoring
- ⭐ **Clustering Evaluation** - Unsupervised quality metrics
- ⭐ **Paraphrase Detection** - Binary classification task
- ⚡ **Caching Layer** - Speed up repeated evaluations
- 🎨 **Advanced CLI Features**:
  - `semeval compare` - Side-by-side model comparison with statistical tests
  - `semeval report` - HTML/PDF report generation with charts
  - `semeval benchmark` - Run standardized benchmarks
  - Config file support for `eval` command

### 🚀 v0.3.0 and Beyond

**Focus**: Advanced analysis and ecosystem integration

- **CKA (Centered Kernel Alignment)** - Compare model representations
- **Token-level Alignment** - Fine-grained semantic matching
- **Question Answering Retrieval** - QA-specific evaluation
- **HuggingFace Hub Integration** - Direct model/dataset access
- **Performance Monitoring** - Profiling and optimization tools
- **Interactive Dashboard** - Web UI for evaluation

---

## 📊 Project Info

**Current Version**: v0.1.0 (Beta)  
**License**: MIT  
**Python**: 3.8+  
**Maintainer**: [@omrylcn](https://github.com/omrylcn)

---

## 🎯 Project Goals

SemEval aims to be:

- **Simple** - From JSON to insights in minutes
- **Flexible** - Any language, any domain, any model
- **Comprehensive** - Beyond accuracy, understand embedding quality
- **Production-Ready** - Testing, error handling, monitoring
- **Community-Driven** - Open source

---
