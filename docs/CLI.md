# SemEval CLI Guide

Complete guide to using SemEval's command-line interface.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Command Reference](#command-reference)
- [Workflows](#common-workflows)
- [Performance](#performance)
- [Troubleshooting](#troubleshooting)

---

## Installation

The CLI is automatically available after installing SemEval:

```bash
# Using uv (recommended)
uv pip install -e .

# Using pip
pip install -e .
```

Verify installation:

```bash
semeval version
```

---

## Quick Start

### 3-Minute Evaluation

```bash
# 1. Create test data from template
semeval init --template basic -o my_test.json

# 2. Validate the data
semeval validate my_test.json

# 3. Run evaluation
semeval eval --model "sentence-transformers/all-MiniLM-L6-v2" --data my_test.json

# 4. Check results in results/ directory
ls results/
```

---

## Command Reference

### `semeval init`

Create template test data files.

**Usage:**
```bash
semeval init [OPTIONS]
```

**Options:**
- `-t, --template TEXT` - Template type: `basic`, `ir`, `similarity`, `robustness` (default: `basic`)
- `-o, --output PATH` - Output JSON file path (default: `test_data.json`)
- `-f, --force` - Overwrite existing file

**Examples:**

```bash
# Create basic template (semantic similarity)
semeval init

# Create IR template
semeval init --template ir -o ir_test.json

# Create comprehensive similarity tests
semeval init --template similarity -o similarity_test.json

# Create robustness tests
semeval init --template robustness -o robustness_test.json

# Overwrite existing file
semeval init -t basic -o test.json --force
```

**Templates:**

| Template | Description | Tasks | Samples |
|----------|-------------|-------|---------|
| `basic` | Semantic similarity examples | Similarity | 2 triplets |
| `ir` | Information retrieval | IR | 4 docs, 2 queries |
| `similarity` | Comprehensive similarity | Similarity | 3 triplets |
| `robustness` | Linguistic robustness | Robustness | 2 morphology, 2 typo, 2 negation pairs |

---

### `semeval validate`

Validate test data format and schema.

**Usage:**
```bash
semeval validate [OPTIONS] DATA
```

**Arguments:**
- `DATA` - Path to test data JSON file

**Options:**
- `--strict` - Fail on warnings (exit code 1)
- `-r, --report PATH` - Save validation report to HTML file (planned)

**Examples:**

```bash
# Basic validation
semeval validate test_data.json

# Strict mode (treats warnings as errors)
semeval validate test_data.json --strict

# Generate validation report (planned)
semeval validate test_data.json --report validation.html
```

**Output:**

```
🔍 Validating: test_data.json

✅ Schema validation passed!

📊 Data Statistics:
✓ Enabled tasks: semantic_similarity
  • Similarity: 2 triplets

ℹ️  Metadata:
  Version: 1.0.0
  Language: en
  Domain: general

⚠️  Warnings:
⚠️  Warning: Semantic similarity has < 50 triplets (recommended: 100+)

✅ Validation passed with warnings
💡 Use --strict to treat warnings as errors
```

---

### `semeval eval`

Run evaluation on a model with test data.

**Usage:**
```bash
semeval eval [OPTIONS]
```

**Options:**
- `-m, --model TEXT` - Model name or path (required)
- `-d, --data PATH` - Path to test data JSON file (required)
- `-o, --output PATH` - Output directory for results (default: `results/`)
- `-t, --tasks TEXT` - Comma-separated list of tasks (e.g., `ir,similarity`)
- `--device TEXT` - Device: `auto`, `cpu`, `cuda`, `mps` (default: `auto`)
- `-e, --encoder TEXT` - Encoder type: `sentence-transformer`, `huggingface` (default: `sentence-transformer`)
- `-v, --verbose` - Verbose output
- `-c, --config PATH` - Path to YAML/JSON config file (planned)

**Examples:**

```bash
# Basic evaluation
semeval eval --model "sentence-transformers/all-MiniLM-L6-v2" --data test.json

# Short form
semeval eval -m "model-name" -d test.json

# Run specific tasks only
semeval eval -m "model" -d test.json --tasks "ir,similarity"

# Use HuggingFace encoder
semeval eval -m "bert-base-uncased" -d test.json --encoder huggingface

# Specify device
semeval eval -m "model" -d test.json --device cuda

# Custom output directory
semeval eval -m "model" -d test.json --output my_results/

# Verbose mode
semeval eval -m "model" -d test.json --verbose

# Use config file (planned)
semeval eval --config config.yaml
```

**Output:**

```
╔═══════════════════════════════════════════════╗
║                                               ║
║   🚀  SemEval - Semantic Evaluation v0.1.1    ║
║                                               ║
╚═══════════════════════════════════════════════╝

📦 Loading model: sentence-transformers/all-MiniLM-L6-v2
✅ Model loaded successfully (384 dimensions)
📊 Running all available tasks

🚀 Starting evaluation...

✅ Evaluation complete!
⏱️  Total time: 2.34s

📊 Results Summary:

Semantic Similarity:
  accuracy: 1.0000
  avg_margin: 0.6245
  margin_distribution_0.1: 1.0000
  margin_distribution_0.2: 1.0000

💾 Results saved to: results/sentence-transformers_all-MiniLM-L6-v2_1234567890.json

💡 Tip: Use 'semeval report' to generate HTML report from results
```

---

### `semeval compare`

Compare multiple models (planned for v0.2.0).

**Usage:**
```bash
semeval compare --models "model1,model2" --data test.json
```

**Planned Features:**
- Side-by-side metric comparison
- Statistical significance tests
- Performance charts
- Speed/accuracy trade-off analysis

---

### `semeval report`

Generate formatted reports (planned for v0.2.0).

**Usage:**
```bash
semeval report results.json [OPTIONS]
```

**Options:**
- `-f, --format TEXT` - Output format: `html`, `markdown`, `pdf` (default: `html`)
- `-o, --output PATH` - Output file path (auto-generated if not provided)

**Planned Features:**
- Beautiful HTML reports with charts
- Markdown reports for documentation
- PDF export capability
- Interactive visualizations

---

### `semeval version`

Show SemEval version information.

**Usage:**
```bash
semeval version
```

**Output:**
```
SemEval version 0.1.1

Semantic Embedding Evaluation Framework
https://github.com/omrylcn/semeval
```

---

### `semeval info`

Show system and environment information.

**Usage:**
```bash
semeval info
```

**Output:**
```
System Information
Python: 3.11.5
Platform: Darwin 24.6.0
PyTorch: 2.9.0
CUDA available: False
MPS available: True
```

---

## Common Workflows

### Workflow 1: Quick Model Test

Test a model with minimal setup:

```bash
# 1. Create test data
semeval init -o quick_test.json

# 2. Run evaluation
semeval eval -m "sentence-transformers/all-MiniLM-L6-v2" -d quick_test.json

# 3. Check results
cat results/*.json
```

### Workflow 2: Custom Data Evaluation

Evaluate with your own data:

```bash
# 1. Create template as starting point
semeval init --template ir -o my_data.json

# 2. Edit my_data.json with your data
# (Use your favorite editor)

# 3. Validate before running
semeval validate my_data.json

# 4. Run evaluation
semeval eval -m "your-model" -d my_data.json
```

### Workflow 3: Multi-Model Comparison

Compare multiple models:

```bash
# Create test data once
semeval init --template similarity -o benchmark.json

# Evaluate each model
for model in "model1" "model2" "model3"; do
  semeval eval -m "$model" -d benchmark.json -o "results_$model/"
done

# Compare results (manually for now, automated in v0.2.0)
```

### Workflow 4: Task-Specific Evaluation

Run only specific evaluation tasks:

```bash
# Create comprehensive test data
semeval init --template similarity -o test.json

# Run only IR task
semeval eval -m "model" -d test.json --tasks "ir"

# Run multiple tasks
semeval eval -m "model" -d test.json --tasks "ir,similarity,robustness"
```

### Workflow 5: GPU Evaluation

Use GPU for faster evaluation:

```bash
# Check GPU availability
semeval info

# Run with CUDA
semeval eval -m "large-model" -d test.json --device cuda

# Or MPS (Apple Silicon)
semeval eval -m "model" -d test.json --device mps
```

---

## Performance

### CLI Startup Times

SemEval CLI uses lazy loading for instant startup:

| Command | Cold Start | Warm Start | Dependencies Loaded |
|---------|-----------|-----------|---------------------|
| `semeval --help` | 0.17s | 0.15s | None |
| `semeval version` | 0.09s | 0.07s | None |
| `semeval info` | 0.12s | 0.10s | PyTorch (for info) |
| `semeval init` | 0.11s | 0.09s | None |
| `semeval validate` | 0.25s | 0.20s | Pydantic only |
| `semeval eval` | ~5-10s | ~5-10s | Full ML stack |

**Optimization Techniques:**

1. **Lazy Module Imports** - `__getattr__` in `__init__.py`
2. **Function-Level Imports** - Heavy dependencies imported inside functions
3. **No Eager Loading** - PyTorch/Transformers only loaded when needed

**Before Optimization:**
```
semeval --help: 3.4s (loading torch, transformers, etc.)
```

**After Optimization:**
```
semeval --help: 0.17s (20x faster!)
```

### Evaluation Performance

Actual evaluation time depends on:
- Model size
- Dataset size
- Hardware (CPU/GPU)
- Batch size

**Tips for Faster Evaluation:**
- Use GPU (`--device cuda`)
- Increase batch size in config
- Use smaller models for prototyping
- Cache encodings (planned for v0.2.0)

---

## Troubleshooting

### Common Issues

#### CLI Command Not Found

**Problem:**
```bash
$ semeval version
zsh: command not found: semeval
```

**Solution:**
```bash
# Make sure package is installed
pip install -e .

# Or use Python module directly
python -m semeval.cli.main version
```

#### Model Not Found

**Problem:**
```
Error: Model 'model-name' not found
```

**Solution:**
- Check model name spelling
- Ensure model exists on HuggingFace Hub
- Try downloading model manually first
- Use full model path for local models

#### CUDA Out of Memory

**Problem:**
```
CUDA out of memory. Tried to allocate...
```

**Solution:**
```bash
# Use CPU instead
semeval eval -m "model" -d test.json --device cpu

# Or reduce batch size in config.yaml
model:
  batch_size: 16  # Reduce from default 32
```

#### Validation Errors

**Problem:**
```
Validation failed: tasks -> semantic_similarity -> name: Field required
```

**Solution:**
- Use `semeval init` to create properly formatted template
- Check schema requirements in error message
- Ensure all required fields are present

### Getting Help

```bash
# Show all available commands
semeval --help

# Show help for specific command
semeval eval --help
semeval validate --help
semeval init --help
```

### Verbose Mode

Enable verbose output for debugging:

```bash
semeval eval -m "model" -d test.json --verbose
```

### Report Issues

If you encounter a bug:

1. Check existing [issues](https://github.com/omrylcn/semeval/issues)
2. Create a new issue with:
   - Command you ran
   - Error message
   - Output of `semeval info`
   - Python and PyTorch versions

---

## Advanced Usage

### Using with uv run

For project isolation:

```bash
# Run without installing globally
uv run python -m semeval.cli.main version

# Evaluate with uv
uv run python -m semeval.cli.main eval -m "model" -d test.json
```

### Scripting

Use in shell scripts:

```bash
#!/bin/bash

# Automated evaluation pipeline
MODEL="sentence-transformers/all-MiniLM-L6-v2"
DATA="test_data.json"
OUTPUT="results_$(date +%Y%m%d)"

# Create output directory
mkdir -p "$OUTPUT"

# Run evaluation
semeval eval -m "$MODEL" -d "$DATA" -o "$OUTPUT/"

# Check exit code
if [ $? -eq 0 ]; then
  echo "Evaluation successful!"
  echo "Results: $OUTPUT"
else
  echo "Evaluation failed!"
  exit 1
fi
```

### CI/CD Integration

Use in continuous integration:

```yaml
# .github/workflows/eval.yml
name: Model Evaluation

on: [push]

jobs:
  evaluate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install SemEval
        run: pip install -e .

      - name: Validate test data
        run: semeval validate data/test.json --strict

      - name: Run evaluation
        run: semeval eval -m "$MODEL" -d data/test.json

      - name: Upload results
        uses: actions/upload-artifact@v2
        with:
          name: evaluation-results
          path: results/
```

---

## Next Steps

- Read [Usage Guide](Usage.md) for Python API
- Check [Metrics Documentation](Metrics.md) for metric details
- View [Roadmap](../ROADMAP.md) for upcoming features
- Contribute to [GitHub](https://github.com/omrylcn/semeval)

---

*CLI designed for speed, simplicity, and power.* ⚡
