# SemEval Package Architecture

**Version:** 0.1.1
**Last Updated:** 2025-11-15

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Package Structure](#package-structure)
4. [Core Components](#core-components)
5. [CLI Architecture](#cli-architecture)
6. [Configuration System](#configuration-system)
7. [Data Flow](#data-flow)
8. [Extensibility](#extensibility)

---

## Overview

SemEval is a modular, extensible evaluation framework for semantic embeddings and NLP models. The architecture follows SOLID principles and emphasizes:

- **Modularity**: Clear separation of concerns with pluggable components
- **Extensibility**: Easy to add new encoders, tasks, and data loaders
- **Type Safety**: Pydantic V2 validation for all data structures
- **Configuration**: Flexible YAML-based config with environment overrides
- **Testability**: Each component is independently testable

### Key Features

- 🔌 **Pluggable Architecture**: Swap encoders, loaders, and tasks easily
- ✅ **Type-Safe**: Pydantic models for all data structures
- 🖥️ **CLI Interface**: Fast command-line tools with lazy loading (~0.2s startup)
- ⚙️ **Configuration**: YAML + environment variables with pydantic-settings
- 📊 **4 Evaluation Tasks**: IR, Semantic Similarity, Linguistic Robustness, Vector Arithmetic
- 📝 **Rich Export**: JSON, CSV, Markdown, per-task reports
- 🚀 **Performance**: Optimized for both CPU and GPU execution

---

## Design Principles

### 1. Dependency Injection

All major components accept dependencies through constructor injection:

```python
# TaskRunner depends on encoder abstraction
runner = TaskRunner(
    encoder=encoder,      # Any BaseEncoder implementation
    settings=settings     # Optional configuration
)

# Tasks depend on encoder, not concrete implementation
task = InformationRetrieval(
    encoder=encoder,
    task_data=data
)
```

**Benefits:**
- Easy testing with mock objects
- Runtime flexibility to swap implementations
- Clear dependency graph

### 2. Interface Segregation

Each component implements a focused interface:

- `BaseEncoder`: Text encoding operations only
- `BaseDataLoader`: Data loading and validation only
- `BaseTask`: Task execution and result generation only

### 3. Configuration as Code

Configuration is managed through type-safe Pydantic settings:

```python
from semeval import load_settings

# Load YAML config with env var overrides
settings = load_settings(env="prod")

# Type-safe access
model_name = settings.model.name  # IDE autocomplete works!
device = settings.model.device
```

### 4. Data Validation First

All data passes through Pydantic validation before use:

```python
# Automatic validation on load
test_data = loader.load("data.json")  # Raises ValidationError if invalid

# Type-safe access
corpus = test_data.tasks.information_retrieval.corpus
```

**Benefits:**
- Fail fast with clear error messages
- Eliminates runtime type errors
- Self-documenting data structures

---

## Package Structure

```
semeval/
├── __init__.py                      # Main API exports (with lazy loading)
├── README.md                        # User documentation
├── USAGE.md                         # Detailed usage guide
├── Architecture.md                  # This document
│
├── cli/                             # Command-line interface
│   ├── __init__.py                 # CLI exports
│   ├── main.py                     # Typer app & command registration
│   │
│   ├── commands/                   # CLI command implementations
│   │   ├── __init__.py
│   │   ├── eval_cmd.py            # eval command (lazy imports)
│   │   ├── validate.py            # validate command (lazy imports)
│   │   ├── compare.py             # compare command (planned)
│   │   ├── report.py              # report command (planned)
│   │   └── init.py                # init command (template generation)
│   │
│   └── utils/                      # CLI utilities
│       ├── __init__.py
│       ├── output.py              # Rich console helpers
│       └── config.py              # Config wrapper for CLI
│
├── core/                            # Core framework components
│   ├── __init__.py                 # Core exports
│   ├── schemas.py                  # Pydantic data models
│   ├── base_encoder.py             # Encoder abstraction
│   ├── base_loader.py              # Loader abstraction
│   ├── runner.py                   # Task orchestration
│   ├── config.py                   # Configuration system
│   ├── yaml_source.py              # YAML config source
│   ├── exceptions.py               # Custom exception hierarchy (17 exceptions)
│   ├── logging.py                  # Structured logging system
│   │
│   ├── encoders/                   # Encoder implementations
│   │   ├── __init__.py
│   │   ├── sentence_transformer_encoder.py
│   │   └── huggingface_encoder.py
│   │
│   └── loaders/                    # Loader implementations
│       ├── __init__.py
│       └── json_loader.py
│
├── tasks/                           # Evaluation tasks
│   ├── __init__.py
│   ├── base.py                     # BaseTask abstraction
│   ├── registry.py                 # Task registry
│   ├── information_retrieval.py    # IR task
│   ├── semantic_similarity.py      # Triplet evaluation
│   ├── linguistic_robustness.py    # Variation testing
│   └── vector_arithmetic.py        # Analogy evaluation
│
├── metrics/                         # Metric computation
│   ├── __init__.py
│   └── ir_metrics.py               # IR metrics (NDCG, MRR, MAP, etc.)
│
├── postprocess/                     # Results export & reporting
│   ├── __init__.py
│   ├── results_exporter.py         # CSV/JSON/Markdown export
│   └── report_generator.py         # Comprehensive reports
│
├── config.yaml                      # Base configuration
├── config.dev.yaml                  # Development config
├── config.prod.yaml                 # Production config
├── .env.example                     # Environment variable template
│
├── data/                            # Test datasets
│   └── test_data.json              # Sample test data (4 tasks)
│
├── tests/                           # Unit tests (153 tests, 51% coverage)
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures
│   ├── TESTING_GUIDE.md           # Comprehensive testing guide (English)
│   ├── TEST_REHBERI_TR.md         # Testing guide (Turkish)
│   ├── test_encoders.py           # Encoder tests (46 tests)
│   ├── test_loaders.py            # Data loader tests (27 tests)
│   ├── test_runner.py             # Runner integration tests (7 tests)
│   ├── test_tasks_integration.py  # Task integration tests (7 tests)
│   ├── test_cli_commands.py       # CLI tests (14 tests)
│   ├── test_config.py             # Config tests
│   ├── test_schemas.py            # Schema tests
│   ├── test_ir_metrics.py         # IR metrics tests
│   ├── test_similarity_metrics.py # Similarity metrics tests
│   ├── test_arithmetic_metrics.py # Arithmetic metrics tests
│   └── test_robustness_metrics.py # Robustness metrics tests
│
└── scripts/                         # Examples and integration tests
    ├── example_usage.py            # Comprehensive examples
    ├── test_all_tasks.py           # Test all 4 tasks
    ├── test_config.py              # Test configuration
    ├── test_with_config.py         # Test with config
    ├── test_per_task_export.py     # Test per-task export
    ├── test_postprocessing_all.py  # Test all post-processing
    └── test_model_comparison.py    # Compare models
```

### Directory Responsibilities

| Directory | Responsibility | Key Files |
|-----------|---------------|-----------|
| `cli/` | Command-line interface | main.py, commands/, utils/ |
| `cli/commands/` | CLI command implementations | eval_cmd.py, validate.py, init.py |
| `cli/utils/` | CLI utilities | output.py (Rich console), config.py |
| `core/` | Framework abstractions and implementations | runner.py, config.py, schemas.py |
| `core/encoders/` | Text encoder implementations | sentence_transformer_encoder.py |
| `core/loaders/` | Data loader implementations | json_loader.py |
| `tasks/` | Evaluation task implementations | base.py, registry.py, 4 tasks |
| `metrics/` | Low-level metric calculations | ir_metrics.py |
| `postprocess/` | Results export and reporting | results_exporter.py, report_generator.py |
| `tests/` | Unit tests (121 tests, >90% coverage) | test_*.py |
| `data/` | Test datasets | test_data.json |
| `scripts/` | Examples and integration tests | example_usage.py, test_*.py |

---

## Core Components

### 1. Schemas (`core/schemas.py`)

Pydantic V2 models defining all data structures.

**Key Models:**

```python
TestDataModel
├── metadata: TestMetadata
└── tasks: TasksModel
    ├── information_retrieval: InformationRetrievalData
    ├── semantic_similarity: SemanticSimilarityData
    ├── linguistic_robustness: LinguisticRobustnessData
    └── vector_arithmetic: VectorArithmeticData
```

Each task data model includes:
- Task-specific data (corpus, queries, triplets, etc.)
- Configuration options
- Pydantic validators for business logic

### 2. Encoders (`core/base_encoder.py`, `core/encoders/`)

Text encoding abstraction with multiple implementations:

- **SentenceTransformerEncoder**: Sentence Transformers models
- **HuggingFaceEncoder**: Raw HuggingFace transformers with mean pooling
- **Custom encoders**: Extend `BaseEncoder` for your own

**Key Interface:**
```python
class BaseEncoder(ABC):
    @abstractmethod
    def encode(self, texts, **kwargs) -> np.ndarray:
        """Encode texts to embeddings."""
        pass

    @abstractmethod
    def get_embedding_dim(self) -> int:
        """Get embedding dimension."""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get model name."""
        pass
```

### 3. Task Runner (`core/runner.py`)

Orchestrates evaluation across all enabled tasks.

**Responsibilities:**
- Load configuration
- Initialize tasks
- Execute tasks (serial or parallel)
- Aggregate results

**Key Features:**
- Backward compatible (works with/without settings)
- Supports running all tasks or specific tasks
- Provides unified result format

### 4. Tasks (`tasks/`)

Each task is a self-contained evaluation module:

**Common Pattern:**
```python
class TaskName(BaseTask):
    def run(self) -> TaskResult:
        """Execute task and return results."""
        # 1. Encode texts
        # 2. Compute metrics
        # 3. Return TaskResult
        pass

    @staticmethod
    def format_markdown_report(result: TaskResult) -> List[str]:
        """Format results as markdown (uses pandas)."""
        pass
```

**Task Registry:**
- Tasks are registered in `tasks/registry.py`
- Dynamic task lookup by name
- Easy to add new tasks

### 5. Error Handling (`core/exceptions.py`)

**Responsibility**: Structured, context-rich error handling.

**Exception Hierarchy**:

```
SemEvalError (base)
├── DataError
│   ├── DataLoadError         # File not found, read errors
│   ├── DataFormatError        # JSON parse errors
│   └── DataValidationError    # Schema validation failures
├── ModelError
│   ├── ModelLoadError         # Model loading failures
│   └── EncoderError           # Encoding errors
├── TaskError
│   ├── TaskConfigError        # Task configuration errors
│   └── TaskExecutionError     # Task execution failures
├── MetricError
│   └── MetricCalculationError # Metric computation errors
├── ConfigError
│   └── InvalidConfigError     # Configuration errors
└── ExportError
    └── ExportFormatError      # Export format errors
```

**Key Features**:

1. **Rich Context**: Each exception carries relevant context
   ```python
   raise DataLoadError(
       "File not found",
       file_path="/path/to/file.json",
       attempted_paths=["/path1", "/path2"]
   )
   ```

2. **User-Friendly Messages**: Clear, actionable error messages
3. **Error Propagation**: Exceptions preserve stack traces with `from e`
4. **Fail-Fast**: Validation errors caught early

**Design Benefits**:
- Specific error handling (catch DataLoadError vs generic Exception)
- Better debugging (context shows exactly what failed)
- User-friendly CLI error messages

### 6. Logging System (`core/logging.py`)

**Responsibility**: Structured logging with performance tracking.

**Key Components**:

```python
# Centralized logger
logger = get_logger("semeval")  # Single instance across the app

# Structured logging with context
class StructuredLogger:
    def info(self, message: str, **context):
        """Log with extra context fields."""
        # Example: logger.info("Loading model", model_name="bert", device="cuda")

# Performance tracking
class PerformanceLogger:
    """Track operation timing and resource usage."""

# Context managers
@contextmanager
def log_execution_time(logger, operation: str, **context):
    """Automatically log operation timing."""
    # Example:
    # with log_execution_time(logger, "model_loading", model="bert"):
    #     model = load_model()
    # Logs: "model_loading completed in 5.23s"

@contextmanager
def log_errors(logger, operation: str, **context):
    """Automatically log and re-raise errors with context."""
```

**Usage Example**:

```python
from semeval.core.logging import get_logger, log_execution_time
from semeval.core.exceptions import EncoderError

logger = get_logger("semeval")

def load_encoder(model_name: str):
    try:
        with log_execution_time(logger, "encoder_loading", model=model_name):
            encoder = SentenceTransformerEncoder(model_name)
            logger.info("Encoder loaded", dim=encoder.get_embedding_dim())
            return encoder
    except Exception as e:
        logger.error("Failed to load encoder", model=model_name, error=str(e))
        raise EncoderError(
            f"Failed to load model {model_name}",
            model_name=model_name
        ) from e
```

**Logging Levels**:
- `DEBUG`: Detailed diagnostic information
- `INFO`: General informational messages
- `WARNING`: Warning messages (non-critical issues)
- `ERROR`: Error messages (failures)

**Performance Tracking**:
- Automatic timing for major operations (model loading, encoding, task execution)
- Logged with structured context for analysis
- Helps identify bottlenecks

**Design Philosophy**:
- **Single logger instance** (`get_logger("semeval")`) for consistency
- **Structured logging** with key-value pairs for easy parsing
- **Context managers** for automatic timing and error handling
- **Graceful degradation** with logging (log errors but continue if possible)

### 7. Post-Processing (`postprocess/`)

Results export and reporting system:

**ResultsExporter:**
- Export to CSV, JSON, Markdown
- Per-task file generation
- Uses task formatters for markdown

**ReportGenerator:**
- Comprehensive markdown reports
- Performance analysis
- Recommendations

---

## CLI Architecture

### Overview

The CLI provides a fast, user-friendly command-line interface built with Typer and Rich. Key design goals:

- **Instant Startup**: ~0.2s performance with lazy loading
- **Beautiful Output**: Rich formatting with progress bars and tables
- **User-Friendly**: Clear help messages and error handling
- **Automation-Ready**: Scriptable commands with proper exit codes

### Technology Stack

- **Typer**: Modern CLI framework with automatic help generation
- **Rich**: Terminal formatting, progress bars, tables
- **Lazy Loading**: Python `__getattr__` for deferred imports

### CLI Components

```
cli/
├── main.py                    # Typer app, command registration
├── commands/
│   ├── eval_cmd.py           # Model evaluation
│   ├── validate.py           # Data validation
│   ├── init.py               # Template generation
│   ├── compare.py            # Model comparison (planned)
│   └── report.py             # Report generation (planned)
└── utils/
    ├── output.py             # Rich console helpers
    └── config.py             # Config wrapper
```

### Performance Optimization

**Problem**: Initial CLI startup was 3.4s due to eager loading of torch, transformers, sentence-transformers.

**Solution**: Two-level lazy loading strategy

#### 1. Package-Level Lazy Loading

```python
# semeval/__init__.py
def __getattr__(name):
    """Lazy import using Python's __getattr__"""
    if name == "TaskRunner":
        from .core.runner import TaskRunner
        return TaskRunner
    # ... other exports
```

**Effect**: `import semeval` no longer loads heavy dependencies.

#### 2. Function-Level Lazy Loading

```python
# semeval/cli/commands/eval_cmd.py
def eval(model, data, ...):
    # Lazy import inside function
    from semeval.core.encoders import SentenceTransformerEncoder
    from semeval.core.runner import TaskRunner

    # Use encoders...
```

**Effect**: Heavy dependencies only loaded when command actually runs.

#### Performance Results

| Command | Before | After | Improvement |
|---------|--------|-------|-------------|
| `semeval --help` | 3.4s | 0.17s | **20x faster** |
| `semeval version` | 3.0s | 0.09s | **33x faster** |
| `semeval init` | 3.0s | 0.11s | **27x faster** |
| `semeval validate` | 3.0s | 0.25s | **12x faster** |

### Commands

#### `semeval init`

**Purpose**: Generate template test data files

**Implementation**:
```python
# Commands/init.py
TEMPLATES = {
    'basic': {...},      # Semantic similarity
    'ir': {...},         # Information retrieval
    'similarity': {...}, # Comprehensive similarity
    'robustness': {...}  # Linguistic robustness
}

def init(template, output, force):
    # No heavy dependencies - instant execution
    data = TEMPLATES[template]
    write_json(output, data)
```

**Performance**: ~0.11s (no ML dependencies)

#### `semeval validate`

**Purpose**: Validate test data schema

**Implementation**:
```python
# commands/validate.py
def validate(data, strict, report):
    # Lazy import
    from semeval.core.loaders import JSONDataLoader

    loader = JSONDataLoader()
    test_data = loader.load(data)  # Pydantic validation

    # Print statistics, warnings
    show_stats(test_data)
```

**Performance**: ~0.25s (only Pydantic, no torch)

#### `semeval eval`

**Purpose**: Run model evaluation

**Implementation**:
```python
# commands/eval_cmd.py
def eval(model, data, output, tasks, device, encoder_type, verbose):
    # Lazy import heavy dependencies
    from semeval.core.encoders import SentenceTransformerEncoder
    from semeval.core.runner import TaskRunner

    # Load model
    encoder = SentenceTransformerEncoder(model, device=device)

    # Run evaluation
    runner = TaskRunner(encoder=encoder)
    result = runner.run(data)

    # Save results
    result.save(output)
```

**Performance**: Model loading time + evaluation time (lazy loading adds ~0s overhead)

#### `semeval compare` & `semeval report`

**Status**: Planned for v0.2.0

**Purpose**:
- `compare`: Side-by-side model comparison with statistical tests
- `report`: HTML/PDF report generation with charts

### Output Formatting

Uses Rich library for beautiful terminal output:

```python
# cli/utils/output.py
from rich.console import Console
from rich.progress import Progress
from rich.panel import Panel

console = Console()

def success(message):
    console.print(f"✅ [bold green]{message}[/bold green]")

def create_progress():
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
    )
```

**Features**:
- Colored output with emojis
- Progress bars for long operations
- Tables for results
- Panels for summaries

### Entry Point

Configured in `pyproject.toml`:

```toml
[project.scripts]
semeval = "semeval.cli.main:app"
```

After installation, `semeval` command is globally available.

### Error Handling

CLI provides user-friendly error messages:

```python
try:
    # Command logic
except DataValidationError as e:
    error(f"Validation failed: {e.message}")
    if e.errors:
        for err in e.errors:
            console.print(f"  • {err}")
    raise typer.Exit(code=1)
except Exception as e:
    error(f"{type(e).__name__}: {e}")
    if verbose:
        console.print_exception()
    raise typer.Exit(code=1)
```

**Exit Codes**:
- `0`: Success
- `1`: General error
- `130`: User interrupt (Ctrl+C)

### CLI Design Patterns

1. **Lazy Loading**: Defer imports until needed
2. **Rich Output**: Beautiful, informative terminal UI
3. **Type Safety**: Typer automatically validates arguments
4. **Help Generation**: Automatic `--help` from docstrings
5. **Scriptable**: Proper exit codes, JSON output for automation

---

## Configuration System

### Architecture

```
Environment Variables (SEMEVAL_*)
           ↓
      .env file
           ↓
   config.{env}.yaml
           ↓
      config.yaml
           ↓
   Default values in code
```

**Priority order** (highest to lowest):
1. Environment variables
2. .env file
3. Environment-specific YAML
4. Base YAML
5. Code defaults

### Configuration Classes

```python
SemEvalSettings
├── model: ModelConfig
│   ├── name: str
│   ├── device: str
│   └── batch_size: int
├── output: OutputConfig
│   ├── base_dir: str
│   ├── export_formats: List[str]
│   └── save_comprehensive_report: bool
├── tasks: TasksConfig
│   ├── information_retrieval: InformationRetrievalConfig
│   ├── semantic_similarity: SemanticSimilarityConfig
│   ├── linguistic_robustness: LinguisticRobustnessConfig
│   └── vector_arithmetic: VectorArithmeticConfig
└── logging: LoggingConfig
    ├── verbose: bool
    ├── level: str
    └── log_dir: str
```

### Custom YAML Source

`YamlConfigSettingsSource` extends pydantic-settings to load YAML files with proper type conversion and validation.

---

## Data Flow

### Evaluation Flow

```
1. Load Configuration
   ↓
2. Load Test Data (JSONDataLoader)
   ↓ (Pydantic validation)
3. Create Encoder
   ↓
4. Initialize TaskRunner
   ↓
5. For each enabled task:
   - Create task instance
   - task.run()
   - Collect TaskResult
   ↓
6. Aggregate into EvaluationResult
   ↓
7. Export/Report
```

### Task Execution Flow

```
1. Task.__init__(encoder, task_data)
   ↓
2. task.run()
   ├─→ Encode texts (corpus, queries, etc.)
   ├─→ Compute similarities/operations
   ├─→ Calculate metrics
   └─→ Return TaskResult
```

### Export Flow

```
EvaluationResult
   ↓
ResultsExporter.export_per_task()
   ├─→ For each task:
   │   ├─→ Create TaskResult
   │   ├─→ Call task.format_markdown_report()
   │   ├─→ Write JSON file
   │   └─→ Write Markdown file
   └─→ Return paths dictionary
```

---

## Extensibility

### Adding a New Encoder

```python
from semeval.core.base_encoder import BaseEncoder
import numpy as np

class MyEncoder(BaseEncoder):
    def __init__(self, model_name: str):
        self._model_name = model_name
        # Initialize your model
        self.model = load_my_model(model_name)

    def encode(self, texts, **kwargs) -> np.ndarray:
        # Your encoding logic
        return self.model.encode(texts)

    def get_embedding_dim(self) -> int:
        return 768

    @property
    def model_name(self) -> str:
        return self._model_name
```

### Adding a New Task

1. **Create task file** in `tasks/new_task.py`:

```python
from .base import BaseTask, TaskResult
import pandas as pd

class NewTask(BaseTask):
    def run(self) -> TaskResult:
        # Your task logic
        metrics = {}  # Compute metrics
        return TaskResult(
            task_name="new_task",
            status="success",
            metrics=metrics,
            runtime_seconds=runtime
        )

    @staticmethod
    def format_markdown_report(result: TaskResult) -> List[str]:
        # Format with pandas
        import pandas as pd
        df = pd.DataFrame(...)
        return [df.to_markdown(index=False)]
```

2. **Register task** in `tasks/registry.py`:

```python
from .new_task import NewTask

TASK_REGISTRY = {
    'new_task': NewTask,
    # ... existing tasks
}
```

3. **Add config** in `core/config.py`:

```python
class NewTaskConfig(BaseModel):
    enabled: bool = True
    # Task-specific config

class TasksConfig(BaseModel):
    new_task: NewTaskConfig = Field(default_factory=NewTaskConfig)
    # ... existing tasks
```

4. **Add to schema** in `core/schemas.py`:

```python
class NewTaskData(BaseModel):
    # Your data structure
    pass

class TasksModel(BaseModel):
    new_task: Optional[NewTaskData] = None
    # ... existing tasks
```

### Adding a New Data Loader

```python
from semeval.core.base_loader import BaseDataLoader
from semeval.core.schemas import TestDataModel

class MyLoader(BaseDataLoader):
    def load(self, source: str, **kwargs) -> TestDataModel:
        # Load from your source
        data = my_load_function(source)

        # Pydantic validation happens automatically
        return TestDataModel(**data)
```

---

## Best Practices

### 1. Type Hints Everywhere

```python
def my_function(text: str, encoder: BaseEncoder) -> np.ndarray:
    embeddings: np.ndarray = encoder.encode(text)
    return embeddings
```

### 2. Pydantic Validation

```python
# Don't manually validate
if 'corpus' not in data:
    raise ValueError("Missing corpus")

# Let Pydantic do it
test_data = TestDataModel(**data)  # Automatic validation
```

### 3. Configuration Over Hard-coding

```python
# Don't hard-code
batch_size = 32

# Use config
batch_size = settings.model.batch_size
```

### 4. Use Static Methods for Formatters

```python
@staticmethod
def format_markdown_report(result: TaskResult) -> List[str]:
    # Can be called without task instance
    pass
```

### 5. Pandas for Tables

```python
# Don't manually format tables
lines = []
lines.append("| Metric | Value |")
lines.append("|--------|-------|")
# ...

# Use pandas
df = pd.DataFrame({'Metric': [...], 'Value': [...]})
lines.append(df.to_markdown(index=False))
```

---

## Performance Considerations

### Device Selection

- **Auto mode** (`device="auto"`): Automatically selects cuda > mps > cpu
- **Explicit mode**: Set `device="cuda"` for GPU, `device="cpu"` for CPU

### Batch Processing

- Configure `batch_size` in settings
- Larger batches = faster (if GPU memory allows)
- Default: 32 (good balance)

### Caching

- Models are cached by sentence-transformers
- Set `cache_folder` in encoder for custom location

---

## Testing Strategy

**Current Status**: 153 tests, 51% coverage

### Test Categories

#### 1. Unit Tests (Isolated Component Testing)

Tests individual functions and classes in isolation using mocks.

**Files**:
- `test_encoders.py` (46 tests) - Encoder implementations
- `test_loaders.py` (27 tests) - Data loading and validation
- `test_config.py` - Configuration system
- `test_schemas.py` - Pydantic schema validation
- `test_*_metrics.py` (52 tests) - Metric calculations

**Coverage Highlights**:
- `json_loader.py`: 96%
- `huggingface_encoder.py`: 86%
- `sentence_transformer_encoder.py`: 80%
- `arithmetic_metrics.py`: 100%
- `ir_metrics.py`: 100%
- `robustness_metrics.py`: 98%

#### 2. Integration Tests (Multi-Component Testing)

Tests multiple components working together with real objects.

**Files**:
- `test_runner.py` (7 tests) - TaskRunner orchestration
  - Full evaluation pipeline
  - Result aggregation
  - Error handling
- `test_tasks_integration.py` (7 tests) - Task execution
  - All 4 task types
  - End-to-end task workflow
  - Metric calculation

**Test Pattern**:
```python
def test_runner_run_with_minimal_data(small_model_encoder, minimal_test_data):
    """Test full evaluation pipeline."""
    # Uses real encoder (small model) + real data
    runner = TaskRunner(encoder=small_model_encoder)
    result = runner.run(test_data_path)

    # Verify entire pipeline works
    assert result.metadata["version"] == "1.0.0"
    assert len(result.task_results) > 0
```

#### 3. CLI Tests (Command-Line Interface Testing)

Tests CLI commands using mocks to avoid heavy dependencies.

**Files**:
- `test_cli_commands.py` (14 tests)
  - Validate command (4 tests)
  - Eval command (6 tests)
  - Utility functions (4 tests)

**Mock-Based Testing**:
```python
@patch("semeval.core.runner.TaskRunner")
@patch("semeval.core.encoders.SentenceTransformerEncoder")
def test_eval_basic_execution(mock_encoder, mock_runner, cli_runner):
    """Test eval command without loading real models."""
    # Mock setup (instant, no model loading)
    mock_encoder.return_value.get_embedding_dim.return_value = 768
    mock_runner.return_value.run.return_value = mock_result

    # Run CLI command
    result = cli_runner.invoke(app, ["--model", "test", "--data", "test.json"])

    # Verify CLI logic (not model loading)
    assert result.exit_code == 0
    assert mock_encoder.called
```

**Coverage Improvements**:
- `eval_cmd.py`: 0% → 91%
- `validate.py`: 0% → 65%
- `cli/main.py`: 0% → 47%

**Performance**: CLI tests run in <1s (vs 30s+ with real models)

### Testing Infrastructure

**Fixtures** (`conftest.py`):
```python
@pytest.fixture
def small_model_encoder():
    """Fast encoder for testing."""
    return SentenceTransformerEncoder(
        "sentence-transformers/paraphrase-albert-small-v2",
        device="cpu"
    )

@pytest.fixture
def minimal_test_data():
    """Minimal valid test data."""
    return {
        "metadata": {...},
        "tasks": {...}
    }
```

**Test Guides**:
- `TESTING_GUIDE.md` - Comprehensive guide (English)
  - How tests work
  - Writing new tests
  - Mock vs integration tests
  - Best practices
- `TEST_REHBERI_TR.md` - Testing guide (Turkish)
  - Simplified explanations
  - Real-world examples
  - Common issues and solutions

### Running Tests

```bash
# All tests
uv run pytest tests/

# With coverage
uv run pytest tests/ --cov=semeval --cov-report=term-missing

# Specific file
uv run pytest tests/test_cli_commands.py

# Verbose mode
uv run pytest tests/ -v

# HTML coverage report
uv run pytest tests/ --cov=semeval --cov-report=html
open htmlcov/index.html
```

### Coverage Analysis

**Current Coverage: 51%**

| Module | Coverage | Status |
|--------|----------|--------|
| **Core** | | |
| json_loader.py | 96% | ✅ Excellent |
| huggingface_encoder.py | 86% | ✅ Good |
| sentence_transformer_encoder.py | 80% | ✅ Good |
| base_loader.py | 92% | ✅ Excellent |
| config.py | 95% | ✅ Excellent |
| schemas.py | 95% | ✅ Excellent |
| runner.py | 50% | ⚠️ Needs improvement |
| exceptions.py | 55% | ⚠️ Needs improvement |
| logging.py | 39% | ⚠️ Needs improvement |
| **Metrics** | | |
| arithmetic_metrics.py | 100% | ✅ Complete |
| ir_metrics.py | 100% | ✅ Complete |
| robustness_metrics.py | 98% | ✅ Excellent |
| similarity_metrics.py | 89% | ✅ Good |
| **Tasks** | | |
| vector_arithmetic.py | 67% | ⚠️ Needs improvement |
| linguistic_robustness.py | 63% | ⚠️ Needs improvement |
| information_retrieval.py | 53% | ⚠️ Needs improvement |
| semantic_similarity.py | 42% | ⚠️ Needs improvement |
| base.py | 66% | ⚠️ Needs improvement |
| **CLI** | | |
| eval_cmd.py | 91% | ✅ Excellent |
| validate.py | 65% | ⚠️ Needs improvement |
| main.py | 47% | ⚠️ Needs improvement |
| **Postprocess** | | |
| All modules | 0% | ❌ Not tested yet |

**Priority for v0.2.0**:
1. Task implementations (42-67% → target 85%)
2. Runner edge cases (50% → 85%)
3. Exception handling paths (55% → 90%)
4. Logging system (39% → 80%)
5. Postprocess modules (0% → 70%)

### Example Integration Tests

Scripts in `scripts/` directory:
- `example_usage.py`: Comprehensive usage examples
- `test_all_tasks.py`: Test all 4 tasks end-to-end
- `test_config.py`: Configuration loading and overrides
- `test_per_task_export.py`: Export functionality
- `test_postprocessing_all.py`: Full pipeline with export
- `test_model_comparison.py`: Model comparison workflow

### Testing Philosophy

1. **Fast Feedback**: Unit tests run in <5s, full suite in ~90s
2. **Isolation**: Mock heavy dependencies (model loading)
3. **Real Integration**: Use small models for integration tests
4. **Documentation**: Test guides for contributors
5. **Coverage Target**: 90% by v0.2.0

---

## Future Enhancements

### Completed in v0.1.1

1. **✅ CLI Interface**
   - Fast command-line tools with ~0.2s startup
   - Progress bars and real-time metrics with Rich
   - Template generation, validation, evaluation
   - 20x performance improvement via lazy loading
   - 91% test coverage for eval command

2. **✅ Error Handling & Logging System**
   - Custom exception hierarchy (17 exceptions)
   - Structured logging with context
   - Performance tracking with context managers
   - Graceful degradation in runner
   - User-friendly error messages in CLI

3. **✅ Comprehensive Testing Infrastructure**
   - 153 tests with 51% coverage (up from 0%)
   - Unit tests (119 tests) for core components
   - Integration tests (14 tests) for runner and tasks
   - CLI tests (14 tests) with mock-based approach
   - Test guides (English + Turkish) for contributors
   - Coverage improvements:
     - `eval_cmd.py`: 0% → 91%
     - `json_loader.py`: 0% → 96%
     - `arithmetic_metrics.py`: 0% → 100%
     - `ir_metrics.py`: 0% → 100%

4. **✅ Performance Optimization**
   - Lazy module imports for instant CLI startup
   - Function-level imports for heavy dependencies
   - No unnecessary torch/transformers loading
   - Batch encoding optimizations

5. **✅ CI/CD Pipeline**
   - GitHub Actions workflow (.github/workflows/tests.yml)
   - Multi-Python version testing (3.8-3.11)
   - Automated linting and formatting checks
   - Coverage reporting

### Planned Features

1. **Advanced CLI Features** (v0.2.0)
   - `semeval compare`: Side-by-side model comparison with statistical tests
   - `semeval report`: HTML/PDF report generation with charts
   - `semeval benchmark`: Run standardized benchmarks
   - Config file support for eval command

2. **Parallel Task Execution** (v0.2.0)
   - Run independent tasks in parallel
   - Reduce total evaluation time
   - Progress tracking for parallel tasks

3. **Additional Encoders** (v0.2.0+)
   - OpenAI embeddings API
   - Cohere embeddings API
   - Custom API encoders with caching

4. **More Export Formats** (v0.2.0+)
   - Excel (.xlsx) with multiple sheets
   - LaTeX tables for papers
   - Interactive HTML reports with charts

5. **Web UI** (v0.3.0+)
   - Visual interface for running evaluations
   - Interactive result exploration
   - Comparison dashboards
   - Real-time progress monitoring

---

## References

- **User Documentation**: [README.md](../README.md)
- **Detailed Usage**: [USAGE.md](../USAGE.md)
- **CLI Guide**: [CLI.md](CLI.md)
- **Metrics Documentation**: [Metrics.md](Metrics.md)
- **Pydantic**: <https://docs.pydantic.dev/>
- **Sentence Transformers**: <https://www.sbert.net/>
- **Typer**: <https://typer.tiangolo.com/>
- **Rich**: <https://rich.readthedocs.io/>

---

**Last Updated:** 2025-11-15
**Version:** 0.1.1
