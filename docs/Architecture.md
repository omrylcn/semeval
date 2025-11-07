# SemEval Package Architecture

**Version:** 0.1.0
**Last Updated:** 2025-11-08

## Table of Contents

1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Package Structure](#package-structure)
4. [Core Components](#core-components)
5. [Configuration System](#configuration-system)
6. [Data Flow](#data-flow)
7. [Extensibility](#extensibility)

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
├── __init__.py                      # Main API exports
├── README.md                        # User documentation
├── USAGE.md                         # Detailed usage guide
├── Architecture.md                  # This document
│
├── core/                            # Core framework components
│   ├── __init__.py                 # Core exports
│   ├── schemas.py                  # Pydantic data models
│   ├── base_encoder.py             # Encoder abstraction
│   ├── base_loader.py              # Loader abstraction
│   ├── runner.py                   # Task orchestration
│   ├── config.py                   # Configuration system
│   ├── yaml_source.py              # YAML config source
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
└── scripts/                         # Examples and tests
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
| `core/` | Framework abstractions and implementations | runner.py, config.py, schemas.py |
| `core/encoders/` | Text encoder implementations | sentence_transformer_encoder.py |
| `core/loaders/` | Data loader implementations | json_loader.py |
| `tasks/` | Evaluation task implementations | base.py, registry.py, 4 tasks |
| `metrics/` | Low-level metric calculations | ir_metrics.py |
| `postprocess/` | Results export and reporting | results_exporter.py, report_generator.py |
| `data/` | Test datasets | test_data.json |
| `scripts/` | Examples and tests | example_usage.py, test_*.py |

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

### 5. Post-Processing (`postprocess/`)

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

### Unit Tests

- Test each component in isolation
- Mock dependencies
- Test edge cases

### Integration Tests

Scripts in `scripts/`:
- `test_all_tasks.py`: Test all 4 tasks
- `test_config.py`: Test configuration loading
- `test_per_task_export.py`: Test per-task export
- `test_postprocessing_all.py`: Test full pipeline

### Example Tests

- `example_usage.py`: Comprehensive usage examples
- `test_model_comparison.py`: Model comparison

---

## Future Enhancements

### Planned Features

1. **CLI Interface** (v0.2.0)
   - Command-line tool for running evaluations
   - Progress bars and real-time metrics

2. **Parallel Task Execution**
   - Run independent tasks in parallel
   - Reduce total evaluation time

3. **Additional Encoders**
   - OpenAI embeddings
   - Cohere embeddings
   - Custom API encoders

4. **More Export Formats**
   - Excel (.xlsx)
   - LaTeX tables
   - HTML reports

5. **Web UI** (v2.0.0)
   - Visual interface for running evaluations
   - Interactive result exploration
   - Comparison dashboards

---

## References

- **User Documentation**: [README.md](../README.md)
- **Detailed Usage**: [USAGE.md](../USAGE.md)
- **Pydantic**: https://docs.pydantic.dev/
- **Sentence Transformers**: https://www.sbert.net/

---

**Last Updated:** 2025-11-08
**Version:** 0.1.0
