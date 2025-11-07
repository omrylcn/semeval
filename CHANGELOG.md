# Changelog

All notable changes to the SemEval project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- CLI interface for command-line evaluation
- Parallel task execution
- Additional encoder support (OpenAI, Cohere, custom APIs)
- More export formats (Excel, LaTeX, HTML)
- Web UI for interactive evaluation

---

## [0.1.0] - 2025-11-08

### Added

#### Core Framework
- **TaskRunner**: Main orchestrator for running evaluation tasks
- **EvaluationResult**: Container for evaluation results from multiple tasks
- **BaseEncoder**: Abstract base class for text encoders
- **BaseDataLoader**: Abstract base class for data loaders
- **BaseTask**: Abstract base class for evaluation tasks

#### Encoders
- **SentenceTransformerEncoder**: Encoder using Sentence Transformers library
  - Auto device detection (cuda > mps > cpu)
  - Batch processing support
  - Model caching
- **HuggingFaceEncoder**: Encoder using raw HuggingFace transformers
  - Mean pooling implementation
  - Custom max length support

#### Evaluation Tasks
- **Information Retrieval**: Evaluate document retrieval and ranking
  - Metrics: NDCG@k, MRR@k, MAP@k, Precision@k, Recall@k, Accuracy@k
  - Configurable k values
  - Uses sentence-transformers InformationRetrievalEvaluator

- **Semantic Similarity**: Evaluate triplet-based similarity understanding
  - Metrics: Triplet accuracy, average margin, margin distribution
  - Performance breakdown by difficulty level
  - Performance breakdown by subcategory
  - Failed triplet reporting

- **Linguistic Robustness**: Test model stability under linguistic variations
  - Morphology testing (case, number, tense changes)
  - Typo testing (spelling errors)
  - Negation testing (negation handling)
  - Per-variation-type metrics

- **Vector Arithmetic**: Test compositional semantic understanding
  - Analogy evaluation (a - b + c ≈ d)
  - Top-k accuracy metrics
  - Category and subcategory breakdowns

#### Configuration System
- **YAML-based configuration**: Flexible, hierarchical config files
  - Base config (`config.yaml`)
  - Environment-specific configs (`config.dev.yaml`, `config.prod.yaml`)
- **Environment variable overrides**: `SEMEVAL_*` prefix
- **`.env` file support**: Local configuration
- **Pydantic-settings integration**: Type-safe configuration with validation
- **Priority system**: env vars > .env > env-specific YAML > base YAML > defaults
- **Configuration classes**:
  - `ModelConfig`: Model settings (name, device, batch_size)
  - `OutputConfig`: Output settings (base_dir, formats, reports)
  - `TasksConfig`: Per-task configuration
  - `LoggingConfig`: Logging settings (verbose, level, log_dir)
  - `SemEvalSettings`: Main settings container

#### Data Management
- **JSONDataLoader**: Load and validate test data from JSON files
- **Pydantic V2 schemas**: Type-safe data models with validation
  - `TestDataModel`: Root model for test data
  - `InformationRetrievalData`: IR task data
  - `SemanticSimilarityData`: Semantic similarity task data
  - `LinguisticRobustnessData`: Linguistic robustness task data
  - `VectorArithmeticData`: Vector arithmetic task data
- **Task registry**: Dynamic task lookup and registration

#### Export & Reporting
- **ResultsExporter**: Export results to multiple formats
  - CSV export with all metrics
  - JSON export with complete structured data
  - Markdown export with formatted tables
  - **Per-task export**: Individual JSON and Markdown files for each task
- **ReportGenerator**: Generate comprehensive markdown reports
  - Executive summary with overall scores
  - Detailed per-task breakdowns
  - Performance analysis by category/difficulty
  - Recommendations for improvement
  - Failed example analysis
- **Pandas-based formatting**: Clean table generation using `to_markdown()`

#### Documentation
- **README.md**: Quick start guide and feature overview
- **USAGE.md**: Comprehensive usage guide (NEW)
  - 11 detailed sections
  - Step-by-step examples
  - Complete API reference
  - Troubleshooting guide
  - Best practices
- **Architecture.md**: Technical architecture documentation (UPDATED)
  - Design principles
  - Component descriptions
  - Data flow diagrams
  - Extensibility guide
- **Example scripts**:
  - `example_usage.py`: 6 comprehensive usage examples
  - `test_all_tasks.py`: Test all 4 tasks together
  - `test_config.py`: Test configuration system
  - `test_with_config.py`: Test evaluation with config
  - `test_per_task_export.py`: Test per-task file export
  - `test_postprocessing_all.py`: Test all post-processing features
  - `test_model_comparison.py`: Compare multiple models

#### Package Metadata
- Complete `pyproject.toml` with:
  - Detailed project description
  - Keywords and classifiers
  - Python version support (3.8+)
  - Complete dependency list
  - Optional dev dependencies
  - Build system configuration
  - Tool configurations (pytest, black, ruff, mypy)
- Comprehensive `.gitignore`
- MIT License

### Technical Details

#### Dependencies
- **Core**: pydantic>=2.0, pydantic-settings>=2.0, pyyaml>=6.0
- **ML**: sentence-transformers>=2.0, torch>=2.0, transformers>=4.30
- **Data**: pandas>=2.0, numpy>=1.24, tabulate>=0.9

#### Python Support
- Python 3.8+
- Tested on Python 3.11

#### Performance
- Auto device detection for optimal performance
- Batch processing for efficient encoding
- Model caching to avoid re-downloads

### Data Format

Sample test data included (`data/test_data.json`) with:
- 35 documents, 12 queries for Information Retrieval
- 40 triplets for Semantic Similarity
- 30 test cases for Linguistic Robustness (10 morphology, 10 typo, 10 negation)
- 20 analogies for Vector Arithmetic

---

## [0.0.1] - 2025-11-07

### Initial Development
- Project structure setup
- Basic task implementation (Information Retrieval only)
- Initial documentation

---

## Release Notes

### Version 0.1.0 Highlights

This is the first official release of SemEval, a comprehensive evaluation framework for semantic embeddings and NLP models, with a focus on Turkish language.

**Key Features:**
- ✅ **4 Evaluation Tasks**: Complete implementation of IR, Semantic Similarity, Linguistic Robustness, and Vector Arithmetic
- ⚙️ **Flexible Configuration**: YAML + environment variables with full type safety
- 📊 **Rich Export Options**: JSON, CSV, Markdown, per-task files, comprehensive reports
- 🔌 **Pluggable Architecture**: Easy to extend with custom encoders, tasks, and loaders
- 📝 **Comprehensive Documentation**: README, USAGE guide, Architecture docs, 7 example scripts

**What's Working:**
- All 4 tasks tested and working
- Configuration system fully functional
- Export and reporting tested
- Example scripts all passing
- Turkish test data included

**What's Next (v0.2.0):**
- CLI interface for command-line usage
- Parallel task execution for faster evaluation
- Additional encoder support
- More export formats

**Getting Started:**
```bash
pip install -e .
python scripts/example_usage.py
```

See [USAGE.md](USAGE.md) for detailed documentation.

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
