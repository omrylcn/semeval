.PHONY: help clean clean-pyc clean-build clean-test clean-all install dev test test-all run example lint format sync

# Default target
help:
	@echo "SemEval - Makefile Commands"
	@echo "============================"
	@echo ""
	@echo "Setup:"
	@echo "  make install        - Install package dependencies"
	@echo "  make dev            - Install with dev dependencies"
	@echo "  make sync           - Sync dependencies with uv"
	@echo ""
	@echo "Run:"
	@echo "  make run            - Run all tasks with example data"
	@echo "  make example        - Run example_usage.py (6 examples)"
	@echo "  make test           - Run test_all_tasks.py"
	@echo "  make test-all       - Run all test scripts"
	@echo "  make config         - Test configuration system"
	@echo "  make export         - Test per-task export"
	@echo "  make compare        - Run model comparison"
	@echo ""
	@echo "Clean:"
	@echo "  make clean          - Remove all generated files"
	@echo "  make clean-pyc      - Remove Python cache files"
	@echo "  make clean-build    - Remove build artifacts"
	@echo "  make clean-test     - Remove test artifacts"
	@echo "  make clean-output   - Remove output files"
	@echo "  make clean-all      - Deep clean (everything)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint           - Run ruff linter"
	@echo "  make format         - Format code with black"
	@echo "  make check          - Run all checks (lint + format check)"
	@echo ""

# Installation
install:
	uv sync

dev:
	uv sync --group dev

sync:
	uv sync --all-groups

# Run commands
run:
	uv run python scripts/test_all_tasks.py

example:
	uv run python scripts/example_usage.py

test:
	uv run python scripts/test_all_tasks.py

test-all:
	@echo "Running all test scripts..."
	@echo "1. Testing all tasks..."
	uv run python scripts/test_all_tasks.py
	@echo ""
	@echo "2. Testing configuration..."
	uv run python scripts/test_config.py
	@echo ""
	@echo "3. Testing with config..."
	uv run python scripts/test_with_config.py
	@echo ""
	@echo "4. Testing per-task export..."
	uv run python scripts/test_per_task_export.py
	@echo ""
	@echo "5. Testing post-processing..."
	uv run python scripts/test_postprocessing_all.py
	@echo ""
	@echo "✅ All tests complete!"

config:
	uv run python scripts/test_config.py

export:
	uv run python scripts/test_per_task_export.py

compare:
	uv run python scripts/test_model_comparison.py

# Cleaning
clean-pyc:
	@echo "Removing Python cache files..."
	find . -type f -name '*.py[co]' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -exec rm -rf {} +
	@echo "✅ Python cache cleaned"

clean-build:
	@echo "Removing build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf .eggs/
	rm -rf *.egg-info
	rm -rf *.egg
	@echo "✅ Build artifacts cleaned"

clean-test:
	@echo "Removing test artifacts..."
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	@echo "✅ Test artifacts cleaned"

clean-output:
	@echo "Removing output files..."
	rm -rf output/
	rm -rf logs/
	@echo "✅ Output files cleaned"

clean: clean-pyc clean-build clean-test
	@echo "✅ Clean complete!"

clean-all: clean clean-output
	@echo "Removing all generated files..."
	find . -type f -name '.DS_Store' -delete
	find . -type f -name 'Thumbs.db' -delete
	@echo "✅ Deep clean complete!"

# Code quality
lint:
	@echo "Running ruff linter..."
	uv run ruff check semeval/
	@echo "✅ Linting complete"

format:
	@echo "Formatting code with black..."
	uv run black semeval/ scripts/
	@echo "✅ Formatting complete"

format-check:
	@echo "Checking code format..."
	uv run black --check semeval/ scripts/
	@echo "✅ Format check complete"

check: lint format-check
	@echo "✅ All checks passed"
