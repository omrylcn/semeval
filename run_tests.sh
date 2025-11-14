#!/bin/bash
# Test runner script to work around PyTorch 2.9 + pytest compatibility issue
# Each test file is run in a separate process to avoid torch import conflicts

set -e

VENV_PYTHON="/Users/omer.yalcin/works/semeval/.venv/bin/python3"

echo "Running tests in separate processes to avoid PyTorch import conflicts..."
echo "=========================================================================="

# Non-torch tests
echo "Running schemas tests..."
$VENV_PYTHON -m pytest tests/test_schemas.py -v

echo ""
echo "Running loader tests..."
$VENV_PYTHON -m pytest tests/test_loaders.py -v

echo ""
echo "Running arithmetic metrics tests..."
$VENV_PYTHON -m pytest tests/test_arithmetic_metrics.py -v

echo ""
echo "Running IR metrics tests..."
$VENV_PYTHON -m pytest tests/test_ir_metrics.py -v

echo ""
echo "Running robustness metrics tests..."
$VENV_PYTHON -m pytest tests/test_robustness_metrics.py -v

echo ""
echo "Running similarity metrics tests..."
$VENV_PYTHON -m pytest tests/test_similarity_metrics.py -v

echo ""
echo "Running encoder tests..."
$VENV_PYTHON -m pytest tests/test_encoders.py -v

echo ""
echo "=========================================================================="
echo "All test suites completed!"
