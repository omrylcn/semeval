"""Pytest configuration and fixtures."""

# Note: PyTorch 2.9 has a known issue with pytest causing:
# RuntimeError: function '_has_torch_function' already has a docstring
#
# Workaround: Run tests in separate processes or upgrade PyTorch
# For now, we run encoder tests separately from other tests
