"""Python package shim for the compiled simple_knn extension."""

# Ensure PyTorch shared libraries are loaded before importing the CUDA extension.
import torch  # noqa: F401

from ._C import distCUDA2

__all__ = ["distCUDA2"]
