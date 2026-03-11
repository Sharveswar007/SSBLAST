# ssblast/solver.py
# Layer 0 — Entry Point
# Validates input and routes to correct GPU path

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    cp = None
    CUPY_AVAILABLE = False

try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    triton = None
    TRITON_AVAILABLE = False


def solve(A, b):
    """Entry point. Routes the linear system Ax=b to the correct backend."""
    if A is None or b is None:
        raise ValueError("A and b must not be None")
    if CUPY_AVAILABLE:
        return _solve_gpu(A, b)
    return _solve_cpu(A, b)


def _solve_gpu(A, b):
    A_gpu = cp.asarray(A)
    b_gpu = cp.asarray(b)
    return cp.linalg.solve(A_gpu, b_gpu)


def _solve_cpu(A, b):
    import numpy as np
    return np.linalg.solve(A, b)
