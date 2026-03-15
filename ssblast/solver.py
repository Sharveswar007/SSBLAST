# ssblast/solver.py
# Layer 0 — Entry Point
# Validates input and routes to correct GPU path

import numpy as np

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


def _validate(A, b):
    """Comprehensive input validation. Raises ValueError with clear messages."""

    # --- None check ---
    if A is None or b is None:
        raise ValueError("A and b must not be None")

    # --- Must be array-like (have ndim attribute) ---
    if not hasattr(A, 'ndim') or not hasattr(b, 'ndim'):
        raise TypeError(
            "A and b must be numpy or cupy arrays. "
            f"Got A={type(A).__name__}, b={type(b).__name__}"
        )

    # --- A must be 2D ---
    if A.ndim != 2:
        raise ValueError(
            f"A must be a 2D matrix, got {A.ndim}D array with shape {A.shape}"
        )

    # --- A must be square ---
    if A.shape[0] != A.shape[1]:
        raise ValueError(
            f"A must be square, got shape {A.shape}"
        )

    # --- b must be 1D ---
    if b.ndim != 1:
        raise ValueError(
            f"b must be a 1D vector, got {b.ndim}D array with shape {b.shape}"
        )

    # --- Shapes must be compatible ---
    if A.shape[0] != b.shape[0]:
        raise ValueError(
            f"Shape mismatch: A is {A.shape} but b has {b.shape[0]} elements. "
            f"A must be (n, n) and b must be (n,)"
        )

    # --- Empty matrix check ---
    if A.shape[0] == 0:
        raise ValueError("A and b must not be empty (size 0)")

    # --- NaN and Inf checks ---
    # Use cupy for cupy arrays, numpy for numpy arrays
    xp = cp if (CUPY_AVAILABLE and hasattr(A, 'get')) else np

    if bool(xp.any(xp.isnan(A))):
        raise ValueError("A contains NaN values. Check your input matrix.")

    if bool(xp.any(xp.isnan(b))):
        raise ValueError("b contains NaN values. Check your input vector.")

    if bool(xp.any(xp.isinf(A))):
        raise ValueError("A contains Inf values. Check your input matrix.")

    if bool(xp.any(xp.isinf(b))):
        raise ValueError("b contains Inf values. Check your input vector.")


def solve(A, b):
    """Entry point. Routes the linear system Ax=b to the correct backend."""
    _validate(A, b)
    if CUPY_AVAILABLE:
        return _solve_gpu(A, b)
    return _solve_cpu(A, b)


def _solve_gpu(A, b):
    from .detector import GPUDetector
    from .precision import PrecisionSelector
    from .dispatcher import Dispatcher

    A_gpu = cp.asarray(A)
    b_gpu = cp.asarray(b)

    config = GPUDetector().detect()
    plan   = PrecisionSelector(config).select()
    return Dispatcher(config, plan).dispatch(A_gpu, b_gpu)


def _solve_cpu(A, b):
    return np.linalg.solve(A, b)
