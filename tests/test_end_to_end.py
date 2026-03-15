# tests/test_end_to_end.py
# Full pipeline test — user calls solve()
import time

import cupy as cp
import numpy as np
import scipy.linalg
from ssblast import solve


def test_full_pipeline_small():
    """Small matrix — full pipeline. User just calls solve(A, b)"""
    cp.random.seed(42)
    n    = 200
    A    = cp.random.randn(n, n)
    b    = cp.random.randn(n)

    x    = solve(A, b)
    x_ref = scipy.linalg.solve(A.get(), b.get())
    diff  = float(np.max(np.abs(x.get() - x_ref)))

    print(f"\nFull pipeline error: {diff:.2e}")
    assert diff < 1e-6
    print("Full pipeline PASSED")


def test_full_pipeline_numpy_input():
    """User passes numpy array — ssBlast auto-converts"""
    np.random.seed(0)
    n    = 200
    A_np = np.random.randn(n, n)
    b_np = np.random.randn(n)

    x    = solve(A_np, b_np)
    x_ref = scipy.linalg.solve(A_np, b_np)
    x_np  = x.get() if hasattr(x, "get") else x
    diff  = float(np.max(np.abs(x_np - x_ref)))

    print(f"\nnumpy input error: {diff:.2e}")
    assert diff < 1e-6
    print("numpy auto-convert PASSED")


def test_full_pipeline_large():
    """Large 1000x1000 matrix — real workload"""
    cp.random.seed(7)
    n  = 1000
    A  = cp.random.randn(n, n)
    b  = cp.random.randn(n)

    t0 = time.perf_counter()
    x  = solve(A, b)
    t1 = time.perf_counter()

    res = float(cp.linalg.norm(b - A @ x))
    print(f"\n1000x1000 residual: {res:.2e}")
    print(f"1000x1000 time:     {t1-t0:.3f}s")
    assert res < 1e-5
    print("Large matrix PASSED")
