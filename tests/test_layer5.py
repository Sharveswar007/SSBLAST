# tests/test_layer5.py
import cupy as cp
import numpy as np
import scipy.linalg
from ssblast.refinement import refine, TOL


def test_refine_identity():
    """
    Identity matrix — perfect answer in 1 iter
    x0 = zeros → refined to ones
    """
    n  = 500
    A  = cp.eye(n, dtype=cp.float64)
    b  = cp.ones(n, dtype=cp.float64)
    x0 = cp.zeros(n, dtype=cp.float64)

    x    = refine(A, b, x0)
    diff = float(cp.max(cp.abs(x - b)))

    print(f"\nIdentity refine error: {diff:.2e}")
    assert diff < TOL
    print("Identity refined to FP64")


def test_refine_random():
    """
    Random matrix — compare to scipy gold standard
    """
    cp.random.seed(0)
    n    = 500
    A_np = np.random.randn(n, n)
    b_np = np.random.randn(n)

    x_true = scipy.linalg.solve(A_np, b_np)

    A  = cp.asarray(A_np)
    b  = cp.asarray(b_np)
    x0 = cp.linalg.solve(
             A.astype(cp.float32),
             b.astype(cp.float32)
         ).astype(cp.float64)

    x    = refine(A, b, x0)
    diff = float(np.max(np.abs(x.get() - x_true)))

    print(f"\nRandom matrix refine error: {diff:.2e}")
    assert diff < 1e-6
    print("Random matrix refined")


def test_refine_improves_fp8_rough():
    """
    Simulate noisy FP8 answer — refinement must fix it
    """
    cp.random.seed(1)
    n  = 500
    A  = cp.eye(n, dtype=cp.float64) * 2
    b  = cp.ones(n, dtype=cp.float64)

    x_rough   = (b / 2) + cp.random.randn(n) * 0.01
    x_refined = refine(A, b, x_rough)
    x_true    = b / 2
    diff      = float(cp.max(cp.abs(x_refined - x_true)))

    print(f"\nFP8 rough → refined error: {diff:.2e}")
    assert diff < 1e-6
    print("FP8 answer refined to FP64 accuracy")


def test_refine_output_fp64():
    """Output must be FP64"""
    n  = 100
    A  = cp.eye(n, dtype=cp.float64)
    b  = cp.ones(n, dtype=cp.float64)
    x0 = cp.zeros(n, dtype=cp.float32)
    x  = refine(A, b, x0)
    assert x.dtype == cp.float64
    print(f"\nOutput dtype: {x.dtype}")


def test_refine_bad_x0_still_works():
    """
    Even with terrible starting point (all zeros)
    refinement should converge
    """
    n  = 200
    A  = cp.eye(n, dtype=cp.float64)
    b  = cp.ones(n, dtype=cp.float64)
    x0 = cp.zeros(n, dtype=cp.float64)

    x    = refine(A, b, x0)
    diff = float(cp.max(cp.abs(x - b)))

    print(f"\nBad start → refined error: {diff:.2e}")
    assert diff < 1e-6
    print("Recovered from bad x0")
