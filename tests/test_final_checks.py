# tests/test_final_checks.py
import cupy as cp
import numpy as np
import pytest
import scipy.linalg
import warnings
from ssblast import solve
from ssblast.solver import TRITON_AVAILABLE
from ssblast.detector import GPUDetector
from ssblast.precision import PrecisionSelector


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_fp8_path_is_active():
    print(f"\nTriton available: {TRITON_AVAILABLE}")
    assert TRITON_AVAILABLE, "Triton not active!"
    print("FP8 path active")


@pytest.mark.skipif(not TRITON_AVAILABLE, reason="Triton not installed")
def test_dispatcher_routes_to_fp8():
    config = GPUDetector().detect()
    plan   = PrecisionSelector(config).select()
    assert config["tier"] == "FP8"
    assert plan["use_triton"] == True
    print(f"\nTier: {config['tier']}")
    print(f"Triton: {plan['use_triton']}")


def test_accuracy_stable_10_runs():
    cp.random.seed(99)
    n      = 2000
    A      = cp.random.randn(n, n)
    b      = cp.random.randn(n)
    x_true = scipy.linalg.solve(A.get(), b.get())

    errors = []
    for i in range(10):
        x    = solve(A, b)
        diff = float(np.max(np.abs(x.get() - x_true)))
        errors.append(diff)

    print(f"\nMax error across 10 runs: {max(errors):.2e}")
    print(f"Min error across 10 runs: {min(errors):.2e}")
    assert max(errors) < 1e-6
    print("Accuracy stable")


def test_ill_conditioned_matrix():
    """Nearly singular matrix - refinement must handle gracefully"""
    n   = 500
    A   = cp.eye(n, dtype=cp.float64)
    A[0, 0] = 1e-10
    b   = cp.ones(n, dtype=cp.float64)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        x = solve(A, b)

    assert x is not None
    print("\nIll-conditioned matrix handled")


def test_near_vram_limit():
    n  = 8000
    A  = cp.random.randn(n, n)
    b  = cp.random.randn(n)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        x = solve(A, b)

    assert x is not None
    assert x.shape == (n,)
    print(f"\nNear-VRAM solve OK  shape={x.shape}")


def test_error_message_wrong_shape():
    with pytest.raises(ValueError, match="[Ss]hape"):
        solve(cp.eye(100), cp.ones(50))
    print("\nShape mismatch correctly raises ValueError")


def test_error_message_nan():
    """NaN must raise ValueError — must not silently return garbage."""
    A = cp.eye(100)
    A[0, 0] = cp.nan
    with pytest.raises(ValueError, match="NaN"):
        solve(A, cp.ones(100))
    print("\nNaN in cupy A correctly raises ValueError")


def test_nan_raises_not_silent():
    """
    NaN must raise ValueError immediately, never silently return garbage.
    This was Bug 2 found in v0.1.1 — NaN was not caught before compute.
    """
    A = cp.eye(500)
    A[0, 0] = cp.nan
    with pytest.raises(ValueError, match="NaN"):
        solve(A, cp.ones(500))
    print("\nNaN correctly caught before compute")


def test_inf_in_A_raises():
    A = cp.eye(100)
    A[5, 5] = cp.inf
    with pytest.raises(ValueError, match="Inf"):
        solve(A, cp.ones(100))
    print("\nInf in cupy A correctly raises ValueError")


def test_inf_in_b_raises():
    A = cp.eye(100)
    b = cp.ones(100)
    b[3] = -cp.inf
    with pytest.raises(ValueError, match="Inf"):
        solve(A, b)
    print("\nInf in cupy b correctly raises ValueError")


def test_b_2d_raises():
    A = cp.eye(100)
    b = cp.ones((100, 1))
    with pytest.raises(ValueError, match="1D"):
        solve(A, b)
    print("\n2D b correctly raises ValueError")


def test_error_message_non_square():
    with pytest.raises(ValueError, match="square"):
        solve(cp.ones((100, 50)), cp.ones(100))
    print("\nNon-square correctly raises ValueError")


def test_numpy_input_works():
    """User passes numpy — should auto convert"""
    n    = 500
    A_np = np.eye(n)
    b_np = np.ones(n)
    x    = solve(A_np, b_np)
    assert x is not None
    print("\nnumpy auto-convert works")


def test_output_always_fp64():
    A = cp.random.randn(500, 500)
    b = cp.random.randn(500)
    x = solve(A, b)
    assert x.dtype == cp.float64
    print(f"\nOutput dtype: {x.dtype}")
