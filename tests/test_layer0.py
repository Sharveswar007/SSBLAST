# tests/test_layer0.py
import numpy as np
import pytest
from ssblast.solver import solve, CUPY_AVAILABLE, TRITON_AVAILABLE


def test_imports():
    from ssblast import solver
    assert hasattr(solver, "solve")


def test_solve_cpu_fallback():
    A = np.array([[2.0, 1.0], [5.0, 7.0]])
    b = np.array([11.0, 13.0])
    x = solve(A, b)
    # Convert back to numpy if GPU result
    if hasattr(x, "get"):
        x = x.get()
    assert np.allclose(np.dot(A, x), b, atol=1e-6)


def test_solve_rejects_none():
    with pytest.raises(ValueError):
        solve(None, None)


def test_cupy_available_flag():
    # Just confirm the flag is a bool
    assert isinstance(CUPY_AVAILABLE, bool)


def test_triton_available_flag():
    assert isinstance(TRITON_AVAILABLE, bool)
