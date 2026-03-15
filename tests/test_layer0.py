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
    assert isinstance(CUPY_AVAILABLE, bool)


def test_triton_available_flag():
    assert isinstance(TRITON_AVAILABLE, bool)


# ----- Validation: NaN and Inf -----

def test_nan_in_A_raises():
    A = np.eye(10)
    A[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        solve(A, np.ones(10))


def test_nan_in_b_raises():
    A = np.eye(10)
    b = np.ones(10)
    b[3] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        solve(A, b)


def test_inf_in_A_raises():
    A = np.eye(10)
    A[2, 2] = np.inf
    with pytest.raises(ValueError, match="Inf"):
        solve(A, np.ones(10))


def test_inf_in_b_raises():
    A = np.eye(10)
    b = np.ones(10)
    b[5] = -np.inf
    with pytest.raises(ValueError, match="Inf"):
        solve(A, b)


# ----- Validation: Shape -----

def test_non_square_A_raises():
    A = np.ones((10, 5))
    b = np.ones(10)
    with pytest.raises(ValueError, match="square"):
        solve(A, b)


def test_shape_mismatch_raises():
    A = np.eye(10)
    b = np.ones(5)
    with pytest.raises(ValueError, match="[Ss]hape"):
        solve(A, b)


def test_A_not_2d_raises():
    A = np.ones((10,))
    b = np.ones(10)
    with pytest.raises(ValueError, match="2D"):
        solve(A, b)


def test_A_3d_raises():
    A = np.ones((5, 5, 5))
    b = np.ones(5)
    with pytest.raises(ValueError, match="2D"):
        solve(A, b)


def test_b_not_1d_raises():
    A = np.eye(10)
    b = np.ones((10, 1))
    with pytest.raises(ValueError, match="1D"):
        solve(A, b)


def test_empty_matrix_raises():
    A = np.empty((0, 0))
    b = np.empty((0,))
    with pytest.raises(ValueError, match="empty"):
        solve(A, b)


# ----- Validation: Type -----

def test_non_array_A_raises():
    with pytest.raises((ValueError, TypeError)):
        solve("not_an_array", np.ones(10))


def test_scalar_inputs_raise():
    with pytest.raises((ValueError, TypeError)):
        solve(1.0, 1.0)

