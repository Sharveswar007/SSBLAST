# tests/test_layer4.py
import cupy as cp
import pytest

triton = pytest.importorskip("triton", reason="Triton not installed — skipping FP8 kernel tests")
torch  = pytest.importorskip("torch",  reason="PyTorch not installed — skipping FP8 kernel tests")

from ssblast.detector import GPUDetector
from ssblast.kernels.ssblast_kernel import fp8_gemm


def get_config():
    return GPUDetector().detect()


def test_kernel_runs():
    """Kernel should run without crashing"""
    config = get_config()
    A = cp.eye(128, dtype=cp.float64)
    b = cp.ones(128, dtype=cp.float64)
    x = fp8_gemm(A, b, config)
    assert x is not None
    print("\nKernel ran without crash")


def test_identity_matrix():
    """
    A @ b where A = identity
    Result should be x = b
    """
    config = get_config()
    n    = 256
    A    = cp.eye(n, dtype=cp.float64)
    b    = cp.ones(n, dtype=cp.float64)
    x    = fp8_gemm(A, b, config)
    diff = float(cp.max(cp.abs(x - b)))
    print(f"\nIdentity test error: {diff:.2e}")
    assert diff < 0.01
    print("Identity matrix test PASSED")


def test_vs_cupy_reference():
    """
    Compare FP8 GEMM result to CuPy reference (A @ b)
    Should be close
    """
    config = get_config()
    cp.random.seed(42)
    n      = 256
    A      = cp.random.randn(n, n).astype(cp.float64)
    b      = cp.random.randn(n).astype(cp.float64)

    x_fp8  = fp8_gemm(A, b, config)        # FP8 A @ b
    x_ref  = (A @ b).astype(cp.float64)    # reference A @ b

    diff = float(cp.max(cp.abs(x_fp8 - x_ref)))
    denom = float(cp.max(cp.abs(x_ref)))
    rel  = diff / denom if denom > 0 else diff

    print(f"\nFP8 vs FP64 max diff:     {diff:.2e}")
    print(f"FP8 vs FP64 relative err: {rel:.2e}")
    assert rel < 0.05, f"Relative error too large: {rel:.2e}"
    print("FP8 rough accuracy OK")


def test_output_shape():
    """Output must be 1D vector"""
    config = get_config()
    n = 128
    A = cp.eye(n, dtype=cp.float64)
    b = cp.ones(n, dtype=cp.float64)
    x = fp8_gemm(A, b, config)
    assert x.shape == (n,)
    print(f"\nOutput shape: {x.shape}")


def test_output_is_fp64():
    """Output must be FP64"""
    config = get_config()
    A = cp.eye(64, dtype=cp.float64)
    b = cp.ones(64, dtype=cp.float64)
    x = fp8_gemm(A, b, config)
    assert x.dtype == cp.float64
    print(f"\nOutput dtype: {x.dtype}")
