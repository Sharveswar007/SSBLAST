# tests/test_layer3.py
import cupy as cp
from ssblast.detector import GPUDetector
from ssblast.precision import PrecisionSelector
from ssblast.dispatcher import Dispatcher


def get_dispatcher():
    config = GPUDetector().detect()
    plan   = PrecisionSelector(config).select()
    return Dispatcher(config, plan)


def test_dispatcher_created():
    d = get_dispatcher()
    assert d is not None
    print("\nDispatcher created")


def test_fp16_path_correct():
    """
    Test FP16 path gives correct answer
    Use identity matrix — answer should = b
    """
    d    = get_dispatcher()
    n    = 500
    A    = cp.eye(n, dtype=cp.float64)
    b    = cp.ones(n, dtype=cp.float64)

    x    = d._fp16_path(A, b)
    diff = float(cp.max(cp.abs(x - b)))

    print(f"\nFP16 path max error: {diff:.2e}")
    assert diff < 1e-3, f"FP16 error too large: {diff}"
    print("FP16 path correct")


def test_fp32_path_correct():
    d    = get_dispatcher()
    n    = 500
    A    = cp.eye(n, dtype=cp.float64)
    b    = cp.ones(n, dtype=cp.float64)

    x    = d._fp32_path(A, b)
    diff = float(cp.max(cp.abs(x - b)))

    print(f"\nFP32 path max error: {diff:.2e}")
    assert diff < 1e-4
    print("FP32 path correct")


def test_fallback_path_correct():
    d    = get_dispatcher()
    n    = 100
    A    = cp.eye(n, dtype=cp.float64)
    b    = cp.random.randn(n)

    x    = d._fallback_path(A, b)
    diff = float(cp.max(cp.abs(x - b)))

    print(f"\nFallback path max error: {diff:.2e}")
    assert diff < 1e-10
    print("Fallback path perfect")


def test_memory_check_runs():
    d = get_dispatcher()
    A = cp.eye(100, dtype=cp.float64)
    d._check_memory(A)
    print("\nMemory check ran fine")
