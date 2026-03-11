# tests/test_layer2.py
import cupy as cp
from ssblast.detector import GPUDetector
from ssblast.precision import PrecisionSelector


def get_plan():
    config = GPUDetector().detect()
    return PrecisionSelector(config).select()


def test_plan_returns_dict():
    plan = get_plan()
    assert isinstance(plan, dict)
    print(f"\nPlan tier: {plan['tier']}")


def test_plan_has_all_keys():
    plan = get_plan()
    required_keys = [
        "tier", "store_dtype", "compute_dtype",
        "accum_dtype", "output_dtype",
        "needs_scaling", "use_triton",
    ]
    for key in required_keys:
        assert key in plan, f"Missing key: {key}"
    print("All keys present")


def test_fp8_needs_scaling():
    plan = get_plan()
    if plan["tier"] == "FP8":
        assert plan["needs_scaling"] is True
        assert plan["scale_block"] == 32
        print("FP8 scaling correctly enabled")


def test_output_always_fp64():
    plan = get_plan()
    assert plan["output_dtype"] == cp.float64
    print("Output is always FP64")


def test_rtx4050_uses_triton():
    plan = get_plan()
    assert plan["use_triton"] is True
    assert plan["tier"] == "FP8"
    print("RTX 4050 uses Triton kernel")
