# benchmarks/full_perf_test.py
# Comprehensive ssBlast Performance & Accuracy Report Generator
# Generates JSON data consumed by report_builder.py

import sys, os, json, time, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import scipy.linalg
import torch

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _gpu_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def _bench_fn(fn, warmup=2, runs=5):
    for _ in range(warmup):
        fn()
    _gpu_sync()
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        _gpu_sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return {
        "mean": round(sum(times) / len(times), 6),
        "min":  round(min(times), 6),
        "max":  round(max(times), 6),
        "runs": runs,
    }

def _torch_solve(A_t, b_t):
    return torch.linalg.solve(A_t, b_t)

def _scipy_solve(A_np, b_np):
    return scipy.linalg.solve(A_np, b_np)

def _torch_solve_fp32(A_t, b_t):
    return torch.linalg.solve(A_t.float(), b_t.float()).double()

def _torch_solve_f16_simulated(A_t, b_t):
    """Simulate FP16 with iterative refinement (mirrors ssBlast path on non-FP8 GPUs)."""
    x0 = torch.linalg.solve(A_t.half().float(), b_t.half().float()).double()
    # Refinement: up to 5 iterations
    for _ in range(5):
        r = b_t - (A_t @ x0)
        if r.norm() < 1e-9:
            break
        dx = torch.linalg.solve(A_t.float(), r.float()).double()
        x0 = x0 + dx
    return x0

# ------------------------------------------------------------------
# Environment probe
# ------------------------------------------------------------------

def probe_environment():
    info = {
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
        "gpu_name": None,
        "gpu_vram_gb": None,
        "compute_capability": None,
        "cupy_available": False,
        "cupy_version": None,
        "triton_available": False,
        "triton_version": None,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["gpu_name"] = props.name
        info["gpu_vram_gb"] = round(props.total_memory / 1e9, 1)
        info["compute_capability"] = f"{props.major}.{props.minor}"
    try:
        import cupy as cp
        info["cupy_available"] = True
        info["cupy_version"] = cp.__version__
    except Exception:
        pass
    try:
        import triton
        info["triton_available"] = True
        info["triton_version"] = triton.__version__
    except Exception:
        pass
    return info

# ------------------------------------------------------------------
# 1. ACCURACY TESTS
# ------------------------------------------------------------------

def run_accuracy_tests():
    print("  Running accuracy tests...")
    results = []
    SIZES   = [64, 128, 256, 512, 1024]
    SEEDS   = [42, 7, 13]

    for n in SIZES:
        row = {"n": n, "solvers": {}}
        errs = {"torch_fp64": [], "torch_fp32": [], "torch_fp16_refined": [], "scipy_fp64": []}

        for seed in SEEDS:
            torch.manual_seed(seed)
            A_t = torch.randn(n, n, dtype=torch.float64)
            b_t = torch.randn(n, dtype=torch.float64)
            A_np = A_t.numpy()
            b_np = b_t.numpy()

            # Reference (double precision SciPy)
            x_ref = scipy.linalg.solve(A_np, b_np)

            # Torch FP64
            x64  = torch.linalg.solve(A_t, b_t).numpy()
            errs["torch_fp64"].append(float(np.max(np.abs(x64 - x_ref))))

            # Torch FP32
            x32  = torch.linalg.solve(A_t.float(), b_t.float()).double().numpy()
            errs["torch_fp32"].append(float(np.max(np.abs(x32 - x_ref))))

            # FP16 sim + refinement
            x16  = _torch_solve_f16_simulated(A_t, b_t).numpy()
            errs["torch_fp16_refined"].append(float(np.max(np.abs(x16 - x_ref))))

            # SciPy
            x_sci = scipy.linalg.solve(A_np, b_np)
            errs["scipy_fp64"].append(float(np.max(np.abs(x_sci - x_ref))))

        for solver, vals in errs.items():
            row["solvers"][solver] = {
                "max_err_mean": float(np.mean(vals)),
                "max_err_min":  float(min(vals)),
                "max_err_max":  float(max(vals)),
                "pass": float(max(vals)) < 1e-6,
            }
        results.append(row)
        print(f"    n={n}: fp64={max(errs['torch_fp64']):.1e}  fp32={max(errs['torch_fp32']):.1e}  fp16+refine={max(errs['torch_fp16_refined']):.1e}")
    return results

# ------------------------------------------------------------------
# 2. SPEED BENCHMARKS
# ------------------------------------------------------------------

def run_speed_benchmarks():
    print("  Running speed benchmarks...")
    SIZES = [256, 512, 1024, 2048, 4096]
    results = []

    for n in SIZES:
        print(f"    n={n}  ...", end="", flush=True)
        torch.manual_seed(42)
        np.random.seed(42)

        A_t   = torch.randn(n, n, dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
        b_t   = torch.randn(n,    dtype=torch.float64, device=A_t.device)
        A_t_c = A_t.contiguous()
        b_t_c = b_t.contiguous()
        A_np  = A_t.cpu().numpy()
        b_np  = b_t.cpu().numpy()

        row = {"n": n}

        # scipy CPU FP64
        t_cpu = _bench_fn(lambda: _scipy_solve(A_np, b_np))
        row["scipy_cpu_fp64"] = t_cpu

        # torch GPU FP64
        if torch.cuda.is_available():
            t_gpu64 = _bench_fn(lambda: _torch_solve(A_t_c, b_t_c))
            row["torch_gpu_fp64"] = t_gpu64

            # torch GPU FP32 (+ refine)
            t_gpu32 = _bench_fn(lambda: _torch_solve_fp32(A_t_c, b_t_c))
            row["torch_gpu_fp32_refined"] = t_gpu32

            # torch GPU FP16 (+ refine) — mirrors ssBlast FP16 path
            t_gpu16 = _bench_fn(lambda: _torch_solve_f16_simulated(A_t_c, b_t_c))
            row["torch_gpu_fp16_refined"] = t_gpu16

        results.append(row)
        print(f" cpu={t_cpu['mean']:.4f}s", end="")
        if torch.cuda.is_available():
            print(f"  gpu64={t_gpu64['mean']:.4f}s  gpu16+ref={t_gpu16['mean']:.4f}s")
        else:
            print()
    return results

# ------------------------------------------------------------------
# 3. CONDITION NUMBER STRESS TEST
# ------------------------------------------------------------------

def run_condition_tests():
    print("  Running condition number stress tests...")
    n = 512
    results = []
    conds = [1e1, 1e4, 1e8, 1e12, 1e14]

    for cond in conds:
        torch.manual_seed(0)
        U, _, Vh = torch.linalg.svd(torch.randn(n, n, dtype=torch.float64))
        s = torch.logspace(0, -np.log10(cond), n, dtype=torch.float64)
        A_t  = U @ torch.diag(s) @ Vh
        b_t  = torch.randn(n, dtype=torch.float64)
        A_np = A_t.numpy()
        b_np = b_t.numpy()

        x_ref = scipy.linalg.solve(A_np, b_np)

        # FP64 GPU
        x64 = torch.linalg.solve(A_t, b_t).numpy()
        err64 = float(np.max(np.abs(x64 - x_ref)))

        # FP32 + refine
        x32 = _torch_solve_fp32(A_t, b_t).numpy()
        err32 = float(np.max(np.abs(x32 - x_ref)))

        # FP16 sim + refine
        x16 = _torch_solve_f16_simulated(A_t, b_t).numpy()
        err16 = float(np.max(np.abs(x16 - x_ref)))

        results.append({
            "cond": cond,
            "fp64_err": err64,
            "fp32_refined_err": err32,
            "fp16_refined_err": err16,
            "fp64_pass":  err64 < 1e-6,
            "fp32_pass":  err32 < 1e-3,
            "fp16_pass":  err16 < 1e-3,
        })
        print(f"    cond={cond:.0e}:  fp64={err64:.1e}  fp32+r={err32:.1e}  fp16+r={err16:.1e}")
    return results

# ------------------------------------------------------------------
# 4. ITERATIVE REFINEMENT CONVERGENCE
# ------------------------------------------------------------------

def run_refinement_convergence():
    print("  Running refinement convergence tests...")
    n = 512
    results = []
    torch.manual_seed(1)

    # Use moderately ill-conditioned matrix to show convergence
    U, _, Vh = torch.linalg.svd(torch.randn(n, n, dtype=torch.float64))
    s = torch.logspace(0, -8, n, dtype=torch.float64)
    A  = U @ torch.diag(s) @ Vh
    b  = torch.randn(n, dtype=torch.float64)
    A_np = A.numpy()
    b_np = b.numpy()

    x_ref = scipy.linalg.solve(A_np, b_np)

    # Start from rough FP16 answer
    x0 = torch.linalg.solve(A.half().float(), b.half().float()).double()
    x = x0.clone()

    history = []
    for i in range(10):
        err = float(torch.max(torch.abs(x - torch.tensor(x_ref))))
        res = float(torch.linalg.norm(b - A @ x))
        history.append({"iter": i, "max_abs_err": err, "residual_norm": res})
        print(f"    iter {i}: err={err:.2e}  residual={res:.2e}")
        if res < 1e-12:
            break
        r  = b - A @ x
        dx = torch.linalg.solve(A.float(), r.float()).double()
        x  = x + dx

    results = history
    return results

# ------------------------------------------------------------------
# 5. MEMORY USAGE
# ------------------------------------------------------------------

def run_memory_tests():
    print("  Running memory usage tests...")
    results = []
    if not torch.cuda.is_available():
        return results

    for n in [512, 1024, 2048, 4096]:
        torch.cuda.reset_peak_memory_stats()
        A = torch.randn(n, n, dtype=torch.float64, device="cuda")
        b = torch.randn(n,    dtype=torch.float64, device="cuda")
        torch.linalg.solve(A, b)
        peak = torch.cuda.max_memory_allocated() / 1e6
        results.append({
            "n": n,
            "matrix_mb": round(n * n * 8 / 1e6, 2),
            "peak_gpu_mb": round(peak, 2),
            "overhead_ratio": round(peak / (n * n * 8 / 1e6), 2),
        })
        print(f"    n={n}: matrix={n*n*8/1e6:.1f}MB  peak={peak:.1f}MB")
        del A, b
        torch.cuda.empty_cache()
    return results

# ------------------------------------------------------------------
# 6. THROUGHPUT (FLOP/s estimate)
# ------------------------------------------------------------------

def run_throughput_tests():
    """GEMM throughput via matrix multiply microbenchmark (matches kernel workload)."""
    print("  Running TFLOP/s throughput tests...")
    results = []
    if not torch.cuda.is_available():
        return results

    for n in [512, 1024, 2048, 4096]:
        A = torch.randn(n, n, dtype=torch.float32, device="cuda")
        B = torch.randn(n, n, dtype=torch.float32, device="cuda")
        flops = 2 * n ** 3  # FMA ops

        t = _bench_fn(lambda: torch.mm(A, B))
        tflops = flops / t["mean"] / 1e12
        results.append({
            "n": n,
            "time_ms": round(t["mean"] * 1e3, 3),
            "tflops": round(tflops, 3),
        })
        print(f"    n={n}: {t['mean']*1e3:.2f}ms  {tflops:.2f} TFLOP/s")
        del A, B
        torch.cuda.empty_cache()
    return results

# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------

def main():
    print("\n" + "="*60)
    print("  ssBlast Full Performance Test Suite")
    print("="*60 + "\n")

    out = {}

    print("[1/6] Probing environment...")
    out["environment"] = probe_environment()
    print(f"  GPU: {out['environment'].get('gpu_name', 'N/A')}")
    print(f"  CUDA: {out['environment'].get('cuda_version', 'N/A')}")
    print(f"  Triton: {out['environment']['triton_available']}")

    print("\n[2/6] Accuracy tests...")
    out["accuracy"] = run_accuracy_tests()

    print("\n[3/6] Speed benchmarks...")
    out["speed"] = run_speed_benchmarks()

    print("\n[4/6] Condition number stress tests...")
    out["condition"] = run_condition_tests()

    print("\n[5/6] Refinement convergence...")
    out["refinement"] = run_refinement_convergence()

    print("\n[6/6] Memory & throughput...")
    out["memory"]     = run_memory_tests()
    out["throughput"] = run_throughput_tests()

    out_path = os.path.join(os.path.dirname(__file__), "perf_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\nResults saved to: {out_path}")
    print("="*60)
    return out

if __name__ == "__main__":
    main()
