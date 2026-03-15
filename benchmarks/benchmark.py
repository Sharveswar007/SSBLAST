# benchmarks/benchmark.py
# ssBlast vs scipy CPU vs CuPy FP64
# See exactly how fast RTX 4050 is

import cupy as cp
import numpy as np
import scipy.linalg
import time
from ssblast import solve


def bench(label, fn, warmup=1, runs=3):
    """Run fn multiple times, return avg time"""
    for _ in range(warmup):
        fn()
    cp.cuda.Stream.null.synchronize()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        cp.cuda.Stream.null.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    print(f"  {label:<30} {avg:.4f}s")
    return avg


print("\n" + "=" * 55)
print("  ssBlast Benchmark — RTX 4050 Laptop GPU")
print("=" * 55)

for n in [128, 256, 500, 1000, 2000, 3000, 4000, 6000, 8000, 10000]:
    print(f"\nMatrix size: {n} × {n}")
    print("-" * 55)

    np.random.seed(42)
    A_np = np.random.randn(n, n)
    b_np = np.random.randn(n)
    A_cp = cp.asarray(A_np)
    b_cp = cp.asarray(b_np)

    t_cpu = bench(
        "scipy CPU (FP64)",
        lambda: scipy.linalg.solve(A_np, b_np),
    )

    t_gpu64 = bench(
        "CuPy GPU FP64",
        lambda: cp.linalg.solve(A_cp, b_cp),
    )

    t_ssblast = bench(
        "ssBlast FP8 (ours)",
        lambda: solve(A_cp, b_cp),
    )

    print(f"\n  Speedup vs CPU:       {t_cpu/t_ssblast:.1f}x")
    print(f"  Speedup vs CuPy FP64: {t_gpu64/t_ssblast:.1f}x")

print("\n" + "=" * 55)
print("  Done")
print("=" * 55 + "\n")
