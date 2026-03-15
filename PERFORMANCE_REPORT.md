# ssBlast - Benchmark Results

**Hardware:** NVIDIA RTX 4050 Laptop GPU - Ada Lovelace CC 8.9 - 6.4 GB VRAM - CUDA 12.8
**Path:** FP8 Triton kernel + iterative refinement (WSL2 Ubuntu, Triton 3.6.0)
**Method:** 3 timed runs + 1 warmup, `cp.cuda.Stream.null.synchronize()` barriers

---

## Results

| Matrix Size     | SciPy CPU FP64 | CuPy GPU FP64 | ssBlast FP8  | vs CPU | vs CuPy  |
|-----------------|----------------|---------------|--------------|--------|----------|
| 128 x 128       | 0.0002s        | 0.0008s       | 0.0063s      | 0.0x   | 0.1x     |
| 256 x 256       | 0.0007s        | 0.0024s       | 0.0079s      | 0.1x   | 0.3x     |
| 500 x 500       | n/a            | 0.0073s       | 0.0097s      | n/a    | 0.7x     |
| 1000 x 1000     | 0.0252s        | 0.0263s       | **0.0202s**  | 1.2x   | **1.3x** |
| 2000 x 2000     | 0.1281s        | 0.1214s       | **0.0497s**  | 2.6x   | **2.4x** |
| 3000 x 3000     | 0.3571s        | 0.2930s       | **0.1028s**  | 3.5x   | **2.8x** |
| 4000 x 4000     | 0.7127s        | 0.5422s       | **0.1882s**  | 3.8x   | **2.9x** |
| 6000 x 6000     | 1.7742s        | 1.1043s       | **0.5458s**  | 3.3x   | **2.0x** |
| 8000 x 8000     | 4.0408s        | 2.0659s       | **1.0213s**  | 4.0x   | **2.0x** |
| 10000 x 10000   | 6.7006s        | 4.0255s       | **1.9198s**  | 3.5x   | **2.1x** |

---

## Speedup vs CuPy FP64

`
  n=128   |=                                    0.1x
  n=256   |===                                  0.3x
  n=500   |=======                              0.7x
  n=1000  |=============                        1.3x
  n=2000  |========================             2.4x
  n=3000  |============================         2.8x
  n=4000  |=============================        2.9x  <-- peak
  n=6000  |====================                 2.0x
  n=8000  |====================                 2.0x
  n=10000 |=====================                2.1x
          +-------+-------+-------+
          0x      1x      2x      3x
`

ssBlast wins for n >= 1000. Peak speedup is ~3x faster than CuPy FP64 at n=3000-4000.

Small n (< 1000) is slower because the LU factorization overhead in iterative refinement
is not yet amortised by the GEMM. ssBlast is designed for large linear systems.

---

## Accuracy

All ssBlast results are FP64-accurate (max error < 1e-11 vs SciPy reference).
FP8 is used only internally for the initial rough solve. Output is always FP64.

---

## Notes

- Benchmarks run from WSL2 Ubuntu with Triton 3.6.0 (FP8 kernel active)
- Running from Windows PowerShell falls back to FP16 path (Triton is Linux-only)
- SciPy n=500 excluded due to OS scheduling spike during that run
- All tests: pytest tests/ -> 33/33 passing