# ssBlast

**First open-source FP8 linear solver for consumer NVIDIA GPUs**

Solves `Ax = b` using FP8 precision with per-tile scaling,
delivering FP64-accurate results in **2-3× faster time** than CuPy FP64.
Works on any NVIDIA GPU from GTX 1060 to RTX 4090.

## Why ssBlast

| Tool        | FP8 | Consumer GPU | Open Source | Speed |
|-------------|-----|--------------|-------------|-------|
| cuSOLVER    | ❌  | Limited      | ❌          | 1x    |
| MAGMA       | ❌  | Limited      | ✅          | 1x    |
| **ssBlast** | ✅  | ✅           | ✅          | 2-3x  |

## Install

```bash
# Core (CPU fallback available)
pip install ssblast

# With FP8 Triton kernel (Linux/WSL2 + NVIDIA GPU)
pip install "ssblast[triton]"
```

## Usage

```python
from ssblast import solve
import cupy as cp

A = cp.random.randn(4000, 4000)
b = cp.random.randn(4000)

x = solve(A, b)  # FP64-accurate result in 0.19s
# vs CuPy FP64:   0.54s (2.9x slower)
# vs SciPy CPU:   0.71s (3.8x slower)
```

## Benchmark — RTX 4050 Laptop GPU (WSL2, Triton 3.6.0)

| Matrix     | SciPy CPU | CuPy FP64 | ssBlast FP8 | Speedup  |
|------------|-----------|-----------|-------------|----------|
| 1000×1000  | 0.025s    | 0.026s    | 0.020s      | 1.3×     |
| 2000×2000  | 0.128s    | 0.121s    | 0.050s      | **2.4×** |
| 3000×3000  | 0.357s    | 0.293s    | 0.103s      | **2.8×** |
| 4000×4000  | 0.713s    | 0.542s    | 0.188s      | **2.9×** |
| 8000×8000  | 4.041s    | 2.066s    | 1.021s      | **2.0×** |
| 10000×10000| 6.701s    | 4.026s    | 1.920s      | **2.1×** |

**Performance characteristics:**
- Peak speedup **~3× at n=3000-4000** for RTX 40-series GPUs
- Designed for **large systems (n >= 2000)**
- All results **FP64-accurate** (max error < 1e-11)
- Graceful fallback chain: FP8 → FP16 → FP32 → FP64 → CPU

## How It Works

```
solve(A, b)
    ↓
Layer 1: Detect GPU (RTX 4050 → tier FP8)
    ↓
Layer 2: Select precision plan (FP8 + per-tile scaling)
    ↓
Layer 3: Dispatch to correct compute path
    ↓
Layer 4: FP8 Triton GEMM kernel
         Each 32×32 tile computes own scale factor
         Keeps values inside FP8 range ±447
         tl.dot automatically uses Tensor Cores
    ↓
Layer 5: Iterative refinement (GPU LU cached)
         Corrects FP8 rough solve → FP64 accuracy
    ↓
Output: FP64 correct solution x
```

## GPU Support

| GPU       | Tier | Path          | Status |
|-----------|------|---------------|--------|
| RTX 40xx  | FP8  | Triton kernel | ✅ Optimized |
| RTX 30xx  | FP16 | CuPy cuBLAS   | ✅ Working  |
| RTX 20xx  | FP16 | CuPy cuBLAS   | ✅ Working  |
| GTX 10xx  | FP32 | CuPy cuBLAS   | ✅ Working  |
| CPU only  | —    | SciPy fallback| ✅ Available |

## Novel Contribution

**Per-tile FP8 scaling** in `ssblast/kernels/ssblast_kernel.py` (~80 lines)

Each 32×32 tile independently computes its own scale factor.
This means:
- No global clipping (which loses precision)
- Every FP8 region uses the full 0-255 value space
- Computed in-kernel (no CPU overhead)

No equivalent exists in:
- cuSOLVER (proprietary, no FP8)
- MAGMA (no FP8 solver)
- SLATE (CPU-focused)
- Any open-source GPU solver

## Test Results

```bash
pytest tests/   # 43/43 passing
```

**Test coverage:**
- Unit tests: 33/33 pass (layers 0-5)
- Final checks: 10/10 pass (production quality)
  - FP8 Triton path verified active
  - Accuracy stable across 10 runs
  - VRAM limit handling
  - Ill-conditioned matrices
  - Error messages clear

## Requirements

- Python ≥ 3.10
- CUDA 12.x
- NVIDIA GPU with compute capability ≥ 7.0
- `cupy-cuda12x`, `scipy`, `numpy`
- Optional: `triton>=3.0.0`, `torch>=2.0` (for FP8 on Linux/WSL2)

## Limitations

- **Linux/WSL2 only** for FP8 path (Triton requirement)
- Windows: falls back to FP16 path (still 2-3× faster than SciPy CPU)
- Speedup **only for n ≥ 2000** (refinement overhead at small n)
- **Input must fit in VRAM** (max ~6 GB on consumer GPUs)

## References

- [How to Optimize GEMM](https://github.com/flame/how-to-optimize-gemm/wiki) —
  FLAME Project, UT Austin. GotoBLAS/BLIS blocking strategy inspired the
  per-tile design in `ssblast_kernel.py`.
- Higham, N.J. (2002). *Accuracy and Stability of Numerical Algorithms* (2nd ed.). SIAM.
- OpenAI Triton — https://github.com/openai/triton

## Author

**SHARVESWAR MADASAMY** — B.Tech CSE, SRM IST KTR

## License

MIT — See LICENSE file