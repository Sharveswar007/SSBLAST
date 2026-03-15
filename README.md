# ssBlast

**FP8 per-tile scaled linear solver for consumer NVIDIA GPUs**

ssBlast solves dense linear systems of the form `Ax = b` using FP8-precision matrix kernels
with per-tile dynamic scaling, followed by iterative refinement to recover full FP64 accuracy.
It is designed to run on any NVIDIA GPU from the GTX 10-series through the RTX 40-series,
with no proprietary solvers or data-center hardware required.

Version: 0.1.3
License: MIT
PyPI: https://pypi.org/project/ssblast
Repository: https://github.com/Sharveswar007/SSBLAST

---

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Full API Reference](#full-api-reference)
6. [Platform Notes](#platform-notes)
7. [GPU Compatibility](#gpu-compatibility)
8. [Benchmark Results](#benchmark-results)
9. [How It Works](#how-it-works)
10. [Accuracy](#accuracy)
11. [Limitations](#limitations)
12. [Running the Tests](#running-the-tests)
13. [Technical Notes](#technical-notes)
14. [References](#references)
15. [Author](#author)
16. [License](#license)

---

## Overview

ssBlast is the first open-source FP8 linear solver designed for consumer NVIDIA GPUs.

Standard solvers such as cuSOLVER and MAGMA operate at FP64 precision throughout.
ssBlast performs the expensive dense matrix-matrix multiply (GEMM) in FP8 using a
Triton kernel where each 32x32 tile independently computes and applies its own scale
factor. This avoids the global clipping that would otherwise destroy precision at FP8
range. The result is then corrected to full FP64 accuracy through iterative refinement
using GPU-cached LU factorization.

Measured performance on an RTX 4050 Laptop GPU (Ada Lovelace) under WSL2 with
Triton 3.6.0 shows 2x to 3x speedup over CuPy FP64 for matrix sizes of 2000 or
larger, with maximum residual error below 1e-8.

---

## Requirements

### Minimum

- Python 3.10 or later
- NVIDIA GPU with CUDA compute capability 7.0 or greater (GTX 10-series and newer)
- CUDA Toolkit 12.x
- `cupy-cuda12x >= 13.0`
- `scipy >= 1.11`
- `numpy >= 1.24`

### For FP8 Triton Path (optional, recommended)

- Linux or WSL2 (Windows Subsystem for Linux 2)
- `triton >= 3.0.0`
- `torch >= 2.0`
- RTX 40-series GPU (compute capability 8.9) for maximum performance

> **Note:** Triton is not supported on native Windows. On native Windows the solver
> automatically falls back to the FP16 CuPy cuBLAS path, which still outperforms
> SciPy CPU for large matrices.

---

## Installation

### Standard Installation (CPU fallback included)

```bash
pip install ssblast
```

This installs the core package with GPU support via CuPy and a CPU fallback via NumPy.
No Triton is required. The FP8 kernel path will not be active, but the FP16 and FP32
GPU paths remain available.

### Full Installation with FP8 Support

```bash
pip install "ssblast[triton]"
```

This additionally installs Triton and PyTorch, enabling the FP8 Triton kernel on
Linux and WSL2 with a supported NVIDIA GPU.

### Installing from Source

```bash
git clone https://github.com/Sharveswar007/SSBLAST.git
cd SSBLAST
pip install .
```

To include Triton dependencies from source:

```bash
pip install ".[triton]"
```

### Verifying the Installation

```python
import ssblast
print(ssblast.__version__)        # 0.1.3
print(ssblast.CUPY_AVAILABLE)     # True if CuPy is installed and a GPU is detected
print(ssblast.TRITON_AVAILABLE)   # True if Triton is installed (Linux/WSL2 only)
```

---

## Quick Start

```python
from ssblast import solve
import numpy as np

# Create a random dense linear system
n = 4000
A = np.random.randn(n, n)
b = np.random.randn(n)

# Solve Ax = b
x = solve(A, b)

# Verify accuracy
residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
print(f"Relative residual: {residual:.2e}")  # typically < 1e-8
```

You can also pass CuPy arrays directly, which avoids the CPU-to-GPU transfer:

```python
from ssblast import solve
import cupy as cp

A = cp.random.randn(4000, 4000)
b = cp.random.randn(4000)

x = solve(A, b)
```

The return value `x` is a CuPy array when a GPU is used, or a NumPy array on
CPU-only systems.

---

## Full API Reference

### `ssblast.solve(A, b)`

Solve the dense linear system `Ax = b`.

**Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | numpy.ndarray or cupy.ndarray | Square coefficient matrix of shape (n, n). Must be non-singular. |
| `b` | numpy.ndarray or cupy.ndarray | Right-hand side vector of shape (n,). |

**Returns**

| Name | Type | Description |
|------|------|-------------|
| `x` | numpy.ndarray or cupy.ndarray | Solution vector of shape (n,). FP64-accurate. |

**Raises**

| Exception | Condition |
|-----------|-----------|
| `ValueError` | If `A` or `b` is `None`. |
| `ValueError` | If `A` is not square. |
| `ValueError` | If `A.shape[0]` does not match `b.shape[0]`. |
| `MemoryError` | If the matrix does not fit within available VRAM. |

**Precision Path Selection**

The solver automatically selects the best available path based on the detected GPU:

| GPU Generation | Path Selected |
|----------------|---------------|
| RTX 40-series (CC 8.9) with Triton on Linux/WSL2 | FP8 Triton kernel + iterative refinement |
| RTX 30-series, RTX 20-series | FP16 CuPy cuBLAS + iterative refinement |
| GTX 10-series | FP32 CuPy cuBLAS + iterative refinement |
| No GPU / CuPy unavailable | NumPy CPU solver |

No configuration is required. The selection is fully automatic.

### Module-Level Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `ssblast.__version__` | str | Package version string. |
| `ssblast.CUPY_AVAILABLE` | bool | True if CuPy is installed and importable. |
| `ssblast.TRITON_AVAILABLE` | bool | True if Triton is installed and importable. |

---

## Platform Notes

### Linux and WSL2 (Recommended for Maximum Performance)

The FP8 Triton kernel is available on Linux and WSL2. This is the primary target
platform for peak performance on RTX 40-series GPUs.

To verify that the FP8 path is active:

```python
from ssblast import TRITON_AVAILABLE
print(TRITON_AVAILABLE)  # True on Linux/WSL2 with Triton installed
```

### Windows (Native)

Triton does not support native Windows. On a native Windows installation, ssBlast
automatically uses the FP16 CuPy cuBLAS path for RTX 20-series and newer GPUs.
This path still outperforms SciPy CPU for large matrices but does not reach the
peak speedup of the FP8 kernel.

No code changes are needed. The fallback is transparent to the caller.

### CPU-Only Systems

If no NVIDIA GPU or CuPy installation is detected, ssBlast falls back to
`numpy.linalg.solve`. The same `solve(A, b)` interface is used. Performance
will be equivalent to NumPy's default linear solver.

---

## GPU Compatibility

| GPU Series | Example Models | Compute Capability | Precision Path | Status |
|------------|---------------|-------------------|----------------|--------|
| RTX 40-series | RTX 4090, 4080, 4070, 4060, 4050 | 8.9 | FP8 Triton kernel | Optimized |
| RTX 30-series | RTX 3090, 3080, 3070, 3060 | 8.6 | FP16 cuBLAS | Working |
| RTX 20-series | RTX 2080, 2070, 2060 | 7.5 | FP16 cuBLAS | Working |
| GTX 10-series | GTX 1080, 1070, 1060 | 6.1 / 7.0 | FP32 cuBLAS | Working |
| CPU fallback | Any system | N/A | NumPy | Available |

> **Compute Capability Threshold:** CUDA compute capability 7.0 is the minimum for
> the CuPy GPU path. GTX 1060 has CC 6.1 and will use the CPU fallback unless a
> compatible GPU is also present.

---

## Benchmark Results

Benchmarks were run on an NVIDIA RTX 4050 Laptop GPU (Ada Lovelace, CC 8.9, 6.4 GB VRAM)
under WSL2 Ubuntu with CUDA 12.8 and Triton 3.6.0. Each measurement uses 3 timed runs
with 1 warmup run and explicit CUDA stream synchronization barriers.

### Solve Time by Matrix Size

| Matrix Size | SciPy CPU FP64 | CuPy GPU FP64 | ssBlast FP8 | vs SciPy CPU | vs CuPy FP64 |
|-------------|----------------|---------------|-------------|--------------|--------------|
| 128 x 128 | 0.0002 s | 0.0008 s | 0.0063 s | 0.0x | 0.1x |
| 256 x 256 | 0.0007 s | 0.0024 s | 0.0079 s | 0.1x | 0.3x |
| 500 x 500 | n/a | 0.0073 s | 0.0097 s | n/a | 0.7x |
| 1000 x 1000 | 0.025 s | 0.026 s | 0.020 s | 1.2x | 1.3x |
| 2000 x 2000 | 0.128 s | 0.121 s | 0.050 s | 2.6x | 2.4x |
| 3000 x 3000 | 0.357 s | 0.293 s | 0.103 s | 3.5x | 2.8x |
| 4000 x 4000 | 0.713 s | 0.542 s | 0.188 s | 3.8x | 2.9x |
| 6000 x 6000 | 1.774 s | 1.104 s | 0.546 s | 3.3x | 2.0x |
| 8000 x 8000 | 4.041 s | 2.066 s | 1.021 s | 4.0x | 2.0x |
| 10000 x 10000 | 6.701 s | 4.026 s | 1.920 s | 3.5x | 2.1x |

### Key Performance Observations

- ssBlast outperforms CuPy FP64 for all matrix sizes of n >= 1000.
- Peak speedup of approximately 3x vs CuPy FP64 occurs at n = 3000 to 4000.
- For n < 1000, iterative refinement overhead is not yet amortized by the GEMM savings.
- For n >= 2000, ssBlast is consistently 2x to 3x faster than CuPy FP64.
- All results are FP64-accurate regardless of the FP8 internal representation.

---

## How It Works

The solver is structured as a six-layer pipeline:

```
Layer 0 — Entry Validation
    Accept A (n x n) and b (n,). Validate types and shapes.
    Route to GPU path if CuPy is available, otherwise CPU path.

Layer 1 — GPU Detection
    Query NVIDIA GPU compute capability, VRAM, and driver version.
    Assign a precision tier: FP8, FP16, FP32, or CPU.

Layer 2 — Precision Plan Selection
    Select the compute plan based on the tier and available libraries.
    Check for Triton availability (Linux/WSL2 required for FP8).

Layer 3 — Dispatcher
    Transfer A and b to the GPU if not already on-device.
    Invoke the correct compute backend.

Layer 4 — FP8 Triton GEMM Kernel (RTX 40-series, Linux/WSL2)
    Tile A into configurable blocks (32x32 to 128x128, selected by autotuner).
    For each tile, compute a local scale factor to keep values within FP8 range.
    Perform GEMM using FP16 arithmetic with per-tile FP8-range scaling.
    Rescale output tiles and accumulate the result in FP32 accumulators.

Layer 5 — Iterative Refinement
    Compute the residual r = b - A*x_rough.
    Solve a correction equation using a cached GPU LU factorization.
    Apply the correction to converge to FP64-accurate x.
    Loop until the relative residual is below the tolerance threshold.
```

### Per-Tile Scaling

The key innovation is in the FP8 kernel (`ssblast/kernels/ssblast_kernel.py`).
Standard FP8 implementations apply a single global scale factor to the entire
matrix. When the matrix has significant dynamic range variation, many values
are clipped, which destroys precision.

ssBlast computes a separate scale factor for each 32x32 tile, independent of
all other tiles. Each tile uses the full FP8 value range without clipping.
The scale factors are computed inside the Triton kernel with no CPU round-trip.

This approach was inspired by the GotoBLAS/BLIS per-block memory access
strategy described in the FLAME How-to-Optimize-GEMM guide, adapted for
FP8 dynamic range management.

---

## Accuracy

The FP8 kernel is used only for the initial rough solve. The output of
`ssblast.solve()` is always FP64-accurate.

Accuracy is guaranteed by the iterative refinement stage, which corrects
the FP8-approximate solution using standard FP64 LU factorization. The
refinement converges in one to two iterations for well-conditioned matrices.

Maximum error vs SciPy reference across all tested matrix sizes: < 1e-8.

ssBlast handles the following cases correctly:

- Matrices with large dynamic range variation between entries
- Ill-conditioned matrices (condition number up to 1e12, tested)
- Matrices of non-power-of-two sizes
- Matrices that do not align exactly to tile block boundaries (boundary padding applied)

---

## Limitations

- The FP8 Triton kernel requires Linux or WSL2. Native Windows uses the FP16 path.
- The FP8 path requires an RTX 40-series GPU. Older GPUs use the FP16 or FP32 path.
- Performance advantage only applies for n >= 1000. For small systems, SciPy CPU is faster.
- The entire matrix must fit in VRAM. A 10000 x 10000 FP64 matrix requires approximately 800 MB.
  Consumer GPUs with 6 GB VRAM can handle matrices up to approximately n = 12000.
- Sparse matrices are not supported. Only dense matrices are handled.
- Batch solving (multiple right-hand sides) is not yet implemented. Call `solve` in a loop.
- The iterative refinement tolerance is fixed internally. There is no public parameter to adjust it.

---

## Running the Tests

The test suite requires a CUDA-capable GPU with CuPy installed.

```bash
# Install test dependencies
pip install pytest

# Run the full test suite
pytest tests/ -v
```

Expected output (exact pass counts vary by GPU and platform):

```
tests/test_layer0.py         17 passed
tests/test_layer1.py          8 passed
tests/test_layer2.py          5 passed
tests/test_layer3.py          5 passed
tests/test_layer4.py          5 passed  (skipped if Triton not installed)
tests/test_layer5.py          5 passed
tests/test_end_to_end.py      3 passed
tests/test_final_checks.py   14 passed  (2 skipped if Triton not installed)
```

### Test Coverage

| Test Module | Area Tested |
|-------------|-------------|
| `test_layer0` | Input validation: None, NaN, Inf, non-square, shape mismatch, 1D/3D inputs, empty matrix |
| `test_layer1` | GPU detection, tier assignment, fallback logic |
| `test_layer2` | Precision plan selection for each GPU tier |
| `test_layer3` | Dispatcher routing, CuPy/NumPy array transfer |
| `test_layer4` | FP8 Triton kernel, per-tile scaling, tile alignment (skipped without Triton) |
| `test_layer5` | Iterative refinement convergence, accuracy |
| `test_end_to_end` | Full pipeline accuracy, numpy input handling, large matrix workload |
| `test_final_checks` | NaN/Inf rejection, FP8 path, accuracy over 10 runs, VRAM limit, ill-conditioned matrix |

---

## Technical Notes

### Comparison with Other Solvers

| Solver | FP8 Support | Consumer GPU | Open Source | Typical Speedup vs FP64 |
|--------|-------------|--------------|-------------|-------------------------|
| cuSOLVER | No | Limited (data center focus) | No | 1x baseline |
| MAGMA | No | Limited | Yes | ~1x |
| ssBlast | Yes | Yes (GTX 1060 and newer) | Yes | 2x to 3x |

### Why Not Use Mixed-Precision LAPACK?

Mixed-precision LAPACK (MPGESV) uses FP32 internally for the main factor and
falls back to FP64 if convergence is not reached. It does not use FP8, which has
only been standardized in NVIDIA hardware since the Ada Lovelace generation (2022).
ssBlast targets FP8 specifically to take advantage of the higher FP8 Tensor Core
throughput on RTX 40-series GPUs.

### Reproducibility

The benchmark results in this document correspond to commit `743e1e0` and were
produced on an RTX 4050 Laptop GPU under WSL2 Ubuntu with CUDA 12.8 and
Triton 3.6.0. Results may vary on different hardware, driver versions, or
operating system environments.

---

## References

- FLAME Project, UT Austin. *How to Optimize GEMM*.
  https://github.com/flame/how-to-optimize-gemm/wiki
  GotoBLAS/BLIS memory blocking strategy inspired the per-tile design.

- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms*, 2nd ed. SIAM.
  The iterative refinement algorithm is described in Chapter 12.

- OpenAI Triton open-source GPU programming language.
  https://github.com/openai/triton

- NVIDIA FP8 Formats for Deep Learning.
  https://arxiv.org/abs/2209.05433

---

## Author

**SHARVESWAR MADASAMY**
B.Tech Computer Science and Engineering, SRM Institute of Science and Technology, Kattankulathur
Contact: msharveswar220@gmail.com

---

## License

MIT License. See the LICENSE file in the repository root for the full license text.
