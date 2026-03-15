# ssblast/refinement.py
# Layer 5 — Iterative Refinement Engine
#
# Problem: FP8 GEMM gives rough answer x0
# Solution: Keep correcting until FP64 accurate
#
# Algorithm:
#   1. Compute residual  r = b - A @ x0
#   2. If r is tiny → already accurate → stop
#   3. Solve correction  A @ dx = r  (FP32)
#   4. Update solution   x0 = x0 + dx
#   5. Repeat until converged or MAX_ITER hit

import cupy as cp
import warnings


MAX_ITER = 10     # max correction rounds
TOL      = 1e-9   # stop when residual < this


def refine(A, b, x0):
    """
    Iterative refinement with LU reuse — fully on GPU.
    Factorize A once in FP32 — reuse for all corrections.

    A  — original matrix  [M x M] FP64
    b  — right hand side  [M]     FP64
    x0 — rough solution   [M]     any dtype
    """
    from cupyx.scipy.linalg import lu_factor, lu_solve

    A  = A.astype(cp.float64)
    b  = b.astype(cp.float64)
    x0 = x0.astype(cp.float64)

    best_x    = x0.copy()
    best_norm = float("inf")

    # Factorize A ONCE in FP32 on GPU — all corrections reuse same factors
    lu, piv = lu_factor(A.astype(cp.float32))

    for i in range(MAX_ITER):

        # Residual r = b - A @ x0
        r    = b - A @ x0
        norm = float(cp.linalg.norm(r))

        if norm < best_norm:
            best_norm = norm
            best_x    = x0.copy()

        if norm < TOL:
            return x0

        # Correction: cheap triangular solve reusing cached LU
        try:
            dx = lu_solve((lu, piv), r.astype(cp.float32)).astype(cp.float64)
        except Exception as e:
            warnings.warn(f"Correction solve failed: {e}")
            break

        x0 = x0 + dx

    if best_norm > 1e-6:
        warnings.warn(
            f"Refinement did not fully converge. "
            f"Best residual: {best_norm:.2e}. "
            f"Matrix may be ill-conditioned."
        )

    return best_x
