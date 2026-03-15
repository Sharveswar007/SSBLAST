# ssblast/dispatcher.py
# Layer 3 — Dispatcher
# Converts matrix dtype
# Routes to correct compute path
# FP32/FP16 → CuPy @
# FP8       → Triton kernel

import cupy as cp


class Dispatcher:

    def __init__(self, config, plan):
        self.config = config
        self.plan   = plan

    def dispatch(self, A, b):
        """
        Route to correct solver path
        based on precision plan
        """
        tier = self.plan["tier"]

        A = self._check_memory(A)

        if tier == "FP8":
            return self._fp8_path(A, b)
        elif tier == "FP16":
            return self._fp16_path(A, b)
        elif tier == "FP32":
            return self._fp32_path(A, b)
        else:
            return self._fallback_path(A, b)

    # ─────────────────────────────────────
    # FP8 Path — RTX 40xx
    # Calls Triton kernel (Layer 4)
    # ─────────────────────────────────────
    def _fp8_path(self, A, b):
        try:
            from .kernels.ssblast_kernel import fp8_gemm
            x0 = fp8_gemm(A, b, self.config)
            from .refinement import refine
            return refine(A, b, x0)
        except Exception as e:
            import warnings
            warnings.warn(f"FP8 kernel failed ({e}), falling back to FP16")
            return self._fp16_path(A, b)

    # ─────────────────────────────────────
    # FP16 Path — RTX 20xx/30xx
    # Pure CuPy — no Triton needed
    # ─────────────────────────────────────
    def _fp16_path(self, A, b):
        try:
            A16 = A.astype(cp.float16)
            b16 = b.astype(cp.float16)
            x0  = cp.linalg.solve(
                      A16.astype(cp.float32),
                      b16.astype(cp.float32)
                  )
            x0  = x0.astype(cp.float64)
            from .refinement import refine
            return refine(A, b, x0)
        except Exception as e:
            import warnings
            warnings.warn(f"FP16 failed ({e}), falling back to FP32")
            return self._fp32_path(A, b)

    # ─────────────────────────────────────
    # FP32 Path — GTX 10xx
    # Pure CuPy
    # ─────────────────────────────────────
    def _fp32_path(self, A, b):
        try:
            A32 = A.astype(cp.float32)
            b32 = b.astype(cp.float32)
            x0  = cp.linalg.solve(A32, b32)
            x0  = x0.astype(cp.float64)
            from .refinement import refine
            return refine(A, b, x0)
        except Exception as e:
            import warnings
            warnings.warn(f"FP32 failed ({e}), falling back to FP64")
            return self._fallback_path(A, b)

    # ─────────────────────────────────────
    # Fallback Path — pure FP64
    # Last resort before CPU
    # ─────────────────────────────────────
    def _fallback_path(self, A, b):
        import warnings
        warnings.warn("Using FP64 GPU fallback path")
        return cp.linalg.solve(A, b)

    # ─────────────────────────────────────
    # Memory Check
    # ─────────────────────────────────────
    def _check_memory(self, A):
        """
        Check if matrix fits in VRAM
        If not → warn user
        """
        matrix_bytes = A.nbytes
        free_bytes   = cp.cuda.Device(0).mem_info[0]

        if matrix_bytes > free_bytes * 0.8:
            import warnings
            warnings.warn(
                f"Matrix size {matrix_bytes/1e9:.1f}GB "
                f"exceeds 80% of free VRAM "
                f"({free_bytes/1e9:.1f}GB free). "
                f"May run out of memory."
            )
        return A
