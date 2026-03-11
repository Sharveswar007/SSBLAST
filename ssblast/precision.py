# ssblast/precision.py
# Layer 2 — Precision Selector
# Reads tier from Layer 1
# Returns exact dtype plan for all layers

import cupy as cp


class PrecisionSelector:

    def __init__(self, config):
        self.config = config

    def select(self):
        """
        Returns precision plan dict
        based on GPU tier from Layer 1
        """
        tier = self.config["tier"]

        if tier == "FP8":
            return self._fp8_plan()
        elif tier == "FP16":
            return self._fp16_plan()
        elif tier == "FP32":
            return self._fp32_plan()
        else:
            return self._fallback_plan()

    def _fp8_plan(self):
        """RTX 40xx — best path"""
        return {
            "tier":          "FP8",
            "store_dtype":   cp.float16,
            "compute_dtype": cp.float16,
            "accum_dtype":   cp.float32,
            "output_dtype":  cp.float64,
            "needs_scaling": True,
            "scale_block":   32,
            "use_triton":    True,
        }

    def _fp16_plan(self):
        """RTX 20xx/30xx"""
        return {
            "tier":          "FP16",
            "store_dtype":   cp.float16,
            "compute_dtype": cp.float16,
            "accum_dtype":   cp.float32,
            "output_dtype":  cp.float64,
            "needs_scaling": False,
            "scale_block":   None,
            "use_triton":    False,
        }

    def _fp32_plan(self):
        """GTX 10xx"""
        return {
            "tier":          "FP32",
            "store_dtype":   cp.float32,
            "compute_dtype": cp.float32,
            "accum_dtype":   cp.float32,
            "output_dtype":  cp.float64,
            "needs_scaling": False,
            "scale_block":   None,
            "use_triton":    False,
        }

    def _fallback_plan(self):
        """Very old GPU or unknown"""
        return {
            "tier":          "FALLBACK",
            "store_dtype":   cp.float32,
            "compute_dtype": cp.float32,
            "accum_dtype":   cp.float64,
            "output_dtype":  cp.float64,
            "needs_scaling": False,
            "scale_block":   None,
            "use_triton":    False,
        }
