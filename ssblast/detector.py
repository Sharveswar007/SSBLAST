# ssblast/detector.py
# Layer 1 — GPU Detector
# Reads GPU hardware properties
# Returns config dict for all other layers

import cupy as cp


class GPUDetector:

    def detect(self):
        """
        Detect GPU and return config dict
        Called once at start of every solve()
        """
        try:
            device = cp.cuda.Device(0)
            props  = cp.cuda.runtime.getDeviceProperties(0)

            major = props["major"]
            minor = props["minor"]
            cc    = float(f"{major}.{minor}")

            shared_mem = props["sharedMemPerBlock"]
            vram_bytes = device.mem_info[1]
            vram_gb    = round(vram_bytes / 1e9, 1)
            name       = props["name"].decode()

            return self._classify(cc, shared_mem, vram_gb, name)

        except Exception as e:
            return self._fallback_config(str(e))

    def _classify(self, cc, shared_mem, vram_gb, name):
        """Map compute capability to tier"""

        # RTX 40xx — Ada Lovelace — FP8
        if cc >= 8.9:
            return {
                "tier":       "FP8",
                "cc":         cc,
                "tile_size":  128,
                "shared_mem": shared_mem,
                "vram_gb":    vram_gb,
                "gpu_name":   name,
            }

        # RTX 30xx — Ampere
        elif cc >= 8.0:
            return {
                "tier":       "FP16",
                "cc":         cc,
                "tile_size":  64,
                "shared_mem": shared_mem,
                "vram_gb":    vram_gb,
                "gpu_name":   name,
            }

        # RTX 20xx — Turing
        elif cc >= 7.0:
            return {
                "tier":       "FP16",
                "cc":         cc,
                "tile_size":  64,
                "shared_mem": shared_mem,
                "vram_gb":    vram_gb,
                "gpu_name":   name,
            }

        # GTX 10xx — Pascal
        elif cc >= 6.0:
            return {
                "tier":       "FP32",
                "cc":         cc,
                "tile_size":  32,
                "shared_mem": shared_mem,
                "vram_gb":    vram_gb,
                "gpu_name":   name,
            }

        else:
            return self._fallback_config("GPU too old")

    def _fallback_config(self, reason):
        """Safe config when detection fails"""
        return {
            "tier":       "FP32",
            "cc":         0.0,
            "tile_size":  16,
            "shared_mem": 49152,
            "vram_gb":    0.0,
            "gpu_name":   f"Unknown ({reason})",
        }
