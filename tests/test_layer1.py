# tests/test_layer1.py
from ssblast.detector import GPUDetector


def test_detect_returns_dict():
    config = GPUDetector().detect()
    assert isinstance(config, dict)
    print(f"\nGPU detected: {config['gpu_name']}")
    print(f"Tier:         {config['tier']}")
    print(f"CC:           {config['cc']}")
    print(f"VRAM:         {config['vram_gb']} GB")
    print(f"Shared mem:   {config['shared_mem']} bytes")


def test_tier_is_valid():
    config = GPUDetector().detect()
    assert config["tier"] in ["FP8", "FP16", "FP32"]
    print(f"Tier valid: {config['tier']}")


def test_rtx4050_is_fp8():
    config = GPUDetector().detect()
    # RTX 4050 = cc 8.9 = FP8 tier
    assert config["tier"] == "FP8"
    assert config["tile_size"] == 128
    print("RTX 4050 correctly detected as FP8")


def test_tile_size_set():
    config = GPUDetector().detect()
    assert config["tile_size"] in [16, 32, 64, 128]
    print(f"Tile size: {config['tile_size']}")


def test_vram_detected():
    config = GPUDetector().detect()
    assert config["vram_gb"] > 0
    print(f"VRAM: {config['vram_gb']} GB")
