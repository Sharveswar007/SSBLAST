# setup.py
from setuptools import setup, find_packages

setup(
    name             = "ssblast",
    version          = "0.1.0",
    author           = "SHARVESWAR MADASAMY",
    description      = "FP8 linear solver for consumer NVIDIA GPUs",
    packages         = find_packages(),
    python_requires  = ">=3.10",
    install_requires = [
        "cupy-cuda12x",
        "triton",
        "scipy",
        "numpy",
        "torch",
    ],
)
