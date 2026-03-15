# setup.py
from setuptools import setup, find_packages

setup(
    name                           = "ssblast",
    version                        = "0.1.2",
    author                         = "SHARVESWAR MADASAMY",
    author_email                   = "msharveswar220@gmail.com",
    description                    = "FP8 per-tile scaled linear solver for consumer NVIDIA GPUs",
    long_description               = open("README.md", encoding="utf-8").read(),
    long_description_content_type  = "text/markdown",
    url                            = "https://github.com/Sharveswar007/SSBLAST",
    packages                       = find_packages(),
    python_requires                = ">=3.10",
    install_requires               = [
        "cupy-cuda12x>=13.0",
        "scipy>=1.11",
        "numpy>=1.24",
    ],
    extras_require                 = {
        "triton": ["triton>=3.0.0", "torch>=2.0"]
    },
    classifiers                    = [
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "Development Status :: 4 - Beta",
    ],
)
