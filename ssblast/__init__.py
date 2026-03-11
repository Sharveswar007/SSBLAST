# ssblast/__init__.py
# Public API — all a user needs to import

from .solver import solve, CUPY_AVAILABLE, TRITON_AVAILABLE

__version__ = "0.1.0"
__author__  = "Sharvesh"

__all__ = ["solve"]
