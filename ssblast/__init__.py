# ssblast/__init__.py
# Public API — all a user needs to import

from .solver import solve, CUPY_AVAILABLE, TRITON_AVAILABLE

__version__ = "0.1.2"
__author__  = "SHARVESWAR MADASAMY"

__all__ = ["solve", "CUPY_AVAILABLE", "TRITON_AVAILABLE"]
