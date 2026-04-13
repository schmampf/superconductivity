"""Shared BCS-style thermal and spectral helper functions."""

from .np import get_Delta_meV, get_T_c_K, get_dos, get_f

__all__ = [
    "get_T_c_K",
    "get_Delta_meV",
    "get_f",
    "get_dos",
]
