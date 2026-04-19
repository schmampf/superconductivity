"""Shared BCS-style thermal and spectral helper functions."""

from .np import get_DeltaT_meV, get_dos, get_f, get_Tc_K

__all__ = [
    "get_Tc_K",
    "get_DeltaT_meV",
    "get_f",
    "get_dos",
]
