"""Public BCS model entry points."""

from __future__ import annotations

from .bcs import Backend, Kernel, get_Ibcs_nA


def get_I_pat_nA(*args, **kwargs):
    """Lazily import the PAT helper so BCS-only imports stay lightweight."""
    from .backend.pat import get_I_pat_nA as _get_I_pat_nA

    return _get_I_pat_nA(*args, **kwargs)

__all__ = [
    "Backend",
    "Kernel",
    "get_I_pat_nA",
    "get_Ibcs_nA",
]
