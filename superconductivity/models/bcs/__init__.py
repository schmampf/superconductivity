"""Public BCS model entry points."""

from __future__ import annotations

from .bcs import Backend, Kernel, get_Ibcs_nA, sim_bcs


def pat_kernel(*args, **kwargs):
    """Lazily import the PAT helper so BCS-only imports stay lightweight."""
    from .backend.pat import pat_kernel as _pat_kernel

    return _pat_kernel(*args, **kwargs)

__all__ = [
    "Backend",
    "Kernel",
    "pat_kernel",
    "get_Ibcs_nA",
    "sim_bcs",
]
