"""Trace metadata types."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TraceMeta:
    """Metadata that identifies one trace within one collection.

    Parameters
    ----------
    specific_key : str
        Exact measurement-specific key in the HDF5 file.
    index : int | None
        Positional index of the trace within the current ordered collection.
    yvalue : float | None
        Parsed scalar value associated with the specific key.
    """

    specific_key: str
    index: int | None
    yvalue: float | None


__all__ = ["TraceMeta"]
