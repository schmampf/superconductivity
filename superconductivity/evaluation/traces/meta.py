"""Trace metadata types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

YValue: TypeAlias = float | str | None


def numeric_yvalue(value: YValue) -> float | None:
    """Return one numeric y-value when available.

    Parameters
    ----------
    value : float | str | None
        Metadata y-value.

    Returns
    -------
    float | None
        Finite numeric value, else ``None``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return numeric


@dataclass(slots=True, frozen=True)
class TraceMeta:
    """Metadata that identifies one trace within one collection.

    Parameters
    ----------
    specific_key : str
        Exact measurement-specific key in the HDF5 file.
    index : int | None
        Positional index of the trace within the current ordered collection.
    yvalue : float | str | None
        Parsed scalar value associated with the specific key. Non-numeric
        fallback strings are allowed when value extraction fails.
    """

    specific_key: str
    index: int | None
    yvalue: YValue


__all__ = ["TraceMeta", "YValue", "numeric_yvalue"]
