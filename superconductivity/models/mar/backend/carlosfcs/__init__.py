"""Python-facing wrapper for the compiled FCS MAR backend."""

from __future__ import annotations

from types import SimpleNamespace

try:
    from . import _fcs_backend as _backend
except ImportError:  # pragma: no cover - build dependent.
    _backend = None
    fcs_api = SimpleNamespace(
        fcs_curve=None,
        fcs_prepare_curve_state=None,
        fcs_prepare_voltage_state=None,
    )
    fcs_curve = None
else:
    fcs_api = _backend.fcs_api
    fcs_curve = fcs_api.fcs_curve

__all__ = [
    "fcs_api",
    "fcs_curve",
]
