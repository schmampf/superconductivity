"""Python-facing wrapper for the compiled asymmetric HA MAR backend."""

from __future__ import annotations

from types import SimpleNamespace

try:
    from . import _ha_asym_backend as _backend
except ImportError:  # pragma: no cover - build dependent.
    _backend = None
    ha_asym_api = SimpleNamespace(
        ha_asym_curve=None,
        ha_prepare_curve_state=None,
        ha_prepare_voltage_state=None,
    )
    ha_asym_curve = None
else:
    ha_asym_api = _backend.ha_asym_api
    ha_asym_curve = ha_asym_api.ha_asym_curve

__all__ = [
    "ha_asym_api",
    "ha_asym_curve",
]
