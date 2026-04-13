"""Python-facing wrapper for the compiled symmetric HA MAR backend."""

from __future__ import annotations

from types import SimpleNamespace

try:
    from . import _ha_sym_backend as _backend
except ImportError:  # pragma: no cover - build dependent.
    _backend = None
    ha_sym_api = SimpleNamespace(ha_sym_curve=None, ha_prepare_state=None)
    ha_sym_curve = None
else:
    ha_sym_api = _backend.ha_sym_api
    ha_sym_curve = ha_sym_api.ha_sym_curve

__all__ = [
    "ha_sym_api",
    "ha_sym_curve",
]
