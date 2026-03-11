"""Top-level package for superconductivity.

The curated API is loaded lazily so that optional dependencies such as JAX do
not prevent importing unrelated subpackages like ``superconductivity.visuals``.
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__all__: list[str] = []


def __getattr__(name: str) -> Any:
    """Lazily forward package attributes to ``superconductivity.api``."""
    from . import api as _api

    if hasattr(_api, name):
        return getattr(_api, name)
    raise AttributeError(f"module 'superconductivity' has no attribute {name!r}")


def __dir__() -> list[str]:
    """Expose lazy API attributes in ``dir(superconductivity)`` when possible."""
    try:
        from . import api as _api
    except ModuleNotFoundError:
        return sorted(globals().keys())

    return sorted(set(globals().keys()) | set(_api.__all__))
