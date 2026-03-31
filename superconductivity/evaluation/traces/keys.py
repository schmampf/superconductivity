"""Helpers for measurement and specific-key handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ...utilities.types import NDArray64
from .file import FileSpec, _require_measurement, list_specific_keys
from .meta import TraceMeta


@dataclass(slots=True)
class KeysSpec:
    """Configuration for parsing and selecting specific keys.

    Parameters
    ----------
    strip0 : str, default="="
        Start delimiter for value parsing.
    strip1 : str | None, default=None
        Optional end delimiter for value parsing.
    remove_key : str or sequence of str, optional
        Exact specific-key names to remove before parsing and sorting.
    add_key : tuple or sequence of tuple, optional
        Exact specific-key additions as ``(key, value)`` or
        ``[(key, value), ...]``.
    norm : float | None, optional
        Optional finite positive normalization factor applied to the parsed
        y-values.
    label : str | None, optional
        Plain-text label for the parsed values.
    html_label : str | None, optional
        HTML-formatted label for the parsed values.
    limits : float, slice, tuple, list, or None, optional
        Optional selection applied after sorting.
    """

    strip0: str = "="
    strip1: str | None = None
    remove_key: str | Sequence[str] | None = None
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None
    norm: float | None = None
    label: str | None = None
    html_label: str | None = None
    limits: float | slice | tuple[int | float | None, int | float | None] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate key-parsing settings."""
        if not isinstance(self.strip0, str) or self.strip0 == "":
            raise ValueError("strip0 must be a non-empty string.")
        if self.strip1 is not None and not isinstance(self.strip1, str):
            raise ValueError("strip1 must be a string or None.")
        if self.norm is not None:
            self.norm = float(self.norm)
            if not np.isfinite(self.norm) or self.norm <= 0.0:
                raise ValueError("norm must be finite and > 0.")
        if self.label is not None and not isinstance(self.label, str):
            raise ValueError("label must be a string or None.")
        if self.html_label is not None and not isinstance(self.html_label, str):
            raise ValueError("html_label must be a string or None.")

        self.remove_key = tuple(
            _normalize_removed_specific_keys(self.remove_key),
        )
        self.add_key = tuple(
            _normalize_added_specific_keys(self.add_key),
        )
        if isinstance(self.limits, list):
            if len(self.limits) != 2:
                raise ValueError(
                    "limits lists must contain exactly two entries.",
                )
            self.limits = (self.limits[0], self.limits[1])


@dataclass(slots=True, eq=False)
class Keys:
    """Sorted key metadata with derived labels."""

    metas: list[TraceMeta]
    _spec: KeysSpec = field(repr=False, compare=False)

    @classmethod
    def from_fields(
        cls,
        *,
        specific_keys: Sequence[str],
        indices: Sequence[int] | NDArray[np.int64],
        yvalues: Sequence[float] | NDArray64,
        spec: KeysSpec,
    ) -> Keys:
        """Build key metadata from aligned field arrays."""
        keys_list = list(specific_keys)
        indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        yvalues_array = np.asarray(yvalues, dtype=np.float64).reshape(-1)
        if len(keys_list) != indices_array.size:
            raise ValueError("specific_keys and indices must have the same length.")
        if len(keys_list) != yvalues_array.size:
            raise ValueError("specific_keys and yvalues must have the same length.")
        metas = [
            TraceMeta(
                specific_key=key,
                index=int(index),
                yvalue=(None if not np.isfinite(float(yvalue)) else float(yvalue)),
            )
            for key, index, yvalue in zip(keys_list, indices_array, yvalues_array)
        ]
        return cls(metas=metas, _spec=spec)

    @property
    def specific_keys(self) -> list[str]:
        """Return ordered specific keys."""
        return [meta.specific_key for meta in self.metas]

    @property
    def indices(self) -> NDArray[np.int64]:
        """Return ordered positional indices."""
        indices: list[int] = []
        for meta in self.metas:
            if meta.index is None:
                raise ValueError("Keys metadata must include indices.")
            indices.append(int(meta.index))
        return np.asarray(indices, dtype=np.int64)

    @property
    def yvalues(self) -> NDArray64:
        """Return ordered parsed y-values."""
        values = [
            np.nan if meta.yvalue is None else float(meta.yvalue) for meta in self.metas
        ]
        return np.asarray(values, dtype=np.float64)

    @property
    def label(self) -> str:
        """Return plain-text label for the y-values."""
        return _infer_keys_labels(
            specific_keys=self.specific_keys,
            spec=self._spec,
        )[0]

    @property
    def html_label(self) -> str:
        """Return HTML-formatted label for the y-values."""
        return _infer_keys_labels(
            specific_keys=self.specific_keys,
            spec=self._spec,
        )[1]

    def __getitem__(self, key: str) -> object:
        """Provide mapping-style access for compatibility."""
        return getattr(self, key)


def _extract_value_from_specific_key(
    specific_key: str,
    strip0: str = "=",
    strip1: str | None = None,
) -> float:
    """Parse one numeric value from a specific key string.

    Parameters
    ----------
    specific_key : str
        Key string such as ``"nu=-31.0dBm"``.
    strip0 : str, default="="
        Start delimiter. Parsing begins right after its first occurrence.
    strip1 : str | None, default=None
        Optional end delimiter. If ``None`` (or not found), parsing continues
        to the end of the key.

    Returns
    -------
    float
        Parsed numeric value.

    Raises
    ------
    ValueError
        If delimiters are invalid or the parsed token is not numeric.
    """
    if strip0 == "":
        raise ValueError("strip0 must not be empty.")

    start_idx = specific_key.find(strip0)
    if start_idx < 0:
        raise ValueError(
            f"strip0 '{strip0}' not found in specific_key '{specific_key}'.",
        )
    start_idx += len(strip0)

    if strip1 is None:
        token = specific_key[start_idx:]
    else:
        end_idx = specific_key.find(strip1, start_idx)
        token = (
            specific_key[start_idx:] if end_idx < 0 else specific_key[start_idx:end_idx]
        )

    token = token.strip()
    if token == "":
        raise ValueError(
            f"No value found in specific_key '{specific_key}'.",
        )

    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse float from specific_key '{specific_key}'.",
        ) from exc


def _normalize_removed_specific_keys(
    remove_key: str | Sequence[str] | None = None,
) -> set[str]:
    """Normalize exact specific-key removals.

    Parameters
    ----------
    remove_key : str or sequence of str, optional
        Exact specific-key names to remove before parsing and sorting.

    Returns
    -------
    set[str]
        Normalized set of keys to remove.

    Raises
    ------
    ValueError
        If a removal key is empty or not a string.
    """
    if remove_key is None:
        return set()

    keys = [remove_key] if isinstance(remove_key, str) else list(remove_key)
    normalized: set[str] = set()
    for key in keys:
        if not isinstance(key, str) or key == "":
            raise ValueError(
                "remove_key entries must be non-empty strings.",
            )
        normalized.add(key)
    return normalized


def _normalize_added_specific_keys(
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None,
) -> list[tuple[str, float]]:
    """Normalize specific keys that should be added before sorting.

    Parameters
    ----------
    add_key : tuple or sequence of tuple, optional
        Exact specific-key additions as ``(key, value)`` or
        ``[(key, value), ...]``.

    Returns
    -------
    list[tuple[str, float]]
        Normalized ``(key, value)`` pairs with finite float values.

    Raises
    ------
    ValueError
        If an added key is malformed, empty, or has a non-finite value.
    """
    if add_key is None:
        return []

    additions_raw = [add_key] if isinstance(add_key, tuple) else list(add_key)
    normalized: list[tuple[str, float]] = []
    for item in additions_raw:
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                "add_key entries must be (key, value) tuples.",
            )

        key, value = item
        if not isinstance(key, str) or key == "":
            raise ValueError(
                "add_key keys must be non-empty strings.",
            )

        value_float = float(value)
        if not np.isfinite(value_float):
            raise ValueError(
                f"add_key value for specific_key '{key}' must be finite.",
            )

        normalized.append((key, value_float))
    return normalized


def _resolve_keys_spec(
    *,
    spec: KeysSpec | None,
    strip0: str,
    strip1: str | None,
    remove_key: str | Sequence[str] | None,
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None,
    limits: (
        float
        | slice
        | tuple[int | float | None, int | float | None]
        | list[int | float | None]
        | None
    ),
) -> KeysSpec:
    """Resolve one explicit key config or one ``KeysSpec`` instance."""
    if spec is not None:
        if (
            strip0 != "="
            or strip1 is not None
            or remove_key is not None
            or add_key is not None
            or limits is not None
        ):
            raise ValueError(
                "Provide either spec or individual key arguments, not both.",
            )
        return spec

    return KeysSpec(
        strip0=strip0,
        strip1=strip1,
        remove_key=remove_key,
        add_key=add_key,
        limits=limits,
    )


def _resolve_get_keys_args(
    *,
    h5path: str | Path | FileSpec | None,
    measurement: str | None,
    filespec: FileSpec | None,
    spec: KeysSpec | None,
    keysspec: KeysSpec | None,
) -> tuple[str | Path | FileSpec, str | None, KeysSpec | None]:
    """Resolve keyword aliases for ``get_keys``."""
    if filespec is not None:
        if h5path is not None:
            raise ValueError(
                "Provide either h5path or filespec, not both.",
            )
        h5path = filespec

    if keysspec is not None:
        if spec is not None:
            raise ValueError(
                "Provide either spec or keysspec, not both.",
            )
        spec = keysspec

    if h5path is None:
        raise ValueError("Provide h5path or filespec.")

    return h5path, measurement, spec


def _infer_keys_labels(
    specific_keys: Sequence[str],
    *,
    spec: KeysSpec,
) -> tuple[str, str]:
    """Infer plain and HTML labels for one key collection."""
    if spec.label is not None:
        label = spec.label
    else:
        prefix = "y"
        for specific_key in specific_keys:
            start_idx = specific_key.find(spec.strip0)
            if start_idx > 0:
                candidate = specific_key[:start_idx].strip()
                if candidate != "":
                    prefix = candidate
                    break
        unit = "" if spec.strip1 is None else spec.strip1.strip()
        label = prefix if unit == "" else f"{prefix} ({unit})"

    if spec.html_label is not None:
        html_label = spec.html_label
    else:
        html_label = label

    return label, html_label


def _build_keys_output(
    specific_keys: list[str],
    yvalues: NDArray64,
    *,
    spec: KeysSpec,
) -> Keys:
    """Build one key-metadata object."""
    values = np.asarray(yvalues, dtype=np.float64)
    if spec.norm is not None:
        values = values / spec.norm
    return Keys.from_fields(
        specific_keys=list(specific_keys),
        indices=np.arange(len(specific_keys), dtype=np.int64),
        yvalues=values,
        spec=spec,
    )


def _is_int_like(value: object) -> bool:
    """Return whether one object should be treated as an integer index."""
    return isinstance(value, (int, np.integer)) and not isinstance(value, bool)


def _is_float_like(value: object) -> bool:
    """Return whether one object should be treated as a fractional bound."""
    return isinstance(value, (float, np.floating)) and not isinstance(value, bool)


def _normalize_limit_bound(
    bound: int | float | None,
    size: int,
    *,
    default: int,
    side: str,
) -> int:
    """Convert one limit bound to one concrete positional index.

    Parameters
    ----------
    bound : int | float | None
        Bound value. Integers are interpreted as positional indices. Floats
        must be finite and lie in ``[0, 1]`` and are interpreted as fractions
        of the sorted list length.
    size : int
        Total number of sorted keys.
    default : int
        Fallback index used when ``bound is None``.
    side : {"start", "stop"}
        Whether the bound acts as a start or stop position.

    Returns
    -------
    int
        Concrete positional index.

    Raises
    ------
    ValueError
        If the bound is malformed or outside the supported range.
    """
    if bound is None:
        return default

    if _is_int_like(bound):
        idx = int(bound)
        if idx < 0:
            idx += size
        return int(np.clip(idx, 0, size))

    if _is_float_like(bound):
        fraction = float(bound)
        if not np.isfinite(fraction) or fraction < 0.0 or fraction > 1.0:
            raise ValueError(
                "Float limits must be finite fractions in [0, 1].",
            )
        if side == "start":
            return int(np.floor(fraction * size))
        return int(np.ceil(fraction * size))

    raise ValueError(
        "limits bounds must be integers, fractions in [0, 1], or None.",
    )


def _apply_limits_to_sorted_keys(
    keys_sorted: list[str],
    values_sorted: np.ndarray,
    limits: (
        float
        | slice
        | tuple[int | float | None, int | float | None]
        | list[int | float | None]
        | None
    ) = None,
) -> tuple[list[str], np.ndarray]:
    """Apply optional positional or fractional limits to sorted keys.

    Parameters
    ----------
    keys_sorted : list[str]
        Sorted specific keys.
    values_sorted : np.ndarray
        Parsed values aligned with ``keys_sorted``.
    limits : float, slice, tuple, list, or None, default=None
        Optional selection applied after sorting.

        - ``None`` keeps the full sorted result.
        - ``0.7`` keeps the first 70 percent of entries.
        - ``(0.2, 0.7)`` keeps entries between 20 and 70 percent.
        - ``(3, 10)`` behaves like positional slicing.
        - ``slice(...)`` is forwarded directly.

    Returns
    -------
    tuple[list[str], np.ndarray]
        Limited ``(specific_keys_sorted, values_sorted)``.

    Raises
    ------
    ValueError
        If ``limits`` is malformed or removes all entries.
    """
    if limits is None:
        return keys_sorted, values_sorted

    size = len(keys_sorted)
    if isinstance(limits, slice):
        start_idx, stop_idx, step = limits.indices(size)
        if step != 1:
            raise ValueError("limits slices must use step=1.")
    elif _is_float_like(limits):
        start_idx = 0
        stop_idx = _normalize_limit_bound(
            limits,
            size,
            default=size,
            side="stop",
        )
    elif isinstance(limits, (tuple, list)) and len(limits) == 2:
        start_idx = _normalize_limit_bound(
            limits[0],
            size,
            default=0,
            side="start",
        )
        stop_idx = _normalize_limit_bound(
            limits[1],
            size,
            default=size,
            side="stop",
        )
    else:
        raise ValueError(
            "limits must be None, a fraction in [0, 1], a slice, or a "
            "(start, stop) pair.",
        )

    if stop_idx < start_idx:
        raise ValueError("limits stop must be >= start.")

    keys_limited = keys_sorted[start_idx:stop_idx]
    values_limited = np.asarray(values_sorted[start_idx:stop_idx], dtype=np.float64)
    if len(keys_limited) == 0:
        raise ValueError("limits removed all specific keys.")

    return keys_limited, values_limited


def get_keys(
    h5path: str | Path | FileSpec | None = None,
    measurement: str | None = None,
    *,
    filespec: FileSpec | None = None,
    keysspec: KeysSpec | None = None,
    strip0: str = "=",
    strip1: str | None = None,
    remove_key: str | Sequence[str] | None = None,
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None,
    spec: KeysSpec | None = None,
    limits: (
        float
        | slice
        | tuple[int | float | None, int | float | None]
        | list[int | float | None]
        | None
    ) = None,
) -> Keys:
    """List, parse, and sort specific keys.

    Parameters
    ----------
    h5path : str | pathlib.Path or FileSpec, optional
        HDF5 file path or one file specification.
    measurement : str | None, default=None
        Measurement name, e.g. ``"frequency_at_15GHz"``. Optional when
        provided by ``FileSpec``.
    filespec : FileSpec | None, optional
        Keyword alias for ``h5path`` when using one file specification.
    keysspec : KeysSpec | None, optional
        Keyword alias for ``spec``.
    strip0 : str, default="="
        Start delimiter for value parsing from each key.
    strip1 : str | None, default=None
        End delimiter for value parsing. If ``None``, parse to key end.
    remove_key : str or sequence of str, optional
        Exact specific-key names to remove before parsing and sorting.
    add_key : tuple or sequence of tuple, optional
        Exact specific-key additions as ``(key, value)`` or
        ``[(key, value), ...]``.
    spec : KeysSpec | None, optional
        Alternative bundled key-parsing configuration. Mutually exclusive with
        the individual key-parsing arguments.
    limits : float, slice, tuple, list, or None, optional
        Optional selection applied after sorting. ``limits=0.7`` keeps the
        first 70 percent of the sorted list.

    Returns
    -------
    Keys
        Sorted key metadata with parsed y-values.
    """
    h5path, measurement, spec = _resolve_get_keys_args(
        h5path=h5path,
        measurement=measurement,
        filespec=filespec,
        spec=spec,
        keysspec=keysspec,
    )
    resolved = _resolve_keys_spec(
        spec=spec,
        strip0=strip0,
        strip1=strip1,
        remove_key=remove_key,
        add_key=add_key,
        limits=limits,
    )
    _, resolved_measurement = _require_measurement(
        h5path=h5path,
        measurement=measurement,
    )
    specific_keys = list_specific_keys(
        h5path=h5path,
        measurement=resolved_measurement,
    )
    keys = list(specific_keys)
    removed_keys = set(resolved.remove_key)
    if len(removed_keys) > 0:
        keys = [key for key in keys if key not in removed_keys]

    values = [
        _extract_value_from_specific_key(
            specific_key=key,
            strip0=resolved.strip0,
            strip1=resolved.strip1,
        )
        for key in keys
    ]

    additions = list(resolved.add_key)
    for key, value in additions:
        keys.append(key)
        values.append(value)

    if len(keys) == 0:
        raise ValueError(
            "specific_keys must not be empty after applying remove_key and add_key.",
        )

    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array)
    keys_sorted = [keys[i] for i in order]
    values_sorted = np.asarray(values_array[order], dtype=np.float64)
    keys_sorted, values_sorted = _apply_limits_to_sorted_keys(
        keys_sorted=keys_sorted,
        values_sorted=values_sorted,
        limits=resolved.limits,
    )
    return _build_keys_output(
        specific_keys=keys_sorted,
        yvalues=values_sorted,
        spec=resolved,
    )


__all__ = [
    "Keys",
    "KeysSpec",
    "get_keys",
]
