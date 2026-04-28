"""Helpers for measurement and specific-key handling."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from ...utilities.meta import AxisSpec, axis
from ...utilities.meta.label import LabelSpec, label as make_label
from ...utilities.types import NDArray64
from .file import FileSpec, _require_measurement, list_specific_keys


def numeric_yvalue(value: object) -> float | None:
    """Return one numeric y-value when possible."""
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool,
    ):
        numeric = float(value)
        if np.isfinite(numeric):
            return numeric
    return None


@dataclass(slots=True)
class KeysSpec:
    """Configuration for parsing and selecting specific keys.

    Parameters
    ----------
    strip0 : str | None, default="="
        Start delimiter for value parsing. ``None`` means parsing starts at
        the beginning of each specific key.
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
    label : str | LabelSpec | None, optional
        Shared label metadata for the parsed values.
    limits : float, slice, tuple, list, or None, optional
        Optional selection applied after sorting.
    """

    strip0: str | None = "="
    strip1: str | None = None
    remove_key: str | Sequence[str] | None = None
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None
    norm: float | None = None
    label: LabelSpec | str | None = None
    limits: float | slice | tuple[int | float | None, int | float | None] | None = None

    def __post_init__(self) -> None:
        """Normalize and validate key-parsing settings."""
        if self.strip0 is not None and not isinstance(self.strip0, str):
            raise ValueError("strip0 must be a string or None.")
        if self.strip1 is not None and not isinstance(self.strip1, str):
            raise ValueError("strip1 must be a string or None.")
        if self.norm is not None:
            self.norm = float(self.norm)
            if not np.isfinite(self.norm) or self.norm <= 0.0:
                raise ValueError("norm must be finite and > 0.")
        if self.label is not None and not isinstance(self.label, (str, LabelSpec)):
            raise ValueError("label must be a string, LabelSpec, or None.")
        if isinstance(self.label, str):
            self.label = make_label(self.label)

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

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        return (
            "strip0",
            "strip1",
            "remove_key",
            "add_key",
            "norm",
            "label",
            "limits",
        )


@dataclass(slots=True, eq=False)
class Keys:
    """Sorted key metadata with derived labels."""

    y: AxisSpec
    indices: NDArray64
    skeys: tuple[str, ...]
    _yvalues: NDArray64 = field(repr=False, compare=False)
    _spec: KeysSpec = field(repr=False, compare=False)

    @classmethod
    def from_fields(
        cls,
        *,
        specific_keys: Sequence[str],
        indices: Sequence[int] | NDArray[np.int64],
        yvalues: Sequence[object] | NDArray[np.object_] | NDArray64,
        spec: KeysSpec,
    ) -> Keys:
        """Build key metadata from aligned field arrays."""
        keys_list = list(specific_keys)
        indices_array = np.asarray(indices, dtype=np.int64).reshape(-1)
        yvalues_list = list(yvalues)
        if len(keys_list) != indices_array.size:
            raise ValueError("specific_keys and indices must have the same length.")
        if len(keys_list) != len(yvalues_list):
            raise ValueError("specific_keys and yvalues must have the same length.")
        return cls(
            y=_build_y_axis(
                specific_keys=keys_list,
                indices=indices_array,
                yvalues=yvalues_list,
                spec=spec,
            ),
            indices=np.asarray(indices_array, dtype=np.float64),
            skeys=tuple(keys_list),
            _yvalues=_coerce_numeric_yvalues(indices_array, yvalues_list),
            _spec=spec,
        )

    @property
    def specific_keys(self) -> list[str]:
        """Return ordered specific keys."""
        return list(self.skeys)

    @property
    def yvalues(self) -> NDArray64:
        """Return ordered parsed y-values."""
        return np.asarray(self._yvalues, dtype=np.float64)

    @property
    def i(self) -> NDArray64:
        """Return ordered positional indices."""
        return np.asarray(self.indices, dtype=np.float64)

    @property
    def label(self) -> str:
        """Return plain-text label for the y-values."""
        return _infer_keys_labels(
            specific_keys=self.specific_keys,
            spec=self._spec,
        )[0]

    def __getattr__(self, name: str):
        if name == self.y.code_label:
            return self.y
        raise AttributeError(name)

    def __getitem__(self, key: str) -> object:
        """Provide mapping-style access for compatibility."""
        if key == "y":
            return self.y
        if key in {"i", "indices", "skeys", "specific_keys"}:
            return getattr(self, key)
        if key == self.y.code_label:
            return self.y
        return getattr(self, key)

    def keys(self) -> tuple[str, ...]:
        """Return public mapping-style keys."""
        keys = ["y", "i", "indices", "skeys", "specific_keys"]
        if self.y.code_label != "y":
            keys.insert(1, self.y.code_label)
        return tuple(keys)


def _extract_value_token_from_specific_key(
    specific_key: str,
    strip0: str | None = "=",
    strip1: str | None = None,
) -> str:
    """Extract one raw value token from a specific key string.

    Parameters
    ----------
    specific_key : str
        Key string such as ``"nu=-31.0dBm"``.
    strip0 : str | None, default="="
        Start delimiter. Parsing begins right after its first occurrence.
        ``None`` means parsing begins at the start of ``specific_key``.
    strip1 : str | None, default=None
        Optional end delimiter. If ``None`` (or not found), parsing continues
        to the end of the key.

    Returns
    -------
    str
        Extracted value token.

    Raises
    ------
    ValueError
        If delimiters are invalid or no token can be extracted.
    """
    if strip0 is None or strip0 == "":
        start_idx = 0
    else:
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
    return token


def _extract_yvalue_from_specific_key(
    specific_key: str,
    strip0: str | None = "=",
    strip1: str | None = None,
) -> float | None:
    """Extract one parsed y-value from a specific key string.

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
    float | None
        Parsed finite float when available, else ``None``.
    """
    try:
        token = _extract_value_token_from_specific_key(
            specific_key=specific_key,
            strip0=strip0,
            strip1=strip1,
        )
    except ValueError:
        return None

    try:
        value = float(token)
    except ValueError:
        return None
    if not np.isfinite(value):
        return None
    return value


def _extract_value_from_specific_key(
    specific_key: str,
    strip0: str | None = "=",
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
    token = _extract_value_token_from_specific_key(
        specific_key=specific_key,
        strip0=strip0,
        strip1=strip1,
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

    if isinstance(add_key, tuple) and len(add_key) == 2 and isinstance(add_key[0], str):
        additions_raw = [add_key]
    else:
        additions_raw = list(add_key)
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
    strip0: str | None,
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
        label = spec.label.print_label
        html_label = spec.label.html_label
    else:
        prefix = "y"
        for specific_key in specific_keys:
            if spec.strip0 is None:
                break
            start_idx = specific_key.find(spec.strip0)
            if start_idx > 0:
                candidate = specific_key[:start_idx].strip()
                if candidate != "":
                    prefix = candidate
                    break
        unit = "" if spec.strip1 is None else spec.strip1.strip()
        label = prefix if unit == "" else f"{prefix} ({unit})"
        html_label = label

    return label, html_label


def _build_keys_output(
    specific_keys: list[str],
    yvalues: Sequence[object],
    *,
    spec: KeysSpec,
) -> Keys:
    """Build one key-metadata object."""
    values: list[object] = []
    for value in yvalues:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool,
        ):
            numeric = float(value)
            if np.isfinite(numeric) and spec.norm is not None:
                values.append(numeric / spec.norm)
            else:
                values.append(numeric)
        else:
            values.append(value)
    return Keys.from_fields(
        specific_keys=list(specific_keys),
        indices=np.arange(len(specific_keys), dtype=np.int64),
        yvalues=values,
        spec=spec,
    )


def _build_y_axis(
    *,
    specific_keys: Sequence[str],
    indices: Sequence[int] | NDArray[np.int64],
    yvalues: Sequence[object],
    spec: KeysSpec,
) -> AxisSpec:
    numeric: list[float] = []
    valid_numeric = True
    for value in yvalues:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool,
        ):
            cast = float(value)
            if not np.isfinite(cast):
                valid_numeric = False
                break
            numeric.append(cast)
        else:
            valid_numeric = False
            break
    if valid_numeric and len(numeric) >= 2 and np.all(np.diff(np.asarray(numeric)) > 0.0):
        values = np.asarray(numeric, dtype=np.float64)
    else:
        values = np.asarray(indices, dtype=np.float64).reshape(-1)
    if values.size == 1:
        values = np.asarray([values[0], values[0] + 1.0], dtype=np.float64)
    return _make_y_axis(values=values, specific_keys=specific_keys, spec=spec)


def _coerce_numeric_yvalues(
    indices: Sequence[int] | NDArray[np.int64],
    yvalues: Sequence[object],
) -> NDArray64:
    numeric: list[float] = []
    valid_numeric = True
    for value in yvalues:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
            value, bool,
        ):
            cast = float(value)
            if not np.isfinite(cast):
                valid_numeric = False
                break
            numeric.append(cast)
        else:
            valid_numeric = False
            break
    if valid_numeric:
        return np.asarray(numeric, dtype=np.float64)
    return np.asarray(indices, dtype=np.float64).reshape(-1)


def _make_y_axis(
    *,
    values: NDArray64,
    specific_keys: Sequence[str],
    spec: KeysSpec,
) -> AxisSpec:
    if spec.label is not None:
        return AxisSpec(
            code_label=spec.label.code_label,
            print_label=spec.label.print_label,
            html_label=spec.label.html_label,
            latex_label=spec.label.latex_label,
            values=values,
            order=0,
        )
    return axis(_infer_code_label(specific_keys, spec), values=values, order=0)


def _infer_code_label(
    specific_keys: Sequence[str],
    spec: KeysSpec,
) -> str:
    """Infer one code label from the specific-key prefix."""
    if spec.label is not None:
        return spec.label.code_label
    for specific_key in specific_keys:
        if spec.strip0 is None:
            break
        start_idx = specific_key.find(spec.strip0)
        if start_idx > 0:
            candidate = specific_key[:start_idx].strip()
            if candidate != "":
                return candidate
    return "y"


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


def _resolve_limits_slice(
    size: int,
    limits: (
        float
        | slice
        | tuple[int | float | None, int | float | None]
        | list[int | float | None]
        | None
    ) = None,
) -> tuple[int, int]:
    """Resolve optional positional or fractional limits into a slice.

    Parameters
    ----------
    size : int
        Number of available entries.
    limits : float, slice, tuple, list, or None, default=None
        Optional selection applied after sorting.

        - ``None`` keeps the full sorted result.
        - ``0.7`` keeps the first 70 percent of entries.
        - ``(0.2, 0.7)`` keeps entries between 20 and 70 percent.
        - ``(3, 10)`` behaves like positional slicing.
        - ``slice(...)`` is forwarded directly.

    Returns
    -------
    tuple[int, int]
        Concrete ``(start, stop)`` slice bounds.

    Raises
    ------
    ValueError
        If ``limits`` is malformed or removes all entries.
    """
    if limits is None:
        return 0, size

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
    if stop_idx == start_idx:
        raise ValueError("limits removed all specific keys.")
    return start_idx, stop_idx


def get_keys(
    h5path: str | Path | FileSpec | None = None,
    measurement: str | None = None,
    *,
    filespec: FileSpec | None = None,
    keysspec: KeysSpec | None = None,
    strip0: str | None = "=",
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
    strip0 : str | None, default="="
        Start delimiter for value parsing from each key. ``None`` means the
        value token starts at the beginning of the key.
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
    parsed_entries: list[tuple[str, float]] = []
    fallback_entries: list[tuple[str, object]] = []
    for key in keys:
        value = _extract_yvalue_from_specific_key(
            specific_key=key,
            strip0=resolved.strip0,
            strip1=resolved.strip1,
        )
        if isinstance(value, float):
            parsed_entries.append((key, value))
        elif value is None:
            fallback_entries.append((key, None))

    additions = list(resolved.add_key)
    for key, value in additions:
        parsed_entries.append((key, value))

    if len(parsed_entries) == 0 and len(fallback_entries) == 0:
        raise ValueError(
            "specific_keys must not be empty after applying remove_key and add_key.",
        )

    if len(parsed_entries) > 0:
        parsed_entries.sort(key=lambda item: item[1])

    keys_sorted = [key for key, _ in parsed_entries]
    values_sorted: list[object] = [value for _, value in parsed_entries]
    keys_sorted.extend(key for key, _ in fallback_entries)
    values_sorted.extend(value for _, value in fallback_entries)

    if resolved.limits is not None:
        start_idx, stop_idx = _resolve_limits_slice(
            len(keys_sorted),
            resolved.limits,
        )
        keys_sorted = keys_sorted[start_idx:stop_idx]
        values_sorted = values_sorted[start_idx:stop_idx]

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
