"""Helpers for measurement and specific-key handling."""

from pathlib import Path
from typing import Sequence

import numpy as np


def _import_h5py():
    """Import h5py lazily.

    Returns
    -------
    module
        Imported ``h5py`` module.

    Raises
    ------
    ImportError
        If ``h5py`` is not installed.
    """
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "h5py is required for HDF5 loading. Install it with " "'pip install h5py'.",
        ) from exc
    return h5py


def _to_measurement_path(
    measurement: str,
    specific_key: str | None = None,
) -> str:
    """Build normalized measurement path below ``measurement/``.

    Parameters
    ----------
    measurement : str
        Measurement group name or path.
    specific_key : str | None, default=None
        Optional measurement key.

    Returns
    -------
    str
        Path relative to root without leading slash.
    """
    meas = measurement.strip("/")
    if meas.startswith("measurement/"):
        meas = meas[len("measurement/") :]
    if specific_key is None:
        return f"measurement/{meas}"
    return f"measurement/{meas}/{specific_key.strip('/')}"


def list_measurement_keys(
    h5path: str | Path,
) -> list[str]:
    """List available measurement names.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file path.

    Returns
    -------
    list[str]
        Sorted measurement names below ``measurement/``.

    Raises
    ------
    KeyError
        If the ``measurement`` group does not exist in the file.
    """
    h5py = _import_h5py()
    p = Path(h5path).expanduser()
    root = "measurement"
    with h5py.File(p, "r") as file:
        if root not in file:
            raise KeyError(f"Measurement root not found: '{root}'.")
        keys = list(file[root].keys())
    return sorted(keys)


def list_specific_keys(
    h5path: str | Path,
    measurement: str,
) -> list[str]:
    """List available specific keys for one measurement.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file path.
    measurement : str
        Measurement name, e.g. ``"frequency_at_15GHz"``.

    Returns
    -------
    list[str]
        Sorted specific keys below ``measurement/<measurement>``.

    Raises
    ------
    KeyError
        If the measurement path does not exist in the file.
    """
    h5py = _import_h5py()
    p = Path(h5path).expanduser()
    measurement_path = _to_measurement_path(measurement)
    with h5py.File(p, "r") as file:
        if measurement_path not in file:
            raise KeyError(
                f"Measurement path not found: '{measurement_path}'.",
            )
        keys = list(file[measurement_path].keys())
    return sorted(keys)


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
    limits: float
    | slice
    | tuple[int | float | None, int | float | None]
    | list[int | float | None]
    | None = None,
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


def sort_specific_keys_by_value(
    specific_keys: Sequence[str],
    strip0: str = "=",
    strip1: str | None = None,
    remove_key: str | Sequence[str] | None = None,
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None,
    limits: float
    | slice
    | tuple[int | float | None, int | float | None]
    | list[int | float | None]
    | None = None,
) -> tuple[list[str], np.ndarray]:
    """Sort specific keys by parsed numeric value.

    Parameters
    ----------
    specific_keys : Sequence[str]
        Input specific keys to parse and sort.
    strip0 : str, default="="
        Start delimiter for value parsing.
    strip1 : str | None, default=None
        End delimiter for value parsing. If ``None``, parse to key end.
    remove_key : str or sequence of str, optional
        Exact specific-key names to remove before parsing and sorting.
    add_key : tuple or sequence of tuple, optional
        Exact specific-key additions as ``(key, value)`` or
        ``[(key, value), ...]``.
    limits : float, slice, tuple, list, or None, optional
        Optional selection applied after sorting. ``limits=0.7`` keeps the
        first 70 percent of the sorted list. Two-element tuples/lists act like
        ``(start, stop)`` bounds, where integers are positional indices and
        floats in ``[0, 1]`` are relative fractions.

    Returns
    -------
    tuple[list[str], np.ndarray]
        ``(specific_keys_sorted, values_sorted)``.

    Raises
    ------
    ValueError
        If no keys remain after filtering/adding or parsing fails.
    """
    keys = list(specific_keys)
    removed_keys = _normalize_removed_specific_keys(remove_key)
    if len(removed_keys) > 0:
        keys = [key for key in keys if key not in removed_keys]

    values = [
        _extract_value_from_specific_key(
            specific_key=key,
            strip0=strip0,
            strip1=strip1,
        )
        for key in keys
    ]

    additions = _normalize_added_specific_keys(add_key)
    for key, value in additions:
        keys.append(key)
        values.append(value)

    if len(keys) == 0:
        raise ValueError(
            "specific_keys must not be empty after applying remove_key " "and add_key.",
        )

    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array)
    keys_sorted = [keys[i] for i in order]
    values_sorted = np.asarray(values_array[order], dtype=np.float64)
    return _apply_limits_to_sorted_keys(
        keys_sorted=keys_sorted,
        values_sorted=values_sorted,
        limits=limits,
    )


def list_specific_keys_and_values(
    h5path: str | Path,
    measurement: str,
    strip0: str = "=",
    strip1: str | None = None,
    remove_key: str | Sequence[str] | None = None,
    add_key: tuple[str, float] | Sequence[tuple[str, float]] | None = None,
    limits: float
    | slice
    | tuple[int | float | None, int | float | None]
    | list[int | float | None]
    | None = None,
) -> tuple[list[str], np.ndarray]:
    """List and sort specific keys with parsed numeric values.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file path.
    measurement : str
        Measurement name, e.g. ``"frequency_at_15GHz"``.
    strip0 : str, default="="
        Start delimiter for value parsing from each key.
    strip1 : str | None, default=None
        End delimiter for value parsing. If ``None``, parse to key end.
    remove_key : str or sequence of str, optional
        Exact specific-key names to remove before parsing and sorting.
    add_key : tuple or sequence of tuple, optional
        Exact specific-key additions as ``(key, value)`` or
        ``[(key, value), ...]``.
    limits : float, slice, tuple, list, or None, optional
        Optional selection applied after sorting. ``limits=0.7`` keeps the
        first 70 percent of the sorted list.

    Returns
    -------
    tuple[list[str], np.ndarray]
        ``(specific_keys_sorted, values_sorted)``.
    """
    specific_keys = list_specific_keys(
        h5path=h5path,
        measurement=measurement,
    )
    return sort_specific_keys_by_value(
        specific_keys=specific_keys,
        strip0=strip0,
        strip1=strip1,
        remove_key=remove_key,
        add_key=add_key,
        limits=limits,
    )


__all__ = [
    "list_measurement_keys",
    "list_specific_keys",
    "list_specific_keys_and_values",
    "sort_specific_keys_by_value",
]
