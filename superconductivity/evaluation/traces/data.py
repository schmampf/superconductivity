"""Discovery and retrieval for raw HDF5-backed data series."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from . import file as file_module
from .file import FileSpec, _to_measurement_path

SeriesTuple: TypeAlias = tuple[NDArray[np.float64], np.ndarray]


def _normalize_h5_path(path: str) -> str:
    """Return one normalized slash-separated HDF5 path."""
    return path.strip("/")


def _is_group_like(node: object) -> bool:
    """Return whether one HDF5-like node behaves like a group."""
    return callable(getattr(node, "keys", None))


def _iter_group_keys(node: object) -> list[str]:
    """Return sorted child keys from one group-like node."""
    keys = getattr(node, "keys", None)
    if not callable(keys):
        raise TypeError("node is not group-like.")
    return sorted(str(name) for name in keys())


def _get_structured_array_with_time(
    node: object,
    *,
    path: str,
) -> np.ndarray:
    """Return one structured dataset with a top-level ``time`` field."""
    if _is_group_like(node):
        raise KeyError(f"time field not found: '{path}'.")

    data = np.atleast_1d(np.asarray(node))
    dtype_names = data.dtype.names
    if dtype_names is None or "time" not in dtype_names:
        raise KeyError(f"time field not found: '{path}'.")
    return data


def _get_structured_field_names(
    node: object,
    *,
    path: str,
) -> tuple[str, ...]:
    """Return non-time field names from one structured dataset."""
    if _is_group_like(node):
        raise KeyError(f"time field not found: '{path}'.")

    dtype = getattr(node, "dtype", None)
    dtype_names = getattr(dtype, "names", None)
    if dtype_names is None:
        dtype_names = np.asarray(node).dtype.names
    if dtype_names is None or "time" not in dtype_names:
        raise KeyError(f"time field not found: '{path}'.")

    return tuple(
        str(field_name)
        for field_name in dtype_names
        if field_name != "time"
    )


def _get_group_series_field_names(
    node: object,
) -> list[str]:
    """Return non-time leaf field names for one group-like time series node."""
    if not _is_group_like(node) or "time" not in node:
        return []

    field_names: list[str] = []
    for name in _iter_group_keys(node):
        if name == "time":
            continue
        child = node[name]
        if _is_group_like(child):
            continue
        field_names.append(name)
    return field_names


def _extract_time_field(
    node: object,
    *,
    path: str,
) -> NDArray[np.float64]:
    """Extract one finite time array from one HDF5-like node."""
    if _is_group_like(node):
        if "time" not in node:
            raise KeyError(f"time field not found: '{path}/time'.")
        time_s = np.asarray(node["time"], dtype=np.float64)
    else:
        data = _get_structured_array_with_time(node, path=path)
        time_s = np.asarray(data["time"], dtype=np.float64)

    time_s = np.asarray(time_s, dtype=np.float64).reshape(-1)
    time_s = time_s[np.isfinite(time_s)]
    if time_s.size == 0:
        raise ValueError(f"time field is empty or non-finite: '{path}'.")
    return time_s


def _crop_field_to_window(
    data: object,
    *,
    path: str,
    field_name: str,
    time_start_s: float,
    time_stop_s: float,
) -> SeriesTuple:
    """Crop one field to the closed measurement window."""
    if _is_group_like(data):
        if "time" not in data:
            raise KeyError(f"time field not found: '{path}/time'.")
        if field_name not in data:
            raise KeyError(f"field not found: {field_name!r}.")
        if _is_group_like(data[field_name]):
            raise KeyError(f"field not found: {field_name!r}.")

        time_s = np.asarray(data["time"], dtype=np.float64).reshape(-1)
        value = np.asarray(data[field_name]).reshape(-1)
        if time_s.size != value.size:
            raise ValueError(
                f"time and value lengths differ at '{path}/{field_name}'.",
            )
        mask = (
            np.isfinite(time_s)
            & (time_s >= time_start_s)
            & (time_s <= time_stop_s)
        )
        return (
            np.asarray(time_s[mask], dtype=np.float64),
            np.asarray(value[mask]),
        )

    structured = _get_structured_array_with_time(data, path=path)
    if field_name not in (structured.dtype.names or ()):
        raise KeyError(f"field not found: {field_name!r}.")

    time_s = np.asarray(structured["time"], dtype=np.float64).reshape(-1)
    mask = (
        np.isfinite(time_s)
        & (time_s >= time_start_s)
        & (time_s <= time_stop_s)
    )
    return (
        np.asarray(time_s[mask], dtype=np.float64),
        np.asarray(structured[field_name][mask]),
    )


def _concat_series_blocks(
    blocks: list[SeriesTuple],
) -> SeriesTuple:
    """Concatenate and time-sort series blocks."""
    if len(blocks) == 0:
        raise ValueError("series blocks must not be empty.")

    time_s = np.concatenate([block[0] for block in blocks])
    value = np.concatenate([np.asarray(block[1]) for block in blocks])
    order = np.argsort(time_s, kind="stable")
    return (
        np.asarray(time_s[order], dtype=np.float64),
        np.asarray(value[order]),
    )


def _iter_series_fields(
    node: object,
    prefix: str,
) -> list[tuple[str, str]]:
    """Return ``(dataset_path, field_name)`` pairs below one node."""
    normalized_prefix = _normalize_h5_path(prefix)
    entries: list[tuple[str, str]] = []

    if _is_group_like(node):
        for field_name in _get_group_series_field_names(node):
            entries.append((normalized_prefix, field_name))
        for name in _iter_group_keys(node):
            child = node[name]
            child_path = f"{normalized_prefix}/{name}"
            entries.extend(_iter_series_fields(child, child_path))
        return entries

    try:
        field_names = _get_structured_field_names(
            node,
            path=normalized_prefix,
        )
    except KeyError:
        return entries

    for field_name in field_names:
        entries.append((normalized_prefix, field_name))
    return entries


def _iter_structured_fields(
    file: object,
    *,
    root_path: str,
    skip_prefixes: tuple[str, ...] = (),
) -> list[tuple[str, str]]:
    """Return ``(relative_dataset_path, field_name)`` pairs below one root."""
    normalized_root = _normalize_h5_path(root_path)
    if normalized_root not in file:
        return []

    entries: list[tuple[str, str]] = []
    normalized_skips = tuple(
        _normalize_h5_path(path) for path in skip_prefixes
    )
    for dataset_path, field_name in _iter_series_fields(
        file[normalized_root],
        normalized_root,
    ):
        if any(
            dataset_path == prefix or dataset_path.startswith(f"{prefix}/")
            for prefix in normalized_skips
        ):
            continue
        relative_path = dataset_path.removeprefix(f"{normalized_root}/")
        entries.append((relative_path, field_name))
    return entries


def _get_specific_keys(
    file: object,
    *,
    measurement: str,
) -> list[str]:
    """Return sorted specific keys for one measurement group."""
    measurement_path = _to_measurement_path(measurement)
    if measurement_path not in file:
        raise KeyError(f"Measurement path not found: '{measurement_path}'.")

    measurement_group = file[measurement_path]
    if not _is_group_like(measurement_group):
        raise KeyError(
            f"Measurement path is not group-like: '{measurement_path}'.",
        )

    specific_keys = _iter_group_keys(measurement_group)
    if len(specific_keys) == 0:
        raise ValueError(
            (
                "No specific keys found below measurement path "
                f"'{measurement_path}'."
            ),
        )
    return specific_keys


def _get_measurement_window(
    file: object,
    *,
    measurement: str,
    specific_keys: list[str],
) -> tuple[float, float]:
    """Return the full measurement window across all specific keys."""
    specific_key_windows = _get_specific_key_windows(
        file,
        measurement=measurement,
    )
    time_start_s: float | None = None
    time_stop_s: float | None = None

    for specific_key in specific_keys:
        if specific_key not in specific_key_windows:
            adwin_path = (
                f"{_to_measurement_path(measurement, specific_key)}"
                "/sweep/adwin"
            )
            raise KeyError(
                f"Measurement sweep adwin data not found: '{adwin_path}'.",
            )

        start, stop = specific_key_windows[specific_key]

        time_start_s = (
            start if time_start_s is None else min(time_start_s, start)
        )
        time_stop_s = stop if time_stop_s is None else max(time_stop_s, stop)

    if time_start_s is None or time_stop_s is None:
        raise ValueError("Could not determine measurement window.")
    return time_start_s, time_stop_s


def _get_specific_key_windows(
    file: object,
    *,
    measurement: str,
) -> dict[str, tuple[float, float]]:
    """Return adwin time windows for all specific keys of one measurement."""
    windows: dict[str, tuple[float, float]] = {}
    for specific_key in _get_specific_keys(file, measurement=measurement):
        sweep_adwin_path = (
            f"{_to_measurement_path(measurement, specific_key)}"
            "/sweep/adwin"
        )
        if sweep_adwin_path not in file:
            raise KeyError(
                "Measurement sweep adwin data not found: "
                f"'{sweep_adwin_path}'.",
            )

        sweep_time_s = _extract_time_field(
            file[sweep_adwin_path],
            path=sweep_adwin_path,
        )
        start = float(np.min(sweep_time_s))
        stop = float(np.max(sweep_time_s))

        offset_adwin_path = (
            f"{_to_measurement_path(measurement, specific_key)}"
            "/offset/adwin"
        )
        if offset_adwin_path in file:
            offset_time_s = _extract_time_field(
                file[offset_adwin_path],
                path=offset_adwin_path,
            )
            start = min(start, float(np.min(offset_time_s)))
            stop = max(stop, float(np.max(offset_time_s)))

        windows[specific_key] = (start, stop)
    return windows


def _resolve_file_path(
    filespec: FileSpec,
) -> tuple[object, str | None]:
    """Resolve file path and optional measurement from one ``FileSpec``."""
    return file_module._resolve_file_spec(filespec)


def _require_file_measurement(
    filespec: FileSpec,
) -> tuple[object, str]:
    """Resolve file path and require measurement from one ``FileSpec``."""
    return file_module._require_measurement(filespec)


def _open_h5():
    """Import ``h5py`` lazily."""
    return file_module._import_h5py()


def _build_status_key_map(
    file: object,
) -> dict[str, str]:
    """Return logical status keys mapped to normalized source paths."""
    mapping: dict[str, str] = {}
    for relative_path, field_name in _iter_structured_fields(
        file,
        root_path="status",
    ):
        key = f"{relative_path}/{field_name}"
        mapping[key] = f"status/{key}"
    return mapping


def get_status_keys(
    filespec: FileSpec,
) -> dict[str, str]:
    """Return available global status keys.

    Parameters
    ----------
    filespec : FileSpec
        File specification for the source HDF5 file.

    Returns
    -------
    dict[str, str]
        Flat mapping from logical key to normalized source path.
    """
    path, _ = _resolve_file_path(filespec)
    h5py = _open_h5()

    with h5py.File(path, "r") as file:
        return _build_status_key_map(file)


def get_status_series(
    filespec: FileSpec,
    key: str,
) -> SeriesTuple:
    """Return one cropped global status time series.

    Parameters
    ----------
    filespec : FileSpec
        File specification with ``measurement`` configured.
    key : str
        Logical status key returned by :func:`get_status_keys`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(time_s, value)`` for the requested key.
    """
    path, measurement = _require_file_measurement(filespec)
    h5py = _open_h5()

    with h5py.File(path, "r") as file:
        key_map = _build_status_key_map(file)
        if key not in key_map:
            raise KeyError(f"Status key not found: {key!r}.")

        specific_keys = _get_specific_keys(file, measurement=measurement)
        time_start_s, time_stop_s = _get_measurement_window(
            file,
            measurement=measurement,
            specific_keys=specific_keys,
        )

        source = _normalize_h5_path(key_map[key])
        dataset_path, _, field_name = source.rpartition("/")
        data = _get_structured_array_with_time(
            file[dataset_path],
            path=dataset_path,
        )
        return _crop_field_to_window(
            data,
            path=dataset_path,
            field_name=field_name,
            time_start_s=time_start_s,
            time_stop_s=time_stop_s,
        )


def _build_measurement_key_map(
    file: object,
    *,
    measurement: str,
) -> dict[str, str]:
    """Return aggregated logical measurement keys and wildcard sources."""
    mapping: dict[str, str] = {}
    measurement_root = _to_measurement_path(measurement)
    specific_keys = _get_specific_keys(file, measurement=measurement)
    group_names = ("sweep", "offset")

    for specific_key in specific_keys:
        measurement_key_root = _to_measurement_path(measurement, specific_key)
        for group_name in group_names:
            group_root = f"{measurement_key_root}/{group_name}"
            for relative_path, field_name in _iter_structured_fields(
                file,
                root_path=group_root,
            ):
                key = f"{relative_path}/{field_name}"
                if key not in mapping:
                    mapping[key] = (
                        f"{measurement_root}/*/*/"
                        f"{relative_path}/{field_name}"
                    )
    return mapping


def get_measurement_keys(
    filespec: FileSpec,
) -> dict[str, str]:
    """Return available measurement-local sweep and offset keys.

    Parameters
    ----------
    filespec : FileSpec
        File specification with ``measurement`` configured.

    Returns
    -------
    dict[str, str]
        Flat mapping from logical key to wildcarded aggregate source path.
    """
    path, measurement = _require_file_measurement(filespec)
    h5py = _open_h5()

    with h5py.File(path, "r") as file:
        return _build_measurement_key_map(file, measurement=measurement)


def get_measurement_series(
    filespec: FileSpec,
    key: str,
) -> SeriesTuple:
    """Return one cropped aggregated measurement-local time series.

    Parameters
    ----------
    filespec : FileSpec
        File specification with ``measurement`` configured.
    key : str
        Logical measurement key returned by :func:`get_measurement_keys`.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(time_s, value)`` for the requested logical key.
    """
    path, measurement = _require_file_measurement(filespec)
    h5py = _open_h5()

    with h5py.File(path, "r") as file:
        key_map = _build_measurement_key_map(file, measurement=measurement)
        if key not in key_map:
            raise KeyError(f"Measurement key not found: {key!r}.")

        specific_keys = _get_specific_keys(file, measurement=measurement)
        time_start_s, time_stop_s = _get_measurement_window(
            file,
            measurement=measurement,
            specific_keys=specific_keys,
        )

        relative_path, _, field_name = key.rpartition("/")

        blocks: list[SeriesTuple] = []
        for specific_key in specific_keys:
            for group_name in ("sweep", "offset"):
                dataset_path = (
                    f"{_to_measurement_path(measurement, specific_key)}"
                    f"/{group_name}/{relative_path}"
                )
                if dataset_path not in file:
                    continue

                blocks.append(
                    _crop_field_to_window(
                        file[dataset_path],
                        path=dataset_path,
                        field_name=field_name,
                        time_start_s=time_start_s,
                        time_stop_s=time_stop_s,
                    ),
                )

        if len(blocks) == 0:
            raise KeyError(f"Measurement key not found: {key!r}.")

        return _concat_series_blocks(blocks)


__all__ = [
    "get_status_keys",
    "get_status_series",
    "get_measurement_keys",
    "get_measurement_series",
]
