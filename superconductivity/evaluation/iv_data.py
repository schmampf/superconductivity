from pathlib import Path
from typing import Sequence

import numpy as np

from ..utilities.safety import require_all_finite


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
    measurement: str,
) -> list[str]:
    """List available dataset keys for one measurement.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file path.
    measurement : str
        Measurement name, e.g. ``"frequency_at_15GHz"``.

    Returns
    -------
    list[str]
        Sorted key names below ``measurement/<measurement>``.

    Raises
    ------
    KeyError
        If measurement path does not exist in file.
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
    """
    return list_measurement_keys(
        h5path=h5path,
        measurement=measurement,
    )


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


def sort_specific_keys_by_value(
    specific_keys: Sequence[str],
    strip0: str = "=",
    strip1: str | None = None,
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

    Returns
    -------
    tuple[list[str], np.ndarray]
        ``(specific_keys_sorted, values_sorted)``.

    Raises
    ------
    ValueError
        If no keys are provided or parsing fails.
    """
    keys = list(specific_keys)
    if len(keys) == 0:
        raise ValueError("specific_keys must not be empty.")

    values = np.asarray(
        [
            _extract_value_from_specific_key(
                specific_key=key,
                strip0=strip0,
                strip1=strip1,
            )
            for key in keys
        ],
        dtype=np.float64,
    )
    order = np.argsort(values)
    keys_sorted = [keys[i] for i in order]
    values_sorted = np.asarray(values[order], dtype=np.float64)
    return keys_sorted, values_sorted


def list_specific_keys_and_values(
    h5path: str | Path,
    measurement: str,
    strip0: str = "=",
    strip1: str | None = None,
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
    )


def get_ivt(
    h5path: str | Path,
    measurement: str,
    specific_key: str,
    amp_voltage: float = 1.0,
    amp_current: float = 1.0,
    r_ref_ohm: float = 51.689e3,
    trigger_values: int | Sequence[int] | None = 1,
    skip: int | tuple[int, int] = 0,
    subtract_offset: bool = True,
    time_relative: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load one sweep trace as ``(I_nA, V_mV, t_s)``.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file path.
    measurement : str
        Measurement name, e.g. ``"frequency_at_15GHz"``.
    specific_key : str
        Dataset key below measurement, e.g. ``"nu=-31.0dBm"``.
    amp_voltage : float, default=1.0
        Voltage-channel amplification factor.
    amp_current : float, default=1.0
        Current-channel amplification factor.
    r_ref_ohm : float, default=51.689e3
        Reference resistor in ohms used for current conversion.
    trigger_values : int | Sequence[int] | None, default=1
        Trigger value(s) to keep from ``sweep/adwin/trigger``.
        If ``None``, all trigger values are kept.
    skip : int | tuple[int, int], default=0
        Number of points to skip from sweep edges after trigger filtering.
        If an integer is provided, it is applied symmetrically to the start
        and end. If a tuple is provided, it is interpreted as
        ``(skip_start, skip_end)``.
    subtract_offset : bool, default=True
        If ``True``, subtract mean offset from ``offset/adwin`` section.
    time_relative : bool, default=True
        If ``True``, return time axis shifted so ``t_s[0] == 0``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(I_nA, V_mV, t_s)`` arrays after filtering.

    Raises
    ------
    ValueError
        If configuration values are invalid.
    KeyError
        If measurement/key paths are missing.
    """
    if amp_voltage <= 0.0 or not np.isfinite(amp_voltage):
        raise ValueError("amp_voltage must be finite and > 0.")
    if amp_current <= 0.0 or not np.isfinite(amp_current):
        raise ValueError("amp_current must be finite and > 0.")
    if r_ref_ohm <= 0.0 or not np.isfinite(r_ref_ohm):
        raise ValueError("r_ref_ohm must be finite and > 0.")
    if isinstance(skip, int):
        if skip < 0:
            raise ValueError("skip must be >= 0.")
        skip_start, skip_end = skip, skip
    elif (
        isinstance(skip, tuple)
        and len(skip) == 2
        and isinstance(skip[0], int)
        and isinstance(skip[1], int)
    ):
        skip_start, skip_end = int(skip[0]), int(skip[1])
        if skip_start < 0 or skip_end < 0:
            raise ValueError("skip values must be >= 0.")
    else:
        raise ValueError("skip must be int or tuple[int, int].")

    h5py = _import_h5py()
    p = Path(h5path).expanduser()
    full_path = _to_measurement_path(
        measurement=measurement,
        specific_key=specific_key,
    )

    with h5py.File(p, "r") as file:
        if full_path not in file:
            raise KeyError(f"Dataset path not found: '{full_path}'.")

        sweep = file[f"{full_path}/sweep/adwin"]
        t_s = np.asarray(sweep["time"], dtype=np.float64)
        v1_sweep_V = np.asarray(sweep["V1"], dtype=np.float64)
        v2_sweep_V = np.asarray(sweep["V2"], dtype=np.float64)
        trigger = np.asarray(sweep["trigger"])

        if trigger_values is not None:
            trigger_keep = np.atleast_1d(np.asarray(trigger_values))
            require_all_finite(trigger_keep, name="trigger_values")
            mask = np.isin(trigger, trigger_keep)
            t_s = t_s[mask]
            v1_sweep_V = v1_sweep_V[mask]
            v2_sweep_V = v2_sweep_V[mask]

        n_total = t_s.size
        n_trim = skip_start + skip_end
        if n_trim > 0:
            if n_trim >= n_total:
                raise ValueError(
                    "skip removes all points. Reduce skip or use another trace.",
                )
            end_idx = None if skip_end == 0 else -skip_end
            t_s = t_s[skip_start:end_idx]
            v1_sweep_V = v1_sweep_V[skip_start:end_idx]
            v2_sweep_V = v2_sweep_V[skip_start:end_idx]

        V_mV = v1_sweep_V / float(amp_voltage) * 1e3
        I_nA = v2_sweep_V / (float(amp_current) * float(r_ref_ohm)) * 1e9

        if subtract_offset:
            offset = file[f"{full_path}/offset/adwin"]
            v1_offset_V = np.asarray(offset["V1"], dtype=np.float64)
            v2_offset_V = np.asarray(offset["V2"], dtype=np.float64)
            v_offset_mV = np.nanmean(v1_offset_V / float(amp_voltage) * 1e3)
            i_offset_nA = np.nanmean(
                v2_offset_V / (float(amp_current) * float(r_ref_ohm)) * 1e9,
            )
            V_mV = V_mV - float(v_offset_mV)
            I_nA = I_nA - float(i_offset_nA)

    finite = np.isfinite(t_s) & np.isfinite(V_mV) & np.isfinite(I_nA)
    if np.any(~finite):
        t_s = t_s[finite]
        V_mV = V_mV[finite]
        I_nA = I_nA[finite]

    if time_relative and t_s.size > 0:
        t_s = t_s - t_s[0]

    return (
        np.asarray(I_nA, dtype=np.float64),
        np.asarray(V_mV, dtype=np.float64),
        np.asarray(t_s, dtype=np.float64),
    )
