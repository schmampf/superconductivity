"""HDF5 cache helpers for MAR model lookups."""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
import threading

import numpy as np

from ....utilities.types import NDArray64
from .voltage import dequantize_voltage_mV
from .voltage import V_TOL_MV

try:  # pragma: no cover - platform dependent.
    import fcntl
except ImportError:  # pragma: no cover - windows only.
    fcntl = None
    import msvcrt
else:  # pragma: no cover - non-windows only.
    msvcrt = None

_LOCKS_GUARD = threading.Lock()
_FILE_LOCKS: dict[Path, threading.RLock] = {}
CurveEvaluator = Callable[[NDArray64], NDArray64]


def import_h5py():
    """Import h5py lazily for MAR caches."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover - environment dependent.
        raise ImportError(
            "h5py is required for MAR HDF5 caching. "
            "Install it with 'pip install h5py'.",
        ) from exc
    return h5py


def _require_group_path(root, group_path: str):
    """Create or reuse one nested HDF5 group path."""
    group = root
    for part in group_path.split("/"):
        if not part:
            continue
        group = group.require_group(part)
    return group


def _ensure_store_metadata(handle) -> None:
    """Ensure the shared MAR cache file has stable root metadata."""
    handle.attrs.setdefault("cache_family", "mar")
    handle.attrs.setdefault("schema_version", 1)


def _lock_file_path(cache_file: Path) -> Path:
    """Return the sidecar lock-file path for one HDF5 cache file."""
    return cache_file.with_name(f"{cache_file.name}.lock")


def _get_thread_lock(lock_file: Path) -> threading.RLock:
    """Return the in-process lock guarding one cache file path."""
    with _LOCKS_GUARD:
        lock = _FILE_LOCKS.get(lock_file)
        if lock is None:
            lock = threading.RLock()
            _FILE_LOCKS[lock_file] = lock
        return lock


def _acquire_process_lock(lock_handle) -> None:
    """Acquire the inter-process sidecar lock."""
    if fcntl is not None:  # pragma: no branch - platform dependent.
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        return
    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_LOCK, 1)


def _release_process_lock(lock_handle) -> None:
    """Release the inter-process sidecar lock."""
    if fcntl is not None:  # pragma: no branch - platform dependent.
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return
    lock_handle.seek(0)
    msvcrt.locking(lock_handle.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def locked_h5_file(cache_file: Path, mode: str) -> Iterator[object]:
    """Open one HDF5 cache file under a thread/process-safe lock."""
    h5py = import_h5py()
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    lock_file = _lock_file_path(cache_file)
    thread_lock = _get_thread_lock(lock_file)

    with thread_lock:
        with open(lock_file, "a+b") as lock_handle:
            lock_handle.seek(0)
            lock_handle.write(b"0")
            lock_handle.flush()
            _acquire_process_lock(lock_handle)
            try:
                with h5py.File(cache_file, mode) as handle:
                    yield handle
            finally:
                _release_process_lock(lock_handle)


def load_curve(
    cache_file: Path,
    group_path: str,
) -> tuple[NDArray64, NDArray64]:
    """Load one cached positive-voltage curve from the HDF5 store."""
    with locked_h5_file(cache_file, "a") as handle:
        _ensure_store_metadata(handle)
        if group_path not in handle:
            return (
                np.empty((0,), dtype=np.int64),
                np.empty((0,), dtype=np.float64),
            )
        group = handle[group_path]
        V_q = np.array(group["V_q"][...], dtype=np.int64)
        I_nA = np.array(group["I_nA"][...], dtype=np.float64)
        return V_q, I_nA


def save_curve(
    cache_file: Path,
    group_path: str,
    attrs: dict[str, float | int | str],
    V_q: NDArray64,
    I_nA: NDArray64,
) -> None:
    """Persist one cached positive-voltage curve in the HDF5 store."""
    with locked_h5_file(cache_file, "a") as handle:
        _ensure_store_metadata(handle)
        group = _require_group_path(handle, group_path)
        for key, value in attrs.items():
            group.attrs[key] = value
        group.attrs["voltage_count"] = int(V_q.size)
        group.attrs["voltage_decimals"] = V_TOL_MV

        if "V_q" in group:
            del group["V_q"]
        if "I_nA" in group:
            del group["I_nA"]

        group.create_dataset(
            "V_q",
            data=V_q,
            compression="gzip",
            shuffle=True,
        )
        group.create_dataset(
            "I_nA",
            data=I_nA,
            compression="gzip",
            shuffle=True,
        )


def merge_sorted_curve(
    V_cached_q: NDArray64,
    I_cached_nA: NDArray64,
    V_missing_q: NDArray64,
    I_missing_nA: NDArray64,
) -> tuple[NDArray64, NDArray64]:
    """Merge cached and newly evaluated positive-voltage points."""
    if V_missing_q.size == 0:
        return V_cached_q, I_cached_nA
    if V_cached_q.size == 0:
        return V_missing_q, I_missing_nA

    V_all_q = np.concatenate((V_cached_q, V_missing_q))
    I_all_nA = np.concatenate((I_cached_nA, I_missing_nA))

    sort_idx = np.argsort(V_all_q)
    V_all_q = V_all_q[sort_idx]
    I_all_nA = I_all_nA[sort_idx]

    keep = np.ones(V_all_q.shape[0], dtype=bool)
    keep[1:] = V_all_q[1:] != V_all_q[:-1]
    return V_all_q[keep], I_all_nA[keep]


def ensure_curve_cached(
    cache_file: Path,
    group_path: str,
    attrs: dict[str, float | int | str],
    V_requested_q: NDArray64,
    evaluate_missing_q: CurveEvaluator,
    caching: bool = True,
) -> tuple[NDArray64, NDArray64]:
    """Ensure one positive-voltage curve exists on the requested grid."""
    V_requested_q = np.unique(np.asarray(V_requested_q, dtype=np.int64))

    if V_requested_q.size == 0:
        return (
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
        )

    if not caching:
        return (
            V_requested_q,
            np.asarray(evaluate_missing_q(V_requested_q), dtype=np.float64),
        )

    V_cached_q, I_cached_nA = load_curve(
        cache_file=cache_file,
        group_path=group_path,
    )

    V_missing_q = np.setdiff1d(V_requested_q, V_cached_q, assume_unique=True)
    if V_missing_q.size == 0:
        return V_cached_q, I_cached_nA

    I_missing_nA = np.asarray(
        evaluate_missing_q(V_missing_q),
        dtype=np.float64,
    )
    V_cached_q, I_cached_nA = merge_sorted_curve(
        V_cached_q=V_cached_q,
        I_cached_nA=I_cached_nA,
        V_missing_q=V_missing_q,
        I_missing_nA=I_missing_nA,
    )
    save_curve(
        cache_file=cache_file,
        group_path=group_path,
        attrs=attrs,
        V_q=V_cached_q,
        I_nA=I_cached_nA,
    )
    return V_cached_q, I_cached_nA


def lookup_currents(
    V_lookup_q: NDArray64,
    V_cached_q: NDArray64,
    I_cached_nA: NDArray64,
) -> NDArray64:
    """Lookup cached currents on the exact integer voltage grid."""
    idx = np.searchsorted(V_cached_q, V_lookup_q)
    if np.any(idx >= V_cached_q.size):
        raise RuntimeError("Requested voltages fall outside the cached range.")
    if not np.array_equal(V_cached_q[idx], V_lookup_q):
        raise RuntimeError("Requested voltages are missing from the cache.")
    return I_cached_nA[idx]


def explore_curve_store(
    cache_file: Path,
    root_group_path: str,
    include_data: bool = False,
) -> list[dict[str, object]]:
    """Return one summary per cached curve below one HDF5 group."""
    if not cache_file.exists():
        return []

    entries: list[dict[str, object]] = []

    with locked_h5_file(cache_file, "r") as handle:
        if root_group_path not in handle:
            return []
        curves = handle[root_group_path]

        for key in sorted(curves.keys()):
            group = curves[key]
            V_q = np.array(group["V_q"][...], dtype=np.int64)
            I_nA = np.array(group["I_nA"][...], dtype=np.float64)
            V_mV = dequantize_voltage_mV(V_q)

            entry: dict[str, object] = {
                "key": key,
                "tau": float(group.attrs["tau"]),
                "T_K": float(group.attrs["T_K"]),
                "Delta_meV": float(group.attrs["Delta_meV"]),
                "gamma_meV": float(group.attrs["gamma_meV"]),
                "voltage_count": int(V_q.size),
                "V_min_mV": None if V_mV.size == 0 else float(V_mV[0]),
                "V_max_mV": None if V_mV.size == 0 else float(V_mV[-1]),
            }
            if include_data:
                entry["V_mV"] = V_mV
                entry["I_nA"] = I_nA
            entries.append(entry)

    return entries
