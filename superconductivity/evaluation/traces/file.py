"""File-level specifications for HDF5-backed evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _import_h5py():
    """Import h5py lazily."""
    try:
        import h5py
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "h5py is required for HDF5 loading. Install it with 'pip install h5py'.",
        ) from exc
    return h5py


def _to_measurement_path(
    measurement: str,
    specific_key: str | None = None,
) -> str:
    """Build normalized measurement path below ``measurement/``."""
    meas = measurement.strip("/")
    if meas.startswith("measurement/"):
        meas = meas[len("measurement/") :]
    if specific_key is None:
        return f"measurement/{meas}"
    return f"measurement/{meas}/{specific_key.strip('/')}"


@dataclass(slots=True)
class FileSpec:
    """Specification for one HDF5 file and optional measurement.

    Parameters
    ----------
    h5path : str | pathlib.Path
        HDF5 file name or path.
    location : str | pathlib.Path | None, default=None
        Optional root folder used when ``h5path`` is relative.
    measurement : str | None, default=None
        Optional measurement name below ``measurement/``.
    """

    h5path: str | Path
    location: str | Path | None = None
    measurement: str | None = None

    @property
    def path(self) -> Path:
        """Return resolved HDF5 path."""
        return _resolve_h5path(
            h5path=self.h5path,
            location=self.location,
        )

    def mkeys(self) -> list[str]:
        """Return measurement keys from the configured file."""
        return list_measurement_keys(self)

    def skeys(self) -> list[str]:
        """Return specific keys from the configured measurement."""
        return list_specific_keys(self)


def _resolve_h5path(
    h5path: str | Path,
    *,
    location: str | Path | None = None,
) -> Path:
    """Resolve one HDF5 path against an optional root folder."""
    path = Path(h5path).expanduser()
    if path.is_absolute() or location is None:
        return path
    return Path(location).expanduser() / path


def _resolve_file_spec(
    h5path: str | Path | FileSpec,
    measurement: str | None = None,
) -> tuple[Path, str | None]:
    """Resolve path and measurement from raw args or one ``FileSpec``."""
    if isinstance(h5path, FileSpec):
        path = h5path.path
        if (
            measurement is not None
            and h5path.measurement is not None
            and measurement != h5path.measurement
        ):
            raise ValueError(
                "measurement conflicts with FileSpec.measurement.",
            )
        return path, h5path.measurement if measurement is None else measurement

    return _resolve_h5path(h5path), measurement


def _require_measurement(
    h5path: str | Path | FileSpec,
    measurement: str | None = None,
) -> tuple[Path, str]:
    """Resolve one file input and require a measurement string."""
    path, resolved_measurement = _resolve_file_spec(
        h5path=h5path,
        measurement=measurement,
    )
    if resolved_measurement is None:
        raise ValueError(
            "measurement is required unless provided by FileSpec.",
        )
    return path, resolved_measurement


def list_measurement_keys(
    h5path: str | Path | FileSpec,
) -> list[str]:
    """List available measurement names."""
    h5py = _import_h5py()
    p, _ = _resolve_file_spec(h5path)
    root = "measurement"
    with h5py.File(p, "r") as file:
        if root not in file:
            raise KeyError(f"Measurement root not found: '{root}'.")
        keys = list(file[root].keys())
    return sorted(keys)


def list_specific_keys(
    h5path: str | Path | FileSpec,
    measurement: str | None = None,
) -> list[str]:
    """List available specific keys for one measurement."""
    h5py = _import_h5py()
    p, resolved_measurement = _require_measurement(
        h5path=h5path,
        measurement=measurement,
    )
    measurement_path = _to_measurement_path(resolved_measurement)
    with h5py.File(p, "r") as file:
        if measurement_path not in file:
            raise KeyError(
                f"Measurement path not found: '{measurement_path}'.",
            )
        keys = list(file[measurement_path].keys())
    return sorted(keys)


__all__ = ["FileSpec", "list_measurement_keys", "list_specific_keys"]
