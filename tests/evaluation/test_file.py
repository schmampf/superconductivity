"""Tests for HDF5 file specifications."""

from __future__ import annotations

from pathlib import Path

import superconductivity.evaluation.traces.file as file_module
from superconductivity.evaluation.traces.file import FileSpec


class _FakeMeasurementGroup:
    def __init__(self, keys_list: list[str]) -> None:
        self._keys = keys_list

    def keys(self) -> list[str]:
        return list(self._keys)


class _FakeFile:
    def __init__(self, groups: dict[str, _FakeMeasurementGroup]) -> None:
        self._groups = groups

    def __enter__(self) -> _FakeFile:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __contains__(self, key: str) -> bool:
        return key in self._groups

    def __getitem__(self, key: str) -> _FakeMeasurementGroup:
        return self._groups[key]


class _FakeH5py:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._file = _FakeFile(
            {
                "measurement": _FakeMeasurementGroup(["a", "b"]),
                "measurement/test": _FakeMeasurementGroup(
                    ["nu=1dBm", "nu=5dBm"],
                ),
            },
        )

    def File(self, path: Path, mode: str) -> _FakeFile:
        assert path == self._path
        assert mode == "r"
        return self._file


def test_filespec_resolves_relative_path_and_lists_measurements(
    monkeypatch,
) -> None:
    """FileSpec.mkeys should resolve relative paths against location."""
    expected_path = Path("/tmp/root/data.h5")
    monkeypatch.setattr(
        file_module,
        "_import_h5py",
        lambda: _FakeH5py(expected_path),
    )

    spec = FileSpec(
        h5path="data.h5",
        location="/tmp/root",
        measurement="test",
    )

    assert spec.path == expected_path
    assert spec.mkeys() == ["a", "b"]


def test_filespec_specific_keys_use_embedded_measurement(
    monkeypatch,
) -> None:
    """FileSpec.skeys should use the configured measurement."""
    expected_path = Path("/tmp/root/data.h5")
    monkeypatch.setattr(
        file_module,
        "_import_h5py",
        lambda: _FakeH5py(expected_path),
    )

    spec = FileSpec(
        h5path="data.h5",
        location="/tmp/root",
        measurement="test",
    )

    assert spec.skeys() == ["nu=1dBm", "nu=5dBm"]
