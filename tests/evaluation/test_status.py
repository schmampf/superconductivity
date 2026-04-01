"""Tests for raw status and measurement data discovery."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from superconductivity.evaluation import (
    get_measurement_keys,
    get_measurement_series,
    get_status_keys,
    get_status_series,
)
from superconductivity.evaluation.traces.file import FileSpec
import superconductivity.evaluation.traces.data as data_module


class _FakeDataset:
    def __init__(self, data: np.ndarray) -> None:
        self._data = np.asarray(data)

    @property
    def dtype(self) -> np.dtype:
        return self._data.dtype

    def __array__(self, dtype=None) -> np.ndarray:
        return np.asarray(self._data, dtype=dtype)


class _NoArrayDataset(_FakeDataset):
    def __array__(self, dtype=None) -> np.ndarray:
        raise AssertionError("discovery should not read dataset payloads.")


class _FakeGroup:
    def __init__(self, children: Mapping[str, object]) -> None:
        self._children = dict(children)

    def __contains__(self, key: str) -> bool:
        try:
            self[key]
        except KeyError:
            return False
        return True

    def __getitem__(self, key: str) -> object:
        if "/" in key:
            head, tail = key.split("/", 1)
            if head not in self._children:
                raise KeyError(key)
            child = self._children[head]
            if not isinstance(child, _FakeGroup):
                raise KeyError(key)
            return child[tail]

        if key not in self._children:
            raise KeyError(key)
        return self._children[key]

    def keys(self) -> list[str]:
        return list(self._children.keys())


class _FakeFile(_FakeGroup):
    def __enter__(self) -> _FakeFile:
        return self

    def __exit__(self, *_args: object) -> None:
        return None


class _FakeH5py:
    def __init__(self, *, path: Path, file: _FakeFile) -> None:
        self._path = path
        self._file = file

    def File(self, path: Path, mode: str) -> _FakeFile:
        assert path == self._path
        assert mode == "r"
        return self._file


def _structured_dataset(
    fields: dict[str, list[float]],
) -> _FakeDataset:
    names = list(fields.keys())
    length = len(fields[names[0]])
    dtype = [(name, np.float64) for name in names]
    data = np.zeros(length, dtype=dtype)
    for name, values in fields.items():
        data[name] = np.asarray(values, dtype=np.float64)
    return _FakeDataset(data)


def _plain_dataset(values: list[float]) -> _FakeDataset:
    return _FakeDataset(np.asarray(values, dtype=np.float64))


def _structured_dataset_no_array(
    fields: dict[str, list[float]],
) -> _NoArrayDataset:
    names = list(fields.keys())
    length = len(fields[names[0]])
    dtype = [(name, np.float64) for name in names]
    data = np.zeros(length, dtype=dtype)
    for name, values in fields.items():
        data[name] = np.asarray(values, dtype=np.float64)
    return _NoArrayDataset(data)


def _adwin_group(times: list[float]) -> _FakeGroup:
    time_array = np.asarray(times, dtype=np.float64)
    return _FakeGroup(
        {
            "time": _FakeDataset(time_array),
            "V1": _FakeDataset(time_array + 100.0),
            "V2": _FakeDataset(time_array + 200.0),
        },
    )


def _make_status_file(
    *,
    include_global_status: bool = True,
    include_sweep_extras: bool = True,
    include_offset_extras: bool = True,
    include_offset_adwin: bool = False,
    missing_adwin_time: bool = False,
) -> _FakeFile:
    nu1_sweep_children: dict[str, object] = {
        "adwin": (
            _FakeGroup({"V1": _FakeDataset(np.zeros(3, dtype=np.float64))})
            if missing_adwin_time
            else _adwin_group([10.0, 11.0, 12.0])
        ),
    }
    nu5_sweep_children: dict[str, object] = {
        "adwin": _adwin_group([20.0, 21.0, 22.0, 23.0]),
    }
    nu1_offset_children: dict[str, object] = {}
    nu5_offset_children: dict[str, object] = {}

    if include_sweep_extras:
        nu1_sweep_children["bluefors"] = _structured_dataset(
            {
                "time": [9.0, 10.0, 12.0, 13.0],
                "Tsample": [100.0, 101.0, 102.0, 103.0],
            },
        )
        nu1_sweep_children["vna"] = _FakeGroup(
            {
                "amplitude": _structured_dataset(
                    {
                        "time": [9.0, 10.5, 11.5],
                        "value": [1.0, 2.0, 3.0],
                    },
                ),
            },
        )
        nu1_sweep_children["note"] = _plain_dataset([1.0, 2.0, 3.0])

        nu5_sweep_children["bluefors"] = _structured_dataset(
            {
                "time": [18.0, 22.0, 24.0],
                "Tsample": [200.0, 201.0, 202.0],
            },
        )
        nu5_sweep_children["vna"] = _FakeGroup(
            {
                "amplitude": _structured_dataset(
                    {
                        "time": [21.5, 24.0],
                        "value": [4.0, 5.0],
                    },
                ),
            },
        )

    if include_offset_extras:
        if include_offset_adwin:
            nu1_offset_children["adwin"] = _adwin_group([9.0, 14.0])
            nu5_offset_children["adwin"] = _adwin_group([19.5, 24.5])
        nu1_offset_children["bluefors"] = _structured_dataset(
            {
                "time": [9.0, 10.5, 12.5, 14.0],
                "Tsample": [300.0, 301.0, 302.0, 303.0],
            },
        )
        nu5_offset_children["bluefors"] = _structured_dataset(
            {
                "time": [19.5, 22.5, 24.5],
                "Tsample": [400.0, 401.0, 402.0],
            },
        )

    measurement_group = _FakeGroup(
        {
            "test": _FakeGroup(
                {
                    "nu=1dBm": _FakeGroup(
                        {
                            "sweep": _FakeGroup(nu1_sweep_children),
                            "offset": _FakeGroup(nu1_offset_children),
                        },
                    ),
                    "nu=5dBm": _FakeGroup(
                        {
                            "sweep": _FakeGroup(nu5_sweep_children),
                            "offset": _FakeGroup(nu5_offset_children),
                        },
                    ),
                },
            ),
        },
    )

    root_children: dict[str, object] = {"measurement": measurement_group}

    if include_global_status:
        root_children["status"] = _FakeGroup(
            {
                "bluefors": _FakeGroup(
                    {
                        "temperature": _FakeGroup(
                            {
                                "MCBJ": _structured_dataset(
                                    {
                                        "time": [9.0, 10.0, 15.0, 23.0, 30.0],
                                        "T": [1.0, 2.0, 3.0, 4.0, 5.0],
                                    },
                                ),
                            },
                        ),
                    },
                ),
                "femtos": _structured_dataset(
                    {
                        "time": [10.0, 16.0, 21.0],
                        "amp_A": [1.0, 2.0, 3.0],
                        "amp_B": [4.0, 5.0, 6.0],
                    },
                ),
                "motor": _FakeGroup(
                    {
                        "position": _structured_dataset(
                            {
                                "time": [5.0, 10.0, 20.0, 25.0],
                                "value": [0.0, 1.0, 2.0, 3.0],
                            },
                        ),
                    },
                ),
                "pressure": _FakeGroup(
                    {
                        "cond": _structured_dataset(
                            {
                                "time": [0.0, 5.0],
                                "value": [7.0, 8.0],
                            },
                        ),
                    },
                ),
                "ignore": _FakeGroup({"plain": _plain_dataset([1.0, 2.0])}),
            },
        )

    return _FakeFile(root_children)


def _patch_fake_h5(
    monkeypatch: pytest.MonkeyPatch,
    *,
    fake_file: _FakeFile,
) -> None:
    monkeypatch.setattr(
        data_module.file_module,
        "_import_h5py",
        lambda: _FakeH5py(path=Path("dummy.h5"), file=fake_file),
    )


def test_get_status_keys_returns_flat_logical_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global status discovery should return logical keys and sources."""
    _patch_fake_h5(monkeypatch, fake_file=_make_status_file())

    mapping = get_status_keys(
        filespec=FileSpec(h5path="dummy.h5"),
    )

    assert mapping == {
        "bluefors/temperature/MCBJ/T": "status/bluefors/temperature/MCBJ/T",
        "femtos/amp_A": "status/femtos/amp_A",
        "femtos/amp_B": "status/femtos/amp_B",
        "motor/position/value": "status/motor/position/value",
        "pressure/cond/value": "status/pressure/cond/value",
    }


def test_get_status_keys_uses_metadata_only_for_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Status key discovery should not materialize structured datasets."""
    fake_file = _make_status_file(include_sweep_extras=False)
    status_group = fake_file["status"]
    assert isinstance(status_group, _FakeGroup)
    status_group._children["bluefors"] = _FakeGroup(
        {
            "temperature": _FakeGroup(
                {
                    "MCBJ": _structured_dataset_no_array(
                        {
                            "time": [9.0, 10.0, 15.0],
                            "T": [1.0, 2.0, 3.0],
                        },
                    ),
                },
            ),
        },
    )
    status_group._children["femtos"] = _structured_dataset_no_array(
        {
            "time": [10.0, 16.0, 21.0],
            "amp_A": [1.0, 2.0, 3.0],
            "amp_B": [4.0, 5.0, 6.0],
        },
    )
    status_group._children["motor"] = _FakeGroup(
        {
            "position": _structured_dataset_no_array(
                {
                    "time": [5.0, 10.0, 20.0],
                    "value": [0.0, 1.0, 2.0],
                },
            ),
        },
    )

    _patch_fake_h5(monkeypatch, fake_file=fake_file)

    mapping = get_status_keys(filespec=FileSpec(h5path="dummy.h5"))

    assert mapping["bluefors/temperature/MCBJ/T"] == (
        "status/bluefors/temperature/MCBJ/T"
    )


def test_get_status_series_returns_cropped_time_value_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Global status retrieval should crop one field to the measurement."""
    _patch_fake_h5(monkeypatch, fake_file=_make_status_file())

    time_s, value = get_status_series(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        key="bluefors/temperature/MCBJ/T",
    )

    assert np.allclose(time_s, np.asarray([10.0, 15.0, 23.0]))
    assert np.allclose(value, np.asarray([2.0, 3.0, 4.0]))

    time_s, value = get_status_series(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        key="femtos/amp_A",
    )
    assert np.allclose(time_s, np.asarray([10.0, 16.0, 21.0]))
    assert np.allclose(value, np.asarray([1.0, 2.0, 3.0]))


def test_get_measurement_keys_returns_aggregated_flat_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measurement discovery should return aggregated wildcard sources."""
    _patch_fake_h5(monkeypatch, fake_file=_make_status_file())

    mapping = get_measurement_keys(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
    )

    assert mapping == {
        "adwin/V1": "measurement/test/*/*/adwin/V1",
        "adwin/V2": "measurement/test/*/*/adwin/V2",
        "bluefors/Tsample": "measurement/test/*/*/bluefors/Tsample",
        "vna/amplitude/value": "measurement/test/*/*/vna/amplitude/value",
    }


def test_get_measurement_series_appends_and_sorts_all_specific_keys(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measurement retrieval should append blocks across all specific keys."""
    _patch_fake_h5(monkeypatch, fake_file=_make_status_file())

    time_s, value = get_measurement_series(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        key="bluefors/Tsample",
    )
    assert np.allclose(
        time_s,
        np.asarray(
            [10.0, 10.5, 12.0, 12.5, 13.0, 14.0, 18.0, 19.5, 22.0, 22.5],
        ),
    )
    assert np.allclose(
        value,
        np.asarray(
            [101.0, 301.0, 102.0, 302.0, 103.0, 303.0, 200.0, 400.0, 201.0, 401.0],
        ),
    )

    time_s, value = get_measurement_series(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        key="vna/amplitude/value",
    )
    assert np.allclose(time_s, np.asarray([10.5, 11.5, 21.5]))
    assert np.allclose(value, np.asarray([2.0, 3.0, 4.0]))

    time_s, value = get_measurement_series(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        key="adwin/V1",
    )
    assert np.allclose(
        time_s,
        np.asarray([10.0, 11.0, 12.0, 20.0, 21.0, 22.0, 23.0]),
    )
    assert np.allclose(
        value,
        np.asarray([110.0, 111.0, 112.0, 120.0, 121.0, 122.0, 123.0]),
    )


def test_get_measurement_series_includes_offset_adwin_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measurement retrieval should include offset adwin blocks too."""
    _patch_fake_h5(
        monkeypatch,
        fake_file=_make_status_file(include_offset_adwin=True),
    )

    time_s, value = get_measurement_series(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        key="adwin/V1",
    )

    assert np.allclose(
        time_s,
        np.asarray(
            [9.0, 10.0, 11.0, 12.0, 14.0, 19.5, 20.0, 21.0, 22.0, 23.0, 24.5],
        ),
    )
    assert np.allclose(
        value,
        np.asarray(
            [
                109.0,
                110.0,
                111.0,
                112.0,
                114.0,
                119.5,
                120.0,
                121.0,
                122.0,
                123.0,
                124.5,
            ],
        ),
    )


def test_data_discovery_skips_datasets_without_time_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Plain arrays should not appear in status or measurement key maps."""
    _patch_fake_h5(monkeypatch, fake_file=_make_status_file())

    status_keys = get_status_keys(filespec=FileSpec(h5path="dummy.h5"))
    measurement_keys = get_measurement_keys(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
    )

    assert "ignore/plain" not in status_keys
    assert "note" not in measurement_keys


def test_missing_measurement_raises_for_retrieval_modules() -> None:
    """Series retrieval should require a configured measurement."""
    with pytest.raises(ValueError, match="measurement is required"):
        get_status_series(
            filespec=FileSpec(h5path="dummy.h5"),
            key="bluefors/temperature/MCBJ/T",
        )

    with pytest.raises(ValueError, match="measurement is required"):
        get_measurement_keys(
            filespec=FileSpec(h5path="dummy.h5"),
        )

    with pytest.raises(ValueError, match="measurement is required"):
        get_measurement_series(
            filespec=FileSpec(h5path="dummy.h5"),
            key="bluefors/Tsample",
        )


def test_missing_sweep_adwin_time_data_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing sweep/adwin time data should fail series retrieval."""
    _patch_fake_h5(
        monkeypatch,
        fake_file=_make_status_file(missing_adwin_time=True),
    )

    with pytest.raises(KeyError, match="time field not found"):
        get_status_series(
            filespec=FileSpec(h5path="dummy.h5", measurement="test"),
            key="bluefors/temperature/MCBJ/T",
        )

    with pytest.raises(KeyError, match="time field not found"):
        get_measurement_series(
            filespec=FileSpec(h5path="dummy.h5", measurement="test"),
            key="bluefors/Tsample",
        )


def test_no_global_status_tree_still_allows_measurement_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing global status should not block measurement-local access."""
    _patch_fake_h5(
        monkeypatch,
        fake_file=_make_status_file(include_global_status=False),
    )

    assert get_status_keys(filespec=FileSpec(h5path="dummy.h5")) == {}
    assert get_measurement_keys(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
    ) == {
        "adwin/V1": "measurement/test/*/*/adwin/V1",
        "adwin/V2": "measurement/test/*/*/adwin/V2",
        "bluefors/Tsample": "measurement/test/*/*/bluefors/Tsample",
        "vna/amplitude/value": "measurement/test/*/*/vna/amplitude/value",
    }


def test_no_measurement_extras_still_allows_global_status_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Measurement-local extras may be absent while status still exists."""
    _patch_fake_h5(
        monkeypatch,
        fake_file=_make_status_file(
            include_sweep_extras=False,
            include_offset_extras=False,
        ),
    )

    assert get_measurement_keys(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
    ) == {
        "adwin/V1": "measurement/test/*/*/adwin/V1",
        "adwin/V2": "measurement/test/*/*/adwin/V2",
    }
    assert get_status_keys(filespec=FileSpec(h5path="dummy.h5")) == {
        "bluefors/temperature/MCBJ/T": "status/bluefors/temperature/MCBJ/T",
        "femtos/amp_A": "status/femtos/amp_A",
        "femtos/amp_B": "status/femtos/amp_B",
        "motor/position/value": "status/motor/position/value",
        "pressure/cond/value": "status/pressure/cond/value",
    }


def test_public_exports_expose_new_data_functions() -> None:
    """Top-level evaluation exports should expose the new data API only."""
    import superconductivity.api as api_module
    import superconductivity.evaluation as evaluation_module

    assert hasattr(api_module, "get_status_keys")
    assert hasattr(api_module, "get_status_series")
    assert hasattr(api_module, "get_measurement_keys")
    assert hasattr(api_module, "get_measurement_series")
    assert not hasattr(api_module, "get_status")

    assert hasattr(evaluation_module, "get_status_keys")
    assert hasattr(evaluation_module, "get_status_series")
    assert hasattr(evaluation_module, "get_measurement_keys")
    assert hasattr(evaluation_module, "get_measurement_series")
    assert not hasattr(evaluation_module, "get_status")
