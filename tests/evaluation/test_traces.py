"""Tests for IV trace loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import superconductivity.evaluation.traces.file as file_module
from superconductivity.evaluation.traces.file import FileSpec
from superconductivity.evaluation.traces.keys import Keys, KeysSpec
import superconductivity.evaluation.traces.traces as traces_module


class _FakeGroup:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]

    def keys(self) -> list[str]:
        return list(self._arrays.keys())


class _FakeFile:
    def __init__(self, groups: dict[str, _FakeGroup]) -> None:
        self._groups = groups

    def __enter__(self) -> _FakeFile:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __contains__(self, key: str) -> bool:
        return key in self._groups

    def __getitem__(self, key: str) -> _FakeGroup:
        return self._groups[key]


class _FakeH5py:
    def __init__(self, groups: dict[str, _FakeGroup]) -> None:
        self._file = _FakeFile(groups)

    def File(self, path: Path, mode: str) -> _FakeFile:
        assert path == Path("dummy.h5")
        assert mode == "r"
        return self._file


def _make_fake_h5py() -> _FakeH5py:
    groups = {
        "measurement/test": _FakeGroup(
            {
                "nu=1dBm": np.asarray([]),
                "nu=5dBm": np.asarray([]),
            },
        ),
        "measurement/test/nu=1dBm": _FakeGroup({}),
        "measurement/test/nu=1dBm/sweep/adwin": _FakeGroup(
            {
                "time": np.asarray([10.0, 11.0, 12.0, 13.0]),
                "V1": np.asarray([1e-3, 2e-3, 9e-3, 4e-3]),
                "V2": np.asarray([1e-9, 2e-9, 9e-9, 4e-9]),
                "trigger": np.asarray([1, 1, 2, 1]),
            },
        ),
        "measurement/test/nu=1dBm/offset/adwin": _FakeGroup(
            {
                "V1": np.asarray([1e-3, 1e-3]),
                "V2": np.asarray([1e-9, 1e-9]),
            },
        ),
        "measurement/test/nu=5dBm": _FakeGroup({}),
        "measurement/test/nu=5dBm/sweep/adwin": _FakeGroup(
            {
                "time": np.asarray([20.0, 21.0, 22.0, 23.0]),
                "V1": np.asarray([2e-3, 4e-3, 8e-3, 6e-3]),
                "V2": np.asarray([2e-9, 4e-9, 8e-9, 6e-9]),
                "trigger": np.asarray([1, 1, 2, 1]),
            },
        ),
        "measurement/test/nu=5dBm/offset/adwin": _FakeGroup(
            {
                "V1": np.asarray([2e-3, 2e-3]),
                "V2": np.asarray([2e-9, 2e-9]),
            },
        ),
    }
    return _FakeH5py(groups)


def _patch_fake_h5(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        traces_module,
        "_import_h5py",
        lambda: _make_fake_h5py(),
    )
    monkeypatch.setattr(file_module, "_import_h5py", lambda: _make_fake_h5py())


def test_get_traces_selects_one_trace_by_specific_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Specific-key selection should return a one-trace collection."""
    _patch_fake_h5(monkeypatch)

    traces = traces_module.get_traces(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        keysspec=KeysSpec(strip1="dBm"),
        tracespec=traces_module.TraceSpec(
            amp_voltage=1.0,
            amp_current=1.0,
            r_ref_ohm=1.0,
            trigger_values=1,
        ),
        specific_key="nu=1dBm",
    )
    trace = traces[0]

    assert len(traces) == 1
    assert isinstance(trace, dict)
    assert trace["meta"].specific_key == "nu=1dBm"
    assert trace["meta"].index == 0
    assert trace["meta"].yvalue == pytest.approx(1.0)
    assert np.allclose(trace["I_nA"], np.asarray([0.0, 1.0, 3.0]))
    assert np.allclose(trace["V_mV"], np.asarray([0.0, 1.0, 3.0]))
    assert np.allclose(trace["t_s"], np.asarray([0.0, 1.0, 3.0]))


def test_get_traces_accepts_filespec_with_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FileSpec should provide the measurement implicitly."""
    _patch_fake_h5(monkeypatch)

    traces = traces_module.get_traces(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        keysspec=KeysSpec(strip1="dBm"),
        tracespec=traces_module.TraceSpec(
            amp_voltage=1.0,
            amp_current=1.0,
            r_ref_ohm=1.0,
            trigger_values=1,
        ),
        specific_key="nu=1dBm",
    )
    trace = traces[0]

    assert trace["meta"].specific_key == "nu=1dBm"
    assert trace["meta"].yvalue == pytest.approx(1.0)


def test_get_traces_resolves_one_trace_by_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Value selection should resolve the matching trace."""
    _patch_fake_h5(monkeypatch)
    monkeypatch.setattr(
        traces_module,
        "get_keys",
        lambda **_kwargs: Keys.from_fields(
            specific_keys=["nu=1dBm", "nu=5dBm"],
            indices=np.asarray([0, 1], dtype=np.int64),
            yvalues=np.asarray([1.0, 5.0]),
            spec=KeysSpec(
                strip0="=",
                strip1="dBm",
                html_label="<i>nu</i> (dBm)",
            ),
        ),
    )

    traces = traces_module.get_traces(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        tracespec=traces_module.TraceSpec(
            amp_voltage=1.0,
            amp_current=1.0,
            r_ref_ohm=1.0,
            trigger_values=1,
        ),
        yvalue=5.0,
    )
    trace = traces[0]

    assert trace["meta"].specific_key == "nu=5dBm"
    assert trace["meta"].index == 0
    assert trace["meta"].yvalue == pytest.approx(5.0)


def test_get_traces_returns_collection_with_lookup_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection loading should provide list views and selectors."""
    _patch_fake_h5(monkeypatch)

    traces = traces_module.get_traces(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        keys=Keys.from_fields(
            specific_keys=["nu=1dBm", "nu=5dBm"],
            indices=np.asarray([0, 1], dtype=np.int64),
            yvalues=np.asarray([1.0, 5.0]),
            spec=KeysSpec(strip0="=", strip1="dBm"),
        ),
        tracespec=traces_module.TraceSpec(
            amp_voltage=1.0,
            amp_current=1.0,
            r_ref_ohm=1.0,
            trigger_values=1,
        ),
    )

    assert len(traces) == 2
    assert traces.specific_keys == ["nu=1dBm", "nu=5dBm"]
    assert np.allclose(traces.yvalues, np.asarray([1.0, 5.0]))
    assert traces[0]["meta"].specific_key == "nu=1dBm"
    assert np.allclose(traces[1]["I_nA"], np.asarray([0.0, 2.0, 4.0]))
    assert len(traces.I_nA) == 2
    assert len(traces.V_mV) == 2
    assert len(traces.t_s) == 2


def test_get_traces_accepts_filespec_keys_and_ivspec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The high-level trace loader should accept the spec objects."""
    _patch_fake_h5(monkeypatch)

    traces = traces_module.get_traces(
        filespec=FileSpec(h5path="dummy.h5", measurement="test"),
        keys=Keys.from_fields(
            specific_keys=["nu=1dBm", "nu=5dBm"],
            indices=np.asarray([0, 1], dtype=np.int64),
            yvalues=np.asarray([1.0, 5.0]),
            spec=KeysSpec(strip0="=", strip1="dBm"),
        ),
        tracespec=traces_module.TraceSpec(
            amp_voltage=1.0,
            amp_current=1.0,
            r_ref_ohm=1.0,
            trigger_values=1,
        ),
    )

    assert len(traces) == 2
    assert traces.specific_keys == ["nu=1dBm", "nu=5dBm"]
    assert np.allclose(traces.yvalues, np.asarray([1.0, 5.0]))


def test_get_traces_rejects_mismatched_keys_and_yvalues() -> None:
    """Collection metadata must stay aligned."""
    with pytest.raises(ValueError, match="same length"):
        traces_module.get_traces(
            filespec=FileSpec(h5path="dummy.h5", measurement="test"),
            keys=Keys.from_fields(
                specific_keys=["nu=1dBm"],
                indices=np.asarray([0], dtype=np.int64),
                yvalues=np.asarray([1.0, 5.0]),
                spec=KeysSpec(strip0="=", strip1="dBm"),
            ),
        )
