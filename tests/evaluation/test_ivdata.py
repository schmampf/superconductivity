"""Tests for IV trace loading helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import superconductivity.evaluation.ivdata as ivdata


class _FakeGroup:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        self._arrays = arrays

    def __getitem__(self, key: str) -> np.ndarray:
        return self._arrays[key]


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
    monkeypatch.setattr(ivdata, "_import_h5py", lambda: _make_fake_h5py())


def _make_trace(
    specific_key: str,
    index: int,
    yvalue: float,
) -> ivdata.IVTrace:
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "I_nA": np.asarray([float(index)], dtype=np.float64),
        "V_mV": np.asarray([float(index)], dtype=np.float64),
        "t_s": np.asarray([float(index)], dtype=np.float64),
    }


def test_get_iv_returns_typed_dict_for_specific_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct specific-key loading should include parsed metadata."""
    _patch_fake_h5(monkeypatch)

    trace = ivdata.get_iv(
        h5path="dummy.h5",
        measurement="test",
        specific_key="nu=1dBm",
        strip1="dBm",
        amp_voltage=1.0,
        amp_current=1.0,
        r_ref_ohm=1.0,
        trigger_values=1,
    )

    assert isinstance(trace, dict)
    assert trace["specific_key"] == "nu=1dBm"
    assert trace["index"] is None
    assert trace["yvalue"] == pytest.approx(1.0)
    assert np.allclose(trace["I_nA"], np.asarray([0.0, 1.0, 3.0]))
    assert np.allclose(trace["V_mV"], np.asarray([0.0, 1.0, 3.0]))
    assert np.allclose(trace["t_s"], np.asarray([0.0, 1.0, 3.0]))


def test_get_iv_resolves_trace_by_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Value selection should resolve the matching trace."""
    _patch_fake_h5(monkeypatch)
    monkeypatch.setattr(
        ivdata,
        "list_specific_keys_and_values",
        lambda **_kwargs: (
            ["nu=1dBm", "nu=5dBm"],
            np.asarray([1.0, 5.0]),
        ),
    )

    trace = ivdata.get_iv(
        h5path="dummy.h5",
        measurement="test",
        yvalue=5.0,
        amp_voltage=1.0,
        amp_current=1.0,
        r_ref_ohm=1.0,
        trigger_values=1,
    )

    assert trace["specific_key"] == "nu=5dBm"
    assert trace["index"] == 1
    assert trace["yvalue"] == pytest.approx(5.0)


def test_get_ivs_returns_collection_with_lookup_methods(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Collection loading should provide list views and selectors."""
    _patch_fake_h5(monkeypatch)

    traces = ivdata.get_ivs(
        h5path="dummy.h5",
        measurement="test",
        keys=["nu=1dBm", "nu=5dBm"],
        yvalues=np.asarray([1.0, 5.0]),
        amp_voltage=1.0,
        amp_current=1.0,
        r_ref_ohm=1.0,
        trigger_values=1,
    )

    assert len(traces) == 2
    assert traces.keys == ["nu=1dBm", "nu=5dBm"]
    assert np.allclose(traces.yvalues, np.asarray([1.0, 5.0]))
    assert np.allclose(traces[1]["I_nA"], np.asarray([0.0, 2.0, 4.0]))
    assert traces.by_key("nu=1dBm")["index"] == 0
    assert traces.by_value(5.0)["specific_key"] == "nu=5dBm"
    assert len(traces.I_nA) == 2
    assert len(traces.V_mV) == 2
    assert len(traces.t_s) == 2


def test_get_ivs_rejects_mismatched_keys_and_yvalues() -> None:
    """Collection metadata must stay aligned."""
    with pytest.raises(ValueError, match="same length"):
        ivdata.get_ivs(
            h5path="dummy.h5",
            measurement="test",
            keys=["nu=1dBm"],
            yvalues=np.asarray([1.0, 5.0]),
        )


def test_ivtraces_by_value_rejects_ambiguous_matches() -> None:
    """Duplicate y-values should require the plural lookup method."""
    traces = ivdata.IVTraces(
        traces=[
            _make_trace("a", 0, 1.0),
            _make_trace("b", 1, 1.0),
        ],
    )

    with pytest.raises(ValueError, match="matches multiple traces"):
        traces.by_value(1.0)

    matches = traces.all_by_value(1.0)
    assert [trace["specific_key"] for trace in matches] == ["a", "b"]


def test_ivtraces_by_key_rejects_ambiguous_matches() -> None:
    """Duplicate keys should require the plural lookup method."""
    traces = ivdata.IVTraces(
        traces=[
            _make_trace("a", 0, 1.0),
            _make_trace("a", 1, 2.0),
        ],
    )

    with pytest.raises(ValueError, match="matches multiple traces"):
        traces.by_key("a")

    matches = traces.all_by_key("a")
    assert [trace["index"] for trace in matches] == [0, 1]
