"""Tests for offset analysis built on IV trace containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.offset as offset
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.utilities.meta.axis import AxisSpec, axis
from superconductivity.utilities.meta.dataset import Dataset


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    v_shift_mV: float,
    i_shift_nA: float,
) -> Trace:
    t_s = np.linspace(0.0, 10.0, 401, dtype=np.float64)
    v_true_mV = np.linspace(-2.0, 2.0, t_s.size, dtype=np.float64)
    i_true_nA = v_true_mV + 0.2 * v_true_mV**3
    return Trace(
        V_mV=v_true_mV + v_shift_mV,
        I_nA=i_true_nA + i_shift_nA,
        t_s=t_s,
    )


def _make_traces() -> Traces:
    return Traces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
        skeys=["a", "b"],
        indices=[0, 1],
        yaxis=axis("Aout_mV", values=np.asarray([1.0, 5.0], dtype=np.float64), order=0),
    )


def _make_spec() -> offset.OffsetSpec:
    return offset.OffsetSpec(
        Vbins_mV=np.linspace(-2.0, 2.0, 81, dtype=np.float64),
        Ibins_nA=np.linspace(-4.0, 4.0, 81, dtype=np.float64),
        Voffscan_mV=np.asarray([-0.4, 0.0, 0.2, 0.4], dtype=np.float64),
        Ioffscan_nA=np.asarray([-0.3, 0.0, 0.1, 0.3], dtype=np.float64),
        cutoff_Hz=10.0,
        sampling_Hz=20.0,
    )


def test_offset_analysis_returns_single_dataset_for_single_trace() -> None:
    """Single-trace offset search should return one stacked dataset."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )

    out = offset.offset_analysis(traces=Traces(traces=[trace], skeys=["a"], indices=[0]), spec=_make_spec())

    assert isinstance(out, Dataset)
    assert isinstance(out.y, AxisSpec)
    assert np.allclose(out.y.values, np.asarray([0.0]))
    assert out.dGerr_G0.values.shape == (1, 4)
    assert out.dRerr_R0.values.shape == (1, 4)
    assert out.Voff_mV.values.shape == (1,)
    assert out.Ioff_nA.values.shape == (1,)
    assert out.keys()[:4] == ("y", "yaxis", "dGerr_G0", "dRerr_R0")
    assert out.Voffscan_mV.code_label == "Voffscan_mV"
    assert out.Ioffscan_nA.code_label == "Ioffscan_nA"


def test_offset_analysis_returns_stacked_dataset_for_collection() -> None:
    """Collection offset search should preserve trace order and y metadata."""
    traces = _make_traces()

    out = offset.offset_analysis(
        traces=traces,
        spec=_make_spec(),
        show_progress=False,
    )

    assert isinstance(out, Dataset)
    assert isinstance(out.y, AxisSpec)
    assert np.allclose(out.y.values, np.asarray([1.0, 5.0]))
    assert out.y.code_label == "Aout_mV"
    assert out.Voffscan_mV.code_label == "Voffscan_mV"
    assert out.Ioffscan_nA.code_label == "Ioffscan_nA"
    assert out.dGerr_G0.values.shape == (2, 4)
    assert out.dRerr_R0.values.shape == (2, 4)
    assert out.Voff_mV.values.shape == (2,)
    assert out.Ioff_nA.values.shape == (2,)
    assert np.allclose(out.Voff_mV.values, np.asarray([0.4, 0.2]))
    assert np.allclose(out.Ioff_nA.values, np.asarray([0.3, 0.1]))
    assert out["Aout_mV"] is out.y
    assert out.keys()[:4] == ("y", "yaxis", "dGerr_G0", "dRerr_R0")


def test_offset_analysis_parallel_matches_single_worker() -> None:
    """Trace-level parallelism should preserve numerical results."""
    traces = _make_traces()
    spec_obj = _make_spec()

    serial = offset.offset_analysis(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
    )
    parallel = offset.offset_analysis(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
        workers=2,
    )

    assert np.allclose(parallel.Voff_mV.values, serial.Voff_mV.values)
    assert np.allclose(parallel.Ioff_nA.values, serial.Ioff_nA.values)
    assert np.allclose(parallel.dGerr_G0.values, serial.dGerr_G0.values)
    assert np.allclose(parallel.dRerr_R0.values, serial.dRerr_R0.values)
