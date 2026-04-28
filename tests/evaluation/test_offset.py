"""Tests for offset analysis built on IV trace containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.analysis.offset as offset
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.utilities.meta.axis import AxisSpec
from superconductivity.utilities.meta.label import LabelSpec
from superconductivity.utilities.meta.param import ParamSpec
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
    return Traces.from_fields(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
        specific_keys=["a", "b"],
        indices=[0, 1],
        yvalues=[1.0, 5.0],
        y_label=LabelSpec(
            code_label="Aout_mV",
            print_label="Aout_mV",
            html_label="Aout_mV",
            latex_label="Aout_mV",
        ),
    )


def _make_spec() -> offset.OffsetSpec:
    return offset.OffsetSpec(
        Vbins_mV=AxisSpec(
            code_label="Vbins_mV",
            print_label="Vbins_mV",
            html_label="Vbins_mV",
            latex_label="Vbins_mV",
            values=np.linspace(-2.0, 2.0, 81, dtype=np.float64),
            order=1,
        ),
        Ibins_nA=AxisSpec(
            code_label="Ibins_nA",
            print_label="Ibins_nA",
            html_label="Ibins_nA",
            latex_label="Ibins_nA",
            values=np.linspace(-4.0, 4.0, 81, dtype=np.float64),
            order=1,
        ),
        Voffscan_mV=AxisSpec(
            code_label="Voffscan_mV",
            print_label="Voffscan_mV",
            html_label="Voffscan_mV",
            latex_label="Voffscan_mV",
            values=np.asarray([-0.4, 0.0, 0.2, 0.4], dtype=np.float64),
            order=1,
        ),
        Ioffscan_nA=AxisSpec(
            code_label="Ioffscan_nA",
            print_label="Ioffscan_nA",
            html_label="Ioffscan_nA",
            latex_label="Ioffscan_nA",
            values=np.asarray([-0.3, 0.0, 0.1, 0.3], dtype=np.float64),
            order=1,
        ),
        nu_Hz=ParamSpec(
            code_label="nu_Hz",
            print_label="nu_Hz",
            html_label="nu_Hz",
            latex_label="nu_Hz",
            values=20.0,
            fixed=True,
        ),
        N_up=ParamSpec(
            code_label="N_up",
            print_label="N_up",
            html_label="N_up",
            latex_label="N_up",
            values=4,
            fixed=True,
        ),
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

    out = offset.offset_analysis(traces=trace, spec=_make_spec())

    assert isinstance(out, Dataset)
    assert isinstance(out.y, AxisSpec)
    assert isinstance(out.index, AxisSpec)
    assert isinstance(out.indices, AxisSpec)
    assert isinstance(out.i, AxisSpec)
    assert np.allclose(out.y.values, np.asarray([0.0]))
    assert np.allclose(out.index.values, np.asarray([0.0]))
    assert np.allclose(out.indices.values, np.asarray([0.0]))
    assert out.dGerr_G0.values.shape == (1, 4)
    assert out.dRerr_R0.values.shape == (1, 4)
    assert out.Voff_mV.values.shape == (1,)
    assert out.Ioff_nA.values.shape == (1,)
    assert out.keys()[:7] == ("y", "i", "indices", "skeys", "specific_keys", "Voffscan_mV", "Ioffscan_nA")
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
    assert isinstance(out.index, AxisSpec)
    assert isinstance(out.indices, AxisSpec)
    assert isinstance(out.i, AxisSpec)
    assert isinstance(out.Aout_mV, AxisSpec)
    assert np.allclose(out.y.values, np.asarray([1.0, 5.0]))
    assert np.allclose(out.index.values, np.asarray([0.0, 1.0]))
    assert np.allclose(out.indices.values, np.asarray([0.0, 1.0]))
    assert out.y.code_label == "Aout_mV"
    assert out.Aout_mV.code_label == "Aout_mV"
    assert out.Voffscan_mV.code_label == "Voffscan_mV"
    assert out.Ioffscan_nA.code_label == "Ioffscan_nA"
    assert out.dGerr_G0.values.shape == (2, 4)
    assert out.dRerr_R0.values.shape == (2, 4)
    assert out.Voff_mV.values.shape == (2,)
    assert out.Ioff_nA.values.shape == (2,)
    assert np.allclose(out.Voff_mV.values, np.asarray([0.4, 0.2]))
    assert np.allclose(out.Ioff_nA.values, np.asarray([0.3, 0.1]))
    assert out["Aout_mV"] is out.y
    assert out.keys()[:8] == ("y", "Aout_mV", "i", "indices", "skeys", "specific_keys", "Voffscan_mV", "Ioffscan_nA")


def test_offset_analysis_parallel_matches_single_worker() -> None:
    """Trace-level parallelism should preserve numerical results."""
    traces = _make_traces()
    spec_obj = _make_spec()

    serial = offset.offset_analysis(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
        backend="numpy",
        workers=1,
    )
    parallel = offset.offset_analysis(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
        backend="numpy",
        workers=2,
    )

    assert np.allclose(parallel.Voff_mV.values, serial.Voff_mV.values)
    assert np.allclose(parallel.Ioff_nA.values, serial.Ioff_nA.values)
    assert np.allclose(parallel.dGerr_G0.values, serial.dGerr_G0.values)
    assert np.allclose(parallel.dRerr_R0.values, serial.dRerr_R0.values)


def test_resolve_backend_auto_uses_numpy_for_parallel_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Auto backend should skip JAX when trace-level parallelism is enabled."""

    def _unexpected_import() -> tuple[object, object]:
        raise AssertionError("JAX import should not run for workers > 1.")

    monkeypatch.setattr(offset, "_import_jax_offset_kernels", _unexpected_import)

    assert offset._resolve_backend("auto", workers=2) == "numpy"


def test_resolve_backend_rejects_parallel_jax() -> None:
    """Parallel trace execution and JAX backend should be mutually exclusive."""
    with pytest.raises(ValueError, match="workers > 1"):
        offset._resolve_backend("jax", workers=2)


def test_get_offset_dispatches_to_jax_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit JAX backend should route through the JAX metric helper."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )
    spec_obj = _make_spec()

    monkeypatch.setattr(
        offset,
        "_import_jax_offset_kernels",
        lambda: (object(), object()),
    )

    def _fake_jax_metrics(
        v_mV: np.ndarray,
        i_nA: np.ndarray,
        spec: offset.OffsetSpec,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert v_mV.ndim == 1
        assert i_nA.ndim == 1
        assert spec is spec_obj
        return (
            np.asarray([3.0, 2.0, 1.0, 4.0]),
            np.asarray([5.0, 1.0, 2.0, 3.0]),
        )

    monkeypatch.setattr(offset, "_compute_offset_errors_jax", _fake_jax_metrics)

    out = offset.offset_analysis(traces=trace, spec=spec_obj, backend="jax")

    assert out.Voff_mV.values.shape == (1,)
    assert out.Ioff_nA.values.shape == (1,)
    assert out.Voff_mV.values[0] == pytest.approx(0.2)
    assert out.Ioff_nA.values[0] == pytest.approx(0.0)
