"""Tests for offset analysis built on IV trace containers."""

from __future__ import annotations

import numpy as np
import pytest

import superconductivity.evaluation.offset as offset
from superconductivity.evaluation.ivdata import IVTrace, IVTraces


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    v_shift_mV: float,
    i_shift_nA: float,
) -> IVTrace:
    t_s = np.linspace(0.0, 10.0, 401, dtype=np.float64)
    v_true_mV = np.linspace(-2.0, 2.0, t_s.size, dtype=np.float64)
    i_true_nA = v_true_mV + 0.2 * v_true_mV**3
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "V_mV": v_true_mV + v_shift_mV,
        "I_nA": i_true_nA + i_shift_nA,
        "t_s": t_s,
    }


def _make_spec() -> offset.OffsetSpec:
    return offset.OffsetSpec(
        Vbins_mV=np.linspace(-2.0, 2.0, 81),
        Ibins_nA=np.linspace(-4.0, 4.0, 81),
        Voff_mV=np.asarray([-0.4, 0.0, 0.2, 0.4]),
        Ioff_nA=np.asarray([-0.3, 0.0, 0.1, 0.3]),
        nu_Hz=20.0,
        upsample=4,
    )


def test_get_offset_returns_offsettrace_for_single_trace() -> None:
    """Single-trace offset search should keep metadata and best offsets."""
    trace = _make_iv_trace(
        specific_key="a",
        index=0,
        yvalue=1.0,
        v_shift_mV=0.4,
        i_shift_nA=0.3,
    )

    out = offset.get_offset(trace=trace, spec=_make_spec())

    assert out["specific_key"] == "a"
    assert out["index"] == 0
    assert out["yvalue"] == pytest.approx(1.0)
    assert out["Voff_mV"] == pytest.approx(0.4)
    assert out["Ioff_nA"] == pytest.approx(0.3)
    assert out["dGerr_G0"].shape == (4,)
    assert out["dRerr_R0"].shape == (4,)


def test_get_offsets_returns_collection_with_lookup_methods() -> None:
    """Collection offset search should mirror the IV/PSD container API."""
    traces = IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )

    out = offset.get_offsets(
        traces=traces,
        spec=_make_spec(),
        show_progress=False,
    )

    assert out.spec.nu_Hz == pytest.approx(20.0)
    assert out.keys == ["a", "b"]
    assert np.allclose(out.yvalues, np.asarray([1.0, 5.0]))
    assert np.allclose(out.Voff_mV, np.asarray([0.4, 0.2]))
    assert np.allclose(out.Ioff_nA, np.asarray([0.3, 0.1]))
    assert out.dGerr_G0.shape == (2, 4)
    assert out.dRerr_R0.shape == (2, 4)
    assert out.by_key("a")["Voff_mV"] == pytest.approx(0.4)
    assert out.by_value(5.0)["specific_key"] == "b"


def test_get_offsets_parallel_matches_single_worker() -> None:
    """Trace-level parallelism should preserve numerical results."""
    traces = IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )
    spec_obj = _make_spec()

    serial = offset.get_offsets(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
        backend="numpy",
        workers=1,
    )
    parallel = offset.get_offsets(
        traces=traces,
        spec=spec_obj,
        show_progress=False,
        backend="numpy",
        workers=2,
    )

    assert np.allclose(parallel.Voff_mV, serial.Voff_mV)
    assert np.allclose(parallel.Ioff_nA, serial.Ioff_nA)
    assert np.allclose(parallel.dGerr_G0, serial.dGerr_G0)
    assert np.allclose(parallel.dRerr_R0, serial.dRerr_R0)


def test_offsettraces_by_value_rejects_ambiguous_matches() -> None:
    """Duplicate yvalues should require the plural lookup method."""
    traces = offset.OffsetTraces(
        spec=_make_spec(),
        traces=[
            {
                "specific_key": "a",
                "index": 0,
                "yvalue": 1.0,
                "dGerr_G0": np.asarray([0.0]),
                "dRerr_R0": np.asarray([0.0]),
                "Voff_mV": 0.0,
                "Ioff_nA": 0.0,
            },
            {
                "specific_key": "b",
                "index": 1,
                "yvalue": 1.0,
                "dGerr_G0": np.asarray([0.0]),
                "dRerr_R0": np.asarray([0.0]),
                "Voff_mV": 0.0,
                "Ioff_nA": 0.0,
            },
        ],
    )

    with pytest.raises(ValueError, match="matches multiple traces"):
        traces.by_value(1.0)

    matches = traces.all_by_value(1.0)
    assert [trace["specific_key"] for trace in matches] == ["a", "b"]


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

    out = offset.get_offset(trace=trace, spec=spec_obj, backend="jax")

    assert out["Voff_mV"] == pytest.approx(0.2)
    assert out["Ioff_nA"] == pytest.approx(0.0)
