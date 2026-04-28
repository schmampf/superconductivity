from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation import reduce, sample
from superconductivity.evaluation.sampling import SamplingSpec
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.utilities.meta import (
    TransportDatasetSpec,
    axis,
    data,
    label,
    switch_bias,
)


def _make_exp_v() -> TransportDatasetSpec:
    return TransportDatasetSpec(
        data=(
            data(
                "I_nA",
                np.asarray(
                    [
                        [-2.0, 0.0, 2.0],
                        [-1.0, 1.0, 3.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 0.0, 1.0], dtype=np.float64), order=1),
            axis("Delta_meV", values=np.asarray([1.0, 1.2], dtype=np.float64), order=0),
        ),
    )


def _make_exp_i() -> TransportDatasetSpec:
    return TransportDatasetSpec(
        data=(
            data(
                "V_mV",
                np.asarray(
                    [
                        [-1.0, 0.0, 1.0],
                        [-0.5, 0.5, 1.5],
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),
            axis("I_nA", values=np.asarray([-2.0, 0.0, 2.0], dtype=np.float64), order=1),
            axis("Delta_meV", values=np.asarray([1.0, 1.2], dtype=np.float64), order=0),
        ),
    )


def test_switch_bias_converts_exp_v_to_exp_i_with_raw_axis() -> None:
    exp_v = _make_exp_v()

    exp_i = switch_bias(exp_v, I_nA=np.linspace(-2.0, 2.0, 5))

    assert exp_i.I_nA.order == 1
    assert np.allclose(exp_i.I_nA.values, np.linspace(-2.0, 2.0, 5))
    assert exp_i.V_mV.values.shape == (2, 5)
    assert np.allclose(exp_i.Aout_mV.values, [0.0, 1.0])
    assert np.allclose(exp_i.Delta_meV.values, [1.0, 1.2])


def test_switch_bias_converts_exp_i_to_exp_v_with_raw_axis() -> None:
    exp_i = _make_exp_i()

    exp_v = switch_bias(exp_i, V_mV=np.linspace(-1.0, 1.0, 5))

    assert exp_v.V_mV.order == 1
    assert np.allclose(exp_v.V_mV.values, np.linspace(-1.0, 1.0, 5))
    assert exp_v.I_nA.values.shape == (2, 5)
    assert np.allclose(exp_v.Aout_mV.values, [0.0, 1.0])
    assert np.allclose(exp_v.Delta_meV.values, [1.0, 1.2])


def test_switch_bias_accepts_axis_spec_target() -> None:
    exp_v = _make_exp_v()

    exp_i = switch_bias(
        exp_v,
        I_nA=axis("I_nA", values=np.linspace(-2.0, 2.0, 5), order=1),
    )

    assert np.allclose(exp_i.I_nA.values, np.linspace(-2.0, 2.0, 5))


def test_switch_bias_passes_through_upsampling_and_fill() -> None:
    exp_v = TransportDatasetSpec(
        data=(
            data(
                "I_nA",
                np.asarray([[-2.0, np.nan, 2.0]], dtype=np.float64),
            ),
        ),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 0.0, 1.0], dtype=np.float64), order=1),
        ),
    )

    exp_i = switch_bias(
        exp_v,
        I_nA=np.linspace(-2.0, 2.0, 5),
        N_up=2,
        fill="interpolate",
    )

    assert exp_i.V_mV.values.shape == (1, 5)
    assert np.isfinite(exp_i.V_mV.values).any()


def test_switch_bias_rejects_both_or_neither_targets() -> None:
    exp_v = _make_exp_v()

    with pytest.raises(ValueError, match="Exactly one of I_nA or V_mV"):
        switch_bias(exp_v)

    with pytest.raises(ValueError, match="Exactly one of I_nA or V_mV"):
        switch_bias(
            exp_v,
            I_nA=np.linspace(-2.0, 2.0, 5),
            V_mV=np.linspace(-1.0, 1.0, 5),
        )


def test_switch_bias_rejects_invalid_target_axis() -> None:
    exp_v = _make_exp_v()

    with pytest.raises(ValueError, match="strictly increasing"):
        switch_bias(exp_v, I_nA=np.asarray([0.0, 0.0, 1.0], dtype=np.float64))


def test_switch_bias_rejects_incompatible_source_shape() -> None:
    with pytest.raises(ValueError, match="missing axis 'V_mV'"):
        switch_bias(_make_exp_i(), I_nA=np.linspace(-1.0, 1.0, 5))


def test_switch_bias_rejects_non_monotonic_transport_relation() -> None:
    exp_v = TransportDatasetSpec(
        data=(data("I_nA", np.asarray([[0.0, 1.0, 0.5]], dtype=np.float64)),),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 0.0, 1.0], dtype=np.float64), order=1),
        ),
    )

    with pytest.raises(ValueError, match="strictly monotonic"):
        switch_bias(exp_v, I_nA=np.linspace(0.0, 1.0, 5))


def test_switch_bias_integrates_with_sampled_transport_dataset() -> None:
    t_s = np.linspace(0.0, 10.0, 401, dtype=np.float64)
    v_true_mV = np.linspace(-2.0, 2.0, t_s.size, dtype=np.float64)
    i_true_nA = v_true_mV + 0.2 * v_true_mV**3
    traces = Traces.from_fields(
        traces=[
            Trace(V_mV=v_true_mV + 0.4, I_nA=i_true_nA + 0.3, t_s=t_s),
            Trace(V_mV=v_true_mV + 0.2, I_nA=i_true_nA + 0.1, t_s=t_s),
        ],
        specific_keys=["a", "b"],
        indices=[0, 1],
        yvalues=[1.0, 5.0],
        y_label=label("Aout_mV"),
    )
    spec = SamplingSpec(
        Vbins_mV=np.linspace(-2.0, 2.0, 81),
        Ibins_nA=np.linspace(-4.0, 4.0, 81),
        apply_smoothing=False,
        nu_Hz=40.0,
        N_up=4,
        Voff_mV=data("Voff_mV", np.asarray([0.4, 0.2], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3, 0.1], dtype=np.float64)),
    )

    exp_v, exp_i = sample(traces, samplingspec=spec, show_progress=False)
    exp_v, _ = reduce(exp_v, exp_i, Delta_meV=np.asarray([1.0, 1.2], dtype=np.float64))
    switched = switch_bias(exp_v, I_nA=np.linspace(-3.0, 3.0, 121))

    assert switched.I_nA.order == 1
    assert switched.V_mV.values.shape == (2, 121)
    assert np.allclose(switched.Aout_mV.values, [1.0, 5.0])
    assert np.allclose(switched.Delta_meV.values, [1.0, 1.2])
