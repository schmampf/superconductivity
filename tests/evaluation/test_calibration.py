from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation import CalibrationSpec, calibrate, sample
from superconductivity.evaluation.sampling import SamplingSpec
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.utilities.meta import TransportDatasetSpec, axis, data


def _make_samples() -> tuple[TransportDatasetSpec, TransportDatasetSpec]:
    exp_v = TransportDatasetSpec(
        data=(
            data(
                "I_nA",
                np.asarray(
                    [
                        [10.0, 20.0],
                        [11.0, 21.0],
                        [12.0, 22.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 1.0], dtype=np.float64), order=1),
        ),
    )
    exp_i = TransportDatasetSpec(
        data=(
            data(
                "V_mV",
                np.asarray(
                    [
                        [30.0, 40.0],
                        [31.0, 41.0],
                        [32.0, 42.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64), order=0),
            axis("I_nA", values=np.asarray([-2.0, 2.0], dtype=np.float64), order=1),
        ),
    )
    return exp_v, exp_i


def test_function_calibration_relabels_collection_axis() -> None:
    exp_v, exp_i = _make_samples()

    cal_v, cal_i = calibrate(
        exp_v,
        exp_i,
        calibrationspec=CalibrationSpec(
            label="A_mV",
            transform=lambda y, a: y * a,
            params=2.0,
        ),
    )

    assert np.allclose(cal_v.A_mV.values, [0.0, 2.0, 4.0])
    assert np.allclose(cal_i.A_mV.values, [0.0, 2.0, 4.0])
    assert np.allclose(cal_v.I_nA.values, exp_v.I_nA.values, equal_nan=True)
    assert np.allclose(cal_i.V_mV.values, exp_i.V_mV.values, equal_nan=True)
    assert not hasattr(cal_v, "Aout_mV")
    assert not hasattr(cal_i, "Aout_mV")


def test_function_calibration_accepts_scalar_and_tuple_params() -> None:
    exp_v, exp_i = _make_samples()

    scalar_v, scalar_i = calibrate(
        exp_v,
        exp_i,
        calibrationspec=CalibrationSpec(
            label="A_mV",
            transform=lambda y, a: y + a,
            params=1.0,
        ),
    )
    tuple_v, tuple_i = calibrate(
        exp_v,
        exp_i,
        calibrationspec=CalibrationSpec(
            label="A_mV",
            transform=lambda y, a: y + a,
            params=(1.0,),
        ),
    )

    assert np.allclose(scalar_v.A_mV.values, tuple_v.A_mV.values)
    assert np.allclose(scalar_i.A_mV.values, tuple_i.A_mV.values)


def test_lookup_calibration_replaces_collection_axis_by_index() -> None:
    exp_v, exp_i = _make_samples()

    cal_v, cal_i = calibrate(
        exp_v,
        exp_i,
        calibrationspec=CalibrationSpec(
            label="A_mV",
            lookup=np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
        ),
    )

    assert np.allclose(cal_v.A_mV.values, [0.0, 5.0, 10.0])
    assert np.allclose(cal_i.A_mV.values, [0.0, 5.0, 10.0])


def test_calibration_rejects_missing_collection_axis() -> None:
    exp_v = TransportDatasetSpec(
        data=(data("I_nA", np.asarray([1.0, 2.0], dtype=np.float64)),),
        axes=(axis("V_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),),
    )
    exp_i = TransportDatasetSpec(
        data=(data("V_mV", np.asarray([1.0, 2.0], dtype=np.float64)),),
        axes=(axis("I_nA", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),),
    )

    with pytest.raises(ValueError, match="exactly one collection axis"):
        calibrate(
            exp_v,
            exp_i,
            calibrationspec=CalibrationSpec(
                label="A_mV",
                transform=lambda y, a: y * a,
                params=2.0,
            ),
        )


def test_calibration_rejects_mismatched_collection_axes() -> None:
    exp_v, exp_i = _make_samples()
    exp_i_bad = TransportDatasetSpec(
        data=exp_i.data,
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 2.0, 4.0], dtype=np.float64), order=0),
            exp_i.I_nA,
        ),
    )

    with pytest.raises(ValueError, match="identical values"):
        calibrate(
            exp_v,
            exp_i_bad,
            calibrationspec=CalibrationSpec(
                label="A_mV",
                transform=lambda y, a: y * a,
                params=2.0,
            ),
        )


def test_calibration_rejects_wrong_transform_shape() -> None:
    exp_v, exp_i = _make_samples()

    with pytest.raises(ValueError, match="same shape"):
        calibrate(
            exp_v,
            exp_i,
            calibrationspec=CalibrationSpec(
                label="A_mV",
                transform=lambda y, a: y[:-1] * a,
                params=2.0,
            ),
        )


def test_calibration_rejects_non_finite_transform_output() -> None:
    exp_v, exp_i = _make_samples()

    with pytest.raises(ValueError, match="must be finite"):
        calibrate(
            exp_v,
            exp_i,
            calibrationspec=CalibrationSpec(
                label="A_mV",
                transform=lambda y, a: np.asarray([0.0, np.nan, 2.0]),
                params=2.0,
            ),
        )


def test_calibration_rejects_lookup_length_mismatch() -> None:
    exp_v, exp_i = _make_samples()

    with pytest.raises(ValueError, match="same shape"):
        calibrate(
            exp_v,
            exp_i,
            calibrationspec=CalibrationSpec(
                label="A_mV",
                lookup=np.asarray([0.0, 1.0], dtype=np.float64),
            ),
        )


def test_calibration_spec_rejects_invalid_transform_lookup_configuration() -> None:
    with pytest.raises(ValueError, match="exactly one of transform or lookup"):
        CalibrationSpec(label="A_mV")

    with pytest.raises(ValueError, match="exactly one of transform or lookup"):
        CalibrationSpec(
            label="A_mV",
            transform=lambda y, a: y * a,
            lookup=np.asarray([0.0, 1.0], dtype=np.float64),
        )


def test_sample_then_calibrate_only_changes_collection_axis() -> None:
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
        y_label=None,
    )
    samplingspec = SamplingSpec(
        Vbins_mV=np.linspace(-2.0, 2.0, 81),
        Ibins_nA=np.linspace(-4.0, 4.0, 81),
        apply_smoothing=False,
        nu_Hz=40.0,
        N_up=4,
        Voff_mV=data("Voff_mV", np.asarray([0.4, 0.2], dtype=np.float64)),
        Ioff_nA=data("Ioff_nA", np.asarray([0.3, 0.1], dtype=np.float64)),
    )

    exp_v, exp_i = sample(traces, samplingspec=samplingspec, show_progress=False)
    cal_v, cal_i = calibrate(
        exp_v,
        exp_i,
        calibrationspec=CalibrationSpec(
            label="A_mV",
            transform=lambda y, a: y * a,
            params=2.0,
        ),
    )

    assert np.allclose(cal_v.A_mV.values, [2.0, 10.0])
    assert np.allclose(cal_i.A_mV.values, [2.0, 10.0])
    assert np.allclose(cal_v.I_nA.values, exp_v.I_nA.values, equal_nan=True)
    assert np.allclose(cal_i.V_mV.values, exp_i.V_mV.values, equal_nan=True)
