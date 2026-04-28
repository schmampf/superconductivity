from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation import CalibrationSpec, calibrate
from superconductivity.evaluation.traces import KeysSpec
from superconductivity.utilities.meta import Dataset, axis, data
from superconductivity.utilities.meta.axis import AxisSpec
from superconductivity.utilities.meta.label import label
from superconductivity.utilities.functions.binning import bin


def _make_samples() -> Dataset:
    return Dataset(
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
        axes=(axis("y", values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64), order=0),),
    )


def _make_keysspec() -> KeysSpec:
    return KeysSpec(label=label("A_mV"))


def test_axis_spec_validates_axis_and_kind() -> None:
    axis_spec = axis("A_mV", values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64), order=7)
    np.testing.assert_allclose(axis_spec.axis, [0.0, 1.0, 2.0])
    assert axis_spec.code_label == "A_mV"
    assert axis_spec.order == 7


def test_function_calibration_updates_trace_metadata() -> None:
    samples = _make_samples()
    axisspec = axis("A_mV", values=np.asarray([0.0, 2.0, 4.0], dtype=np.float64), order=1)
    spec = CalibrationSpec(
        mode="function",
        transform=lambda y, a: y * a,
        params=2.0,
        gap_fill="nearest",
    )

    result = calibrate(
        samples=samples,
        axisspec=axisspec,
        keysspec=_make_keysspec(),
        calibrationspec=spec,
    )

    np.testing.assert_allclose(result.source_axis, [0.0, 1.0, 2.0])
    np.testing.assert_allclose(result.mapped_axis, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(result.calibrated_axis, [0.0, 2.0, 4.0])
    np.testing.assert_allclose(result.samples["y"].values, [0.0, 2.0, 4.0])
    assert result.axisspec.code_label == "A_mV"
    assert result.axisspec.print_label == "A (mV)"


def test_function_calibration_accepts_scalar_and_tuple_params() -> None:
    samples = _make_samples()
    axisspec = axis("A_mV", values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64), order=1)

    scalar_result = calibrate(
        samples=samples,
        axisspec=axisspec,
        keysspec=_make_keysspec(),
        calibrationspec=CalibrationSpec(
            mode="function",
            transform=lambda y, a: y + a,
            params=1.0,
        ),
    )
    tuple_result = calibrate(
        samples=samples,
        axisspec=axisspec,
        keysspec=_make_keysspec(),
        calibrationspec=CalibrationSpec(
            mode="function",
            transform=lambda y, a: y + a,
            params=(1.0,),
        ),
    )

    np.testing.assert_allclose(scalar_result.mapped_axis, tuple_result.mapped_axis)


def test_lookup_calibration_uses_table_mapping() -> None:
    samples = _make_samples()
    axisspec = axis("A_mV", values=np.asarray([0.0, 10.0, 20.0], dtype=np.float64), order=1)
    spec = CalibrationSpec(
        mode="lookup",
        lookup=np.asarray([0.0, 5.0, 10.0], dtype=np.float64),
        gap_fill="nan",
    )

    result = calibrate(
        samples=samples,
        axisspec=axisspec,
        keysspec=_make_keysspec(),
        calibrationspec=spec,
    )

    np.testing.assert_allclose(result.mapped_axis, [0.0, 5.0, 10.0])
    np.testing.assert_allclose(result.samples["y"].values, [0.0, 10.0, 20.0])


def test_gap_fill_nearest_fills_missing_values() -> None:
    samples = _make_samples()
    axisspec = axis("A_mV", values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64), order=1)
    spec = CalibrationSpec(
        mode="function",
        transform=lambda y, a: np.asarray([y[0] * a, np.nan, y[2] * a]),
        params=2.0,
        gap_fill="nearest",
    )

    result = calibrate(
        samples=samples,
        axisspec=axisspec,
        keysspec=_make_keysspec(),
        calibrationspec=spec,
    )

    np.testing.assert_allclose(result.mapped_axis, [0.0, 0.0, 4.0])


def test_gap_fill_interpolate_fills_missing_values() -> None:
    samples = _make_samples()
    axisspec = axis("A_mV", values=np.asarray([0.0, 1.0, 2.0], dtype=np.float64), order=1)
    spec = CalibrationSpec(
        mode="function",
        transform=lambda y, a: np.asarray([y[0] * a, np.nan, y[2] * a]),
        params=2.0,
        gap_fill="interpolate",
    )

    result = calibrate(
        samples=samples,
        axisspec=axisspec,
        keysspec=_make_keysspec(),
        calibrationspec=spec,
    )

    np.testing.assert_allclose(result.mapped_axis, [0.0, 2.0, 4.0])


def test_invalid_axis_spec_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="order must be"):
        AxisSpec(
            values=np.asarray([0.0, 1.0], dtype=np.float64),
            code_label="A_mV",
            print_label="A (mV)",
            html_label="<i>A</i> (mV)",
            latex_label=r"$A$ (mV)",
            order=-1,
        )


def test_invalid_lookup_grid_is_rejected() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        CalibrationSpec(
            mode="lookup",
            lookup=np.asarray([0.0, 1.0, 1.0], dtype=np.float64),
        )


def test_calibrated_output_is_compatible_with_binning() -> None:
    samples = _make_samples()
    axisspec = axis("A_mV", values=np.asarray([0.0, 2.0, 4.0], dtype=np.float64), order=1)
    result = calibrate(
        samples=samples,
        axisspec=axisspec,
        keysspec=_make_keysspec(),
        calibrationspec=CalibrationSpec(
            mode="function",
            transform=lambda y, a: y * a,
            params=2.0,
        ),
    )

    rebinned = bin(result.mapped_axis, result.source_axis, axisspec.axis)
    assert rebinned.shape == result.mapped_axis.shape
