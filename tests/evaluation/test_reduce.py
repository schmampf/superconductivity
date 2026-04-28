from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation import reduce
from superconductivity.utilities.meta import TransportDatasetSpec, axis, data, param


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


def test_reduce_adds_scalar_params_to_both_samples() -> None:
    exp_v, exp_i = _make_samples()

    red_v, red_i = reduce(exp_v, exp_i, Delta_meV=1.35, GN_G0=0.8, nu_GHz=13.7)

    assert float(red_v.Delta_meV) == pytest.approx(1.35)
    assert float(red_i.Delta_meV) == pytest.approx(1.35)
    assert float(red_v.GN_G0) == pytest.approx(0.8)
    assert float(red_i.GN_G0) == pytest.approx(0.8)
    assert float(red_v.nu_GHz) == pytest.approx(13.7)
    assert float(red_i.nu_GHz) == pytest.approx(13.7)


def test_reduce_promotes_collection_length_arrays_to_axes() -> None:
    exp_v, exp_i = _make_samples()

    red_v, red_i = reduce(
        exp_v,
        exp_i,
        Delta_meV=np.asarray([1.1, 1.2, 1.3], dtype=np.float64),
    )

    assert red_v.Delta_meV.order == 0
    assert red_i.Delta_meV.order == 0
    assert np.allclose(red_v.Delta_meV.values, [1.1, 1.2, 1.3])
    assert np.allclose(red_i.Delta_meV.values, [1.1, 1.2, 1.3])


def test_reduce_accepts_explicit_param_and_axis_specs() -> None:
    exp_v, exp_i = _make_samples()

    red_v, red_i = reduce(
        exp_v,
        exp_i,
        GN_G0=param("GN_G0", 0.7),
        nu_GHz=axis("nu_GHz", values=np.asarray([10.0, 11.0, 12.0], dtype=np.float64), order=0),
    )

    assert float(red_v.GN_G0) == pytest.approx(0.7)
    assert float(red_i.GN_G0) == pytest.approx(0.7)
    assert np.allclose(red_v.nu_GHz.values, [10.0, 11.0, 12.0])
    assert np.allclose(red_i.nu_GHz.values, [10.0, 11.0, 12.0])


def test_reduce_replaces_existing_entries() -> None:
    exp_v, exp_i = _make_samples()
    exp_v = exp_v.add(Delta_meV=1.0)
    exp_i = exp_i.add(Delta_meV=1.0)

    red_v, red_i = reduce(exp_v, exp_i, Delta_meV=1.5)

    assert float(red_v.Delta_meV) == pytest.approx(1.5)
    assert float(red_i.Delta_meV) == pytest.approx(1.5)


def test_reduce_rejects_missing_collection_axis() -> None:
    exp_v = TransportDatasetSpec(
        data=(data("I_nA", np.asarray([1.0, 2.0], dtype=np.float64)),),
        axes=(axis("V_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),),
    )
    exp_i = TransportDatasetSpec(
        data=(data("V_mV", np.asarray([1.0, 2.0], dtype=np.float64)),),
        axes=(axis("I_nA", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),),
    )

    with pytest.raises(ValueError, match="exactly one collection axis"):
        reduce(exp_v, exp_i, Delta_meV=1.0)


def test_reduce_rejects_mismatched_collection_axes() -> None:
    exp_v, exp_i = _make_samples()
    exp_i_bad = TransportDatasetSpec(
        data=exp_i.data,
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 2.0, 4.0], dtype=np.float64), order=0),
            axis("I_nA", values=np.asarray([-2.0, 2.0], dtype=np.float64), order=1),
        ),
    )

    with pytest.raises(ValueError, match="identical values"):
        reduce(exp_v, exp_i_bad, Delta_meV=1.0)


def test_reduce_rejects_raw_arrays_with_wrong_shape() -> None:
    exp_v, exp_i = _make_samples()

    with pytest.raises(ValueError, match="must be scalar or a 1D array matching"):
        reduce(
            exp_v,
            exp_i,
            Delta_meV=np.asarray([1.0, 2.0], dtype=np.float64),
        )
