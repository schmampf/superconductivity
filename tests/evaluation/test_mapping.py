from __future__ import annotations

import numpy as np
import pytest

from superconductivity.evaluation import reduce, sample
from superconductivity.evaluation.sampling import SamplingSpec
from superconductivity.evaluation.traces import Trace, Traces
from superconductivity.utilities import mapping
from superconductivity.utilities.meta import TransportDatasetSpec, axis, data, label, param


def _make_transport_sample() -> TransportDatasetSpec:
    return TransportDatasetSpec(
        data=(
            data(
                "I_nA",
                np.asarray(
                    [
                        [1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 0.0, 1.0], dtype=np.float64), order=1),
            axis("Delta_meV", values=np.asarray([1.0, 2.0], dtype=np.float64), order=0),
        ),
        params=(param("GN_G0", 0.8),),
    )


def test_mapping_upsample_only_on_collection_axis() -> None:
    sample_in = _make_transport_sample()

    mapped = mapping(sample_in, axis="Aout_mV", N_up=2)

    assert np.allclose(mapped.Aout_mV.values, np.linspace(0.0, 1.0, 4))
    assert mapped.I_nA.values.shape == (4, 3)
    assert np.allclose(mapped.Delta_meV.values, np.linspace(1.0, 2.0, 4))
    assert float(mapped.GN_G0) == pytest.approx(0.8)


def test_mapping_remap_only_on_transport_axis() -> None:
    sample_in = _make_transport_sample()

    mapped = mapping(
        sample_in,
        axis="V_mV",
        xbins=np.asarray([-0.5, 0.5], dtype=np.float64),
    )

    assert np.allclose(mapped.V_mV.values, [-0.5, 0.5])
    assert mapped.I_nA.values.shape == (2, 2)
    assert np.allclose(mapped.Aout_mV.values, sample_in.Aout_mV.values)


def test_mapping_fill_only_on_collection_axis() -> None:
    sample_in = TransportDatasetSpec(
        data=(
            data(
                "I_nA",
                np.asarray(
                    [
                        [1.0, np.nan, 3.0],
                        [4.0, 6.0, 6.0],
                    ],
                    dtype=np.float64,
                ),
            ),
            data(
                "aux",
                np.asarray(
                    [
                        [10.0, np.nan, 30.0],
                        [40.0, 60.0, 60.0],
                    ],
                    dtype=np.float64,
                ),
            ),
        ),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 0.0, 1.0], dtype=np.float64), order=1),
            axis("Delta_meV", values=np.asarray([1.0, 2.0], dtype=np.float64), order=0),
        ),
        params=(),
    )

    mapped = mapping(sample_in, axis="Aout_mV", fill="nearest")

    assert np.isfinite(mapped.aux.values).all()
    assert np.allclose(mapped.aux.values[:, 1], [60.0, 60.0])


def test_mapping_combined_pipeline_has_fixed_order() -> None:
    sample_in = _make_transport_sample()

    mapped = mapping(
        sample_in,
        axis="Aout_mV",
        N_up=2,
        xbins=np.asarray([0.0, 0.5, 1.0], dtype=np.float64),
        fill="interpolate",
    )

    assert np.allclose(mapped.Aout_mV.values, [0.0, 0.5, 1.0])
    assert mapped.I_nA.values.shape == (3, 3)
    assert np.allclose(mapped.Delta_meV.values, [1.0, 1.5, 2.0])


def test_mapping_rejects_missing_axis_label() -> None:
    with pytest.raises(ValueError, match="missing axis"):
        mapping(_make_transport_sample(), axis="nu_GHz", N_up=2)


def test_mapping_rejects_non_finite_selected_axis() -> None:
    sample_in = TransportDatasetSpec(
        data=(data("I_nA", np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 1.0], dtype=np.float64), order=1),
        ),
        params=(),
    )
    sample_bad = TransportDatasetSpec(
        data=sample_in.data,
        axes=(
            sample_in.Aout_mV,
            axis("V_mV", values=np.asarray([-1.0, 1.0], dtype=np.float64), order=1),
        ),
    )
    object.__setattr__(sample_bad.V_mV, "values", np.asarray([-1.0, np.nan]))

    with pytest.raises(ValueError, match="selected axis must contain only finite"):
        mapping(sample_bad, axis="V_mV", N_up=2)


def test_mapping_rejects_invalid_xbins() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        mapping(
            _make_transport_sample(),
            axis="Aout_mV",
            xbins=np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
        )


def test_mapping_rejects_unsupported_fill_method() -> None:
    with pytest.raises(ValueError, match="Unsupported fill method"):
        mapping(_make_transport_sample(), axis="Aout_mV", fill="spline")


def test_mapping_rejects_ambiguous_broadcastable_entry() -> None:
    sample_in = TransportDatasetSpec(
        data=(data("I_nA", np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)),),
        axes=(
            axis("Aout_mV", values=np.asarray([0.0, 1.0], dtype=np.float64), order=0),
            axis("V_mV", values=np.asarray([-1.0, 1.0], dtype=np.float64), order=1),
        ),
        params=(param("aux_param", np.asarray([[1.0], [2.0]], dtype=np.float64)),),
    )

    with pytest.raises(ValueError, match="cannot be transformed unambiguously"):
        mapping(sample_in, axis="Aout_mV", N_up=2)


def test_mapping_integrates_with_sampled_transport_dataset() -> None:
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
    mapped = mapping(
        exp_v,
        axis="Aout_mV",
        N_up=2,
        xbins=np.asarray([1.0, 3.0, 5.0], dtype=np.float64),
        fill="interpolate",
    )

    assert np.allclose(mapped.Aout_mV.values, [1.0, 3.0, 5.0])
    assert mapped.I_nA.values.shape[0] == 3
    assert np.allclose(mapped.Delta_meV.values, [1.0, 1.1, 1.2])
