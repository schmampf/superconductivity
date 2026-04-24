from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.constants import G0_muS, kB_meV_K
from superconductivity.utilities.meta import (
    DataSpec,
    Dataset,
    TransportDatasetSpec,
    ParamSpec,
    axis,
    data,
    dataset,
    gridded_dataset,
    param,
    reduced_dataset,
    validate_gridded_dataset,
)


def test_dataset_label_lookup() -> None:
    dataset = Dataset(
        data=data("I_nA", [1.0, 2.0, 3.0]),
        axes=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        params=param("nu_Hz", 13.7),
    )

    assert dataset.I_nA.values.tolist() == [1.0, 2.0, 3.0]
    assert dataset.V_mV.values.tolist() == [-1.0, 0.0, 1.0]
    assert float(dataset.nu_Hz) == pytest.approx(13.7)
    assert "I_nA" in dataset
    assert "V_mV" in dataset
    assert "nu_Hz" in dataset


def test_dataset_ragged_payload_preserved() -> None:
    dataset = Dataset(
        data=[np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
        axes=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        params=param("nu_Hz", 13.7),
    )

    assert len(dataset.data) == 2
    assert dataset.data[0].code_label == "data_0"
    assert dataset.data[1].code_label == "data_1"
    assert dataset["data_0"].values.tolist() == [1.0, 2.0]
    assert dataset["data_1"].values.tolist() == [3.0, 4.0, 5.0]


def test_dataset_mapping_input() -> None:
    dataset = Dataset(
        data={"I_nA": [1.0, 2.0, 3.0]},
        axes={"V_mV": [-1.0, 0.0, 1.0]},
        params={"nu_Hz": 13.7},
    )

    assert dataset.I_nA.values.tolist() == [1.0, 2.0, 3.0]
    assert dataset.V_mV.values.tolist() == [-1.0, 0.0, 1.0]
    assert float(dataset.nu_Hz) == pytest.approx(13.7)


def test_dataset_rejects_duplicate_labels() -> None:
    with pytest.raises(ValueError, match="Duplicate code_label"):
        Dataset(
            data=data("I_nA", [1.0, 2.0]),
            axes=axis("I_nA", values=[-1.0, 0.0, 1.0], order=0),
        )


def test_dataset_constructor_helper() -> None:
    ds = dataset(
        V_mV=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        I_nA=data("I_nA", [1.0, 2.0, 3.0]),
        nu_Hz=param("nu_Hz", 13.7),
    )

    assert ds.V_mV.values.tolist() == [-1.0, 0.0, 1.0]
    assert ds.I_nA.values.tolist() == [1.0, 2.0, 3.0]
    assert float(ds.nu_Hz) == pytest.approx(13.7)


def test_dataset_constructor_helper_ignores_keyword_mismatch() -> None:
    ds = dataset(V_nV=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0))

    assert ds.V_mV.values.tolist() == [-1.0, 0.0, 1.0]
    assert "V_nV" not in ds


def test_dataset_add_and_remove() -> None:
    ds = dataset(
        V_mV=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        I_nA=data("I_nA", [1.0, 2.0, 3.0]),
    )

    ds2 = ds.add(dIdV_muS=data("dIdV_muS", [0.1, 0.2, 0.3]))
    assert ds2.dIdV_muS.values.tolist() == [0.1, 0.2, 0.3]
    assert ds2.V_mV.values.tolist() == [-1.0, 0.0, 1.0]

    ds3 = ds2.remove("I_nA", "dIdV_muS")
    assert "I_nA" not in ds3
    assert "dIdV_muS" not in ds3
    assert ds3.V_mV.values.tolist() == [-1.0, 0.0, 1.0]


def test_gridded_dataset_valid_aligned() -> None:
    ds = gridded_dataset(
        I_nA=data("I_nA", [1.0, 2.0, 3.0]),
        V_mV=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        nu_Hz=param("nu_Hz", 13.7),
    )
    assert ds.I_nA.values.tolist() == [1.0, 2.0, 3.0]
    assert ds.V_mV.values.tolist() == [-1.0, 0.0, 1.0]


def test_gridded_dataset_rejects_mismatched_data_shapes() -> None:
    with pytest.raises(ValueError, match="all data entries must have identical shape"):
        validate_gridded_dataset(
            Dataset(
                data=(
                    data("I_nA", [1.0, 2.0, 3.0]),
                    data("G_muS", [1.0, 2.0]),
                ),
                axes=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
                params=param("nu_Hz", 13.7),
            )
        )


def test_gridded_dataset_rejects_axis_data_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match data dimension"):
        validate_gridded_dataset(
            Dataset(
                data=data("I_nA", [1.0, 2.0, 3.0]),
                axes=axis("V_mV", values=[-1.0, 0.0], order=0),
                params=param("nu_Hz", 13.7),
            )
        )


def test_gridded_dataset_uses_axis_order_for_2d_data() -> None:
    ds = Dataset(
        data=data(
            "I_nA",
            np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64),
        ),
        axes=(
            axis("A_mV", values=[0.0, 0.1], order=0),
            axis("V_mV", values=[-1.0, 0.0, 1.0], order=1),
        ),
        params=param("nu_Hz", 13.7),
    )
    validate_gridded_dataset(ds)


def test_gridded_dataset_rejects_axis_order_out_of_bounds() -> None:
    ds = Dataset(
        data=data("I_nA", [1.0, 2.0, 3.0]),
        axes=axis("V_mV", values=[-1.0, 0.0, 1.0], order=1),
        params=param("nu_Hz", 13.7),
    )
    with pytest.raises(ValueError, match="out of bounds"):
        validate_gridded_dataset(ds)


def test_gridded_dataset_rejects_missing_required_labels() -> None:
    ds = Dataset(
        data=data("I_nA", [1.0, 2.0, 3.0]),
        axes=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        params=param("nu_Hz", 13.7),
    )
    with pytest.raises(ValueError, match="missing required data labels"):
        validate_gridded_dataset(ds, required_data=("G_muS",))
    with pytest.raises(ValueError, match="missing required axis labels"):
        validate_gridded_dataset(ds, required_axes=("A_mV",))
    with pytest.raises(ValueError, match="missing required param labels"):
        validate_gridded_dataset(ds, required_params=("N_up",))


def test_gridded_dataset_rejects_non_scalar_required_param() -> None:
    ds = Dataset(
        data=data("I_nA", [1.0, 2.0, 3.0]),
        axes=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        params=param("N_up", [10.0, 11.0]),
    )
    with pytest.raises(ValueError, match="must be scalar-valued"):
        validate_gridded_dataset(ds, required_params=("N_up",))


def test_gridded_dataset_enforces_finite_axis() -> None:
    bad_axis = axis("V_mV", values=[-1.0, 0.0, 1.0], order=0)
    object.__setattr__(bad_axis, "values", np.array([-1.0, np.nan, 1.0]))
    ds = Dataset(
        data=data("I_nA", [1.0, 2.0, 3.0]),
        axes=bad_axis,
        params=param("nu_Hz", 13.7),
    )
    with pytest.raises(ValueError, match="must be finite"):
        validate_gridded_dataset(ds)


def test_plain_dataset_still_allows_non_gridded_payloads() -> None:
    ds = dataset(
        data=[np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0])],
        V_mV=axis("V_mV", values=[-1.0, 0.0, 1.0], order=0),
        nu_Hz=13.7,
    )
    assert len(ds.data) == 2


def test_reduced_dataset_valid_voltage_bias() -> None:
    ds = reduced_dataset(
        V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
        I_nA=data("I_nA", [0.0, 2.0, 4.0]),
        Delta_meV=param("Delta_meV", 2.0),
        GN_G0=param("GN_G0", 4.0),
        gamma_meV=param("gamma_meV", 0.5),
    )

    assert isinstance(ds, TransportDatasetSpec)
    assert ds.V_mV.values.tolist() == [0.0, 1.0, 2.0]
    assert ds.I_nA.values.tolist() == [0.0, 2.0, 4.0]


def test_reduced_dataset_valid_current_bias() -> None:
    ds = reduced_dataset(
        I_nA=axis("I_nA", values=[0.0, 1.0, 2.0], order=0),
        V_mV=data("V_mV", [0.0, 2.0, 4.0]),
        Delta_meV=param("Delta_meV", 2.0),
        GN_G0=param("GN_G0", 4.0),
    )

    assert isinstance(ds, TransportDatasetSpec)
    assert ds.I_nA.values.tolist() == [0.0, 1.0, 2.0]
    assert ds.V_mV.values.tolist() == [0.0, 2.0, 4.0]


def test_reduced_dataset_rejects_both_transport_axes() -> None:
    with pytest.raises(
        ValueError,
        match="Exactly one of 'I_nA' and 'V_mV' must be a data entry",
    ):
        reduced_dataset(
            I_nA=axis("I_nA", values=[0.0, 1.0, 2.0], order=0),
            V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
            Delta_meV=param("Delta_meV", 2.0),
            GN_G0=param("GN_G0", 4.0),
        )


def test_reduced_dataset_rejects_both_transport_data() -> None:
    with pytest.raises(
        ValueError,
        match="Exactly one of 'I_nA' and 'V_mV' must be an axis entry",
    ):
        reduced_dataset(
            I_nA=data("I_nA", [0.0, 1.0, 2.0]),
            V_mV=data("V_mV", [0.0, 2.0, 4.0]),
            Delta_meV=param("Delta_meV", 2.0),
            GN_G0=param("GN_G0", 4.0),
        )


def test_reduced_dataset_rejects_missing_required_labels() -> None:
    with pytest.raises(ValueError, match="requires 'Delta_meV'"):
        reduced_dataset(
            V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
            I_nA=data("I_nA", [0.0, 2.0, 4.0]),
            GN_G0=param("GN_G0", 4.0),
        )
    with pytest.raises(ValueError, match="requires 'GN_G0'"):
        reduced_dataset(
            V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
            I_nA=data("I_nA", [0.0, 2.0, 4.0]),
            Delta_meV=param("Delta_meV", 2.0),
        )


def test_reduced_dataset_lazy_properties_voltage_bias() -> None:
    ds = reduced_dataset(
        V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
        I_nA=data("I_nA", [0.0, 2.0, 4.0]),
        Delta_meV=param("Delta_meV", 2.0),
        GN_G0=param("GN_G0", 4.0),
        gamma_meV=param("gamma_meV", 0.5),
        A_mV=param("A_mV", 0.25),
        nu_GHz=param("nu_GHz", 1.0),
        sigmaV_mV=param("sigmaV_mV", 0.0),
        T_K=param("T_K", 4.0),
    )

    G0_uS = float(G0_muS)
    GN_uS = 4.0 * G0_uS

    np.testing.assert_allclose(ds.dG_uS.values, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(ds.dR_MOhm.values, [0.5, 0.5, 0.5])
    np.testing.assert_allclose(ds.G_uS.values[1:], [2.0, 2.0])
    np.testing.assert_allclose(ds.R_MOhm.values[1:], [0.5, 0.5])
    assert np.isnan(np.asarray(ds.G_uS.values)[0])
    assert np.isnan(np.asarray(ds.R_MOhm.values)[0])
    np.testing.assert_allclose(
        ds.dG_G0.values,
        np.asarray(ds.dG_uS.values) / G0_uS,
    )
    np.testing.assert_allclose(
        ds.dG_GN.values,
        np.asarray(ds.dG_uS.values) / GN_uS,
    )
    np.testing.assert_allclose(
        ds.dR_R0.values,
        np.asarray(ds.dR_MOhm.values) * G0_uS,
    )
    np.testing.assert_allclose(
        ds.dR_RN.values,
        np.asarray(ds.dR_MOhm.values) * GN_uS,
    )
    np.testing.assert_allclose(
        ds.G_G0.values[1:],
        np.asarray(ds.G_uS.values)[1:] / G0_uS,
    )
    np.testing.assert_allclose(
        ds.G_GN.values[1:],
        np.asarray(ds.G_uS.values)[1:] / GN_uS,
    )
    np.testing.assert_allclose(
        ds.R_R0.values[1:],
        np.asarray(ds.R_MOhm.values)[1:] * G0_uS,
    )
    np.testing.assert_allclose(
        ds.R_RN.values[1:],
        np.asarray(ds.R_MOhm.values)[1:] * GN_uS,
    )
    np.testing.assert_allclose(ds.gamma_Delta.values, [0.25, 0.25, 0.25])
    np.testing.assert_allclose(
        ds.Tc_K.values,
        2.0 / (1.764 * float(kB_meV_K)),
    )
    np.testing.assert_allclose(ds.eV_Delta.values, [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        ds.eI_DeltaG0.values,
        np.asarray(ds.I_nA.values) / (2.0 * G0_uS),
    )
    np.testing.assert_allclose(
        ds.eI_DeltaGN.values,
        np.asarray(ds.I_nA.values) / (2.0 * GN_uS),
    )
    np.testing.assert_allclose(ds.eA_hnu.values, 0.25)
    np.testing.assert_allclose(ds.eA_Delta.values, 0.125)
    np.testing.assert_allclose(ds.hnu_Delta.values, 0.5)
    np.testing.assert_allclose(
        ds.T_Tc.values,
        4.0 / (2.0 / (1.764 * float(kB_meV_K))),
    )
    np.testing.assert_allclose(
        ds.DeltaT_meV.values,
        2.0 * np.tanh(1.74 * np.sqrt((2.0 / (1.764 * float(kB_meV_K))) / 4.0 - 1.0)),
    )
    np.testing.assert_allclose(
        ds.DeltaT_Delta.values,
        ds.DeltaT_meV.values / ds.Delta_meV.values,
    )
    np.testing.assert_allclose(ds.sigmaV_Delta.values, 0.0)
    assert isinstance(ds.eV_Delta, DataSpec)
    assert isinstance(ds.eA_hnu, DataSpec)
    assert isinstance(ds.T_Tc, DataSpec)


def test_reduced_dataset_lazy_properties_current_bias() -> None:
    ds = reduced_dataset(
        I_nA=axis("I_nA", values=[0.0, 1.0, 2.0], order=0),
        V_mV=data("V_mV", [0.0, 2.0, 4.0]),
        Delta_meV=param("Delta_meV", 2.0),
        GN_G0=param("GN_G0", 4.0),
    )

    np.testing.assert_allclose(ds.dR_MOhm.values, [2.0, 2.0, 2.0])
    np.testing.assert_allclose(ds.dG_uS.values, [0.5, 0.5, 0.5])


def test_reduced_dataset_gamma_delta_requires_gamma_mev() -> None:
    ds = reduced_dataset(
        V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
        I_nA=data("I_nA", [0.0, 2.0, 4.0]),
        Delta_meV=param("Delta_meV", 2.0),
        GN_G0=param("GN_G0", 4.0),
    )

    with pytest.raises(ValueError, match="gamma_meV"):
        _ = ds.gamma_Delta


def test_reduced_dataset_supports_dataset_methods() -> None:
    ds = reduced_dataset(
        V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
        I_nA=data("I_nA", [0.0, 2.0, 4.0]),
        Delta_meV=param("Delta_meV", 2.0),
        GN_G0=param("GN_G0", 4.0),
    )

    assert "V_mV" in ds.keys()
    ds2 = ds.add(gamma_meV=param("gamma_meV", 0.5))
    assert isinstance(ds2, TransportDatasetSpec)
    assert float(ds2.gamma_meV) == pytest.approx(0.5)

    ds3 = ds2.remove("gamma_meV")
    assert isinstance(ds3, TransportDatasetSpec)
    assert "gamma_meV" not in ds3


def test_transport_dataset_keys_include_available_lazy_properties() -> None:
    ds = reduced_dataset(
        V_mV=axis("V_mV", values=[0.0, 1.0, 2.0], order=0),
        I_nA=data("I_nA", [0.0, 2.0, 4.0]),
        Delta_meV=param("Delta_meV", 2.0),
        GN_G0=param("GN_G0", 4.0),
        gamma_meV=param("gamma_meV", 0.5),
        A_mV=param("A_mV", 0.25),
        nu_GHz=param("nu_GHz", 1.0),
        sigmaV_mV=param("sigmaV_mV", 0.0),
        T_K=param("T_K", 4.0),
    )

    keys = ds.keys()
    assert "I_nA" in keys
    assert "V_mV" in keys
    assert "dG_uS" in keys
    assert "eV_Delta" in keys
    assert "Tc_K" in keys
    assert "DeltaT_meV" in keys
    assert "gamma_Delta" in keys
    assert "sigmaV_Delta" in keys
    assert "dG_uS" in ds
    assert "gamma_Delta" in ds
    assert "sigmaV_Delta" in ds
    assert "hnu_Delta" in keys
    assert ds["dG_uS"].code_label == "dG_uS"
    assert ds["Tc_K"].code_label == "Tc_K"
    assert ds["sigmaV_Delta"].code_label == "sigmaV_Delta"
