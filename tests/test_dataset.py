from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.meta import (
    Dataset,
    axis,
    data,
    dataset,
    gridded_dataset,
    param,
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
