from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.meta.axis import AxisSpec, axis
from superconductivity.utilities.meta.dataset import Dataset, dataset
from superconductivity.utilities.meta.dataset import Dataset, dataset
from superconductivity.utilities.meta.label import LABELS, LabelSpec, label
from superconductivity.utilities.meta.param import ParamSpec, param
from superconductivity.utilities.constants import G0_muS, h_Vs, kB_meV_K


def test_label_meta_construction() -> None:
    meta = label("V_mV")
    assert isinstance(meta, LabelSpec)
    assert meta.code_label == "V_mV"
    assert meta.print_label == "V_mV"
    assert meta.html_label == "<i>V</i> (mV)"
    assert meta.latex_label == r"$V$ (mV)"


def test_label_lookup_returns_matching_meta() -> None:
    meta = label("V_mV")
    assert meta == LABELS["V_mV"]


def test_label_lookup_rejects_unknown_name() -> None:
    with pytest.raises(KeyError, match="Unknown label"):
        label("missing")


def test_label_meta_reuse() -> None:
    meta = label("dIdV")
    spec = axis("dIdV", values=[0.0, 0.5, 1.0])
    assert spec.code_label == meta.code_label
    assert spec.print_label == meta.print_label
    assert spec.html_label == meta.html_label
    assert spec.latex_label == meta.latex_label


def test_axis_spec_inherits_label_meta() -> None:
    spec = axis("V_mV", -1.0, 1.0, 5, order=2)
    assert isinstance(spec, LabelSpec)
    assert spec.code_label == "V_mV"
    assert spec.print_label == "V_mV"
    assert spec.order == 2


def test_param_spec_inherits_label_meta() -> None:
    spec = param("tau", value=0.1, lower=0.0, upper=1.0)
    assert isinstance(spec, LabelSpec)
    assert spec.code_label == "tau"
    assert spec.print_label == "tau"
    assert spec.value == 0.1
    assert spec.lower == 0.0
    assert spec.upper == 1.0


def test_param_spec_accepts_sequence_error() -> None:
    spec = param(
        "tau",
        value=[0.1, 0.2],
        error=[0.01, 0.02],
    )
    np.testing.assert_allclose(spec.value, [0.1, 0.2])
    np.testing.assert_allclose(spec.error, [0.01, 0.02])


def test_param_spec_rejects_mismatched_error_shape() -> None:
    with pytest.raises(ValueError, match="error must match value shape"):
        param("tau", value=[0.1, 0.2], error=0.01)


def test_dataset_collects_axes_and_params() -> None:
    x_axis = axis("V_mV", 0.0, 1.0, 5)
    temperature = param("T_K", value=0.12, fixed=True)

    spec = dataset("measurement", [1.0, 2.0, 3.0], axes=x_axis, params=temperature)

    assert isinstance(spec, Dataset)
    assert spec.code_label == "measurement"
    assert spec.print_label == "measurement"
    np.testing.assert_allclose(spec.values, [1.0, 2.0, 3.0])
    assert spec.axes == (x_axis,)
    assert spec.params == (temperature,)


def test_construct_axis_uses_values_over_linspace() -> None:
    spec = axis("V_mV", values=np.asarray([1.0, 2.0, 4.0], dtype=np.float64))
    np.testing.assert_allclose(spec.axis, [1.0, 2.0, 4.0])


def test_axis_V_mV_uses_linspace_semantics() -> None:
    spec = axis("V_mV", -1.0, 1.0, 5, order=2)
    np.testing.assert_allclose(spec.axis, np.linspace(-1.0, 1.0, 5))
    assert spec.code_label == "V_mV"
    assert spec.latex_label == r"$V$ (mV)"


def test_axis_I_nA_accepts_values_override() -> None:
    spec = axis("I_nA", values=[0.0, 1.0, 3.0, 4.0], order=1)
    np.testing.assert_allclose(spec.axis, [0.0, 1.0, 3.0, 4.0])
    assert spec.order == 1


def test_axis_A_mV_uses_linspace_semantics() -> None:
    spec = axis("A_mV", -1.0, 1.0, 5)
    np.testing.assert_allclose(spec.axis, np.linspace(-1.0, 1.0, 5))
    assert spec.code_label == "A_mV"
    assert spec.latex_label == r"$A$ (mV)"


def test_axis_nu_GHz_accepts_values_override() -> None:
    spec = axis("nu_GHz", values=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(spec.axis, [0.0, 0.5, 1.0])
    assert spec.code_label == "nu_GHz"


def test_axis_T_K_uses_linspace_semantics() -> None:
    spec = axis("T_K", 0.0, 3.0, 4)
    np.testing.assert_allclose(spec.axis, np.linspace(0.0, 3.0, 4))
    assert spec.code_label == "T_K"


def test_axis_V_Delta_uses_linspace_semantics() -> None:
    spec = axis("V", -1.0, 1.0, 5, order=3)
    np.testing.assert_allclose(spec.axis, np.linspace(-1.0, 1.0, 5))
    assert spec.code_label == "V"


def test_axis_I_GNDelta_accepts_values_override() -> None:
    spec = axis("I", values=[0.0, 0.5, 1.0], order=4)
    np.testing.assert_allclose(spec.axis, [0.0, 0.5, 1.0])
    assert spec.code_label == "I"


def test_axis_A_hnu_accepts_values_override() -> None:
    spec = axis("A_hnu", values=[0.0, 0.25, 0.5])
    np.testing.assert_allclose(spec.axis, [0.0, 0.25, 0.5])


def test_axis_hnu_Delta_accepts_values_override() -> None:
    spec = axis("hnu_Delta", values=[0.0, 0.2, 0.4])
    np.testing.assert_allclose(spec.axis, [0.0, 0.2, 0.4])


def test_axis_T_Tc_accepts_values_override() -> None:
    spec = axis("T_Tc", values=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(spec.axis, [0.0, 0.5, 1.0])


def test_axis_spec_validates_non_monotonic_input() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        AxisSpec(
            values=[0.0, 1.0, 1.0],
            code_label="A_mV",
            html_label="<i>A</i> (mV)",
            latex_label=r"$A$ (mV)",
            order=1,
        )


def test_param_lookup_falls_back_to_name() -> None:
    spec = param("tau", value=0.2)
    assert spec.code_label == "tau"
    assert spec.print_label == "tau"
    assert spec.html_label == "tau"
    assert spec.latex_label == "tau"


def test_constant_param_specs_still_behave_numerically() -> None:
    assert float(G0_muS) == pytest.approx(77.48091729863648)
    assert h_Vs * 1e12 == pytest.approx(0.004135667696923859)
    assert 1.0 / kB_meV_K == pytest.approx(11.604518121550084)


def test_data_spec_inherits_label_meta() -> None:
    spec = dataset("I_nA", [1.0, 2.0, 3.0])
    assert isinstance(spec, LabelSpec)
    assert isinstance(spec, Dataset)
    np.testing.assert_allclose(spec.values, [1.0, 2.0, 3.0])


def test_data_lookup_falls_back_to_name() -> None:
    spec = dataset("custom", [1.0, 2.0])
    assert spec.code_label == "custom"
    assert spec.print_label == "custom"
    assert spec.html_label == "custom"
    assert spec.latex_label == "custom"
