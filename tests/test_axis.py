from __future__ import annotations

import numpy as np
import pytest

from superconductivity.utilities.axis import AxisSpec, construct_axis
from superconductivity.utilities.axis_constructors import (
    axis_A_hnu,
    axis_A_mV,
    axis_I_GNDelta,
    axis_I_nA,
    axis_T_K,
    axis_T_Tc,
    axis_V_Delta,
    axis_V_mV,
    axis_hnu_Delta,
    axis_nu_GHz,
)
from superconductivity.utilities.label import LabelMeta, label_V_mV, label_dIdV
from superconductivity.utilities.params import ParamSpec


def test_label_meta_construction() -> None:
    meta = label_V_mV()
    assert isinstance(meta, LabelMeta)
    assert meta.label == "V_mV"
    assert meta.html_label == "<i>V</i> (mV)"
    assert meta.latex_label == r"$V$ (mV)"


def test_label_meta_reuse() -> None:
    meta = label_dIdV()
    axis = construct_axis(0.0, 1.0, 3, values=[0.0, 0.5, 1.0], meta=meta)
    assert axis.label == meta.label
    assert axis.html_label == meta.html_label
    assert axis.latex_label == meta.latex_label


def test_axis_spec_inherits_label_meta() -> None:
    axis = axis_V_mV(-1.0, 1.0, 5)
    assert isinstance(axis, LabelMeta)
    assert axis.label == "V_mV"
    assert axis.kind == "x"


def test_param_spec_inherits_label_meta() -> None:
    param = ParamSpec(label="tau", html_label="<i>&tau;</i>", latex_label=r"$\\tau$")
    assert isinstance(param, LabelMeta)
    assert param.label == "tau"


def test_construct_axis_uses_values_over_linspace() -> None:
    axis = construct_axis(
        0.0,
        10.0,
        5,
        values=np.asarray([1.0, 2.0, 4.0], dtype=np.float64),
        meta=label_V_mV(),
    )
    np.testing.assert_allclose(axis.axis, [1.0, 2.0, 4.0])


def test_axis_V_mV_uses_linspace_semantics() -> None:
    axis = axis_V_mV(-1.0, 1.0, 5)
    np.testing.assert_allclose(axis.axis, np.linspace(-1.0, 1.0, 5))
    assert axis.label == "V_mV"
    assert axis.latex_label == r"$V$ (mV)"


def test_axis_I_nA_accepts_values_override() -> None:
    axis = axis_I_nA(values=[0.0, 1.0, 3.0, 4.0])
    np.testing.assert_allclose(axis.axis, [0.0, 1.0, 3.0, 4.0])
    assert axis.kind == "y"


def test_axis_A_mV_uses_linspace_semantics() -> None:
    axis = axis_A_mV(-1.0, 1.0, 5)
    np.testing.assert_allclose(axis.axis, np.linspace(-1.0, 1.0, 5))
    assert axis.label == "A_mV"
    assert axis.latex_label == r"$A$ (mV)"


def test_axis_nu_GHz_accepts_values_override() -> None:
    axis = axis_nu_GHz(values=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(axis.axis, [0.0, 0.5, 1.0])
    assert axis.label == "nu_GHz"


def test_axis_T_K_uses_linspace_semantics() -> None:
    axis = axis_T_K(0.0, 3.0, 4)
    np.testing.assert_allclose(axis.axis, np.linspace(0.0, 3.0, 4))
    assert axis.label == "T_K"


def test_axis_V_Delta_uses_linspace_semantics() -> None:
    axis = axis_V_Delta(-1.0, 1.0, 5)
    np.testing.assert_allclose(axis.axis, np.linspace(-1.0, 1.0, 5))
    assert axis.label == "V"


def test_axis_I_GNDelta_accepts_values_override() -> None:
    axis = axis_I_GNDelta(values=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(axis.axis, [0.0, 0.5, 1.0])
    assert axis.label == "I"


def test_axis_A_hnu_accepts_values_override() -> None:
    axis = axis_A_hnu(values=[0.0, 0.25, 0.5])
    np.testing.assert_allclose(axis.axis, [0.0, 0.25, 0.5])


def test_axis_hnu_Delta_accepts_values_override() -> None:
    axis = axis_hnu_Delta(values=[0.0, 0.2, 0.4])
    np.testing.assert_allclose(axis.axis, [0.0, 0.2, 0.4])


def test_axis_T_Tc_accepts_values_override() -> None:
    axis = axis_T_Tc(values=[0.0, 0.5, 1.0])
    np.testing.assert_allclose(axis.axis, [0.0, 0.5, 1.0])


def test_axis_spec_validates_non_monotonic_input() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        AxisSpec(
            axis=[0.0, 1.0, 1.0],
            label="A_mV",
            html_label="<i>A</i> (mV)",
            latex_label=r"$A$ (mV)",
            kind="y",
        )
