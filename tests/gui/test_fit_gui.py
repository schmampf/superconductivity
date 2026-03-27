from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("panel")

from superconductivity.optimizers.gui import FitPanel, fit_gui_app
from superconductivity.optimizers.models import get_model_spec


def _bcs_trace(V_mV: np.ndarray) -> np.ndarray:
    spec = get_model_spec("bcs_sis_int")
    return spec.function(V_mV, *[parameter.guess for parameter in spec.parameters])


def test_fit_gui_app_builds_without_server() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41)
    I_nA = _bcs_trace(V_mV)

    app = fit_gui_app(V_mV, I_nA, model="bcs_sis_int")

    assert hasattr(app, "_fit_gui_panel")


def test_parameter_edit_updates_initial_curve() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41)
    I_nA = _bcs_trace(V_mV)
    panel = FitPanel(V_mV, I_nA, model="bcs_sis_int")
    before = np.array(panel._iv_figure.data[1].y, dtype=np.float64)

    panel._on_parameter_edit(
        SimpleNamespace(
            column="guess",
            row=0,
            value=panel._parameters[0].guess + 0.2,
        )
    )

    after = np.array(panel._iv_figure.data[1].y, dtype=np.float64)
    assert not np.allclose(before, after)


def test_slice_and_model_switch_update_gui_state() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41)
    first = _bcs_trace(V_mV)
    second = first * 1.02
    panel = FitPanel(V_mV, np.vstack([first, second]), model="bcs_sis_int")

    panel._slice_selector.value = 1
    assert panel._slice_index == 1
    assert np.allclose(panel._iv_figure.data[0].y, second)

    panel._model_selector.value = "bcs_sis_conv_noise"
    assert panel.model_key == "bcs_sis_conv_noise"
    assert len(panel._parameters) == 5
    assert panel._parameters[-1].name == "sigma_V_mV"

    panel._model_selector.value = "pat_sis_int_jax"
    assert panel.model_key == "pat_sis_int_jax"
    assert len(panel._parameters) == 6
    assert panel._model_info_table.value.loc[0, "value"] == "pat_sis_int_jax"
