import numpy as np
import pytest

pytest.importorskip("panel")

from superconductivity.optimizers.gui.pat_panel import PatFitPanel


def _make_solution(V: np.ndarray, I: np.ndarray) -> dict:
    guess = np.array([1.0, 0.2, 0.195, 1e-3, 1.0, 7.8], dtype=np.float64)
    popt = guess + np.array([0.1, 0.05, 0.005, 1e-3, 0.2, 0.5], dtype=np.float64)
    return {
        "optimizer": "curve_fit",
        "model": "pat",
        "V_mV": V,
        "I_exp_nA": I,
        "I_ini_nA": I,
        "I_fit_nA": I + 0.2,
        "guess": guess,
        "lower": guess * 0.5,
        "upper": guess * 1.5,
        "fixed": np.zeros_like(guess, dtype=bool),
        "popt": popt,
        "pcov": np.eye(6, dtype=np.float64) * 0.01,
        "perr": np.full_like(popt, 0.1),
        "E_mV": None,
        "weights": None,
        "maxfev": None,
        "G_N": popt[0],
        "T_K": popt[1],
        "Delta_mV": popt[2],
        "gamma_mV": popt[3],
        "A_mV": popt[4],
        "nu_GHz": popt[5],
    }


def test_slider_change_updates_initial_curve() -> None:
    V = np.linspace(-1.0, 1.0, 51)
    I = np.sin(V) * 0.1
    widget = PatFitPanel(V, I)
    initial_curve = widget._iv_figure.data[1].y.copy()

    widget._sliders["GN_G0"].value = widget._sliders["GN_G0"].value + 0.5
    assert not np.allclose(initial_curve, widget._iv_figure.data[1].y)


def test_fit_button_triggers_optimizer(monkeypatch) -> None:
    V = np.linspace(-2.0, 2.0, 61)
    I = V**2 * 0.01
    widget = PatFitPanel(V, I)

    captured: dict[str, object] = {}

    def fake_fit(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return _make_solution(V, I)

    monkeypatch.setattr("superconductivity.optimizers.gui.pat_panel.fit_I_nA", fake_fit)
    widget._on_fit_click(None)

    assert captured, "fit_I_nA was not called"
    assert np.allclose(widget._iv_figure.data[2].y, I + 0.2)
    assert widget._solution is not None


def test_trace_selector_switches_rows() -> None:
    V = np.linspace(-1.0, 1.0, 41)
    first = np.sin(V) * 0.1
    second = np.cos(V) * 0.1
    data = np.vstack((first, second))
    widget = PatFitPanel(V, data)

    start = widget._iv_figure.data[0].y.copy()
    widget._trace_selector.value = 1

    assert not np.allclose(start, widget._iv_figure.data[0].y)
    assert np.allclose(widget.I_nA, second)
    assert "trace 2 of 2" in widget._trace_header.object
