import numpy as np
import pytest
from dataclasses import replace
from pathlib import Path

pytest.importorskip("panel")

from superconductivity.optimizers.pat.process import (
    clear_solution,
    load_solution,
    save_solution,
)
from superconductivity.optimizers.pat.panel import PatFitPanel
from superconductivity.optimizers.pat import DEFAULT_PARAMETERS


def _make_solution(V: np.ndarray, I: np.ndarray) -> dict:
    guess = np.array([1.0, 0.2, 0.195, 1e-3, 1.0, 7.8], dtype=np.float64)
    popt = guess + np.array([0.1, 0.05, 0.005, 1e-3, 0.2, 0.5], dtype=np.float64)
    params = []
    for idx, base in enumerate(DEFAULT_PARAMETERS):
        params.append(
            replace(
                base,
                guess=float(guess[idx]),
                fit_value=float(popt[idx]),
                fit_error=0.1,
            )
        )
    return {
        "V_mV": V,
        "I_exp_nA": I,
        "I_ini_nA": I,
        "I_fit_nA": I + 0.2,
        "params": tuple(params),
        "weights": None,
        "maxfev": None,
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

    monkeypatch.setattr("superconductivity.optimizers.pat.panel.fit_pat", fake_fit)
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


def test_solution_callback_persists_and_clears(monkeypatch, tmp_path: Path) -> None:
    V = np.linspace(-2.0, 2.0, 61)
    I = V**2 * 0.01

    def callback(solution):
        if solution is None:
            clear_solution(tmp_path)
        else:
            save_solution(tmp_path, solution)

    widget = PatFitPanel(
        V,
        np.vstack((I, I + 0.1)),
        on_solution_changed=callback,
    )

    monkeypatch.setattr(
        "superconductivity.optimizers.pat.panel.fit_pat",
        lambda *args, **kwargs: _make_solution(V, I),
    )

    widget._on_fit_click(None)
    saved = load_solution(tmp_path)
    assert saved is not None

    widget._trace_selector.value = 1
    assert load_solution(tmp_path) is None
