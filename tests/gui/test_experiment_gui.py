from __future__ import annotations

import importlib
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("panel")

from superconductivity.evaluation.ivdata import IVTrace, IVTraces
from superconductivity.evaluation.offset import get_offset
from superconductivity.evaluation.psd import PSDSpec, get_psd
from superconductivity.evaluation.sampling import (
    fill_sampling_spec_from_offset,
    get_sampling,
)
from superconductivity.gui import GUIPanel, gui_app

gui_mod = importlib.import_module(
    "superconductivity.gui.app"
)


def _trace_by_name(figure, name: str):
    for trace in figure.data:
        if trace.name == name:
            return trace
    raise AssertionError(f"trace {name!r} not found")


def _experimental_rows(panel: GUIPanel):
    return panel._experimental_table.value.reset_index(drop=True).set_index("key")


def _sampling_info_rows(panel: GUIPanel):
    return panel._sampling_info_table.value.reset_index(drop=True).set_index("key")


def _offset_info_rows(panel: GUIPanel):
    return panel._offset_info_table.value.reset_index(drop=True).set_index("key")


def _sampling_smoothing_rows(panel: GUIPanel):
    return (
        panel._sampling_smoothing_table.value.reset_index(drop=True).set_index("key")
    )


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    v_shift_mV: float,
    i_shift_nA: float,
) -> IVTrace:
    t_s = np.linspace(0.0, 10.0, 401, dtype=np.float64)
    v_true_mV = np.linspace(-2.0, 2.0, t_s.size, dtype=np.float64)
    i_true_nA = v_true_mV + 0.2 * v_true_mV**3
    return {
        "specific_key": specific_key,
        "index": index,
        "yvalue": yvalue,
        "V_mV": v_true_mV + v_shift_mV,
        "I_nA": i_true_nA + i_shift_nA,
        "t_s": t_s,
    }


def _make_traces() -> IVTraces:
    return IVTraces(
        traces=[
            _make_iv_trace("a", 0, 1.0, v_shift_mV=0.4, i_shift_nA=0.3),
            _make_iv_trace("b", 1, 5.0, v_shift_mV=0.2, i_shift_nA=0.1),
        ],
    )


def _fake_solution(panel: GUIPanel):
    sampling = panel.state["sampling"]
    params = []
    for parameter in panel._parameters:
        fitted = replace(parameter)
        fitted.value = float(parameter.guess)
        fitted.error = 0.0
        params.append(fitted)
    return {
        "V_mV": np.asarray(sampling["Vbin_mV"], dtype=np.float64),
        "I_exp_nA": np.asarray(sampling["I_nA"], dtype=np.float64),
        "I_ini_nA": np.asarray(panel._initial_curve, dtype=np.float64),
        "I_fit_nA": np.asarray(panel._initial_curve, dtype=np.float64) * 0.98,
        "params": tuple(params),
        "weights": None,
        "maxfev": panel.maxfev,
    }


def test_gui_app_builds_without_server() -> None:
    app = gui_app(_make_traces(), model="bcs_sis_int")

    assert hasattr(app, "_gui_panel")
    panel = app._gui_panel
    assert panel.state["active_index"] == 0
    assert panel.state["sampling"]["I_nA"].shape == (51,)
    assert panel._left_stage_selector.value == ["binned", "initial", "fit"]
    assert len(panel._iv_figure.data) == 5
    assert len(panel._vi_figure.data) == 5
    assert _trace_by_name(panel._iv_figure, "Raw").visible is False
    assert _trace_by_name(panel._iv_figure, "Binned").visible is True
    experimental_rows = _experimental_rows(panel)
    assert panel._experimental_psd_figure.layout.yaxis.type == "log"
    assert panel._experimental_psd_figure.layout.yaxis2.type == "log"
    assert len(panel._experimental_psd_figure.data) == 4
    assert panel._experimental_psd_figure.layout.showlegend is False
    assert panel._experimental_time_figure.layout.showlegend is False
    assert panel._experimental_psd_figure.data[0].name == "Raw"
    assert panel._experimental_psd_figure.data[0].mode == "lines"
    assert panel._experimental_psd_figure.data[1].name == "Downsampled"
    assert panel._experimental_psd_figure.data[1].mode == "lines"
    assert panel._experimental_table.value.shape == (4, 3)
    assert panel._experimental_table.titles["parameter"] == "Parameter"
    assert panel._experimental_table.titles["value"] == "Value"
    assert experimental_rows.at["nu_Hz", "parameter"] == "<i>&nu;</i> (Hz)"
    assert experimental_rows.at["detrend", "parameter"] == "Detrend"
    assert experimental_rows.at["sigma_V_mV", "parameter"] == (
        "<i>&sigma;<sub>V</sub></i> (mV)"
    )
    assert experimental_rows.at["sigma_I_nA", "parameter"] == (
        "<i>&sigma;<sub>I</sub></i> (nA)"
    )
    assert len(panel._experimental_time_figure.layout.annotations) == 0
    assert panel._experimental_apply_button.name == "PSD Analysis"
    assert panel._experimental_plot_tabs.active == 0
    assert panel._experimental_plot_tabs._names == ["S(f)", "V(t) / I(t)"]
    assert panel._offset_apply_button.name == "Offset Analysis"
    assert panel._sampling_apply_button.name == "Sampling"
    assert "Downsampled" in panel._experimental_legend.object
    assert "flex-direction:row" in panel._experimental_legend.object
    offset_info_rows = _offset_info_rows(panel)
    assert panel._offset_grid_table.value.shape == (4, 4)
    assert panel._offset_info_table.value.shape == (4, 3)
    assert np.issubdtype(panel._offset_grid_table.value["count"].dtype, np.integer)
    assert panel._offset_info_table.titles["parameter"] == "Parameter"
    assert panel._offset_info_table.titles["value"] == "Value"
    assert offset_info_rows.at["nu_Hz", "parameter"] == "<i>&nu;</i> (Hz)"
    assert offset_info_rows.at["upsample", "parameter"] == (
        "<i>N</i><sub>up</sub>"
    )
    assert offset_info_rows.at["Voff_mV", "parameter"] == (
        "<i>V</i><sub>off</sub> (mV)"
    )
    assert isinstance(offset_info_rows.at["upsample", "value"], int)
    assert panel._offset_grid_table.value.iloc[0]["parameter"] == (
        "<i>V</i><sub>bins</sub> (mV)"
    )
    assert panel._offset_grid_table.value.iloc[2]["parameter"] == (
        "<i>V</i><sub>off</sub> (mV)"
    )
    assert panel._offset_g_figure.layout.title.text is None
    assert panel._offset_r_figure.layout.title.text is None
    sampling_info_rows = _sampling_info_rows(panel)
    sampling_smoothing_rows = _sampling_smoothing_rows(panel)
    assert panel._sampling_grid_table.value.shape == (2, 4)
    assert panel._sampling_info_table.value.shape == (3, 3)
    assert panel._sampling_smoothing_table.value.shape == (2, 3)
    assert np.issubdtype(
        panel._sampling_grid_table.value["count"].dtype,
        np.integer,
    )
    assert panel._sampling_grid_table.value.iloc[0]["parameter"] == (
        "<i>V</i><sub>bins</sub> (mV)"
    )
    assert panel._sampling_info_table.titles["parameter"] == "Parameter"
    assert panel._sampling_info_table.titles["value"] == "Value"
    assert sampling_info_rows.at["upsample", "parameter"] == (
        "<i>N</i><sub>up</sub>"
    )
    assert sampling_info_rows.at["Voff_mV", "parameter"] == (
        "<i>V</i><sub>off</sub> (mV)"
    )
    assert sampling_info_rows.at["Ioff_nA", "parameter"] == (
        "<i>I</i><sub>off</sub> (nA)"
    )
    assert sampling_info_rows.at["upsample", "value"] == pytest.approx(
        panel._sampling_spec.upsample
    )
    assert isinstance(sampling_info_rows.at["upsample", "value"], int)
    assert panel._sampling_smooth_toggle.value is False
    assert sampling_smoothing_rows.at["median_bins", "parameter"] == (
        "<i>N</i><sub>med</sub>"
    )
    assert sampling_smoothing_rows.at["sigma_bins", "parameter"] == (
        "<i>&sigma;</i><sub>bins</sub>"
    )
    assert sampling_smoothing_rows.at["median_bins", "value"] == pytest.approx(
        panel._smoothing_spec.median_bins
    )
    assert isinstance(
        sampling_smoothing_rows.at["median_bins", "value"],
        int,
    )
    assert panel._sampling_iv_figure.layout.title.text is None
    assert panel._sampling_vi_figure.layout.title.text is None


def test_trace_change_repopulates_pipeline_and_left_plots() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_sis_int")
    before = np.asarray(panel.state["sampling"]["I_nA"], dtype=np.float64)

    panel._trace_selector.value = 1

    after = np.asarray(panel.state["sampling"]["I_nA"], dtype=np.float64)
    assert panel.active_index == 1
    assert not np.allclose(before, after, equal_nan=True)
    assert np.allclose(
        _trace_by_name(panel._iv_figure, "Binned").y,
        after,
        equal_nan=True,
    )


def test_left_stage_selector_can_show_raw_and_downsampled() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_sis_int")

    panel._left_stage_selector.value = [
        "raw",
        "downsampled",
        "binned",
        "initial",
        "fit",
    ]

    assert _trace_by_name(panel._iv_figure, "Raw").visible is True
    assert _trace_by_name(panel._iv_figure, "Downsampled").visible is True
    assert np.allclose(
        np.asarray(_trace_by_name(panel._iv_figure, "Raw").x, dtype=np.float64),
        np.asarray(panel.state["trace"]["V_mV"], dtype=np.float64),
        equal_nan=True,
    )


def test_apply_buttons_update_expected_pipeline_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"psd": 0, "offset": 0, "sampling": 0}
    real_get_psd = gui_mod.get_psd
    real_get_offset = gui_mod.get_offset
    real_get_sampling = gui_mod.get_sampling

    def wrapped_psd(*args, **kwargs):
        calls["psd"] += 1
        return real_get_psd(*args, **kwargs)

    def wrapped_offset(*args, **kwargs):
        calls["offset"] += 1
        return real_get_offset(*args, **kwargs)

    def wrapped_sampling(*args, **kwargs):
        calls["sampling"] += 1
        return real_get_sampling(*args, **kwargs)

    monkeypatch.setattr(gui_mod, "get_psd", wrapped_psd)
    monkeypatch.setattr(gui_mod, "get_offset", wrapped_offset)
    monkeypatch.setattr(gui_mod, "get_sampling", wrapped_sampling)

    panel = GUIPanel(_make_traces(), model="bcs_sis_int")
    panel._set_fit_solution(_fake_solution(panel))
    panel._fit_curve = np.asarray(panel._fit_solution["I_fit_nA"], dtype=np.float64)
    calls.update(psd=0, offset=0, sampling=0)

    offset_info_table = panel._offset_info_table.value.copy()
    offset_info_table.loc[
        offset_info_table["key"] == "upsample",
        "value",
    ] = panel._offset_spec.upsample + 1
    panel._offset_info_table.value = offset_info_table
    panel._on_offset_apply(SimpleNamespace())

    assert calls == {"psd": 0, "offset": 1, "sampling": 1}
    assert panel.state["fit"] is None

    calls.update(psd=0, offset=0, sampling=0)
    sampling_info_table = panel._sampling_info_table.value.copy()
    sampling_info_table.loc[
        sampling_info_table["key"] == "upsample",
        "value",
    ] = panel._sampling_spec.upsample + 1
    panel._sampling_info_table.value = sampling_info_table
    panel._on_sampling_apply(SimpleNamespace())

    assert calls == {"psd": 0, "offset": 0, "sampling": 1}

    calls.update(psd=0, offset=0, sampling=0)
    experimental_table = panel._experimental_table.value.copy()
    experimental_table.loc[
        experimental_table["key"] == "nu_Hz",
        "value",
    ] = panel._shared_nu_Hz + 1.0
    panel._experimental_table.value = experimental_table
    panel._on_experimental_apply(SimpleNamespace())

    assert calls == {"psd": 1, "offset": 0, "sampling": 1}


def test_gui_panel_matches_backend_outputs() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_sis_int")
    trace = panel.state["trace"]

    downsampled_trace, psd_expected = get_psd(
        trace,
        spec=PSDSpec(nu_Hz=panel._shared_nu_Hz, detrend=True),
    )
    offset_expected = get_offset(trace, spec=panel._offset_spec)
    sampling_expected = get_sampling(
        downsampled_trace,
        spec=fill_sampling_spec_from_offset(
            panel._sampling_spec,
            offset_expected,
        ),
    )

    assert panel.state["psd"]["raw_sigma_I_nA"] == pytest.approx(
        psd_expected["raw_sigma_I_nA"]
    )
    assert panel.state["psd"]["raw_sigma_V_mV"] == pytest.approx(
        psd_expected["raw_sigma_V_mV"]
    )
    assert np.allclose(
        panel.state["offset"]["dGerr_G0"],
        offset_expected["dGerr_G0"],
        equal_nan=True,
    )
    assert np.allclose(
        panel.state["sampling"]["I_nA"],
        sampling_expected["I_nA"],
        equal_nan=True,
    )
    experimental_rows = _experimental_rows(panel)
    assert experimental_rows.at["nu_Hz", "value"] == pytest.approx(
        panel._shared_nu_Hz
    )
    assert bool(experimental_rows.at["detrend", "value"]) is True
    assert experimental_rows.at["sigma_I_nA", "value"] == pytest.approx(
        psd_expected["raw_sigma_I_nA"]
    )
    assert experimental_rows.at["sigma_V_mV", "value"] == pytest.approx(
        psd_expected["raw_sigma_V_mV"]
    )
    offset_rows = _offset_info_rows(panel)
    assert offset_rows.at["nu_Hz", "value"] == pytest.approx(
        panel._offset_spec.nu_Hz
    )
    assert offset_rows.at["upsample", "value"] == panel._offset_spec.upsample
    assert offset_rows.at["Voff_mV", "value"] == pytest.approx(
        offset_expected["Voff_mV"]
    )
    assert offset_rows.at["Ioff_nA", "value"] == pytest.approx(
        offset_expected["Ioff_nA"]
    )
    sampling_rows = _sampling_info_rows(panel)
    assert sampling_rows.at["Voff_mV", "value"] == pytest.approx(
        offset_expected["Voff_mV"]
    )
    assert sampling_rows.at["Ioff_nA", "value"] == pytest.approx(
        offset_expected["Ioff_nA"]
    )


def test_fit_uses_sampled_curve_and_updates_iv_overlay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces(), model="bcs_sis_int")
    captured = {}
    fake_solution = _fake_solution(panel)

    class DummyFuture:
        def __init__(self, result):
            self._result = result

        def result(self):
            return self._result

        def add_done_callback(self, callback):
            callback(self)

    class DummyExecutor:
        def submit(self, function, sampling, **kwargs):
            _ = function
            captured["sampling"] = sampling
            captured["kwargs"] = kwargs
            return DummyFuture(fake_solution)

    monkeypatch.setattr(panel._pn.state, "execute", lambda callback: callback())
    panel._executor = DummyExecutor()

    panel._on_fit_clicked(SimpleNamespace())

    assert captured["sampling"] == panel.state["sampling"]
    assert panel.state["fit"] is not None
    assert np.allclose(
        _trace_by_name(panel._iv_figure, "Fit").y,
        fake_solution["I_fit_nA"],
    )
    assert len(panel._vi_figure.data) == 5
    assert _trace_by_name(panel._vi_figure, "Fit").visible is True


def test_sampling_apply_can_enable_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces(), model="bcs_sis_int")
    raw_sampling = np.asarray(panel.state["sampling"]["I_nA"], dtype=np.float64)

    def fake_smoothing(*args, **kwargs):
        _ = args, kwargs
        smoothed = panel._require_raw_sampling().copy()
        smoothed["I_nA"] = np.asarray(raw_sampling, dtype=np.float64) * 0.9
        return smoothed

    monkeypatch.setattr(
        gui_mod,
        "get_smoothed_sampling",
        fake_smoothing,
    )

    sampling_smoothing_table = panel._sampling_smoothing_table.value.copy()
    panel._sampling_smooth_toggle.value = True
    sampling_smoothing_table.loc[
        sampling_smoothing_table["key"] == "median_bins",
        "value",
    ] = 3
    sampling_smoothing_table.loc[
        sampling_smoothing_table["key"] == "sigma_bins",
        "value",
    ] = 1.5
    panel._sampling_smoothing_table.value = sampling_smoothing_table
    panel._on_sampling_apply(SimpleNamespace())

    assert panel._smoothing_enabled is True
    assert np.allclose(
        panel.state["sampling"]["I_nA"],
        raw_sampling * 0.9,
        equal_nan=True,
    )
    assert np.allclose(
        _trace_by_name(panel._iv_figure, "Binned").y,
        raw_sampling * 0.9,
        equal_nan=True,
    )


def test_sampling_offsets_follow_active_trace() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_sis_int")
    rows_0 = _sampling_info_rows(panel)
    offset_0 = panel.state["offset"]
    assert rows_0.at["Voff_mV", "value"] == pytest.approx(offset_0["Voff_mV"])
    assert rows_0.at["Ioff_nA", "value"] == pytest.approx(offset_0["Ioff_nA"])

    panel._trace_selector.value = 1

    sampling_rows = _sampling_info_rows(panel)
    offset_1 = panel.state["offset"]
    assert sampling_rows.at["Voff_mV", "value"] == pytest.approx(offset_1["Voff_mV"])
    assert sampling_rows.at["Ioff_nA", "value"] == pytest.approx(offset_1["Ioff_nA"])


def test_run_gui_returns_last_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "active_index": 0,
        "trace": _make_traces()[0],
        "psd": {"raw_sigma_I_nA": 0.0, "raw_sigma_V_mV": 0.0},
        "offset": {"Voff_mV": 0.0, "Ioff_nA": 0.0},
        "sampling": {"I_nA": np.zeros(3), "Vbin_mV": np.zeros(3)},
        "fit": None,
    }

    class DummyServer:
        def is_alive(self) -> bool:
            return False

    def fake_serve(*args, **kwargs):
        _ = args, kwargs
        gui_mod._ACTIVE_GUI_PANEL = SimpleNamespace(
            state=expected,
        )
        return DummyServer()

    monkeypatch.setattr(gui_mod, "serve_gui", fake_serve)

    result = gui_mod.run_gui(
        _make_traces(),
        model="bcs_sis_int",
    )

    assert result is expected
