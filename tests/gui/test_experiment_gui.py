from __future__ import annotations

import importlib
import time
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("panel")

from superconductivity.evaluation.traces import (
    FileSpec,
    Keys,
    KeysSpec,
    Trace,
    TraceMeta,
    TraceSpec,
    Traces,
)
from superconductivity.evaluation.analysis.offset import OffsetSpec, offset_analysis
from superconductivity.evaluation.analysis.psd import PSDTraces, PSDSpec, psd_analysis
from superconductivity.evaluation.sampling import (
    SamplingSpec,
    Samples,
    sample,
)
from superconductivity.evaluation.sampling import SmoothingSpec
from superconductivity.gui import GUIPanel, gui, gui_app

gui_mod = importlib.import_module("superconductivity.gui.app")


def _trace_by_name(figure, name: str):
    for trace in figure.data:
        if trace.name == name:
            return trace
    raise AssertionError(f"trace {name!r} not found")


def _experimental_setting_rows(panel: GUIPanel):
    return panel._experimental_table.value.reset_index(drop=True).set_index("key")


def _sampling_info_rows(panel: GUIPanel):
    return panel._sampling_info_table.value.reset_index(drop=True).set_index("key")


def _offset_info_rows(panel: GUIPanel):
    return panel._offset_info_table.value.reset_index(drop=True).set_index("key")


def _offset_batch_rows(panel: GUIPanel):
    return panel._offset_batch_table.value.reset_index(drop=True)


def _fit_config_rows(panel: GUIPanel):
    return panel._fit_config_table.value.reset_index(drop=True).set_index("key")


def _fit_parameter_rows(panel: GUIPanel):
    return panel._parameter_table.value.reset_index(drop=True).set_index("name")


def _optimizer_rows(panel: GUIPanel):
    return panel._optimizer_info_table.value.reset_index(drop=True).set_index("key")


def _model_rows(panel: GUIPanel):
    return panel._model_info_table.value.reset_index(drop=True).set_index("key")


def _sampling_smoothing_rows(panel: GUIPanel):
    return panel._sampling_smoothing_table.value.reset_index(drop=True).set_index("key")


def _make_iv_trace(
    specific_key: str,
    index: int,
    yvalue: float,
    v_shift_mV: float,
    i_shift_nA: float,
) -> Trace:
    t_s = np.linspace(0.0, 10.0, 401, dtype=np.float64)
    v_true_mV = np.linspace(-2.0, 2.0, t_s.size, dtype=np.float64)
    i_true_nA = v_true_mV + 0.2 * v_true_mV**3
    return {
        "meta": TraceMeta(
            specific_key=specific_key,
            index=index,
            yvalue=yvalue,
        ),
        "V_mV": v_true_mV + v_shift_mV,
        "I_nA": i_true_nA + i_shift_nA,
        "t_s": t_s,
    }


def _make_traces() -> Traces:
    return Traces(
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
        "V_mV": np.asarray(sampling["Vbins_mV"], dtype=np.float64),
        "I_exp_nA": np.asarray(sampling["I_nA"], dtype=np.float64),
        "I_ini_nA": np.asarray(panel._initial_curve, dtype=np.float64),
        "I_fit_nA": np.asarray(panel._initial_curve, dtype=np.float64) * 0.98,
        "params": tuple(params),
        "weights": None,
        "maxfev": panel.maxfev,
    }


def test_gui_app_builds_without_server() -> None:
    app = gui_app(_make_traces())

    assert hasattr(app, "_gui_panel")
    panel = app._gui_panel
    assert panel.state["active_index"] == 0
    assert panel.state["sampling"]["I_nA"].shape == (51,)
    assert panel._left_stage_selector.value == ["binned", "initial", "fit"]
    assert panel._left_v_quantity_selector.value == ["iv_v", "didv_v"]
    assert panel._left_i_quantity_selector.value == ["vi_i", "dvdi_i"]
    assert len(panel._iv_figure.data) == 10
    assert len(panel._vi_figure.data) == 10
    assert _trace_by_name(panel._iv_figure, "Raw").visible is False
    assert _trace_by_name(panel._iv_figure, "Binned").visible is True
    assert panel._iv_figure.layout.xaxis.matches == "x2"
    assert panel._vi_figure.layout.xaxis.matches == "x2"
    assert panel._iv_figure.layout.yaxis.title.text == "<i>I</i> (nA)"
    assert panel._iv_figure.layout.yaxis2.title.text == (
        "<i>dI/dV</i> (<i>G</i><sub>0</sub>)"
    )
    assert panel._vi_figure.layout.yaxis.title.text == "<i>V</i> (mV)"
    assert panel._vi_figure.layout.yaxis2.title.text == (
        "<i>dV/dI</i> (<i>R</i><sub>0</sub>)"
    )
    experimental_setting_rows = _experimental_setting_rows(panel)
    assert panel._experimental_psd_figure.layout.yaxis.type == "log"
    assert panel._experimental_psd_figure.layout.yaxis2.type == "log"
    assert len(panel._experimental_psd_figure.data) == 4
    assert panel._experimental_psd_figure.layout.showlegend is False
    assert panel._experimental_time_figure.layout.showlegend is False
    assert panel._experimental_psd_figure.data[0].name == "Raw"
    assert panel._experimental_psd_figure.data[0].mode == "lines"
    assert panel._experimental_psd_figure.data[1].name == "Downsampled"
    assert panel._experimental_psd_figure.data[1].mode == "lines"
    assert panel._experimental_table.value.shape == (2, 3)
    assert panel._experimental_table.titles["parameter"] == "Parameter"
    assert panel._experimental_table.titles["value"] == "Value"
    assert experimental_setting_rows.at["nu_Hz", "parameter"] == "<i>&nu;</i> (Hz)"
    assert experimental_setting_rows.at["detrend", "parameter"] == "Detrend"
    assert len(panel._experimental_time_figure.layout.annotations) == 0
    assert panel._experimental_apply_button.name == "PSD Analysis"
    assert panel._experimental_plot_tabs.active == 0
    assert panel._experimental_plot_tabs._names == ["S(f)", "V(t) / I(t)"]
    assert panel._offset_apply_button.name == "Offset Analysis"
    assert panel._offset_batch_apply_button.name == "Offset Analysis (All)"
    assert panel._offset_batch_stop_button.name == "Stop"
    assert panel._offset_batch_stop_button.disabled is True
    assert panel._offset_batch_progress.value == 0
    assert panel._offset_batch_state.object == "Idle"
    assert panel._sampling_apply_button.name == "Sampling"
    assert "Downsampled" in panel._experimental_legend.object
    assert "Cutoff" not in panel._experimental_legend.object
    assert "flex-direction:row" in panel._experimental_legend.object
    offset_info_rows = _offset_info_rows(panel)
    offset_batch_rows = _offset_batch_rows(panel)
    assert panel._offset_grid_table.value.shape == (4, 4)
    assert panel._offset_info_table.value.shape == (4, 3)
    assert panel._offset_batch_table.value.shape == (2, 6)
    assert len(panel._offset_batch_v_figure.data[0].x) == 0
    assert len(panel._offset_batch_i_figure.data[0].x) == 0
    assert len(panel._offset_batch_v_figure.data[1].x) == 0
    assert len(panel._offset_batch_i_figure.data[1].x) == 0
    assert len(panel._offset_batch_v_figure.data[2].x) == 0
    assert len(panel._offset_batch_i_figure.data[2].x) == 0
    assert np.issubdtype(panel._offset_grid_table.value["count"].dtype, np.integer)
    assert panel._offset_info_table.titles["parameter"] == "Parameter"
    assert panel._offset_info_table.titles["value"] == "Value"
    assert offset_info_rows.at["nu_Hz", "parameter"] == "<i>&nu;</i> (Hz)"
    assert offset_info_rows.at["upsample", "parameter"] == ("<i>N</i><sub>up</sub>")
    assert offset_info_rows.at["Voff_mV", "parameter"] == (
        "<i>V</i><sub>off</sub> (mV)"
    )
    assert offset_info_rows.at["Ioff_nA", "parameter"] == (
        "<i>I</i><sub>off</sub> (nA)"
    )
    assert isinstance(offset_info_rows.at["upsample", "value"], int)
    assert panel._offset_grid_table.value.iloc[0]["parameter"] == (
        "<i>V</i><sub>bins</sub> (mV)"
    )
    assert panel._offset_grid_table.value.iloc[2]["parameter"] == (
        "<i>V</i><sub>off</sub> (mV)"
    )
    assert list(offset_batch_rows["status"]) == ["idle", "idle"]
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
    assert sampling_info_rows.at["upsample", "parameter"] == ("<i>N</i><sub>up</sub>")
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
    fit_config_rows = _fit_config_rows(panel)
    fit_parameter_rows = _fit_parameter_rows(panel)
    optimizer_rows = _optimizer_rows(panel)
    model_rows = _model_rows(panel)
    assert panel._fit_button.name == "Fit"
    assert panel._fit_state.object == "Idle"
    assert panel._fit_inner_tabs._names == ["Model", "Optimizer"]
    assert panel._fit_config_table.value.shape == (4, 3)
    assert fit_config_rows.at["jax", "value"] is True
    assert fit_config_rows.at["conv", "value"] is True
    assert fit_config_rows.at["pat_enabled", "value"] is False
    assert fit_config_rows.at["noise_enabled", "value"] is False
    assert list(fit_parameter_rows.index) == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
    ]
    assert optimizer_rows.at["status", "value"] == "idle"
    assert optimizer_rows.at["solver", "value"] == "scipy.optimize.curve_fit"
    assert model_rows.at["noise_oversample", "value"] == "64"
    assert "\\begin{gathered}" in panel._model_html.object


def test_gui_app_accepts_spec_presets() -> None:
    traces = _make_traces()
    psd_spec = PSDSpec(detrend=False)
    offset_spec = OffsetSpec(
        Vbins_mV=np.linspace(-0.3, 0.3, 31, dtype=np.float64),
        Ibins_nA=np.linspace(-4.0, 4.0, 81, dtype=np.float64),
        Voff_mV=np.linspace(-0.02, 0.02, 21, dtype=np.float64),
        Ioff_nA=np.linspace(-0.15, 0.15, 31, dtype=np.float64),
        nu_Hz=11.0,
        upsample=7,
    )
    sampling_spec = SamplingSpec(
        upsample=6,
        Vbins_mV=np.linspace(-0.25, 0.25, 41, dtype=np.float64),
        Ibins_nA=np.linspace(-3.0, 3.0, 61, dtype=np.float64),
    )
    smoothing_spec = SmoothingSpec(
        median_bins=5,
        sigma_bins=1.25,
    )

    app = gui_app(
        traces,
        psd_spec=psd_spec,
        psd_nu_Hz=7.5,
        offset_spec=offset_spec,
        sampling_spec=sampling_spec,
        smoothing_spec=smoothing_spec,
    )

    assert hasattr(app, "_gui_panel")
    panel = app._gui_panel
    experimental_setting_rows = _experimental_setting_rows(panel)
    offset_rows = _offset_info_rows(panel)
    sampling_rows = _sampling_info_rows(panel)
    smoothing_rows = _sampling_smoothing_rows(panel)
    assert panel._shared_nu_Hz == pytest.approx(7.5)
    assert panel._experimental_detrend is psd_spec.detrend
    assert experimental_setting_rows.at["nu_Hz", "value"] == pytest.approx(7.5)
    assert experimental_setting_rows.at["detrend", "value"] is False
    assert offset_rows.at["nu_Hz", "value"] == pytest.approx(11.0)
    assert offset_rows.at["upsample", "value"] == 7
    assert np.isfinite(float(offset_rows.at["Voff_mV", "value"]))
    assert np.isfinite(float(offset_rows.at["Ioff_nA", "value"]))
    assert sampling_rows.at["upsample", "value"] == 6
    assert sampling_rows.at["Voff_mV", "value"] == pytest.approx(
        panel.state["offset"]["Voff_mV"]
    )
    assert sampling_rows.at["Ioff_nA", "value"] == pytest.approx(
        panel.state["offset"]["Ioff_nA"]
    )
    assert panel._sampling_smooth_toggle.value is True
    assert panel._smoothing_enabled is True
    assert smoothing_rows.at["median_bins", "value"] == 5
    assert smoothing_rows.at["sigma_bins", "value"] == pytest.approx(1.25)


def test_gui_app_accepts_offset_analysis_preset() -> None:
    traces = _make_traces()
    offset_spec = OffsetSpec(
        Vbins_mV=np.linspace(-0.3, 0.3, 31, dtype=np.float64),
        Ibins_nA=np.linspace(-4.0, 4.0, 81, dtype=np.float64),
        Voff_mV=np.linspace(-0.02, 0.02, 21, dtype=np.float64),
        Ioff_nA=np.linspace(-0.15, 0.15, 31, dtype=np.float64),
        nu_Hz=11.0,
        upsample=7,
    )
    offsets = offset_analysis(
        traces,
        spec=offset_spec,
        show_progress=False,
        backend="numpy",
    )

    app = gui_app(
        traces,
        offset_spec=offset_spec,
        offset_analysis=offsets,
    )

    panel = app._gui_panel
    offset_rows = _offset_info_rows(panel)

    assert panel._offset_batch_spec is not None
    assert panel._offset_specs_match(panel._offset_spec, offset_spec)
    assert panel._offset_batch_state.object == "Loaded"
    assert panel._offset_batch_status == ["done", "done"]
    assert panel.state["offset"]["Voff_mV"] == pytest.approx(offsets[0]["Voff_mV"])
    assert panel.state["offset"]["Ioff_nA"] == pytest.approx(offsets[0]["Ioff_nA"])
    assert offset_rows.at["Voff_mV", "value"] == pytest.approx(offsets[0]["Voff_mV"])
    assert offset_rows.at["Ioff_nA", "value"] == pytest.approx(offsets[0]["Ioff_nA"])


def test_gui_dispatches_to_run_or_serve(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    traces = _make_traces()
    expected = {"state": "done"}
    calls: list[str] = []

    def fake_run(*args, **kwargs):
        _ = args, kwargs
        calls.append("run")
        return expected

    def fake_serve(*args, **kwargs):
        _ = args, kwargs
        calls.append("serve")
        return object()

    monkeypatch.setattr(gui_mod, "run_gui", fake_run)
    monkeypatch.setattr(gui_mod, "serve_gui", fake_serve)

    result = gui(traces, wait=True)
    assert result is expected
    assert calls == ["run"]

    result = gui(traces, wait=False)
    assert result is None
    assert calls == ["run", "serve"]


def test_trace_change_repopulates_pipeline_and_left_plots() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
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
    panel = GUIPanel(_make_traces(), model="bcs_int")

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


def test_left_quantity_selectors_rebuild_shared_x_stacks() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")

    panel._left_v_quantity_selector.value = [
        "iv_v",
        "didv_v",
        "i_over_v_v",
        "dvdi_v",
    ]
    panel._left_i_quantity_selector.value = [
        "vi_i",
        "dvdi_i",
        "v_over_i_i",
        "didv_i",
    ]

    assert len(panel._iv_figure.data) == 20
    assert len(panel._vi_figure.data) == 20
    assert panel._iv_figure.layout.xaxis.matches == "x4"
    assert panel._vi_figure.layout.xaxis.matches == "x4"
    assert panel._iv_figure.layout.yaxis3.title.text == (
        "<i>I/V</i> (<i>G</i><sub>0</sub>)"
    )
    assert panel._iv_figure.layout.yaxis4.title.text == (
        "<i>dV/dI</i> (<i>R</i><sub>0</sub>)"
    )
    assert panel._vi_figure.layout.yaxis3.title.text == (
        "<i>V/I</i> (<i>R</i><sub>0</sub>)"
    )
    assert panel._vi_figure.layout.yaxis4.title.text == (
        "<i>dI/dV</i> (<i>G</i><sub>0</sub>)"
    )


def test_left_zoom_ranges_are_preserved_across_refresh() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")

    panel._iv_pane.relayout_data = {
        "xaxis.range[0]": -0.4,
        "xaxis.range[1]": 0.5,
        "yaxis.range[0]": -2.0,
        "yaxis.range[1]": 3.0,
        "yaxis2.range[0]": -0.2,
        "yaxis2.range[1]": 0.8,
    }

    panel._refresh_left_plots()

    assert tuple(panel._iv_figure.layout.xaxis.range) == (-0.4, 0.5)
    assert tuple(panel._iv_figure.layout.xaxis2.range) == (-0.4, 0.5)
    assert tuple(panel._iv_figure.layout.yaxis.range) == (-2.0, 3.0)
    assert tuple(panel._iv_figure.layout.yaxis2.range) == (-0.2, 0.8)


def test_apply_buttons_update_expected_pipeline_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"psd": 0, "offset": 0, "sampling": 0}
    real_psd_analysis = gui_mod.psd_analysis
    real_offset_analysis = gui_mod.offset_analysis
    real_sample = gui_mod.sample

    def wrapped_psd(*args, **kwargs):
        calls["psd"] += 1
        return real_psd_analysis(*args, **kwargs)

    def wrapped_offset(*args, **kwargs):
        calls["offset"] += 1
        return real_offset_analysis(*args, **kwargs)

    def wrapped_sampling(*args, **kwargs):
        calls["sampling"] += 1
        return real_sample(*args, **kwargs)

    monkeypatch.setattr(gui_mod, "psd_analysis", wrapped_psd)
    monkeypatch.setattr(gui_mod, "offset_analysis", wrapped_offset)
    monkeypatch.setattr(gui_mod, "sample", wrapped_sampling)

    panel = GUIPanel(_make_traces(), model="bcs_int")
    panel._set_fit_solution(_fake_solution(panel))
    panel._fit_curve = np.asarray(panel._fit_solution["I_fit_nA"], dtype=np.float64)
    calls.update(psd=0, offset=0, sampling=0)

    offset_info_table = panel._offset_info_table.value.copy()
    offset_info_table.loc[
        offset_info_table["key"] == "upsample",
        "value",
    ] = (
        panel._offset_spec.upsample + 1
    )
    panel._offset_info_table.value = offset_info_table
    panel._on_offset_apply(SimpleNamespace())

    assert calls == {"psd": 0, "offset": 1, "sampling": 1}
    assert panel.state["fit"] is None
    assert panel._offset_batch_spec is not None
    assert panel._offset_batch_status[0] == "done"
    assert panel._offset_batch_display_index == 0

    calls.update(psd=0, offset=0, sampling=0)
    sampling_info_table = panel._sampling_info_table.value.copy()
    sampling_info_table.loc[
        sampling_info_table["key"] == "upsample",
        "value",
    ] = (
        panel._sampling_spec.upsample + 1
    )
    panel._sampling_info_table.value = sampling_info_table
    panel._on_sampling_apply(SimpleNamespace())

    assert calls == {"psd": 0, "offset": 0, "sampling": 1}

    calls.update(psd=0, offset=0, sampling=0)
    experimental_table = panel._experimental_table.value.copy()
    experimental_table.loc[
        experimental_table["key"] == "nu_Hz",
        "value",
    ] = (
        panel._shared_nu_Hz + 1.0
    )
    panel._experimental_table.value = experimental_table
    panel._on_experimental_apply(SimpleNamespace())

    assert calls == {"psd": 2, "offset": 0, "sampling": 1}


def test_offset_batch_run_updates_progress_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    panel._trace_selector.value = 1
    captured: dict[str, object] = {}

    class DummyTimer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class DummyFuture:
        def __init__(self):
            self._done = False

        def done(self):
            return self._done

        def result(self):
            return None

        def finish(self):
            self._done = True

    future = DummyFuture()

    class DummyExecutor:
        def submit(
            self,
            function,
            traces,
            order,
            spec,
            event_queue,
            stop_event,
        ):
            captured["function"] = function
            captured["traces"] = traces
            captured["order"] = order
            captured["spec"] = spec
            captured["queue"] = event_queue
            captured["stop_event"] = stop_event
            return future

    def fake_add_periodic_callback(callback, period, start):
        _ = callback, period, start
        timer = DummyTimer()
        captured["timer"] = timer
        return timer

    monkeypatch.setattr(
        panel._pn.state,
        "add_periodic_callback",
        fake_add_periodic_callback,
    )
    panel._executor = DummyExecutor()

    panel._on_offset_batch_apply(SimpleNamespace())

    assert captured["order"] == [1, 0]
    assert panel._offset_batch_running is True
    assert panel._offset_apply_button.disabled is True
    assert panel._offset_batch_apply_button.disabled is True
    assert panel._fit_button.disabled is True
    assert panel._offset_grid_table.disabled is True
    assert panel._offset_info_table.disabled is True
    assert panel._offset_batch_spinner.value is True
    assert panel._offset_batch_progress.value == 0
    assert panel._offset_batch_state.object == "Running"

    offset_1 = offset_analysis(panel.traces[1], spec=replace(panel._offset_spec))
    offset_0 = offset_analysis(panel.traces[0], spec=replace(panel._offset_spec))
    queue = captured["queue"]
    queue.put(("running", 1, None, ""))
    panel._update_offset_batch_timer()
    rows = _offset_batch_rows(panel)
    assert rows.at[1, "status"] == "running"

    queue.put(("done", 1, offset_1, ""))
    panel._update_offset_batch_timer()
    rows = _offset_batch_rows(panel)
    assert panel._offset_batch_progress.value == 1
    assert panel._offset_batch_state.object == "Running"
    assert rows.at[1, "status"] == "done"
    assert rows.at[1, "Voff_mV"] == pytest.approx(offset_1["Voff_mV"])
    assert panel.active_index == 1
    assert panel.state["offset"]["Voff_mV"] == pytest.approx(offset_1["Voff_mV"])
    assert np.allclose(
        np.asarray(panel._offset_batch_v_figure.data[2].customdata, dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )

    queue.put(("done", 0, offset_0, ""))
    panel._update_offset_batch_timer()
    offset_rows = _offset_info_rows(panel)
    assert panel.active_index == 1
    assert panel._offset_batch_running is True
    assert offset_rows.at["Voff_mV", "value"] == pytest.approx(offset_1["Voff_mV"])
    assert offset_rows.at["Ioff_nA", "value"] == pytest.approx(offset_1["Ioff_nA"])
    assert np.asarray(panel._offset_batch_v_figure.data[1].customdata).size == 0

    future.finish()
    panel._update_offset_batch_timer()

    rows = _offset_batch_rows(panel)
    assert panel._offset_batch_running is False
    assert panel._offset_apply_button.disabled is False
    assert panel._offset_batch_apply_button.disabled is False
    assert panel._fit_button.disabled is False
    assert panel._offset_grid_table.disabled is False
    assert panel._offset_info_table.disabled is False
    assert panel._offset_batch_spinner.value is False
    assert panel._offset_batch_progress.value == 2
    assert panel._offset_batch_state.object == "Done"
    assert captured["timer"].stopped is True
    assert list(rows["status"]) == ["done", "done"]
    assert np.allclose(
        np.asarray(panel._offset_batch_v_figure.data[0].x, dtype=np.float64),
        np.asarray([1.0, 5.0], dtype=np.float64),
    )
    assert np.allclose(
        np.asarray(panel._offset_batch_i_figure.data[0].customdata, dtype=np.int64),
        np.asarray([0, 1], dtype=np.int64),
    )
    assert np.asarray(panel._offset_batch_i_figure.data[1].customdata).size == 0
    assert np.allclose(
        np.asarray(panel._offset_batch_i_figure.data[2].customdata, dtype=np.int64),
        np.asarray([1], dtype=np.int64),
    )

    panel._on_offset_batch_table_selection(SimpleNamespace(new=[0]))
    assert panel.active_index == 1
    assert _offset_info_rows(panel).at["Voff_mV", "value"] == pytest.approx(
        offset_1["Voff_mV"]
    )


def test_offset_batch_failure_is_recorded_without_aborting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    captured: dict[str, object] = {}

    class DummyTimer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class DummyFuture:
        def __init__(self):
            self._done = False

        def done(self):
            return self._done

        def result(self):
            return None

        def finish(self):
            self._done = True

    future = DummyFuture()

    class DummyExecutor:
        def submit(
            self,
            function,
            traces,
            order,
            spec,
            event_queue,
            stop_event,
        ):
            _ = function, traces, order, spec, stop_event
            captured["queue"] = event_queue
            return future

    def fake_add_periodic_callback(callback, period, start):
        _ = callback, period, start
        timer = DummyTimer()
        captured["timer"] = timer
        return timer

    monkeypatch.setattr(
        panel._pn.state,
        "add_periodic_callback",
        fake_add_periodic_callback,
    )
    panel._executor = DummyExecutor()

    panel._on_offset_batch_apply(SimpleNamespace())

    offset_1 = offset_analysis(panel.traces[1], spec=replace(panel._offset_spec))
    queue = captured["queue"]
    queue.put(("failed", 0, None, "RuntimeError: broken"))
    queue.put(("done", 1, offset_1, ""))
    future.finish()
    panel._update_offset_batch_timer()

    rows = _offset_batch_rows(panel)
    assert list(rows["status"]) == ["failed", "done"]
    assert np.isnan(rows.at[0, "Voff_mV"])
    assert rows.at[1, "Ioff_nA"] == pytest.approx(offset_1["Ioff_nA"])
    assert panel._offset_batch_errors[0] == "RuntimeError: broken"
    assert panel._offset_batch_state.object == "Done with failures"
    assert panel._offset_batch_running is False
    assert captured["timer"].stopped is True


def test_offset_batch_can_be_stopped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    captured: dict[str, object] = {}

    class DummyTimer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class DummyFuture:
        def __init__(self):
            self._done = False

        def done(self):
            return self._done

        def result(self):
            return None

        def finish(self):
            self._done = True

    future = DummyFuture()

    class DummyExecutor:
        def submit(
            self,
            function,
            traces,
            order,
            spec,
            event_queue,
            stop_event,
        ):
            captured["function"] = function
            captured["traces"] = traces
            captured["order"] = order
            captured["spec"] = spec
            captured["queue"] = event_queue
            captured["stop_event"] = stop_event
            return future

    def fake_add_periodic_callback(callback, period, start):
        _ = callback, period, start
        timer = DummyTimer()
        captured["timer"] = timer
        return timer

    monkeypatch.setattr(
        panel._pn.state,
        "add_periodic_callback",
        fake_add_periodic_callback,
    )
    panel._executor = DummyExecutor()

    panel._on_offset_batch_apply(SimpleNamespace())

    assert panel._offset_batch_stop_button.disabled is False
    panel._on_offset_batch_stop(SimpleNamespace())
    assert panel._offset_batch_stop_requested is True
    assert captured["stop_event"].is_set() is True
    assert panel._offset_batch_stop_button.disabled is True
    assert panel._offset_batch_state.object == "Stopping"

    offset_1 = offset_analysis(panel.traces[0], spec=replace(panel._offset_spec))
    captured["queue"].put(("done", 0, offset_1, ""))
    future.finish()
    panel._update_offset_batch_timer()

    rows = _offset_batch_rows(panel)
    assert list(rows["status"]) == ["done", "stopped"]
    assert panel._offset_batch_running is False
    assert panel._offset_batch_stop_button.disabled is True
    assert panel._offset_batch_state.object == "Stopped"
    assert captured["timer"].stopped is True


def test_offset_spec_edit_clears_cached_batch_results() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    cached = offset_analysis(panel.traces[0], spec=replace(panel._offset_spec))
    panel._offset_batch_spec = replace(panel._offset_spec)
    panel._offset_batch_results[0] = cached
    panel._offset_batch_status[0] = "done"
    panel._refresh_offset_batch_views()

    assert len(panel._offset_batch_v_figure.data[0].x) == 1

    panel._on_offset_spec_edited(SimpleNamespace())

    rows = _offset_batch_rows(panel)
    assert panel._offset_batch_spec is None
    assert panel._offset_batch_running is False
    assert list(rows["status"]) == ["idle", "idle"]
    assert len(panel._offset_batch_v_figure.data[0].x) == 0
    assert len(panel._offset_batch_i_figure.data[0].x) == 0


def test_gui_panel_matches_backend_outputs() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    trace = panel.state["trace"]

    psd_expected = psd_analysis(
        trace,
        spec=PSDSpec(detrend=True),
    )
    offset_expected = offset_analysis(trace, spec=panel._offset_spec)
    sampling_expected = sample(
        trace,
        samplingspec=panel._sampling_spec,
        offsetanalysis=offset_expected,
    )

    assert np.allclose(
        panel.state["psd"]["f_Hz"],
        psd_expected["f_Hz"],
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
    experimental_setting_rows = _experimental_setting_rows(panel)
    assert experimental_setting_rows.at["nu_Hz", "value"] == pytest.approx(
        panel._shared_nu_Hz
    )
    assert bool(experimental_setting_rows.at["detrend", "value"]) is True
    offset_rows = _offset_info_rows(panel)
    assert offset_rows.at["nu_Hz", "value"] == pytest.approx(panel._offset_spec.nu_Hz)
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
    panel = GUIPanel(_make_traces(), model="bcs_int")
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
    assert len(panel._vi_figure.data) == 10
    assert _trace_by_name(panel._vi_figure, "Fit").visible is True


def test_fit_model_table_controls_active_model_and_parameters() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")

    panel._on_fit_config_edit(SimpleNamespace(row=0, column="value", value=True))
    assert panel.model_key == "bcs_int_jax"

    panel._on_fit_config_edit(SimpleNamespace(row=1, column="value", value=True))
    assert panel.model_key == "bcs_conv_jax"

    panel._on_fit_config_edit(SimpleNamespace(row=2, column="value", value=True))
    assert panel.model_key == "bcs_conv_jax_pat"
    assert list(_fit_parameter_rows(panel).index) == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "A_mV",
        "nu_GHz",
    ]

    panel._on_parameter_edit(SimpleNamespace(row=4, column="guess", value=0.25))
    assert _fit_parameter_rows(panel).at["A_mV", "guess"] == pytest.approx(0.25)

    panel._on_fit_config_edit(SimpleNamespace(row=2, column="value", value=False))
    assert list(_fit_parameter_rows(panel).index) == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
    ]

    panel._on_fit_config_edit(SimpleNamespace(row=2, column="value", value=True))
    assert _fit_parameter_rows(panel).at["A_mV", "guess"] == pytest.approx(0.25)

    panel._on_fit_config_edit(SimpleNamespace(row=3, column="value", value=True))
    assert panel.model_key == "bcs_conv_jax_pat_noise"
    assert list(_fit_parameter_rows(panel).index[-1:]) == ["sigma_V_mV"]
    assert _model_rows(panel).at["noise_oversample", "value"] == "64"
    assert _optimizer_rows(panel).at["solver", "value"] == "scipy.optimize.curve_fit"


def test_fit_controls_show_running_state_and_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    fake_solution = _fake_solution(panel)
    captured: dict[str, object] = {}

    class DummyTimer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

    class DummyFuture:
        def __init__(self, result):
            self._result = result
            self._callback = None

        def result(self):
            return self._result

        def add_done_callback(self, callback):
            self._callback = callback

        def finish(self):
            assert self._callback is not None
            self._callback(self)

    class DummyExecutor:
        def submit(self, function, sampling, **kwargs):
            _ = function, sampling, kwargs
            future = DummyFuture(fake_solution)
            captured["future"] = future
            return future

    monkeypatch.setattr(panel._pn.state, "execute", lambda callback: callback())

    def fake_add_periodic_callback(callback, period, start):
        _ = callback, period, start
        timer = DummyTimer()
        captured["timer"] = timer
        return timer

    monkeypatch.setattr(
        panel._pn.state,
        "add_periodic_callback",
        fake_add_periodic_callback,
    )
    panel._executor = DummyExecutor()

    panel._on_fit_clicked(SimpleNamespace())

    assert panel._fit_running is True
    assert panel._fit_button.disabled is True
    assert panel._spinner.value is True
    assert panel._fit_state.object == "Running (0.0 s)"

    panel._fit_started_at = time.perf_counter() - 1.2
    panel._update_fit_timer()
    assert panel._fit_state.object.startswith("Running (1.")

    captured["future"].finish()

    assert panel._fit_running is False
    assert panel._fit_button.disabled is False
    assert panel._spinner.value is False
    assert panel._fit_state.object.startswith("Done (")
    assert captured["timer"].stopped is True
    assert _optimizer_rows(panel).at["status", "value"] == "done"


def test_sampling_apply_can_enable_smoothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    raw_sampling = np.asarray(panel.state["sampling"]["I_nA"], dtype=np.float64)

    def fake_smoothing(*args, **kwargs):
        _ = args, kwargs
        smoothed = panel._require_raw_sampling().copy()
        smoothed["I_nA"] = np.asarray(raw_sampling, dtype=np.float64) * 0.9
        return smoothed

    monkeypatch.setattr(
        gui_mod,
        "smooth",
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
    panel = GUIPanel(_make_traces(), model="bcs_int")
    rows_0 = _sampling_info_rows(panel)
    offset_0 = panel.state["offset"]
    assert rows_0.at["Voff_mV", "value"] == pytest.approx(offset_0["Voff_mV"])
    assert rows_0.at["Ioff_nA", "value"] == pytest.approx(offset_0["Ioff_nA"])

    panel._trace_selector.value = 1

    sampling_rows = _sampling_info_rows(panel)
    offset_1 = panel.state["offset"]
    assert sampling_rows.at["Voff_mV", "value"] == pytest.approx(offset_1["Voff_mV"])
    assert sampling_rows.at["Ioff_nA", "value"] == pytest.approx(offset_1["Ioff_nA"])


def test_offset_info_values_are_display_rounded() -> None:
    panel = GUIPanel(_make_traces(), model="bcs_int")
    panel._offset = {
        **panel._offset,
        "Voff_mV": 0.00450000000000001,
        "Ioff_nA": -0.003200000000000001,
    }

    rows = panel._offset_info_frame().set_index("key")

    assert rows.at["Voff_mV", "value"] == 0.0045
    assert rows.at["Ioff_nA", "value"] == -0.0032


def test_run_gui_returns_last_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = {
        "active_index": 0,
        "trace": _make_traces()[0],
        "psd": {"f_Hz": np.zeros(1)},
        "offset": {"Voff_mV": 0.0, "Ioff_nA": 0.0},
        "sampling": {"I_nA": np.zeros(3), "Vbins_mV": np.zeros(3)},
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
        model="bcs_int",
    )

    assert result is expected
