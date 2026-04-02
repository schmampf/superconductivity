from __future__ import annotations

import importlib
import time
from dataclasses import replace
from pathlib import Path
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
gui_data_mod = importlib.import_module("superconductivity.gui.tabs.data")
gui_measurement_mod = importlib.import_module(
    "superconductivity.gui.tabs.measurement",
)
gui_file_mod = importlib.import_module("superconductivity.evaluation.traces.file")


def _trace_by_name(figure, name: str):
    for trace in figure.data:
        if trace.name == name:
            return trace
    raise AssertionError(f"trace {name!r} not found")


def _experimental_setting_rows(panel: GUIPanel):
    return panel._experimental_table.value.reset_index(drop=True).set_index("key")


def _filespec_rows(panel: GUIPanel):
    return panel._filespec_table.value.reset_index(drop=True).set_index("key")


def _keysspec_rows(panel: GUIPanel):
    return panel._keysspec_table.value.reset_index(drop=True).set_index("key")


def _tracespec_rows(panel: GUIPanel):
    return panel._tracespec_table.value.reset_index(drop=True).set_index("key")


def _measurement_rows(panel: GUIPanel):
    return panel._measurement_table.value.reset_index(drop=True)


def _data_y_rows(panel: GUIPanel):
    return panel._data_y_table.value.reset_index(drop=True)


def _data_x_rows(panel: GUIPanel):
    return panel._data_x_table.value.reset_index(drop=True)


def _data_timeframe_rows(panel: GUIPanel):
    return panel._data_timeframe_table.value.reset_index(drop=True)


def _specific_key_name_rows(panel: GUIPanel):
    return panel._specific_key_name_table.value.reset_index(drop=True)


def _keys_rows(panel: GUIPanel):
    return panel._keys_table.value.reset_index(drop=True)


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
    yvalue: float | str | None,
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
        "maxfev": None,
    }


def test_gui_app_builds_without_server() -> None:
    app = gui_app(_make_traces())

    assert hasattr(app, "_gui_panel")
    panel = app._gui_panel
    assert panel.state["active_index"] == 0
    assert panel.state["sampling"]["I_nA"].shape == (1601,)
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
    assert panel._right_tabs._names == [
        "Measurement",
        "Data",
        "PSD Analysis",
        "Offset Analysis",
        "Sampling",
        "BCS fitting",
    ]
    assert panel._filespec_browse_button.name == "Browse..."
    assert panel._filespec_update_button.name == "Update All"
    assert panel._measurement_table.value.shape == (0, 1)
    assert panel._data_x_table.value.shape == (1, 4)
    assert panel._data_y_table.titles["quantity"] == "Y quantities"
    assert panel._data_x_table.titles["quantity"] == "X quantity"
    assert panel._data_timeframe_table.titles["specific_key"] == "Time frame"
    assert panel._data_figure.layout.legend.orientation == "h"
    assert panel._specific_key_name_table.value.shape == (0, 1)
    assert panel._keys_table.value.shape == (0, 3)
    assert panel._keys_table.titles["yvalue"] == "Value"
    assert panel._data_status.visible is True
    assert "Select a file and measurement" in panel._data_status.object
    assert panel._fit_inner_tabs._names == ["Model", "Optimizer"]
    assert panel._filespec_table.value.shape == (3, 3)
    assert panel._keysspec_table.value.shape == (8, 3)
    assert panel._tracespec_table.value.shape == (7, 3)
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
    filespec = FileSpec(
        h5path="demo.h5",
        location="/tmp",
        measurement="measurement_a",
    )
    keysspec = KeysSpec(
        strip0="=",
        label="Power",
        html_label="<i>P</i>",
    )
    tracespec = TraceSpec(
        amp_voltage=2.0,
        amp_current=3.0,
        skip=5,
    )
    keys = Keys.from_fields(
        specific_keys=traces.specific_keys,
        indices=np.asarray([0, 1], dtype=np.int64),
        yvalues=np.asarray([1.0, 5.0], dtype=np.float64),
        spec=keysspec,
    )
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
        filespec=filespec,
        keysspec=keysspec,
        tracespec=tracespec,
        keys=keys,
        psdspec=psd_spec,
        offsetspec=offset_spec,
        samplingspec=sampling_spec,
        smoothingspec=smoothing_spec,
    )

    assert hasattr(app, "_gui_panel")
    panel = app._gui_panel
    assert panel._filespec == filespec
    assert panel._keysspec == keysspec
    assert panel._tracespec == tracespec
    assert panel._keys is keys
    filespec_rows = _filespec_rows(panel)
    keysspec_rows = _keysspec_rows(panel)
    tracespec_rows = _tracespec_rows(panel)
    experimental_setting_rows = _experimental_setting_rows(panel)
    offset_rows = _offset_info_rows(panel)
    sampling_rows = _sampling_info_rows(panel)
    smoothing_rows = _sampling_smoothing_rows(panel)
    assert filespec_rows.at["h5path", "value"] == "demo.h5"
    assert filespec_rows.at["location", "value"] == "/tmp"
    assert filespec_rows.at["measurement", "value"] == "measurement_a"
    assert keysspec_rows.at["strip0", "value"] == "="
    assert keysspec_rows.at["label", "value"] == "Power"
    assert keysspec_rows.at["html_label", "value"] == "<i>P</i>"
    assert panel._keys_table.titles["yvalue"] == "Value"
    assert tracespec_rows.at["amp_voltage", "value"] == pytest.approx(2.0)
    assert tracespec_rows.at["amp_current", "value"] == pytest.approx(3.0)
    assert tracespec_rows.at["skip", "value"] == 5
    assert panel._shared_nu_Hz == pytest.approx(13.7)
    assert panel._experimental_detrend is psd_spec.detrend
    assert experimental_setting_rows.at["nu_Hz", "value"] == pytest.approx(13.7)
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


def test_data_tab_loads_inventory_and_defaults_to_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_measurement_keys",
        lambda filespec: {
            "adwin/V1": "measurement/a/*/*/adwin/V1",
            "bluefors/Tsample": "measurement/a/*/*/bluefors/Tsample",
        },
    )
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_status_keys",
        lambda filespec: {
            "motor/position/value": "status/motor/position/value",
        },
    )
    monkeypatch.setattr(
        gui_data_mod.GUIDataTabMixin,
        "_load_data_specific_key_windows",
        lambda self, filespec: {
            "raw_b": (2.0, 3.0),
            "raw_a": (0.0, 1.0),
        },
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: ["raw_a", "raw_b"],
    )

    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    y_rows = _data_y_rows(panel)
    x_rows = _data_x_rows(panel)
    timeframe_rows = _data_timeframe_rows(panel)

    assert list(y_rows["quantity"]) == [
        "clear",
        "measurement: adwin/V1",
        "measurement: bluefors/Tsample",
        "status: motor/position/value",
    ]
    assert panel._data_y_table.selection == [0]
    assert x_rows.iloc[0]["key"] == "time"
    assert panel._data_x_table.selection == [0]
    assert list(timeframe_rows["specific_key"]) == ["all", "raw_a", "raw_b"]
    assert panel._data_timeframe_table.selection == [0]


def test_data_tab_interpolates_against_selected_x_quantity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_measurement_keys",
        lambda filespec: {
            "adwin/V1": "measurement/a/*/*/adwin/V1",
            "adwin/V2": "measurement/a/*/*/adwin/V2",
        },
    )
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_status_keys",
        lambda filespec: {
            "motor/position/value": "status/motor/position/value",
        },
    )
    monkeypatch.setattr(
        gui_data_mod.GUIDataTabMixin,
        "_load_data_specific_key_windows",
        lambda self, filespec: {"raw_a": (0.0, 2.0)},
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: ["raw_a"],
    )

    def _measurement_series(filespec: FileSpec, key: str):
        if key == "adwin/V1":
            return (
                np.asarray([0.0, 1.0, 2.0]),
                np.asarray([10.0, 20.0, 30.0]),
            )
        if key == "adwin/V2":
            return (
                np.asarray([0.0, 1.0, 2.0]),
                np.asarray([100.0, 200.0, 300.0]),
            )
        raise KeyError(key)

    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_measurement_series",
        _measurement_series,
    )
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_status_series",
        lambda filespec, key: (
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([1.0, 1.5, 2.0]),
        ),
    )

    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    y_rows = _data_y_rows(panel)
    v1_row = int(y_rows.index[y_rows["key"] == "adwin/V1"][0])
    motor_row = int(
        y_rows.index[y_rows["key"] == "motor/position/value"][0]
    )
    panel._on_data_y_selection_changed(SimpleNamespace(new=[v1_row, motor_row]))

    x_rows = _data_x_rows(panel)
    v2_row = int(x_rows.index[x_rows["key"] == "adwin/V2"][0])
    panel._on_data_x_selection_changed(SimpleNamespace(new=[v2_row]))

    assert len(panel._data_figure.data) == 2
    assert np.allclose(panel._data_figure.data[0].x, [100.0, 200.0, 300.0])
    assert np.allclose(panel._data_figure.data[0].y, [10.0, 20.0, 30.0])
    assert panel._data_figure.layout.yaxis.title.text is None
    assert panel._data_figure.layout.yaxis.side == "right"
    assert panel._data_figure.layout.yaxis.tickformat == ".2g"
    assert panel._data_figure.layout.yaxis2.title.text is None
    assert panel._data_figure.layout.yaxis2.side == "right"
    assert panel._data_figure.layout.yaxis2.tickformat == ".2g"


def test_data_tab_timeframe_selection_crops_visible_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_measurement_keys",
        lambda filespec: {"adwin/V1": "measurement/a/*/*/adwin/V1"},
    )
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_status_keys",
        lambda filespec: {},
    )
    monkeypatch.setattr(
        gui_data_mod.GUIDataTabMixin,
        "_load_data_specific_key_windows",
        lambda self, filespec: {
            "raw_a": (0.0, 1.0),
            "raw_b": (2.0, 3.0),
        },
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: ["raw_a", "raw_b"],
    )
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_measurement_series",
        lambda filespec, key: (
            np.asarray([0.0, 1.0, 2.0, 3.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0]),
        ),
    )

    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    y_rows = _data_y_rows(panel)
    v1_row = int(y_rows.index[y_rows["key"] == "adwin/V1"][0])
    panel._on_data_y_selection_changed(SimpleNamespace(new=[v1_row]))
    assert np.allclose(panel._data_figure.data[0].x, [0.0, 1.0, 2.0, 3.0])

    panel._on_data_timeframe_selection_changed(SimpleNamespace(new=[1]))
    assert np.allclose(panel._data_figure.data[0].x, [0.0, 1.0])

    panel._on_data_timeframe_selection_changed(SimpleNamespace(new=[]))
    assert np.allclose(panel._data_figure.data[0].x, [0.0, 1.0, 2.0, 3.0])


def test_data_tab_time_axis_uses_measurement_relative_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_measurement_keys",
        lambda filespec: {"adwin/V1": "measurement/a/*/*/adwin/V1"},
    )
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_status_keys",
        lambda filespec: {},
    )
    monkeypatch.setattr(
        gui_data_mod.GUIDataTabMixin,
        "_load_data_specific_key_windows",
        lambda self, filespec: {
            "raw_a": (10.0, 11.0),
            "raw_b": (12.0, 13.0),
        },
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: ["raw_a", "raw_b"],
    )
    monkeypatch.setattr(
        gui_data_mod.trace_data_module,
        "get_measurement_series",
        lambda filespec, key: (
            np.asarray([10.0, 11.0, 12.0, 13.0]),
            np.asarray([0.0, 1.0, 2.0, 3.0]),
        ),
    )

    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    y_rows = _data_y_rows(panel)
    v1_row = int(y_rows.index[y_rows["key"] == "adwin/V1"][0])
    panel._on_data_y_selection_changed(SimpleNamespace(new=[v1_row]))

    assert np.allclose(panel._data_figure.data[0].x, [0.0, 1.0, 2.0, 3.0])
    assert panel._data_figure.layout.xaxis.title.text == (
        "<i>t</i> - <i>t</i><sub>0</sub> (s)"
    )


def test_measurement_tab_tracespec_tables_are_editable() -> None:
    panel = GUIPanel(_make_traces())

    panel._on_tracespec_core_edit(
        SimpleNamespace(row=0, column="value", value="2.5")
    )
    panel._on_tracespec_other_edit(
        SimpleNamespace(row=1, column="value", value="(2, 3)")
    )
    panel._on_tracespec_other_edit(
        SimpleNamespace(row=3, column="value", value="false")
    )

    tracespec_rows = _tracespec_rows(panel)
    assert panel._tracespec is not None
    assert panel._tracespec_core_table.layout == "fit_columns"
    assert panel._tracespec_other_table.layout == "fit_columns"
    assert panel._tracespec.amp_voltage == pytest.approx(2.5)
    assert panel._tracespec.skip == (2, 3)
    assert panel._tracespec.time_relative is False
    assert tracespec_rows.at["amp_voltage", "value"] == pytest.approx(2.5)
    assert tracespec_rows.at["skip", "value"] == "(2, 3)"
    assert tracespec_rows.at["time_relative", "value"] is False


def test_traces_tab_keysspec_table_is_editable() -> None:
    panel = GUIPanel(_make_traces())

    edits = [
        (0, "GHz_"),
        (1, "V"),
        (2, "no_irradiation"),
        (4, "1000"),
        (5, "Aout_mV"),
        (6, "<i>A</i>"),
    ]
    for row, value in edits:
        panel._on_keysspec_edit(
            SimpleNamespace(row=row, column="value", value=value)
        )

    assert panel._keysspec is None
    assert panel._keys is None
    rows = _keysspec_rows(panel)
    assert rows.at["strip0", "value"] == "GHz_"
    assert rows.at["strip1", "value"] == "V"
    assert rows.at["remove_key", "value"] == "no_irradiation"
    assert rows.at["norm", "value"] == "1000"
    assert rows.at["label", "value"] == "Aout_mV"
    assert rows.at["html_label", "value"] == "<i>A</i>"
    assert panel._keys_table.titles["yvalue"] == "Value"


def test_traces_tab_keysspec_edit_does_not_auto_update_keys_and_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    loaded_keys = Keys.from_fields(
        specific_keys=["mode=alpha", "mode=beta"],
        indices=np.asarray([0, 1], dtype=np.int64),
        yvalues=np.asarray(["alpha", "beta"], dtype=object),
        spec=KeysSpec(strip0="=", label="mode"),
    )
    loaded_traces = Traces(
        traces=[
            _make_iv_trace("mode=alpha", 0, "alpha", v_shift_mV=0.1, i_shift_nA=0.2),
            _make_iv_trace("mode=beta", 1, "beta", v_shift_mV=0.3, i_shift_nA=0.4),
        ],
    )

    monkeypatch.setattr(
        gui_measurement_mod,
        "get_keys",
        lambda *, filespec, keysspec: loaded_keys,
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_traces",
        lambda *, filespec, keys, tracespec: loaded_traces,
    )

    event = SimpleNamespace(row=0, column="value", value="GHz_")
    panel._on_keysspec_edit(event)

    specific_key_rows = _keys_rows(panel)
    assert panel._keysspec is None
    rows = _keysspec_rows(panel)
    assert rows.at["strip0", "value"] == "GHz_"
    assert panel._keys is None
    assert panel.traces is not loaded_traces
    assert list(specific_key_rows["specific_key"]) == []
    assert list(specific_key_rows["yvalue"]) == []


def test_traces_tab_keysspec_reverts_invalid_text_and_shows_error() -> None:
    panel = GUIPanel(_make_traces())

    event = SimpleNamespace(row=7, column="value", value="not valid")
    panel._on_keysspec_edit(event)

    rows = _keysspec_rows(panel)
    assert rows.at["limits", "value"] == ""
    assert panel._keysspec is None
    assert panel._keysspec_error.visible is True
    assert "limits must be a valid Python literal." in panel._keysspec_error.object


def test_traces_tab_update_all_applies_edited_keysspec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    loaded_keys = Keys.from_fields(
        specific_keys=["mode=alpha", "mode=beta", "mode=gamma"],
        indices=np.asarray([0, 1, 2], dtype=np.int64),
        yvalues=np.asarray(["alpha", "beta", "gamma"], dtype=object),
        spec=KeysSpec(strip0="=", label="mode"),
    )
    loaded_traces = Traces(
        traces=[
            _make_iv_trace("mode=alpha", 0, "alpha", v_shift_mV=0.1, i_shift_nA=0.2),
            _make_iv_trace("mode=beta", 1, "beta", v_shift_mV=0.3, i_shift_nA=0.4),
            _make_iv_trace("mode=gamma", 2, "gamma", v_shift_mV=0.5, i_shift_nA=0.6),
        ],
    )

    monkeypatch.setattr(
        gui_measurement_mod,
        "get_keys",
        lambda *, filespec, keysspec: loaded_keys,
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_traces",
        lambda *, filespec, keys, tracespec: loaded_traces,
    )

    event = SimpleNamespace(row=0, column="value", value="GHz_")
    panel._on_keysspec_edit(event)
    assert panel._keysspec is None
    rows = _keysspec_rows(panel)
    assert rows.at["strip0", "value"] == "GHz_"
    assert list(_keys_rows(panel)["specific_key"]) == []

    panel._on_update_file(SimpleNamespace())

    specific_key_rows = _keys_rows(panel)
    assert panel._keys is loaded_keys
    assert panel._keysspec is not None
    assert panel._keysspec.strip0 == "GHz_"
    assert panel.traces is loaded_traces
    assert list(specific_key_rows["specific_key"]) == [
        "mode=alpha",
        "mode=beta",
        "mode=gamma",
    ]
    assert list(specific_key_rows["yvalue"]) == ["alpha", "beta", "gamma"]


def test_traces_tab_lists_and_selects_measurements(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeMeasurementGroup:
        def __init__(self, keys_list: list[str]) -> None:
            self._keys = keys_list

        def keys(self) -> list[str]:
            return list(self._keys)

    class _FakeFile:
        def __init__(self, groups: dict[str, _FakeMeasurementGroup]) -> None:
            self._groups = groups

        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def __contains__(self, key: str) -> bool:
            return key in self._groups

        def __getitem__(self, key: str) -> _FakeMeasurementGroup:
            return self._groups[key]

    class _FakeH5py:
        def __init__(self, path: Path) -> None:
            self._path = path
            self._file = _FakeFile(
                {
                    "measurement": _FakeMeasurementGroup(
                        ["measurement_b", "measurement_a"]
                    ),
                }
            )

        def File(self, path: Path, mode: str):
            assert path == self._path
            assert mode == "r"
            return self._file

    expected_path = Path("/tmp/root/data.h5")
    monkeypatch.setattr(
        gui_file_mod,
        "_import_h5py",
        lambda: _FakeH5py(expected_path),
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: (
            ["raw_a", "raw_b"]
            if self.measurement == "measurement_a"
            else ["raw_c", "raw_d", "raw_e"]
        ),
    )
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="data.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    measurement_rows = _measurement_rows(panel)
    assert list(measurement_rows["measurement"]) == [
        "measurement_a",
        "measurement_b",
    ]
    assert panel._measurement_table.selection == [0]
    assert list(_specific_key_name_rows(panel)["specific_key"]) == ["raw_a", "raw_b"]

    panel._on_measurement_selection_changed(SimpleNamespace(new=[1]))

    assert panel._filespec is not None
    assert panel._filespec.measurement == "measurement_b"
    assert _filespec_rows(panel).at["measurement", "value"] == "measurement_b"
    assert list(_specific_key_name_rows(panel)["specific_key"]) == [
        "raw_c",
        "raw_d",
        "raw_e",
    ]


def test_traces_tab_lists_keys_preview_only_when_file_is_available() -> None:
    panel = GUIPanel(_make_traces())

    specific_key_name_rows = _specific_key_name_rows(panel)
    specific_key_rows = _keys_rows(panel)
    assert list(specific_key_name_rows["specific_key"]) == []
    assert list(specific_key_rows["trace_index"]) == []
    assert list(specific_key_rows["specific_key"]) == []
    assert list(specific_key_rows["yvalue"]) == []



def test_traces_tab_keys_preview_is_independent_of_active_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    loaded_keys = Keys.from_fields(
        specific_keys=["mode=alpha", "mode=beta"],
        indices=np.asarray([0, 1], dtype=np.int64),
        yvalues=np.asarray(["alpha", "beta"], dtype=object),
        spec=KeysSpec(strip0="=", label="mode"),
    )

    monkeypatch.setattr(
        gui_measurement_mod,
        "get_keys",
        lambda *, filespec, keysspec: loaded_keys,
    )

    panel._sync_trace_widgets_from_state()
    before = _keys_rows(panel).copy()

    panel._trace_selector.value = 1

    after = _keys_rows(panel)
    assert after.equals(before)


def test_traces_tab_browse_updates_filespec_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    monkeypatch.setattr(
        gui_measurement_mod,
        "_browse_h5_file",
        lambda filespec: Path("/tmp/root/sub/demo_2.hdf5"),
    )

    panel._on_filespec_browse(SimpleNamespace())

    filespec_rows = _filespec_rows(panel)
    assert panel._filespec is not None
    assert panel._filespec.location == "/tmp/root"
    assert panel._filespec.h5path == "sub/demo_2.hdf5"
    assert panel._filespec.measurement == "measurement_a"
    assert filespec_rows.at["h5path", "value"] == "sub/demo_2.hdf5"
    assert filespec_rows.at["location", "value"] == "/tmp/root"
    assert filespec_rows.at["measurement", "value"] == "measurement_a"


def test_traces_tab_update_file_reloads_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
        keysspec=KeysSpec(
            strip0="=",
            html_label="<i>P</i>",
        ),
    )

    loaded_keys = Keys.from_fields(
        specific_keys=["c", "d"],
        indices=np.asarray([0, 1], dtype=np.int64),
        yvalues=np.asarray([2.5, 7.5], dtype=np.float64),
        spec=panel._keysspec,
    )
    loaded_traces = Traces(
        traces=[
            _make_iv_trace("c", 0, 2.5, v_shift_mV=0.1, i_shift_nA=0.2),
            _make_iv_trace("d", 1, 7.5, v_shift_mV=0.3, i_shift_nA=0.4),
        ],
    )

    monkeypatch.setattr(
        gui_measurement_mod,
        "get_keys",
        lambda *, filespec, keysspec: loaded_keys,
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_traces",
        lambda *, filespec, keys, tracespec: loaded_traces,
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: ["raw_c", "raw_d", "raw_e"],
    )

    panel._trace_selector.value = 1
    panel._on_update_file(SimpleNamespace())

    specific_key_rows = _keys_rows(panel)
    assert panel.active_index == 0
    assert panel._trace_selector.value == 0
    assert panel.traces is loaded_traces
    assert panel._keys is loaded_keys
    assert list(_specific_key_name_rows(panel)["specific_key"]) == [
        "raw_c",
        "raw_d",
        "raw_e",
    ]
    assert list(specific_key_rows["specific_key"]) == ["c", "d"]
    assert list(specific_key_rows["yvalue"]) == ["2.5", "7.5"]
    assert panel._keys_table.titles["yvalue"] == "Value"


def test_traces_tab_update_file_uses_selected_measurement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    loaded_keys_a = Keys.from_fields(
        specific_keys=["a0", "a1"],
        indices=np.asarray([0, 1], dtype=np.int64),
        yvalues=np.asarray([1.0, 2.0], dtype=np.float64),
        spec=KeysSpec(strip0="="),
    )
    loaded_keys_b = Keys.from_fields(
        specific_keys=["b0", "b1", "b2"],
        indices=np.asarray([0, 1, 2], dtype=np.int64),
        yvalues=np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
        spec=KeysSpec(strip0="="),
    )
    loaded_traces_a = Traces(
        traces=[
            _make_iv_trace("a0", 0, 1.0, v_shift_mV=0.1, i_shift_nA=0.1),
            _make_iv_trace("a1", 1, 2.0, v_shift_mV=0.2, i_shift_nA=0.2),
        ],
    )
    loaded_traces_b = Traces(
        traces=[
            _make_iv_trace("b0", 0, 3.0, v_shift_mV=0.1, i_shift_nA=0.1),
            _make_iv_trace("b1", 1, 4.0, v_shift_mV=0.2, i_shift_nA=0.2),
            _make_iv_trace("b2", 2, 5.0, v_shift_mV=0.3, i_shift_nA=0.3),
        ],
    )

    monkeypatch.setattr(
        FileSpec,
        "mkeys",
        lambda self: ["measurement_a", "measurement_b"],
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: (
            ["raw_a0", "raw_a1"]
            if self.measurement == "measurement_a"
            else ["raw_b0", "raw_b1", "raw_b2"]
        ),
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_keys",
        lambda *, filespec, keysspec: (
            loaded_keys_a
            if filespec.measurement == "measurement_a"
            else loaded_keys_b
        ),
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_traces",
        lambda *, filespec, keys, tracespec: (
            loaded_traces_a
            if filespec.measurement == "measurement_a"
            else loaded_traces_b
        ),
    )

    panel._on_measurement_selection_changed(SimpleNamespace(new=[0]))
    panel._on_update_file(SimpleNamespace())
    assert list(_specific_key_name_rows(panel)["specific_key"]) == ["raw_a0", "raw_a1"]
    assert list(_keys_rows(panel)["specific_key"]) == ["a0", "a1"]

    panel._on_measurement_selection_changed(SimpleNamespace(new=[1]))
    panel._on_update_file(SimpleNamespace())

    assert panel._filespec is not None
    assert panel._filespec.measurement == "measurement_b"
    assert list(_specific_key_name_rows(panel)["specific_key"]) == [
        "raw_b0",
        "raw_b1",
        "raw_b2",
    ]
    assert list(_keys_rows(panel)["specific_key"]) == ["b0", "b1", "b2"]


def test_traces_tab_measurement_selection_auto_updates_keys_and_traces(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_keys_a = Keys.from_fields(
        specific_keys=["a0", "a1"],
        indices=np.asarray([0, 1], dtype=np.int64),
        yvalues=np.asarray([1.0, 2.0], dtype=np.float64),
        spec=KeysSpec(strip0="="),
    )
    loaded_keys_b = Keys.from_fields(
        specific_keys=["b0", "b1", "b2"],
        indices=np.asarray([0, 1, 2], dtype=np.int64),
        yvalues=np.asarray([3.0, 4.0, 5.0], dtype=np.float64),
        spec=KeysSpec(strip0="="),
    )
    loaded_traces_a = Traces(
        traces=[
            _make_iv_trace("a0", 0, 1.0, v_shift_mV=0.1, i_shift_nA=0.1),
            _make_iv_trace("a1", 1, 2.0, v_shift_mV=0.2, i_shift_nA=0.2),
        ],
    )
    loaded_traces_b = Traces(
        traces=[
            _make_iv_trace("b0", 0, 3.0, v_shift_mV=0.1, i_shift_nA=0.1),
            _make_iv_trace("b1", 1, 4.0, v_shift_mV=0.2, i_shift_nA=0.2),
            _make_iv_trace("b2", 2, 5.0, v_shift_mV=0.3, i_shift_nA=0.3),
        ],
    )

    monkeypatch.setattr(
        FileSpec,
        "mkeys",
        lambda self: ["measurement_a", "measurement_b"],
    )
    monkeypatch.setattr(
        FileSpec,
        "skeys",
        lambda self: (
            ["raw_a0", "raw_a1"]
            if self.measurement == "measurement_a"
            else ["raw_b0", "raw_b1", "raw_b2"]
        ),
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_keys",
        lambda *, filespec, keysspec: (
            loaded_keys_a
            if filespec.measurement == "measurement_a"
            else loaded_keys_b
        ),
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_traces",
        lambda *, filespec, keys, tracespec: (
            loaded_traces_a
            if filespec.measurement == "measurement_a"
            else loaded_traces_b
        ),
    )

    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
    )

    panel._on_measurement_selection_changed(SimpleNamespace(new=[1]))

    assert panel._filespec is not None
    assert panel._filespec.measurement == "measurement_b"
    assert panel._keys is loaded_keys_b
    assert panel.traces is loaded_traces_b
    assert list(_specific_key_name_rows(panel)["specific_key"]) == [
        "raw_b0",
        "raw_b1",
        "raw_b2",
    ]
    assert list(_keys_rows(panel)["specific_key"]) == ["b0", "b1", "b2"]


def test_traces_tab_update_file_shows_raw_yvalues_when_parsing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(
        _make_traces(),
        filespec=FileSpec(
            h5path="demo.h5",
            location="/tmp/root",
            measurement="measurement_a",
        ),
        keysspec=KeysSpec(
            strip0="=",
            label="mode",
        ),
    )

    loaded_keys = Keys.from_fields(
        specific_keys=["mode=alpha", "mode=beta"],
        indices=np.asarray([0, 1], dtype=np.int64),
        yvalues=np.asarray(["alpha", "beta"], dtype=object),
        spec=panel._keysspec,
    )
    loaded_traces = Traces(
        traces=[
            _make_iv_trace("mode=alpha", 0, "alpha", v_shift_mV=0.1, i_shift_nA=0.2),
            _make_iv_trace("mode=beta", 1, "beta", v_shift_mV=0.3, i_shift_nA=0.4),
        ],
    )

    monkeypatch.setattr(
        gui_measurement_mod,
        "get_keys",
        lambda *, filespec, keysspec: loaded_keys,
    )
    monkeypatch.setattr(
        gui_measurement_mod,
        "get_traces",
        lambda *, filespec, keys, tracespec: loaded_traces,
    )

    panel._on_update_file(SimpleNamespace())

    specific_key_rows = _keys_rows(panel)
    assert list(specific_key_rows["specific_key"]) == ["mode=alpha", "mode=beta"]
    assert list(specific_key_rows["yvalue"]) == ["alpha", "beta"]


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
        offsetspec=offset_spec,
        offsetanalysis=offsets,
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


def test_gui_app_accepts_psd_analysis_preset() -> None:
    traces = _make_traces()
    psd_spec = PSDSpec(detrend=False)
    psds = psd_analysis(traces, spec=psd_spec)

    app = gui_app(
        traces,
        psdspec=psd_spec,
        psdanalysis=psds,
    )

    panel = app._gui_panel

    assert panel._psd_stage_spec is not None
    assert panel._psd_specs_match(panel._psd_stage_spec, psd_spec)
    assert np.allclose(panel.state["psd"]["f_Hz"], psds[0]["f_Hz"])
    assert np.allclose(
        panel.state["psd"]["I_psd_nA2_per_Hz"],
        psds[0]["I_psd_nA2_per_Hz"],
    )


def test_gui_app_accepts_samples_preset() -> None:
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
    sampling_spec = SamplingSpec(
        upsample=6,
        Vbins_mV=np.linspace(-0.25, 0.25, 41, dtype=np.float64),
        Ibins_nA=np.linspace(-3.0, 3.0, 61, dtype=np.float64),
        nu_Hz=13.7,
    )
    smoothing_spec = SmoothingSpec(
        median_bins=5,
        sigma_bins=1.25,
    )
    samples = sample(
        traces,
        samplingspec=sampling_spec,
        smoothingspec=smoothing_spec,
        offsetanalysis=offsets,
        show_progress=False,
    )

    app = gui_app(
        traces,
        offsetspec=offset_spec,
        offsetanalysis=offsets,
        samplingspec=sampling_spec,
        smoothingspec=smoothing_spec,
        samples=samples,
    )

    panel = app._gui_panel

    assert panel._sampling_stage_spec is not None
    assert panel._sampling_specs_match(panel._sampling_stage_spec, sampling_spec)
    assert panel._sampling_stage_smoothing_enabled is True
    assert np.allclose(panel.state["sampling"]["Vbins_mV"], samples[0]["Vbins_mV"])
    assert np.allclose(
        panel.state["sampling"]["I_nA"],
        samples[0]["I_nA"],
        equal_nan=True,
    )


def test_gui_app_accepts_combined_stage_presets() -> None:
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
        nu_Hz=13.7,
    )
    smoothing_spec = SmoothingSpec(
        median_bins=5,
        sigma_bins=1.25,
    )
    psds = psd_analysis(traces, spec=psd_spec)
    offsets = offset_analysis(
        traces,
        spec=offset_spec,
        show_progress=False,
        backend="numpy",
    )
    samples = sample(
        traces,
        samplingspec=sampling_spec,
        smoothingspec=smoothing_spec,
        offsetanalysis=offsets,
        show_progress=False,
    )

    app = gui_app(
        traces,
        psdspec=psd_spec,
        psdanalysis=psds,
        offsetspec=offset_spec,
        offsetanalysis=offsets,
        samplingspec=sampling_spec,
        smoothingspec=smoothing_spec,
        samples=samples,
    )

    panel = app._gui_panel

    assert np.allclose(panel.state["psd"]["f_Hz"], psds[0]["f_Hz"])
    assert panel.state["offset"]["Voff_mV"] == pytest.approx(offsets[0]["Voff_mV"])
    assert np.allclose(
        panel.state["sampling"]["I_nA"],
        samples[0]["I_nA"],
        equal_nan=True,
    )


def test_gui_rejects_invalid_stage_preset_shapes() -> None:
    traces = _make_traces()
    one_trace = Traces(traces=[traces[0]])
    psd_spec = PSDSpec(detrend=False)
    single_psd = psd_analysis(one_trace, spec=psd_spec)[0]
    short_psds = psd_analysis(one_trace, spec=psd_spec)
    sampling_spec = SamplingSpec(
        upsample=10,
        Vbins_mV=np.linspace(-0.5, 0.5, 51, dtype=np.float64),
        Ibins_nA=np.linspace(-5.0, 5.0, 181, dtype=np.float64),
        nu_Hz=13.7,
    )
    short_samples = Samples(
        traces=[
            sample(
                traces[0],
                samplingspec=sampling_spec,
                show_progress=False,
            )
        ]
    )

    with pytest.raises(ValueError, match="bare PSDTrace preset"):
        gui_app(traces, psdanalysis=single_psd)

    with pytest.raises(ValueError, match="PSDTraces preset length must match"):
        gui_app(traces, psdanalysis=short_psds)

    with pytest.raises(ValueError, match="Samples preset length must match"):
        gui_app(traces, samples=short_samples)


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
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())

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
    panel = GUIPanel(_make_traces())

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
    panel = GUIPanel(_make_traces())

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

    panel = GUIPanel(_make_traces())
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

    assert calls == {"psd": 2, "offset": 0, "sampling": 0}


def test_offset_batch_run_updates_progress_and_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())
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


def test_psd_apply_replaces_staged_preset_and_clears_stale_entries() -> None:
    traces = _make_traces()
    psd_spec = PSDSpec(detrend=False)
    psds = psd_analysis(traces, spec=psd_spec)
    panel = GUIPanel(
        traces,
                psdspec=psd_spec,
        psdanalysis=psds,
    )

    experimental_table = panel._experimental_table.value.copy()
    experimental_table.loc[
        experimental_table["key"] == "detrend",
        "value",
    ] = True
    panel._experimental_table.value = experimental_table
    panel._on_experimental_apply(SimpleNamespace())

    assert panel._psd_stage_spec is not None
    assert panel._psd_stage_spec.detrend is True
    assert panel._psd_stage_results[0] is not None
    assert panel._psd_stage_results[1] is None
    assert not np.allclose(
        panel.state["psd"]["I_psd_nA2_per_Hz"],
        psds[0]["I_psd_nA2_per_Hz"],
    )


def test_sampling_apply_replaces_staged_preset_and_clears_stale_entries() -> None:
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
    sampling_spec = SamplingSpec(
        upsample=6,
        Vbins_mV=np.linspace(-0.25, 0.25, 41, dtype=np.float64),
        Ibins_nA=np.linspace(-3.0, 3.0, 61, dtype=np.float64),
        nu_Hz=13.7,
    )
    smoothing_spec = SmoothingSpec(
        median_bins=5,
        sigma_bins=1.25,
    )
    samples = sample(
        traces,
        samplingspec=sampling_spec,
        smoothingspec=smoothing_spec,
        offsetanalysis=offsets,
        show_progress=False,
    )
    panel = GUIPanel(
        traces,
                offsetspec=offset_spec,
        offsetanalysis=offsets,
        samplingspec=sampling_spec,
        smoothingspec=smoothing_spec,
        samples=samples,
    )

    sampling_grid_table = panel._sampling_grid_table.value.copy()
    sampling_grid_table.loc[0, "count"] = int(sampling_grid_table.loc[0, "count"]) - 4
    panel._sampling_grid_table.value = sampling_grid_table
    panel._on_sampling_apply(SimpleNamespace())

    assert panel._sampling_stage_spec is not None
    assert panel._sampling_stage_results[0] is not None
    assert panel._sampling_stage_results[1] is None
    assert panel.state["sampling"]["Vbins_mV"].size == 37


def test_gui_panel_matches_backend_outputs() -> None:
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())

    panel._on_fit_config_edit(SimpleNamespace(row=1, column="value", value=False))
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
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())
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
    panel = GUIPanel(_make_traces())
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
            )

    assert result is expected
