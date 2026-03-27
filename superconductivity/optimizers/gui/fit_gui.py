from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ...gui.common import _import_panel, _mapping_frame
from ...utilities.safety import require_all_finite, require_min_size, to_1d_float64
from ...utilities.types import NDArray64
from ..fit_model import SolutionDict, fit_model
from ..models import MODEL_OPTIONS, ParameterSpec, get_model_spec

_ACTIVE_SERVER = None
_ACTIVE_PANEL: "FitPanel | None" = None
_DEFAULT_MODEL = "pat_sis_int_jax"


def _prepare_trace_matrix(
    values: NDArray64,
    *,
    V_size: int,
    name: str,
) -> NDArray64:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 1:
        matrix = array.reshape(1, -1)
    elif array.ndim == 2:
        if array.shape[-1] == V_size:
            matrix = array.reshape(-1, V_size)
        elif array.shape[0] == V_size:
            matrix = array.T
        else:
            raise ValueError(
                f"{name} must have a trailing axis that matches V_mV."
            )
    else:
        raise ValueError(f"{name} must be one- or two-dimensional.")

    require_min_size(matrix, 3, name)
    require_all_finite(matrix, name)
    if matrix.shape[1] != V_size:
        raise ValueError(f"{name} must have {V_size} samples per trace.")
    return matrix


def _prepare_weight_matrix(
    weights: Optional[NDArray64],
    *,
    data_shape: tuple[int, int],
) -> Optional[NDArray64]:
    if weights is None:
        return None

    traces, points = data_shape
    array = np.asarray(weights, dtype=np.float64)
    if array.ndim == 1:
        if array.shape != (points,):
            raise ValueError("1D weights must match the V_mV axis length.")
        matrix = np.broadcast_to(array, data_shape).copy()
    elif array.ndim == 2:
        if array.shape == data_shape:
            matrix = array.copy()
        elif array.shape == (points, traces):
            matrix = array.T.copy()
        else:
            raise ValueError("2D weights must match I_nA or its transpose.")
    else:
        raise ValueError("weights must be one- or two-dimensional.")

    require_all_finite(matrix, "weights")
    if np.any(matrix < 0.0):
        raise ValueError("weights must be non-negative.")
    return matrix


class FitPanel:
    def __init__(
        self,
        V_mV: NDArray64,
        I_nA: NDArray64,
        *,
        weights: Optional[NDArray64] = None,
        model: str = _DEFAULT_MODEL,
        maxfev: Optional[int] = None,
        on_solution_changed: Optional[Callable[[Optional[SolutionDict]], None]] = None,
    ) -> None:
        self._pn = _import_panel()
        self.V_mV = to_1d_float64(V_mV, "V_mV")
        require_all_finite(self.V_mV, "V_mV")
        require_min_size(self.V_mV, 3, "V_mV")

        self._trace_matrix = _prepare_trace_matrix(I_nA, V_size=self.V_mV.size, name="I_nA")
        self._weights_matrix = _prepare_weight_matrix(
            weights,
            data_shape=self._trace_matrix.shape,
        )
        self._slice_index = 0
        self.maxfev = maxfev
        self.model_key = model
        self._on_solution_changed = on_solution_changed
        self._solution: Optional[SolutionDict] = None
        self._fit_running = False
        self._fit_started_at: float | None = None
        self._fit_timer = None
        self._fit_future = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._last_error = ""
        self._last_elapsed_s = 0.0
        self._last_status = "idle"

        self._parameters = self._default_parameters()
        self._initial_curve = self._compute_initial_curve()
        self._fit_curve = self._initial_curve.copy()

        self._model_selector = self._pn.widgets.Select(
            name="Model",
            options=MODEL_OPTIONS,
            value=self.model_key,
            sizing_mode="stretch_width",
        )
        self._model_selector.param.watch(self._on_model_changed, "value")

        self._slice_selector = self._pn.widgets.IntSlider(
            name="Slice",
            start=0,
            end=max(0, self._trace_matrix.shape[0] - 1),
            value=0,
            step=1,
            disabled=self._trace_matrix.shape[0] == 1,
            visible=self._trace_matrix.shape[0] > 1,
            sizing_mode="stretch_width",
        )
        self._slice_selector.param.watch(self._on_slice_changed, "value")

        self._fit_button = self._pn.widgets.Button(name="Fit", button_type="primary")
        self._fit_button.on_click(self._on_fit_clicked)
        self._spinner = self._pn.indicators.LoadingSpinner(
            value=False,
            width=20,
            height=20,
        )
        self._fit_state = self._pn.pane.Markdown("Idle", sizing_mode="stretch_width")

        self._parameter_table = self._pn.widgets.Tabulator(
            self._parameter_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=260,
            editors={
                "name": None,
                "label": None,
                "guess": {"type": "number"},
                "lower": {"type": "number"},
                "upper": {"type": "number"},
                "fixed": {"type": "tickCross"},
                "fit": None,
            },
            formatters={
                "label": {"type": "html"},
                "fixed": {"type": "tickCross"},
            },
        )
        self._parameter_table.on_edit(self._on_parameter_edit)

        self._model_info_table = self._pn.widgets.Tabulator(
            _mapping_frame(self._model_info()),
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=220,
        )
        self._optimizer_info_table = self._pn.widgets.Tabulator(
            _mapping_frame(self._optimizer_info()),
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=220,
        )
        self._model_html = self._pn.pane.LaTeX(
            get_model_spec(self.model_key).html,
            sizing_mode="stretch_width",
        )

        self._iv_figure = self._build_iv_figure()
        self._didv_figure = self._build_didv_figure()
        self._iv_pane = self._pn.pane.Plotly(
            self._iv_figure,
            sizing_mode="stretch_width",
            height=340,
            config={"responsive": True},
        )
        self._didv_pane = self._pn.pane.Plotly(
            self._didv_figure,
            sizing_mode="stretch_width",
            height=340,
            config={"responsive": True},
        )

        self.layout = self.get_layout()

    @property
    def solution(self) -> Optional[SolutionDict]:
        return self._solution

    def get_layout(self):
        return self._pn.Row(
            self._pn.Column(
                self._iv_pane,
                self._didv_pane,
                sizing_mode="stretch_width",
            ),
            self._pn.Column(
                self._pn.Row(
                    self._model_selector,
                    self._fit_button,
                    self._spinner,
                    sizing_mode="stretch_width",
                ),
                self._fit_state,
                self._slice_selector,
                self._pn.pane.Markdown("### Parameters"),
                self._parameter_table,
                self._pn.pane.Markdown("### Model"),
                self._model_info_table,
                self._pn.pane.Markdown("### Equation"),
                self._model_html,
                self._pn.pane.Markdown("### Optimizer"),
                self._optimizer_info_table,
                min_width=560,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def _default_parameters(self) -> list[ParameterSpec]:
        return [replace(parameter) for parameter in get_model_spec(self.model_key).parameters]

    def _active_trace(self) -> NDArray64:
        return np.asarray(self._trace_matrix[self._slice_index], dtype=np.float64)

    def _active_weights(self) -> Optional[NDArray64]:
        if self._weights_matrix is None:
            return None
        return np.asarray(self._weights_matrix[self._slice_index], dtype=np.float64)

    def _current_guess(self) -> NDArray64:
        return np.array([parameter.guess for parameter in self._parameters], dtype=np.float64)

    def _compute_initial_curve(self) -> NDArray64:
        function = get_model_spec(self.model_key).function
        return np.asarray(function(self.V_mV, *self._current_guess()), dtype=np.float64)

    @staticmethod
    def _gradient(V_mV: NDArray64, values: NDArray64) -> NDArray64:
        return np.gradient(np.asarray(values, dtype=np.float64), np.asarray(V_mV, dtype=np.float64))

    def _build_iv_figure(self) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._active_trace(),
                mode="markers",
                marker={"size": 4, "opacity": 0.6, "color": "#111111"},
                name="Data",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._initial_curve,
                mode="lines",
                line={"dash": "dash", "color": "#777777"},
                name="Initial",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._fit_curve,
                mode="lines",
                line={"color": "#005ea8"},
                name="Fit",
            )
        )
        figure.update_layout(
            margin={"l": 80, "r": 20, "t": 20, "b": 20},
            yaxis_title="I (nA)",
            showlegend=True,
            legend={"orientation": "h", "y": 1.05},
        )
        figure.update_xaxes(showticklabels=False)
        return figure

    def _build_didv_figure(self) -> go.Figure:
        figure = go.Figure()
        figure.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._gradient(self.V_mV, self._active_trace()),
                mode="markers",
                marker={"size": 4, "opacity": 0.6, "color": "#111111"},
                name="Data",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._gradient(self.V_mV, self._initial_curve),
                mode="lines",
                line={"dash": "dash", "color": "#777777"},
                name="Initial",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._gradient(self.V_mV, self._fit_curve),
                mode="lines",
                line={"color": "#005ea8"},
                name="Fit",
            )
        )
        figure.update_layout(
            margin={"l": 80, "r": 20, "t": 20, "b": 40},
            xaxis_title="V (mV)",
            yaxis_title="dI/dV",
            showlegend=False,
        )
        return figure

    def _parameter_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        result_params = self._solution["params"] if self._solution else None
        for index, parameter in enumerate(self._parameters):
            fit_text = self._format_fit_text(
                None if result_params is None else result_params[index].value,
                None if result_params is None else result_params[index].error,
            )
            rows.append(
                {
                    "name": parameter.name,
                    "label": parameter.label,
                    "guess": float(parameter.guess),
                    "lower": float(parameter.lower),
                    "upper": float(parameter.upper),
                    "fixed": bool(parameter.fixed),
                    "fit": fit_text,
                }
            )
        return pd.DataFrame(rows)

    def _model_info(self) -> OrderedDict[str, object]:
        model_spec = get_model_spec(self.model_key)
        info = OrderedDict()
        info["model"] = model_spec.key
        info["label"] = model_spec.label
        info.update(model_spec.info)
        return info

    def _optimizer_info(self) -> OrderedDict[str, object]:
        free_parameters = sum(not parameter.fixed for parameter in self._parameters)
        chi2, reduced_chi2 = self._chi2_metrics()
        info = OrderedDict()
        info["solver"] = "scipy.optimize.curve_fit"
        info["maxfev"] = self.maxfev
        info["active_slice"] = self._slice_index
        info["free_parameters"] = free_parameters
        info["chi2"] = f"{chi2:.6g}" if np.isfinite(chi2) else "n/a"
        info["reduced_chi2"] = (
            f"{reduced_chi2:.6g}" if np.isfinite(reduced_chi2) else "n/a"
        )
        info["elapsed_s"] = f"{self._last_elapsed_s:.3f}"
        info["status"] = self._last_status
        info["last_error"] = self._last_error or ""
        return info

    def _chi2_metrics(self) -> tuple[float, float]:
        if self._solution is None:
            return np.nan, np.nan

        residual = self._solution["I_exp_nA"] - self._solution["I_fit_nA"]
        weights = self._solution["weights"]
        if weights is None:
            chi2 = float(np.sum(residual * residual))
            effective_points = residual.size
        else:
            mask = weights > 0.0
            if not np.any(mask):
                return np.nan, np.nan
            chi2 = float(np.sum(weights[mask] * residual[mask] * residual[mask]))
            effective_points = int(np.sum(mask))

        dof = effective_points - sum(not parameter.fixed for parameter in self._parameters)
        reduced = chi2 / dof if dof > 0 else np.nan
        return chi2, reduced

    def _refresh_tables(self) -> None:
        self._parameter_table.value = self._parameter_frame()
        self._model_info_table.value = _mapping_frame(self._model_info())
        self._optimizer_info_table.value = _mapping_frame(self._optimizer_info())
        self._model_html.object = get_model_spec(self.model_key).html

    def _refresh_plots(self) -> None:
        active_trace = self._active_trace()
        active_didv = self._gradient(self.V_mV, active_trace)
        initial_didv = self._gradient(self.V_mV, self._initial_curve)
        fit_didv = self._gradient(self.V_mV, self._fit_curve)

        self._iv_figure.data[0].y = active_trace
        self._iv_figure.data[1].y = self._initial_curve
        self._iv_figure.data[2].y = self._fit_curve
        self._didv_figure.data[0].y = active_didv
        self._didv_figure.data[1].y = initial_didv
        self._didv_figure.data[2].y = fit_didv
        self._iv_pane.object = self._iv_figure
        self._didv_pane.object = self._didv_figure

    def _clear_solution(self) -> None:
        self._solution = None
        if self._on_solution_changed is not None:
            self._on_solution_changed(None)

    def _set_solution(self, solution: Optional[SolutionDict]) -> None:
        self._solution = solution
        if self._on_solution_changed is not None:
            self._on_solution_changed(solution)

    def _format_fit_text(
        self,
        value: float | None,
        error: float | None,
    ) -> str:
        if value is None:
            return "n/a"
        if error is None or not np.isfinite(error) or error <= 0.0:
            return f"{float(value):.6g}"
        return f"{float(value):.6g} ± {float(error):.2g}"

    def _recompute_initial_curve(self) -> None:
        self._initial_curve = self._compute_initial_curve()
        self._refresh_plots()
        self._refresh_tables()

    def _on_parameter_edit(self, event: object) -> None:
        column = str(getattr(event, "column", ""))
        row = int(getattr(event, "row"))
        value = getattr(event, "value")
        parameter = self._parameters[row]

        if column == "guess":
            parameter.guess = float(np.clip(float(value), parameter.lower, parameter.upper))
            self._recompute_initial_curve()
            return

        if column == "lower":
            parameter.lower = float(value)
            if parameter.lower > parameter.upper:
                parameter.upper = parameter.lower
            parameter.guess = float(np.clip(parameter.guess, parameter.lower, parameter.upper))
            self._recompute_initial_curve()
            return

        if column == "upper":
            parameter.upper = float(value)
            if parameter.upper < parameter.lower:
                parameter.lower = parameter.upper
            parameter.guess = float(np.clip(parameter.guess, parameter.lower, parameter.upper))
            self._recompute_initial_curve()
            return

        if column == "fixed":
            parameter.fixed = bool(value)
            self._refresh_tables()

    def _on_model_changed(self, event: object) -> None:
        new_value = str(getattr(event, "new"))
        old_value = str(getattr(event, "old"))
        if new_value == old_value:
            return
        if self._fit_running:
            self._model_selector.value = old_value
            return

        self.model_key = new_value
        self._parameters = self._default_parameters()
        self._clear_solution()
        self._fit_curve = self._initial_curve = self._compute_initial_curve()
        self._last_error = ""
        self._last_elapsed_s = 0.0
        self._last_status = "idle"
        self._refresh_plots()
        self._refresh_tables()

    def _on_slice_changed(self, event: object) -> None:
        new_value = int(getattr(event, "new"))
        old_value = int(getattr(event, "old"))
        if new_value == old_value:
            return
        self._slice_index = new_value
        self._clear_solution()
        self._fit_curve = self._initial_curve = self._compute_initial_curve()
        self._last_error = ""
        self._last_elapsed_s = 0.0
        self._last_status = "idle"
        self._refresh_plots()
        self._refresh_tables()

    def _on_fit_clicked(self, _: object) -> None:
        if self._fit_running:
            return

        self._fit_running = True
        self._fit_started_at = time.perf_counter()
        self._spinner.value = True
        self._fit_button.disabled = True
        self._last_status = "running"
        self._last_error = ""
        self._fit_state.object = "Fit running..."
        self._refresh_tables()
        self._start_fit_timer()

        self._fit_future = self._executor.submit(
            fit_model,
            self.V_mV,
            self._active_trace(),
            model=self.model_key,
            parameters=self._parameters,
            weights=self._active_weights(),
            maxfev=self.maxfev,
        )
        self._fit_future.add_done_callback(self._on_fit_finished)

    def _on_fit_finished(self, future: object) -> None:
        def finalize() -> None:
            if self._fit_started_at is not None:
                self._last_elapsed_s = time.perf_counter() - self._fit_started_at
            try:
                solution = future.result()
            except Exception as exc:
                self._clear_solution()
                self._fit_curve = self._initial_curve.copy()
                self._last_status = "failed"
                self._last_error = f"{type(exc).__name__}: {exc}"
                self._fit_state.object = f"Fit failed: `{self._last_error}`"
            else:
                self._set_solution(solution)
                self._fit_curve = np.asarray(solution["I_fit_nA"], dtype=np.float64)
                self._last_status = "done"
                self._last_error = ""
                self._fit_state.object = (
                    f"Fit finished in {self._last_elapsed_s:.3f} s"
                )
            finally:
                self._fit_running = False
                self._fit_started_at = None
                self._spinner.value = False
                self._fit_button.disabled = False
                if self._fit_timer is not None:
                    self._fit_timer.stop()
                    self._fit_timer = None
                self._refresh_plots()
                self._refresh_tables()

        self._pn.state.execute(finalize)

    def _start_fit_timer(self) -> None:
        if self._fit_timer is not None:
            self._fit_timer.stop()
        self._fit_timer = self._pn.state.add_periodic_callback(
            self._update_fit_timer,
            period=200,
            start=True,
        )

    def _update_fit_timer(self) -> None:
        if not self._fit_running or self._fit_started_at is None:
            if self._fit_timer is not None:
                self._fit_timer.stop()
                self._fit_timer = None
            return
        elapsed_s = time.perf_counter() - self._fit_started_at
        self._fit_state.object = f"Fit running for {elapsed_s:.1f} s..."


def fit_gui_app(
    V_mV: NDArray64,
    I_nA: NDArray64,
    *,
    weights: Optional[NDArray64] = None,
    model: str = _DEFAULT_MODEL,
    maxfev: Optional[int] = None,
    on_solution_changed: Optional[Callable[[Optional[SolutionDict]], None]] = None,
):
    panel = FitPanel(
        V_mV,
        I_nA,
        weights=weights,
        model=model,
        maxfev=maxfev,
        on_solution_changed=on_solution_changed,
    )
    layout = panel.layout
    setattr(layout, "_fit_gui_panel", panel)
    return layout


def serve_fit_gui(
    V_mV: NDArray64,
    I_nA: NDArray64,
    *,
    weights: Optional[NDArray64] = None,
    model: str = _DEFAULT_MODEL,
    maxfev: Optional[int] = None,
    port: int = 0,
    open_browser: bool = True,
    threaded: bool = True,
    title: str = "Optimizer GUI",
    verbose: bool = True,
    stop_existing: bool = True,
):
    global _ACTIVE_PANEL, _ACTIVE_SERVER

    pn = _import_panel()
    if stop_existing and _ACTIVE_SERVER is not None and hasattr(_ACTIVE_SERVER, "stop"):
        _ACTIVE_SERVER.stop()
        _ACTIVE_SERVER = None
        _ACTIVE_PANEL = None

    app = fit_gui_app(
        V_mV,
        I_nA,
        weights=weights,
        model=model,
        maxfev=maxfev,
    )
    _ACTIVE_PANEL = getattr(app, "_fit_gui_panel")
    _ACTIVE_SERVER = pn.serve(
        app,
        port=port,
        show=open_browser,
        threaded=threaded,
        title=title,
        verbose=verbose,
    )
    return _ACTIVE_SERVER


def run_fit_gui(
    V_mV: NDArray64,
    I_nA: NDArray64,
    *,
    weights: Optional[NDArray64] = None,
    model: str = _DEFAULT_MODEL,
    maxfev: Optional[int] = None,
    port: int = 0,
    open_browser: bool = True,
    threaded: bool = True,
    title: str = "Optimizer GUI",
    verbose: bool = True,
    stop_existing: bool = True,
    wait: bool = True,
    poll_interval: float = 0.5,
) -> Optional[SolutionDict]:
    global _ACTIVE_PANEL

    server = serve_fit_gui(
        V_mV,
        I_nA,
        weights=weights,
        model=model,
        maxfev=maxfev,
        port=port,
        open_browser=open_browser,
        threaded=threaded,
        title=title,
        verbose=verbose,
        stop_existing=stop_existing,
    )
    if not wait:
        return None

    try:
        while getattr(server, "is_alive", lambda: False)():
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        if hasattr(server, "stop"):
            server.stop()
    return None if _ACTIVE_PANEL is None else _ACTIVE_PANEL.solution


def fit_gui(
    V_mV: NDArray64,
    I_nA: NDArray64,
    *,
    weights: Optional[NDArray64] = None,
    model: str = _DEFAULT_MODEL,
    maxfev: Optional[int] = None,
    port: int = 0,
    open_browser: bool = True,
    threaded: bool = True,
    title: str = "Optimizer GUI",
    verbose: bool = True,
    stop_existing: bool = True,
    wait: bool = True,
    poll_interval: float = 0.5,
) -> Optional[SolutionDict]:
    if wait:
        return run_fit_gui(
            V_mV,
            I_nA,
            weights=weights,
            model=model,
            maxfev=maxfev,
            port=port,
            open_browser=open_browser,
            threaded=threaded,
            title=title,
            verbose=verbose,
            stop_existing=stop_existing,
            wait=True,
            poll_interval=poll_interval,
        )

    serve_fit_gui(
        V_mV,
        I_nA,
        weights=weights,
        model=model,
        maxfev=maxfev,
        port=port,
        open_browser=open_browser,
        threaded=threaded,
        title=title,
        verbose=verbose,
        stop_existing=stop_existing,
    )
    return None


__all__ = [
    "FitPanel",
    "fit_gui",
    "fit_gui_app",
    "run_fit_gui",
    "serve_fit_gui",
]
