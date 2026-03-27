from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Callable, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from numpy.typing import NDArray

from superconductivity.style.cpd5 import schwarz, seeblau100, seegrau100
from superconductivity.style.plotly import mpl_color_to_plotly
from superconductivity.utilities.safety import (
    require_all_finite,
    require_min_size,
    to_1d_float64,
)
from superconductivity.utilities.types import NDArray64

from .fit_pat import (
    DEFAULT_PARAMETERS,
    PARAMETER_NAMES,
    ParameterSpec,
    SolutionDict,
    fit_pat,
)
from .models import get_model

try:
    from panel.util import _NoValue
except (ImportError, ModuleNotFoundError):
    _NoValue = object()

schwarz = mpl_color_to_plotly(schwarz)
seeblau100 = mpl_color_to_plotly(seeblau100)
seegrau100 = mpl_color_to_plotly(seegrau100)


def _pat_trace(V_mV: NDArray64, params: NDArray64, *, model: str) -> NDArray64:
    function, parameter_mask = get_model(model=model)
    return function(V_mV, *params[parameter_mask])


def _import_panel() -> "panel":
    try:
        import panel as pn
    except ImportError as exc:
        raise ImportError(
            "Panel must be installed to build the PAT fit widget. "
            "Install panel and rerun the code."
        ) from exc

    try:
        pn.extension("plotly", "mathjax")
    except Exception as exc:
        raise RuntimeError("Failed to initialize Panel Plotly extension.") from exc

    return pn


class PatFitPanel:
    """Interactive Panel widget for PAT data and optimizer control.

    Parameters
    ----------
    V_mV : NDArray64
        Voltage axis of the experimental trace in millivolts.
    I_nA : NDArray64
        Current axis of the experimental trace in nanoamperes.
    weights : NDArray64 | None, optional
        Optional per-point reliability weights for the optimizer. Defaults to None.
    maxfev : int | None, optional
        Upper limit for the optimizer iterations.
    model : str, optional
        Active fit model. Supported GUI options are ``"pat"`` and
        ``"conv_pat"``.
    on_solution_changed : callable | None, optional
        Callback invoked whenever the active fit solution is updated or cleared.

    Notes
    -----
    The widget exposes :meth:`build_plots`, :meth:`build_control_panel`, and
    :meth:`update_solution` so that atomic-contact or other fit pipelines can extend
    the same layout with additional traces or metadata.
    """

    def __init__(
        self,
        V_mV: NDArray64,
        I_nA: NDArray64,
        *,
        weights: Optional[NDArray64] = None,
        maxfev: Optional[int] = None,
        model: str = "pat",
        on_solution_changed: Optional[Callable[[Optional[SolutionDict]], None]] = None,
    ) -> None:
        self._pn = _import_panel()
        self.V_mV: NDArray64 = to_1d_float64(V_mV, "V_mV")
        require_all_finite(self.V_mV, "V_mV")
        require_min_size(self.V_mV, 3, "V_mV")

        self._trace_matrix: NDArray64 = self._prepare_trace_matrix(I_nA)
        self._trace_index = 0
        self._trace_selector = self._pn.widgets.IntSlider(
            name="Trace index",
            start=0,
            end=self._trace_matrix.shape[0] - 1,
            value=0,
            step=1,
            sizing_mode="stretch_width",
        )
        self._trace_selector.param.watch(self._on_trace_selected, "value")
        self._trace_header = self._pn.pane.Markdown(
            self._format_trace_header(),
            sizing_mode="stretch_width",
        )
        self.I_nA: NDArray64 = self._trace_matrix[0]
        self.weights: Optional[NDArray64] = weights
        self.maxfev = maxfev
        self.model = model
        self._model_selector_sync = False
        self._model_selector = self._pn.widgets.Select(
            name="Model",
            options={
                "Integral PAT": "pat",
                "Convolution PAT": "conv_pat",
            },
            value=self.model,
            width=180,
        )
        self._model_selector.param.watch(self._on_model_selected, "value")
        self._on_solution_changed = on_solution_changed

        self._parameter_templates = DEFAULT_PARAMETERS
        self._display_keys = list(PARAMETER_NAMES)
        self._sliders: OrderedDict[str, "pn.widgets.FloatSlider"] = OrderedDict()
        self._lock_checkboxes: OrderedDict[str, "pn.widgets.Checkbox"] = OrderedDict()
        self._slider_rows: list["pn.Row"] = []
        self._build_sliders()

        self._initial_curve = self._compute_curve_from_guess()
        self._fit_curve = self._initial_curve.copy()
        self._data_derivative = self._gradient(self.I_nA)
        self._initial_derivative = self._gradient(self._initial_curve)
        self._fit_derivative = self._gradient(self._fit_curve)

        self._iv_figure = self._create_iv_figure()
        self._derivative_figure = self._create_derivative_figure()
        self._fit_button = self._pn.widgets.Button(name="Fit", button_type="primary")
        self._fit_running = False
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._running_indicator = self._pn.indicators.LoadingSpinner(
            value=False,
            width=20,
            height=20,
        )
        self._fit_state_text = self._pn.pane.Markdown(
            "Idle",
            sizing_mode="stretch_width",
        )
        self._fit_started_at: float | None = None
        self._fit_elapsed_text = self._pn.pane.Markdown(
            "Elapsed: 0.0 s",
            sizing_mode="stretch_width",
        )
        self._fit_button.on_click(self._on_fit_click)
        self._fit_future = None
        self._fit_timer = None
        self._syncing_xrange = False

        self._solution: Optional[SolutionDict] = None

        self._parameter_table = self._build_parameter_table_widget()
        self._status_panel = self._pn.pane.Markdown(self._format_status_text())
        self._model_panel = self._pn.pane.LaTeX(
            self._format_model_text(),
            sizing_mode="stretch_width",
            styles={"padding": "0.5rem"},
        )

        self._solution: Optional[SolutionDict] = None

        self._iv_pane = self._pn.pane.Plotly(
            self._iv_figure,
            sizing_mode="stretch_width",
            height=350,
            config={"responsive": True},
            viewport_update_policy="continuous",
            viewport_update_throttle=100,
        )
        self._derivative_pane = self._pn.pane.Plotly(
            self._derivative_figure,
            sizing_mode="stretch_width",
            height=350,
            config={"responsive": True},
            viewport_update_policy="continuous",
            viewport_update_throttle=100,
        )
        self._iv_pane.param.watch(self._on_iv_viewport_changed, "viewport")
        self._derivative_pane.param.watch(
            self._on_derivative_viewport_changed,
            "viewport",
        )
        self._update_plot_traces()

        self.layout = self.get_layout()

    def build_plots(self) -> "pn.Column":
        """Return the column containing the IV and dI/dV plots."""
        return self._pn.Column(
            self._iv_pane,
            self._derivative_pane,
            sizing_mode="stretch_width",
            height=350,
        )

    def build_control_panel(self) -> "pn.Column":
        """Return the control column for sliders, button, and readouts."""
        return self._pn.Column(
            self._pn.pane.Markdown("### PAT controls"),
            self._pn.Row(
                self._model_selector,
                self._fit_button,
                self._running_indicator,
                self._fit_state_text,
                self._fit_elapsed_text,
                sizing_mode="stretch_width",
            ),
            self._parameter_table,
            self._status_panel,
            self._pn.Spacer(height=10),
            self._trace_header,
            self._trace_selector,
            self._pn.Spacer(height=15),
            self._pn.pane.Markdown("### Model"),
            self._model_panel,
            sizing_mode="stretch_width",
        )

    def get_layout(self) -> "pn.Row":
        """Return the composed Row layout for embedding."""
        return self._pn.Row(
            self.build_plots(),
            self.build_control_panel(),
            sizing_mode="stretch_width",
        )

    def update_solution(self, solution: SolutionDict) -> None:
        """Update the UI with the latest solution from :func:`fit_pat`."""
        self._set_solution(solution)
        self._fit_curve = solution["I_fit_nA"].copy()
        self._fit_derivative = self._gradient(self._fit_curve)
        self._iv_figure.data[2].y = self._fit_curve
        self._derivative_figure.data[2].y = self._fit_derivative
        self._iv_pane.object = self._iv_figure
        self._derivative_pane.object = self._derivative_figure
        self._refresh_parameter_table()
        self._status_panel.object = self._format_status_text()

    def _build_sliders(self) -> None:
        for spec in self._parameter_templates:
            step = max((spec.upper - spec.lower) / 200.0, 1e-6)
            slider = self._pn.widgets.FloatSlider(
                name=f"{spec.name} ({spec.label})",
                start=spec.lower,
                end=spec.upper,
                value=spec.guess,
                step=step,
                sizing_mode="stretch_width",
            )
            slider.param.watch(self._on_slider_changed, "value")

            checkbox = self._pn.widgets.Checkbox(
                name=f"Fix {spec.name}",
                value=bool(spec.fixed),
            )
            checkbox.param.watch(partial(self._on_lock_toggled, key=spec.name), "value")
            slider.disabled = checkbox.value

            self._slider_rows.append(
                self._pn.Row(slider, checkbox, sizing_mode="stretch_width")
            )
            self._sliders[spec.name] = slider
            self._lock_checkboxes[spec.name] = checkbox

    def _create_iv_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self.I_nA,
                name="Data",
                mode="markers",
                marker=dict(size=4, opacity=0.6, color=schwarz),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._initial_curve,
                name="Initial",
                line=dict(dash="dash", width=2, color=seegrau100),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._fit_curve,
                name="Fit",
                line=dict(width=2, color=seeblau100),
            )
        )
        fig.update_layout(
            yaxis_title="<i>I</i> (nA)",
            showlegend=False,
            margin=dict(l=85, r=20, t=30, b=10),
        )
        fig.update_xaxes(showticklabels=False, title_text=None)
        fig.update_yaxes(automargin=False, title_standoff=8)
        return fig

    def _create_derivative_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._data_derivative,
                name="Data",
                mode="markers",
                marker=dict(size=4, opacity=0.6, color=schwarz),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._initial_derivative,
                name="Initial",
                line=dict(dash="dash", width=2, color=seegrau100),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._fit_derivative,
                name="Fit",
                line=dict(width=2, color=seeblau100),
            )
        )
        fig.update_layout(
            xaxis_title="<i>V</i> (mV)",
            yaxis_title="d<i>I</i>/d<i>V</i> (<i>G<sub>0</sub></i>)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=85, r=20, t=30, b=30),
        )
        fig.update_yaxes(automargin=False, title_standoff=8)
        return fig

    def _on_slider_changed(self, event: "pn.parameterized.Event") -> None:
        if event.new == event.old:
            return
        self._initial_curve = self._compute_curve_from_guess()
        self._initial_derivative = self._gradient(self._initial_curve)
        self._iv_figure.data[1].y = self._initial_curve
        self._derivative_figure.data[1].y = self._initial_derivative
        self._update_plot_traces()
        self._refresh_parameter_table()

    def _on_trace_selected(self, event: "pn.parameterized.Event") -> None:
        if event.new == event.old:
            return
        self._trace_index = int(event.new)
        self.I_nA = self._trace_matrix[self._trace_index]
        self._set_solution(None)
        self._trace_header.object = self._format_trace_header()
        self._initial_curve = self._compute_curve_from_guess()
        self._fit_curve = self._initial_curve.copy()
        self._data_derivative = self._gradient(self.I_nA)
        self._initial_derivative = self._gradient(self._initial_curve)
        self._fit_derivative = self._gradient(self._fit_curve)
        self._update_plot_traces()
        self._refresh_parameter_table()
        if self._fit_timer is not None:
            self._fit_timer.stop()
            self._fit_timer = None
        self._fit_state_text.object = "Idle"
        self._fit_elapsed_text.object = "Elapsed: 0.0 s"
        self._status_panel.object = self._format_status_text()

    def _on_model_selected(self, event: "pn.parameterized.Event") -> None:
        if event.new == event.old or self._model_selector_sync:
            return
        if self._fit_running:
            self._model_selector_sync = True
            try:
                self._model_selector.value = event.old
            finally:
                self._model_selector_sync = False
            return

        self.model = str(event.new)
        self._set_solution(None)
        self._initial_curve = self._compute_curve_from_guess()
        self._fit_curve = self._initial_curve.copy()
        self._initial_derivative = self._gradient(self._initial_curve)
        self._fit_derivative = self._gradient(self._fit_curve)
        self._update_plot_traces()
        self._refresh_parameter_table()
        self._model_panel.object = self._format_model_text()
        self._status_panel.object = self._format_status_text()

    def _on_lock_toggled(self, event: "pn.parameterized.Event", *, key: str) -> None:
        slider = self._sliders[key]
        slider.disabled = bool(event.new)
        self._refresh_parameter_table()

    def _build_parameter_table_widget(self) -> "pn.widgets.Tabulator":
        table = self._pn.widgets.Tabulator(
            self._parameter_table_frame(),
            show_index=False,
            disabled=False,
            editors={
                "Label": None,
                "Guess": {"type": "number"},
                "Lower": {"type": "number"},
                "Upper": {"type": "number"},
                "Fixed": {"type": "tickCross"},
                "Fit": None,
            },
            formatters={
                "Label": {"type": "html"},
                "Fixed": {"type": "tickCross"},
            },
            widths={
                "Label": 80,
                "Guess": 100,
                "Lower": 100,
                "Upper": 100,
                "Fixed": 80,
                "Fit": 200,
            },
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=220,
        )
        table.on_edit(self._on_parameter_table_edit)
        return table

    def _parameter_table_frame(self) -> pd.DataFrame:
        guess = self._current_guess()
        results = self._solution["params"] if self._solution else None
        rows: list[dict[str, object]] = []
        for idx, display in enumerate(self._display_keys):
            spec = self._parameter_templates[idx]
            if results is None:
                fit_text = self._format_fit_value_with_error(guess[idx], None)
            else:
                fit_text = self._format_fit_value_with_error(
                    results[idx].value,
                    results[idx].error,
                )
            rows.append(
                {
                    "Label": spec.label,
                    "Guess": float(guess[idx]),
                    "Lower": float(spec.lower),
                    "Upper": float(spec.upper),
                    "Fixed": bool(self._lock_checkboxes[display].value),
                    "Fit": fit_text,
                }
            )
        return pd.DataFrame(rows)

    def _format_fit_value_with_error(
        self,
        value: float | None,
        error: float | None,
    ) -> str:
        if value is None:
            return "n/a"

        value_f = float(value)
        if error is None or not np.isfinite(error) or float(error) <= 0.0:
            return f"{value_f:.6g}"

        error_f = float(error)
        exponent = int(np.floor(np.log10(error_f)))
        decimals = max(0, 1 - exponent)
        rounded_error = round(error_f, decimals)
        rounded_value = round(value_f, decimals)

        if rounded_error >= 10 ** (2 - decimals):
            decimals = max(0, decimals - 1)
            rounded_error = round(error_f, decimals)
            rounded_value = round(value_f, decimals)

        if decimals > 0:
            error_digits = int(round(rounded_error * (10**decimals)))
            value_text = f"{rounded_value:.{decimals}f}"
        else:
            error_digits = int(round(rounded_error))
            value_text = f"{rounded_value:.0f}"

        return f"{value_text}({error_digits})"

    def _refresh_parameter_table(self) -> None:
        self._parameter_table.value = self._parameter_table_frame()

    def _on_parameter_table_edit(self, event: object) -> None:
        column = getattr(event, "column", None)
        row = int(event.row)
        key = self._display_keys[row]

        if column == "Guess":
            slider = self._sliders[key]
            value = float(event.value)
            clipped = min(max(value, slider.start), slider.end)
            slider.value = clipped
            if clipped != value:
                self._refresh_parameter_table()
            return

        if column == "Fixed":
            checkbox = self._lock_checkboxes[key]
            checkbox.value = bool(event.value)
            self._refresh_parameter_table()
            return

        if column in {"Lower", "Upper"}:
            slider = self._sliders[key]
            spec_index = self._display_keys.index(key)
            spec = self._parameter_templates[spec_index]
            value = float(event.value)

            if column == "Lower":
                new_lower = min(value, spec.upper)
                self._parameter_templates[spec_index] = ParameterSpec(
                    name=spec.name,
                    label=spec.label,
                    lower=float(new_lower),
                    upper=float(spec.upper),
                    guess=float(slider.value),
                    fixed=bool(self._lock_checkboxes[key].value),
                )
                slider.start = float(new_lower)
                if slider.value < slider.start:
                    slider.value = slider.start
                self._refresh_parameter_table()
                return

            new_upper = max(value, spec.lower)
            self._parameter_templates[spec_index] = ParameterSpec(
                name=spec.name,
                label=spec.label,
                lower=float(spec.lower),
                upper=float(new_upper),
                guess=float(slider.value),
                fixed=bool(self._lock_checkboxes[key].value),
            )
            slider.end = float(new_upper)
            if slider.value > slider.end:
                slider.value = slider.end
            self._refresh_parameter_table()
            return

    def _on_fit_click(self, _: object) -> None:
        if self._fit_running:
            return

        self._fit_running = True
        self._fit_started_at = time.perf_counter()
        self._fit_button.disabled = True
        self._running_indicator.value = True
        self._fit_state_text.object = "Fit running for 0.0 s ..."
        self._fit_elapsed_text.object = "Elapsed: 0.0 s"
        self._status_panel.object = "Fit running ..."
        self._start_fit_timer()

        parameters = self._current_parameters()
        self._fit_future = self._executor.submit(
            fit_pat,
            self.V_mV,
            self.I_nA,
            parameters=parameters,
            weights=self.weights,
            maxfev=self.maxfev,
            model=self.model,
        )
        self._fit_future.add_done_callback(self._on_fit_finished)

    def _on_fit_finished(self, future: object) -> None:
        def finalize() -> None:
            elapsed_s = 0.0
            if self._fit_started_at is not None:
                elapsed_s = time.perf_counter() - self._fit_started_at
            try:
                solution = future.result()
            except Exception as exc:
                self._set_solution(None)
                self._status_panel.object = (
                    "Fit failed with error: " f"`{type(exc).__name__}: {exc}`"
                )
            else:
                self.update_solution(solution)
            finally:
                self._fit_running = False
                if self._fit_timer is not None:
                    self._fit_timer.stop()
                    self._fit_timer = None
                self._fit_button.disabled = False
                self._running_indicator.value = False
                self._fit_state_text.object = "Idle"
                self._fit_elapsed_text.object = f"Elapsed: {elapsed_s:.1f} s"
                self._fit_started_at = None

        self._pn.state.execute(finalize)

    def _start_fit_timer(self) -> None:
        if self._fit_timer is not None:
            self._fit_timer.stop()
            self._fit_timer = None

        self._fit_timer = self._pn.state.add_periodic_callback(
            self._update_fit_elapsed,
            period=200,
            start=True,
        )

    def _update_fit_elapsed(self) -> None:
        if not self._fit_running or self._fit_started_at is None:
            if self._fit_timer is not None:
                self._fit_timer.stop()
                self._fit_timer = None
            return

        elapsed_s = time.perf_counter() - self._fit_started_at
        self._fit_state_text.object = f"Fit running for {elapsed_s:.1f} s..."

    def _on_iv_viewport_changed(self, event: "pn.parameterized.Event") -> None:
        self._sync_xrange_from_viewport(
            viewport=event.new,
            source="iv",
        )

    def _on_derivative_viewport_changed(
        self,
        event: "pn.parameterized.Event",
    ) -> None:
        self._sync_xrange_from_viewport(
            viewport=event.new,
            source="derivative",
        )

    def _sync_xrange_from_viewport(
        self,
        *,
        viewport: object,
        source: str,
    ) -> None:
        if self._syncing_xrange:
            return

        x_range = self._extract_xrange_from_viewport(viewport)
        if x_range is None:
            if self._viewport_requests_autorange(viewport):
                self._syncing_xrange = True
                try:
                    if source == "iv":
                        self._derivative_figure.update_xaxes(autorange=True)
                    else:
                        self._iv_figure.update_xaxes(autorange=True)
                finally:
                    self._syncing_xrange = False
            return

        self._syncing_xrange = True
        try:
            if source == "iv":
                self._derivative_figure.update_xaxes(
                    range=list(x_range),
                    autorange=False,
                )
            else:
                self._iv_figure.update_xaxes(
                    range=list(x_range),
                    autorange=False,
                )
        finally:
            self._syncing_xrange = False

    def _extract_xrange_from_viewport(
        self,
        viewport: object,
    ) -> tuple[float, float] | None:
        if not isinstance(viewport, dict):
            return None

        range_pair = viewport.get("xaxis.range")
        if isinstance(range_pair, (list, tuple)) and len(range_pair) == 2:
            return (float(range_pair[0]), float(range_pair[1]))

        lower = viewport.get("xaxis.range[0]")
        upper = viewport.get("xaxis.range[1]")
        if lower is not None and upper is not None:
            return (float(lower), float(upper))

        return None

    def _viewport_requests_autorange(self, viewport: object) -> bool:
        return isinstance(viewport, dict) and bool(viewport.get("xaxis.autorange"))

    def _format_parameter_table(self) -> str:
        rows = ["| Parameter | Locked | Guess | Fit |", "| --- | --- | --- | --- |"]
        guess = self._current_guess()
        results = self._solution["params"] if self._solution else None
        for idx, display in enumerate(self._display_keys):
            locked = "✅" if self._lock_checkboxes[display].value else ""
            fit_value = results[idx].value if results is not None else guess[idx]
            rows.append(
                f"| {display} | {locked} | {guess[idx]:.4g} | {fit_value:.4g} |"
            )
        return "\n".join(rows)

    def _format_status_text(self) -> str:
        if self._solution is None:
            return "No fit performed yet. Click **Fit** to start optimization."
        sol = self._solution
        chi2 = float(np.sum((sol["I_fit_nA"] - sol["I_exp_nA"]) ** 2))
        formatted_params: list[str] = []
        for param in sol["params"]:
            value_text = f"{param.value:.3g}" if param.value is not None else "n/a"
            error_text = f"{param.error:.3g}" if param.error is not None else "n/a"
            formatted_params.append(f"{param.name}={value_text}±{error_text}")
        param_text = ", ".join(formatted_params)
        maxfev = sol.get("maxfev")
        maxfev_text = str(maxfev) if maxfev is not None else "∞"
        return (
            f"**Chi²:** {chi2:.3g}  \n"
            f"**Maxfev:** {maxfev_text}  \n"
            f"**Parameters:** {param_text}"
        )

    def _current_parameters(self) -> list[ParameterSpec]:
        result: list[ParameterSpec] = []
        for template in self._parameter_templates:
            value = self._sliders[template.name].value
            locked = bool(self._lock_checkboxes[template.name].value)
            result.append(
                ParameterSpec(
                    name=template.name,
                    label=template.label,
                    lower=template.lower,
                    upper=template.upper,
                    guess=float(value),
                    fixed=locked,
                )
            )
        return result

    def _current_guess(self) -> NDArray64:
        return np.array(
            [self._sliders[display].value for display in self._display_keys],
            dtype=np.float64,
        )

    def _set_solution(self, solution: Optional[SolutionDict]) -> None:
        self._solution = solution
        if self._on_solution_changed is not None:
            self._on_solution_changed(solution)

    def _compute_curve_from_guess(self) -> NDArray64:
        return _pat_trace(self.V_mV, self._current_guess(), model=self.model)

    def _clean_numeric_trace(self, arr: NDArray) -> NDArray64:
        try:
            data = np.asarray(arr, dtype=np.float64)
        except TypeError:
            data = np.asarray(arr, dtype=object)
            if _NoValue is not None:
                mask = np.equal(data, _NoValue)
                data = np.where(mask, np.nan, data)
            data = np.asarray(data, dtype=np.float64)
        return data

    def _update_plot_traces(self) -> None:
        self._iv_figure.data[0].y = self.I_nA
        self._iv_figure.data[1].y = self._initial_curve
        self._iv_figure.data[2].y = self._fit_curve
        self._derivative_figure.data[0].y = self._data_derivative
        self._derivative_figure.data[1].y = self._initial_derivative
        self._derivative_figure.data[2].y = self._fit_derivative
        self._iv_pane.object = self._iv_figure
        self._derivative_pane.object = self._derivative_figure

    def _format_trace_header(self) -> str:
        return f"Showing trace {self._trace_index + 1} of {self._trace_matrix.shape[0]}"

    def _prepare_trace_matrix(self, I_nA: NDArray) -> NDArray64:
        if np.ma.isMaskedArray(I_nA):
            data = self._clean_numeric_trace(I_nA.filled(np.nan))
        else:
            data = self._clean_numeric_trace(I_nA)

        if data.ndim == 0:
            raise ValueError("I_nA must contain at least one data point.")
        if data.ndim == 1:
            data = data[np.newaxis, :]
        elif data.ndim > 2:
            raise ValueError("I_nA must be 1D or 2D.")

        if data.shape[1] != self.V_mV.size:
            raise ValueError("Each trace must match the length of V_mV.")

        require_all_finite(data, "I_nA")
        require_min_size(data, 1, "I_nA")
        return data

    def _gradient(self, values: NDArray, axis: Optional[NDArray] = None) -> NDArray64:
        axis_values = axis if axis is not None else self.V_mV
        return np.gradient(values, axis_values)

    def _format_model_text(self) -> str:
        if self.model in {"conv", "conv_pat", "conv+pat", "conv_dynes"}:
            return r"""
\[
I_\mathrm{BCS}(V)=G_\mathrm{N}\left[
N_2(1-f_2)\otimes N_1 -
N_1 f_1 \otimes N_2
\right](eV) \qquad (A=0)
\]

\[
N_i(E)=\Re\!\left(
\frac{E+i\gamma_i}{\sqrt{(E+i\gamma_i)^2-\Delta_i^2(T)}}
\right), \qquad
f_i(E)=\frac{1}{1+\exp\!\left(\frac{E}{k_\mathrm{B}T}\right)}
\]

\[
I_\mathrm{PAT}(V)=\sum_{n=-\infty}^{\infty}
J_n^2\!\left(\frac{eA}{h \nu}\right)
I_\mathrm{BCS}\!\left(V + n\,\frac{h \nu}{e}\right) \qquad (A>0)
\]
"""

        return r"""
\[
I_\mathrm{BCS}(V) = \int_{-\infty}^{\infty} N(E) \cdot N(E+eV) \cdot \left(f(E)-f(E+eV)\right)\mathrm{d}E \qquad (A=0)
\]

\[
N(E) = \Re\!\left( \frac{E+i\gamma}{\sqrt{(E+i\gamma)^{2}-\Delta^{2}(T)}} \right)\,, \quad
f(E) = \frac{1}{1+\exp\!\left(\frac{E}{k_\mathrm{B}T}\right)}\,, \quad
\frac{\Delta(T)}{\Delta_0} \approx \tanh\!\left( 1.74\,\sqrt{\frac{\Delta_0}{1.764\,k_\mathrm{B}T} - 1} \right)
\]

\[
I_\mathrm{PAT}(V)=\sum_{n=-\infty}^{\infty}
J_n^2\!\left(\frac{eA}{h \nu}\right)
I_\mathrm{BCS}\!\left(V + n\,\frac{h \nu}{e}\right) \qquad (A>0)
\]
"""
