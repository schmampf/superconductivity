from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import numpy as np
import plotly.graph_objects as go
from numpy.typing import NDArray

from superconductivity.utilities.safety import (
    require_all_finite,
    require_min_size,
    to_1d_float64,
)
from superconductivity.utilities.types import NDArray64

from .fit_pat import DEFAULT_PARAMETERS, PARAMETER_NAMES, Parameter, SolutionDict, fit_pat
from .models import get_model

try:
    from panel.util import _NoValue
except (ImportError, ModuleNotFoundError):
    _NoValue = object()


def _pat_trace(V_mV: NDArray64, params: NDArray64) -> NDArray64:
    function, parameter_mask = get_model(model="pat")
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
        pn.extension("plotly")
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
        on_solution_changed: Optional[
            Callable[[Optional[SolutionDict]], None]
        ] = None,
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
        self._fit_button.on_click(self._on_fit_click)

        self._solution: Optional[SolutionDict] = None

        self._parameter_table = self._pn.pane.Markdown(
            self._format_parameter_table(), sizing_mode="stretch_width"
        )
        self._status_panel = self._pn.pane.Markdown(self._format_status_text())

        self._solution: Optional[SolutionDict] = None

        self._iv_pane = self._pn.pane.Plotly(
            self._iv_figure,
            sizing_mode="stretch_width",
            height=350,
            config={"responsive": True},
        )
        self._derivative_pane = self._pn.pane.Plotly(
            self._derivative_figure,
            sizing_mode="stretch_width",
            height=350,
            config={"responsive": True},
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
            self._trace_header,
            self._trace_selector,
            *self._slider_rows,
            self._fit_button,
            self._parameter_table,
            self._status_panel,
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
        self._parameter_table.object = self._format_parameter_table()
        self._status_panel.object = self._format_status_text()

    def _build_sliders(self) -> None:
        for spec in self._parameter_templates:
            step = max((spec.upper - spec.lower) / 200.0, 1e-6)
            slider = self._pn.widgets.FloatSlider(
                name=f"{spec.name} ({spec.description})",
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
                marker=dict(size=4, opacity=0.6),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._initial_curve,
                name="Initial",
                line=dict(dash="dash", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._fit_curve,
                name="Fit",
                line=dict(width=2),
            )
        )
        fig.update_layout(
            title="I-V trace",
            xaxis_title="Voltage (mV)",
            yaxis_title="Current (nA)",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=30, r=20, t=30, b=30),
        )
        return fig

    def _create_derivative_figure(self) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._data_derivative,
                name="Data",
                mode="lines",
                line=dict(color="grey", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._initial_derivative,
                name="Initial",
                line=dict(dash="dash", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=self.V_mV,
                y=self._fit_derivative,
                name="Fit",
                line=dict(width=2),
            )
        )
        fig.update_layout(
            title="dI/dV",
            xaxis_title="Voltage (mV)",
            yaxis_title="Current derivative",
            legend=dict(orientation="h", y=-0.2),
            margin=dict(l=30, r=20, t=30, b=30),
        )
        return fig

    def _on_slider_changed(self, event: "pn.parameterized.Event") -> None:
        if event.new == event.old:
            return
        self._initial_curve = self._compute_curve_from_guess()
        self._initial_derivative = self._gradient(self._initial_curve)
        self._iv_figure.data[1].y = self._initial_curve
        self._derivative_figure.data[1].y = self._initial_derivative
        self._update_plot_traces()
        self._parameter_table.object = self._format_parameter_table()

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
        self._parameter_table.object = self._format_parameter_table()
        self._status_panel.object = self._format_status_text()

    def _on_lock_toggled(self, event: "pn.parameterized.Event", *, key: str) -> None:
        slider = self._sliders[key]
        slider.disabled = bool(event.new)
        self._parameter_table.object = self._format_parameter_table()

    def _on_fit_click(self, _: object) -> None:
        solution = fit_pat(
            self.V_mV,
            self.I_nA,
            parameters=self._current_parameters(),
            weights=self.weights,
            maxfev=self.maxfev,
        )
        self.update_solution(solution)

    def _format_parameter_table(self) -> str:
        rows = ["| Parameter | Locked | Guess | Fit |", "| --- | --- | --- | --- |"]
        guess = self._current_guess()
        results = self._solution["params"] if self._solution else None
        for idx, display in enumerate(self._display_keys):
            locked = "✅" if self._lock_checkboxes[display].value else ""
            fit_value = results[idx].fit_value if results is not None else guess[idx]
            rows.append(
                f"| {display} | {locked} | {guess[idx]:.4g} | {fit_value:.4g} |"
            )
        return "\n".join(rows)

    def _format_status_text(self) -> str:
        if self._solution is None:
            return "No fit performed yet. Click **Fit** to start optimization."
        sol = self._solution
        chi2 = float(np.sum((sol["I_fit_nA"] - sol["I_exp_nA"]) ** 2))
        param_text = ", ".join(
            f"{param.name}={param.fit_value:.3g}±{param.fit_error:.3g}"
            for param in sol["params"]
        )
        maxfev = sol.get("maxfev")
        maxfev_text = str(maxfev) if maxfev is not None else "∞"
        return (
            f"**Chi²:** {chi2:.3g}  \n"
            f"**Maxfev:** {maxfev_text}  \n"
            f"**Parameters:** {param_text}"
        )

    def _current_parameters(self) -> list[Parameter]:
        result: list[Parameter] = []
        for template in self._parameter_templates:
            value = self._sliders[template.name].value
            locked = bool(self._lock_checkboxes[template.name].value)
            result.append(
                Parameter(
                    name=template.name,
                    description=template.description,
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
        return _pat_trace(self.V_mV, self._current_guess())

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
