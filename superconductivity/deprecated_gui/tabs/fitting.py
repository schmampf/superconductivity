from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import replace

import numpy as np
import pandas as pd

from .._compat import BCSModelConfig, get_model_spec
from ..common import _mapping_frame
from ..state import _fit_sampling_trace

_FIT_CONFIG_LABELS = OrderedDict(
    [
        ("jax", "JAX"),
        ("conv", "Conv"),
        ("pat_enabled", "PAT"),
        ("noise_enabled", "Noise"),
    ]
)
_FIT_CONFIG_TITLES = {
    "parameter": "Parameter",
    "value": "Value",
}


class GUIFitTabMixin:
    def _build_fit_widgets(self) -> None:
        self._fit_button = self._pn.widgets.Button(
            name="Fit",
            button_type="primary",
            width=90,
            margin=0,
        )
        self._fit_button.on_click(self._on_fit_clicked)
        self._spinner = self._pn.indicators.LoadingSpinner(
            value=False,
            width=20,
            height=20,
            margin=0,
        )

        self._fit_config_table = self._pn.widgets.Tabulator(
            self._fit_config_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_data_fill",
            sizing_mode="fixed",
            width=220,
            height=170,
            widths={
                "parameter": 110,
            },
            editors={
                "parameter": None,
                "value": {"type": "tickCross"},
            },
            formatters={
                "value": {"type": "tickCross"},
            },
            titles=_FIT_CONFIG_TITLES,
            title_formatters={key: {"type": "html"} for key in _FIT_CONFIG_TITLES},
            margin=0,
        )
        self._fit_config_table.on_edit(self._on_fit_config_edit)

        self._parameter_table = self._build_parameter_table(
            self._parameter_frame(self._parameters),
            height=self._parameter_table_height(),
        )
        self._parameter_table.on_edit(self._on_parameter_edit)

        self._model_info_table = self._pn.widgets.Tabulator(
            _mapping_frame(self._model_info()),
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=False,
            layout="fit_data_fill",
            sizing_mode="stretch_width",
            height=220,
            widths={
                "key": 150,
            },
            margin=0,
        )
        self._optimizer_info_table = self._pn.widgets.Tabulator(
            _mapping_frame(self._optimizer_info()),
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=False,
            layout="fit_data_fill",
            sizing_mode="stretch_width",
            height=220,
            widths={
                "key": 150,
            },
            margin=0,
        )
        self._model_html = self._pn.pane.LaTeX(
            get_model_spec(self._fit_model_config).html,
            sizing_mode="stretch_width",
            margin=0,
        )

    def _build_parameter_table(
        self,
        frame: pd.DataFrame,
        *,
        height: int,
    ):
        return self._pn.widgets.Tabulator(
            frame,
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["name"],
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=height,
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
            margin=0,
        )

    def _fit_tab(self):
        self._fit_inner_tabs = self._pn.Tabs(
            (
                "Model",
                self._pn.Column(
                    self._parameter_table,
                    self._pn.Row(
                        self._fit_config_table,
                        self._pn.Spacer(width=16),
                        self._pn.Column(
                            self._model_html,
                            sizing_mode="stretch_width",
                            margin=0,
                        ),
                        sizing_mode="stretch_width",
                        margin=0,
                    ),
                    sizing_mode="stretch_width",
                    margin=0,
                ),
            ),
            (
                "Optimizer",
                self._pn.Column(
                    self._pn.pane.Markdown("### Backend model", margin=0),
                    self._model_info_table,
                    self._pn.pane.Markdown("### Optimizer", margin=0),
                    self._optimizer_info_table,
                    sizing_mode="stretch_width",
                    margin=0,
                ),
            ),
            sizing_mode="stretch_width",
            margin=0,
        )
        return self._pn.Column(
            self._pn.Row(
                self._fit_button,
                self._spinner,
                self._fit_state,
                sizing_mode="stretch_width",
                margin=0,
            ),
            self._fit_inner_tabs,
            margin=0,
            sizing_mode="stretch_width",
        )

    def _parameter_table_height(self) -> int:
        return min(max(130, 42 * len(self._parameters) + 30), 360)

    def _parameter_frame(self, parameters: list[object]) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        result_lookup = {}
        if self._fit_solution is not None:
            result_lookup = {
                parameter.name: parameter for parameter in self._fit_solution["params"]
            }
        for parameter in parameters:
            result = result_lookup.get(parameter.name)
            fit_text = self._format_fit_text(
                None if result is None else result.value,
                None if result is None else result.error,
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

    def _fit_config_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "key": "jax",
                    "parameter": _FIT_CONFIG_LABELS["jax"],
                    "value": bool(self._fit_model_config.backend == "jax"),
                },
                {
                    "key": "conv",
                    "parameter": _FIT_CONFIG_LABELS["conv"],
                    "value": bool(self._fit_model_config.kernel == "conv"),
                },
                {
                    "key": "pat_enabled",
                    "parameter": _FIT_CONFIG_LABELS["pat_enabled"],
                    "value": bool(self._fit_model_config.pat_enabled),
                },
                {
                    "key": "noise_enabled",
                    "parameter": _FIT_CONFIG_LABELS["noise_enabled"],
                    "value": bool(self._fit_model_config.noise_enabled),
                },
            ],
            dtype=object,
        )

    def _model_info(self) -> OrderedDict[str, object]:
        model_spec = get_model_spec(self._fit_model_config)
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
        info["active_trace"] = self.active_index
        info["free_parameters"] = free_parameters
        info["chi2"] = f"{chi2:.6g}" if np.isfinite(chi2) else "n/a"
        info["reduced_chi2"] = (
            f"{reduced_chi2:.6g}" if np.isfinite(reduced_chi2) else "n/a"
        )
        info["elapsed_s"] = f"{self._last_fit_elapsed_s:.3f}"
        info["status"] = self._last_fit_status
        info["last_error"] = self._last_fit_error or ""
        return info

    def _chi2_metrics(self) -> tuple[float, float]:
        if self._fit_solution is None:
            return np.nan, np.nan
        residual = np.asarray(
            self._fit_solution["I_exp_nA"] - self._fit_solution["I_fit_nA"],
            dtype=np.float64,
        )
        finite = np.isfinite(residual)
        if not np.any(finite):
            return np.nan, np.nan
        chi2 = float(np.sum(residual[finite] * residual[finite]))
        dof = int(np.sum(finite)) - sum(
            not parameter.fixed for parameter in self._parameters
        )
        reduced = chi2 / dof if dof > 0 else np.nan
        return chi2, reduced

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

    def _reset_fit_status(self) -> None:
        self._last_fit_error = ""
        self._last_fit_elapsed_s = 0.0
        self._last_fit_status = "idle"
        self._fit_state.object = "Idle"

    def _refresh_fit_views(self) -> None:
        self._fit_config_table.value = self._fit_config_frame()
        self._parameter_table.value = self._parameter_frame(self._parameters)
        self._parameter_table.height = self._parameter_table_height()
        self._model_info_table.value = _mapping_frame(self._model_info())
        self._optimizer_info_table.value = _mapping_frame(self._optimizer_info())
        self._model_html.object = get_model_spec(self._fit_model_config).html

    def _recompute_after_parameter_change(self) -> None:
        self._sync_active_fit_model()
        self._recompute_fit_curves()
        self._refresh_left_plots()
        self._refresh_fit_views()

    def _apply_parameter_edit(self, event: object) -> None:
        column = str(getattr(event, "column", ""))
        row = int(getattr(event, "row"))
        value = getattr(event, "value")
        parameter = self._parameters[row]

        if column == "guess":
            parameter.guess = float(
                np.clip(float(value), parameter.lower, parameter.upper)
            )
            self._recompute_after_parameter_change()
            return

        if column == "lower":
            parameter.lower = float(value)
            if parameter.lower > parameter.upper:
                parameter.upper = parameter.lower
            parameter.guess = float(
                np.clip(parameter.guess, parameter.lower, parameter.upper)
            )
            self._recompute_after_parameter_change()
            return

        if column == "upper":
            parameter.upper = float(value)
            if parameter.upper < parameter.lower:
                parameter.lower = parameter.upper
            parameter.guess = float(
                np.clip(parameter.guess, parameter.lower, parameter.upper)
            )
            self._recompute_after_parameter_change()
            return

        if column == "fixed":
            parameter.fixed = bool(value)
            self._sync_active_fit_model()
            self._refresh_fit_views()

    def _on_parameter_edit(self, event: object) -> None:
        self._apply_parameter_edit(event)

    def _on_fit_config_edit(self, event: object) -> None:
        if str(getattr(event, "column", "")) != "value":
            return
        if self._fit_running:
            self._refresh_fit_views()
            return
        row = int(getattr(event, "row"))
        value = bool(getattr(event, "value"))
        frame = self._fit_config_table.value.reset_index(drop=True)
        key = str(frame.at[row, "key"])

        if key == "jax":
            self._update_fit_model_config(backend="jax" if value else "np")
            return
        if key == "conv":
            self._update_fit_model_config(kernel="conv" if value else "int")
            return
        if key == "pat_enabled":
            self._update_fit_model_config(pat_enabled=value)
            return
        if key == "noise_enabled":
            self._update_fit_model_config(noise_enabled=value)

    def _update_fit_model_config(
        self,
        *,
        kernel: str | None = None,
        backend: str | None = None,
        pat_enabled: bool | None = None,
        noise_enabled: bool | None = None,
    ) -> None:
        if self._fit_running:
            self._refresh_fit_views()
            return
        self._fit_model_config = BCSModelConfig(
            kernel=self._fit_model_config.kernel if kernel is None else kernel,
            backend=self._fit_model_config.backend if backend is None else backend,
            pat_enabled=(
                self._fit_model_config.pat_enabled
                if pat_enabled is None
                else pat_enabled
            ),
            noise_enabled=(
                self._fit_model_config.noise_enabled
                if noise_enabled is None
                else noise_enabled
            ),
            noise_oversample=self._fit_model_config.noise_oversample,
        )
        self._sync_active_fit_model()
        self._clear_fit_solution()
        self._recompute_fit_curves()
        self._refresh_left_plots()
        self._refresh_fit_views()
        self._notify_state_changed()

    def _on_fit_clicked(self, _: object) -> None:
        if self._fit_running or self._offset_batch_running:
            return

        self._fit_running = True
        self._fit_started_at = time.perf_counter()
        self._spinner.value = True
        self._fit_button.disabled = True
        self._last_fit_status = "running"
        self._last_fit_error = ""
        self._fit_state.object = "Running (0.0 s)"
        self._refresh_fit_views()
        self._start_fit_timer()

        self._fit_future = self._executor.submit(
            _fit_sampling_trace,
            self._require_sampling(),
            model=self.model_key,
            parameters=[replace(parameter) for parameter in self._parameters],
            maxfev=None,
        )
        self._fit_future.add_done_callback(self._on_fit_finished)

    def _on_fit_finished(self, future: object) -> None:
        def finalize() -> None:
            if self._fit_started_at is not None:
                self._last_fit_elapsed_s = time.perf_counter() - self._fit_started_at
            try:
                solution = future.result()
            except Exception as exc:
                self._clear_fit_solution()
                self._fit_curve = self._initial_curve.copy()
                self._last_fit_status = "failed"
                self._last_fit_error = f"{type(exc).__name__}: {exc}"
                self._fit_state.object = (
                    f"Failed after {self._last_fit_elapsed_s:.3f} s: "
                    f"`{self._last_fit_error}`"
                )
            else:
                self._set_fit_solution(solution)
                self._fit_curve = np.asarray(
                    solution["I_fit_nA"],
                    dtype=np.float64,
                )
                self._last_fit_status = "done"
                self._last_fit_error = ""
                self._fit_state.object = f"Done ({self._last_fit_elapsed_s:.3f} s)"
            finally:
                self._fit_running = False
                self._fit_started_at = None
                self._spinner.value = False
                self._fit_button.disabled = False
                if self._fit_timer is not None:
                    self._fit_timer.stop()
                    self._fit_timer = None
                self._refresh_left_plots()
                self._refresh_fit_views()
                self._notify_state_changed()

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
        self._fit_state.object = f"Running ({elapsed_s:.1f} s)"
