from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..evaluation.traces import numeric_yvalue
from .style import gui_trace_label, gui_trace_style
from ..utilities.constants import G_0_muS
from ..utilities.types import NDArray64

_LEFT_STAGE_BUTTON_ORDER = (
    "raw",
    "downsampled",
    "offset",
    "binned",
    "smoothed",
    "initial",
    "fit",
)
_LEFT_STAGE_TRACE_ORDER = _LEFT_STAGE_BUTTON_ORDER
_LEFT_STAGE_DEFAULTS = ["binned", "initial", "fit"]
_LEFT_V_QUANTITY_ORDER = (
    "iv_v",
    "didv_v",
    "i_over_v_v",
    "dvdi_v",
)
_LEFT_I_QUANTITY_ORDER = (
    "vi_i",
    "dvdi_i",
    "v_over_i_i",
    "didv_i",
)
_LEFT_V_QUANTITY_DEFAULTS = ["iv_v", "didv_v"]
_LEFT_I_QUANTITY_DEFAULTS = ["vi_i", "dvdi_i"]
_LEFT_V_QUANTITY_LABELS = {
    "iv_v": "I (V)",
    "didv_v": "dI/dV (V)",
    "i_over_v_v": "I/V (V)",
    "dvdi_v": "dV/dI (V)",
}
_LEFT_I_QUANTITY_LABELS = {
    "vi_i": "V (I)",
    "dvdi_i": "dV/dI (I)",
    "v_over_i_i": "V/I (I)",
    "didv_i": "dI/dV (I)",
}


class GUILeftMixin:
    def _y_axis_html_label(self) -> str:
        keys = getattr(self, "_keys", None)
        if keys is not None:
            html_label = getattr(keys, "html_label", None)
            if isinstance(html_label, str) and html_label.strip() != "":
                return html_label
        keysspec = getattr(self, "_keysspec", None)
        if keysspec is not None:
            html_label = getattr(keysspec, "html_label", None)
            if isinstance(html_label, str) and html_label.strip() != "":
                return html_label
        return "y"

    def _build_left_controls(self) -> None:
        self._left_stage_selector = self._pn.widgets.CheckButtonGroup(
            options=OrderedDict(
                (
                    gui_trace_label(stage),
                    stage,
                )
                for stage in _LEFT_STAGE_BUTTON_ORDER
            ),
            value=list(_LEFT_STAGE_DEFAULTS),
            sizing_mode="stretch_width",
        )
        self._left_stage_selector.param.watch(
            self._on_left_stage_changed,
            "value",
        )
        self._left_v_quantity_selector = self._pn.widgets.CheckButtonGroup(
            options=OrderedDict(
                (
                    _LEFT_V_QUANTITY_LABELS[key],
                    key,
                )
                for key in _LEFT_V_QUANTITY_ORDER
            ),
            value=list(_LEFT_V_QUANTITY_DEFAULTS),
            sizing_mode="stretch_width",
        )
        self._left_v_quantity_selector.param.watch(
            self._on_left_quantity_changed,
            "value",
        )
        self._left_i_quantity_selector = self._pn.widgets.CheckButtonGroup(
            options=OrderedDict(
                (
                    _LEFT_I_QUANTITY_LABELS[key],
                    key,
                )
                for key in _LEFT_I_QUANTITY_ORDER
            ),
            value=list(_LEFT_I_QUANTITY_DEFAULTS),
            sizing_mode="stretch_width",
        )
        self._left_i_quantity_selector.param.watch(
            self._on_left_quantity_changed,
            "value",
        )

    def _on_left_stage_changed(self, _: object) -> None:
        self._refresh_left_plots()

    def _on_left_quantity_changed(self, _: object) -> None:
        self._refresh_left_plots()

    def _build_plot_panes(self) -> None:
        self._left_v_view_state: dict[str, object] = {
            "x_range": None,
            "y_ranges": {},
        }
        self._left_i_view_state: dict[str, object] = {
            "x_range": None,
            "y_ranges": {},
        }
        self._left_v_row_keys: list[str] = []
        self._left_i_row_keys: list[str] = []
        self._iv_figure = self._build_left_stack_figure(axis_kind="V")
        self._vi_figure = self._build_left_stack_figure(axis_kind="I")

        self._experimental_time_figure = self._build_experimental_time_figure()
        self._experimental_psd_figure = self._build_experimental_psd_figure()

        self._offset_g_figure = self._build_offset_figure(kind="G")
        self._offset_r_figure = self._build_offset_figure(kind="R")
        self._offset_batch_v_figure = self._build_offset_batch_figure(kind="V")
        self._offset_batch_i_figure = self._build_offset_batch_figure(kind="I")

        self._sampling_offset_v_figure = self._build_sampling_offset_figure(
            kind="V"
        )
        self._sampling_offset_i_figure = self._build_sampling_offset_figure(
            kind="I"
        )

        self._iv_pane = self._pn.pane.Plotly(
            self._iv_figure,
            sizing_mode="stretch_width",
            height=self._left_figure_height(axis_kind="V"),
            config={"responsive": True},
        )
        self._vi_pane = self._pn.pane.Plotly(
            self._vi_figure,
            sizing_mode="stretch_width",
            height=self._left_figure_height(axis_kind="I"),
            config={"responsive": True},
        )
        self._iv_pane.param.watch(self._on_left_v_relayout, "relayout_data")
        self._vi_pane.param.watch(self._on_left_i_relayout, "relayout_data")
        self._experimental_time_pane = self._pn.pane.Plotly(
            self._experimental_time_figure,
            sizing_mode="stretch_width",
            height=420,
            config={"responsive": True},
            margin=0,
        )
        self._experimental_psd_pane = self._pn.pane.Plotly(
            self._experimental_psd_figure,
            sizing_mode="stretch_width",
            height=420,
            config={"responsive": True},
            margin=0,
        )
        self._offset_g_pane = self._pn.pane.Plotly(
            self._offset_g_figure,
            sizing_mode="stretch_width",
            height=480,
            config={"responsive": True},
            margin=0,
        )
        self._offset_r_pane = self._pn.pane.Plotly(
            self._offset_r_figure,
            sizing_mode="stretch_width",
            height=480,
            config={"responsive": True},
            margin=0,
        )
        self._offset_batch_v_pane = self._pn.pane.Plotly(
            self._offset_batch_v_figure,
            sizing_mode="stretch_width",
            height=240,
            config={"responsive": True},
            margin=0,
        )
        self._offset_batch_i_pane = self._pn.pane.Plotly(
            self._offset_batch_i_figure,
            sizing_mode="stretch_width",
            height=240,
            config={"responsive": True},
            margin=0,
        )
        self._sampling_offset_v_pane = self._pn.pane.Plotly(
            self._sampling_offset_v_figure,
            sizing_mode="stretch_width",
            height=240,
            config={"responsive": True},
            margin=0,
        )
        self._sampling_offset_i_pane = self._pn.pane.Plotly(
            self._sampling_offset_i_figure,
            sizing_mode="stretch_width",
            height=240,
            config={"responsive": True},
            margin=0,
        )

    def _left_trace(self, stage: str) -> go.Scatter:
        style = gui_trace_style(stage)
        return go.Scatter(
            x=np.empty((0,), dtype=np.float64),
            y=np.empty((0,), dtype=np.float64),
            name=gui_trace_label(stage),
            mode=style["mode"],
            line=style.get("line"),
            marker=style.get("marker"),
            visible=stage in _LEFT_STAGE_DEFAULTS,
            showlegend=False,
        )

    def _left_trace_visible(self, stage: str) -> bool:
        if stage == "fit" and self._fit_solution is None:
            return False
        if stage == "smoothed" and self._smoothed_sampling is None:
            return False
        return stage in self._left_stage_selector.value

    def _on_left_v_relayout(self, event: object) -> None:
        self._update_left_view_state(
            axis_kind="V",
            relayout_data=getattr(event, "new", None),
        )

    def _on_left_i_relayout(self, event: object) -> None:
        self._update_left_view_state(
            axis_kind="I",
            relayout_data=getattr(event, "new", None),
        )

    def _left_iv_stage_data(self, stage: str) -> tuple[NDArray64, NDArray64]:
        trace = self._active_trace()
        if stage == "raw":
            return (
                np.asarray(trace["V_mV"], dtype=np.float64),
                np.asarray(trace["I_nA"], dtype=np.float64),
            )
        if stage == "downsampled":
            return (
                np.asarray(self._downsampled_V_mV, dtype=np.float64),
                np.asarray(self._downsampled_I_nA, dtype=np.float64),
            )
        if stage == "offset":
            offset_trace = self._require_offset_corrected_trace()
            return (
                np.asarray(offset_trace["V_mV"], dtype=np.float64),
                np.asarray(offset_trace["I_nA"], dtype=np.float64),
            )
        if stage == "binned":
            sampling = self._require_raw_sampling()
            return (
                np.asarray(sampling["Vbins_mV"], dtype=np.float64),
                np.asarray(sampling["I_nA"], dtype=np.float64),
            )
        if stage == "smoothed":
            sampling = self._require_sampling()
            return (
                np.asarray(sampling["Vbins_mV"], dtype=np.float64),
                np.asarray(sampling["I_nA"], dtype=np.float64),
            )
        sampling = self._require_sampling()
        if stage == "initial":
            return (
                np.asarray(sampling["Vbins_mV"], dtype=np.float64),
                np.asarray(self._initial_curve, dtype=np.float64),
            )
        return (
            np.asarray(sampling["Vbins_mV"], dtype=np.float64),
            np.asarray(self._fit_curve, dtype=np.float64),
        )

    def _left_didv_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        V_mV, I_nA = self._left_iv_stage_data(stage)
        return V_mV, self._gradient(V_mV, I_nA) / G_0_muS

    @staticmethod
    def _sorted_unique_curve(
        x: NDArray64,
        y: NDArray64,
    ) -> tuple[NDArray64, NDArray64]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        x = x[finite]
        y = y[finite]
        if x.size == 0:
            return x, y
        order = np.argsort(x, kind="mergesort")
        x = x[order]
        y = y[order]
        unique_x, start = np.unique(x, return_index=True)
        if unique_x.size == x.size:
            return x, y
        stop = np.r_[start[1:], x.size]
        unique_y = np.asarray(
            [np.mean(y[i:j]) for i, j in zip(start, stop)],
            dtype=np.float64,
        )
        return unique_x, unique_y

    def _left_iv_over_v_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        V_mV, I_nA = self._left_iv_stage_data(stage)
        return V_mV, self._safe_divide(I_nA, V_mV) / G_0_muS

    def _left_dvdi_vs_v_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        V_mV, didv_G0 = self._left_didv_stage_data(stage)
        return V_mV, self._safe_divide(1.0, didv_G0)

    def _left_vi_stage_data(self, stage: str) -> tuple[NDArray64, NDArray64]:
        V_mV, I_nA = self._left_iv_stage_data(stage)
        return self._sorted_unique_curve(I_nA, V_mV)

    def _left_dvdi_vs_i_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        I_nA, V_mV = self._left_vi_stage_data(stage)
        return I_nA, self._gradient(I_nA, V_mV) * G_0_muS

    def _left_v_over_i_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        I_nA, V_mV = self._left_vi_stage_data(stage)
        return I_nA, self._safe_divide(V_mV, I_nA) * G_0_muS

    def _left_didv_vs_i_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        I_nA, dvdi_R0 = self._left_dvdi_vs_i_stage_data(stage)
        return I_nA, self._safe_divide(1.0, dvdi_R0)

    @staticmethod
    def _safe_divide(
        numerator: NDArray64 | float,
        denominator: NDArray64 | float,
    ) -> NDArray64:
        numerator_array = np.asarray(numerator, dtype=np.float64)
        denominator_array = np.asarray(denominator, dtype=np.float64)
        out = np.full(
            np.broadcast_shapes(
                numerator_array.shape,
                denominator_array.shape,
            ),
            np.nan,
            dtype=np.float64,
        )
        return np.divide(
            numerator_array,
            denominator_array,
            out=out,
            where=np.isfinite(denominator_array) & (denominator_array != 0.0),
        )

    def _left_quantity_keys(self, *, axis_kind: str) -> list[str]:
        if axis_kind == "V":
            return list(self._left_v_quantity_selector.value)
        return list(self._left_i_quantity_selector.value)

    def _left_view_state(self, *, axis_kind: str) -> dict[str, object]:
        if axis_kind == "V":
            return self._left_v_view_state
        return self._left_i_view_state

    def _left_row_keys(self, *, axis_kind: str) -> list[str]:
        if axis_kind == "V":
            return list(self._left_v_row_keys)
        return list(self._left_i_row_keys)

    @staticmethod
    def _axis_name(prefix: str, row_index: int) -> str:
        if row_index == 1:
            return prefix
        return f"{prefix}{row_index}"

    @staticmethod
    def _extract_axis_range(
        relayout_data: dict[str, object],
        *,
        axis_name: str,
    ) -> tuple[float, float] | None:
        combined = relayout_data.get(f"{axis_name}.range")
        if isinstance(combined, (list, tuple)) and len(combined) == 2:
            return (float(combined[0]), float(combined[1]))
        key0 = f"{axis_name}.range[0]"
        key1 = f"{axis_name}.range[1]"
        if key0 in relayout_data and key1 in relayout_data:
            return (
                float(relayout_data[key0]),
                float(relayout_data[key1]),
            )
        return None

    def _update_left_view_state(
        self,
        *,
        axis_kind: str,
        relayout_data: object,
    ) -> None:
        if not isinstance(relayout_data, dict) or len(relayout_data) == 0:
            return
        state = self._left_view_state(axis_kind=axis_kind)
        row_keys = self._left_row_keys(axis_kind=axis_kind)

        x_range = None
        x_autorange = False
        for row_index in range(1, max(1, len(row_keys)) + 1):
            axis_name = self._axis_name("xaxis", row_index)
            if bool(relayout_data.get(f"{axis_name}.autorange", False)):
                x_autorange = True
            extracted = self._extract_axis_range(
                relayout_data,
                axis_name=axis_name,
            )
            if extracted is not None:
                x_range = extracted
        if x_autorange:
            state["x_range"] = None
        elif x_range is not None:
            state["x_range"] = x_range

        y_ranges = dict(state["y_ranges"])
        for row_index, key in enumerate(row_keys, start=1):
            axis_name = self._axis_name("yaxis", row_index)
            if bool(relayout_data.get(f"{axis_name}.autorange", False)):
                y_ranges.pop(key, None)
                continue
            extracted = self._extract_axis_range(
                relayout_data,
                axis_name=axis_name,
            )
            if extracted is not None:
                y_ranges[key] = extracted
        state["y_ranges"] = y_ranges

    def _left_quantity_specs(
        self,
        *,
        axis_kind: str,
    ) -> list[
        tuple[
            str,
            str,
            Callable[[str], tuple[NDArray64, NDArray64]],
        ]
    ]:
        if axis_kind == "V":
            return [
                ("iv_v", "<i>I</i> (nA)", self._left_iv_stage_data),
                (
                    "didv_v",
                    "<i>dI/dV</i> (<i>G</i><sub>0</sub>)",
                    self._left_didv_stage_data,
                ),
                (
                    "i_over_v_v",
                    "<i>I/V</i> (<i>G</i><sub>0</sub>)",
                    self._left_iv_over_v_stage_data,
                ),
                (
                    "dvdi_v",
                    "<i>dV/dI</i> (<i>R</i><sub>0</sub>)",
                    self._left_dvdi_vs_v_stage_data,
                ),
            ]
        return [
            ("vi_i", "<i>V</i> (mV)", self._left_vi_stage_data),
            (
                "dvdi_i",
                "<i>dV/dI</i> (<i>R</i><sub>0</sub>)",
                self._left_dvdi_vs_i_stage_data,
            ),
            (
                "v_over_i_i",
                "<i>V/I</i> (<i>R</i><sub>0</sub>)",
                self._left_v_over_i_stage_data,
            ),
            (
                "didv_i",
                "<i>dI/dV</i> (<i>G</i><sub>0</sub>)",
                self._left_didv_vs_i_stage_data,
            ),
        ]

    def _left_figure_height(self, *, axis_kind: str) -> int:
        row_count = max(1, len(self._left_quantity_keys(axis_kind=axis_kind)))
        return max(620, 220 * row_count + 40)

    def _build_left_stack_figure(self, *, axis_kind: str) -> go.Figure:
        quantity_map = OrderedDict(
            (key, (ylabel, getter))
            for key, ylabel, getter in self._left_quantity_specs(axis_kind=axis_kind)
        )
        selected_keys = [
            key
            for key in self._left_quantity_keys(axis_kind=axis_kind)
            if key in quantity_map
        ]
        if not selected_keys:
            selected_keys = [
                (
                    _LEFT_V_QUANTITY_ORDER[0]
                    if axis_kind == "V"
                    else _LEFT_I_QUANTITY_ORDER[0]
                )
            ]
        if axis_kind == "V":
            self._left_v_row_keys = list(selected_keys)
        else:
            self._left_i_row_keys = list(selected_keys)
        row_count = len(selected_keys)
        figure = make_subplots(
            rows=row_count,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        for row_index, key in enumerate(selected_keys, start=1):
            ylabel, getter = quantity_map[key]
            for stage in _LEFT_STAGE_TRACE_ORDER:
                x, y = getter(stage)
                figure.add_trace(
                    go.Scatter(
                        x=x,
                        y=y,
                        name=gui_trace_label(stage),
                        mode=gui_trace_style(stage)["mode"],
                        line=gui_trace_style(stage).get("line"),
                        marker=gui_trace_style(stage).get("marker"),
                        visible=self._left_trace_visible(stage),
                        showlegend=False,
                    ),
                    row=row_index,
                    col=1,
                )
            figure.update_yaxes(title_text=ylabel, row=row_index, col=1)

        figure.update_layout(
            margin={"l": 80, "r": 20, "t": 20, "b": 40},
            showlegend=False,
            uirevision=f"left-{axis_kind}",
        )
        figure.update_xaxes(
            title_text="<i>V</i> (mV)" if axis_kind == "V" else "<i>I</i> (nA)",
            row=row_count,
            col=1,
        )
        self._apply_left_view_state(
            figure=figure,
            axis_kind=axis_kind,
            selected_keys=selected_keys,
        )
        return figure

    def _apply_left_view_state(
        self,
        *,
        figure: go.Figure,
        axis_kind: str,
        selected_keys: list[str],
    ) -> None:
        state = self._left_view_state(axis_kind=axis_kind)
        x_range = state.get("x_range")
        if x_range is not None:
            for row_index in range(1, len(selected_keys) + 1):
                figure.update_xaxes(range=list(x_range), row=row_index, col=1)
        y_ranges = state.get("y_ranges", {})
        if not isinstance(y_ranges, dict):
            return
        for row_index, key in enumerate(selected_keys, start=1):
            y_range = y_ranges.get(key)
            if y_range is not None:
                figure.update_yaxes(range=list(y_range), row=row_index, col=1)

    def _build_experimental_time_figure(self) -> go.Figure:
        trace = self._active_trace()
        raw_style = gui_trace_style("raw")
        downsampled_style = gui_trace_style("downsampled")
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(trace["t_s"], dtype=np.float64),
                y=np.asarray(trace["V_mV"], dtype=np.float64),
                mode=raw_style["mode"],
                line=raw_style["line"],
                name=gui_trace_label("raw"),
                legendgroup="raw",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=self._psd_downsampled_t_s,
                y=self._psd_downsampled_V_mV,
                mode=downsampled_style["mode"],
                marker=downsampled_style.get("marker"),
                line=downsampled_style["line"],
                name=gui_trace_label("downsampled"),
                legendgroup="downsampled",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(trace["t_s"], dtype=np.float64),
                y=np.asarray(trace["I_nA"], dtype=np.float64),
                mode=raw_style["mode"],
                line=raw_style["line"],
                name=gui_trace_label("raw"),
                legendgroup="raw",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=self._psd_downsampled_t_s,
                y=self._psd_downsampled_I_nA,
                mode=downsampled_style["mode"],
                marker=downsampled_style.get("marker"),
                line=downsampled_style["line"],
                name=gui_trace_label("downsampled"),
                legendgroup="downsampled",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.update_layout(
            margin={"l": 70, "r": 20, "t": 20, "b": 40},
            showlegend=False,
        )
        figure.update_yaxes(title_text="<i>V</i> (mV)", row=1, col=1)
        figure.update_yaxes(title_text="<i>I</i> (nA)", row=2, col=1)
        figure.update_xaxes(title_text="<i>t</i> (s)", row=2, col=1)
        return figure

    def _build_experimental_psd_figure(self) -> go.Figure:
        raw_psd = self._require_psd()
        downsampled_psd = self._require_downsampled_psd()
        raw_style = gui_trace_style("raw")
        downsampled_style = gui_trace_style("downsampled")
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(raw_psd["f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(raw_psd["V_psd_mV2_per_Hz"], dtype=np.float64)
                ),
                mode=raw_style["mode"],
                line=raw_style["line"],
                name=gui_trace_label("raw"),
                legendgroup="raw",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(downsampled_psd["f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(
                        downsampled_psd["V_psd_mV2_per_Hz"],
                        dtype=np.float64,
                    )
                ),
                mode=downsampled_style["mode"],
                line=downsampled_style["line"],
                marker=downsampled_style.get("marker"),
                name=gui_trace_label("downsampled"),
                legendgroup="downsampled",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(raw_psd["f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(raw_psd["I_psd_nA2_per_Hz"], dtype=np.float64)
                ),
                mode=raw_style["mode"],
                line=raw_style["line"],
                name=gui_trace_label("raw"),
                legendgroup="raw",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(downsampled_psd["f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(
                        downsampled_psd["I_psd_nA2_per_Hz"],
                        dtype=np.float64,
                    )
                ),
                mode=downsampled_style["mode"],
                line=downsampled_style["line"],
                marker=downsampled_style.get("marker"),
                name=gui_trace_label("downsampled"),
                legendgroup="downsampled",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        figure.update_layout(
            margin={"l": 70, "r": 20, "t": 20, "b": 40},
            showlegend=False,
        )
        figure.update_yaxes(
            title_text="S<sub>V</sub> (mV<sup>2</sup>/Hz)",
            type="log",
            row=1,
            col=1,
        )
        figure.update_yaxes(
            title_text="S<sub>I</sub> (nA<sup>2</sup>/Hz)",
            type="log",
            row=2,
            col=1,
        )
        figure.update_xaxes(title_text="<i>f</i> (Hz)", row=2, col=1)
        return figure

    def _build_offset_figure(self, *, kind: str) -> go.Figure:
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        offset = self._require_offset()
        fit_style = gui_trace_style("fit")
        cutoff_style = gui_trace_style("cutoff")["line"]
        if kind == "G":
            x = np.asarray(self._offset_spec.Voff_mV, dtype=np.float64)
            y = np.asarray(offset["dGerr_G0"], dtype=np.float64)
            vline = float(offset["Voff_mV"])
            xlabel = "<i>V</i><sub>off</sub> (mV)"
            ylabel_top = "<i>dG</i><sub>err</sub> / <i>G</i><sub>0</sub>"
            batch_kind = "V"
        else:
            x = np.asarray(self._offset_spec.Ioff_nA, dtype=np.float64)
            y = np.asarray(offset["dRerr_R0"], dtype=np.float64)
            vline = float(offset["Ioff_nA"])
            xlabel = "<i>I</i><sub>off</sub> (nA)"
            ylabel_top = "<i>dR</i><sub>err</sub> / <i>R</i><sub>0</sub>"
            batch_kind = "I"
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=fit_style["mode"],
                line=fit_style["line"],
                name=gui_trace_label("fit"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        batch_style = gui_trace_style("fit")
        for marker, name in (
            (
                {
                    "size": batch_style.get("marker", {}).get("size", 7),
                    "color": batch_style["line"]["color"],
                },
                "Batch",
            ),
            (
                {
                    "size": batch_style.get("marker", {}).get("size", 7),
                    "color": "black",
                    "symbol": "circle",
                },
                "Selected",
            ),
            (
                {
                    "size": batch_style.get("marker", {}).get("size", 7) + 1,
                    "color": "black",
                    "symbol": "circle",
                },
                "Active",
            ),
        ):
            figure.add_trace(
                go.Scatter(
                    x=np.empty((0,), dtype=np.float64),
                    y=np.empty((0,), dtype=np.float64),
                    customdata=np.empty((0,), dtype=np.int64),
                    mode="markers",
                    marker=marker,
                    name=name,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        figure.add_vline(
            x=vline,
            row=1,
            col=1,
            line_dash=cutoff_style["dash"],
            line_color=cutoff_style["color"],
            line_width=cutoff_style["width"],
        )
        figure.update_layout(
            margin={"l": 70, "r": 0, "t": 20, "b": 40},
            showlegend=False,
            uirevision=f"offset-{kind}",
        )
        figure.update_yaxes(title_text=ylabel_top, row=1, col=1)
        figure.update_yaxes(
            title_text=self._y_axis_html_label(),
            row=2,
            col=1,
        )
        figure.update_xaxes(
            title_text=xlabel,
            range=self._offset_batch_y_range(kind=batch_kind),
            row=2,
            col=1,
        )
        figure.update_yaxes(
            range=self._offset_batch_x_range(),
            row=2,
            col=1,
        )
        return figure

    def _build_sampling_offset_figure(self, *, kind: str) -> go.Figure:
        figure = go.Figure()
        fit_style = gui_trace_style("fit")
        ylabel = "<i>V</i><sub>off</sub> (mV)" if kind == "V" else (
            "<i>I</i><sub>off</sub> (nA)"
        )
        figure.add_trace(
            go.Scatter(
                x=np.empty((0,), dtype=np.float64),
                y=np.empty((0,), dtype=np.float64),
                customdata=np.empty((0,), dtype=np.int64),
                mode="markers",
                marker={
                    "size": fit_style.get("marker", {}).get("size", 7),
                    "color": fit_style["line"]["color"],
                },
                name="Batch",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=np.empty((0,), dtype=np.float64),
                y=np.empty((0,), dtype=np.float64),
                customdata=np.empty((0,), dtype=np.int64),
                mode="markers",
                marker={
                    "size": fit_style.get("marker", {}).get("size", 7),
                    "color": "black",
                    "symbol": "circle",
                },
                name="Active",
                showlegend=False,
            )
        )
        figure.update_layout(
            margin={"l": 70, "r": 0, "t": 20, "b": 40},
            xaxis_title=ylabel,
            yaxis_title=self._y_axis_html_label(),
            showlegend=False,
            uirevision=f"sampling-offset-{kind}",
        )
        figure.update_xaxes(range=self._offset_batch_y_range(kind=kind))
        figure.update_yaxes(range=self._offset_batch_x_range())
        return figure

    def _build_offset_batch_figure(self, *, kind: str) -> go.Figure:
        figure = go.Figure()
        fit_style = gui_trace_style("fit")
        ylabel = (
            "<i>V</i><sub>off</sub> (mV)"
            if kind == "V"
            else "<i>I</i><sub>off</sub> (nA)"
        )
        figure.add_trace(
            go.Scatter(
                x=np.empty((0,), dtype=np.float64),
                y=np.empty((0,), dtype=np.float64),
                customdata=np.empty((0,), dtype=np.int64),
                mode="markers",
                marker={
                    "size": fit_style.get("marker", {}).get("size", 7),
                    "color": fit_style["line"]["color"],
                },
                name="Batch",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=np.empty((0,), dtype=np.float64),
                y=np.empty((0,), dtype=np.float64),
                customdata=np.empty((0,), dtype=np.int64),
                mode="markers",
                marker={
                    "size": fit_style.get("marker", {}).get("size", 7),
                    "color": "black",
                    "symbol": "circle",
                },
                name="Selected",
                showlegend=False,
            )
        )
        figure.add_trace(
            go.Scatter(
                x=np.empty((0,), dtype=np.float64),
                y=np.empty((0,), dtype=np.float64),
                customdata=np.empty((0,), dtype=np.int64),
                mode="markers",
                marker={
                    "size": fit_style.get("marker", {}).get("size", 7) + 1,
                    "color": "black",
                    "symbol": "circle",
                },
                name="Active",
                showlegend=False,
            )
        )
        figure.update_layout(
            margin={"l": 70, "r": 0, "t": 20, "b": 40},
            xaxis_title=self._y_axis_html_label(),
            yaxis_title=ylabel,
            showlegend=False,
            uirevision=f"offset-batch-{kind}",
        )
        figure.update_xaxes(
            range=self._offset_batch_x_range(),
        )
        figure.update_yaxes(
            range=self._offset_batch_y_range(kind=kind),
        )
        return figure

    def _offset_batch_x_range(self) -> list[float]:
        x_values: list[float] = []
        for index, trace in enumerate(self.traces):
            yvalue = numeric_yvalue(trace["meta"].yvalue)
            if yvalue is None:
                x_values.append(float(index))
            else:
                x_values.append(float(yvalue))
        return self._padded_axis_range(np.asarray(x_values, dtype=np.float64))

    def _offset_batch_y_range(self, *, kind: str) -> list[float]:
        values = (
            np.asarray(self._offset_spec.Voff_mV, dtype=np.float64)
            if kind == "V"
            else np.asarray(self._offset_spec.Ioff_nA, dtype=np.float64)
        )
        return self._padded_axis_range(values)

    def _sampling_offset_plot_points(
        self,
        *,
        kind: str,
    ) -> tuple[NDArray64, NDArray64, NDArray64]:
        x_values: list[float] = []
        y_values: list[float] = []
        indices: list[int] = []
        for index, trace in enumerate(self.traces):
            if index == int(self.active_index):
                continue
            offset_values = self._sampling_display_offset_values_for_index(index)
            if offset_values is None:
                continue
            yvalue = numeric_yvalue(trace["meta"].yvalue)
            x_values.append(
                float(offset_values[0] if kind == "V" else offset_values[1])
            )
            y_values.append(float(index) if yvalue is None else float(yvalue))
            indices.append(int(index))
        return (
            np.asarray(x_values, dtype=np.float64),
            np.asarray(y_values, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
        )

    def _sampling_offset_active_point(
        self,
        *,
        kind: str,
    ) -> tuple[NDArray64, NDArray64, NDArray64]:
        offset_values = self._sampling_display_offset_values_for_index(
            self.active_index
        )
        if offset_values is None:
            empty = np.empty((0,), dtype=np.float64)
            empty_index = np.empty((0,), dtype=np.int64)
            return empty, empty.copy(), empty_index
        trace = self.traces[int(self.active_index)]
        yvalue = numeric_yvalue(trace["meta"].yvalue)
        y_axis_value = float(self.active_index) if yvalue is None else float(yvalue)
        return (
            np.asarray(
                [float(offset_values[0] if kind == "V" else offset_values[1])],
                dtype=np.float64,
            ),
            np.asarray([y_axis_value], dtype=np.float64),
            np.asarray([int(self.active_index)], dtype=np.int64),
        )

    @staticmethod
    def _padded_axis_range(values: NDArray64) -> list[float]:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if finite.size == 0:
            return [-1.0, 1.0]
        minimum = float(np.min(finite))
        maximum = float(np.max(finite))
        span = maximum - minimum
        if span <= 0.0:
            pad = max(1.0, abs(minimum) * 0.1, abs(maximum) * 0.1)
        else:
            pad = span * 0.05
        return [minimum - pad, maximum + pad]

    def _offset_batch_plot_points(
        self,
        *,
        kind: str,
    ) -> tuple[NDArray64, NDArray64, NDArray64]:
        x_values: list[float] = []
        y_values: list[float] = []
        indices: list[int] = []
        for index, trace in enumerate(self.traces):
            offset = self._get_cached_offset_batch_result(index)
            if offset is None:
                continue
            yvalue = numeric_yvalue(trace["meta"].yvalue)
            if yvalue is None:
                x_values.append(float(index))
            else:
                x_values.append(float(yvalue))
            if kind == "V":
                y_values.append(float(offset["Voff_mV"]))
            else:
                y_values.append(float(offset["Ioff_nA"]))
            indices.append(int(index))
        return (
            np.asarray(x_values, dtype=np.float64),
            np.asarray(y_values, dtype=np.float64),
            np.asarray(indices, dtype=np.int64),
        )

    def _offset_batch_selected_point(
        self,
        *,
        kind: str,
    ) -> tuple[NDArray64, NDArray64, NDArray64]:
        index = self._offset_batch_display_index
        if index is None:
            empty_x = np.empty((0,), dtype=np.float64)
            empty_index = np.empty((0,), dtype=np.int64)
            return empty_x, empty_x.copy(), empty_index
        offset = self._get_cached_offset_batch_result(int(index))
        if offset is None:
            empty_x = np.empty((0,), dtype=np.float64)
            empty_index = np.empty((0,), dtype=np.int64)
            return empty_x, empty_x.copy(), empty_index
        trace = self.traces[int(index)]
        yvalue = numeric_yvalue(trace["meta"].yvalue)
        x_value = float(int(index)) if yvalue is None else float(yvalue)
        y_value = float(offset["Voff_mV"]) if kind == "V" else float(offset["Ioff_nA"])
        return (
            np.asarray([x_value], dtype=np.float64),
            np.asarray([y_value], dtype=np.float64),
            np.asarray([int(index)], dtype=np.int64),
        )

    def _offset_batch_active_point(
        self,
        *,
        kind: str,
    ) -> tuple[NDArray64, NDArray64, NDArray64]:
        _ = kind
        empty_x = np.empty((0,), dtype=np.float64)
        empty_index = np.empty((0,), dtype=np.int64)
        return empty_x, empty_x.copy(), empty_index

    def _display_offset(self):
        return self._offset_display_result()

    def _gradient(self, x: NDArray64, y: NDArray64) -> NDArray64:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        if x.size < 2 or y.size < 2:
            return np.full(y.shape, np.nan, dtype=np.float64)
        return np.gradient(y, x)

    @staticmethod
    def _positive_log_values(values: NDArray64) -> NDArray64:
        out = np.asarray(values, dtype=np.float64).copy()
        out[out <= 0.0] = np.nan
        return out

    def _refresh_left_plots(self) -> None:
        self._update_left_view_state(
            axis_kind="V",
            relayout_data=self._iv_pane.relayout_data,
        )
        self._update_left_view_state(
            axis_kind="I",
            relayout_data=self._vi_pane.relayout_data,
        )
        self._iv_figure = self._build_left_stack_figure(axis_kind="V")
        self._vi_figure = self._build_left_stack_figure(axis_kind="I")
        self._iv_pane.object = self._iv_figure
        self._iv_pane.height = self._left_figure_height(axis_kind="V")
        self._vi_pane.object = self._vi_figure
        self._vi_pane.height = self._left_figure_height(axis_kind="I")

    def _refresh_experimental_views(self) -> None:
        trace = self._active_trace()
        raw_psd = self._require_psd()
        downsampled_psd = self._require_downsampled_psd()

        self._experimental_time_figure.data[0].x = np.asarray(
            trace["t_s"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[0].y = np.asarray(
            trace["V_mV"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[1].x = self._psd_downsampled_t_s
        self._experimental_time_figure.data[1].y = self._psd_downsampled_V_mV
        self._experimental_time_figure.data[2].x = np.asarray(
            trace["t_s"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[2].y = np.asarray(
            trace["I_nA"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[3].x = self._psd_downsampled_t_s
        self._experimental_time_figure.data[3].y = self._psd_downsampled_I_nA

        self._experimental_psd_figure.data[0].x = np.asarray(
            raw_psd["f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[0].y = self._positive_log_values(
            np.asarray(raw_psd["V_psd_mV2_per_Hz"], dtype=np.float64)
        )
        self._experimental_psd_figure.data[1].x = np.asarray(
            downsampled_psd["f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[1].y = self._positive_log_values(
            np.asarray(
                downsampled_psd["V_psd_mV2_per_Hz"],
                dtype=np.float64,
            )
        )
        self._experimental_psd_figure.data[2].x = np.asarray(
            raw_psd["f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[2].y = self._positive_log_values(
            np.asarray(raw_psd["I_psd_nA2_per_Hz"], dtype=np.float64)
        )
        self._experimental_psd_figure.data[3].x = np.asarray(
            downsampled_psd["f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[3].y = self._positive_log_values(
            np.asarray(
                downsampled_psd["I_psd_nA2_per_Hz"],
                dtype=np.float64,
            )
        )

        self._experimental_time_pane.object = self._experimental_time_figure
        self._experimental_psd_pane.object = self._experimental_psd_figure
        self._sync_psd_widgets_from_state()

    def _refresh_offset_views(self) -> None:
        offset = self._display_offset()
        self._offset_g_figure.data[0].x = np.asarray(
            self._offset_spec.Voff_mV,
            dtype=np.float64,
        )
        self._offset_g_figure.layout.shapes = ()
        if offset is None:
            self._offset_g_figure.data[0].y = np.empty((0,), dtype=np.float64)
        else:
            self._offset_g_figure.data[0].y = np.asarray(
                offset["dGerr_G0"],
                dtype=np.float64,
            )
            self._offset_g_figure.add_vline(
                x=float(offset["Voff_mV"]),
                row=1,
                col=1,
                line_dash=gui_trace_style("cutoff")["line"]["dash"],
                line_color=gui_trace_style("cutoff")["line"]["color"],
                line_width=gui_trace_style("cutoff")["line"]["width"],
            )
        x_v, y_v, indices_v = self._offset_batch_plot_points(kind="V")
        self._offset_g_figure.data[1].x = y_v
        self._offset_g_figure.data[1].y = x_v
        self._offset_g_figure.data[1].customdata = indices_v
        selected_x_v, selected_y_v, selected_indices_v = (
            self._offset_batch_selected_point(kind="V")
        )
        self._offset_g_figure.data[2].x = selected_y_v
        self._offset_g_figure.data[2].y = selected_x_v
        self._offset_g_figure.data[2].customdata = selected_indices_v
        active_x_v, active_y_v, active_indices_v = self._offset_batch_active_point(
            kind="V"
        )
        self._offset_g_figure.data[3].x = active_y_v
        self._offset_g_figure.data[3].y = active_x_v
        self._offset_g_figure.data[3].customdata = active_indices_v
        self._offset_g_figure.update_yaxes(
            title_text=self._y_axis_html_label(),
            row=2,
            col=1,
        )

        self._offset_r_figure.data[0].x = np.asarray(
            self._offset_spec.Ioff_nA,
            dtype=np.float64,
        )
        self._offset_r_figure.layout.shapes = ()
        if offset is None:
            self._offset_r_figure.data[0].y = np.empty((0,), dtype=np.float64)
        else:
            self._offset_r_figure.data[0].y = np.asarray(
                offset["dRerr_R0"],
                dtype=np.float64,
            )
            self._offset_r_figure.add_vline(
                x=float(offset["Ioff_nA"]),
                row=1,
                col=1,
                line_dash=gui_trace_style("cutoff")["line"]["dash"],
                line_color=gui_trace_style("cutoff")["line"]["color"],
                line_width=gui_trace_style("cutoff")["line"]["width"],
            )
        x_i, y_i, indices_i = self._offset_batch_plot_points(kind="I")
        self._offset_r_figure.data[1].x = y_i
        self._offset_r_figure.data[1].y = x_i
        self._offset_r_figure.data[1].customdata = indices_i
        selected_x_i, selected_y_i, selected_indices_i = (
            self._offset_batch_selected_point(kind="I")
        )
        self._offset_r_figure.data[2].x = selected_y_i
        self._offset_r_figure.data[2].y = selected_x_i
        self._offset_r_figure.data[2].customdata = selected_indices_i
        active_x_i, active_y_i, active_indices_i = self._offset_batch_active_point(
            kind="I"
        )
        self._offset_r_figure.data[3].x = active_y_i
        self._offset_r_figure.data[3].y = active_x_i
        self._offset_r_figure.data[3].customdata = active_indices_i
        self._offset_r_figure.update_yaxes(
            title_text=self._y_axis_html_label(),
            row=2,
            col=1,
        )

        self._offset_g_pane.object = self._offset_g_figure
        self._offset_r_pane.object = self._offset_r_figure
        self._offset_info_table.value = self._offset_info_frame()

    def _refresh_offset_batch_views(self) -> None:
        x_v, y_v, indices_v = self._offset_batch_plot_points(kind="V")
        with self._offset_batch_v_figure.batch_update():
            self._offset_batch_v_figure.data[0].x = x_v
            self._offset_batch_v_figure.data[0].y = y_v
            self._offset_batch_v_figure.data[0].customdata = indices_v
            selected_x_v, selected_y_v, selected_indices_v = (
                self._offset_batch_selected_point(kind="V")
            )
            self._offset_batch_v_figure.data[1].x = selected_x_v
            self._offset_batch_v_figure.data[1].y = selected_y_v
            self._offset_batch_v_figure.data[1].customdata = selected_indices_v
            active_x_v, active_y_v, active_indices_v = self._offset_batch_active_point(
                kind="V"
            )
            self._offset_batch_v_figure.data[2].x = active_x_v
            self._offset_batch_v_figure.data[2].y = active_y_v
            self._offset_batch_v_figure.data[2].customdata = active_indices_v
            self._offset_batch_v_figure.update_xaxes(
                title_text=self._y_axis_html_label(),
            )

        x_i, y_i, indices_i = self._offset_batch_plot_points(kind="I")
        with self._offset_batch_i_figure.batch_update():
            self._offset_batch_i_figure.data[0].x = x_i
            self._offset_batch_i_figure.data[0].y = y_i
            self._offset_batch_i_figure.data[0].customdata = indices_i
            selected_x_i, selected_y_i, selected_indices_i = (
                self._offset_batch_selected_point(kind="I")
            )
            self._offset_batch_i_figure.data[1].x = selected_x_i
            self._offset_batch_i_figure.data[1].y = selected_y_i
            self._offset_batch_i_figure.data[1].customdata = selected_indices_i
            active_x_i, active_y_i, active_indices_i = self._offset_batch_active_point(
                kind="I"
            )
            self._offset_batch_i_figure.data[2].x = active_x_i
            self._offset_batch_i_figure.data[2].y = active_y_i
            self._offset_batch_i_figure.data[2].customdata = active_indices_i
            self._offset_batch_i_figure.update_xaxes(
                title_text=self._y_axis_html_label(),
            )

        self._offset_batch_v_pane.object = self._offset_batch_v_figure
        self._offset_batch_i_pane.object = self._offset_batch_i_figure
        self._offset_batch_table.value = self._offset_batch_frame()
        self._refresh_offset_views()

    def _refresh_sampling_views(self) -> None:
        x_v, y_v, indices_v = self._sampling_offset_plot_points(kind="V")
        active_x_v, active_y_v, active_indices_v = (
            self._sampling_offset_active_point(kind="V")
        )
        with self._sampling_offset_v_figure.batch_update():
            self._sampling_offset_v_figure.data[0].x = x_v
            self._sampling_offset_v_figure.data[0].y = y_v
            self._sampling_offset_v_figure.data[0].customdata = indices_v
            self._sampling_offset_v_figure.data[1].x = active_x_v
            self._sampling_offset_v_figure.data[1].y = active_y_v
            self._sampling_offset_v_figure.data[1].customdata = active_indices_v
            self._sampling_offset_v_figure.update_xaxes(
                range=self._offset_batch_y_range(kind="V"),
            )
            self._sampling_offset_v_figure.update_yaxes(
                title_text=self._y_axis_html_label(),
                range=self._offset_batch_x_range(),
            )

        x_i, y_i, indices_i = self._sampling_offset_plot_points(kind="I")
        active_x_i, active_y_i, active_indices_i = (
            self._sampling_offset_active_point(kind="I")
        )
        with self._sampling_offset_i_figure.batch_update():
            self._sampling_offset_i_figure.data[0].x = x_i
            self._sampling_offset_i_figure.data[0].y = y_i
            self._sampling_offset_i_figure.data[0].customdata = indices_i
            self._sampling_offset_i_figure.data[1].x = active_x_i
            self._sampling_offset_i_figure.data[1].y = active_y_i
            self._sampling_offset_i_figure.data[1].customdata = active_indices_i
            self._sampling_offset_i_figure.update_xaxes(
                range=self._offset_batch_y_range(kind="I"),
            )
            self._sampling_offset_i_figure.update_yaxes(
                title_text=self._y_axis_html_label(),
                range=self._offset_batch_x_range(),
            )

        self._sampling_offset_v_pane.object = self._sampling_offset_v_figure
        self._sampling_offset_i_pane.object = self._sampling_offset_i_figure
