from __future__ import annotations

from collections import OrderedDict

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .style import gui_trace_label, gui_trace_style
from ..utilities.constants import G_0_muS
from ..utilities.types import NDArray64

_LEFT_STAGE_BUTTON_ORDER = (
    "raw",
    "downsampled",
    "binned",
    "initial",
    "fit",
)
_LEFT_STAGE_TRACE_ORDER = _LEFT_STAGE_BUTTON_ORDER
_LEFT_STAGE_DEFAULTS = ["binned", "initial", "fit"]


class GUILeftMixin:
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

    def _on_left_stage_changed(self, _: object) -> None:
        self._refresh_left_plots()

    def _build_plot_panes(self) -> None:
        self._iv_figure = self._build_iv_figure()
        self._didv_figure = self._build_didv_figure()
        self._vi_figure = self._build_vi_figure()
        self._dvdi_figure = self._build_dvdi_figure()

        self._experimental_time_figure = self._build_experimental_time_figure()
        self._experimental_psd_figure = self._build_experimental_psd_figure()

        self._offset_g_figure = self._build_offset_figure(kind="G")
        self._offset_r_figure = self._build_offset_figure(kind="R")

        self._sampling_iv_figure = self._build_sampling_preview(kind="IV")
        self._sampling_vi_figure = self._build_sampling_preview(kind="VI")

        self._iv_pane = self._pn.pane.Plotly(
            self._iv_figure,
            sizing_mode="stretch_width",
            height=320,
            config={"responsive": True},
        )
        self._didv_pane = self._pn.pane.Plotly(
            self._didv_figure,
            sizing_mode="stretch_width",
            height=320,
            config={"responsive": True},
        )
        self._vi_pane = self._pn.pane.Plotly(
            self._vi_figure,
            sizing_mode="stretch_width",
            height=320,
            config={"responsive": True},
        )
        self._dvdi_pane = self._pn.pane.Plotly(
            self._dvdi_figure,
            sizing_mode="stretch_width",
            height=320,
            config={"responsive": True},
        )
        self._experimental_time_pane = self._pn.pane.Plotly(
            self._experimental_time_figure,
            sizing_mode="stretch_width",
            height=420,
            config={"responsive": True},
        )
        self._experimental_psd_pane = self._pn.pane.Plotly(
            self._experimental_psd_figure,
            sizing_mode="stretch_width",
            height=420,
            config={"responsive": True},
        )
        self._offset_g_pane = self._pn.pane.Plotly(
            self._offset_g_figure,
            sizing_mode="stretch_width",
            height=260,
            config={"responsive": True},
        )
        self._offset_r_pane = self._pn.pane.Plotly(
            self._offset_r_figure,
            sizing_mode="stretch_width",
            height=260,
            config={"responsive": True},
        )
        self._sampling_iv_pane = self._pn.pane.Plotly(
            self._sampling_iv_figure,
            sizing_mode="stretch_width",
            height=260,
            config={"responsive": True},
        )
        self._sampling_vi_pane = self._pn.pane.Plotly(
            self._sampling_vi_figure,
            sizing_mode="stretch_width",
            height=260,
            config={"responsive": True},
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
        return stage in self._left_stage_selector.value

    def _left_iv_stage_data(self, stage: str) -> tuple[NDArray64, NDArray64]:
        trace = self._active_trace()
        sampling = self._require_sampling()
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
        if stage == "binned":
            return (
                np.asarray(sampling["Vbin_mV"], dtype=np.float64),
                np.asarray(sampling["I_nA"], dtype=np.float64),
            )
        if stage == "initial":
            return (
                np.asarray(sampling["Vbin_mV"], dtype=np.float64),
                np.asarray(self._initial_curve, dtype=np.float64),
            )
        return (
            np.asarray(sampling["Vbin_mV"], dtype=np.float64),
            np.asarray(self._fit_curve, dtype=np.float64),
        )

    def _left_didv_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        sampling = self._require_sampling()
        if stage == "binned":
            return (
                np.asarray(sampling["Vbin_mV"], dtype=np.float64),
                np.asarray(sampling["dG_G0"], dtype=np.float64),
            )
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

    def _left_vi_stage_data(self, stage: str) -> tuple[NDArray64, NDArray64]:
        sampling = self._require_sampling()
        if stage == "binned":
            return (
                np.asarray(sampling["Ibin_nA"], dtype=np.float64),
                np.asarray(sampling["V_mV"], dtype=np.float64),
            )
        V_mV, I_nA = self._left_iv_stage_data(stage)
        return self._sorted_unique_curve(I_nA, V_mV)

    def _left_dvdi_stage_data(
        self,
        stage: str,
    ) -> tuple[NDArray64, NDArray64]:
        sampling = self._require_sampling()
        if stage == "binned":
            return (
                np.asarray(sampling["Ibin_nA"], dtype=np.float64),
                np.asarray(sampling["dR_R0"], dtype=np.float64),
            )
        I_nA, V_mV = self._left_vi_stage_data(stage)
        return I_nA, self._gradient(I_nA, V_mV) * G_0_muS

    def _build_iv_figure(self) -> go.Figure:
        figure = go.Figure()
        for stage in _LEFT_STAGE_TRACE_ORDER:
            figure.add_trace(self._left_trace(stage))
        figure.update_layout(
            margin={"l": 80, "r": 20, "t": 20, "b": 20},
            yaxis_title="I (nA)",
            showlegend=False,
        )
        figure.update_xaxes(showticklabels=False)
        return figure


    def _build_didv_figure(self) -> go.Figure:
        figure = go.Figure()
        for stage in _LEFT_STAGE_TRACE_ORDER:
            figure.add_trace(self._left_trace(stage))
        figure.update_layout(
            margin={"l": 80, "r": 20, "t": 20, "b": 40},
            xaxis_title="V (mV)",
            yaxis_title="dI/dV (G/G0)",
            showlegend=False,
        )
        return figure


    def _build_vi_figure(self) -> go.Figure:
        figure = go.Figure()
        for stage in _LEFT_STAGE_TRACE_ORDER:
            figure.add_trace(self._left_trace(stage))
        figure.update_layout(
            margin={"l": 80, "r": 20, "t": 20, "b": 20},
            yaxis_title="V (mV)",
            showlegend=False,
        )
        figure.update_xaxes(showticklabels=False)
        return figure


    def _build_dvdi_figure(self) -> go.Figure:
        figure = go.Figure()
        for stage in _LEFT_STAGE_TRACE_ORDER:
            figure.add_trace(self._left_trace(stage))
        figure.update_layout(
            margin={"l": 80, "r": 20, "t": 20, "b": 40},
            xaxis_title="I (nA)",
            yaxis_title="dV/dI (R/R0)",
            showlegend=False,
        )
        return figure


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
                x=self._downsampled_t_s,
                y=self._downsampled_V_mV,
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
                x=self._downsampled_t_s,
                y=self._downsampled_I_nA,
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
        psd = self._require_psd()
        raw_style = gui_trace_style("raw")
        downsampled_style = gui_trace_style("downsampled")
        cutoff_style = gui_trace_style("cutoff")["line"]
        figure = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
        )
        figure.add_trace(
            go.Scatter(
                x=np.asarray(psd["raw_f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(
                        psd["raw_V_psd_mV2_per_Hz"],
                        dtype=np.float64,
                    )
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
                x=np.asarray(psd["downsampled_f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(
                        psd["downsampled_V_psd_mV2_per_Hz"],
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
                x=np.asarray(psd["raw_f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(
                        psd["raw_I_psd_nA2_per_Hz"],
                        dtype=np.float64,
                    )
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
                x=np.asarray(psd["downsampled_f_Hz"], dtype=np.float64),
                y=self._positive_log_values(
                    np.asarray(
                        psd["downsampled_I_psd_nA2_per_Hz"],
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
        figure.add_vline(
            x=float(psd["downsampled_sigma_cutoff_Hz"]),
            line_dash=cutoff_style["dash"],
            line_color=cutoff_style["color"],
            line_width=cutoff_style["width"],
            row=1,
            col=1,
        )
        figure.add_vline(
            x=float(psd["downsampled_sigma_cutoff_Hz"]),
            line_dash=cutoff_style["dash"],
            line_color=cutoff_style["color"],
            line_width=cutoff_style["width"],
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
        figure = go.Figure()
        offset = self._require_offset()
        fit_style = gui_trace_style("fit")
        cutoff_style = gui_trace_style("cutoff")["line"]
        if kind == "G":
            x = np.asarray(self._offset_spec.Voff_mV, dtype=np.float64)
            y = np.asarray(offset["dGerr_G0"], dtype=np.float64)
            vline = float(offset["Voff_mV"])
            xlabel = "<i>V</i><sub>off</sub> (mV)"
            ylabel = "<i>dG</i><sub>err</sub> / <i>G</i><sub>0</sub>"
        else:
            x = np.asarray(self._offset_spec.Ioff_nA, dtype=np.float64)
            y = np.asarray(offset["dRerr_R0"], dtype=np.float64)
            vline = float(offset["Ioff_nA"])
            xlabel = "<i>I</i><sub>off</sub> (nA)"
            ylabel = "<i>dR</i><sub>err</sub> / <i>R</i><sub>0</sub>"
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=fit_style["mode"],
                line=fit_style["line"],
                name=gui_trace_label("fit"),
            )
        )
        figure.add_vline(
            x=vline,
            line_dash=cutoff_style["dash"],
            line_color=cutoff_style["color"],
            line_width=cutoff_style["width"],
        )
        figure.update_layout(
            margin={"l": 70, "r": 20, "t": 20, "b": 40},
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=False,
        )
        return figure


    def _build_sampling_preview(self, *, kind: str) -> go.Figure:
        figure = go.Figure()
        sampling = self._require_sampling()
        binned_style = gui_trace_style("binned")
        if kind == "IV":
            x = np.asarray(sampling["Vbin_mV"], dtype=np.float64)
            y = np.asarray(sampling["I_nA"], dtype=np.float64)
            xlabel = "<i>V</i> (mV)"
            ylabel = "<i>I</i> (nA)"
        else:
            x = np.asarray(sampling["Ibin_nA"], dtype=np.float64)
            y = np.asarray(sampling["V_mV"], dtype=np.float64)
            xlabel = "<i>I</i> (nA)"
            ylabel = "<i>V</i> (mV)"
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode=binned_style["mode"],
                marker=binned_style.get("marker"),
                line=binned_style["line"],
                name=gui_trace_label("binned"),
            )
        )
        figure.update_layout(
            margin={"l": 70, "r": 20, "t": 20, "b": 40},
            xaxis_title=xlabel,
            yaxis_title=ylabel,
            showlegend=False,
        )
        return figure


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
        for index, stage in enumerate(_LEFT_STAGE_TRACE_ORDER):
            visible = self._left_trace_visible(stage)
            x, y = self._left_iv_stage_data(stage)
            self._iv_figure.data[index].x = x
            self._iv_figure.data[index].y = y
            self._iv_figure.data[index].visible = visible

            x, y = self._left_didv_stage_data(stage)
            self._didv_figure.data[index].x = x
            self._didv_figure.data[index].y = y
            self._didv_figure.data[index].visible = visible

            x, y = self._left_vi_stage_data(stage)
            self._vi_figure.data[index].x = x
            self._vi_figure.data[index].y = y
            self._vi_figure.data[index].visible = visible

            x, y = self._left_dvdi_stage_data(stage)
            self._dvdi_figure.data[index].x = x
            self._dvdi_figure.data[index].y = y
            self._dvdi_figure.data[index].visible = visible

        self._iv_pane.object = self._iv_figure
        self._didv_pane.object = self._didv_figure
        self._vi_pane.object = self._vi_figure
        self._dvdi_pane.object = self._dvdi_figure


    def _refresh_experimental_views(self) -> None:
        trace = self._active_trace()
        psd = self._require_psd()

        self._experimental_time_figure.data[0].x = np.asarray(
            trace["t_s"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[0].y = np.asarray(
            trace["V_mV"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[1].x = self._downsampled_t_s
        self._experimental_time_figure.data[1].y = self._downsampled_V_mV
        self._experimental_time_figure.data[2].x = np.asarray(
            trace["t_s"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[2].y = np.asarray(
            trace["I_nA"],
            dtype=np.float64,
        )
        self._experimental_time_figure.data[3].x = self._downsampled_t_s
        self._experimental_time_figure.data[3].y = self._downsampled_I_nA

        self._experimental_psd_figure.data[0].x = np.asarray(
            psd["raw_f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[0].y = self._positive_log_values(
            np.asarray(psd["raw_V_psd_mV2_per_Hz"], dtype=np.float64)
        )
        self._experimental_psd_figure.data[1].x = np.asarray(
            psd["downsampled_f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[1].y = self._positive_log_values(
            np.asarray(
                psd["downsampled_V_psd_mV2_per_Hz"],
                dtype=np.float64,
            )
        )
        self._experimental_psd_figure.data[2].x = np.asarray(
            psd["raw_f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[2].y = self._positive_log_values(
            np.asarray(psd["raw_I_psd_nA2_per_Hz"], dtype=np.float64)
        )
        self._experimental_psd_figure.data[3].x = np.asarray(
            psd["downsampled_f_Hz"],
            dtype=np.float64,
        )
        self._experimental_psd_figure.data[3].y = self._positive_log_values(
            np.asarray(
                psd["downsampled_I_psd_nA2_per_Hz"],
                dtype=np.float64,
            )
        )
        self._experimental_psd_figure.layout.shapes = ()
        self._experimental_psd_figure.add_vline(
            x=float(psd["downsampled_sigma_cutoff_Hz"]),
            line_dash=gui_trace_style("cutoff")["line"]["dash"],
            line_color=gui_trace_style("cutoff")["line"]["color"],
            line_width=gui_trace_style("cutoff")["line"]["width"],
            row=1,
            col=1,
        )
        self._experimental_psd_figure.add_vline(
            x=float(psd["downsampled_sigma_cutoff_Hz"]),
            line_dash=gui_trace_style("cutoff")["line"]["dash"],
            line_color=gui_trace_style("cutoff")["line"]["color"],
            line_width=gui_trace_style("cutoff")["line"]["width"],
            row=2,
            col=1,
        )

        self._experimental_time_pane.object = self._experimental_time_figure
        self._experimental_psd_pane.object = self._experimental_psd_figure
        self._experimental_table.value = self._experimental_frame()


    def _refresh_offset_views(self) -> None:
        offset = self._require_offset()
        self._offset_g_figure.data[0].x = np.asarray(
            self._offset_spec.Voff_mV,
            dtype=np.float64,
        )
        self._offset_g_figure.data[0].y = np.asarray(
            offset["dGerr_G0"],
            dtype=np.float64,
        )
        self._offset_g_figure.layout.shapes = ()
        self._offset_g_figure.add_vline(
            x=float(offset["Voff_mV"]),
            line_dash=gui_trace_style("cutoff")["line"]["dash"],
            line_color=gui_trace_style("cutoff")["line"]["color"],
            line_width=gui_trace_style("cutoff")["line"]["width"],
        )

        self._offset_r_figure.data[0].x = np.asarray(
            self._offset_spec.Ioff_nA,
            dtype=np.float64,
        )
        self._offset_r_figure.data[0].y = np.asarray(
            offset["dRerr_R0"],
            dtype=np.float64,
        )
        self._offset_r_figure.layout.shapes = ()
        self._offset_r_figure.add_vline(
            x=float(offset["Ioff_nA"]),
            line_dash=gui_trace_style("cutoff")["line"]["dash"],
            line_color=gui_trace_style("cutoff")["line"]["color"],
            line_width=gui_trace_style("cutoff")["line"]["width"],
        )

        self._offset_g_pane.object = self._offset_g_figure
        self._offset_r_pane.object = self._offset_r_figure
        self._offset_info_table.value = self._offset_info_frame()


    def _refresh_sampling_views(self) -> None:
        sampling = self._require_sampling()
        self._sampling_iv_figure.data[0].x = np.asarray(
            sampling["Vbin_mV"],
            dtype=np.float64,
        )
        self._sampling_iv_figure.data[0].y = np.asarray(
            sampling["I_nA"],
            dtype=np.float64,
        )
        self._sampling_vi_figure.data[0].x = np.asarray(
            sampling["Ibin_nA"],
            dtype=np.float64,
        )
        self._sampling_vi_figure.data[0].y = np.asarray(
            sampling["V_mV"],
            dtype=np.float64,
        )
        self._sampling_iv_pane.object = self._sampling_iv_figure
        self._sampling_vi_pane.object = self._sampling_vi_figure
