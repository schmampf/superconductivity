from __future__ import annotations

import numpy as np
import pandas as pd
from panel.io.model import JSCode

from ...evaluation.sampling import SamplingSpec
from ...evaluation.sampling import SmoothingSpec
from ..state import _linspace_from_values

_SAMPLING_GRID_PARAMETER_LABELS = {
    "Vbins_mV": "<i>V</i><sub>bins</sub> (mV)",
    "Ibins_nA": "<i>I</i><sub>bins</sub> (nA)",
}
_SAMPLING_GRID_TITLES = {
    "parameter": "Parameter",
    "start": "Start",
    "stop": "Stop",
    "count": "Count",
}
_SAMPLING_INFO_TITLES = {
    "parameter": "Parameter",
    "value": "Value",
}
_SAMPLING_INFO_PARAMETER_LABELS = {
    "upsample": "<i>N</i><sub>up</sub>",
    "median_bins": "<i>N</i><sub>med</sub>",
    "sigma_bins": "<i>&sigma;</i><sub>bins</sub>",
}
_SAMPLING_OFFSET_PARAMETER_LABELS = {
    "Voff_mV": "<i>V</i><sub>off</sub> (mV)",
    "Ioff_nA": "<i>I</i><sub>off</sub> (nA)",
}


class GUISamplingTabMixin:
    def _build_sampling_widgets(self) -> None:
        self._sampling_grid_table = self._pn.widgets.Tabulator(
            self._sampling_grid_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=120,
            editors={
                "parameter": None,
                "start": {"type": "number"},
                "stop": {"type": "number"},
                "count": {"type": "number", "step": 1},
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_SAMPLING_GRID_TITLES,
            title_formatters={key: {"type": "html"} for key in _SAMPLING_GRID_TITLES},
        )
        self._sampling_info_table = self._pn.widgets.Tabulator(
            self._sampling_info_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_data_fill",
            sizing_mode="fixed",
            width=320,
            height=145,
            widths={
                "parameter": 150,
            },
            editors={
                "parameter": None,
                "value": {"type": "number"},
            },
            editables={
                "value": JSCode(
                    "function(cell) { " "return cell.getData().key === 'upsample'; " "}"
                )
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_SAMPLING_INFO_TITLES,
            title_formatters={key: {"type": "html"} for key in _SAMPLING_INFO_TITLES},
        )
        self._sampling_smoothing_table = self._pn.widgets.Tabulator(
            self._sampling_smoothing_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_data_fill",
            sizing_mode="fixed",
            width=320,
            height=110,
            widths={
                "parameter": 150,
            },
            editors={
                "parameter": None,
                "value": {"type": "number"},
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_SAMPLING_INFO_TITLES,
            title_formatters={key: {"type": "html"} for key in _SAMPLING_INFO_TITLES},
        )
        self._sampling_smooth_toggle = self._pn.widgets.Checkbox(
            name="Smooth",
            value=bool(self._smoothing_enabled),
            sizing_mode="stretch_width",
        )
        self._sampling_apply_button = self._pn.widgets.Button(
            name="Sampling",
            button_type="primary",
        )
        self._sampling_apply_button.on_click(self._on_sampling_apply)

    def _sampling_tab(self):
        return self._pn.Column(
            self._sampling_apply_button,
            self._sampling_grid_table,
            self._pn.Row(
                self._pn.Column(
                    self._sampling_info_table,
                    self._sampling_smooth_toggle,
                    self._sampling_smoothing_table,
                    width=320,
                    sizing_mode="fixed",
                ),
                self._pn.Column(
                    self._sampling_iv_pane,
                    self._sampling_vi_pane,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def _sampling_grid_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "parameter": _SAMPLING_GRID_PARAMETER_LABELS["Vbins_mV"],
                    "start": float(self._sampling_spec.Vbins_mV[0]),
                    "stop": float(self._sampling_spec.Vbins_mV[-1]),
                    "count": int(self._sampling_spec.Vbins_mV.size),
                },
                {
                    "parameter": _SAMPLING_GRID_PARAMETER_LABELS["Ibins_nA"],
                    "start": float(self._sampling_spec.Ibins_nA[0]),
                    "stop": float(self._sampling_spec.Ibins_nA[-1]),
                    "count": int(self._sampling_spec.Ibins_nA.size),
                },
            ]
        )
        frame["count"] = frame["count"].astype(np.int64)
        return frame

    def _sampling_info_frame(self) -> pd.DataFrame:
        Voff_mV, Ioff_nA = self._active_sampling_offset_values()
        return pd.DataFrame(
            [
                {
                    "key": "upsample",
                    "parameter": _SAMPLING_INFO_PARAMETER_LABELS["upsample"],
                    "value": int(self._sampling_spec.upsample),
                },
                {
                    "key": "Voff_mV",
                    "parameter": _SAMPLING_OFFSET_PARAMETER_LABELS["Voff_mV"],
                    "value": float(Voff_mV),
                },
                {
                    "key": "Ioff_nA",
                    "parameter": _SAMPLING_OFFSET_PARAMETER_LABELS["Ioff_nA"],
                    "value": float(Ioff_nA),
                },
            ],
            dtype=object,
        )

    def _sampling_smoothing_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "key": "median_bins",
                    "parameter": _SAMPLING_INFO_PARAMETER_LABELS["median_bins"],
                    "value": int(self._smoothing_spec.median_bins),
                },
                {
                    "key": "sigma_bins",
                    "parameter": _SAMPLING_INFO_PARAMETER_LABELS["sigma_bins"],
                    "value": float(self._smoothing_spec.sigma_bins),
                },
            ],
            dtype=object,
        )

    def _build_sampling_spec_from_tables(self) -> SamplingSpec:
        grid_frame = self._sampling_grid_table.value.reset_index(drop=True)
        info_frame = self._sampling_info_table.value.reset_index(drop=True).set_index(
            "key"
        )
        return SamplingSpec(
            upsample=int(info_frame.at["upsample", "value"]),
            Vbins_mV=_linspace_from_values(
                grid_frame.at[0, "start"],
                grid_frame.at[0, "stop"],
                grid_frame.at[0, "count"],
                name="Vbins_mV",
                min_count=2,
            ),
            Ibins_nA=_linspace_from_values(
                grid_frame.at[1, "start"],
                grid_frame.at[1, "stop"],
                grid_frame.at[1, "count"],
                name="Ibins_nA",
                min_count=2,
            ),
        )

    def _build_smoothing_state_from_tables(
        self,
    ) -> SmoothingSpec:
        smoothing_frame = self._sampling_smoothing_table.value.reset_index(
            drop=True
        ).set_index("key")
        return SmoothingSpec(
            median_bins=int(smoothing_frame.at["median_bins", "value"]),
            sigma_bins=float(smoothing_frame.at["sigma_bins", "value"]),
            mode=self._smoothing_spec.mode,
        )

    def _sync_sampling_widgets_from_spec(self) -> None:
        self._sampling_smooth_toggle.value = bool(self._smoothing_enabled)
        self._sampling_grid_table.value = self._sampling_grid_frame()
        self._sampling_info_table.value = self._sampling_info_frame()
        self._sampling_smoothing_table.value = self._sampling_smoothing_frame()

    def _on_sampling_apply(self, _: object) -> None:
        new_sampling_spec = self._build_sampling_spec_from_tables()
        new_smoothing_spec = self._build_smoothing_state_from_tables()
        new_smoothing_enabled = bool(self._sampling_smooth_toggle.value)
        previous_sampling_spec = self._sampling_spec
        previous_smoothing_enabled = self._smoothing_enabled
        previous_smoothing_spec = self._smoothing_spec
        self._sampling_spec = new_sampling_spec
        self._smoothing_spec = new_smoothing_spec
        self._smoothing_enabled = new_smoothing_enabled
        if (
            not self._sampling_specs_match(previous_sampling_spec, self._sampling_spec)
            or previous_smoothing_enabled != self._smoothing_enabled
            or (
                self._smoothing_enabled
                and not self._smoothing_specs_match(
                    previous_smoothing_spec,
                    self._smoothing_spec,
                )
            )
        ):
            self._clear_sampling_stage_cache()
        self._recompute_pipeline(
            clear_fit=True,
            recompute_psd=False,
            recompute_offset=False,
            recompute_sampling=True,
        )
        self._stage_sampling_result(self.active_index, self._require_sampling())
        self._sync_control_widgets_from_specs()
        self._refresh_all_views()
        self._notify_state_changed()
