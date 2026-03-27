from __future__ import annotations

import numpy as np
import pandas as pd
from panel.io.model import JSCode

from ...evaluation.offset import OffsetSpec
from ..state import _linspace_from_values

_OFFSET_GRID_PARAMETER_LABELS = {
    "Vbins_mV": "<i>V</i><sub>bins</sub> (mV)",
    "Ibins_nA": "<i>I</i><sub>bins</sub> (nA)",
    "Voff_mV": "<i>V</i><sub>off</sub> (mV)",
    "Ioff_nA": "<i>I</i><sub>off</sub> (nA)",
}
_OFFSET_GRID_TITLES = {
    "parameter": "Parameter",
    "start": "Start",
    "stop": "Stop",
    "count": "Count",
}
_OFFSET_INFO_TITLES = {
    "parameter": "Parameter",
    "value": "Value",
}
_OFFSET_INFO_PARAMETER_LABELS = {
    "nu_Hz": "<i>&nu;</i> (Hz)",
    "upsample": "<i>N</i><sub>up</sub>",
    "Voff_mV": "<i>V</i><sub>off</sub> (mV)",
    "Ioff_nA": "<i>I</i><sub>off</sub> (nA)",
}


class GUIOffsetTabMixin:
    def _build_offset_widgets(self) -> None:
        self._offset_grid_table = self._pn.widgets.Tabulator(
            self._offset_grid_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=180,
            editors={
                "parameter": None,
                "start": {"type": "number"},
                "stop": {"type": "number"},
                "count": {"type": "number", "step": 1},
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_OFFSET_GRID_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _OFFSET_GRID_TITLES
            },
        )
        self._offset_info_table = self._pn.widgets.Tabulator(
            self._offset_info_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=320,
            height=145,
            editors={
                "parameter": None,
                "value": {"type": "number"},
            },
            editables={
                "value": JSCode(
                    "function(cell) { "
                    "const key = cell.getData().key; "
                    "return key === 'nu_Hz' || key === 'upsample'; "
                    "}"
                )
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_OFFSET_INFO_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _OFFSET_INFO_TITLES
            },
        )
        self._offset_apply_button = self._pn.widgets.Button(
            name="Offset Analysis",
            button_type="primary",
        )
        self._offset_apply_button.on_click(self._on_offset_apply)

    def _offset_tab(self):
        return self._pn.Column(
            self._offset_apply_button,
            self._offset_grid_table,
            self._pn.Row(
                self._offset_info_table,
                self._pn.Column(
                    self._offset_g_pane,
                    self._offset_r_pane,
                    sizing_mode="stretch_width",
                ),
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def _offset_grid_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Vbins_mV"],
                    "start": float(self._offset_spec.Vbins_mV[0]),
                    "stop": float(self._offset_spec.Vbins_mV[-1]),
                    "count": int(self._offset_spec.Vbins_mV.size),
                },
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Ibins_nA"],
                    "start": float(self._offset_spec.Ibins_nA[0]),
                    "stop": float(self._offset_spec.Ibins_nA[-1]),
                    "count": int(self._offset_spec.Ibins_nA.size),
                },
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Voff_mV"],
                    "start": float(self._offset_spec.Voff_mV[0]),
                    "stop": float(self._offset_spec.Voff_mV[-1]),
                    "count": int(self._offset_spec.Voff_mV.size),
                },
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Ioff_nA"],
                    "start": float(self._offset_spec.Ioff_nA[0]),
                    "stop": float(self._offset_spec.Ioff_nA[-1]),
                    "count": int(self._offset_spec.Ioff_nA.size),
                },
            ]
        )
        frame["count"] = frame["count"].astype(np.int64)
        return frame

    def _offset_info_frame(self) -> pd.DataFrame:
        Voff_mV = np.nan
        Ioff_nA = np.nan
        if self._offset is not None:
            Voff_mV = float(self._offset["Voff_mV"])
            Ioff_nA = float(self._offset["Ioff_nA"])
        return pd.DataFrame(
            [
                {
                    "key": "nu_Hz",
                    "parameter": _OFFSET_INFO_PARAMETER_LABELS["nu_Hz"],
                    "value": float(self._offset_spec.nu_Hz),
                },
                {
                    "key": "upsample",
                    "parameter": _OFFSET_INFO_PARAMETER_LABELS["upsample"],
                    "value": int(self._offset_spec.upsample),
                },
                {
                    "key": "Voff_mV",
                    "parameter": _OFFSET_INFO_PARAMETER_LABELS["Voff_mV"],
                    "value": Voff_mV,
                },
                {
                    "key": "Ioff_nA",
                    "parameter": _OFFSET_INFO_PARAMETER_LABELS["Ioff_nA"],
                    "value": Ioff_nA,
                },
            ],
            dtype=object,
        )

    def _build_offset_spec_from_table(self) -> OffsetSpec:
        grid_frame = self._offset_grid_table.value.reset_index(drop=True)
        info_frame = (
            self._offset_info_table.value.reset_index(drop=True).set_index("key")
        )
        return OffsetSpec(
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
            Voff_mV=_linspace_from_values(
                grid_frame.at[2, "start"],
                grid_frame.at[2, "stop"],
                grid_frame.at[2, "count"],
                name="Voff_mV",
                min_count=1,
            ),
            Ioff_nA=_linspace_from_values(
                grid_frame.at[3, "start"],
                grid_frame.at[3, "stop"],
                grid_frame.at[3, "count"],
                name="Ioff_nA",
                min_count=1,
            ),
            nu_Hz=float(info_frame.at["nu_Hz", "value"]),
            upsample=int(info_frame.at["upsample", "value"]),
        )

    def _sync_offset_widgets_from_spec(self) -> None:
        self._offset_grid_table.value = self._offset_grid_frame()
        self._offset_info_table.value = self._offset_info_frame()

    def _on_offset_apply(self, _: object) -> None:
        offset_spec = self._build_offset_spec_from_table()
        self._offset_spec = offset_spec
        self._recompute_pipeline(
            clear_fit=True,
            recompute_psd=False,
            recompute_offset=True,
            recompute_sampling=True,
        )
        self._sync_control_widgets_from_specs()
        self._refresh_all_views()
        self._notify_state_changed()
