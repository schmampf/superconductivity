from __future__ import annotations

import numpy as np
import pandas as pd
from panel.io.model import JSCode

from ..style import gui_legend_html
from ..state import _EXPERIMENTAL_TITLES
_PSD_INFO_TITLES = {
    "parameter": "Parameter",
    "value": "Value",
}


class GUIPSDTabMixin:
    def _build_psd_widgets(self) -> None:
        self._experimental_table = self._pn.widgets.Tabulator(
            self._experimental_frame(),
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
                "value": "adaptable",
            },
            editables={
                "value": JSCode(
                    "function(cell) { "
                    "const key = cell.getData().key; "
                    "return key === 'nu_Hz' || key === 'detrend'; "
                    "}"
                )
            },
            formatters={
                "parameter": {"type": "html"},
                "value": JSCode(
                    "function(cell) { "
                    "const value = cell.getValue(); "
                    "if (typeof value === 'boolean') { "
                    "return value ? '&#10003;' : '&#10007;'; "
                    "} "
                    "return value; "
                    "}"
                ),
            },
            titles=_PSD_INFO_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _PSD_INFO_TITLES
            },
        )
        self._experimental_apply_button = self._pn.widgets.Button(
            name="PSD Analysis",
            button_type="primary",
        )
        self._experimental_apply_button.on_click(self._on_experimental_apply)
        self._experimental_legend = self._pn.pane.HTML(
            gui_legend_html(("raw", "downsampled", "cutoff")),
            sizing_mode="fixed",
            width=320,
        )

    def _experimental_tab(self):
        self._experimental_plot_tabs = self._pn.Tabs(
            (
                "S(f)",
                self._experimental_psd_pane,
            ),
            (
                "V(t) / I(t)",
                self._experimental_time_pane,
            ),
            active=0,
            sizing_mode="stretch_width",
        )
        return self._pn.Column(
            self._experimental_apply_button,
            self._pn.Row(
                self._pn.Column(
                    self._experimental_table,
                    self._experimental_legend,
                    width=320,
                    sizing_mode="fixed",
                ),
                self._pn.Spacer(width=16),
                self._experimental_plot_tabs,
                sizing_mode="stretch_width",
            ),
            sizing_mode="stretch_width",
        )

    def _experimental_frame(self) -> pd.DataFrame:
        sigma_I_nA = np.nan
        sigma_V_mV = np.nan
        if self._psd is not None:
            sigma_I_nA = float(self._psd["raw_sigma_I_nA"])
            sigma_V_mV = float(self._psd["raw_sigma_V_mV"])
        return pd.DataFrame(
            [
                {
                    "key": "nu_Hz",
                    "parameter": _EXPERIMENTAL_TITLES["nu_Hz"],
                    "value": float(self._shared_nu_Hz),
                },
                {
                    "key": "detrend",
                    "parameter": _EXPERIMENTAL_TITLES["detrend"],
                    "value": bool(self._experimental_detrend),
                },
                {
                    "key": "sigma_V_mV",
                    "parameter": _EXPERIMENTAL_TITLES["sigma_V_mV"],
                    "value": sigma_V_mV,
                },
                {
                    "key": "sigma_I_nA",
                    "parameter": _EXPERIMENTAL_TITLES["sigma_I_nA"],
                    "value": sigma_I_nA,
                },
            ],
            dtype=object,
        )

    def _sync_psd_widgets_from_state(self) -> None:
        self._experimental_table.value = self._experimental_frame()

    def _experimental_settings_from_table(self) -> tuple[float, bool]:
        frame = self._experimental_table.value.reset_index(drop=True).set_index("key")
        try:
            nu_Hz = float(frame.at["nu_Hz", "value"])
        except (TypeError, ValueError):
            raise ValueError("nu_Hz must be finite and > 0.") from None
        if not np.isfinite(nu_Hz) or nu_Hz <= 0.0:
            raise ValueError("nu_Hz must be finite and > 0.")
        detrend = bool(frame.at["detrend", "value"])
        return nu_Hz, detrend

    def _on_experimental_apply(self, _: object) -> None:
        try:
            nu_Hz, detrend = self._experimental_settings_from_table()
        except ValueError:
            self._experimental_table.value = self._experimental_frame()
            return
        self._set_shared_nu_Hz(nu_Hz)
        self._experimental_detrend = detrend
        self._recompute_pipeline(
            clear_fit=True,
            recompute_psd=True,
            recompute_offset=False,
            recompute_sampling=True,
        )
        self._sync_control_widgets_from_specs()
        self._refresh_all_views()
        self._notify_state_changed()
