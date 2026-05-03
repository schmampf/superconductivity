from __future__ import annotations

import numpy as np

from ..style import gui_legend_html
from ..state import _EXPERIMENTAL_TITLES


class GUIPSDTabMixin:
    def _build_psd_widgets(self) -> None:
        self._syncing_psd_widgets = False
        start, end, step = self._experimental_nu_slider_bounds()
        value = min(max(float(self._shared_nu_Hz), start), end)
        self._shared_nu_Hz = value
        self._experimental_nu_input = self._pn.widgets.FloatInput(
            name="",
            start=start,
            end=end,
            step=step,
            value=value,
            width=110,
            sizing_mode="fixed",
            margin=0,
        )
        self._experimental_nu_input.param.watch(
            self._on_experimental_nu_input_changed,
            "value_throttled",
        )
        self._experimental_nu_slider = self._pn.widgets.FloatSlider(
            name="Sample Frequency (Hz)",
            start=start,
            end=end,
            step=step,
            value=value,
            sizing_mode="stretch_width",
            margin=0,
        )
        self._experimental_nu_slider.param.watch(
            self._on_experimental_nu_changed,
            "value",
        )
        self._experimental_detrend_toggle = self._pn.widgets.Checkbox(
            name=_EXPERIMENTAL_TITLES["detrend"],
            value=bool(self._experimental_detrend),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._experimental_detrend_toggle.param.watch(
            self._on_experimental_detrend_changed,
            "value",
        )
        self._experimental_legend = self._pn.pane.HTML(
            gui_legend_html(("raw", "downsampled")),
            sizing_mode="fixed",
            width=320,
            margin=0,
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
            margin=0,
        )
        return self._pn.Column(
            self._pn.Row(
                self._experimental_nu_input,
                self._experimental_nu_slider,
                sizing_mode="stretch_width",
            ),
            self._pn.Row(
                self._experimental_detrend_toggle,
                self._experimental_legend,
                sizing_mode="stretch_width",
            ),
            self._experimental_plot_tabs,
            margin=0,
            sizing_mode="stretch_width",
        )

    def _experimental_nu_slider_bounds(self) -> tuple[float, float, float]:
        t_s = np.asarray(self._active_trace()["t_s"], dtype=np.float64)
        diffs = np.diff(t_s)
        diffs = diffs[np.isfinite(diffs) & (diffs > 0.0)]
        if diffs.size == 0:
            max_nu_Hz = max(float(self._shared_nu_Hz), 0.1)
        else:
            max_nu_Hz = max(float(1.0 / np.median(diffs)), 0.1)
        step = 0.01
        start = min(step, max_nu_Hz)
        return start, max_nu_Hz, step

    def _sync_psd_widgets_from_state(self) -> None:
        start, end, step = self._experimental_nu_slider_bounds()
        value = min(max(float(self._shared_nu_Hz), start), end)
        self._syncing_psd_widgets = True
        try:
            self._shared_nu_Hz = value
            self._experimental_nu_input.start = start
            self._experimental_nu_input.end = end
            self._experimental_nu_input.step = step
            self._experimental_nu_input.value = value
            self._experimental_nu_slider.start = start
            self._experimental_nu_slider.end = end
            self._experimental_nu_slider.step = step
            self._experimental_nu_slider.value = value
            self._experimental_detrend_toggle.value = bool(
                self._experimental_detrend
            )
        finally:
            self._syncing_psd_widgets = False

    def _apply_experimental_nu_value(self, value: object) -> None:
        if self._syncing_psd_widgets:
            return
        try:
            changed = self._set_shared_nu_Hz(float(value))
        except (TypeError, ValueError):
            self._sync_psd_widgets_from_state()
            return
        if not changed:
            return
        self._sync_psd_widgets_from_state()
        self._compute_psd_stage()
        self._refresh_experimental_views()
        self._notify_state_changed()

    def _on_experimental_nu_changed(self, event: object) -> None:
        self._apply_experimental_nu_value(getattr(event, "new"))

    def _on_experimental_nu_input_changed(self, event: object) -> None:
        self._apply_experimental_nu_value(getattr(event, "new"))

    def _on_experimental_detrend_changed(self, event: object) -> None:
        if self._syncing_psd_widgets:
            return
        previous_raw_spec = self._current_psd_stage_spec()
        self._experimental_detrend = bool(getattr(event, "new"))
        if not self._psd_specs_match(
            previous_raw_spec,
            self._current_psd_stage_spec(),
        ):
            self._clear_psd_stage_cache()
        self._compute_psd_stage()
        self._stage_psd_result(self.active_index, self._require_psd())
        self._refresh_experimental_views()
        self._notify_state_changed()
