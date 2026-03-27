from __future__ import annotations

from collections import OrderedDict

from ..state import _trace_label


class GUITabsBaseMixin:
    def _build_control_widgets(self) -> None:
        options = OrderedDict(
            (_trace_label(index, trace), index)
            for index, trace in enumerate(self.traces)
        )
        self._trace_selector = self._pn.widgets.Select(
            name="Trace",
            options=options,
            value=0,
            sizing_mode="stretch_width",
        )
        self._trace_selector.param.watch(self._on_trace_changed, "value")

        self._build_psd_widgets()
        self._build_offset_widgets()
        self._build_sampling_widgets()

    def _sync_control_widgets_from_specs(self) -> None:
        self._sync_psd_widgets_from_state()
        self._sync_offset_widgets_from_spec()
        self._sync_sampling_widgets_from_spec()
