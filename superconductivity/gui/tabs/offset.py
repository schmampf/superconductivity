from __future__ import annotations
from queue import Empty, Queue
from threading import Event

import numpy as np
import pandas as pd
from panel.io.model import JSCode

from ...evaluation.analysis import (
    OffsetSpec,
    OffsetTrace,
    OffsetTraces,
    offset_analysis,
)
from ...evaluation.traces import numeric_yvalue
from ...utilities.meta import param
from ..state import _linspace_from_values, _trace_label

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
    "N_up": "<i>N</i><sub>up</sub>",
    "Voff_mV": "<i>V</i><sub>off</sub> (mV)",
    "Ioff_nA": "<i>I</i><sub>off</sub> (nA)",
}
_OFFSET_BATCH_TITLES = {
    "index": "Index",
    "specific_key": "Key",
    "yvalue": "y",
    "status": "Status",
    "Voff_mV": "<i>V</i><sub>off</sub> (mV)",
    "Ioff_nA": "<i>I</i><sub>off</sub> (nA)",
}
_OFFSET_PAIR_TABLE_WIDTH = 360
_OFFSET_PAIR_GAP_WIDTH = 10
_OFFSET_ROW_WIDTH = 2 * _OFFSET_PAIR_TABLE_WIDTH + _OFFSET_PAIR_GAP_WIDTH
_OFFSET_SECTION_GAP_HEIGHT = 24
_OFFSET_ACTION_HEIGHT = 32
_OFFSET_ACTION_ROW_GAP_WIDTH = 10
_OFFSET_ACTION_CONTAINER_WIDTH = _OFFSET_PAIR_TABLE_WIDTH
_OFFSET_ACTION_BUTTON_WIDTH = 80
_OFFSET_PROGRESS_WIDTH = _OFFSET_ACTION_CONTAINER_WIDTH


def _offset_heading_html(text: str) -> str:
    """Return heading markup styled close to table headers."""
    return (
        '<div style="font-size: 14px; font-weight: 600; line-height: 1.4;">'
        f"{text}"
        "</div>"
    )


def _run_offset_batch(
    traces: list[object],
    order: list[int],
    spec: OffsetSpec,
    event_queue: Queue,
    stop_event: Event,
) -> None:
    for index in order:
        if stop_event.is_set():
            break
        trace = traces[index]
        event_queue.put(("running", index, None, ""))
        try:
            offset = offset_analysis(trace, spec=spec)
        except Exception as exc:  # pragma: no cover - exercised via GUI tests
            event_queue.put(("failed", index, None, f"{type(exc).__name__}: {exc}"))
        else:
            event_queue.put(("done", index, offset, ""))


class GUIOffsetTabMixin:
    def _init_offset_batch_state(self) -> None:
        self._offset_batch_spec: OffsetSpec | None = None
        self._offset_batch_results: list[OffsetTrace | None] = [
            None for _ in range(len(self.traces))
        ]
        self._offset_batch_status: list[str] = ["idle" for _ in range(len(self.traces))]
        self._offset_batch_errors: list[str] = ["" for _ in range(len(self.traces))]
        self._offset_batch_queue: Queue | None = None
        self._offset_batch_running = False
        self._offset_batch_future = None
        self._offset_batch_timer = None
        self._offset_batch_completed = 0
        self._offset_batch_display_index: int | None = int(self.active_index)
        self._offset_display_selector_syncing = False
        self._offset_batch_stop_event = Event()
        self._offset_batch_stop_requested = False

    @staticmethod
    def _offset_specs_match(left: OffsetSpec, right: OffsetSpec) -> bool:
        return bool(
            np.array_equal(left.Vbins_mV.values, right.Vbins_mV.values)
            and np.array_equal(left.Ibins_nA.values, right.Ibins_nA.values)
            and np.array_equal(left.Voff_mV.values, right.Voff_mV.values)
            and np.array_equal(left.Ioff_nA.values, right.Ioff_nA.values)
            and float(left.nu_Hz.value) == float(right.nu_Hz.value)
            and int(left.N_up.value) == int(right.N_up.value)
        )

    def _ensure_offset_stage_spec(self, spec: OffsetSpec) -> None:
        if self._offset_batch_spec is None:
            self._offset_batch_spec = spec
            return
        if self._offset_specs_match(self._offset_batch_spec, spec):
            return
        self._clear_offset_batch_cache()
        self._offset_batch_spec = spec

    def _stage_offset_result(
        self,
        index: int,
        offset: OffsetTrace,
        *,
        select: bool = True,
    ) -> None:
        self._offset_batch_results[index] = offset
        self._offset_batch_status[index] = "done"
        self._offset_batch_errors[index] = ""
        if select:
            self._offset_batch_display_index = int(index)

    def _offset_display_selector_options(self) -> dict[str, int]:
        return {
            _trace_label(index, trace): int(index)
            for index, trace in enumerate(self.traces)
        }

    def _selected_offset_index(self) -> int:
        if self._offset_batch_display_index is not None:
            return int(self._offset_batch_display_index)
        if hasattr(self, "_offset_display_selector"):
            return int(self._offset_display_selector.value)
        return int(self.active_index)

    def _get_offset_result_for_index(self, index: int) -> OffsetTrace | None:
        cached = self._get_cached_offset_batch_result(int(index))
        if cached is not None:
            return cached
        if int(index) == int(self.active_index):
            return self._offset
        return None

    def _set_offset_display_index(
        self,
        index: int,
        *,
        refresh_views: bool,
    ) -> None:
        index = int(index)
        self._offset_batch_display_index = index
        if hasattr(self, "_offset_display_selector"):
            if int(self._offset_display_selector.value) != index:
                self._offset_display_selector_syncing = True
                try:
                    self._offset_display_selector.value = index
                finally:
                    self._offset_display_selector_syncing = False
        if refresh_views:
            self._refresh_offset_batch_views()

    def _sync_offset_display_index_from_active_change(
        self,
        *,
        old_index: int,
        new_index: int,
    ) -> None:
        if self._offset_batch_display_index != int(old_index):
            return
        self._set_offset_display_index(int(new_index), refresh_views=False)

    @staticmethod
    def _copy_offset_trace(offset: OffsetTrace) -> OffsetTrace:
        return {
            "dGerr_G0": np.asarray(offset["dGerr_G0"], dtype=np.float64).copy(),
            "dRerr_R0": np.asarray(offset["dRerr_R0"], dtype=np.float64).copy(),
            "Voff_mV": float(offset["Voff_mV"]),
            "Ioff_nA": float(offset["Ioff_nA"]),
        }

    def _load_offset_analysis_preset(
        self,
        offset_analysis: OffsetTrace | OffsetTraces,
    ) -> None:
        copied = self._copy_stage_preset_entries(
            offset_analysis,
            collection_type=OffsetTraces,
            single_name="OffsetTrace",
            copy_fn=self._copy_offset_trace,
        )
        self._offset_batch_spec = self._offset_spec
        self._offset_batch_results = [None for _ in range(len(self.traces))]
        self._offset_batch_status = ["idle" for _ in range(len(self.traces))]
        self._offset_batch_errors = ["" for _ in range(len(self.traces))]
        display_index: int | None = None
        completed = 0
        for index, result in enumerate(copied):
            self._offset_batch_results[index] = result
            self._offset_batch_status[index] = "done"
            completed += 1
            if index == int(self.active_index):
                display_index = index
            elif display_index is None:
                display_index = index
        self._offset_batch_completed = completed
        self._offset_batch_display_index = display_index
        self._offset_batch_progress.max = len(self.traces)
        self._offset_batch_progress.value = completed
        self._offset_batch_state.object = "Loaded"

    def _build_offset_widgets(self) -> None:
        self._offset_heading = self._pn.pane.HTML(
            _offset_heading_html("Offset Analysis:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._offset_specs_heading = self._pn.pane.HTML(
            _offset_heading_html("Offset Specs:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._offset_trace_selection_heading = self._pn.pane.HTML(
            _offset_heading_html("Trace Selection:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._offset_results_heading = self._pn.pane.HTML(
            _offset_heading_html("Offset Results:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._offset_grid_table = self._pn.widgets.Tabulator(
            self._offset_grid_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="fixed",
            width=_OFFSET_PAIR_TABLE_WIDTH,
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
            title_formatters={key: {"type": "html"} for key in _OFFSET_GRID_TITLES},
            margin=0,
        )
        self._offset_grid_table.on_edit(self._on_offset_spec_edited)
        self._offset_info_table = self._pn.widgets.Tabulator(
            self._offset_info_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_OFFSET_PAIR_TABLE_WIDTH,
            height=180,
            widths={
                "parameter": 150,
            },
            editors={
                "parameter": None,
                "value": {"type": "number"},
            },
            editables={
                "value": JSCode(
                    "function(cell) { "
                    "const key = cell.getData().key; "
                    "return key === 'nu_Hz' || key === 'N_up'; "
                    "}"
                )
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_OFFSET_INFO_TITLES,
            title_formatters={key: {"type": "html"} for key in _OFFSET_INFO_TITLES},
            margin=0,
        )
        self._offset_info_table.on_edit(self._on_offset_spec_edited)
        self._offset_apply_button = self._pn.widgets.Button(
            name="Single",
            button_type="primary",
            sizing_mode="fixed",
            width=_OFFSET_ACTION_BUTTON_WIDTH,
            height=_OFFSET_ACTION_HEIGHT,
            margin=0,
        )
        self._offset_apply_button.on_click(self._on_offset_apply)
        self._offset_batch_apply_button = self._pn.widgets.Button(
            name="All",
            button_type="default",
            sizing_mode="fixed",
            width=_OFFSET_ACTION_BUTTON_WIDTH,
            height=_OFFSET_ACTION_HEIGHT,
            margin=0,
        )
        self._offset_batch_apply_button.on_click(self._on_offset_batch_apply)
        self._offset_batch_stop_button = self._pn.widgets.Button(
            name="Stop",
            button_type="danger",
            disabled=True,
            sizing_mode="fixed",
            width=_OFFSET_ACTION_BUTTON_WIDTH,
            height=_OFFSET_ACTION_HEIGHT,
            margin=0,
        )
        self._offset_batch_stop_button.on_click(self._on_offset_batch_stop)
        self._offset_batch_spinner = self._pn.indicators.LoadingSpinner(
            value=False,
            width=20,
            height=20,
            margin=0,
        )
        self._offset_batch_progress = self._pn.indicators.Progress(
            value=0,
            max=len(self.traces),
            sizing_mode="stretch_width",
            height=_OFFSET_ACTION_HEIGHT,
            margin=0,
        )
        self._offset_batch_state = self._pn.pane.Markdown(
            "Idle",
            sizing_mode="stretch_width",
            margin=0,
        )
        self._offset_display_selector = self._pn.widgets.Select(
            name="",
            options=self._offset_display_selector_options(),
            value=self._selected_offset_index(),
            sizing_mode="fixed",
            width=_OFFSET_PAIR_TABLE_WIDTH,
            margin=0,
        )
        self._offset_display_selector.param.watch(
            self._on_offset_display_index_changed,
            "value",
        )
        self._offset_batch_table = self._pn.widgets.Tabulator(
            self._offset_batch_frame(),
            show_index=False,
            selectable=1,
            sortable=False,
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=240,
            editors={
                "index": None,
                "specific_key": None,
                "yvalue": None,
                "status": None,
                "Voff_mV": None,
                "Ioff_nA": None,
            },
            formatters={
                "Voff_mV": {"type": "plaintext"},
                "Ioff_nA": {"type": "plaintext"},
            },
            titles=_OFFSET_BATCH_TITLES,
            title_formatters={key: {"type": "html"} for key in _OFFSET_BATCH_TITLES},
            margin=0,
        )

    @staticmethod
    def _display_scalar(value: float, *, significant_digits: int = 7) -> float:
        value = float(value)
        if not np.isfinite(value):
            return value
        return float(f"{value:.{significant_digits}g}")

    def _offset_tab(self):
        return self._pn.Column(
            self._pn.Row(
                self._pn.Column(
                    self._offset_heading,
                    self._pn.Spacer(height=_OFFSET_ACTION_ROW_GAP_WIDTH),
                    self._pn.Row(
                        self._offset_apply_button,
                        self._offset_batch_apply_button,
                        self._offset_batch_stop_button,
                        sizing_mode="fixed",
                        width=_OFFSET_ACTION_CONTAINER_WIDTH,
                        margin=0,
                        styles={"gap": f"{_OFFSET_ACTION_ROW_GAP_WIDTH}px"},
                    ),
                    self._pn.Spacer(height=_OFFSET_ACTION_ROW_GAP_WIDTH),
                    self._pn.Row(
                        self._offset_batch_progress,
                        sizing_mode="fixed",
                        width=_OFFSET_PROGRESS_WIDTH,
                        margin=0,
                    ),
                    width=_OFFSET_ACTION_CONTAINER_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                self._pn.Column(
                    self._offset_trace_selection_heading,
                    self._pn.Spacer(height=_OFFSET_ACTION_ROW_GAP_WIDTH),
                    self._offset_display_selector,
                    width=_OFFSET_PAIR_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                sizing_mode="fixed",
                width=_OFFSET_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_OFFSET_PAIR_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_OFFSET_SECTION_GAP_HEIGHT),
            self._offset_specs_heading,
            self._pn.Row(
                self._pn.Column(
                    self._offset_grid_table,
                    width=_OFFSET_PAIR_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                self._pn.Column(
                    self._offset_info_table,
                    width=_OFFSET_PAIR_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                sizing_mode="fixed",
                width=_OFFSET_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_OFFSET_PAIR_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_OFFSET_SECTION_GAP_HEIGHT),
            self._offset_results_heading,
            self._pn.Row(
                self._pn.Column(
                    self._offset_g_pane,
                    width=_OFFSET_PAIR_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                self._pn.Column(
                    self._offset_r_pane,
                    width=_OFFSET_PAIR_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                sizing_mode="fixed",
                width=_OFFSET_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_OFFSET_PAIR_GAP_WIDTH}px"},
            ),
            margin=0,
            sizing_mode="stretch_width",
        )

    def _offset_grid_frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(
            [
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Vbins_mV"],
                    "start": float(self._offset_spec.Vbins_mV.values[0]),
                    "stop": float(self._offset_spec.Vbins_mV.values[-1]),
                    "count": int(self._offset_spec.Vbins_mV.values.size),
                },
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Ibins_nA"],
                    "start": float(self._offset_spec.Ibins_nA.values[0]),
                    "stop": float(self._offset_spec.Ibins_nA.values[-1]),
                    "count": int(self._offset_spec.Ibins_nA.values.size),
                },
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Voff_mV"],
                    "start": float(self._offset_spec.Voff_mV.values[0]),
                    "stop": float(self._offset_spec.Voff_mV.values[-1]),
                    "count": int(self._offset_spec.Voff_mV.values.size),
                },
                {
                    "parameter": _OFFSET_GRID_PARAMETER_LABELS["Ioff_nA"],
                    "start": float(self._offset_spec.Ioff_nA.values[0]),
                    "stop": float(self._offset_spec.Ioff_nA.values[-1]),
                    "count": int(self._offset_spec.Ioff_nA.values.size),
                },
            ]
        )
        frame["count"] = frame["count"].astype(np.int64)
        return frame

    def _offset_info_frame(self) -> pd.DataFrame:
        Voff_mV = np.nan
        Ioff_nA = np.nan
        display_offset = self._offset_display_result()
        if display_offset is not None:
            Voff_mV = self._display_scalar(float(display_offset["Voff_mV"]))
            Ioff_nA = self._display_scalar(float(display_offset["Ioff_nA"]))
        return pd.DataFrame(
            [
                {
                    "key": "nu_Hz",
                    "parameter": _OFFSET_INFO_PARAMETER_LABELS["nu_Hz"],
                    "value": float(self._offset_spec.nu_Hz.value),
                },
                {
                    "key": "N_up",
                    "parameter": _OFFSET_INFO_PARAMETER_LABELS["N_up"],
                    "value": int(self._offset_spec.N_up.value),
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

    def _offset_display_result(self) -> OffsetTrace | None:
        return self._get_offset_result_for_index(self._selected_offset_index())

    def _build_offset_spec_from_table(self) -> OffsetSpec:
        grid_frame = self._offset_grid_table.value.reset_index(drop=True)
        info_frame = self._offset_info_table.value.reset_index(drop=True).set_index(
            "key"
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
            nu_Hz=param("nu_Hz", float(info_frame.at["nu_Hz", "value"]), fixed=True),
            N_up=param("N_up", int(info_frame.at["N_up", "value"]), fixed=True),
        )

    def _sync_offset_widgets_from_spec(self) -> None:
        self._offset_grid_table.value = self._offset_grid_frame()
        self._offset_info_table.value = self._offset_info_frame()
        options = self._offset_display_selector_options()
        self._offset_display_selector.options = options
        selected_index = self._selected_offset_index()
        if selected_index < 0 or selected_index >= len(self.traces):
            selected_index = 0
        self._set_offset_display_index(int(selected_index), refresh_views=False)

    def _on_offset_spec_edited(self, _: object) -> None:
        if self._offset_batch_running or self._offset_batch_spec is None:
            return
        self._clear_offset_batch_cache()
        self._refresh_offset_batch_views()

    def _on_offset_display_index_changed(self, event: object) -> None:
        if self._offset_display_selector_syncing:
            return
        new_value = int(getattr(event, "new"))
        old_value = int(getattr(event, "old"))
        if new_value == old_value:
            return
        self._set_offset_display_index(new_value, refresh_views=True)

    def _on_offset_apply(self, _: object) -> None:
        if self._offset_batch_running or self._fit_running:
            return
        selected_index = self._selected_offset_index()
        offset_spec = self._build_offset_spec_from_table()
        self._offset_spec = offset_spec
        self._ensure_offset_stage_spec(offset_spec)
        self._clear_sampling_batch_cache(indices=[int(self.active_index)])
        self._recompute_pipeline(
            clear_fit=True,
            recompute_psd=False,
            recompute_offset=True,
            recompute_sampling=True,
        )
        self._stage_offset_result(
            self.active_index,
            self._require_offset(),
            select=(selected_index == int(self.active_index)),
        )
        if selected_index != int(self.active_index):
            display_offset = offset_analysis(
                self.traces[selected_index],
                spec=self._offset_spec,
            )
            self._stage_offset_result(selected_index, display_offset)
        self._sync_control_widgets_from_specs()
        self._refresh_all_views()
        self._notify_state_changed()

    def _offset_batch_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for index, trace in enumerate(self.traces):
            result = self._offset_batch_results[index]
            meta = trace["meta"]
            yvalue = numeric_yvalue(meta.yvalue)
            rows.append(
                {
                    "index": int(index),
                    "specific_key": str(meta.specific_key),
                    "yvalue": (
                        meta.yvalue
                        if yvalue is None
                        else float(yvalue)
                    ),
                    "status": self._offset_batch_status[index],
                    "Voff_mV": (
                        np.nan
                        if result is None
                        else self._display_scalar(float(result["Voff_mV"]))
                    ),
                    "Ioff_nA": (
                        np.nan
                        if result is None
                        else self._display_scalar(float(result["Ioff_nA"]))
                    ),
                }
            )
        return pd.DataFrame(rows, dtype=object)

    def _clear_offset_batch_cache(self) -> None:
        if self._offset_batch_timer is not None:
            self._offset_batch_timer.stop()
            self._offset_batch_timer = None
        self._offset_batch_spec = None
        self._offset_batch_results = [None for _ in range(len(self.traces))]
        self._offset_batch_status = ["idle" for _ in range(len(self.traces))]
        self._offset_batch_errors = ["" for _ in range(len(self.traces))]
        self._offset_batch_queue = None
        self._offset_batch_future = None
        self._offset_batch_running = False
        self._offset_batch_completed = 0
        self._set_offset_display_index(
            self._selected_offset_index(),
            refresh_views=False,
        )
        self._offset_batch_stop_event = Event()
        self._offset_batch_stop_requested = False
        self._offset_batch_spinner.value = False
        self._offset_batch_progress.max = len(self.traces)
        self._offset_batch_progress.value = 0
        self._offset_batch_state.object = "Idle"
        self._offset_grid_table.disabled = False
        self._offset_info_table.disabled = False
        self._offset_apply_button.disabled = False
        self._offset_batch_apply_button.disabled = False
        self._offset_batch_stop_button.disabled = True
        if hasattr(self, "_fit_button"):
            self._fit_button.disabled = False

    def _get_cached_offset_batch_result(
        self,
        index: int,
    ) -> OffsetTrace | None:
        if self._offset_batch_spec is None:
            return None
        if index < 0 or index >= len(self._offset_batch_results):
            return None
        return self._offset_batch_results[index]

    def _on_offset_batch_apply(self, _: object) -> None:
        if self._offset_batch_running or self._fit_running:
            return
        self._offset_spec = self._build_offset_spec_from_table()
        self._clear_offset_batch_cache()
        self._offset_batch_spec = self._offset_spec
        self._offset_batch_queue = Queue()
        self._offset_batch_status = ["queued" for _ in range(len(self.traces))]
        self._offset_batch_running = True
        self._set_offset_display_index(
            self._selected_offset_index(),
            refresh_views=False,
        )
        self._offset_batch_stop_event = Event()
        self._offset_batch_stop_requested = False
        self._offset_batch_spinner.value = True
        self._offset_batch_progress.max = len(self.traces)
        self._offset_batch_progress.value = 0
        self._offset_batch_state.object = "Running"
        self._offset_grid_table.disabled = True
        self._offset_info_table.disabled = True
        self._offset_apply_button.disabled = True
        self._offset_batch_apply_button.disabled = True
        self._offset_batch_stop_button.disabled = False
        if hasattr(self, "_fit_button"):
            self._fit_button.disabled = True
        self._refresh_offset_batch_views()
        run_order: list[int] = []
        for index in (
            self._selected_offset_index(),
            int(self.active_index),
        ):
            if index not in run_order:
                run_order.append(int(index))
        run_order.extend(
            index
            for index in range(len(self.traces))
            if index not in run_order
        )
        self._offset_batch_future = self._executor.submit(
            _run_offset_batch,
            list(self.traces),
            run_order,
            self._offset_spec,
            self._offset_batch_queue,
            self._offset_batch_stop_event,
        )
        self._start_offset_batch_timer()

    def _on_offset_batch_stop(self, _: object) -> None:
        if not self._offset_batch_running or self._offset_batch_stop_requested:
            return
        self._offset_batch_stop_requested = True
        self._offset_batch_stop_event.set()
        self._offset_batch_stop_button.disabled = True
        self._offset_batch_state.object = "Stopping"

    def _start_offset_batch_timer(self) -> None:
        if self._offset_batch_timer is not None:
            self._offset_batch_timer.stop()
        self._offset_batch_timer = self._pn.state.add_periodic_callback(
            self._update_offset_batch_timer,
            period=400,
            start=True,
        )

    def _update_offset_batch_timer(self) -> None:
        if self._offset_batch_queue is None:
            return

        refresh_active_trace = False
        refresh_display_trace = False
        status_changed = False
        plot_changed = False
        while True:
            try:
                state, index, offset, message = self._offset_batch_queue.get_nowait()
            except Empty:
                break

            if state == "running":
                self._offset_batch_status[index] = "running"
                self._set_offset_display_index(int(index), refresh_views=False)
                status_changed = True
                continue
            if state == "failed":
                self._offset_batch_status[index] = "failed"
                self._offset_batch_errors[index] = str(message)
                self._offset_batch_completed += 1
                self._set_offset_display_index(int(index), refresh_views=False)
                status_changed = True
                continue

            self._offset_batch_status[index] = "done"
            self._offset_batch_errors[index] = ""
            self._offset_batch_results[index] = offset
            self._offset_batch_completed += 1
            self._clear_sampling_batch_cache(indices=[int(index)])
            self._set_offset_display_index(int(index), refresh_views=False)
            status_changed = True
            plot_changed = True
            if int(index) == int(self.active_index):
                refresh_active_trace = True
            if self._offset_batch_display_index == int(index):
                refresh_display_trace = True

        if status_changed:
            self._offset_batch_progress.value = self._offset_batch_completed
            if self._offset_batch_stop_requested:
                self._offset_batch_state.object = "Stopping"
            else:
                self._offset_batch_state.object = "Running"
            self._offset_batch_table.value = self._offset_batch_frame()

        if plot_changed:
            self._refresh_offset_batch_views()

        if refresh_display_trace:
            self._refresh_offset_views()

        if refresh_active_trace:
            self._recompute_pipeline(
                clear_fit=True,
                recompute_psd=False,
                recompute_offset=True,
                recompute_sampling=True,
            )
            self._sync_control_widgets_from_specs()
            self._refresh_all_views()

        if (
            self._offset_batch_future is not None
            and self._offset_batch_future.done()
            and self._offset_batch_queue.empty()
        ):
            self._finalize_offset_batch()

    def _finalize_offset_batch(self) -> None:
        failed = sum(status == "failed" for status in self._offset_batch_status)
        self._offset_batch_running = False
        self._offset_batch_spinner.value = False
        self._offset_grid_table.disabled = False
        self._offset_info_table.disabled = False
        self._offset_apply_button.disabled = False
        self._offset_batch_apply_button.disabled = False
        self._offset_batch_stop_button.disabled = True
        if hasattr(self, "_fit_button"):
            self._fit_button.disabled = False
        if self._offset_batch_timer is not None:
            self._offset_batch_timer.stop()
            self._offset_batch_timer = None

        future = self._offset_batch_future
        self._offset_batch_future = None
        if future is not None:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover
                self._offset_batch_state.object = (
                    f"Failed: `{type(exc).__name__}: {exc}`"
                )
            else:
                if self._offset_batch_stop_requested:
                    for index, status in enumerate(self._offset_batch_status):
                        if status == "queued":
                            self._offset_batch_status[index] = "stopped"
                    self._offset_batch_state.object = "Stopped"
                elif failed == 0:
                    self._offset_batch_state.object = "Done"
                else:
                    self._offset_batch_state.object = "Done with failures"
        self._refresh_offset_batch_views()

    def _on_offset_batch_table_selection(self, event: object) -> None:
        _ = event
        return
