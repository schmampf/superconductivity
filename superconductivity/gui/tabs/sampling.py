from __future__ import annotations

from dataclasses import replace
from queue import Empty, Queue
from threading import Event

import numpy as np
import pandas as pd

from ...evaluation.sampling import Sample, SamplingSpec, sample
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
    "nu_Hz": "<i>&nu;</i> (Hz)",
    "N_up": "<i>N</i><sub>up</sub>",
    "median_bins": "<i>N</i><sub>med</sub>",
    "sigma_bins": "<i>&sigma;</i><sub>bins</sub>",
}
_SAMPLING_STAGE_OPTIONS = {
    "Offset": "apply_offset_correction",
    "Downsample": "apply_downsampling",
    "Upsample": "apply_upsampling",
    "Smooth": "apply_smoothing",
}
_SAMPLING_ACTION_CONTAINER_WIDTH = 360
_SAMPLING_ACTION_BUTTON_WIDTH = 80
_SAMPLING_ACTION_HEIGHT = 32
_SAMPLING_ACTION_ROW_GAP_WIDTH = 10
_SAMPLING_PROGRESS_WIDTH = _SAMPLING_ACTION_CONTAINER_WIDTH
_SAMPLING_PAIR_GAP_WIDTH = 10
_SAMPLING_ROW_WIDTH = 730
_SAMPLING_SECTION_GAP_HEIGHT = 24
_SAMPLING_INFO_TABLE_WIDTH = 180
_SAMPLING_GRID_TABLE_WIDTH = 350
_SAMPLING_SMOOTHING_TABLE_WIDTH = 180
_SAMPLING_RESULT_WIDTH = 360


def _sampling_heading_html(text: str) -> str:
    return (
        '<div style="font-size: 14px; font-weight: 600; line-height: 1.4;">'
        f"{text}"
        "</div>"
    )


def _run_sampling_batch(
    traces: list[object],
    order: list[int],
    spec: SamplingSpec,
    offsetanalysis_by_index: list[object | None],
    event_queue: Queue,
    stop_event: Event,
) -> None:
    for index in order:
        if stop_event.is_set():
            break
        trace = traces[index]
        offsetanalysis = offsetanalysis_by_index[index]
        event_queue.put(("running", index, None, ""))
        try:
            sampled = sample(
                trace,
                samplingspec=replace(spec),
                offsetanalysis=offsetanalysis,
                show_progress=False,
            )
        except Exception as exc:  # pragma: no cover - exercised via GUI tests
            event_queue.put(("failed", index, None, f"{type(exc).__name__}: {exc}"))
        else:
            event_queue.put(("done", index, sampled, ""))


class GUISamplingTabMixin:
    def _init_sampling_batch_state(self) -> None:
        self._sampling_batch_status: list[str] = [
            "idle" for _ in range(len(self.traces))
        ]
        self._sampling_batch_errors: list[str] = [
            "" for _ in range(len(self.traces))
        ]
        self._sampling_batch_queue: Queue | None = None
        self._sampling_batch_running = False
        self._sampling_batch_future = None
        self._sampling_batch_timer = None
        self._sampling_batch_completed = 0
        self._sampling_batch_stop_event = Event()
        self._sampling_batch_stop_requested = False
        self._sampling_stage_group_syncing = False

    def _build_sampling_widgets(self) -> None:
        self._sampling_heading = self._pn.pane.HTML(
            _sampling_heading_html("Sampling:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._sampling_specs_heading = self._pn.pane.HTML(
            _sampling_heading_html("Sampling Specs:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._sampling_stages_heading = self._pn.pane.HTML(
            _sampling_heading_html("Stages:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._sampling_results_heading = self._pn.pane.HTML(
            _sampling_heading_html("Offset Results:"),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._sampling_grid_table = self._pn.widgets.Tabulator(
            self._sampling_grid_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="fixed",
            width=_SAMPLING_GRID_TABLE_WIDTH,
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
            title_formatters={
                key: {"type": "html"}
                for key in _SAMPLING_GRID_TITLES
            },
            margin=0,
        )
        self._sampling_grid_table.on_edit(self._on_sampling_spec_edited)
        self._sampling_info_table = self._pn.widgets.Tabulator(
            self._sampling_info_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_SAMPLING_INFO_TABLE_WIDTH,
            height=110,
            widths={
                "parameter": 105,
            },
            editors={
                "parameter": None,
                "value": {"type": "number"},
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_SAMPLING_INFO_TITLES,
            title_formatters={
                key: {"type": "html"}
                for key in _SAMPLING_INFO_TITLES
            },
            margin=0,
        )
        self._sampling_info_table.on_edit(self._on_sampling_spec_edited)
        self._sampling_smoothing_table = self._pn.widgets.Tabulator(
            self._sampling_smoothing_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_SAMPLING_SMOOTHING_TABLE_WIDTH,
            height=110,
            widths={
                "parameter": 110,
            },
            editors={
                "parameter": None,
                "value": {"type": "number"},
            },
            formatters={
                "parameter": {"type": "html"},
            },
            titles=_SAMPLING_INFO_TITLES,
            title_formatters={
                key: {"type": "html"}
                for key in _SAMPLING_INFO_TITLES
            },
            margin=0,
        )
        self._sampling_smoothing_table.on_edit(self._on_sampling_spec_edited)
        self._sampling_apply_button = self._pn.widgets.Button(
            name="Single",
            button_type="primary",
            sizing_mode="fixed",
            width=_SAMPLING_ACTION_BUTTON_WIDTH,
            height=_SAMPLING_ACTION_HEIGHT,
            margin=0,
        )
        self._sampling_apply_button.on_click(self._on_sampling_apply)
        self._sampling_batch_apply_button = self._pn.widgets.Button(
            name="All",
            button_type="default",
            sizing_mode="fixed",
            width=_SAMPLING_ACTION_BUTTON_WIDTH,
            height=_SAMPLING_ACTION_HEIGHT,
            margin=0,
        )
        self._sampling_batch_apply_button.on_click(self._on_sampling_batch_apply)
        self._sampling_batch_stop_button = self._pn.widgets.Button(
            name="Stop",
            button_type="danger",
            disabled=True,
            sizing_mode="fixed",
            width=_SAMPLING_ACTION_BUTTON_WIDTH,
            height=_SAMPLING_ACTION_HEIGHT,
            margin=0,
        )
        self._sampling_batch_stop_button.on_click(self._on_sampling_batch_stop)
        self._sampling_refresh_button = self._pn.widgets.Button(
            name="Update",
            button_type="default",
            sizing_mode="fixed",
            width=_SAMPLING_ACTION_BUTTON_WIDTH,
            height=_SAMPLING_ACTION_HEIGHT,
            margin=0,
        )
        self._sampling_refresh_button.on_click(self._on_sampling_refresh)
        self._sampling_batch_progress = self._pn.indicators.Progress(
            value=0,
            max=len(self.traces),
            sizing_mode="stretch_width",
            height=_SAMPLING_ACTION_HEIGHT,
            margin=0,
        )
        self._sampling_batch_state = self._pn.pane.Markdown(
            "Idle",
            sizing_mode="stretch_width",
            margin=0,
        )
        self._sampling_stage_group = self._pn.widgets.CheckButtonGroup(
            options=_SAMPLING_STAGE_OPTIONS,
            value=self._sampling_stage_group_value(),
            sizing_mode="stretch_width",
            margin=0,
        )
        self._sampling_stage_group.param.watch(
            self._on_sampling_stage_group_changed,
            "value",
        )

    def _sampling_tab(self):
        return self._pn.Column(
            self._pn.Row(
                self._pn.Column(
                    self._sampling_heading,
                    self._pn.Spacer(height=_SAMPLING_ACTION_ROW_GAP_WIDTH),
                    self._pn.Row(
                        self._sampling_apply_button,
                        self._sampling_batch_apply_button,
                        self._sampling_batch_stop_button,
                        self._sampling_refresh_button,
                        sizing_mode="fixed",
                        width=_SAMPLING_ACTION_CONTAINER_WIDTH,
                        margin=0,
                        styles={
                            "gap": f"{_SAMPLING_ACTION_ROW_GAP_WIDTH}px"
                        },
                    ),
                    self._pn.Spacer(height=_SAMPLING_ACTION_ROW_GAP_WIDTH),
                    self._pn.Row(
                        self._sampling_batch_progress,
                        sizing_mode="fixed",
                        width=_SAMPLING_PROGRESS_WIDTH,
                        margin=0,
                    ),
                    width=_SAMPLING_ACTION_CONTAINER_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                self._pn.Column(
                    self._sampling_stages_heading,
                    self._pn.Spacer(height=_SAMPLING_ACTION_ROW_GAP_WIDTH),
                    self._sampling_stage_group,
                    width=_SAMPLING_ACTION_CONTAINER_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                sizing_mode="fixed",
                width=_SAMPLING_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_SAMPLING_PAIR_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_SAMPLING_SECTION_GAP_HEIGHT),
            self._sampling_specs_heading,
            self._pn.Row(
                self._pn.Column(
                    self._sampling_info_table,
                    width=_SAMPLING_INFO_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                self._pn.Column(
                    self._sampling_grid_table,
                    width=_SAMPLING_GRID_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                self._pn.Column(
                    self._sampling_smoothing_table,
                    width=_SAMPLING_SMOOTHING_TABLE_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                sizing_mode="fixed",
                width=_SAMPLING_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_SAMPLING_PAIR_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_SAMPLING_SECTION_GAP_HEIGHT),
            self._sampling_results_heading,
            self._pn.Row(
                self._pn.Column(
                    self._sampling_offset_v_pane,
                    width=_SAMPLING_RESULT_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                self._pn.Column(
                    self._sampling_offset_i_pane,
                    width=_SAMPLING_RESULT_WIDTH,
                    sizing_mode="fixed",
                    margin=0,
                ),
                sizing_mode="fixed",
                width=_SAMPLING_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_SAMPLING_PAIR_GAP_WIDTH}px"},
            ),
            margin=0,
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
        return pd.DataFrame(
            [
                {
                    "key": "nu_Hz",
                    "parameter": _SAMPLING_INFO_PARAMETER_LABELS["nu_Hz"],
                    "value": float(self._sampling_spec.nu_Hz),
                },
                {
                    "key": "N_up",
                    "parameter": _SAMPLING_INFO_PARAMETER_LABELS["N_up"],
                    "value": int(self._sampling_spec.N_up),
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
                    "value": int(self._sampling_spec.median_bins),
                },
                {
                    "key": "sigma_bins",
                    "parameter": _SAMPLING_INFO_PARAMETER_LABELS["sigma_bins"],
                    "value": float(self._sampling_spec.sigma_bins),
                },
            ],
            dtype=object,
        )

    def _build_sampling_spec_from_tables(self) -> SamplingSpec:
        grid_frame = self._sampling_grid_table.value.reset_index(drop=True)
        info_frame = self._sampling_info_table.value.reset_index(drop=True).set_index(
            "key"
        )
        smoothing_frame = (
            self._sampling_smoothing_table.value.reset_index(drop=True).set_index(
                "key"
            )
        )
        selected_stages = set(self._sampling_stage_group.value)
        return SamplingSpec(
            apply_offset_correction=(
                "apply_offset_correction" in selected_stages
            ),
            apply_downsampling=("apply_downsampling" in selected_stages),
            apply_upsampling=("apply_upsampling" in selected_stages),
            apply_smoothing=("apply_smoothing" in selected_stages),
            nu_Hz=float(info_frame.at["nu_Hz", "value"]),
            N_up=int(info_frame.at["N_up", "value"]),
            median_bins=int(smoothing_frame.at["median_bins", "value"]),
            sigma_bins=float(smoothing_frame.at["sigma_bins", "value"]),
            mode=self._sampling_spec.mode,
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

    def _sync_sampling_widgets_from_spec(self) -> None:
        self._sampling_grid_table.value = self._sampling_grid_frame()
        self._sampling_info_table.value = self._sampling_info_frame()
        self._sampling_smoothing_table.value = self._sampling_smoothing_frame()
        self._sampling_stage_group_syncing = True
        try:
            self._sampling_stage_group.value = self._sampling_stage_group_value()
        finally:
            self._sampling_stage_group_syncing = False
        self._sampling_batch_progress.max = len(self.traces)

    def _sampling_stage_group_value(self) -> list[str]:
        values: list[str] = []
        if self._sampling_spec.apply_offset_correction:
            values.append("apply_offset_correction")
        if self._sampling_spec.apply_downsampling:
            values.append("apply_downsampling")
        if self._sampling_spec.apply_upsampling:
            values.append("apply_upsampling")
        if self._sampling_spec.apply_smoothing:
            values.append("apply_smoothing")
        return values

    def _on_sampling_stage_group_changed(self, _: object) -> None:
        if self._sampling_stage_group_syncing or self._sampling_batch_running:
            return
        self._clear_sampling_batch_cache()
        self._refresh_sampling_views()

    def _on_sampling_spec_edited(self, _: object) -> None:
        if self._sampling_batch_running:
            return
        self._clear_sampling_batch_cache()
        self._refresh_sampling_views()

    def _on_sampling_apply(self, _: object) -> None:
        if self._sampling_batch_running or self._fit_running:
            return
        new_sampling_spec = self._build_sampling_spec_from_tables()
        previous_sampling_spec = replace(self._sampling_spec)
        self._sampling_spec = new_sampling_spec
        if not self._sampling_specs_match(previous_sampling_spec, self._sampling_spec):
            self._clear_sampling_batch_cache()
        self._recompute_pipeline(
            clear_fit=True,
            recompute_psd=False,
            recompute_offset=False,
            recompute_sampling=True,
        )
        self._stage_sampling_result(self.active_index, self._require_sampling())
        self._sampling_batch_status[self.active_index] = "done"
        self._sampling_batch_errors[self.active_index] = ""
        self._sampling_batch_completed = sum(
            status == "done" for status in self._sampling_batch_status
        )
        self._sampling_batch_progress.value = self._sampling_batch_completed
        self._sampling_batch_state.object = "Done"
        self._sync_control_widgets_from_specs()
        self._refresh_all_views()
        self._notify_state_changed()

    def _clear_sampling_batch_cache(
        self,
        *,
        indices: list[int] | None = None,
    ) -> None:
        if indices is None:
            if self._sampling_batch_timer is not None:
                self._sampling_batch_timer.stop()
                self._sampling_batch_timer = None
            self._clear_sampling_stage_cache()
            self._sampling_batch_status = ["idle" for _ in range(len(self.traces))]
            self._sampling_batch_errors = ["" for _ in range(len(self.traces))]
            self._sampling_batch_queue = None
            self._sampling_batch_running = False
            self._sampling_batch_future = None
            self._sampling_batch_completed = 0
            self._sampling_batch_stop_event = Event()
            self._sampling_batch_stop_requested = False
            if hasattr(self, "_sampling_batch_progress"):
                self._sampling_batch_progress.max = len(self.traces)
                self._sampling_batch_progress.value = 0
            if hasattr(self, "_sampling_batch_state"):
                self._sampling_batch_state.object = "Idle"
            if hasattr(self, "_sampling_grid_table"):
                self._sampling_grid_table.disabled = False
            if hasattr(self, "_sampling_info_table"):
                self._sampling_info_table.disabled = False
            if hasattr(self, "_sampling_smoothing_table"):
                self._sampling_smoothing_table.disabled = False
            if hasattr(self, "_sampling_stage_group"):
                self._sampling_stage_group.disabled = False
            if hasattr(self, "_sampling_apply_button"):
                self._sampling_apply_button.disabled = False
            if hasattr(self, "_sampling_batch_apply_button"):
                self._sampling_batch_apply_button.disabled = False
            if hasattr(self, "_sampling_batch_stop_button"):
                self._sampling_batch_stop_button.disabled = True
            if hasattr(self, "_fit_button"):
                self._fit_button.disabled = False
            return

        self._clear_sampling_stage_cache(indices=indices)
        for index in indices:
            if 0 <= int(index) < len(self._sampling_batch_status):
                self._sampling_batch_status[int(index)] = "idle"
                self._sampling_batch_errors[int(index)] = ""
        self._sampling_batch_completed = sum(
            status == "done" for status in self._sampling_batch_status
        )
        if hasattr(self, "_sampling_batch_progress"):
            self._sampling_batch_progress.value = self._sampling_batch_completed

    def _sampling_offsetanalysis_snapshot(self) -> list[object | None]:
        snapshot: list[object | None] = []
        for index in range(len(self.traces)):
            offsetanalysis = self._sampling_offsetanalysis_for_index(index)
            if offsetanalysis is None:
                snapshot.append(None)
            else:
                snapshot.append(self._copy_offset_trace(offsetanalysis))
        return snapshot

    def _on_sampling_batch_apply(self, _: object) -> None:
        if self._sampling_batch_running or self._fit_running:
            return
        self._sampling_spec = self._build_sampling_spec_from_tables()
        self._clear_sampling_batch_cache()
        self._sampling_batch_queue = Queue()
        self._sampling_batch_status = ["queued" for _ in range(len(self.traces))]
        self._sampling_batch_running = True
        self._sampling_batch_stop_event = Event()
        self._sampling_batch_stop_requested = False
        self._sampling_batch_progress.max = len(self.traces)
        self._sampling_batch_progress.value = 0
        self._sampling_batch_state.object = "Running"
        self._sampling_grid_table.disabled = True
        self._sampling_info_table.disabled = True
        self._sampling_smoothing_table.disabled = True
        self._sampling_stage_group.disabled = True
        self._sampling_apply_button.disabled = True
        self._sampling_batch_apply_button.disabled = True
        self._sampling_batch_stop_button.disabled = False
        if hasattr(self, "_fit_button"):
            self._fit_button.disabled = True
        run_order = [int(self.active_index)]
        run_order.extend(
            index for index in range(len(self.traces)) if index != int(self.active_index)
        )
        self._sampling_batch_future = self._executor.submit(
            _run_sampling_batch,
            list(self.traces),
            run_order,
            replace(self._sampling_spec),
            self._sampling_offsetanalysis_snapshot(),
            self._sampling_batch_queue,
            self._sampling_batch_stop_event,
        )
        self._start_sampling_batch_timer()

    def _on_sampling_batch_stop(self, _: object) -> None:
        if not self._sampling_batch_running or self._sampling_batch_stop_requested:
            return
        self._sampling_batch_stop_requested = True
        self._sampling_batch_stop_event.set()
        self._sampling_batch_stop_button.disabled = True
        self._sampling_batch_state.object = "Stopping"

    def _on_sampling_refresh(self, _: object) -> None:
        self._refresh_sampling_views()

    def _start_sampling_batch_timer(self) -> None:
        if self._sampling_batch_timer is not None:
            self._sampling_batch_timer.stop()
        self._sampling_batch_timer = self._pn.state.add_periodic_callback(
            self._update_sampling_batch_timer,
            period=400,
            start=True,
        )

    def _update_sampling_batch_timer(self) -> None:
        if self._sampling_batch_queue is None:
            return

        refresh_active_trace = False
        plot_changed = False
        while True:
            try:
                state, index, sampled, message = (
                    self._sampling_batch_queue.get_nowait()
                )
            except Empty:
                break

            if state == "running":
                self._sampling_batch_status[index] = "running"
                continue
            if state == "failed":
                self._sampling_batch_status[index] = "failed"
                self._sampling_batch_errors[index] = str(message)
                self._sampling_batch_completed += 1
                continue

            self._stage_sampling_result(index, sampled)
            self._sampling_batch_status[index] = "done"
            self._sampling_batch_errors[index] = ""
            self._sampling_batch_completed += 1
            plot_changed = True
            if int(index) == int(self.active_index):
                refresh_active_trace = True

        self._sampling_batch_progress.value = self._sampling_batch_completed
        if self._sampling_batch_stop_requested:
            self._sampling_batch_state.object = "Stopping"
        elif self._sampling_batch_running:
            self._sampling_batch_state.object = "Running"

        if plot_changed:
            self._refresh_sampling_views()

        if refresh_active_trace:
            self._recompute_pipeline(
                clear_fit=True,
                recompute_psd=False,
                recompute_offset=False,
                recompute_sampling=True,
            )
            self._sync_control_widgets_from_specs()
            self._refresh_all_views()
            self._notify_state_changed()

        if (
            self._sampling_batch_future is not None
            and self._sampling_batch_future.done()
            and self._sampling_batch_queue.empty()
        ):
            self._finalize_sampling_batch()

    def _finalize_sampling_batch(self) -> None:
        failed = sum(status == "failed" for status in self._sampling_batch_status)
        self._sampling_batch_running = False
        self._sampling_grid_table.disabled = False
        self._sampling_info_table.disabled = False
        self._sampling_smoothing_table.disabled = False
        self._sampling_stage_group.disabled = False
        self._sampling_apply_button.disabled = False
        self._sampling_batch_apply_button.disabled = False
        self._sampling_batch_stop_button.disabled = True
        if hasattr(self, "_fit_button"):
            self._fit_button.disabled = False
        if self._sampling_batch_timer is not None:
            self._sampling_batch_timer.stop()
            self._sampling_batch_timer = None

        future = self._sampling_batch_future
        self._sampling_batch_future = None
        if future is not None:
            try:
                future.result()
            except Exception as exc:  # pragma: no cover
                self._sampling_batch_state.object = (
                    f"Failed: `{type(exc).__name__}: {exc}`"
                )
            else:
                if self._sampling_batch_stop_requested:
                    for index, status in enumerate(self._sampling_batch_status):
                        if status == "queued":
                            self._sampling_batch_status[index] = "stopped"
                    self._sampling_batch_state.object = "Stopped"
                elif failed == 0:
                    self._sampling_batch_state.object = "Done"
                else:
                    self._sampling_batch_state.object = "Done with failures"
        self._refresh_sampling_views()
