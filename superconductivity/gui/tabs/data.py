from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from ...evaluation.traces import FileSpec
from ...evaluation.traces import data as trace_data_module

_DATA_Y_TITLES = {
    "quantity": "Y quantities",
}
_DATA_X_TITLES = {
    "quantity": "X quantity",
}
_DATA_TIMEFRAME_TITLES = {
    "specific_key": "Time frame",
}
_DATA_COLORWAY = tuple(qualitative.Plotly)
_ALL_TIMEFRAME_ID = "__all__"
_ALL_TIMEFRAME_LABEL = "all"
_CLEAR_Y_ID = "__clear__"
_CLEAR_Y_LABEL = "clear"
_DATA_TABLE_HEIGHT = 240
_DATA_AXIS_STEP = 0.07


def _series_label(source: str, key: str) -> str:
    """Return a stable legend label for one raw series."""
    return f"{source}: {key}"


def _data_quantity_label(source: str, key: str) -> str:
    """Return one display label for the quantity tables."""
    return f"{source}: {key}"


class GUIDataTabMixin:
    def _build_data_widgets(self) -> None:
        self._data_inventory_signature: tuple[str | None, str | None] | None = None
        self._data_selected_y_ids: set[str] = set()
        self._data_selected_x_id = "time"
        self._data_selected_specific_keys: set[str] | None = None
        self._data_quantity_rows: list[dict[str, object]] = []
        self._data_specific_key_rows: list[dict[str, object]] = []
        self._data_specific_key_windows: dict[str, tuple[float, float]] = {}
        self._data_dirty = True
        self._data_inventory_message = ""
        self._suppress_data_x_selection_watch = False
        self._suppress_data_y_selection_watch = False
        self._suppress_data_timeframe_selection_watch = False

        self._data_status = self._pn.pane.HTML(
            "",
            visible=False,
            sizing_mode="stretch_width",
            margin=0,
        )
        self._data_y_table = self._pn.widgets.Tabulator(
            self._data_y_frame(),
            show_index=False,
            selectable=True,
            sortable=False,
            hidden_columns=["id", "source", "key"],
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=_DATA_TABLE_HEIGHT,
            editors={
                "quantity": None,
            },
            titles=_DATA_Y_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _DATA_Y_TITLES
            },
            margin=0,
        )
        self._data_y_table.param.watch(
            self._on_data_y_selection_changed,
            "selection",
        )
        self._data_x_table = self._pn.widgets.Tabulator(
            self._data_x_frame(),
            show_index=False,
            selectable=1,
            sortable=False,
            hidden_columns=["id", "source", "key"],
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=_DATA_TABLE_HEIGHT,
            editors={
                "quantity": None,
            },
            titles=_DATA_X_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _DATA_X_TITLES
            },
            margin=0,
        )
        self._data_x_table.param.watch(
            self._on_data_x_selection_changed,
            "selection",
        )
        self._data_timeframe_table = self._pn.widgets.Tabulator(
            self._data_timeframe_frame(),
            show_index=False,
            selectable=True,
            sortable=False,
            hidden_columns=["id"],
            layout="fit_columns",
            sizing_mode="stretch_width",
            height=_DATA_TABLE_HEIGHT,
            editors={
                "specific_key": None,
            },
            titles=_DATA_TIMEFRAME_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _DATA_TIMEFRAME_TITLES
            },
            margin=0,
        )
        self._data_timeframe_table.param.watch(
            self._on_data_timeframe_selection_changed,
            "selection",
        )
        self._data_figure = self._build_data_figure()
        self._data_pane = self._pn.pane.Plotly(
            self._data_figure,
            sizing_mode="stretch_width",
            height=480,
            config={"responsive": True},
            margin=0,
        )

    def _data_tab(self):
        return self._pn.Column(
            self._pn.Row(
                self._data_y_table,
                self._data_x_table,
                self._data_timeframe_table,
                sizing_mode="stretch_width",
            ),
            self._data_status,
            self._data_pane,
            margin=0,
            sizing_mode="stretch_width",
        )

    def _data_signature(self) -> tuple[str | None, str | None]:
        """Return the inventory signature for the current file selection."""
        if self._filespec is None:
            return (None, None)
        try:
            path = str(self._filespec.path)
        except Exception:
            path = str(self._filespec.h5path)
        return path, self._filespec.measurement

    def _data_y_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._data_quantity_rows,
            columns=["id", "source", "key", "quantity"],
            dtype=object,
        )

    def _data_x_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = [
            {
                "id": "time",
                "source": "",
                "key": "time",
                "quantity": "time",
            }
        ]
        for row in self._data_quantity_rows:
            if str(row["id"]) == _CLEAR_Y_ID:
                continue
            rows.append(
                {
                    "id": row["id"],
                    "source": row["source"],
                    "key": row["key"],
                    "quantity": row["quantity"],
                },
            )
        return pd.DataFrame(
            rows,
            columns=["id", "source", "key", "quantity"],
            dtype=object,
        )

    def _data_timeframe_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            self._data_specific_key_rows,
            columns=["id", "specific_key"],
            dtype=object,
        )

    def _set_data_status(self, message: str, *, kind: str = "info") -> None:
        """Show one short status or warning message for the Data tab."""
        if message == "":
            self._data_status.object = ""
            self._data_status.visible = False
            return
        color = "#475467" if kind == "info" else "#b54708"
        self._data_status.object = (
            f'<div style="color: {color}; font-size: 12px; line-height: 1.4;">'
            f"{message}"
            "</div>"
        )
        self._data_status.visible = True

    def _set_data_x_selection(self) -> None:
        """Sync the X quantity row selection from the stored selection id."""
        frame = self._data_x_table.value.reset_index(drop=True)
        selected = frame.index[frame["id"] == self._data_selected_x_id].tolist()
        if not selected:
            selected = [0]
            self._data_selected_x_id = "time"
        self._suppress_data_x_selection_watch = True
        try:
            self._data_x_table.selection = selected[:1]
        finally:
            self._suppress_data_x_selection_watch = False

    def _sync_data_widgets_from_state(self) -> None:
        signature = self._data_signature()
        if signature == self._data_inventory_signature:
            return
        self._data_inventory_signature = signature
        self._reload_data_inventory()

    def _reload_data_inventory(self) -> None:
        """Rebuild available raw-data inventories for the current file."""
        self._data_quantity_rows = [
            {
                "id": _CLEAR_Y_ID,
                "source": "",
                "key": "",
                "quantity": _CLEAR_Y_LABEL,
            },
        ]
        self._data_specific_key_rows = []
        self._data_specific_key_windows = {}
        self._data_inventory_message = ""
        previous_y_ids = set(self._data_selected_y_ids)
        previous_x_id = str(self._data_selected_x_id)
        previous_specific_keys = (
            None
            if self._data_selected_specific_keys is None
            else set(self._data_selected_specific_keys)
        )

        if self._filespec is None or self._filespec.measurement is None:
            self._data_selected_y_ids = set()
            self._data_selected_x_id = "time"
            self._data_selected_specific_keys = None
            self._apply_data_frames()
            self._data_dirty = True
            return

        try:
            measurement_keys = trace_data_module.get_measurement_keys(
                self._filespec,
            )
            status_keys = trace_data_module.get_status_keys(self._filespec)
            self._data_specific_key_windows = (
                self._load_data_specific_key_windows(self._filespec)
            )
        except Exception as exc:
            self._data_selected_y_ids = set()
            self._data_selected_x_id = "time"
            self._data_selected_specific_keys = None
            self._data_inventory_message = str(exc)
            self._apply_data_frames()
            self._data_dirty = True
            return

        quantity_rows: list[dict[str, object]] = [
            {
                "id": _CLEAR_Y_ID,
                "source": "",
                "key": "",
                "quantity": _CLEAR_Y_LABEL,
            },
        ]
        available_ids: set[str] = set()
        for source, keys in (
            ("measurement", measurement_keys),
            ("status", status_keys),
        ):
            for key in sorted(keys):
                row_id = f"{source}::{key}"
                available_ids.add(row_id)
                quantity_rows.append(
                    {
                        "id": row_id,
                        "source": source,
                        "key": key,
                        "quantity": _data_quantity_label(source, key),
                    },
                )
        self._data_quantity_rows = quantity_rows
        self._data_selected_y_ids = previous_y_ids & available_ids
        if previous_x_id == "time" or previous_x_id not in available_ids:
            self._data_selected_x_id = "time"
        else:
            self._data_selected_x_id = previous_x_id

        ordered_specific_keys = self._ordered_data_specific_keys()
        if previous_specific_keys is None:
            selected_specific_keys = None
        else:
            selected_specific_keys = (
                previous_specific_keys & set(ordered_specific_keys)
            )
            if len(selected_specific_keys) == 0:
                selected_specific_keys = None
        self._data_selected_specific_keys = selected_specific_keys
        self._data_specific_key_rows = [
            {
                "id": _ALL_TIMEFRAME_ID,
                "specific_key": _ALL_TIMEFRAME_LABEL,
            },
        ]
        self._data_specific_key_rows.extend(
            [
                {
                    "id": specific_key,
                    "specific_key": specific_key,
                }
                for specific_key in ordered_specific_keys
            ]
        )
        self._apply_data_frames()
        self._data_dirty = True

    def _apply_data_frames(self) -> None:
        """Push the current raw-data state into the tab widgets."""
        self._data_y_table.value = self._data_y_frame()
        self._set_data_y_selection()
        self._data_x_table.value = self._data_x_frame()
        self._set_data_x_selection()
        self._data_timeframe_table.value = self._data_timeframe_frame()
        self._set_data_timeframe_selection()

    def _set_data_y_selection(self) -> None:
        """Sync the Y quantity row selections from stored selection ids."""
        frame = self._data_y_table.value.reset_index(drop=True)
        if "id" not in frame.columns:
            frame = self._data_y_frame()
        if len(self._data_selected_y_ids) == 0:
            selection = frame.index[frame["id"] == _CLEAR_Y_ID].tolist()
        else:
            selection = frame.index[
                frame["id"].astype(str).isin(self._data_selected_y_ids)
            ].tolist()
        self._suppress_data_y_selection_watch = True
        try:
            self._data_y_table.selection = selection
        finally:
            self._suppress_data_y_selection_watch = False

    def _ordered_data_specific_keys(self) -> list[str]:
        """Return raw file specific keys ordered for the time-frame table."""
        windows = self._data_specific_key_windows
        if self._filespec is None or self._filespec.measurement is None:
            return []
        try:
            raw_specific_keys = list(self._filespec.skeys())
        except Exception:
            raw_specific_keys = []
        if not windows:
            return raw_specific_keys
        ordered = [key for key in raw_specific_keys if key in windows]
        ordered.extend(key for key in windows if key not in ordered)
        return ordered

    def _set_data_timeframe_selection(self) -> None:
        """Sync the time-frame row selections from stored selection ids."""
        frame = self._data_timeframe_table.value.reset_index(drop=True)
        if "id" not in frame.columns:
            frame = self._data_timeframe_frame()
        if self._data_selected_specific_keys is None:
            selection = frame.index[frame["id"] == _ALL_TIMEFRAME_ID].tolist()
        else:
            selection = frame.index[
                frame["id"].astype(str).isin(self._data_selected_specific_keys)
            ].tolist()
        self._suppress_data_timeframe_selection_watch = True
        try:
            self._data_timeframe_table.selection = selection
        finally:
            self._suppress_data_timeframe_selection_watch = False

    def _load_data_specific_key_windows(
        self,
        filespec: FileSpec,
    ) -> dict[str, tuple[float, float]]:
        """Return per-specific-key adwin time windows for one file."""
        if filespec.measurement is None:
            return {}
        path, measurement = trace_data_module._require_file_measurement(filespec)
        h5py = trace_data_module._open_h5()
        with h5py.File(path, "r") as file:
            return trace_data_module._get_specific_key_windows(
                file,
                measurement=measurement,
            )

    def _selected_data_quantity_rows(self) -> list[dict[str, object]]:
        """Return selected Y quantity rows in display order."""
        return [
            row
            for row in self._data_quantity_rows
            if str(row["id"]) in self._data_selected_y_ids
        ]

    def _data_row_by_id(self, row_id: str) -> dict[str, object] | None:
        """Return one quantity row by id."""
        for row in self._data_quantity_rows:
            if str(row["id"]) == row_id:
                return row
        return None

    def _selected_data_window(self) -> tuple[float, float] | None:
        """Return the union time window selected in the time-frame table."""
        windows = self._data_specific_key_windows
        if len(windows) == 0:
            return None
        selected_keys = (
            set(windows)
            if not self._data_selected_specific_keys
            else {
                key
                for key in self._data_selected_specific_keys
                if key in windows
            }
        )
        if len(selected_keys) == 0:
            selected_keys = set(windows)
        starts = [windows[key][0] for key in selected_keys]
        stops = [windows[key][1] for key in selected_keys]
        return min(starts), max(stops)

    def _measurement_time_origin(self) -> float | None:
        """Return the earliest measurement time point for display."""
        if len(self._data_specific_key_windows) == 0:
            return None
        return min(window[0] for window in self._data_specific_key_windows.values())

    @staticmethod
    def _crop_data_series(
        series: tuple[np.ndarray, np.ndarray],
        *,
        window: tuple[float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Crop one raw data series to one optional closed time window."""
        time_s = np.asarray(series[0], dtype=np.float64).reshape(-1)
        value = np.asarray(series[1]).reshape(-1)
        if window is None:
            return time_s, value
        mask = (
            np.isfinite(time_s)
            & (time_s >= float(window[0]))
            & (time_s <= float(window[1]))
        )
        return time_s[mask], value[mask]

    @staticmethod
    def _coerce_numeric_series(
        value: np.ndarray,
    ) -> np.ndarray | None:
        """Return one numeric value array or ``None`` if not numeric."""
        array = np.asarray(value)
        if array.size == 0:
            return np.asarray(array, dtype=np.float64)
        if np.iscomplexobj(array):
            return None
        try:
            numeric = np.asarray(array, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        return numeric

    @staticmethod
    def _sorted_unique_curve(
        x: np.ndarray,
        y: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sort one numeric curve and average repeated x positions."""
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

    def _load_data_series(
        self,
        *,
        source: str,
        key: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Load one raw series from the configured data source."""
        if self._filespec is None:
            raise RuntimeError("filespec is required for raw data loading.")
        if source == "measurement":
            return trace_data_module.get_measurement_series(self._filespec, key)
        if source == "status":
            return trace_data_module.get_status_series(self._filespec, key)
        raise KeyError(f"Unknown data source: {source!r}.")

    def _data_axis_name(self, axis_index: int) -> str:
        """Return one Plotly axis key."""
        return "yaxis" if axis_index == 1 else f"yaxis{axis_index}"

    def _data_trace_axis_name(self, axis_index: int) -> str:
        """Return one Plotly trace axis reference."""
        return "y" if axis_index == 1 else f"y{axis_index}"

    def _build_data_figure(self) -> go.Figure:
        """Build the current raw-data figure."""
        figure = go.Figure()
        figure.update_layout(
            margin={"l": 20, "r": 20, "t": 20, "b": 110},
            showlegend=True,
            xaxis_title="<i>t</i> (s)",
            yaxis={
                "side": "right",
                "tickformat": ".2g",
                "ticks": "",
                "zeroline": False,
            },
            legend={
                "orientation": "h",
                "x": 0.0,
                "xanchor": "left",
                "y": -0.22,
                "yanchor": "top",
            },
            uirevision="data",
        )

        if self._filespec is None or self._filespec.measurement is None:
            self._set_data_status(
                "Select a file and measurement to inspect raw data.",
                kind="info",
            )
            return figure
        if self._data_inventory_message != "":
            self._set_data_status(self._data_inventory_message, kind="warning")
            return figure

        selected_rows = self._selected_data_quantity_rows()
        if len(selected_rows) == 0:
            self._set_data_status("", kind="info")
            return figure

        window = self._selected_data_window()
        time_origin_s = self._measurement_time_origin()
        warnings: list[str] = []
        plotted_traces: list[tuple[go.Scatter, str, str]] = []
        xaxis_title = (
            "<i>t</i> - <i>t</i><sub>0</sub> (s)"
            if time_origin_s is not None
            else "<i>t</i> (s)"
        )

        if self._data_selected_x_id == "time":
            for row in selected_rows:
                label = _series_label(str(row["source"]), str(row["key"]))
                try:
                    time_s, value = self._load_data_series(
                        source=str(row["source"]),
                        key=str(row["key"]),
                    )
                except Exception as exc:
                    warnings.append(f"{label}: {exc}")
                    continue
                time_s, value = self._crop_data_series(
                    (time_s, value),
                    window=window,
                )
                numeric = self._coerce_numeric_series(value)
                if numeric is None:
                    warnings.append(f"{label}: non-numeric series skipped.")
                    continue
                finite = np.isfinite(time_s) & np.isfinite(numeric)
                if not np.any(finite):
                    continue
                x_plot = np.asarray(time_s[finite], dtype=np.float64)
                if time_origin_s is not None:
                    x_plot = x_plot - float(time_origin_s)
                plotted_traces.append(
                    (
                        go.Scatter(
                            x=x_plot,
                            y=np.asarray(numeric[finite], dtype=np.float64),
                            mode="lines",
                            name=label,
                        ),
                        label,
                        label,
                    ),
                )
        else:
            x_row = self._data_row_by_id(self._data_selected_x_id)
            if x_row is None:
                self._set_data_status(
                    "Selected x quantity is no longer available.",
                    kind="warning",
                )
                return figure
            xaxis_title = _series_label(
                str(x_row["source"]),
                str(x_row["key"]),
            )
            try:
                x_time_s, x_value = self._load_data_series(
                    source=str(x_row["source"]),
                    key=str(x_row["key"]),
                )
            except Exception as exc:
                self._set_data_status(f"{xaxis_title}: {exc}", kind="warning")
                return figure
            x_time_s, x_value = self._crop_data_series(
                (x_time_s, x_value),
                window=window,
            )
            x_numeric = self._coerce_numeric_series(x_value)
            if x_numeric is None:
                self._set_data_status(
                    f"{xaxis_title}: non-numeric x series skipped.",
                    kind="warning",
                )
                return figure
            x_time_s, x_numeric = self._sorted_unique_curve(x_time_s, x_numeric)
            if x_time_s.size == 0:
                self._set_data_status(
                    f"{xaxis_title}: no samples in the selected window.",
                    kind="warning",
                )
                return figure

            for row in selected_rows:
                label = _series_label(str(row["source"]), str(row["key"]))
                try:
                    y_time_s, y_value = self._load_data_series(
                        source=str(row["source"]),
                        key=str(row["key"]),
                    )
                except Exception as exc:
                    warnings.append(f"{label}: {exc}")
                    continue
                y_time_s, y_value = self._crop_data_series(
                    (y_time_s, y_value),
                    window=window,
                )
                y_numeric = self._coerce_numeric_series(y_value)
                if y_numeric is None:
                    warnings.append(f"{label}: non-numeric series skipped.")
                    continue
                y_time_s, y_numeric = self._sorted_unique_curve(
                    y_time_s,
                    y_numeric,
                )
                if y_time_s.size == 0:
                    warnings.append(f"{label}: no samples in the selected window.")
                    continue
                overlap = (
                    np.isfinite(x_numeric)
                    & (x_time_s >= float(np.min(y_time_s)))
                    & (x_time_s <= float(np.max(y_time_s)))
                )
                if not np.any(overlap):
                    warnings.append(
                        f"{label}: no overlap with selected x quantity."
                    )
                    continue
                y_interp = np.interp(x_time_s[overlap], y_time_s, y_numeric)
                x_plot = np.asarray(x_numeric[overlap], dtype=np.float64)
                y_plot = np.asarray(y_interp, dtype=np.float64)
                finite = np.isfinite(x_plot) & np.isfinite(y_plot)
                if not np.any(finite):
                    warnings.append(
                        f"{label}: no finite overlap with selected x quantity."
                    )
                    continue
                plotted_traces.append(
                    (
                        go.Scatter(
                            x=x_plot[finite],
                            y=y_plot[finite],
                            mode="lines",
                            name=label,
                        ),
                        label,
                        label,
                    ),
                )

        if len(plotted_traces) == 0:
            self._set_data_status("<br>".join(warnings), kind="warning")
            figure.update_layout(xaxis_title=xaxis_title)
            return figure

        extra_axes = max(0, len(plotted_traces) - 1)
        figure.update_layout(
            margin={
                "l": 20,
                "r": 20,
                "t": 20,
                "b": 110,
            },
            xaxis={"title": xaxis_title, "domain": [0.0, 1.0]},
        )

        for axis_index, (trace, axis_title, legend_label) in enumerate(
            plotted_traces,
            start=1,
        ):
            color = _DATA_COLORWAY[(axis_index - 1) % len(_DATA_COLORWAY)]
            trace.name = legend_label
            trace.line = {"color": color}
            trace.yaxis = self._data_trace_axis_name(axis_index)
            figure.add_trace(trace)

            axis_name = self._data_axis_name(axis_index)
            axis_layout: dict[str, Any] = {
                "tickfont": {"color": color},
                "tickformat": ".2g",
                "ticks": "",
                "linecolor": color,
                "mirror": False,
                "zeroline": False,
                "side": "right",
            }
            if axis_index == 1:
                axis_layout["showgrid"] = True
            else:
                axis_layout.update(
                    {
                        "overlaying": "y",
                        "anchor": "free",
                        "position": max(
                            0.55,
                            1.0 - _DATA_AXIS_STEP * (axis_index - 1),
                        ),
                        "showgrid": False,
                    },
                )
            figure.layout[axis_name] = axis_layout

        self._set_data_status("<br>".join(warnings), kind="warning")
        return figure

    def _refresh_data_views(self) -> None:
        """Refresh the raw-data figure only when the tab state changed."""
        if not self._data_dirty:
            return
        self._data_figure = self._build_data_figure()
        self._data_pane.object = self._data_figure
        self._data_dirty = False

    def _on_data_y_selection_changed(self, event: object) -> None:
        """Update Y quantity selections from the table selection state."""
        if self._suppress_data_y_selection_watch:
            return
        frame = self._data_y_table.value.reset_index(drop=True)
        selection = [
            int(index)
            for index in getattr(event, "new", [])
            if 0 <= int(index) < len(frame)
        ]
        selected_ids = {
            str(frame.at[index, "id"])
            for index in selection
        }
        if _CLEAR_Y_ID in selected_ids:
            self._data_selected_y_ids = set()
            self._set_data_y_selection()
        else:
            self._data_selected_y_ids = selected_ids
        self._data_dirty = True
        self._refresh_data_views()

    def _on_data_x_selection_changed(self, event: object) -> None:
        if self._suppress_data_x_selection_watch:
            return
        selection = list(getattr(event, "new", []))
        if len(selection) == 0:
            return
        row = int(selection[0])
        frame = self._data_x_table.value.reset_index(drop=True)
        if row < 0 or row >= len(frame):
            return
        self._data_selected_x_id = str(frame.at[row, "id"])
        self._data_dirty = True
        self._refresh_data_views()

    def _on_data_timeframe_selection_changed(self, event: object) -> None:
        """Update the visible time window from the table selection state."""
        if self._suppress_data_timeframe_selection_watch:
            return
        frame = self._data_timeframe_table.value.reset_index(drop=True)
        selection = [
            int(index)
            for index in getattr(event, "new", [])
            if 0 <= int(index) < len(frame)
        ]
        selected_ids = [str(frame.at[index, "id"]) for index in selection]
        if _ALL_TIMEFRAME_ID in selected_ids:
            self._data_selected_specific_keys = None
            self._set_data_timeframe_selection()
        else:
            selected_keys = {
                row_id
                for row_id in selected_ids
                if row_id != _ALL_TIMEFRAME_ID
            }
            if len(selected_keys) == 0:
                self._data_selected_specific_keys = None
                self._set_data_timeframe_selection()
            else:
                self._data_selected_specific_keys = selected_keys
        self._data_dirty = True
        self._refresh_data_views()
