from __future__ import annotations

import ast
import math
import shutil
import subprocess
import sys
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...evaluation.traces import (
    FileSpec,
    Keys,
    KeysSpec,
    TraceSpec,
    get_keys,
    get_traces,
)
from ..state import _trace_label

_TRACES_TABLE_TITLES = {
    "parameter": "Parameter",
    "value": "Value",
}
_MEASUREMENT_TABLE_TITLES = {
    "measurement": "Measurement",
}
_SPECIFIC_KEY_NAME_TABLE_TITLES = {
    "specific_key": "Specific Key",
}
_SPECIFIC_KEY_TABLE_TITLES = {
    "trace_index": "Index",
    "specific_key": "Extraction Key",
    "yvalue": "yvalue",
}
_FILE_PARAMETER_LABELS = {
    "h5path": "h5path",
    "location": "location",
    "measurement": "measurement",
}
_KEYS_PARAMETER_LABELS = {
    "strip0": "strip0",
    "strip1": "strip1",
    "remove_key": "remove_key",
    "add_key": "add_key",
    "norm": "norm",
    "label": "label",
    "html_label": "html_label",
    "limits": "limits",
}
_TRACE_PARAMETER_LABELS = {
    "amp_voltage": "amp_voltage",
    "amp_current": "amp_current",
    "r_ref_ohm": "r_ref_ohm",
    "trigger_values": "trigger_values",
    "skip": "skip",
    "subtract_offset": "subtract_offset",
    "time_relative": "time_relative",
}
_MEASUREMENT_PAIR_TABLE_WIDTH = 360
_MEASUREMENT_PAIR_GAP_WIDTH = 10
_MEASUREMENT_ROW_WIDTH = (
    2 * _MEASUREMENT_PAIR_TABLE_WIDTH + _MEASUREMENT_PAIR_GAP_WIDTH
)
_MEASUREMENT_SECTION_GAP_HEIGHT = 24
_MEASUREMENT_ERROR_GAP_HEIGHT = 12
_FILESPEC_BUTTON_STACK_WIDTH = 110
_FILESPEC_ROW_GAP_WIDTH = _MEASUREMENT_PAIR_GAP_WIDTH
_FILESPEC_TABLE_WIDTH = (
    _MEASUREMENT_ROW_WIDTH
    - _FILESPEC_BUTTON_STACK_WIDTH
    - _FILESPEC_ROW_GAP_WIDTH
)


def _trace_table_value(value: Any) -> object:
    """Format one trace-side value for compact table display."""
    if value is None:
        return ""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return repr(tuple(value))
    if isinstance(value, list):
        return repr(list(value))
    return value


def _trace_yvalue_value(value: Any) -> object:
    """Format one y-value for compact display."""
    if value is None:
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return _trace_table_value(value)
    if not math.isfinite(numeric):
        return ""
    return f"{numeric:.6g}"


def _table_heading_html(text: str) -> str:
    """Return heading markup styled close to table headers."""
    return (
        '<div style="font-size: 14px; font-weight: 600; line-height: 1.4;">'
        f"{text}"
        "</div>"
    )


def _browse_h5_file(
    filespec: FileSpec | None,
) -> Path | None:
    """Open the native macOS file dialog and return the selected HDF5 path."""
    initial_dir = None
    if filespec is not None:
        if filespec.location is not None:
            initial_dir = str(Path(filespec.location).expanduser())
        try:
            resolved = filespec.path
        except Exception:
            resolved = None
        if resolved is not None:
            initial_dir = str(resolved.parent)

    if sys.platform != "darwin" or shutil.which("osascript") is None:
        return None
    return _browse_h5_file_macos(initial_dir=initial_dir)


def _browse_h5_file_macos(
    *,
    initial_dir: str | None,
) -> Path | None:
    """Open the native macOS file picker via AppleScript."""

    def _apple_string(value: str) -> str:
        return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'

    lines = []
    if initial_dir is not None:
        initial_dir_path = str(Path(initial_dir).expanduser())
        lines.append(
            f"set initialDir to POSIX file {_apple_string(initial_dir_path)} as alias"
        )
        lines.append(
            'set chosenFile to choose file with prompt "Select HDF5 file" '
            "default location initialDir"
        )
    else:
        lines.append('set chosenFile to choose file with prompt "Select HDF5 file"')
    lines.append("POSIX path of chosenFile")

    completed = subprocess.run(
        ["osascript", "-e", "\n".join(lines)],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return None

    selected = completed.stdout.strip()
    if selected == "":
        return None
    return Path(selected).expanduser()


class GUIMeasurementTabMixin:
    def _effective_keysspec(self) -> KeysSpec:
        """Return the effective KeysSpec used for reloads."""
        return KeysSpec() if self._keysspec is None else self._keysspec

    def _preview_keys(self) -> Keys | None:
        """Return the current extraction preview from ``get_keys``."""
        if self._filespec is None:
            return None
        try:
            return get_keys(
                filespec=self._filespec,
                keysspec=self._effective_keysspec(),
            )
        except Exception:
            return None

    def _keys_preview_titles(self) -> dict[str, str]:
        """Return table titles for the keys preview table."""
        return {
            "trace_index": _SPECIFIC_KEY_TABLE_TITLES["trace_index"],
            "specific_key": _SPECIFIC_KEY_TABLE_TITLES["specific_key"],
            "yvalue": "Value",
        }

    def _build_trace_widgets(self) -> None:
        self._traces_file_heading = self._pn.pane.HTML(
            _table_heading_html("Choose File:"),
            sizing_mode="stretch_width",
        )
        self._traces_measurement_heading = self._pn.pane.HTML(
            _table_heading_html("Choose Measurement:"),
            sizing_mode="stretch_width",
        )
        self._traces_keys_heading = self._pn.pane.HTML(
            _table_heading_html("Handle Value Extraction:"),
            sizing_mode="stretch_width",
        )
        self._traces_spec_heading = self._pn.pane.HTML(
            _table_heading_html("Handle Trace Extraction:"),
            sizing_mode="stretch_width",
        )
        self._keysspec_error = self._pn.pane.HTML(
            "",
            visible=False,
            sizing_mode="stretch_width",
        )
        self._tracespec_error = self._pn.pane.HTML(
            "",
            visible=False,
            sizing_mode="stretch_width",
        )
        self._filespec_browse_button = self._pn.widgets.Button(
            name="Browse...",
            button_type="default",
            width=_FILESPEC_BUTTON_STACK_WIDTH,
        )
        self._filespec_browse_button.on_click(self._on_filespec_browse)
        self._filespec_update_button = self._pn.widgets.Button(
            name="Update All",
            button_type="primary",
            width=_FILESPEC_BUTTON_STACK_WIDTH,
        )
        self._filespec_update_button.on_click(self._on_update_file)
        self._filespec_button_stack = self._pn.Column(
            self._pn.VSpacer(),
            self._filespec_update_button,
            self._pn.VSpacer(),
            self._filespec_browse_button,
            self._pn.VSpacer(),
            width=_FILESPEC_BUTTON_STACK_WIDTH,
            height=112,
            margin=0,
        )
        self._measurement_table = self._pn.widgets.Tabulator(
            self._measurement_frame(),
            show_index=False,
            selectable=1,
            sortable=False,
            layout="fit_columns",
            sizing_mode="fixed",
            width=_MEASUREMENT_PAIR_TABLE_WIDTH,
            height=190,
            editors={
                "measurement": None,
            },
            titles=_MEASUREMENT_TABLE_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _MEASUREMENT_TABLE_TITLES
            },
        )
        self._measurement_table.param.watch(
            self._on_measurement_selection_changed,
            "selection",
        )
        self._specific_key_name_table = self._pn.widgets.Tabulator(
            self._specific_key_name_frame(),
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="fixed",
            width=_MEASUREMENT_PAIR_TABLE_WIDTH,
            height=190,
            editors={
                "specific_key": None,
            },
            titles=_SPECIFIC_KEY_NAME_TABLE_TITLES,
            title_formatters={
                key: {"type": "html"} for key in _SPECIFIC_KEY_NAME_TABLE_TITLES
            },
        )
        self._keys_table = self._pn.widgets.Tabulator(
            self._keys_preview_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            layout="fit_columns",
            sizing_mode="fixed",
            width=_MEASUREMENT_PAIR_TABLE_WIDTH,
            height=278,
            widths={
                "trace_index": 70,
            },
            editors={
                "trace_index": None,
                "specific_key": None,
                "yvalue": None,
            },
            titles=self._keys_preview_titles(),
            title_formatters={
                key: {"type": "html"} for key in _SPECIFIC_KEY_TABLE_TITLES
            },
        )
        self._filespec_table = self._pn.widgets.Tabulator(
            self._filespec_frame(),
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_FILESPEC_TABLE_WIDTH,
            height=112,
            widths={
                "parameter": 140,
            },
            titles=_TRACES_TABLE_TITLES,
            title_formatters={key: {"type": "html"} for key in _TRACES_TABLE_TITLES},
        )
        self._keysspec_table = self._pn.widgets.Tabulator(
            self._keysspec_frame(),
            show_index=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_MEASUREMENT_PAIR_TABLE_WIDTH,
            height=278,
            widths={
                "parameter": 140,
            },
            editors={
                "parameter": None,
                "value": {"type": "input"},
            },
            titles=_TRACES_TABLE_TITLES,
            title_formatters={key: {"type": "html"} for key in _TRACES_TABLE_TITLES},
        )
        self._keysspec_table.on_edit(self._on_keysspec_edit)
        self._tracespec_table = self._pn.widgets.Tabulator(
            self._tracespec_frame(),
            show_index=False,
            disabled=True,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_MEASUREMENT_PAIR_TABLE_WIDTH,
            height=220,
            widths={
                "parameter": 150,
            },
            titles=_TRACES_TABLE_TITLES,
            title_formatters={key: {"type": "html"} for key in _TRACES_TABLE_TITLES},
        )
        self._tracespec_core_table = self._pn.widgets.Tabulator(
            self._tracespec_core_frame(),
            show_index=False,
            disabled=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_MEASUREMENT_PAIR_TABLE_WIDTH,
            height=112,
            widths={
                "parameter": 150,
            },
            editors={
                "parameter": None,
                "value": {"type": "input"},
            },
            titles=_TRACES_TABLE_TITLES,
            title_formatters={key: {"type": "html"} for key in _TRACES_TABLE_TITLES},
        )
        self._tracespec_core_table.on_edit(self._on_tracespec_core_edit)
        self._tracespec_other_table = self._pn.widgets.Tabulator(
            self._tracespec_other_frame(),
            show_index=False,
            disabled=False,
            selectable=False,
            sortable=False,
            hidden_columns=["key"],
            layout="fit_columns",
            sizing_mode="fixed",
            width=_MEASUREMENT_PAIR_TABLE_WIDTH,
            height=148,
            widths={
                "parameter": 150,
            },
            editors={
                "parameter": None,
                "value": {"type": "input"},
            },
            titles=_TRACES_TABLE_TITLES,
            title_formatters={key: {"type": "html"} for key in _TRACES_TABLE_TITLES},
        )
        self._tracespec_other_table.on_edit(self._on_tracespec_other_edit)

    def _measurement_tab(self):
        return self._pn.Column(
            self._traces_file_heading,
            self._pn.Row(
                self._filespec_table,
                self._filespec_button_stack,
                sizing_mode="fixed",
                width=_MEASUREMENT_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_FILESPEC_ROW_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_MEASUREMENT_SECTION_GAP_HEIGHT),
            self._traces_measurement_heading,
            self._pn.Row(
                self._measurement_table,
                self._specific_key_name_table,
                sizing_mode="fixed",
                width=_MEASUREMENT_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_MEASUREMENT_PAIR_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_MEASUREMENT_SECTION_GAP_HEIGHT),
            self._traces_keys_heading,
            self._pn.Row(
                self._keysspec_table,
                self._keys_table,
                sizing_mode="fixed",
                width=_MEASUREMENT_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_MEASUREMENT_PAIR_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_MEASUREMENT_SECTION_GAP_HEIGHT),
            self._traces_spec_heading,
            self._pn.Row(
                self._tracespec_core_table,
                self._tracespec_other_table,
                sizing_mode="fixed",
                width=_MEASUREMENT_ROW_WIDTH,
                margin=0,
                styles={"gap": f"{_MEASUREMENT_PAIR_GAP_WIDTH}px"},
            ),
            self._pn.Spacer(height=_MEASUREMENT_ERROR_GAP_HEIGHT),
            self._keysspec_error,
            self._tracespec_error,
            sizing_mode="stretch_width",
        )

    def _filespec_frame(self) -> pd.DataFrame:
        spec: FileSpec | None = self._filespec
        return pd.DataFrame(
            [
                {
                    "key": "h5path",
                    "parameter": _FILE_PARAMETER_LABELS["h5path"],
                    "value": _trace_table_value(
                        None if spec is None else spec.h5path,
                    ),
                },
                {
                    "key": "location",
                    "parameter": _FILE_PARAMETER_LABELS["location"],
                    "value": _trace_table_value(
                        None if spec is None else spec.location,
                    ),
                },
                {
                    "key": "measurement",
                    "parameter": _FILE_PARAMETER_LABELS["measurement"],
                    "value": _trace_table_value(
                        None if spec is None else spec.measurement,
                    ),
                },
            ],
            dtype=object,
        )

    def _specific_key_name_frame(self) -> pd.DataFrame:
        if self._filespec is None:
            keys: list[str] = []
        else:
            try:
                keys = list(self._filespec.skeys())
            except Exception:
                keys = []
        return pd.DataFrame(
            {
                "specific_key": list(keys),
            },
            dtype=object,
        )

    def _keysspec_frame(self) -> pd.DataFrame:
        spec = self._effective_keysspec()
        return pd.DataFrame(
            [
                {
                    "key": "strip0",
                    "parameter": _KEYS_PARAMETER_LABELS["strip0"],
                    "value": _trace_table_value(None if spec is None else spec.strip0),
                },
                {
                    "key": "strip1",
                    "parameter": _KEYS_PARAMETER_LABELS["strip1"],
                    "value": _trace_table_value(None if spec is None else spec.strip1),
                },
                {
                    "key": "remove_key",
                    "parameter": _KEYS_PARAMETER_LABELS["remove_key"],
                    "value": _trace_table_value(
                        None if spec is None else spec.remove_key,
                    ),
                },
                {
                    "key": "add_key",
                    "parameter": _KEYS_PARAMETER_LABELS["add_key"],
                    "value": _trace_table_value(None if spec is None else spec.add_key),
                },
                {
                    "key": "norm",
                    "parameter": _KEYS_PARAMETER_LABELS["norm"],
                    "value": _trace_table_value(None if spec is None else spec.norm),
                },
                {
                    "key": "label",
                    "parameter": _KEYS_PARAMETER_LABELS["label"],
                    "value": _trace_table_value(None if spec is None else spec.label),
                },
                {
                    "key": "html_label",
                    "parameter": _KEYS_PARAMETER_LABELS["html_label"],
                    "value": _trace_table_value(
                        None if spec is None else spec.html_label,
                    ),
                },
                {
                    "key": "limits",
                    "parameter": _KEYS_PARAMETER_LABELS["limits"],
                    "value": _trace_table_value(None if spec is None else spec.limits),
                },
            ],
            dtype=object,
        )

    @staticmethod
    def _parse_keysspec_value(
        *,
        key: str,
        value: object,
    ) -> object:
        """Parse one editable ``KeysSpec`` table value."""
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                return None
        else:
            text = None

        if key in {"strip0", "strip1", "label", "html_label"}:
            if text is None:
                return value
            return text

        if key == "norm":
            if text is not None:
                return float(text)
            return float(value)

        if key in {"remove_key", "add_key", "limits"}:
            if text is None:
                return value
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                if key == "remove_key":
                    return text
                raise ValueError(f"{key} must be a valid Python literal.") from None
            if key in {"remove_key", "add_key"} and parsed in ((), [], set(), {}):
                return None
            return parsed

        return value

    def _build_keysspec_from_frame(self, frame: pd.DataFrame) -> KeysSpec:
        """Build one ``KeysSpec`` from one table frame."""
        frame = frame.reset_index(drop=True).set_index("key")
        values: dict[str, object] = {}
        for key in _KEYS_PARAMETER_LABELS:
            parsed = self._parse_keysspec_value(
                key=key,
                value=frame.at[key, "value"],
            )
            if key == "strip0":
                values[key] = "" if parsed is None else str(parsed)
            elif key in {"strip1", "label", "html_label"}:
                values[key] = None if parsed is None else str(parsed)
            else:
                values[key] = parsed
        return KeysSpec(**values)

    def _tracespec_frame(self) -> pd.DataFrame:
        spec: TraceSpec | None = self._tracespec
        return pd.DataFrame(
            [
                {
                    "key": "amp_voltage",
                    "parameter": _TRACE_PARAMETER_LABELS["amp_voltage"],
                    "value": _trace_table_value(
                        None if spec is None else spec.amp_voltage,
                    ),
                },
                {
                    "key": "amp_current",
                    "parameter": _TRACE_PARAMETER_LABELS["amp_current"],
                    "value": _trace_table_value(
                        None if spec is None else spec.amp_current,
                    ),
                },
                {
                    "key": "r_ref_ohm",
                    "parameter": _TRACE_PARAMETER_LABELS["r_ref_ohm"],
                    "value": _trace_table_value(
                        None if spec is None else spec.r_ref_ohm,
                    ),
                },
                {
                    "key": "trigger_values",
                    "parameter": _TRACE_PARAMETER_LABELS["trigger_values"],
                    "value": _trace_table_value(
                        None if spec is None else spec.trigger_values,
                    ),
                },
                {
                    "key": "skip",
                    "parameter": _TRACE_PARAMETER_LABELS["skip"],
                    "value": _trace_table_value(None if spec is None else spec.skip),
                },
                {
                    "key": "subtract_offset",
                    "parameter": _TRACE_PARAMETER_LABELS["subtract_offset"],
                    "value": _trace_table_value(
                        None if spec is None else spec.subtract_offset,
                    ),
                },
                {
                    "key": "time_relative",
                    "parameter": _TRACE_PARAMETER_LABELS["time_relative"],
                    "value": _trace_table_value(
                        None if spec is None else spec.time_relative,
                    ),
                },
            ],
            dtype=object,
        )

    def _tracespec_core_frame(self) -> pd.DataFrame:
        """Return the core gain/reference trace extraction parameters."""
        frame = self._tracespec_frame().reset_index(drop=True)
        return frame.loc[
            frame["key"].isin(("amp_voltage", "amp_current", "r_ref_ohm"))
        ].reset_index(drop=True)

    def _tracespec_other_frame(self) -> pd.DataFrame:
        """Return the remaining trace extraction parameters."""
        frame = self._tracespec_frame().reset_index(drop=True)
        return frame.loc[
            frame["key"].isin(
                (
                    "trigger_values",
                    "skip",
                    "subtract_offset",
                    "time_relative",
                )
            )
        ].reset_index(drop=True)

    @staticmethod
    def _parse_tracespec_value(
        *,
        key: str,
        value: object,
    ) -> object:
        """Parse one editable ``TraceSpec`` table value."""
        if isinstance(value, str):
            text = value.strip()
            if text == "":
                value = None
            else:
                value = text

        if key in {"amp_voltage", "amp_current", "r_ref_ohm"}:
            parsed = float(value)
            if not math.isfinite(parsed):
                raise ValueError(f"{key} must be finite.")
            return parsed

        if key == "trigger_values":
            if value is None:
                return None
            try:
                parsed = ast.literal_eval(value) if isinstance(value, str) else value
            except (SyntaxError, ValueError):
                parsed = value
            if isinstance(parsed, int) and not isinstance(parsed, bool):
                return int(parsed)
            if isinstance(parsed, list):
                parsed = tuple(parsed)
            if isinstance(parsed, tuple) and all(
                isinstance(item, int) and not isinstance(item, bool) for item in parsed
            ):
                return parsed
            raise ValueError(
                "trigger_values must be an int, a tuple/list of ints, or empty."
            )

        if key == "skip":
            if value is None:
                raise ValueError("skip must not be empty.")
            try:
                parsed = ast.literal_eval(value) if isinstance(value, str) else value
            except (SyntaxError, ValueError):
                parsed = value
            if isinstance(parsed, int) and not isinstance(parsed, bool):
                return int(parsed)
            if isinstance(parsed, list):
                parsed = tuple(parsed)
            if (
                isinstance(parsed, tuple)
                and len(parsed) == 2
                and all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in parsed
                )
            ):
                return parsed
            raise ValueError("skip must be an int or a tuple/list of two ints.")

        if key in {"subtract_offset", "time_relative"}:
            if isinstance(value, bool):
                return value
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered in {"true", "1", "yes", "y"}:
                    return True
                if lowered in {"false", "0", "no", "n"}:
                    return False
            raise ValueError(f"{key} must be a boolean.")

        return value

    def _measurement_names(self) -> list[str]:
        if self._filespec is None:
            return []
        try:
            return list(self._filespec.mkeys())
        except Exception:
            return []

    def _measurement_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"measurement": self._measurement_names()},
            dtype=object,
        )

    def _keys_preview_frame(self) -> pd.DataFrame:
        preview = getattr(self, "_keys_preview", None)
        metas = [] if preview is None else list(preview.metas)
        return pd.DataFrame(
            {
                "trace_index": [
                    ("" if meta.index is None else int(meta.index)) for meta in metas
                ],
                "specific_key": [meta.specific_key for meta in metas],
                "yvalue": [_trace_yvalue_value(meta.yvalue) for meta in metas],
            },
            dtype=object,
        )

    def _sync_trace_widgets_from_state(self) -> None:
        self._keys_preview = self._preview_keys()
        measurement_frame = self._measurement_frame()
        self._measurement_table.value = measurement_frame
        selected_measurement = None
        if self._filespec is not None:
            selected_measurement = self._filespec.measurement
        if selected_measurement is None:
            self._measurement_table.selection = []
        else:
            matches = measurement_frame.index[
                measurement_frame["measurement"] == selected_measurement
            ].tolist()
            self._measurement_table.selection = matches[:1]
        self._keys_table.value = self._keys_preview_frame()
        self._keys_table.titles = self._keys_preview_titles()
        self._specific_key_name_table.value = self._specific_key_name_frame()
        self._filespec_table.value = self._filespec_frame()
        if (
            not hasattr(self, "_keysspec_draft_frame")
            or self._keysspec_draft_frame is None
        ):
            self._keysspec_draft_frame = self._keysspec_frame()
        self._set_keysspec_table_value(self._keysspec_draft_frame.copy())
        self._set_tracespec_table_values()

    def _set_keysspec_table_value(self, frame: pd.DataFrame) -> None:
        """Set the editable KeysSpec table without re-entering the watcher."""
        self._suppress_keysspec_table_watch = True
        try:
            self._keysspec_table.value = frame.copy()
        finally:
            self._suppress_keysspec_table_watch = False

    def _set_keysspec_error(self, message: str) -> None:
        """Show one inline validation error for the KeysSpec table."""
        if message == "":
            self._keysspec_error.object = ""
            self._keysspec_error.visible = False
            return
        self._keysspec_error.object = (
            '<div style="color: #b42318; font-size: 12px; line-height: 1.4;">'
            f"{message}"
            "</div>"
        )
        self._keysspec_error.visible = True

    def _set_tracespec_error(self, message: str) -> None:
        """Show one inline validation error for the TraceSpec tables."""
        if message == "":
            self._tracespec_error.object = ""
            self._tracespec_error.visible = False
            return
        self._tracespec_error.object = (
            '<div style="color: #b42318; font-size: 12px; line-height: 1.4;">'
            f"{message}"
            "</div>"
        )
        self._tracespec_error.visible = True

    def _set_tracespec_table_values(self) -> None:
        """Sync the hidden/full and visible TraceSpec tables from state."""
        self._tracespec_table.value = self._tracespec_frame()
        self._tracespec_core_table.value = self._tracespec_core_frame()
        self._tracespec_other_table.value = self._tracespec_other_frame()

    def _apply_tracespec_field_edit(self, *, key: str, value: object) -> None:
        """Validate and store one edited TraceSpec field."""
        spec = TraceSpec() if self._tracespec is None else self._tracespec
        values = asdict(spec)
        try:
            values[key] = self._parse_tracespec_value(key=key, value=value)
        except (TypeError, ValueError) as exc:
            self._set_tracespec_error(str(exc))
            self._set_tracespec_table_values()
            return
        self._tracespec = TraceSpec(**values)
        self._set_tracespec_error("")
        self._set_tracespec_table_values()

    def _on_tracespec_table_edit(self, table: object, event: object) -> None:
        """Handle one TraceSpec table edit event."""
        if str(getattr(event, "column", "")) != "value":
            return
        try:
            row = int(getattr(event, "row"))
        except (TypeError, ValueError):
            return
        frame = table.value.reset_index(drop=True)
        if row < 0 or row >= len(frame):
            return
        key = str(frame.at[row, "key"])
        self._apply_tracespec_field_edit(
            key=key,
            value=getattr(event, "value"),
        )

    def _on_tracespec_core_edit(self, event: object) -> None:
        """Handle edits for the core TraceSpec table."""
        self._on_tracespec_table_edit(self._tracespec_core_table, event)

    def _on_tracespec_other_edit(self, event: object) -> None:
        """Handle edits for the auxiliary TraceSpec table."""
        self._on_tracespec_table_edit(self._tracespec_other_table, event)

    def _reload_file_selection(self) -> None:
        """Reload keys and traces from the current file-related specs."""
        if self._filespec is None:
            return

        loaded_keys = get_keys(
            filespec=self._filespec,
            keysspec=self._keysspec,
        )
        self._keys = loaded_keys
        loaded_traces = get_traces(
            filespec=self._filespec,
            keys=loaded_keys,
            tracespec=self._tracespec,
        )

        self.traces = loaded_traces
        self.active_index = 0
        self._sampling_offset_override_enabled = np.zeros(
            len(self.traces),
            dtype=bool,
        )
        self._initialize_sampling_offset_overrides()
        self._clear_psd_stage_cache()
        self._clear_sampling_stage_cache()
        self._clear_offset_batch_cache()

        options = OrderedDict(
            (_trace_label(index, trace), index)
            for index, trace in enumerate(self.traces)
        )
        previous_value = int(self._trace_selector.value)
        self._trace_selector.options = options
        self._trace_selector.value = 0
        self._sync_trace_widgets_from_state()
        if previous_value == 0:
            self._recompute_pipeline(clear_fit=True)
            self._sync_control_widgets_from_specs()
            self._refresh_all_views()
            self._notify_state_changed()

    def _coerce_keysspec_value(self, *, key: str, value: object) -> object:
        """Parse and coerce one edited KeysSpec field value."""
        parsed = self._parse_keysspec_value(key=key, value=value)
        if key == "strip0":
            return "" if parsed is None else str(parsed)
        if key in {"strip1", "label", "html_label"}:
            return None if parsed is None else str(parsed)
        return parsed

    def _apply_keysspec_field_edit(self, *, key: str, value: object) -> None:
        """Validate one edited KeysSpec field and keep it as draft."""
        frame = (
            self._keysspec_draft_frame.copy().reset_index(drop=True)
            if hasattr(self, "_keysspec_draft_frame")
            and self._keysspec_draft_frame is not None
            else self._keysspec_frame().reset_index(drop=True)
        )
        rows = frame.index[frame["key"] == key].tolist()
        if len(rows) != 1:
            return
        row = int(rows[0])

        try:
            self._coerce_keysspec_value(key=key, value=value)
        except ValueError as exc:
            self._set_keysspec_error(str(exc))
            self._set_keysspec_table_value(frame)
            return

        self._set_keysspec_error("")
        frame.at[row, "value"] = value
        self._keysspec_draft_frame = frame
        self._set_keysspec_table_value(self._keysspec_draft_frame)

    def _on_keysspec_edit(self, event: object) -> None:
        if str(getattr(event, "column", "")) != "value":
            return
        try:
            row = int(getattr(event, "row"))
        except (TypeError, ValueError):
            return
        frame = self._keysspec_table.value.reset_index(drop=True)
        if row < 0 or row >= len(frame):
            return
        key = str(frame.at[row, "key"])
        self._apply_keysspec_field_edit(
            key=key,
            value=getattr(event, "value"),
        )

    def _apply_selected_measurement(self) -> None:
        """Reload the current file selection after one measurement change."""
        self._sync_control_widgets_from_specs()
        if self._filespec is None or self._fit_running or self._offset_batch_running:
            return
        try:
            self._reload_file_selection()
        except Exception:
            self._sync_control_widgets_from_specs()

    def _on_filespec_browse(self, _: object) -> None:
        selected = _browse_h5_file(self._filespec)
        if selected is None:
            return

        measurement = None if self._filespec is None else self._filespec.measurement
        location: str | Path | None
        h5path: str | Path
        if self._filespec is not None and self._filespec.location is not None:
            location_path = Path(self._filespec.location).expanduser()
            try:
                h5path = str(selected.relative_to(location_path))
                location = self._filespec.location
            except ValueError:
                location = str(selected.parent)
                h5path = selected.name
        else:
            location = str(selected.parent)
            h5path = selected.name

        self._filespec = FileSpec(
            h5path=h5path,
            location=location,
            measurement=measurement,
        )
        self._sync_control_widgets_from_specs()

    def _on_update_file(self, _: object) -> None:
        if self._filespec is None or self._fit_running or self._offset_batch_running:
            return
        draft = (
            self._keysspec_draft_frame.copy()
            if hasattr(self, "_keysspec_draft_frame")
            and self._keysspec_draft_frame is not None
            else self._keysspec_frame()
        )
        try:
            self._keysspec = self._build_keysspec_from_frame(draft)
        except ValueError as exc:
            self._set_keysspec_error(str(exc))
            self._set_keysspec_table_value(draft)
            return
        self._set_keysspec_error("")
        self._keysspec_draft_frame = self._keysspec_frame()
        self._set_keysspec_table_value(self._keysspec_draft_frame)
        self._reload_file_selection()

    def _on_measurement_selection_changed(self, event: object) -> None:
        if self._filespec is None:
            return
        selection = list(getattr(event, "new", []))
        if len(selection) == 0:
            return
        row = int(selection[0])
        frame = self._measurement_table.value.reset_index(drop=True)
        if row < 0 or row >= len(frame):
            return
        measurement = str(frame.at[row, "measurement"])
        if measurement == self._filespec.measurement:
            return
        self._filespec = FileSpec(
            h5path=self._filespec.h5path,
            location=self._filespec.location,
            measurement=measurement,
        )
        self._apply_selected_measurement()
