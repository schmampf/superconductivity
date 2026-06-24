"""Cache workspace tab for TransportLab."""

from __future__ import annotations

import subprocess
from pathlib import Path
from shutil import which
from typing import Any

from ..utilities.cache import list_caches, load_cache


def cache_tab(pn: Any, session: Any):
    """Build the cache workspace tab."""
    button = pn.widgets.Button(name="Choose folder", button_type="primary")
    path_field = pn.widgets.TextInput(
        name="Project path",
        value=str(session.project_path),
        sizing_mode="stretch_width",
    )
    table = pn.widgets.Tabulator(
        _cache_frame(session.project_path),
        selectable=1,
        show_index=False,
        sortable=False,
        editors={"name": None},
        layout="fit_columns",
        sizing_mode="stretch_width",
        height=280,
    )
    keys_table = pn.widgets.Tabulator(
        _keys_frame(None),
        show_index=False,
        sortable=False,
        editors={"key": None},
        layout="fit_columns",
        sizing_mode="stretch_width",
        height=200,
    )
    subkeys_table = pn.widgets.Tabulator(
        _keys_frame(None),
        show_index=False,
        sortable=False,
        editors={"key": None},
        layout="fit_columns",
        sizing_mode="stretch_width",
        height=200,
    )
    output = pn.pane.Markdown("", sizing_mode="stretch_width")

    def _choose_project_path(_event: Any) -> None:
        chosen = _pick_folder(session.project_path)
        if chosen is None:
            return
        session.project_path = chosen
        path_field.value = str(chosen)
        _refresh_table(table, session)
        _select_current_cache(table, session)

    def _sync_project_path(event: Any) -> None:
        value = str(getattr(event, "new", "")).strip()
        if value:
            session.project_path = Path(value)
            _refresh_table(table, session)
            _select_current_cache(table, session)

    def _sync_selected_cache(event: Any) -> None:
        selection = list(getattr(event, "new", []) or [])
        if not selection:
            keys_table.value = _keys_frame(None)
            subkeys_table.value = _keys_frame(None)
            output.object = ""
            return
        frame = table.value.reset_index(drop=True)
        row = int(selection[0])
        if row < 0 or row >= len(frame):
            return
        selected = frame.at[row, "name"]
        _load_selected_cache(selected, session)
        _notify_cache_changed(session)
        keys_table.value = _keys_frame(session.cache)
        subkeys_table.value = _keys_frame(None)
        output.object = ""

    def _sync_selected_key(event: Any) -> None:
        selection = list(getattr(event, "new", []) or [])
        if not selection or session.cache is None:
            subkeys_table.value = _keys_frame(None)
            output.object = ""
            return
        frame = keys_table.value.reset_index(drop=True)
        row = int(selection[0])
        if row < 0 or row >= len(frame):
            subkeys_table.value = _keys_frame(None)
            output.object = ""
            return
        key = str(frame.at[row, "key"])
        subkeys_table.value = _keys_frame(_selected_value(session.cache, key))
        output.object = ""

    def _sync_selected_subkey(event: Any) -> None:
        selection = list(getattr(event, "new", []) or [])
        if not selection or session.cache is None:
            output.object = ""
            return
        key_frame = keys_table.value.reset_index(drop=True)
        subkey_frame = subkeys_table.value.reset_index(drop=True)
        key_selection = list(keys_table.selection or [])
        row = int(selection[0])
        if not key_selection or row < 0 or row >= len(subkey_frame):
            output.object = ""
            return
        key = str(key_frame.at[int(key_selection[0]), "key"])
        subkey = str(subkey_frame.at[row, "key"])
        value = _selected_value(_selected_value(session.cache, key), subkey)
        output.object = _output_text(value)

    button.on_click(_choose_project_path)
    path_field.param.watch(_sync_project_path, "value")
    table.param.watch(_sync_selected_cache, "selection")
    keys_table.param.watch(_sync_selected_key, "selection")
    subkeys_table.param.watch(_sync_selected_subkey, "selection")
    return pn.Column(
        pn.Row(button, path_field, sizing_mode="stretch_width"),
        table,
        keys_table,
        subkeys_table,
        output,
        sizing_mode="stretch_width",
    )


def _pick_folder(initial_path: Path) -> Path | None:
    if which("osascript") is None:
        return None

    initial_dir = _apple_string(str(initial_path.expanduser()))
    script = f"""
        set chosenItem to choose folder with prompt "Select project folder" default location POSIX file {initial_dir} as alias
        POSIX path of chosenItem
    """
    result = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    chosen = result.stdout.strip()
    return Path(chosen) if chosen else None


def _cache_frame(path: Path):
    import pandas as pd

    rows = [{"name": name} for name in list_caches(path)]
    return pd.DataFrame(rows, columns=("name",))


def _keys_frame(cache: Any):
    import pandas as pd

    if cache is None:
        return pd.DataFrame([], columns=("key",))
    keys = getattr(cache, "keys", None)
    if not callable(keys):
        return pd.DataFrame([], columns=("key",))
    try:
        rows = [{"key": key} for key in keys()]
    except Exception:
        return pd.DataFrame([], columns=("key",))
    return pd.DataFrame(rows, columns=("key",))


def _selected_value(cache: Any, key: str) -> Any:
    if cache is None:
        return None
    try:
        return cache[key]
    except Exception:
        pass
    try:
        return getattr(cache, key)
    except Exception:
        return None


def _output_text(value: Any) -> str:
    return f"```python\n{repr(value)}\n```"


def _refresh_table(table: Any, session: Any) -> None:
    table.value = _cache_frame(session.project_path)


def _select_current_cache(table: Any, session: Any) -> None:
    if session.cache is None:
        table.selection = []
        return
    frame = table.value.reset_index(drop=True)
    matches = frame.index[frame["name"] == session.cache.name].tolist()
    table.selection = matches[:1]


def _load_selected_cache(selected: str, session: Any):
    try:
        session.cache = load_cache(selected, path=session.project_path)
    except Exception:
        session.cache = None
    return session.cache


def _notify_cache_changed(session: Any) -> None:
    notify = getattr(session, "notify_cache_changed", None)
    if callable(notify):
        notify()


def _apple_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
