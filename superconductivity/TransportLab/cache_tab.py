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
            return
        frame = table.value.reset_index(drop=True)
        row = int(selection[0])
        if row < 0 or row >= len(frame):
            return
        selected = frame.at[row, "name"]
        _load_selected_cache(selected, session)
        keys_table.value = _keys_frame(session.cache)

    button.on_click(_choose_project_path)
    path_field.param.watch(_sync_project_path, "value")
    table.param.watch(_sync_selected_cache, "selection")
    return pn.Column(
        pn.Row(button, path_field, sizing_mode="stretch_width"),
        table,
        keys_table,
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
    rows = [{"key": key} for key in cache.keys()]
    return pd.DataFrame(rows, columns=("key",))


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


def _apple_string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"') + '"'
