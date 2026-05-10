"""Visualization browser page for TransportLab."""

from __future__ import annotations

from typing import Any

from ..utilities.cache import cache_summary


def visualization_app(pn: Any, session: Any):
    """Build the barebone visualization page."""
    rows = _table_frame(_active_cache_summary(session))
    selectable = pn.widgets.Tabulator(
        rows,
        disabled=True,
        height=260,
        sizing_mode="stretch_width",
    )
    return pn.Column(
        pn.pane.Markdown("# TransportLab Visualization"),
        pn.pane.Markdown(
            "Select cache entries here for visualization. Plot rendering will be "
            "added step by step.",
        ),
        pn.pane.Markdown(f"**Active cache:** {_active_cache_label(session)}"),
        selectable,
        sizing_mode="stretch_width",
    )


def _active_cache_summary(session: Any) -> tuple[dict[str, object], ...]:
    if session.cache is None:
        return ()
    return cache_summary(session.cache)


def _active_cache_label(session: Any) -> str:
    if session.cache is None:
        return "`None`"
    return f"`{session.cache.name}`"


def _table_frame(rows: tuple[dict[str, object], ...]):
    import pandas as pd

    return pd.DataFrame(
        list(rows),
        columns=("key", "kind", "type", "summary"),
    )
