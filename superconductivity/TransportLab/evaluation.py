"""Evaluation workspace tab for TransportLab."""

from __future__ import annotations

from typing import Any


def evaluation_tab(pn: Any, session: Any):
    """Build the evaluation workspace tab."""
    return pn.Column(
        pn.pane.Markdown("## Evaluation"),
        pn.pane.Markdown("This workspace tab is intentionally empty for now."),
        sizing_mode="stretch_width",
    )
