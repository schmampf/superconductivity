"""Fitting workspace tab for TransportLab."""

from __future__ import annotations

from typing import Any


def fitting_tab(pn: Any, session: Any):
    """Build the fitting workspace tab."""
    return pn.Column(
        pn.pane.Markdown("## Fitting"),
        pn.pane.Markdown("This workspace tab is intentionally empty for now."),
        sizing_mode="stretch_width",
    )
