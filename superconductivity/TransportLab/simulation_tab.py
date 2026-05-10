"""Simulation workspace tab for TransportLab."""

from __future__ import annotations

from typing import Any


def simulation_tab(pn: Any, session: Any):
    """Build the simulation workspace tab."""
    return pn.Column(
        pn.pane.Markdown("## Simulation"),
        pn.pane.Markdown("This workspace tab is intentionally empty for now."),
        sizing_mode="stretch_width",
    )
