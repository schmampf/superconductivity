"""Visualization browser page for TransportLab."""

from __future__ import annotations

from typing import Any


def visualization_app(pn: Any, session: Any):
    """Build the barebone visualization page."""
    return pn.Column(sizing_mode="stretch_width")
