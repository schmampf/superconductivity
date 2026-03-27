from __future__ import annotations

from copy import deepcopy
from typing import Sequence

from ..style.cpd5 import rot, schwarz, seeblau65, seeblau100, seegrau65, seegrau100
from ..style.plotly import mpl_color_to_plotly

GUI_LINE_WIDTH = 2.0
GUI_MARKER_SIZE = 4

_GUI_COLORS = {
    "raw": mpl_color_to_plotly(seegrau65),
    "downsampled": mpl_color_to_plotly(seegrau100),
    "binned": mpl_color_to_plotly(schwarz),
    "initial": mpl_color_to_plotly(seeblau65),
    "fit": mpl_color_to_plotly(seeblau100),
    "cutoff": mpl_color_to_plotly(rot),
}

_GUI_TRACE_LABELS = {
    "raw": "Raw",
    "downsampled": "Downsampled",
    "binned": "Binned",
    "initial": "Initial",
    "fit": "Fit",
    "cutoff": "&sigma; cutoff",
}

_GUI_TRACE_STYLES = {
    "raw": {
        "mode": "lines",
        "line": {"color": _GUI_COLORS["raw"], "width": GUI_LINE_WIDTH},
    },
    "downsampled": {
        "mode": "lines",
        "line": {
            "color": _GUI_COLORS["downsampled"],
            "width": GUI_LINE_WIDTH,
        },
    },
    "binned": {
        "mode": "lines+markers",
        "line": {"color": _GUI_COLORS["binned"], "width": GUI_LINE_WIDTH},
        "marker": {
            "size": GUI_MARKER_SIZE,
            "color": _GUI_COLORS["binned"],
        },
    },
    "initial": {
        "mode": "lines",
        "line": {
            "color": _GUI_COLORS["initial"],
            "width": GUI_LINE_WIDTH,
            "dash": "dash",
        },
    },
    "fit": {
        "mode": "lines",
        "line": {"color": _GUI_COLORS["fit"], "width": GUI_LINE_WIDTH},
    },
    "cutoff": {
        "line": {
            "color": _GUI_COLORS["cutoff"],
            "width": GUI_LINE_WIDTH,
            "dash": "dash",
        },
    },
}


def gui_trace_style(name: str) -> dict[str, object]:
    """Return a copy of the configured GUI trace style."""
    return deepcopy(_GUI_TRACE_STYLES[name])


def gui_trace_label(name: str) -> str:
    """Return the display label for a GUI trace style."""
    return _GUI_TRACE_LABELS[name]


def gui_legend_html(
    names: Sequence[str],
    *,
    direction: str = "row",
    gap_px: int = 12,
) -> str:
    """Build a small HTML legend from the configured GUI styles."""
    items: list[str] = []
    for name in names:
        style = _GUI_TRACE_STYLES[name]
        line = style["line"]
        color = str(line["color"])
        dash = str(line.get("dash", "solid"))
        border_style = "dashed" if dash == "dash" else "solid"
        marker = style.get("marker")
        if marker is None:
            swatch = (
                "<span style=\"display:inline-block; width:26px; "
                f"border-top:{GUI_LINE_WIDTH}px {border_style} {color};\"></span>"
            )
        else:
            swatch = (
                "<span style=\"position:relative; display:inline-block; "
                "width:26px; height:10px;\">"
                f"<span style=\"position:absolute; left:0; right:0; top:4px; "
                f"border-top:{GUI_LINE_WIDTH}px {border_style} {color};\"></span>"
                f"<span style=\"position:absolute; left:10px; top:1px; "
                f"width:{GUI_MARKER_SIZE + 2}px; height:{GUI_MARKER_SIZE + 2}px; "
                f"background:{color}; border-radius:50%;\"></span>"
                "</span>"
            )
        items.append(
            "<span style=\"display:flex; align-items:center; gap:8px;\">"
            f"{swatch}{gui_trace_label(name)}"
            "</span>"
        )
    return (
        "<div style=\"display:flex; "
        f"flex-direction:{direction}; flex-wrap:wrap; gap:{gap_px}px; "
        "align-items:flex-start;\">"
        + "".join(items)
        + "</div>"
    )
