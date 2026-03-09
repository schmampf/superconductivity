"""Shared Plotly template configuration for superconductivity plots."""

import plotly.graph_objects as go
import plotly.io as pio

from superconductivity.style.cpd4 import cmap
from superconductivity.utilities.types import COLOR

from .helper import mpl_cmap_to_plotly, mpl_color_to_plotly


def build_house_template() -> go.layout.Template:
    """Build the shared Plotly template.

    Returns
    -------
    go.layout.Template
        Plotly template containing project-wide layout and trace defaults.
    """
    gridcolor_mpl: COLOR = (0, 0, 0, 0.15)
    gridwidth: float = 1.0
    backgroundcolor_mpl: COLOR = (1, 1, 1, 1)

    gridcolor = mpl_color_to_plotly(gridcolor_mpl)
    backgroundcolor = mpl_color_to_plotly(backgroundcolor_mpl)

    colormap = mpl_cmap_to_plotly(cmap_mpl=cmap())

    global_axis = dict(
        showgrid=True,
        gridcolor=gridcolor,
        gridwidth=gridwidth,
        showline=True,
        linecolor="rgba(0,0,0,0.35)",
        linewidth=1,
        zeroline=False,
        tickformat=".3g",
    )
    local_axis = dict(showgrid=True, gridcolor=gridcolor, tickformat=".3g")

    template = go.layout.Template()
    template.layout = dict(
        template="simple_white",
        paper_bgcolor=backgroundcolor,
        plot_bgcolor=backgroundcolor,
        font=dict(size=14),
        margin=dict(l=40, r=40, t=40, b=40, pad=4),
        hoverlabel=dict(font_size=12),
        xaxis=global_axis,
        yaxis=global_axis,
        scene=dict(
            bgcolor=backgroundcolor,
            aspectmode="cube",
            xaxis=local_axis,
            yaxis=local_axis,
            zaxis=local_axis,
            camera=dict(eye=dict(x=1.6, y=1.3, z=0.9)),
        ),
    )
    template.data.scatter = [go.Scatter(mode="lines", line=dict(width=2))]
    template.data.heatmap = [
        go.Heatmap(colorscale=colormap, showscale=True),
    ]
    template.data.surface = [
        go.Surface(colorscale=colormap, showscale=False),
    ]

    return template


def register_house_template(set_default: bool = True) -> None:
    """Register the shared Plotly template.

    Parameters
    ----------
    set_default
        If ``True``, set the registered template as Plotly's default.
    """
    pio.templates["house"] = build_house_template()
    if set_default:
        pio.templates.default = "house"
        pio.templates.default = "house"
