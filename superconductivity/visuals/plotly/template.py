"""Shared Plotly template configuration for superconductivity plots."""

import plotly.graph_objects as go
import plotly.io as pio
from matplotlib.colors import ListedColormap

from superconductivity.style.cpd4 import cmap
from superconductivity.style.cpd5 import (
    mpl_cmap_to_plotly,
    mpl_color_to_plotly,
    seeblau100,
    seegrau35,
    seegrau65,
    seegrau100,
    weiss,
)
from superconductivity.utilities.types import COLOR


def build_house_template() -> go.layout.Template:
    """Build the shared Plotly template.

    Returns
    -------
    go.layout.Template
        Plotly template containing project-wide layout and trace defaults.
    """
    axiscolor_mpl: COLOR = seegrau100
    axiscolor = mpl_color_to_plotly(axiscolor_mpl)

    gridcolor_mpl: COLOR = seegrau65
    gridcolor = mpl_color_to_plotly(gridcolor_mpl)

    backgroundcolor_mpl: COLOR = weiss
    backgroundcolor = mpl_color_to_plotly(backgroundcolor_mpl)

    colormap_mpl: ListedColormap = cmap()
    colormap = mpl_cmap_to_plotly(colormap_mpl)

    gridwidth: float = 1.0
    axiswidth: float = 1.0
    linewidth: float = 2.0

    tickformat: str = ".3g"
    fontsize: float = 14

    global_axis = dict(
        showgrid=True,
        gridcolor=gridcolor,
        gridwidth=gridwidth,
        showline=True,
        linecolor=axiscolor,
        linewidth=axiswidth,
        zeroline=False,
        tickformat=tickformat,
    )
    local_axis = dict(
        showgrid=True,
        gridcolor=gridcolor,
        tickformat=tickformat,
    )

    template = go.layout.Template()
    template.layout = dict(
        template="simple_white",
        paper_bgcolor=backgroundcolor,
        plot_bgcolor=backgroundcolor,
        font=dict(size=fontsize),
        margin=dict(
            l=50,
            r=0,
            t=0,
            b=50,
            pad=3,
        ),
        hoverlabel=dict(font_size=fontsize),
        xaxis=global_axis,
        yaxis=global_axis,
        scene=dict(
            bgcolor=backgroundcolor,
            aspectmode="cube",
            xaxis=local_axis,
            yaxis=local_axis,
            zaxis=local_axis,
            camera=dict(
                eye=dict(
                    x=1.6,
                    y=1.3,
                    z=0.9,
                )
            ),
        ),
    )
    template.data.scatter = [
        go.Scatter(
            mode="lines",
            line=dict(width=linewidth),
        ),
    ]
    template.data.heatmap = [
        go.Heatmap(
            colorscale=colormap,
            showscale=True,
        ),
    ]
    template.data.surface = [
        go.Surface(
            colorscale=colormap,
            showscale=False,
        ),
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
