"""Plotly helpers for tuple/list trace data."""

from typing import Optional, Sequence

import numpy as np
import plotly.graph_objects as go

from superconductivity.utilities.types import LIM
from .maps import get_axis, save_figure


def _prepare_xy_trace_lists(
    x_list: Sequence[Sequence[float] | np.ndarray],
    y_list: Sequence[Sequence[float] | np.ndarray],
    x_name: str,
    y_name: str,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Convert list-like traces to finite float64 arrays.

    Parameters
    ----------
    x_list : Sequence[Sequence[float] | np.ndarray]
        Sequence of x-traces.
    y_list : Sequence[Sequence[float] | np.ndarray]
        Sequence of y-traces.
    x_name : str
        Name used in error messages for x-traces.
    y_name : str
        Name used in error messages for y-traces.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        Prepared ``(x_traces, y_traces)`` lists with matching lengths per
        trace and only finite values.
    """
    if len(x_list) != len(y_list):
        raise ValueError(f"{x_name} and {y_name} must have the same length.")
    if len(x_list) == 0:
        raise ValueError(f"{x_name} and {y_name} must not be empty.")

    x_out: list[np.ndarray] = []
    y_out: list[np.ndarray] = []

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        x_arr = np.asarray(x, dtype=np.float64).reshape(-1)
        y_arr = np.asarray(y, dtype=np.float64).reshape(-1)
        if x_arr.size != y_arr.size:
            raise ValueError(
                f"{x_name}[{i}] and {y_name}[{i}] must have the same size.",
            )

        finite = np.isfinite(x_arr) & np.isfinite(y_arr)
        x_f = x_arr[finite]
        y_f = y_arr[finite]
        if x_f.size == 0:
            raise ValueError(
                f"{x_name}[{i}] / {y_name}[{i}] has no finite points.",
            )
        x_out.append(x_f)
        y_out.append(y_f)

    return x_out, y_out


def get_tuple_slider(
    x_list: Sequence[Sequence[float] | np.ndarray],
    y_list: Sequence[Sequence[float] | np.ndarray],
    slider_values: Sequence[float] | np.ndarray,
    xlabel: str,
    ylabel: str,
    slider_label: str = "y",
    xlim: LIM = None,
    ylim: LIM = None,
    ylogscale: bool = False,
    name: Optional[str] = None,
    scheme: str = "standard",
    dataset: str = "dataset",
) -> go.Figure:
    """Create a slider figure from lists of x/y traces.

    Parameters
    ----------
    x_list : Sequence[Sequence[float] | np.ndarray]
        Sequence of x-traces, one trace per slider step.
    y_list : Sequence[Sequence[float] | np.ndarray]
        Sequence of y-traces with same outer length as ``x_list``.
    slider_values : Sequence[float] | np.ndarray
        Slider values associated with each trace.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    slider_label : str, default="y"
        Prefix shown in slider current-value display.
    xlim : LIM, default=None
        Optional x-axis limits.
    ylim : LIM, default=None
        Optional y-axis limits.
    ylogscale : bool, default=False
        If ``True``, use logarithmic scaling on the y-axis.
    name : str | None, default=None
        If provided, save the slider figure using this title.
    scheme : str, default="standard"
        HTML save scheme passed to :func:`save_figure`.
    dataset : str, default="dataset"
        Output directory passed to :func:`save_figure`.

    Returns
    -------
    go.Figure
        Plotly figure with one dynamic trace and slider steps.
    """
    slider_arr = np.asarray(slider_values, dtype=np.float64).reshape(-1)
    if slider_arr.size == 0:
        raise ValueError("slider_values must not be empty.")
    if not np.all(np.isfinite(slider_arr)):
        raise ValueError("slider_values must contain only finite values.")

    x_pre, y_pre = _prepare_xy_trace_lists(
        x_list=x_list,
        y_list=y_list,
        x_name="x_list",
        y_name="y_list",
    )
    if len(x_pre) != slider_arr.size:
        raise ValueError(
            "slider_values length must match number of traces.",
        )

    order = np.argsort(slider_arr)
    slider_sorted = slider_arr[order]
    x_sorted = [x_pre[i] for i in order]
    y_sorted_traces = [y_pre[i] for i in order]
    has_nonpositive = any(
        np.any(y_trace <= 0.0) for y_trace in y_sorted_traces
    )
    if ylogscale and has_nonpositive:
        raise ValueError(
            "y_list must be strictly positive when ylogscale=True.",
        )

    trace_dyn = go.Scatter(
        x=x_sorted[0],
        y=y_sorted_traces[0],
        mode="lines",
        name=f"{slider_label}={slider_sorted[0]:g}",
    )

    steps = []
    for k, slider_k in enumerate(slider_sorted):
        steps.append(
            dict(
                label=f"{slider_k:g}",
                method="restyle",
                args=[
                    {
                        "x": [x_sorted[k]],
                        "y": [y_sorted_traces[k]],
                        "name": [f"{slider_label}={slider_k:g}"],
                    },
                    [0],
                ],
            )
        )

    xaxis = get_axis(lim=xlim, label=xlabel)
    yaxis = get_axis(
        lim=ylim,
        label=ylabel,
        logscale=ylogscale,
    )

    layout = go.Layout(
        xaxis=xaxis,
        yaxis=yaxis,
        sliders=[
            dict(
                active=0,
                currentvalue={"prefix": f"{slider_label}: "},
                pad={"t": 20},
                steps=steps,
            )
        ],
    )

    fig = go.Figure(
        data=[trace_dyn],
        layout=layout,
        skip_invalid=False,
    )
    if name is not None:
        save_figure(
            fig=fig,
            title=name,
            scheme=scheme,
            out_dir=dataset,
        )

    return fig
