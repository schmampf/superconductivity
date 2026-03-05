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


def _get_xy_slider_from_lists(
    x_list: Sequence[Sequence[float] | np.ndarray],
    y_list: Sequence[Sequence[float] | np.ndarray],
    y_values: Sequence[float] | np.ndarray,
    xlabel: str,
    ylabel: str,
    slider_label: str = "y",
    xlim: LIM = None,
    ylim: LIM = None,
) -> go.Figure:
    """Create a slider figure from lists of x/y traces.

    Parameters
    ----------
    x_list : Sequence[Sequence[float] | np.ndarray]
        Sequence of x-traces, one trace per slider step.
    y_list : Sequence[Sequence[float] | np.ndarray]
        Sequence of y-traces with same outer length as ``x_list``.
    y_values : Sequence[float] | np.ndarray
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

    Returns
    -------
    go.Figure
        Plotly figure with one dynamic trace and slider steps.
    """
    y_arr = np.asarray(y_values, dtype=np.float64).reshape(-1)
    if y_arr.size == 0:
        raise ValueError("y_values must not be empty.")
    if not np.all(np.isfinite(y_arr)):
        raise ValueError("y_values must contain only finite values.")

    x_pre, y_pre = _prepare_xy_trace_lists(
        x_list=x_list,
        y_list=y_list,
        x_name="x_list",
        y_name="y_list",
    )
    if len(x_pre) != y_arr.size:
        raise ValueError(
            "y_values length must match number of traces.",
        )

    order = np.argsort(y_arr)
    y_sorted = y_arr[order]
    x_sorted = [x_pre[i] for i in order]
    y_sorted_traces = [y_pre[i] for i in order]

    trace_dyn = go.Scatter(
        x=x_sorted[0],
        y=y_sorted_traces[0],
        mode="lines",
        name=f"{slider_label}={y_sorted[0]:g}",
    )

    steps = []
    for k, yk in enumerate(y_sorted):
        steps.append(
            dict(
                label=f"{yk:g}",
                method="restyle",
                args=[
                    {
                        "x": [x_sorted[k]],
                        "y": [y_sorted_traces[k]],
                        "name": [f"{slider_label}={yk:g}"],
                    },
                    [0],
                ],
            )
        )

    xaxis = get_axis(lim=xlim, label=xlabel)
    yaxis = get_axis(lim=ylim, label=ylabel)

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

    return go.Figure(
        data=[trace_dyn],
        layout=layout,
        skip_invalid=False,
    )


def get_ivt_sliders(
    y_values: Sequence[float] | np.ndarray,
    t_s_list: Sequence[Sequence[float] | np.ndarray],
    I_nA_list: Sequence[Sequence[float] | np.ndarray],
    V_mV_list: Sequence[Sequence[float] | np.ndarray],
    y_label: str = "y",
    tlim: LIM = None,
    ilim: LIM = None,
    vlim: LIM = None,
    name: Optional[str] = None,
    scheme: str = "standard",
    dataset: str = "dataset",
) -> tuple[go.Figure, go.Figure, go.Figure]:
    """Create slider figures for ``I(t)``, ``V(t)``, and ``I(V)``.

    Parameters
    ----------
    y_values : Sequence[float] | np.ndarray
        One y-value per trace (e.g., amplitude or power).
    t_s_list : Sequence[Sequence[float] | np.ndarray]
        Time traces in seconds, one per y-value.
    I_nA_list : Sequence[Sequence[float] | np.ndarray]
        Current traces in nA, one per y-value.
    V_mV_list : Sequence[Sequence[float] | np.ndarray]
        Voltage traces in mV, one per y-value.
    y_label : str, default="y"
        Label used for slider current value display.
    tlim : LIM, default=None
        Optional limits for time axis.
    ilim : LIM, default=None
        Optional limits for current axis.
    vlim : LIM, default=None
        Optional limits for voltage axis.
    name : str | None, default=None
        If provided, save three HTML files using this prefix.
    scheme : str, default="standard"
        HTML save scheme passed to :func:`save_figure`.
    dataset : str, default="dataset"
        Output directory passed to :func:`save_figure`.

    Returns
    -------
    tuple[go.Figure, go.Figure, go.Figure]
        ``(fig_I_t, fig_V_t, fig_I_V)``.
    """
    fig_I_t = _get_xy_slider_from_lists(
        x_list=t_s_list,
        y_list=I_nA_list,
        y_values=y_values,
        xlabel="t (s)",
        ylabel="I (nA)",
        slider_label=y_label,
        xlim=tlim,
        ylim=ilim,
    )
    fig_V_t = _get_xy_slider_from_lists(
        x_list=t_s_list,
        y_list=V_mV_list,
        y_values=y_values,
        xlabel="t (s)",
        ylabel="V (mV)",
        slider_label=y_label,
        xlim=tlim,
        ylim=vlim,
    )
    fig_I_V = _get_xy_slider_from_lists(
        x_list=V_mV_list,
        y_list=I_nA_list,
        y_values=y_values,
        xlabel="V (mV)",
        ylabel="I (nA)",
        slider_label=y_label,
        xlim=vlim,
        ylim=ilim,
    )

    if name is not None:
        save_figure(
            fig=fig_I_t,
            title=f"{name}_heatmap",
            scheme=scheme,
            out_dir=dataset,
        )
        save_figure(
            fig=fig_V_t,
            title=f"{name}_slider",
            scheme=scheme,
            out_dir=dataset,
        )
        save_figure(
            fig=fig_I_V,
            title=f"{name}_surface",
            scheme=scheme,
            out_dir=dataset,
        )

    return fig_I_t, fig_V_t, fig_I_V
