"""Visualization browser page for TransportLab."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

from ..utilities.transport import TransportDatasetSpec
from ..visuals.plotly import get_heatmap, get_plain, get_slider, get_surface

_MODES = ("trace", "heatmap", "surface")


def visualization_app(pn: Any, session: Any):
    """Build the TransportLab visualization page."""
    datasets = pn.widgets.Tabulator(
        _dataset_frame(session),
        selectable=True,
        show_index=False,
        sortable=False,
        editors={"key": None, "type": None},
        layout="fit_columns",
        sizing_mode="stretch_width",
        height=220,
    )
    modes = pn.widgets.Tabulator(
        _mode_frame(),
        selectable=1,
        show_index=False,
        sortable=False,
        editors={"mode": None},
        layout="fit_columns",
        sizing_mode="stretch_width",
        height=130,
    )
    x_table = _quantity_table(pn, name="x")
    y_table = _quantity_table(pn, name="y")
    z_table = _quantity_table(pn, name="z")
    alert = pn.pane.Alert("", alert_type="warning", visible=False)
    output = pn.GridBox(ncols=2, sizing_mode="stretch_width")

    def _refresh_quantities() -> None:
        selected = _selected_datasets(datasets, session)
        dataset = selected[0][1] if selected else None
        z_table.value = _z_frame(dataset)
        x_table.value = _axis_frame(dataset, order=1)
        y_table.value = _axis_frame(dataset, order=0)
        for table in (x_table, y_table, z_table):
            table.selection = []

    def _render(_event: Any = None) -> None:
        selected = _selected_datasets(datasets, session)
        mode = _selected_mode(modes)
        z_key = _selected_quantity(z_table)
        x_key = _selected_quantity(x_table)
        y_key = _selected_quantity(y_table)
        output.objects = []
        alert.visible = False
        if not selected or mode is None:
            return
        figures = []
        errors = []
        for name, dataset in selected:
            try:
                figure = _figure_for_dataset(
                    dataset,
                    mode,
                    x_key,
                    y_key,
                    z_key,
                )
                figure.update_layout(title=name)
                figures.append((name, figure))
            except Exception as exc:
                errors.append(f"{name}: {exc}")
        if errors:
            alert.object = "Cannot render selection. " + " ".join(errors)
            alert.visible = True
        output.objects = [
            pn.pane.Plotly(figure, sizing_mode="stretch_width", height=460)
            for _, figure in figures
        ]

    def _refresh(_event: Any = None) -> None:
        datasets.value = _dataset_frame(session)
        datasets.selection = []
        _refresh_quantities()
        _render()

    def _datasets_changed(event: Any) -> None:
        _refresh_quantities()
        _render(event)

    datasets.param.watch(_datasets_changed, "selection")
    for table in (modes, x_table, y_table, z_table):
        table.param.watch(_render, "selection")
    watch_cache = getattr(session, "watch_cache", None)
    if callable(watch_cache):
        watch_cache(_refresh)

    return pn.Column(
        pn.Row(datasets, modes, sizing_mode="stretch_width"),
        pn.Row(z_table, x_table, y_table, sizing_mode="stretch_width"),
        alert,
        output,
        sizing_mode="stretch_width",
    )


def _quantity_table(pn: Any, *, name: str):
    return pn.widgets.Tabulator(
        _quantity_frame(None),
        name=name,
        selectable=1,
        show_index=False,
        sortable=False,
        editors={"key": None, "kind": None, "shape": None},
        layout="fit_columns",
        sizing_mode="stretch_width",
        height=220,
    )


def _dataset_frame(session: Any):
    import pandas as pd

    rows = [
        {"key": key, "type": type(value).__name__}
        for key, value in _transport_datasets(session)
    ]
    return pd.DataFrame(rows, columns=("key", "type"))


def _mode_frame():
    import pandas as pd

    return pd.DataFrame([{"mode": mode} for mode in _MODES], columns=("mode",))


def _quantity_frame(dataset: TransportDatasetSpec | None):
    import pandas as pd

    if dataset is None:
        return pd.DataFrame([], columns=("key", "kind", "shape"))
    rows = []
    for key in dataset.keys():
        try:
            entry = dataset[key]
        except Exception:
            continue
        rows.append(
            {
                "key": key,
                "kind": type(entry).__name__,
                "shape": _shape_text(entry),
            }
        )
    return pd.DataFrame(rows, columns=("key", "kind", "shape"))


def _z_frame(dataset: TransportDatasetSpec | None):
    import pandas as pd

    if dataset is None:
        return pd.DataFrame([], columns=("key", "kind", "shape"))
    rows = []
    for key in dataset.keys():
        try:
            entry = dataset[key]
        except Exception:
            continue
        values = getattr(entry, "values", None)
        if values is None or np.asarray(values).ndim != 2:
            continue
        rows.append(
            {
                "key": key,
                "kind": type(entry).__name__,
                "shape": _shape_text(entry),
            }
        )
    return pd.DataFrame(rows, columns=("key", "kind", "shape"))


def _axis_frame(dataset: TransportDatasetSpec | None, *, order: int):
    import pandas as pd

    if dataset is None:
        return pd.DataFrame([], columns=("key", "kind", "shape"))
    rows = []
    for key in dataset.keys():
        try:
            entry = dataset[key]
        except Exception:
            continue
        values = getattr(entry, "values", None)
        entry_order = getattr(entry, "order", None)
        if values is None or np.asarray(values).ndim != 1 or entry_order != order:
            continue
        rows.append(
            {
                "key": key,
                "kind": type(entry).__name__,
                "shape": _shape_text(entry),
            }
        )
    return pd.DataFrame(rows, columns=("key", "kind", "shape"))


def _transport_datasets(session: Any) -> tuple[tuple[str, TransportDatasetSpec], ...]:
    cache = getattr(session, "cache", None)
    if cache is None:
        return ()
    return tuple(
        (key, value)
        for key, value in cache.items.items()
        if isinstance(value, TransportDatasetSpec)
    )


def _selected_datasets(
    table: Any,
    session: Any,
) -> tuple[tuple[str, TransportDatasetSpec], ...]:
    frame = table.value.reset_index(drop=True)
    datasets = dict(_transport_datasets(session))
    selected = []
    for index in list(table.selection or []):
        row = int(index)
        if 0 <= row < len(frame):
            key = str(frame.at[row, "key"])
            if key in datasets:
                selected.append((key, datasets[key]))
    return tuple(selected)


def _selected_mode(table: Any) -> str | None:
    frame = table.value.reset_index(drop=True)
    selection = list(table.selection or [])
    if not selection:
        return None
    row = int(selection[0])
    if row < 0 or row >= len(frame):
        return None
    return str(frame.at[row, "mode"])


def _selected_quantity(table: Any) -> str | None:
    frame = table.value.reset_index(drop=True)
    selection = list(table.selection or [])
    if not selection:
        return None
    row = int(selection[0])
    if row < 0 or row >= len(frame):
        return None
    return str(frame.at[row, "key"])


def _figure_for_dataset(
    dataset: TransportDatasetSpec,
    mode: str,
    x_key: str | None,
    y_key: str | None,
    z_key: str | None,
) -> go.Figure:
    if mode == "trace":
        x, y, z = _xyz_values(dataset, x_key=x_key, y_key=y_key, z_key=z_key)
        x_label, y_label, z_label = _plot_labels(
            dataset,
            x_key=x_key,
            y_key=y_key,
            z_key=z_key,
            z=z,
        )
        if z.ndim == 1:
            return get_plain(
                x=x,
                y=np.asarray([0.0]),
                z=z.reshape(1, -1),
                xlabel=x_label,
                zlabel=z_label,
            )
        return get_slider(
            x=x,
            y=y,
            z=z,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
        )
    if mode == "heatmap":
        x, y, z = _xyz_values(dataset, x_key=x_key, y_key=y_key, z_key=z_key)
        x_label, y_label, z_label = _plot_labels(
            dataset,
            x_key=x_key,
            y_key=y_key,
            z_key=z_key,
            z=z,
        )
        return get_heatmap(
            x=x,
            y=y,
            z=z,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
        )
    if mode == "surface":
        x, y, z = _xyz_values(dataset, x_key=x_key, y_key=y_key, z_key=z_key)
        x_label, y_label, z_label = _plot_labels(
            dataset,
            x_key=x_key,
            y_key=y_key,
            z_key=z_key,
            z=z,
        )
        return get_surface(
            x=x,
            y=y,
            z=z,
            xlabel=x_label,
            ylabel=y_label,
            zlabel=z_label,
        )
    raise ValueError(f"Unknown plot mode: {mode}.")


def _xyz_values(
    dataset: TransportDatasetSpec,
    *,
    x_key: str | None,
    y_key: str | None,
    z_key: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if z_key is None:
        raise ValueError("Select z quantity.")
    z = _entry_values(dataset, z_key)
    if z.ndim == 1:
        x = _optional_axis(dataset, x_key, length=z.size)
        y = np.asarray([0.0], dtype=np.float64)
        return x, y, z
    if z.ndim != 2:
        raise ValueError("Only 1D and 2D z quantities are supported.")
    x = _optional_axis(dataset, x_key, length=z.shape[1])
    y = _optional_axis(dataset, y_key, length=z.shape[0])
    return x, y, z


def _optional_axis(
    dataset: TransportDatasetSpec,
    key: str | None,
    *,
    length: int,
) -> np.ndarray:
    fallback = np.arange(length, dtype=np.float64)
    if key is None:
        return fallback
    try:
        values = _entry_values(dataset, key).reshape(-1)
    except Exception:
        return fallback
    if values.size != length:
        return fallback
    return values


def _plot_labels(
    dataset: TransportDatasetSpec,
    *,
    x_key: str | None,
    y_key: str | None,
    z_key: str | None,
    z: np.ndarray,
) -> tuple[str, str, str]:
    if z.ndim == 1:
        x_label = _axis_label(
            dataset,
            x_key,
            length=z.size,
            fallback="x index",
        )
        y_label = "y index"
    else:
        x_label = _axis_label(
            dataset,
            x_key,
            length=z.shape[1],
            fallback="x index",
        )
        y_label = _axis_label(
            dataset,
            y_key,
            length=z.shape[0],
            fallback="y index",
        )
    return x_label, y_label, _entry_label(dataset, z_key, fallback="z")


def _axis_label(
    dataset: TransportDatasetSpec,
    key: str | None,
    *,
    length: int,
    fallback: str,
) -> str:
    if key is None:
        return fallback
    try:
        values = _entry_values(dataset, key).reshape(-1)
    except Exception:
        return fallback
    if values.size != length:
        return fallback
    return _entry_label(dataset, key, fallback=fallback)


def _entry_label(
    dataset: TransportDatasetSpec,
    key: str | None,
    *,
    fallback: str,
) -> str:
    if key is None:
        return fallback
    try:
        entry = dataset[key]
    except Exception:
        return fallback
    label = getattr(entry, "html_label", None)
    return str(label) if label else str(key)


def _entry_values(dataset: TransportDatasetSpec, key: str) -> np.ndarray:
    entry = dataset[key]
    values = getattr(entry, "values", None)
    if values is None:
        raise ValueError(f"{key} has no values.")
    array = np.asarray(values, dtype=np.float64)
    if array.ndim == 0:
        raise ValueError(f"{key} is scalar.")
    if not np.any(np.isfinite(array)):
        raise ValueError(f"{key} has no finite values.")
    return array


def _shape_text(entry: object) -> str:
    values = getattr(entry, "values", None)
    if values is None:
        return ""
    shape = tuple(np.asarray(values).shape)
    if shape == ():
        return "scalar"
    return "x".join(str(size) for size in shape)
