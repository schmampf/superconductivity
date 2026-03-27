from __future__ import annotations

from collections import OrderedDict

import pandas as pd


def _import_panel() -> "panel":
    try:
        import panel as pn
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Panel must be installed to build the GUI."
        ) from exc

    try:
        pn.extension("plotly", "tabulator", "mathjax")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Failed to initialize Panel extensions."
        ) from exc

    return pn


def _mapping_frame(mapping: OrderedDict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "key": list(mapping.keys()),
            "value": [str(value) for value in mapping.values()],
        }
    )
