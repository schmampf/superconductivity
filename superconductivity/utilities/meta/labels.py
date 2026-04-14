"""Shared label registry."""

from __future__ import annotations

LABEL_ROWS = {
    "V_mV": {
        "html_label": "<i>V</i> (mV)",
        "latex_label": r"$V$ (mV)",
    },
    "I_nA": {
        "html_label": "<i>I</i> (nA)",
        "latex_label": r"$I$ (nA)",
    },
    "A_mV": {
        "html_label": "<i>A</i> (mV)",
        "latex_label": r"$A$ (mV)",
    },
    "nu_GHz": {
        "html_label": "<i>&nu;</i> (GHz)",
        "latex_label": r"$\nu$ (GHz)",
    },
    "T_K": {
        "html_label": "<i>T</i> (K)",
        "latex_label": r"$T$ (K)",
    },
    "dIdV": {
        "html_label": "d<i>I</i>/d<i>V</i>",
        "latex_label": r"$dI/dV$",
    },
    "dVdI": {
        "html_label": "d<i>V</i>/d<i>I</i>",
        "latex_label": r"$dV/dI$",
    },
    "V": {
        "html_label": "<i>V</i> / <i>&Delta;</i>",
        "latex_label": r"$V\,/\,\Delta$",
    },
    "I": {
        "html_label": "<i>I</i> / <i>G</i><sub>N</sub> <i>&Delta;</i>",
        "latex_label": r"$I\,/\,G_N \Delta$",
    },
    "A_hnu": {
        "html_label": "<i>A</i> / <i>h&nu;</i>",
        "latex_label": r"$A / h\nu$",
    },
    "hnu_Delta": {
        "html_label": "<i>h&nu;</i> / <i>&Delta;</i>",
        "latex_label": r"$h\nu / \Delta$",
    },
    "T_Tc": {
        "html_label": "<i>T</i> / <i>T</i><sub>c</sub>",
        "latex_label": r"$T / T_c$",
    },
}


__all__ = ["LABEL_ROWS"]
