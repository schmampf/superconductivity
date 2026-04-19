"""Shared label registry."""

from __future__ import annotations

LABEL_ROWS = {
    "e": {
        "print_label": "e",
        "html_label": "<i>e</i> (As)",
        "latex_label": r"$e$ (As)",
    },
    "h": {
        "print_label": "h",
        "html_label": "<i>h</i> (J s)",
        "latex_label": r"$h$ (J s)",
    },
    "kB": {
        "print_label": "kB",
        "html_label": "<i>k</i><sub>B</sub> (J/K)",
        "latex_label": r"$k_B$ (J/K)",
    },
    "G0_S": {
        "print_label": "G0 S",
        "html_label": "<i>G<sub>0</sub></i> (S)",
        "latex_label": r"$G_0$ (S)",
    },
    "h_Vs": {
        "print_label": "h (V s)",
        "html_label": "<i>h</i> (Vs)",
        "latex_label": r"$h$ (V s)",
    },
    "kB_eV_K": {
        "print_label": "kB (eV/K)",
        "html_label": "<i>k</i><sub>B</sub> (eV/K)",
        "latex_label": r"$k_B$ (eV/K)",
    },
    "G0_muS": {
        "print_label": "G0 uS",
        "html_label": "<i>G<sub>0</sub></i> (&micro;S)",
        "latex_label": r"$G_0$ (\mu S)",
    },
    "h_pVs": {
        "print_label": "h (pV s)",
        "html_label": "<i>h</i> (pV s)",
        "latex_label": r"$h$ (pV s)",
    },
    "kB_meV_K": {
        "print_label": "kB (meV/K)",
        "html_label": "<i>k</i><sub>B</sub> (meV/K)",
        "latex_label": r"$k_B$ (meV/K)",
    },
    "V_mV": {
        "print_label": "V (mV)",
        "html_label": "<i>V</i> (mV)",
        "latex_label": r"$V$ (mV)",
    },
    "I_nA": {
        "print_label": "I (nA)",
        "html_label": "<i>I</i> (nA)",
        "latex_label": r"$I$ (nA)",
    },
    "A_mV": {
        "print_label": "A (mV)",
        "html_label": "<i>A</i> (mV)",
        "latex_label": r"$A$ (mV)",
    },
    "Aout_mV": {
        "print_label": "Aout (mV)",
        "html_label": "<i>A</i><sub>out</sub> (mV)",
        "latex_label": r"$A_\mathrm{out}$ (mV)",
    },
    "nu_GHz": {
        "print_label": "nu (GHz)",
        "html_label": "<i>&nu;</i> (GHz)",
        "latex_label": r"$\nu$ (GHz)",
    },
    "T_K": {
        "print_label": "T (K)",
        "html_label": "<i>T</i> (K)",
        "latex_label": r"$T$ (K)",
    },
    "dIdV": {
        "print_label": "dIdV",
        "html_label": "d<i>I</i>/d<i>V</i>",
        "latex_label": r"$dI/dV$",
    },
    "dVdI": {
        "print_label": "dVdI",
        "html_label": "d<i>V</i>/d<i>I</i>",
        "latex_label": r"$dV/dI$",
    },
    "V": {
        "print_label": "V / Delta",
        "html_label": "<i>V</i> / <i>&Delta;</i>",
        "latex_label": r"$V\,/\,\Delta$",
    },
    "I": {
        "print_label": "I / (GN Delta)",
        "html_label": "<i>I</i> / <i>G</i><sub>N</sub> <i>&Delta;</i>",
        "latex_label": r"$I\,/\,G_N \Delta$",
    },
    "A_hnu": {
        "print_label": "A / hnu",
        "html_label": "<i>A</i> / <i>h&nu;</i>",
        "latex_label": r"$A / h\nu$",
    },
    "hnu_Delta": {
        "print_label": "hnu / Delta",
        "html_label": "<i>h&nu;</i> / <i>&Delta;</i>",
        "latex_label": r"$h\nu / \Delta$",
    },
    "T_Tc": {
        "print_label": "T / Tc",
        "html_label": "<i>T</i> / <i>T</i><sub>c</sub>",
        "latex_label": r"$T / T_c$",
    },
}


__all__ = ["LABEL_ROWS"]
