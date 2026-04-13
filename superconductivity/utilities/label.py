"""Shared label metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LabelMeta:
    """Label-only metadata shared by axes and parameters."""

    label: str
    html_label: str
    latex_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(self, "html_label", str(self.html_label))
        object.__setattr__(self, "latex_label", str(self.latex_label))


def label_V_mV() -> LabelMeta:
    return LabelMeta(
        label="V_mV",
        html_label="<i>V</i> (mV)",
        latex_label=r"$V$ (mV)",
    )


def label_I_nA() -> LabelMeta:
    return LabelMeta(
        label="I_nA",
        html_label="<i>I</i> (nA)",
        latex_label=r"$I$ (nA)",
    )


def label_A_mV() -> LabelMeta:
    return LabelMeta(
        label="A_mV",
        html_label="<i>A</i> (mV)",
        latex_label=r"$A$ (mV)",
    )


def label_nu_GHz() -> LabelMeta:
    return LabelMeta(
        label="nu_GHz",
        html_label="<i>&nu;</i> (GHz)",
        latex_label=r"$\nu$ (GHz)",
    )


def label_T_K() -> LabelMeta:
    return LabelMeta(
        label="T_K",
        html_label="<i>T</i> (K)",
        latex_label=r"$T$ (K)",
    )


def label_dIdV() -> LabelMeta:
    return LabelMeta(
        label="dIdV",
        html_label="d<i>I</i>/d<i>V</i>",
        latex_label=r"$dI/dV$",
    )


def label_dVdI() -> LabelMeta:
    return LabelMeta(
        label="dVdI",
        html_label="d<i>V</i>/d<i>I</i>",
        latex_label=r"$dV/dI$",
    )


def label_V_Delta() -> LabelMeta:
    return LabelMeta(
        label="V",
        html_label="<i>V</i> / <i>&Delta;</i>",
        latex_label=r"$V\,/\,\Delta$",
    )


def label_I_GNDelta() -> LabelMeta:
    return LabelMeta(
        label="I",
        html_label="<i>I</i> / <i>G</i><sub>N</sub> <i>&Delta;</i>",
        latex_label=r"$I\,/\,G_N \Delta$",
    )


def label_A_hnu() -> LabelMeta:
    return LabelMeta(
        label="A_hnu",
        html_label="<i>A</i> / <i>h&nu;</i>",
        latex_label=r"$A / h\nu$",
    )


def label_hnu_Delta() -> LabelMeta:
    return LabelMeta(
        label="hnu_Delta",
        html_label="<i>h&nu;</i> / <i>&Delta;</i>",
        latex_label=r"$h\nu / \Delta$",
    )


def label_T_Tc() -> LabelMeta:
    return LabelMeta(
        label="T_Tc",
        html_label="<i>T</i> / <i>T</i><sub>c</sub>",
        latex_label=r"$T / T_c$",
    )


__all__ = [
    "LabelMeta",
    "label_V_mV",
    "label_I_nA",
    "label_A_mV",
    "label_nu_GHz",
    "label_T_K",
    "label_dIdV",
    "label_dVdI",
    "label_V_Delta",
    "label_I_GNDelta",
    "label_A_hnu",
    "label_hnu_Delta",
    "label_T_Tc",
]
