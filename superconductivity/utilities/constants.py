"""Physical constants.

The canonical constant objects are fixed ``ParamSpec`` instances. Numeric
aliases are kept for legacy code that still expects plain floats.
"""

from __future__ import annotations

from scipy.constants import Boltzmann as _kB
from scipy.constants import e as _e
from scipy.constants import h as _h

from .meta.param import ParamSpec

e = ParamSpec(
    label="e",
    html_label="<i>e</i> (As)",
    latex_label=r"$e$ (As)",
    value=float(_e),
    fixed=True,
)

h = ParamSpec(
    label="h",
    html_label="<i>h</i> (J s)",
    latex_label=r"$h$ (J s)",
    value=float(_h),
    fixed=True,
)

kB = ParamSpec(
    label="kB",
    html_label="<i>k</i><sub>B</sub> (J/K)",
    latex_label=r"$k_B$ (J/K)",
    value=float(_kB),
    fixed=True,
)

G0_S = ParamSpec(
    label="G0_S",
    html_label="<i>G<sub>0</sub></i> (S)",
    latex_label=r"$G_0$ (S)",
    value=2.0 * e.value * e.value / h.value,
    fixed=True,
)

h_Vs = ParamSpec(
    label="h_Vs",
    html_label="<i>h</i> (Vs)",
    latex_label=r"$h$ (V s)",
    value=h.value / e.value,
    fixed=True,
)

kB_eV_K = ParamSpec(
    label="kB_eV_K",
    html_label="<i>k</i><sub>B</sub> (eV/K)",
    latex_label=r"$k_B$ (eV/K)",
    value=kB.value / e.value,
    fixed=True,
)

G0_muS = ParamSpec(
    label="G0_muS",
    html_label="<i>G<sub>0</sub></i> (&micro;S)",
    latex_label=r"$G_0$ (\mu S)",
    value=float(G0_S.value) * 1e6,
    fixed=True,
)

h_pVs = ParamSpec(
    label="h_pVs",
    html_label="<i>h</i> (pV s)",
    latex_label=r"$h$ (pV s)",
    value=h_Vs.value * 1e12,
    fixed=True,
)

kB_meV_K = ParamSpec(
    label="kB_meV_K",
    html_label="<i>k</i><sub>B</sub> (meV/K)",
    latex_label=r"$k_B$ (meV/K)",
    value=kB_eV_K.value * 1e3,
    fixed=True,
)

# e = 1.602176634e-19 As
# h = 6.62607015e-34 J s
# kB = 1.380649e-23 J/K
# G0_S = 7.748091729863649e-05 S
# h_Vs = 4.135667696923859e-15 V s
# kB_eV_K = 8.617333262145179e-05 eV/K
# G0_muS = 77.48091729863648 uS
# h_pVs = 0.004135667696923859 pV s
# kB_meV_K = 0.08617333262145178 meV/K

__all__ = [
    "e",
    "h",
    "kB",
    "G0_S",
    "h_Vs",
    "kB_eV_K",
    "G0_muS",
    "h_pVs",
    "kB_meV_K",
]
