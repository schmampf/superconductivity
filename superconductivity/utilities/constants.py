"""Physical constants.

The canonical constant objects are fixed ``ParamSpec`` instances. Numeric
aliases are kept for legacy code that still expects plain floats.
"""

from __future__ import annotations

from scipy.constants import Boltzmann as _kB
from scipy.constants import e as _e
from scipy.constants import h as _h

from .meta.param import param

e = param("e", float(_e), fixed=True)
h = param("h", float(_h), fixed=True)
kB = param("kB", float(_kB), fixed=True)

G0_S = param("G0_S", 2.0 * e * e / h, fixed=True)
h_Vs = param("h_Vs", h / e, fixed=True)
kB_eV_K = param("kB_eV_K", kB / e, fixed=True)
G0_muS = param("G0_muS", G0_S * 1e6, fixed=True)
h_pVs = param("h_pVs", h_Vs * 1e12, fixed=True)
kB_meV_K = param("kB_meV_K", kB_eV_K * 1e3, fixed=True)

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
