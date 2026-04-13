# nature constants

from scipy.constants import Boltzmann as kB
from scipy.constants import e, h

G0_S: float = 2 * e * e / h
h_Vs: float = h / e
kB_eV_K: float = kB / e

G0_muS: float = G0_S * 1e6
h_pVs: float = h_Vs * 1e12
kB_meV_K: float = kB_eV_K * 1e3

# e = 1.602176634e-19 As
# h = 6.62607015e-34 AVs²
# kB = 1.380649e-23 J/K
# G0_S = 7.748091729863649e-05 A/V
# h_Vs = 4.135667696923859e-15 Vs
# kB_eV_K = 8.617333262145179e-05 V/K
# G0_muS = 77.48091729863648 µS
# h_pVs 0.004135667696923859 pVs
# kB_meV_K = 0.08617333262145178 mV/K

__all__ = [
    "e",
    "h",
    "kB",
    "G0_S",
    "kB_eV_K",
    "h_e_Vs",
    "G0_muS",
    "h_pVs",
    "kB_meV_K",
]
