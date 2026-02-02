# nature constants

from scipy.constants import Boltzmann as k_B
from scipy.constants import e, h

h_e_Vs: float = h / e
G_0_S: float = 2 * e * e / h
k_B_eV: float = k_B / e

h_e_pVs: float = h_e_Vs * 1e12
G_0_muS: float = G_0_S * 1e6
k_B_meV: float = k_B_eV * 1e3

h_e_nVs: float = h_e_Vs * 1e9
h_e_fVs: float = h_e_Vs * 1e15

# e = 1.602176634e-19 As
# h = 6.62607015e-34 AVs²
# k_B = 1.380649e-23 J/K
# h_e_Vs = 4.135667696923859e-15 Vs
# G_0_S = 7.748091729863649e-05 A/V
# k_B_eV = 8.617333262145179e-05 V/K
# h_e_pVs = 0.004135667696923859 pVs
# G_0_muS = 77.48091729863648 µS
# k_B_meV = 0.08617333262145178 mV/K


# parameter tolerances
V_tol_mV: int = 6  # meV
tau_tol: int = 4
T_tol_K: int = 4  # K
Delta_tol_meV: int = 6  # meV
gamma_tol_meV: int = 9  # meV

# FCS settings
m_max: int = 10
iw: int = 2003
nchi: int = 66
