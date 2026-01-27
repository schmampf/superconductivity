# Let's begin translating the structure of the main Fortran program into Python.
# This will focus on replicating the logic and key operations correctly.
# We'll use NumPy and SciPy for numerical operations.

import numpy as np
from scipy.integrate import quad
from scipy.special import expi

# Constants
pi = np.pi
gamma_euler = 0.5772156649
e_charge = 1.602e-19  # elementary charge (C)
hbar = 1.055e-34  # reduced Planck constant (J·s)
G0 = 2 * e_charge**2 / hbar  # Quantum of conductance (S)


# BCS gap function as a function of reduced temperature
def bcs_gap(temp_reduced):
    if temp_reduced >= 1.0:
        return 0.0
    if temp_reduced == 0.0:
        return 1.0
    if temp_reduced > 0.99:
        return np.exp(gamma_euler) * np.sqrt(8 * (1 - temp_reduced) / (7 * 1.202))

    def integrand(x, gapt, tempf):
        w = gapt * np.sqrt(x**2 + 1) * pi * np.exp(-gamma_euler)
        if w / tempf < 50:
            fermi = 1.0 / (1.0 + np.exp(w / tempf))
        else:
            fermi = 0.0
        return -2.0 * fermi / np.sqrt(x**2 + 1)

    gapt = 1.0
    tempf = temp_reduced * pi / np.exp(gamma_euler)
    for _ in range(100):
        x_max = 50.0 * tempf / (gapt * pi * np.exp(-gamma_euler))
        x_vals = np.linspace(0, x_max, 10000)
        dx = x_vals[1] - x_vals[0]
        integral = np.sum([integrand(x, gapt, tempf) for x in x_vals]) * dx
        gaptf = np.exp(integral)
        if np.abs(gaptf - gapt) < 1e-5:
            return gaptf
        gapt = gaptf
    return gapt


# Dynes broadened BCS density of states
def dynes_dos(E, delta, gamma):
    z = E + 1j * gamma
    return np.real(np.abs(z) / np.sqrt(z**2 - delta**2))


# Fermi function
def fermi(E, T):
    kB = 8.617333262145e-5  # eV/K
    beta = 1 / (kB * T)
    return 1.0 / (1.0 + np.exp(E * beta))


# Current integration placeholder (actual recursive Green’s function logic omitted for now)
def iv_curve(voltage, delta1, delta2, trans, temp, eta1, eta2):
    # Placeholder for actual integration using Green's functions
    # Using a dummy current value to allow code testing
    return np.tanh(voltage / (2 * temp)) * trans


# Main simulation
def simulate_iv_curve(
    temp_K, delta1_meV, delta2_meV, trans, eta1, eta2, vi_meV, vf_meV, vstep_meV
):
    delta1 = delta1_meV
    delta2 = delta2_meV
    eta1 /= delta1
    eta2 /= delta1
    temp = 0.08617 * temp_K / delta1
    ratio = delta2 / delta1
    vi = vi_meV / delta1
    vf = vf_meV / delta1
    vstep = vstep_meV / delta1

    voltages = np.arange(vi, vf + vstep, vstep)
    currents = []

    gap1 = bcs_gap(temp)
    gap2 = bcs_gap(temp / ratio) * ratio

    for v in voltages:
        if np.abs(v) < 0.002:
            curr = 0.0
        else:
            curr = iv_curve(v, gap1, gap2, trans, temp, eta1, eta2)
        # Convert to original units: V (mV) and I (nA)
        currents.append([v * delta1, curr * delta1 * 77.48])

    return np.array(currents)


# Example usage:
# result = simulate_iv_curve(1.0, 1.0, 0.5, 0.8, 0.001, 0.001, -2.0, 2.0, 0.1)

# Returns array of shape (N, 2): columns are [Voltage (mV), Current (nA)]


# First part of the Green's function integrand logic
import numpy as np


def bcs_greens_function(z, delta, eta):
    """
    Returns the 2x2 Nambu BCS Green's function for a given complex energy z.

    Parameters
    ----------
    z : complex
        Energy variable (real + i * imaginary)
    delta : float
        Superconducting gap
    eta : float
        Dynes parameter (broadening)

    Returns
    -------
    g : np.ndarray
        2x2 complex Nambu Green's function
    """
    u = (z - 1j * eta) / delta
    denom = np.sqrt(1.0 - u**2)
    g = np.zeros((2, 2), dtype=np.complex128)
    g[0, 0] = -u / denom
    g[1, 1] = g[0, 0]
    g[0, 1] = 1.0 / denom
    g[1, 0] = g[0, 1]
    return g


def fermi_function(w, T):
    """Symmetric Fermi function."""
    return 0.5 * (1.0 - np.tanh(0.5 * w / T))


def compute_gpm_gmm(g, fermi):
    """Compute greater/lesser components (2x2) of the uncoupled Green's functions."""
    imag_g = np.imag(g)
    gpm = 2j * imag_g * fermi
    gmm = 2j * imag_g * (fermi - 1)
    return gpm, gmm


# Example usage for a grid of energies
def zintegrand(w, v, deltaL, deltaR, etaL, etaR, T):
    """
    Approximate translation of part of zintegrand, computing uncoupled Green's functions.

    Parameters
    ----------
    w : float
        Integration variable
    v : float
        Voltage (dimensionless)
    deltaL, deltaR : float
        Superconducting gaps
    etaL, etaR : float
        Dynes parameters
    T : float
        Temperature (normalized by gap)

    Returns
    -------
    dict
        Dictionary of Green's functions and spectral components at shifted frequencies
    """
    n = int(2.0 / abs(v))
    if n % 2 == 0:
        n += 6
    else:
        n += 7

    w_vals = w + v * np.arange(-n - 1, n + 2)
    g0l = {}
    g0r = {}
    gpml = {}
    gpmr = {}
    gmpl = {}
    gmpr = {}

    for j in range(-n - 1, n + 2):
        wj = w + j * v
        gl = bcs_greens_function(wj, deltaL, etaL)
        gr = bcs_greens_function(wj, deltaR, etaR)
        fermi = fermi_function(wj, T)

        g0l[j] = gl
        g0r[j] = gr

        gpml[j], gmpl[j] = compute_gpm_gmm(gl, fermi)
        gpmr[j], gmpr[j] = compute_gpm_gmm(gr, fermi)

    return {
        "g0l": g0l,
        "g0r": g0r,
        "gpml": gpml,
        "gpmr": gpmr,
        "gmpl": gmpl,
        "gmpr": gmpr,
        "n": n,
    }


# Step 1: Compute uncoupled BCS Green's functions with temperature and Dynes broadening

import numpy as np


def green_bcs(z, delta, eta):
    """
    Computes the 2x2 BCS Green's function matrix at energy z,
    with superconducting gap delta and Dynes parameter eta.
    """
    if delta == 0:
        return np.array([[1j, 0], [0, 1j]], dtype=complex)

    u = (z - 1j * eta) / delta
    denom = np.sqrt(1 - u**2)
    G = np.zeros((2, 2), dtype=complex)
    G[0, 0] = -u / denom
    G[1, 1] = G[0, 0]
    G[0, 1] = 1 / denom
    G[1, 0] = G[0, 1]
    return G


def fermi_dist(E, T):
    return 0.5 * (1 - np.tanh(E / (2 * T)))


def compute_green_functions(E, delta_l, delta_r, eta_l, eta_r, T):
    """
    Computes the uncoupled Green's functions and spectral functions
    for both leads over all harmonics.
    """
    n = len(E)
    gpml = np.zeros((n, 2, 2), dtype=complex)
    gmpl = np.zeros((n, 2, 2), dtype=complex)
    gpmr = np.zeros((n, 2, 2), dtype=complex)
    gmpr = np.zeros((n, 2, 2), dtype=complex)
    G0L = np.zeros((n, 2, 2), dtype=complex)
    G0R = np.zeros((n, 2, 2), dtype=complex)

    for j in range(n):
        z = E[j]
        GL = green_bcs(z, delta_l, eta_l)
        GR = green_bcs(z, delta_r, eta_r)
        f = fermi_dist(z, T)

        G0L[j] = GL
        G0R[j] = GR

        gpml[j] = 2j * np.imag(GL) * f
        gmpl[j] = 2j * np.imag(GL) * (f - 1)
        gpmr[j] = 2j * np.imag(GR) * f
        gmpr[j] = 2j * np.imag(GR) * (f - 1)

    return G0L, G0R, gpml, gmpl, gpmr, gmpr
