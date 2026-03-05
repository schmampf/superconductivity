"""Cache-key helper functions."""

from .constants import Delta_tol_meV, T_tol_K, V_tol_mV, gamma_tol_meV, tau_tol


def cache_hash(
    V_max_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
    string: str = "HA",
) -> str:
    """Build a cache key for asymmetric two-gap model grids.

    Parameters
    ----------
    V_max_mV, dV_mV, tau, T_K, Delta_1_meV, Delta_2_meV, gamma_1_meV,
    gamma_2_meV
        Model parameters encoded into the key.
    string : str, default="HA"
        Prefix of the key.

    Returns
    -------
    str
        Deterministic cache key string.
    """
    string += "_"
    string += f"V_max={V_max_mV:.{V_tol_mV}f}mV_"
    string += f"dV={dV_mV:.{V_tol_mV}f}mV_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"gamma=({gamma_1_meV:.{gamma_tol_meV}f},"
    string += f"{gamma_2_meV:.{gamma_tol_meV}f})meV"
    return string


def cache_hash_pbar(
    tau: float,
    T_K: float,
    Delta_1_meV: float,
    Delta_2_meV: float,
    gamma_1_meV: float,
    gamma_2_meV: float,
    string: str = "FCS",
) -> str:
    """Build a cache key for pbar-based FCS evaluation.

    Parameters
    ----------
    tau, T_K, Delta_1_meV, Delta_2_meV, gamma_1_meV, gamma_2_meV
        Model parameters encoded into the key.
    string : str, default="FCS"
        Prefix of the key.

    Returns
    -------
    str
        Deterministic cache key string.
    """
    string += "_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta=({Delta_1_meV:.{Delta_tol_meV}f},"
    string += f"{Delta_2_meV:.{Delta_tol_meV}f})meV_"
    string += f"gamma=({gamma_1_meV:.{gamma_tol_meV}f},"
    string += f"{gamma_2_meV:.{gamma_tol_meV}f})meV"
    return string


def cache_hash_nuni(
    tau: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    string: str = "FCS",
) -> str:
    """Build a cache key for single-gap (nuni) FCS evaluation.

    Parameters
    ----------
    tau, T_K, Delta_meV, gamma_meV
        Model parameters encoded into the key.
    string : str, default="FCS"
        Prefix of the key.

    Returns
    -------
    str
        Deterministic cache key string.
    """
    string += "_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta={Delta_meV:.{Delta_tol_meV}f}meV_"
    string += f"gamma={gamma_meV:.{gamma_tol_meV}f}meV"
    return string


def cache_hash_sym(
    V_max_mV: float,
    dV_mV: float,
    tau: float,
    T_K: float,
    Delta_meV: float,
    gamma_meV: float,
    string: str = "ha_sym",
) -> str:
    """Build a cache key for symmetric one-gap model grids.

    Parameters
    ----------
    V_max_mV, dV_mV, tau, T_K, Delta_meV, gamma_meV
        Model parameters encoded into the key.
    string : str, default="ha_sym"
        Prefix of the key.

    Returns
    -------
    str
        Deterministic cache key string.
    """
    string += "_"
    string += f"V_max={V_max_mV:.{V_tol_mV}f}mV_"
    string += f"dV={dV_mV:.{V_tol_mV}f}mV_"
    string += f"tau={tau:.{tau_tol}f}_"
    string += f"T={T_K:.{T_tol_K}f}K_"
    string += f"Delta={Delta_meV:.{Delta_tol_meV}f}meV_"
    string += f"gamma=({gamma_meV:.{gamma_tol_meV}f})meV"
    return string

