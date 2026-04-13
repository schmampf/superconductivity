"""Unified MAR current front-end with automatic model dispatch."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, overload

import numpy as np

from ...utilities.constants import G0_muS
from ...utilities.types import NDArray64
from ..basics import get_Delta_meV
from .models.btk import get_I_btk_nA
from .models.fcs import NMAX_DEFAULT as QMAX_DEFAULT
from .models.fcs import get_I_fcs_nA
from .models.ha_asym import get_I_ha_asym_nA
from .models.ha_sym import get_I_ha_sym_nA


def _charge_resolved_ohmic(
    I_nA: NDArray64,
    qmax: int,
) -> tuple[NDArray64, NDArray64]:
    """Return one-electron-only channels for an ohmic current."""
    Iq_nA = np.zeros((I_nA.size, qmax), dtype=np.float64)
    Iq_nA[:, 0] = I_nA
    return I_nA, Iq_nA


def _charge_resolved_btk(
    I_btk_nA: NDArray64,
    qmax: int,
) -> tuple[NDArray64, NDArray64]:
    """Map BTK total, 1e, and 2e currents to ``(I, I_q)`` output."""
    I_nA = np.asarray(I_btk_nA[:, 0], dtype=np.float64)
    Iq_nA = np.zeros((I_btk_nA.shape[0], qmax), dtype=np.float64)
    Iq_nA[:, 0] = I_btk_nA[:, 1]
    Iq_nA[:, 1] = I_btk_nA[:, 2]
    return I_nA, Iq_nA


def _charge_resolved_fcs(
    I_fcs_nA: NDArray64,
) -> tuple[NDArray64, NDArray64]:
    """Split FCS output into total and charge-resolved channels."""
    return (
        np.asarray(I_fcs_nA[:, 0], dtype=np.float64),
        np.asarray(I_fcs_nA[:, 1:], dtype=np.float64),
    )


def _as_pair(value: float | tuple[float, float]) -> tuple[float, float]:
    """Return one scalar or tuple input as a left/right pair."""
    if isinstance(value, tuple):
        left, right = value
        return float(left), float(right)
    scalar = float(value)
    return scalar, scalar


def _normalize_tau(tau: float | Sequence[float]) -> NDArray64:
    """Return ``tau`` as one non-empty 1D float array."""
    tau_x = np.asarray(tau, dtype=np.float64)
    if tau_x.ndim == 0:
        tau_x = tau_x.reshape(1)
    if tau_x.ndim != 1:
        raise ValueError("tau must be a scalar or 1D sequence.")
    if tau_x.size == 0:
        raise ValueError("tau must contain at least one value.")
    return tau_x


@overload
def get_Imar_nA(
    V_mV: NDArray64,
    tau: float | Sequence[float] = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    charge_resolved: Literal[False] = False,
    tau_resolved: Literal[False] = False,
    delta_tol_meV: float = 1e-6,
    qmax: int = QMAX_DEFAULT,
    caching: bool = True,
) -> NDArray64: ...


@overload
def get_Imar_nA(
    V_mV: NDArray64,
    tau: float | Sequence[float] = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    charge_resolved: Literal[False] = False,
    tau_resolved: Literal[True] = True,
    delta_tol_meV: float = 1e-6,
    qmax: int = QMAX_DEFAULT,
    caching: bool = True,
) -> tuple[NDArray64, NDArray64]: ...


@overload
def get_Imar_nA(
    V_mV: NDArray64,
    tau: float | Sequence[float] = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    charge_resolved: Literal[True] = True,
    tau_resolved: Literal[False] = False,
    delta_tol_meV: float = 1e-6,
    qmax: int = QMAX_DEFAULT,
    caching: bool = True,
) -> tuple[NDArray64, NDArray64]: ...


@overload
def get_Imar_nA(
    V_mV: NDArray64,
    tau: float | Sequence[float] = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    charge_resolved: Literal[True] = True,
    tau_resolved: Literal[True] = True,
    delta_tol_meV: float = 1e-6,
    qmax: int = QMAX_DEFAULT,
    caching: bool = True,
) -> tuple[NDArray64, NDArray64, NDArray64, NDArray64]: ...


def get_Imar_nA(
    V_mV: NDArray64,
    tau: float | Sequence[float] = 0.5,
    T_K: float = 0.0,
    Delta_meV: float | tuple[float, float] = (0.18, 0.18),
    gamma_meV: float | tuple[float, float] = 0.0,
    gamma_meV_min: float = 1e-4,
    charge_resolved: bool = False,
    tau_resolved: bool = False,
    delta_tol_meV: float = 1e-6,
    qmax: int = QMAX_DEFAULT,
    caching: bool = True,
) -> (
    NDArray64
    | tuple[NDArray64, NDArray64]
    | tuple[NDArray64, NDArray64, NDArray64, NDArray64]
):
    """Return the MAR current on ``V_mV`` with automatic model dispatch.

    Parameters
    ----------
    V_mV
        Bias voltages in mV.
    tau
        Junction transmission or one pincode-like sequence of transmissions.
        Sequence inputs are evaluated one transmission at a time and summed.
    T_K
        Temperature in kelvin.
    Delta_meV
        Zero-temperature gap input, either one shared value or one
        ``(left, right)`` tuple in meV.
    gamma_meV
        Dynes broadening, either one shared value or one ``(left, right)``
        tuple in meV.
    gamma_meV_min
        Minimum broadening passed to the wrapped MAR models.
    charge_resolved
        If ``True``, return ``(I_nA, Iq_nA)`` instead of only ``I_nA``. The
        channel array uses ``Iq_nA[:, q - 1]`` for the ``q e`` current.
        The wrapper uses FCS when both leads are superconducting, BTK when
        exactly one lead is superconducting, and an ohmic 1e channel when
        both leads are normal.
    tau_resolved
        If ``True``, also return the transmission-resolved current
        ``Ix_nA[:, x]`` for each input transmission. When both
        ``charge_resolved`` and ``tau_resolved`` are ``True``, also return
        ``Ixq_nA[:, x, q - 1]``.
    delta_tol_meV
        Small tolerance used when classifying thermal gaps as zero.
    qmax
        Highest charge index for the FCS backend. Must be at least ``2`` when
        ``charge_resolved=True``.
    caching
        Forwarded to the cached HA and FCS wrappers.

    Returns
    -------
    NDArray64 or tuple
        ``I_nA`` by default; ``(I_nA, Ix_nA)`` when ``tau_resolved=True``;
        ``(I_nA, Iq_nA)`` when ``charge_resolved=True``; or
        ``(I_nA, Ix_nA, Iq_nA, Ixq_nA)`` when both resolution flags are set.
    """
    V_mV = np.asarray(V_mV, dtype=np.float64)
    tau_x = _normalize_tau(tau)
    if charge_resolved and qmax < 2:
        raise ValueError("qmax must be at least 2 when charge_resolved=True.")

    Delta_1_meV, Delta_2_meV = _as_pair(Delta_meV)
    gamma_1_meV, gamma_2_meV = _as_pair(gamma_meV)

    Delta_1_T_meV = get_Delta_meV(Delta_1_meV, T_K)
    Delta_2_T_meV = get_Delta_meV(Delta_2_meV, T_K)

    symmetric_gap = np.isclose(
        Delta_1_T_meV,
        Delta_2_T_meV,
        atol=delta_tol_meV,
        rtol=0.0,
    )
    symmetric_gamma = np.isclose(
        max(gamma_1_meV, gamma_meV_min),
        max(gamma_2_meV, gamma_meV_min),
        atol=1e-9,
        rtol=0.0,
    )
    symmetric = np.logical_and(
        symmetric_gap,
        symmetric_gamma,
    )

    si_ = Delta_1_T_meV > delta_tol_meV
    _is = Delta_2_T_meV > delta_tol_meV

    ni_ = np.logical_not(si_)
    _in = np.logical_not(_is)

    sis = np.logical_and(si_, _is)
    nin = np.logical_and(ni_, _in)

    def evaluate_single_tau(
        tau_single: float,
    ) -> NDArray64 | tuple[NDArray64, NDArray64]:

        if tau_single == 0.0:
            I_nA = np.zeros_like(V_mV)
            if charge_resolved:
                return _charge_resolved_ohmic(I_nA=I_nA, qmax=qmax)
            return I_nA
        if nin:
            I_nA = V_mV * tau_single * G0_muS
            if charge_resolved:
                return _charge_resolved_ohmic(I_nA=I_nA, qmax=qmax)
            return I_nA
        elif sis:
            if charge_resolved:
                I_fcs_nA = get_I_fcs_nA(
                    V_mV=V_mV,
                    tau=tau_single,
                    T_K=T_K,
                    Delta_meV=(Delta_1_meV, Delta_2_meV),
                    gamma_meV=(gamma_1_meV, gamma_2_meV),
                    gamma_meV_min=gamma_meV_min,
                    nmax=qmax,
                    caching=caching,
                )
                return _charge_resolved_fcs(I_fcs_nA=I_fcs_nA)
            else:
                if symmetric:
                    return get_I_ha_sym_nA(
                        V_mV=V_mV,
                        tau=tau_single,
                        T_K=T_K,
                        Delta_meV=0.5 * (Delta_1_meV + Delta_2_meV),
                        gamma_meV=0.5 * (gamma_1_meV + gamma_2_meV),
                        gamma_meV_min=gamma_meV_min,
                        caching=caching,
                    )
                else:
                    return get_I_ha_asym_nA(
                        V_mV=V_mV,
                        tau=tau_single,
                        T_K=T_K,
                        Delta_meV=(Delta_1_meV, Delta_2_meV),
                        gamma_meV=(gamma_1_meV, gamma_2_meV),
                        gamma_meV_min=gamma_meV_min,
                        caching=caching,
                    )
        else:
            Ibtk_nA = get_I_btk_nA(
                V_mV=V_mV,
                Delta_meV=Delta_1_meV if si_ else Delta_2_meV,
                tau=tau_single,
                T_K=T_K,
                gamma_meV=gamma_1_meV if si_ else gamma_2_meV,
                gamma_meV_min=gamma_meV_min,
            )
            if charge_resolved:
                return _charge_resolved_btk(I_btk_nA=Ibtk_nA, qmax=qmax)
            return Ibtk_nA[:, 0]

    if charge_resolved:
        resolved = [evaluate_single_tau(float(tau_single)) for tau_single in tau_x]
        Ix_nA = np.column_stack([I_nA for I_nA, _ in resolved])
        Ixq_nA = np.stack([Iq_nA for _, Iq_nA in resolved], axis=1)
        I_nA = Ix_nA.sum(axis=1)
        Iq_nA = Ixq_nA.sum(axis=1)
        if tau_resolved:
            return I_nA, Ix_nA, Iq_nA, Ixq_nA
        return I_nA, Iq_nA

    Ix_nA = np.column_stack(
        [evaluate_single_tau(float(tau_single)) for tau_single in tau_x]
    )
    I_nA = Ix_nA.sum(axis=1)
    if tau_resolved:
        return I_nA, Ix_nA
    return I_nA


__all__ = ["get_Imar_nA"]
