"""Parameter normalization helpers for MAR models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# These values are owned by the MAR submodule and intentionally live here so
# the cache/key behavior is defined together with parameter normalization.
TAU_TOL: int = 4
T_TOL_K: int = 4
DELTA_TOL_MEV: int = 6
GAMMA_TOL_MEV: int = 9


def _normalize_asymmetric_lead_inputs(
    Delta_meV: float | tuple[float, float],
    gamma_meV: float | tuple[float, float],
    gamma_meV_min: float,
) -> tuple[float, float, float, float]:
    """Return rounded left/right gap and broadening inputs."""
    if isinstance(Delta_meV, tuple):
        Delta_1_meV, Delta_2_meV = Delta_meV
    else:
        Delta_1_meV = Delta_meV
        Delta_2_meV = Delta_meV

    if isinstance(gamma_meV, tuple):
        gamma_1_meV, gamma_2_meV = gamma_meV
    else:
        gamma_1_meV = gamma_meV
        gamma_2_meV = gamma_meV

    gamma_1_meV = max(gamma_1_meV, gamma_meV_min)
    gamma_2_meV = max(gamma_2_meV, gamma_meV_min)

    return (
        float(np.round(Delta_1_meV, decimals=DELTA_TOL_MEV)),
        float(np.round(Delta_2_meV, decimals=DELTA_TOL_MEV)),
        float(np.round(gamma_1_meV, decimals=GAMMA_TOL_MEV)),
        float(np.round(gamma_2_meV, decimals=GAMMA_TOL_MEV)),
    )


@dataclass(frozen=True, slots=True)
class SymmetricHAParams:
    """Rounded parameter tuple for the symmetric HA model."""

    tau: float
    T_K: float
    Delta_meV: float
    gamma_meV: float

    @classmethod
    def from_raw(
        cls,
        tau: float,
        T_K: float,
        Delta_meV: float,
        gamma_meV: float,
        gamma_meV_min: float,
    ) -> "SymmetricHAParams":
        """Build one rounded cacheable parameter tuple."""
        gamma_meV = gamma_meV_min if gamma_meV < gamma_meV_min else gamma_meV
        return cls(
            tau=float(np.round(tau, decimals=TAU_TOL)),
            T_K=float(np.round(T_K, decimals=T_TOL_K)),
            Delta_meV=float(np.round(Delta_meV, decimals=DELTA_TOL_MEV)),
            gamma_meV=float(np.round(gamma_meV, decimals=GAMMA_TOL_MEV)),
        )

    def cache_key(self) -> str:
        """Return one human-readable deterministic cache key."""
        return (
            f"tau={self.tau:.{TAU_TOL}f}_"
            f"T={self.T_K:.{T_TOL_K}f}K_"
            f"Delta={self.Delta_meV:.{DELTA_TOL_MEV}f}meV_"
            f"gamma={self.gamma_meV:.{GAMMA_TOL_MEV}f}meV"
        )

    def attrs(self) -> dict[str, float | int | str]:
        """Return HDF5 attributes describing this parameter tuple."""
        return {
            "model": "ha_sym",
            "tau": self.tau,
            "T_K": self.T_K,
            "Delta_meV": self.Delta_meV,
            "gamma_meV": self.gamma_meV,
        }


@dataclass(frozen=True, slots=True)
class AsymmetricHAParams:
    """Rounded parameter tuple for the asymmetric HA model."""

    tau: float
    T_K: float
    Delta_1_meV: float
    Delta_2_meV: float
    gamma_1_meV: float
    gamma_2_meV: float

    @classmethod
    def from_raw(
        cls,
        tau: float,
        T_K: float,
        Delta_meV: float | tuple[float, float],
        gamma_meV: float | tuple[float, float],
        gamma_meV_min: float,
    ) -> "AsymmetricHAParams":
        """Build one rounded cacheable asymmetric parameter tuple."""
        (
            Delta_1_meV,
            Delta_2_meV,
            gamma_1_meV,
            gamma_2_meV,
        ) = _normalize_asymmetric_lead_inputs(
            Delta_meV=Delta_meV,
            gamma_meV=gamma_meV,
            gamma_meV_min=gamma_meV_min,
        )

        return cls(
            tau=float(np.round(tau, decimals=TAU_TOL)),
            T_K=float(np.round(T_K, decimals=T_TOL_K)),
            Delta_1_meV=Delta_1_meV,
            Delta_2_meV=Delta_2_meV,
            gamma_1_meV=gamma_1_meV,
            gamma_2_meV=gamma_2_meV,
        )

    def cache_key(self) -> str:
        """Return one human-readable deterministic cache key."""
        return (
            f"tau={self.tau:.{TAU_TOL}f}_"
            f"T={self.T_K:.{T_TOL_K}f}K_"
            f"Delta1={self.Delta_1_meV:.{DELTA_TOL_MEV}f}meV_"
            f"Delta2={self.Delta_2_meV:.{DELTA_TOL_MEV}f}meV_"
            f"gamma1={self.gamma_1_meV:.{GAMMA_TOL_MEV}f}meV_"
            f"gamma2={self.gamma_2_meV:.{GAMMA_TOL_MEV}f}meV"
        )

    def attrs(self) -> dict[str, float | int | str]:
        """Return HDF5 attributes describing this parameter tuple."""
        return {
            "model": "ha_asym",
            "tau": self.tau,
            "T_K": self.T_K,
            "Delta_1_meV": self.Delta_1_meV,
            "Delta_2_meV": self.Delta_2_meV,
            "gamma_1_meV": self.gamma_1_meV,
            "gamma_2_meV": self.gamma_2_meV,
        }


@dataclass(frozen=True, slots=True)
class FCSParams:
    """Rounded parameter tuple for the FCS model."""

    tau: float
    T_K: float
    Delta_1_meV: float
    Delta_2_meV: float
    gamma_1_meV: float
    gamma_2_meV: float
    nmax: int
    iw: int
    nchi: int

    @classmethod
    def from_raw(
        cls,
        tau: float,
        T_K: float,
        Delta_meV: float | tuple[float, float],
        gamma_meV: float | tuple[float, float],
        gamma_meV_min: float,
        nmax: int,
        iw: int,
        nchi: int,
    ) -> "FCSParams":
        """Build one rounded cacheable FCS parameter tuple."""
        (
            Delta_1_meV,
            Delta_2_meV,
            gamma_1_meV,
            gamma_2_meV,
        ) = _normalize_asymmetric_lead_inputs(
            Delta_meV=Delta_meV,
            gamma_meV=gamma_meV,
            gamma_meV_min=gamma_meV_min,
        )

        return cls(
            tau=float(np.round(tau, decimals=TAU_TOL)),
            T_K=float(np.round(T_K, decimals=T_TOL_K)),
            Delta_1_meV=Delta_1_meV,
            Delta_2_meV=Delta_2_meV,
            gamma_1_meV=gamma_1_meV,
            gamma_2_meV=gamma_2_meV,
            nmax=int(nmax),
            iw=int(iw),
            nchi=int(nchi),
        )

    def cache_key(self) -> str:
        """Return one human-readable deterministic cache key."""
        return (
            f"tau={self.tau:.{TAU_TOL}f}_"
            f"T={self.T_K:.{T_TOL_K}f}K_"
            f"Delta1={self.Delta_1_meV:.{DELTA_TOL_MEV}f}meV_"
            f"Delta2={self.Delta_2_meV:.{DELTA_TOL_MEV}f}meV_"
            f"gamma1={self.gamma_1_meV:.{GAMMA_TOL_MEV}f}meV_"
            f"gamma2={self.gamma_2_meV:.{GAMMA_TOL_MEV}f}meV_"
            f"nmax={self.nmax}_iw={self.iw}_nchi={self.nchi}"
        )

    def attrs(self) -> dict[str, float | int | str]:
        """Return HDF5 attributes describing this parameter tuple."""
        return {
            "model": "fcs",
            "tau": self.tau,
            "T_K": self.T_K,
            "Delta_1_meV": self.Delta_1_meV,
            "Delta_2_meV": self.Delta_2_meV,
            "gamma_1_meV": self.gamma_1_meV,
            "gamma_2_meV": self.gamma_2_meV,
            "nmax": self.nmax,
            "iw": self.iw,
            "nchi": self.nchi,
        }


__all__ = [
    "AsymmetricHAParams",
    "DELTA_TOL_MEV",
    "FCSParams",
    "GAMMA_TOL_MEV",
    "SymmetricHAParams",
    "TAU_TOL",
    "T_TOL_K",
]
