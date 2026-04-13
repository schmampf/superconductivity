from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParameterSpec:
    name: str
    label: str
    lower: float
    upper: float
    guess: float
    fixed: bool = False
    value: Optional[float] = None
    error: Optional[float] = None


def make_bcs_parameters() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(
            name="GN_G0",
            label="<i>G</i><sub>N</sub> (<i>G</i><sub>0</sub>)",
            lower=0.0,
            upper=10.0,
            guess=0.189,
        ),
        ParameterSpec(
            name="T_K",
            label="<i>T</i> (K)",
            lower=0.0,
            upper=1.5,
            guess=0.236,
        ),
        ParameterSpec(
            name="Delta_meV",
            label="<i>&Delta;</i> (meV)",
            lower=0.18,
            upper=0.21,
            guess=0.195,
        ),
        ParameterSpec(
            name="gamma_meV",
            label="<i>&gamma;</i> (meV)",
            lower=1e-6,
            upper=25e-3,
            guess=4e-3,
        ),
    )


def make_pat_parameters() -> tuple[ParameterSpec, ...]:
    return make_bcs_parameters() + make_pat_addon_parameters()


def make_pat_addon_parameters() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(
            name="A_mV",
            label="<i>A</i> (mV)",
            lower=0.0,
            upper=1.0,
            guess=0.0,
            fixed=True,
        ),
        ParameterSpec(
            name="nu_GHz",
            label="<i>&nu;</i> (GHz)",
            lower=1.0,
            upper=20.0,
            guess=7.8,
        ),
    )


def make_noise_parameters() -> tuple[ParameterSpec, ...]:
    return (
        ParameterSpec(
            name="sigmaV_mV",
            label="<i>&sigma;</i><sub>V</sub> (mV)",
            lower=0.0,
            upper=1.0,
            guess=0.0,
        ),
    )
