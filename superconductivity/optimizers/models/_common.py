from __future__ import annotations

from collections import OrderedDict

import numpy as np

from .parameters import ParameterSpec

DEFAULT_E_MV = np.linspace(-4.0, 4.0, 4001, dtype=np.float64)
PAT_N_MAX = 50


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
    return make_bcs_parameters() + (
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


def make_bcs_noise_parameters() -> tuple[ParameterSpec, ...]:
    return make_bcs_parameters() + (
        ParameterSpec(
            name="sigma_V_mV",
            label="<i>&sigma;</i><sub>V</sub> (mV)",
            lower=0.0,
            upper=1.0,
            guess=0.0,
        ),
    )


def energy_grid_summary(E_mV: np.ndarray) -> str:
    return (
        f"{float(E_mV[0]):.1f}..{float(E_mV[-1]):.1f} meV, " f"{int(E_mV.size)} points"
    )


def make_model_info(
    *,
    backend: str,
    junction: str,
    kernel: str,
    E_mV: np.ndarray,
    n_max: int | None = None,
) -> OrderedDict[str, str]:
    info = OrderedDict(
        [
            ("backend", backend),
            ("junction", junction),
            ("kernel", kernel),
            ("energy_grid", energy_grid_summary(E_mV)),
        ]
    )
    if n_max is not None:
        info["n_max"] = str(int(n_max))
    return info


BCS_SIS_INT_HTML = r"""
\[
I(V)=G_\mathrm{N}\int_{-\infty}^{\infty}
N(E-\tfrac{eV}{2})N(E+\tfrac{eV}{2})
\left[f(E-\tfrac{eV}{2})-f(E+\tfrac{eV}{2})\right]\mathrm{d}E
\]
"""

BCS_SIS_CONV_HTML = r"""
\[
I(V)=G_\mathrm{N}\left[
N_2(1-f_2)\otimes N_1 f_1 -
N_1(1-f_1)\otimes N_2 f_2
\right](eV)
\]
"""

BCS_SIN_INT_HTML = r"""
\[
I(V)=G_\mathrm{N}\int_{-\infty}^{\infty}
N_\mathrm{S}(E+\tfrac{eV}{2})
\left[f(E-\tfrac{eV}{2})-f(E+\tfrac{eV}{2})\right]\mathrm{d}E
\]
"""

BCS_SIN_CONV_HTML = r"""
\[
I(V)=G_\mathrm{N}\left[
(1-f)\otimes N_\mathrm{S}f -
f \otimes N_\mathrm{S}(1-f)
\right](eV)
\]
"""

PAT_SUFFIX_HTML = r"""
\[
I_\mathrm{PAT}(V)=\sum_{n=-N}^{N}
J_n^2\!\left(\frac{eA}{h\nu}\right)
I_\mathrm{BCS}\!\left(V-n\frac{h\nu}{e}\right)
\]
"""

NOISE_SUFFIX_HTML = r"""
\[
I_\mathrm{noise}(V)=
\int \mathrm{d}V'\,
\frac{1}{\sqrt{2\pi}\sigma_V}
\exp\!\left[-\frac{(V'-V)^2}{2\sigma_V^2}\right]
I_0(V')
\]
"""
