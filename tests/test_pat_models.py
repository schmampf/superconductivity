from __future__ import annotations

from dataclasses import replace

import numpy as np

from superconductivity.optimizers.pat import DEFAULT_PARAMETERS, fit_pat
from superconductivity.optimizers.pat.models import get_model


def _default_parameter_values() -> dict[str, float]:
    return {
        "GN_G0": 0.189,
        "T_K": 0.236,
        "Delta_meV": 0.195,
        "gamma_meV": 4e-3,
        "A_mV": 0.0,
        "nu_GHz": 7.8,
    }


def test_convolution_model_matches_integral_model() -> None:
    V = np.linspace(-1.0, 1.0, 401, dtype=np.float64)
    E = np.linspace(-2.0, 2.0, 2001, dtype=np.float64)
    params = _default_parameter_values()

    integral, _ = get_model(model="dynes", E_mV=E)
    convolution, _ = get_model(model="conv", E_mV=E)

    I_integral = integral(
        V,
        params["GN_G0"],
        params["T_K"],
        params["Delta_meV"],
        params["gamma_meV"],
    )
    I_convolution = convolution(
        V,
        params["GN_G0"],
        params["T_K"],
        params["Delta_meV"],
        params["gamma_meV"],
    )

    assert np.allclose(I_convolution, I_integral, rtol=6e-3, atol=5e-2)


def test_convolution_pat_matches_integral_pat_when_pat_is_off() -> None:
    V = np.linspace(-1.0, 1.0, 401, dtype=np.float64)
    E = np.linspace(-2.0, 2.0, 2001, dtype=np.float64)
    params = _default_parameter_values()

    integral_pat, _ = get_model(model="pat", E_mV=E)
    convolution_pat, _ = get_model(model="conv_pat", E_mV=E)

    I_integral = integral_pat(
        V,
        params["GN_G0"],
        params["T_K"],
        params["Delta_meV"],
        params["gamma_meV"],
        params["A_mV"],
        params["nu_GHz"],
    )
    I_convolution = convolution_pat(
        V,
        params["GN_G0"],
        params["T_K"],
        params["Delta_meV"],
        params["gamma_meV"],
        params["A_mV"],
        params["nu_GHz"],
    )

    assert np.allclose(I_convolution, I_integral, rtol=6e-3, atol=5e-2)


def test_fit_pat_accepts_convolution_pat_model() -> None:
    V = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
    params = _default_parameter_values()
    function, _ = get_model(model="conv_pat")
    I = function(
        V,
        params["GN_G0"],
        params["T_K"],
        params["Delta_meV"],
        params["gamma_meV"],
        params["A_mV"],
        params["nu_GHz"],
    )
    parameter_specs = []
    for parameter in DEFAULT_PARAMETERS:
        parameter_specs.append(
            replace(
                parameter,
                guess=params[parameter.name],
                fixed=True,
            )
        )

    solution = fit_pat(
        V,
        I,
        parameters=parameter_specs,
        model="conv_pat",
    )

    assert np.allclose(solution["I_fit_nA"], I)
    assert solution["params"][0].value == params["GN_G0"]
