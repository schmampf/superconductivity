from dataclasses import replace

import numpy as np

from superconductivity.optimizers import fit_model
from superconductivity.optimizers.models import get_model_spec


def test_fit_model_returns_compact_solution() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41)
    spec = get_model_spec("bcs_sis_int")
    guess = [parameter.guess for parameter in spec.parameters]
    I_nA = spec.function(V_mV, *guess)
    parameters = [
        replace(parameter, fixed=(index > 0))
        for index, parameter in enumerate(spec.parameters)
    ]

    solution = fit_model(
        V_mV,
        I_nA,
        model="bcs_sis_int",
        parameters=parameters,
        maxfev=20,
    )

    assert sorted(solution) == [
        "I_exp_nA",
        "I_fit_nA",
        "I_ini_nA",
        "V_mV",
        "maxfev",
        "params",
        "weights",
    ]
    assert solution["weights"] is None
    assert solution["maxfev"] == 20
    assert solution["I_fit_nA"].shape == V_mV.shape
    assert np.allclose(solution["I_ini_nA"], I_nA)
    assert np.isclose(solution["params"][0].value, guess[0])
    assert solution["params"][1].value == guess[1]
