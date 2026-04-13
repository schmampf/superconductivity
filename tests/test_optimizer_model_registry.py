import importlib.util

import numpy as np
import pytest

from superconductivity.models.basics.noise import (
    apply_voltage_noise,
    make_bias_support_grid,
)
from superconductivity.models.bcs import get_I_pat_nA
from superconductivity.optimizers.bcs import (
    MODEL_OPTIONS,
    BCSModelConfig,
    get_model_key,
    get_model_spec,
)
from superconductivity.optimizers.bcs.registry import PAT_N_MAX

_SCIPY_AVAILABLE = importlib.util.find_spec("scipy") is not None
_JAX_AVAILABLE = importlib.util.find_spec("jax") is not None

pytestmark = pytest.mark.skipif(not _JAX_AVAILABLE, reason="jax is unavailable")

BASE_CONFIGS = [
    BCSModelConfig("int", "np"),
    BCSModelConfig("int", "jax"),
    BCSModelConfig("conv", "np"),
    BCSModelConfig("conv", "jax"),
]


def test_registry_resolves_full_base_matrix() -> None:
    V_mV = np.linspace(-1.0, 1.0, 21)
    for config in BASE_CONFIGS:
        spec = get_model_spec(config)
        guess = [parameter.guess for parameter in spec.parameters]
        curve = spec.function(V_mV, *guess)

        assert spec.key == get_model_key(config)
        assert callable(spec.function)
        assert len(spec.parameters) == 4
        assert curve.shape == V_mV.shape
        assert np.all(np.isfinite(curve))


def test_registry_composes_parameter_sets_deterministically() -> None:
    plain = get_model_spec(BCSModelConfig("int", "jax"))
    pat = get_model_spec(BCSModelConfig("int", "jax", pat_enabled=True))
    noise = get_model_spec(BCSModelConfig("int", "jax", noise_enabled=True))
    pat_noise = get_model_spec(
        BCSModelConfig(
            "int",
            "jax",
            pat_enabled=True,
            noise_enabled=True,
        )
    )

    assert [parameter.name for parameter in plain.parameters] == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
    ]
    assert [parameter.name for parameter in pat.parameters] == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "A_mV",
        "nu_GHz",
    ]
    assert [parameter.name for parameter in noise.parameters] == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "sigmaV_mV",
    ]
    assert [parameter.name for parameter in pat_noise.parameters] == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "A_mV",
        "nu_GHz",
        "sigmaV_mV",
    ]
    assert noise.info["noise_oversample"] == "64"


@pytest.mark.skipif(not _SCIPY_AVAILABLE, reason="scipy is unavailable")
def test_registry_stacks_noise_after_pat() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41)
    config = BCSModelConfig(
        "conv",
        "np",
        pat_enabled=True,
        noise_enabled=True,
    )
    spec = get_model_spec(config)
    guess = [parameter.guess for parameter in spec.parameters]
    guess[4] = 0.15
    guess[5] = 8.5
    guess[6] = 0.04
    composed = spec.function(V_mV, *guess)
    V_support = make_bias_support_grid(V_mV, guess[6])
    pat_support = get_I_pat_nA(
        V_support,
        get_model_spec(BCSModelConfig("conv", "np")).function(
            V_support,
            guess[0],
            guess[1],
            guess[2],
            guess[3],
        ),
        guess[4],
        nu_GHz=guess[5],
        n_max=PAT_N_MAX,
    )
    pat_noise = apply_voltage_noise(
        V_support,
        pat_support,
        guess[6],
        config.noise_oversample,
        V_out_mV=V_mV,
    )

    assert np.allclose(composed, pat_noise)


def test_legacy_aliases_still_resolve() -> None:
    assert "bcs_conv_noise" in MODEL_OPTIONS.values()
    assert "pat_int_jax" in MODEL_OPTIONS.values()
    assert get_model_spec("bcs_int").key == "bcs_int_np"
    assert get_model_spec("bcs_conv_noise").key == "bcs_conv_jax_noise"
    assert get_model_spec("pat_int_jax").key == "bcs_int_jax_pat"
