import numpy as np

from superconductivity.optimizers.bcs import (
    BCSModelConfig,
    MODEL_OPTIONS,
    get_model_key,
    get_model_spec,
)
from superconductivity.optimizers.bcs.gap_distribution import (
    apply_gap_distribution,
)
from superconductivity.optimizers.bcs.noise import (
    apply_voltage_noise,
    make_bias_support_grid,
)
from superconductivity.optimizers.bcs.pat import get_I_pat_nA
from superconductivity.optimizers.bcs.registry import PAT_N_MAX

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
    gap = get_model_spec(
        BCSModelConfig("int", "jax", gap_distribution_enabled=True)
    )
    noise = get_model_spec(BCSModelConfig("int", "jax", noise_enabled=True))
    pat_gap_noise = get_model_spec(
        BCSModelConfig(
            "int",
            "jax",
            pat_enabled=True,
            gap_distribution_enabled=True,
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
    assert [parameter.name for parameter in gap.parameters] == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "sigma_Delta_meV",
    ]
    assert [parameter.name for parameter in noise.parameters] == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "sigma_V_mV",
    ]
    assert [parameter.name for parameter in pat_gap_noise.parameters] == [
        "GN_G0",
        "T_K",
        "Delta_meV",
        "gamma_meV",
        "A_mV",
        "nu_GHz",
        "sigma_Delta_meV",
        "sigma_V_mV",
    ]
    assert gap.info["gap_distribution_order"] == "41"
    assert noise.info["noise_oversample"] == "64"


def test_registry_stacks_gap_distribution_before_noise() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41)
    config = BCSModelConfig(
        "conv",
        "np",
        gap_distribution_enabled=True,
        noise_enabled=True,
    )
    spec = get_model_spec(config)
    guess = [parameter.guess for parameter in spec.parameters]
    guess[4] = 0.006
    guess[5] = 0.04
    composed = spec.function(V_mV, *guess)
    V_support = make_bias_support_grid(V_mV, guess[5])
    gap_support = apply_gap_distribution(
        lambda delta_meV: get_model_spec(BCSModelConfig("conv", "np")).function(
            V_support,
            guess[0],
            guess[1],
            delta_meV,
            guess[3],
        ),
        guess[2],
        guess[4],
        config.gap_distribution_order,
    )
    gap_noise = apply_voltage_noise(
        V_support,
        gap_support,
        guess[5],
        config.noise_oversample,
        V_out_mV=V_mV,
    )

    assert np.allclose(composed, gap_noise)


def test_registry_stacks_pat_before_gap_distribution_and_noise() -> None:
    V_mV = np.linspace(-1.0, 1.0, 41)
    config = BCSModelConfig(
        "conv",
        "np",
        pat_enabled=True,
        gap_distribution_enabled=True,
        noise_enabled=True,
    )
    spec = get_model_spec(config)
    guess = [parameter.guess for parameter in spec.parameters]
    guess[4] = 0.15
    guess[5] = 8.5
    guess[6] = 0.006
    guess[7] = 0.04
    composed = spec.function(V_mV, *guess)
    V_support = make_bias_support_grid(V_mV, guess[7])
    gap_support = apply_gap_distribution(
        lambda delta_meV: get_model_spec(BCSModelConfig("conv", "np")).function(
            V_support,
            guess[0],
            guess[1],
            delta_meV,
            guess[3],
        ),
        guess[2],
        guess[6],
        config.gap_distribution_order,
    )
    pat_support = get_I_pat_nA(
        V_support,
        gap_support,
        guess[4],
        nu_GHz=guess[5],
        n_max=PAT_N_MAX,
    )
    pat_noise = apply_voltage_noise(
        V_support,
        pat_support,
        guess[7],
        config.noise_oversample,
        V_out_mV=V_mV,
    )

    assert np.allclose(composed, pat_noise)


def test_legacy_aliases_still_resolve() -> None:
    assert "bcs_conv_gapdist" in MODEL_OPTIONS.values()
    assert "pat_int_jax" in MODEL_OPTIONS.values()
    assert get_model_spec("bcs_int").key == "bcs_int_np"
    assert get_model_spec("bcs_conv_gapdist").key == "bcs_conv_jax_gapdist"
    assert get_model_spec("pat_int_jax").key == "bcs_int_jax_pat"
