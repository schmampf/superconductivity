import numpy as np

from superconductivity.optimizers.models import MODEL_OPTIONS, get_model_spec

EXPECTED_MODELS = [
    "bcs_sis_int",
    "bcs_sis_int_jax",
    "bcs_sis_conv_jax",
    "bcs_sis_conv_noise",
    "bcs_sin_int",
    "bcs_sin_int_jax",
    "bcs_sin_conv_jax",
    "pat_sis_int_jax",
    "pat_sis_conv_jax",
]


def test_registry_resolves_all_declared_models() -> None:
    assert list(MODEL_OPTIONS.values()) == EXPECTED_MODELS

    V_mV = np.linspace(-1.0, 1.0, 21)
    for key in EXPECTED_MODELS:
        spec = get_model_spec(key)
        guess = [parameter.guess for parameter in spec.parameters]
        curve = spec.function(V_mV, *guess)

        assert spec.key == key
        assert callable(spec.function)
        assert len(spec.parameters) in {4, 5, 6}
        assert curve.shape == V_mV.shape
        assert np.all(np.isfinite(curve))
