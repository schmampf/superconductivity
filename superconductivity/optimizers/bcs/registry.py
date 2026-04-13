from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np

from ...models.basics.noise import apply_voltage_noise, make_bias_support_grid
from ...models.bcs.backend import (
    DEFAULT_E_MV,
    PAT_N_MAX,
    Backend,
    Kernel,
)
from ...utilities.types import NDArray64
from .parameters import (
    ParameterSpec,
    make_bcs_parameters,
    make_noise_parameters,
    make_pat_addon_parameters,
)

ModelFunction = Callable[..., NDArray64]
BaseModelFunction = Callable[
    [NDArray64, NDArray64, float, float, float, float],
    NDArray64,
]

BCS_INT_HTML = r"""
I(V)=G_\mathrm{N}\int_{-\infty}^{\infty}
N(E-\tfrac{eV}{2})N(E+\tfrac{eV}{2})
\left[f(E-\tfrac{eV}{2})-f(E+\tfrac{eV}{2})\right]\mathrm{d}E
"""

BCS_CONV_HTML = r"""
I(V)=G_\mathrm{N}\left[
N_2(1-f_2)\otimes N_1 f_1 -
N_1(1-f_1)\otimes N_2 f_2
\right](eV)
"""

PAT_SUFFIX_HTML = r"""
I_\mathrm{PAT}(V)=\sum_{n=-N}^{N}
J_n^2\!\left(\frac{eA}{h\nu}\right)
I_\mathrm{BCS}\!\left(V-n\frac{h\nu}{e}\right)
"""

NOISE_SUFFIX_HTML = r"""
\delta V \sim \mathcal{N}(0,\sigma_V^2),\qquad
I_\mathrm{noise}(V)=\left\langle I_0(V+\delta V)\right\rangle
"""


@dataclass(frozen=True)
class BCSModelConfig:
    kernel: Kernel = "conv"
    backend: Backend = "jax"
    pat_enabled: bool = False
    noise_enabled: bool = False
    noise_oversample: int = 64

    def __post_init__(self) -> None:
        noise_oversample = int(self.noise_oversample)
        if noise_oversample < 2:
            raise ValueError("noise_oversample must be >= 2.")
        object.__setattr__(self, "noise_oversample", noise_oversample)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    function: ModelFunction
    parameters: tuple[ParameterSpec, ...]
    info: Mapping[str, str]
    html: str


@dataclass(frozen=True)
class _BaseModelSpec:
    label: str
    html: str


_BASE_MODEL_SPECS: dict[tuple[Kernel, Backend], _BaseModelSpec] = {
    ("int", "np"): _BaseModelSpec(
        label="BCS integral (NumPy)",
        html=BCS_INT_HTML,
    ),
    ("int", "jax"): _BaseModelSpec(
        label="BCS integral (JAX)",
        html=BCS_INT_HTML,
    ),
    ("conv", "np"): _BaseModelSpec(
        label="BCS convolution (NumPy)",
        html=BCS_CONV_HTML,
    ),
    ("conv", "jax"): _BaseModelSpec(
        label="BCS convolution (JAX)",
        html=BCS_CONV_HTML,
    ),
}

_MODEL_CONFIGS = OrderedDict(
    [
        ("bcs_int", BCSModelConfig("int", "np")),
        ("bcs_int_jax", BCSModelConfig("int", "jax")),
        ("bcs_conv_jax", BCSModelConfig("conv", "jax")),
        ("bcs_conv_noise", BCSModelConfig("conv", "jax", noise_enabled=True)),
        ("pat_int_jax", BCSModelConfig("int", "jax", pat_enabled=True)),
        ("pat_conv_jax", BCSModelConfig("conv", "jax", pat_enabled=True)),
    ]
)



def get_model_key(config: BCSModelConfig) -> str:
    key = f"bcs_{config.kernel}_{config.backend}"
    if config.pat_enabled:
        key += "_pat"
    if config.noise_enabled:
        key += "_noise"
    return key



def _parse_canonical_model(model: str) -> BCSModelConfig:
    parts = model.split("_")
    if len(parts) < 3 or parts[0] != "bcs":
        raise KeyError(model)
    kernel = parts[1]
    backend = parts[2]
    if kernel not in {"int", "conv"}:
        raise KeyError(model)
    if backend not in {"np", "jax"}:
        raise KeyError(model)
    suffixes = parts[3:]
    allowed_suffixes = {"pat", "noise"}
    if any(suffix not in allowed_suffixes for suffix in suffixes):
        raise KeyError(model)
    return BCSModelConfig(
        kernel=kernel,
        backend=backend,
        pat_enabled="pat" in suffixes,
        noise_enabled="noise" in suffixes,
    )



def get_model_config(model: str | BCSModelConfig) -> BCSModelConfig:
    if isinstance(model, BCSModelConfig):
        return model
    if model in _MODEL_CONFIGS:
        return _MODEL_CONFIGS[model]
    try:
        return _parse_canonical_model(model)
    except KeyError as exc:
        raise KeyError(f"Unknown optimizer model '{model}'.") from exc



def _clone_parameters(
    parameters: tuple[ParameterSpec, ...],
) -> tuple[ParameterSpec, ...]:
    return tuple(
        ParameterSpec(
            name=parameter.name,
            label=parameter.label,
            lower=parameter.lower,
            upper=parameter.upper,
            guess=parameter.guess,
            fixed=parameter.fixed,
            value=parameter.value,
            error=parameter.error,
        )
        for parameter in parameters
    )



def _compose_parameters(config: BCSModelConfig) -> tuple[ParameterSpec, ...]:
    parameters = list(_clone_parameters(make_bcs_parameters()))
    if config.pat_enabled:
        parameters.extend(_clone_parameters(make_pat_addon_parameters()))
    if config.noise_enabled:
        parameters.extend(_clone_parameters(make_noise_parameters()))
    return tuple(parameters)



def _compose_label(config: BCSModelConfig) -> str:
    base_label = _BASE_MODEL_SPECS[(config.kernel, config.backend)].label
    suffixes: list[str] = []
    if config.pat_enabled:
        suffixes.append("PAT")
    if config.noise_enabled:
        suffixes.append("noise")
    if not suffixes:
        return base_label
    return base_label + " + " + " + ".join(suffixes)



def _energy_grid_summary(E_mV: NDArray64) -> str:
    return (
        f"{float(E_mV[0]):.1f}..{float(E_mV[-1]):.1f} meV, "
        f"{int(E_mV.size)} points"
    )



def _compose_info(config: BCSModelConfig) -> OrderedDict[str, str]:
    info = OrderedDict()
    info["kernel"] = "integral" if config.kernel == "int" else "convolution"
    info["backend"] = "NumPy" if config.backend == "np" else "JAX"
    info["PAT"] = "yes" if config.pat_enabled else "no"
    info["noise"] = "yes" if config.noise_enabled else "no"
    info["noise_oversample"] = str(config.noise_oversample)
    info["energy_grid"] = _energy_grid_summary(DEFAULT_E_MV)
    if config.pat_enabled:
        info["n_max"] = str(PAT_N_MAX)
    return info



def _compose_html(config: BCSModelConfig) -> str:
    equations = [_BASE_MODEL_SPECS[(config.kernel, config.backend)].html.strip()]
    if config.pat_enabled:
        equations.append(PAT_SUFFIX_HTML.strip())
    if config.noise_enabled:
        equations.append(NOISE_SUFFIX_HTML.strip())
    return (
        "\\[\n"
        "\\begin{gathered}\n"
        + "\\\\[0.4em]\n".join(equations)
        + "\n\\end{gathered}\n"
        "\\]"
    )



def _compose_function(config: BCSModelConfig) -> ModelFunction:
    base_function = _resolve_base_function(config.kernel, config.backend)

    def function(
        V_mV: NDArray64,
        *parameters: float,
    ) -> NDArray64:
        V_requested = np.asarray(V_mV, dtype=np.float64)
        values = np.asarray(parameters, dtype=np.float64)
        expected = len(_compose_parameters(config))
        if values.size != expected:
            raise ValueError(
                f"Model '{get_model_key(config)}' expects {expected} parameters, "
                f"got {values.size}."
            )

        GN_G0, T_K, Delta_meV, gamma_meV = values[:4]
        index = 4
        A_mV = 0.0
        nu_GHz = 0.0
        if config.pat_enabled:
            A_mV = float(values[index])
            nu_GHz = float(values[index + 1])
            index += 2
        sigma_V_mV = float(values[index]) if config.noise_enabled else 0.0

        V_evaluate = (
            make_bias_support_grid(V_requested, sigma_V_mV)
            if config.noise_enabled and sigma_V_mV > 0.0
            else V_requested
        )
        current = np.asarray(
            base_function(
                V_evaluate,
                DEFAULT_E_MV,
                float(GN_G0),
                float(T_K),
                float(Delta_meV),
                float(gamma_meV),
            ),
            dtype=np.float64,
        )

        if config.pat_enabled and A_mV != 0.0:
            from ...models.bcs import get_I_pat_nA

            current = np.asarray(
                get_I_pat_nA(
                    V_evaluate,
                    current,
                    A_mV,
                    nu_GHz=nu_GHz,
                    n_max=PAT_N_MAX,
                ),
                dtype=np.float64,
            )

        if config.noise_enabled:
            current = apply_voltage_noise(
                V_evaluate,
                current,
                sigma_V_mV,
                config.noise_oversample,
                V_out_mV=V_requested,
            )

        return np.asarray(current, dtype=np.float64)

    return function


def _resolve_base_function(
    kernel: Kernel,
    backend: Backend,
) -> BaseModelFunction:
    if backend == "np":
        from ...models.bcs.backend.np import convolution_np, integral_np

        return integral_np if kernel == "int" else convolution_np
    if backend == "jax":
        from ...models.bcs.backend.jax import convolution_jax, integral_jax

        return integral_jax if kernel == "int" else convolution_jax
    raise ValueError("Unknown backend/kernel combination.")


@lru_cache(maxsize=None)
def _cached_model_spec(config: BCSModelConfig) -> ModelSpec:
    return ModelSpec(
        key=get_model_key(config),
        label=_compose_label(config),
        function=_compose_function(config),
        parameters=_compose_parameters(config),
        info=_compose_info(config),
        html=_compose_html(config),
    )



def get_model_spec(model: str | BCSModelConfig) -> ModelSpec:
    return _cached_model_spec(get_model_config(model))


_MODEL_OPTIONS_ENTRIES = (
    ("BCS integral", "bcs_int"),
    ("BCS integral (JAX)", "bcs_int_jax"),
    ("BCS convolution (JAX)", "bcs_conv_jax"),
    ("BCS convolution + noise", "bcs_conv_noise"),
    ("PAT integral (JAX)", "pat_int_jax"),
    ("PAT convolution (JAX)", "pat_conv_jax"),
)

MODEL_OPTIONS = OrderedDict(_MODEL_OPTIONS_ENTRIES)


class _ModelRegistry(Mapping[str, ModelSpec]):
    def __iter__(self) -> Iterator[str]:
        for config in _MODEL_CONFIGS.values():
            yield get_model_key(config)

    def __len__(self) -> int:
        return len(_MODEL_CONFIGS)

    def __getitem__(self, key: str) -> ModelSpec:
        return get_model_spec(key)


MODEL_REGISTRY: Mapping[str, ModelSpec] = _ModelRegistry()


__all__ = [
    "BCSModelConfig",
    "MODEL_OPTIONS",
    "MODEL_REGISTRY",
    "ModelSpec",
    "PAT_N_MAX",
    "get_model_config",
    "get_model_key",
    "get_model_spec",
]
