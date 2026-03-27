from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from importlib import import_module
from typing import Callable

from ...utilities.types import NDArray64
from .parameters import ParameterSpec

ModelFunction = Callable[..., NDArray64]


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    function: ModelFunction
    parameters: tuple[ParameterSpec, ...]
    info: Mapping[str, str]
    html: str


@dataclass(frozen=True)
class _ModelEntry:
    key: str
    label: str
    module_name: str


_MODEL_ENTRIES: tuple[_ModelEntry, ...] = (
    _ModelEntry("bcs_sis_int", "BCS SIS integral", "bcs_sis_int"),
    _ModelEntry("bcs_sis_int_jax", "BCS SIS integral (JAX)", "bcs_sis_int_jax"),
    _ModelEntry(
        "bcs_sis_conv_jax",
        "BCS SIS convolution (JAX)",
        "bcs_sis_conv_jax",
    ),
    _ModelEntry(
        "bcs_sis_conv_noise",
        "BCS SIS convolution + noise",
        "bcs_sis_conv_noise",
    ),
    _ModelEntry("bcs_sin_int", "BCS SIN integral", "bcs_sin_int"),
    _ModelEntry("bcs_sin_int_jax", "BCS SIN integral (JAX)", "bcs_sin_int_jax"),
    _ModelEntry(
        "bcs_sin_conv_jax",
        "BCS SIN convolution (JAX)",
        "bcs_sin_conv_jax",
    ),
    _ModelEntry("pat_sis_int_jax", "PAT SIS integral (JAX)", "pat_sis_int_jax"),
    _ModelEntry(
        "pat_sis_conv_jax",
        "PAT SIS convolution (JAX)",
        "pat_sis_conv_jax",
    ),
)


class _LazyModelRegistry(Mapping[str, ModelSpec]):
    def __init__(self, entries: tuple[_ModelEntry, ...]) -> None:
        self._entries = OrderedDict((entry.key, entry) for entry in entries)
        self._cache: dict[str, ModelSpec] = {}

    def __getitem__(self, key: str) -> ModelSpec:
        if key not in self._entries:
            raise KeyError(key)
        if key not in self._cache:
            module_name = self._entries[key].module_name
            module = import_module(f"{__package__}.{module_name}")
            self._cache[key] = module.MODEL
        return self._cache[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._entries)

    def __len__(self) -> int:
        return len(self._entries)


MODEL_REGISTRY: Mapping[str, ModelSpec] = _LazyModelRegistry(_MODEL_ENTRIES)
MODEL_OPTIONS = OrderedDict((entry.label, entry.key) for entry in _MODEL_ENTRIES)


def get_model_spec(model: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[model]
    except KeyError as exc:
        raise KeyError(f"Unknown optimizer model '{model}'.") from exc
