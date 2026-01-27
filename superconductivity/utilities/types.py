from typing import TypeAlias
from typing import Callable

import numpy as np
from numpy.typing import NDArray

import jax.numpy as jnp

NDArray64: TypeAlias = NDArray[np.float64]
JNDArray: TypeAlias = jnp.ndarray

JInterpolator: TypeAlias = Callable[[JNDArray], JNDArray]
ModelFunction: TypeAlias = Callable[..., NDArray64]

ModelType: TypeAlias = tuple[ModelFunction, NDArray[np.bool]]
ParameterType: TypeAlias = tuple[float, tuple[float, float], bool]
