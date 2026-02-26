from typing import Callable, Optional, Sequence, Tuple, TypeAlias, Union

import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

NDArray64: TypeAlias = NDArray[np.float64]
JNDArray: TypeAlias = jnp.ndarray

JInterpolator: TypeAlias = Callable[[JNDArray], JNDArray]
ModelFunction: TypeAlias = Callable[..., NDArray64]

ModelType: TypeAlias = tuple[ModelFunction, NDArray[np.bool]]
ParameterType: TypeAlias = tuple[float, tuple[float, float], bool]


Number = Union[int, float]
RGB = Tuple[Number, Number, Number]
RGBA = Tuple[Number, Number, Number, Number]
COLOR = Union[RGB, RGBA, Sequence[Number]]
LIM = Optional[Tuple[Optional[float], Optional[float]]]
