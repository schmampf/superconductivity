from typing import Any, Callable, Optional, Sequence, Tuple, TypeAlias, Union

import numpy as np
from numpy.typing import NDArray

try:
    import jax.numpy as jnp
except ModuleNotFoundError:  # pragma: no cover - exercised via import only.
    jnp = None

NDArray64: TypeAlias = NDArray[np.float64]
if jnp is not None:
    JNDArray: TypeAlias = jnp.ndarray
else:
    JNDArray: TypeAlias = Any

JInterpolator: TypeAlias = Callable[[JNDArray], JNDArray]
ModelFunction: TypeAlias = Callable[..., NDArray64]

ModelType: TypeAlias = tuple[ModelFunction, NDArray[np.bool_]]
ParameterType: TypeAlias = tuple[float, tuple[float, float], bool]


Number = Union[int, float]
RGB = Tuple[Number, Number, Number]
RGBA = Tuple[Number, Number, Number, Number]
COLOR = Union[RGB, RGBA, Sequence[Number]]
LIM = Optional[Tuple[Optional[float], Optional[float]]]
