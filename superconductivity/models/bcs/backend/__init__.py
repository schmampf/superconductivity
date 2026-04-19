"""Source-of-truth BCS backend constants and type aliases."""

from __future__ import annotations

from typing import Literal

import numpy as np

Backend = Literal["np", "jax"]
Kernel = Literal["int", "conv"]

E0_meV = np.linspace(-4.0, 4.0, 4001, dtype=np.float64)
Nmax_ = 50

__all__ = [
    "Backend",
    "Kernel",
    "E0_meV",
    "Nmax_",
]
