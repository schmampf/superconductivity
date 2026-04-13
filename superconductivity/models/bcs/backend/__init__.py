"""Source-of-truth BCS backend constants and type aliases."""

from __future__ import annotations

from typing import Literal

import numpy as np

Backend = Literal["np", "jax"]
Kernel = Literal["int", "conv"]

DEFAULT_E_MV = np.linspace(-4.0, 4.0, 4001, dtype=np.float64)
PAT_N_MAX = 50

__all__ = [
    "Backend",
    "Kernel",
    "DEFAULT_E_MV",
    "PAT_N_MAX",
]
