from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ParameterSpec:
    name: str
    label: str
    lower: float
    upper: float
    guess: float
    fixed: bool = False
    value: Optional[float] = None
    error: Optional[float] = None
