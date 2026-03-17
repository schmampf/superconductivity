from __future__ import annotations

from .models import get_model
from ...utilities.types import NDArray64


def pat_trace(V_mV: NDArray64, params: NDArray64) -> NDArray64:
    function, parameter_mask = get_model(model="pat")
    return function(V_mV, *params[parameter_mask])
