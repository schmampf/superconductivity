"""Curated public API.

This module re-exports the functions/constants you most frequently use, so that
users can do either:

    import superconductivity.api as sc
    sc.get_Ibcs_nA(...)

or:

    from superconductivity.api import *

The package-level convenience import (`import superconductivity as sc`) is
implemented in `superconductivity/__init__.py` and forwards missing attributes
to this module.
"""

from .evaluation.analysis import (
    OffsetSpec,
    OffsetTrace,
    OffsetTraces,
    PSDSpec,
    PSDTrace,
    PSDTraces,
    offset_analysis,
    psd_analysis,
)
from .evaluation.calibration import CalibrationSpec, calibrate
from .evaluation.sampling import (
    SamplingSpec,
    binning,
    downsample_trace,
    downsample_traces,
    downsampling,
    offset_correction,
    sample,
    smooth,
    upsampling,
)
from .utilities.transport import reduced_units
from .evaluation.traces import (
    FileSpec,
    Keys,
    KeysSpec,
    Trace,
    Traces,
    TraceSpec,
    get_keys,
    get_measurement_keys,
    get_measurement_series,
    get_status_keys,
    get_status_series,
    get_traces,
    list_measurement_keys,
    list_specific_keys,
)

# imports from BCS basics / tunneling
from .models.basics import get_DeltaT_meV, get_dos, get_f, get_Tc_K
from .models.bcs import get_Ibcs_nA, pat_kernel, sim_bcs

# imports from btk
from .models.mar import get_AB_btk, get_I_btk_nA, get_I_fcs_nA, get_I_ha_asym_nA
from .models.mar import get_I_ha_sym_nA as get_I_ha_nA
from .models.mar import get_Imar_nA, get_Z_btk

# import from abs
from .models.rcsj.josephson.abs import (
    get_cpr_ab,
    get_cpr_ab_nA,
    get_cpr_abs,
    get_cpr_abs_nA,
    get_cpr_ko1,
    get_cpr_ko1_nA,
    get_cpr_ko2,
    get_cpr_ko2_nA,
    get_E_abs,
    get_E_abs_meV,
    get_Ic_ab,
    get_Ic_ab_nA,
    get_Ic_abs,
    get_Ic_abs_nA,
    get_Ic_ko1,
    get_Ic_ko1_nA,
    get_Ic_ko2,
    get_Ic_ko2_nA,
    get_IcT_ab,
    get_IcT_ab_nA,
    get_IcT_abs,
    get_IcT_abs_nA,
    get_IcT_ko1,
    get_IcT_ko1_nA,
    get_IcT_ko2,
    get_IcT_ko2_nA,
    get_rho,
)

# import from ss
try:
    from .models.rcsj.josephson.sp import get_I_p_abs_nA
except ModuleNotFoundError as exc:
    if exc.name != "scipy":
        raise

    def get_I_p_abs_nA(*args, **kwargs):
        raise ImportError("get_I_p_abs_nA requires scipy in the active environment.")


# import colors
from .style.cpd4 import cmap as get_cmap
from .style.cpd5 import (
    get_color,
    get_colors,
    seeblau20,
    seeblau35,
    seeblau65,
    seeblau80,
    seeblau100,
    seeblau120,
    seegrau10,
    seegrau20,
    seegrau35,
    seegrau65,
    seegrau80,
    seegrau100,
    seegrau120,
)

# import styles
from .style.thesislayout import get_ext, get_figure, map_layout, theory_layout

# import utilities
from .utilities.constants import G0_muS, h_pVs, kB_meV_K
from .utilities.functions.binning import bin
from .utilities.functions.binning import bin as bin_y_over_x
from .utilities.legacy.functions import ragged_to_array, upsample
from .utilities.legacy.functions_jax import jinterp_y_of_x
from .utilities.types import NDArray64

# import from rsj
# from .models.rsj_old import get_I_rsj_meso_nA, get_I_rsj_nA


seeblau = [seeblau120, seeblau100, seeblau65, seeblau35, seeblau20]
seegrau = [
    seegrau120,
    seegrau100,
    seegrau65,
    seegrau35,
    seegrau20,
    seegrau10,
]

# Public API for `from superconductivity.api import *`
__all__ = [name for name in list(globals().keys()) if not name.startswith("_")]
