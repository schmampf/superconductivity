"""Curated public API.

This module re-exports the functions/constants you most frequently use, so that
users can do either:

    import superconductivity.api as sc
    sc.get_I_bcs_nA(...)

or:

    from superconductivity.api import *

The package-level convenience import (`import superconductivity as sc`) is
implemented in `superconductivity/__init__.py` and forwards missing attributes
to this module.
"""

# import from abs
from .models.abs import (
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

# imports from bcs_jnp
from .models.bcs_jnp import get_I_bcs_jnp_nA as get_I_bcs_nA

# imports from bcs
from .models.bcs_np import get_Delta_meV, get_dos, get_f, get_T_c_K

# imports from btk
from .models.btk import get_AB_btk, get_I_btk_nA, get_Z_btk

# import from fcs
from .models.fcs_pbar import get_I_fcs_pbar_nA as get_I_fcs_nA
from .models.ha_asym import get_I_ha_asym_nA

# import from ha
from .models.ha_sym import get_I_ha_sym_nA as get_I_ha_nA

# import from pat
from .models.pat import get_I_pat_nA

# import from rsj
from .models.rsj import get_I_rsj_meso_nA, get_I_rsj_nA

# import from ss
from .models.ss import get_I_p_abs_nA

# import colors
from .style.cpd5 import (
    get_color,
    get_colors,
    seeblau20,
    seeblau35,
    seeblau65,
    seeblau80,
    seeblau100,
    seeblau120,
    seegrau20,
    seegrau35,
    seegrau65,
    seegrau80,
    seegrau100,
    seegrau120,
)

# import styles
from .style.man import man
from .style.thesislayout import get_ext, get_figure, map_layout, theory_layout

# import utilities
from .utilities.constants import G_0_muS, h_e_pVs, k_B_meV
from .utilities.functions import bin_y_over_x, oversample
from .utilities.functions_jax import jnp_interp_y_of_x
from .utilities.types import NDArray64

seeblau = [seeblau120, seeblau100, seeblau65, seeblau35, seeblau20]
seegrau = [seegrau120, seegrau100, seegrau65, seegrau35, seegrau20]

# Public API for `from superconductivity.api import *`
__all__ = [name for name in list(globals().keys()) if not name.startswith("_")]
