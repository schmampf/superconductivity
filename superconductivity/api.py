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
from .models.abs import get_rho
from .models.abs import get_E_abs
from .models.abs import get_E_abs_meV

from .models.abs import get_cpr_ab
from .models.abs import get_cpr_ab_nA
from .models.abs import get_cpr_abs
from .models.abs import get_cpr_abs_nA
from .models.abs import get_cpr_ko1
from .models.abs import get_cpr_ko1_nA
from .models.abs import get_cpr_ko2
from .models.abs import get_cpr_ko2_nA

from .models.abs import get_Ic_ab
from .models.abs import get_Ic_ab_nA
from .models.abs import get_Ic_abs
from .models.abs import get_Ic_abs_nA
from .models.abs import get_Ic_ko1
from .models.abs import get_Ic_ko1_nA
from .models.abs import get_Ic_ko2
from .models.abs import get_Ic_ko2_nA

from .models.abs import get_IcT_ab
from .models.abs import get_IcT_ab_nA
from .models.abs import get_IcT_abs
from .models.abs import get_IcT_abs_nA
from .models.abs import get_IcT_ko1
from .models.abs import get_IcT_ko1_nA
from .models.abs import get_IcT_ko2
from .models.abs import get_IcT_ko2_nA

# imports from bcs
from .models.bcs_np import get_T_c_K
from .models.bcs_np import get_Delta_meV
from .models.bcs_np import get_f
from .models.bcs_np import get_dos

# imports from bcs_jnp
from .models.bcs_jnp import get_I_bcs_jnp_nA as get_I_bcs_nA

# imports from btk
from .models.btk import get_Z_btk
from .models.btk import get_AB_btk
from .models.btk import get_I_btk_nA

# import from ha
from .models.ha_sym import get_I_ha_sym_nA as get_I_ha_nA
from .models.ha_asym import get_I_ha_asym_nA

# import from fcs
from .models.fcs_pbar import get_I_fcs_pbar_nA as get_I_fcs_nA

# import from pat
from .models.pat import get_I_pat_nA

# import utilities
from .utilities.constants import G_0_muS
from .utilities.constants import k_B_meV
from .utilities.constants import h_e_pVs

from .utilities.functions import bin_y_over_x
from .utilities.functions import oversample

from .utilities.types import NDArray64

# import styles
from .style.man import man

from .style.thesislayout import get_ext
from .style.thesislayout import get_figure
from .style.thesislayout import theory_layout

from .style.cpd5 import get_color, get_colors
from .style.cpd5 import (
    seeblau120,
    seeblau100,
    seeblau80,
    seeblau65,
    seeblau35,
    seeblau20,
    seegrau120,
    seegrau100,
    seegrau80,
    seegrau65,
    seegrau35,
    seegrau20,
)

seeblau = [seeblau120, seeblau100, seeblau65, seeblau35, seeblau20]
seegrau = [seegrau120, seegrau100, seegrau65, seegrau35, seegrau20]

# Public API for `from superconductivity.api import *`
__all__ = [name for name in list(globals().keys()) if not name.startswith("_")]
