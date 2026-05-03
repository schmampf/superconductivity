from .bcs_theory import GUIBCSTheoryTabMixin
from .base import GUITabsBaseMixin
from .data import GUIDataTabMixin
from .fitting import GUIFitTabMixin
from .mar_fitting import GUIMARFitTabMixin
from .mar_theory import GUIMARTheoryTabMixin
from .measurement import GUIMeasurementTabMixin
from .offset import GUIOffsetTabMixin
from .psd import GUIPSDTabMixin
from .rcsj_theory import GUIRCSJTheoryTabMixin
from .sampling import GUISamplingTabMixin


class GUITabsMixin(
    GUITabsBaseMixin,
    GUIMeasurementTabMixin,
    GUIDataTabMixin,
    GUIPSDTabMixin,
    GUIOffsetTabMixin,
    GUISamplingTabMixin,
    GUIFitTabMixin,
    GUIMARFitTabMixin,
    GUIBCSTheoryTabMixin,
    GUIMARTheoryTabMixin,
    GUIRCSJTheoryTabMixin,
):
    pass


__all__ = ["GUITabsMixin"]
