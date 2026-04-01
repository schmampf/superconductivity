from .base import GUITabsBaseMixin
from .data import GUIDataTabMixin
from .fitting import GUIFitTabMixin
from .measurement import GUIMeasurementTabMixin
from .offset import GUIOffsetTabMixin
from .psd import GUIPSDTabMixin
from .sampling import GUISamplingTabMixin


class GUITabsMixin(
    GUITabsBaseMixin,
    GUIMeasurementTabMixin,
    GUIDataTabMixin,
    GUIPSDTabMixin,
    GUIOffsetTabMixin,
    GUISamplingTabMixin,
    GUIFitTabMixin,
):
    pass


__all__ = ["GUITabsMixin"]
