from .base import GUITabsBaseMixin
from .fitting import GUIFitTabMixin
from .offset import GUIOffsetTabMixin
from .psd import GUIPSDTabMixin
from .sampling import GUISamplingTabMixin


class GUITabsMixin(
    GUITabsBaseMixin,
    GUIPSDTabMixin,
    GUIOffsetTabMixin,
    GUISamplingTabMixin,
    GUIFitTabMixin,
):
    pass


__all__ = ["GUITabsMixin"]
