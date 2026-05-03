from __future__ import annotations


class GUIMARFitTabMixin:
    def _mar_fit_tab(self):
        return self._pn.Column(
            sizing_mode="stretch_width",
            margin=0,
        )
