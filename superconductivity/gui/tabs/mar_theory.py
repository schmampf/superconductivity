from __future__ import annotations


class GUIMARTheoryTabMixin:
    def _mar_theory_tab(self):
        return self._pn.Column(
            sizing_mode="stretch_width",
            margin=0,
        )
