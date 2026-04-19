"""Shared label metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .labels import LABEL_ROWS


@dataclass(frozen=True, slots=True)
class LabelSpec:
    """Label-only metadata shared by axes, parameters, and data."""

    code_label: str
    print_label: str
    html_label: str
    latex_label: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "code_label", str(self.code_label))
        object.__setattr__(self, "print_label", str(self.print_label))
        object.__setattr__(self, "html_label", str(self.html_label))
        object.__setattr__(self, "latex_label", str(self.latex_label))


LABELS: dict[str, LabelSpec] = {
    name: LabelSpec(
        code_label=name,
        print_label=str(row["print_label"]),
        html_label=str(row["html_label"]),
        latex_label=str(row["latex_label"]),
    )
    for name, row in LABEL_ROWS.items()
}


def label(name: str) -> LabelSpec:
    """Return a shared label metadata object by name."""
    if name in LABELS:
        return LABELS[name]
    text = str(name)
    return LabelSpec(
        code_label=text,
        print_label=text,
        html_label=text,
        latex_label=text,
    )


__all__ = ["LabelSpec", "LABELS", "label"]
