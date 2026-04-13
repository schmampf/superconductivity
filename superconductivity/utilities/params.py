"""Shared parameter metadata helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .label import LabelMeta


@dataclass(frozen=True, slots=True)
class ParamSpec(LabelMeta):
    """Parameter metadata shared by fitting and GUI tables."""

    def __post_init__(self) -> None:
        LabelMeta.__post_init__(self)


__all__ = ["ParamSpec"]
