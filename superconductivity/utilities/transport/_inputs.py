"""Shared input normalization for transport helpers."""

from __future__ import annotations

from collections.abc import Sequence

from .dataset import TransportDatasetSpec


def normalize_samples(
    sample_or_samples: TransportDatasetSpec | Sequence[TransportDatasetSpec],
) -> tuple[tuple[TransportDatasetSpec, ...], bool, type[Sequence[TransportDatasetSpec]] | None]:
    """Normalize one transport dataset or a sequence of them."""
    if isinstance(sample_or_samples, TransportDatasetSpec):
        return (sample_or_samples,), False, None
    if isinstance(sample_or_samples, Sequence):
        samples = tuple(sample_or_samples)
        if not samples:
            raise ValueError("at least one transport dataset is required.")
        for sample in samples:
            if not isinstance(sample, TransportDatasetSpec):
                raise TypeError(
                    "samples must be TransportDatasetSpec instances.",
                )
        return samples, True, type(sample_or_samples)
    raise TypeError(
        "expected a TransportDatasetSpec or a sequence of TransportDatasetSpec.",
    )


def restore_samples(
    samples: tuple[TransportDatasetSpec, ...],
    *,
    was_sequence: bool,
    sequence_type: type[Sequence[TransportDatasetSpec]] | None,
) -> TransportDatasetSpec | Sequence[TransportDatasetSpec]:
    """Restore normalized samples to a single dataset or a sequence."""
    if not was_sequence:
        return samples[0]
    if sequence_type is tuple:
        return samples
    if sequence_type is list:
        return list(samples)
    try:
        return sequence_type(samples)  # type: ignore[misc]
    except TypeError:
        return samples


__all__ = ["normalize_samples", "restore_samples"]
