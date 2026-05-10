"""Small pickle-backed project cache."""

from __future__ import annotations

import pickle
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_PROJECTS_PATH = Path(__file__).resolve().parents[2] / "projects"

_INTERNAL_NAMES = {
    "name",
    "path",
    "items",
    "file_path",
    "save",
    "save_cache",
    "keys",
    "remove",
    "values",
    "get",
    "update",
}


@dataclass(slots=True)
class ProjectCache:
    """Persistent namespace for arbitrary trusted Python objects."""

    name: str
    path: str | Path = field(default_factory=lambda: _DEFAULT_PROJECTS_PATH)
    items: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.path = Path(self.path)
        self.items = dict(self.items)

    @property
    def file_path(self) -> Path:
        """Return the default pickle file for this cache."""
        return project_cache_path(self.path, self.name)

    def save_cache(self, path: str | Path | None = None) -> Path:
        """Save this cache and return the written file path."""
        return save_cache(self, path=path)

    def save(self, path: str | Path | None = None) -> Path:
        """Short alias for :meth:`save_cache`."""
        return self.save_cache(path=path)

    def keys(self) -> tuple[str, ...]:
        """Return stored item keys."""
        return tuple(self.items)

    def values(self) -> tuple[object, ...]:
        """Return stored item values."""
        return tuple(self.items.values())

    def get(self, key: str, default: object = None) -> object:
        """Return one item or ``default`` when absent."""
        return self.items.get(str(key), default)

    def remove(self, *keys: str) -> None:
        """Remove one or more stored items by key."""
        for key in keys:
            del self.items[str(key)]

    def update(
        self,
        entries: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Update cache items from a mapping and keyword entries."""
        if entries is not None:
            self.items.update({str(key): value for key, value in entries.items()})
        self.items.update({str(key): value for key, value in kwargs.items()})

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self.items

    def __iter__(self) -> Iterator[str]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, key: str) -> object:
        return self.items[str(key)]

    def __setitem__(self, key: str, value: object) -> None:
        self.items[str(key)] = value

    def __delitem__(self, key: str) -> None:
        del self.items[str(key)]

    def __getattr__(self, name: str) -> object:
        try:
            items = object.__getattribute__(self, "items")
        except AttributeError as exc:
            raise AttributeError(name) from exc
        try:
            return items[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value: object) -> None:
        if name in _INTERNAL_NAMES or name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self.items[name] = value

    def __getstate__(self) -> dict[str, object]:
        """Return pickle state for slotted instances."""
        return {
            "name": self.name,
            "path": self.path,
            "items": self.items,
        }

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore pickle state for slotted instances."""
        object.__setattr__(self, "name", str(state["name"]))
        object.__setattr__(self, "path", Path(state["path"]))
        object.__setattr__(self, "items", dict(state["items"]))


def project_cache_path(path: str | Path, name: str) -> Path:
    """Return the default single-file cache path."""
    safe_name = str(name).strip() or "project"
    return Path(path) / f"{safe_name}.pkl"


def list_caches(path: str | Path) -> tuple[str, ...]:
    """Return available project cache names in ``path``."""
    root = Path(path)
    return tuple(sorted(file.stem for file in root.glob("*.pkl") if file.is_file()))


def make_cache(name: str, path: str | Path | None = None) -> ProjectCache:
    """Create an empty project cache."""
    return ProjectCache(
        name=name,
        path=_DEFAULT_PROJECTS_PATH if path is None else path,
    )


def save_cache(
    cache: ProjectCache,
    path: str | Path | None = None,
) -> Path:
    """Save one cache with pickle and return the written path.

    Pickle cache files should only be loaded from trusted local sources.
    """
    if not isinstance(cache, ProjectCache):
        raise TypeError("cache must be a ProjectCache.")
    out_path = Path(path) if path is not None else cache.file_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as file:
        pickle.dump(cache, file, protocol=pickle.HIGHEST_PROTOCOL)
    return out_path


def load_cache(
    name_or_path: str | Path,
    path: str | Path | None = None,
) -> ProjectCache:
    """Load a cache by file path or by name plus optional project path.

    Pickle cache files should only be loaded from trusted local sources.
    """
    cache_path = _resolve_cache_file(name_or_path=name_or_path, path=path)
    with cache_path.open("rb") as file:
        cache = pickle.load(file)
    if not isinstance(cache, ProjectCache):
        raise TypeError(f"{cache_path} does not contain a ProjectCache.")
    return cache


def entry_kind(value: object) -> str:
    """Return a coarse UI grouping for one cache entry."""
    type_name = type(value).__name__
    if type_name == "TransportDatasetSpec":
        return "transport"
    if type_name in {"Trace", "Traces"}:
        return "traces"
    if type_name.endswith("Spec"):
        return "spec"
    if type_name.endswith("Dataset"):
        return "dataset"
    return "misc"


def cache_summary(cache: ProjectCache) -> tuple[dict[str, object], ...]:
    """Return table-friendly summary rows for cache entries."""
    if not isinstance(cache, ProjectCache):
        raise TypeError("cache must be a ProjectCache.")
    return tuple(
        {
            "key": key,
            "kind": entry_kind(value),
            "type": type(value).__name__,
            "summary": _entry_summary(value),
        }
        for key, value in cache.items.items()
    )


def _resolve_cache_file(
    *,
    name_or_path: str | Path,
    path: str | Path | None,
) -> Path:
    raw = Path(name_or_path)
    if path is not None:
        root = Path(path)
        name = raw.name if raw.suffix != ".pkl" else raw.stem
        direct = root / f"{name}.pkl"
        if direct.exists():
            return direct
        return direct
    if raw.suffix == ".pkl" or raw.parent != Path("."):
        return raw
    direct = _DEFAULT_PROJECTS_PATH / f"{raw.name}.pkl"
    return direct


def _entry_summary(value: object) -> str:
    shape = _shape_summary(value)
    if shape:
        return shape
    if isinstance(value, Mapping):
        return f"{len(value)} keys"
    if isinstance(value, (list, tuple, set, frozenset)):
        return f"{len(value)} items"
    if np.isscalar(value):
        return repr(value)
    keys = getattr(value, "keys", None)
    if callable(keys):
        try:
            return f"{len(tuple(keys()))} keys"
        except Exception:
            return ""
    return ""


def _shape_summary(value: object) -> str:
    data = getattr(value, "data", None)
    axes = getattr(value, "axes", None)
    if data is not None and axes is not None:
        shapes = [
            _array_shape(getattr(entry, "values", None))
            for entry in data
            if getattr(entry, "values", None) is not None
        ]
        axis_labels = [
            str(getattr(entry, "code_label", ""))
            for entry in axes
            if getattr(entry, "code_label", None) is not None
        ]
        shape_text = ", ".join(shape for shape in shapes if shape)
        axis_text = ", ".join(label for label in axis_labels if label)
        if shape_text and axis_text:
            return f"shape {shape_text}; axes {axis_text}"
        if shape_text:
            return f"shape {shape_text}"

    traces = getattr(value, "traces", None)
    if traces is not None:
        try:
            return f"{len(traces)} traces"
        except TypeError:
            return ""

    values = getattr(value, "values", None)
    if values is not None:
        shape = _array_shape(values)
        if shape:
            return f"shape {shape}"
    return ""


def _array_shape(value: Any) -> str:
    try:
        shape = tuple(np.asarray(value).shape)
    except Exception:
        return ""
    if shape == ():
        return "scalar"
    return "x".join(str(size) for size in shape)


__all__ = [
    "ProjectCache",
    "cache_summary",
    "entry_kind",
    "list_caches",
    "load_cache",
    "make_cache",
    "project_cache_path",
    "save_cache",
]
