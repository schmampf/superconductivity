"""Corporate design colors and perceptual colormaps (University of Konstanz).

This module provides a compact color toolbox for Matplotlib. It contains the
official corporate design base colors as sRGB values, 5-step named palettes
(dark -> light), and colormap constructors based on CAM02-UCS for perceptually
meaningful ramps.

Conventions
-----------
- All RGB values are represented as float arrays in sRGB1, i.e. in [0, 1].
- Palettes are 5-step ramps ordered dark -> light with shades
  ["100", "80", "65", "35", "20"] mapped to indices [0, 1, 2, 3, 4].
- Optional channel permutations can be applied to palette swatches via either
  a permutation key (e.g. "magenta") or an explicit index permutation
  (2, 0, 1).

Perceptual colormaps
--------------------
Two colormap families are provided:

1) `cmap_tinted_black_to_white`
   Black -> white with perceived lightness shaped in CAM02-UCS J' and a
   symmetric tint envelope that vanishes at both ends. Gamut safety is
   enforced by reducing chroma (a', b') while keeping J' fixed.

2) `cmap_tinted_grey`
   Interpolation between multiple RGB anchor points at constant CAM02-UCS
   lightness J'. The path is piecewise-linear in (a', b'). Endpoints are
   projected onto the constant-J' plane, so output endpoints generally do not
   exactly match the input RGB anchor colors unless they share the same J'.

Out-of-range and invalid values
-------------------------------
Wrapper functions (`get_cmap`, `get_cmap_multi`) support setting `under`,
`over`, and `bad` colors for colormaps. If these are not specified, the
intended defaults are:
- under: first colormap color (min)
- over: last colormap color (max)
- bad: `rot`

Public API
----------
Color access:
- `get_color(palette, shade, permutation) -> (3,)`
- `get_colors(index, palettes, shades, permutations, ...) -> (4,)`

Colormap construction:
- `cmap_tinted_black_to_white(...) -> ListedColormap`
- `cmap_tinted_grey(...) -> ListedColormap`
- `get_cmap(...) -> ListedColormap`
- `get_cmap_multi(...) -> ListedColormap`

Dependencies
------------
- NumPy
- Matplotlib
- colorspacious (CAM02-UCS conversions)

Version
-------
2025-12-28
Oliver Aschenbrenner
"""

from typing import Dict
from typing import TypeAlias
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from matplotlib.colors import ListedColormap
from matplotlib.colors import to_rgba

from colorspacious import cspace_convert

NDArray64: TypeAlias = NDArray[np.float64]

# region color base

# seeblau
seeblau20: NDArray64 = np.array([204, 238, 249], dtype=np.float64) / 255
seeblau35: NDArray64 = np.array([166, 225, 244], dtype=np.float64) / 255
seeblau65: NDArray64 = np.array([89, 199, 235], dtype=np.float64) / 255
seeblau80: NDArray64 = np.array([46, 185, 230], dtype=np.float64) / 255
seeblau100: NDArray64 = np.array([0, 169, 224], dtype=np.float64) / 255
seeblau120: NDArray64 = np.array([0, 142, 206], dtype=np.float64) / 255

# peach
peach20: NDArray64 = np.array([254, 226, 221], dtype=np.float64) / 255
peach35: NDArray64 = np.array([254, 207, 199], dtype=np.float64) / 255
peach65: NDArray64 = np.array([255, 184, 172], dtype=np.float64) / 255
peach80: NDArray64 = np.array([254, 160, 144], dtype=np.float64) / 255
peach100: NDArray64 = np.array([255, 142, 123], dtype=np.float64) / 255

# seegrau
seegrau20: NDArray64 = np.array([225, 226, 229], dtype=np.float64) / 255
seegrau35: NDArray64 = np.array([184, 188, 193], dtype=np.float64) / 255
seegrau65: NDArray64 = np.array([154, 160, 167], dtype=np.float64) / 255
seegrau80: NDArray64 = np.array([137, 143, 149], dtype=np.float64) / 255
seegrau100: NDArray64 = np.array([115, 120, 126], dtype=np.float64) / 255
seegrau120: NDArray64 = np.array([77, 80, 84], dtype=np.float64) / 255

# petrol
petrol20: NDArray64 = np.array([156, 198, 207], dtype=np.float64) / 255
petrol35: NDArray64 = np.array([106, 170, 183], dtype=np.float64) / 255
petrol65: NDArray64 = np.array([57, 141, 159], dtype=np.float64) / 255
petrol80: NDArray64 = np.array([7, 113, 135], dtype=np.float64) / 255
petrol100: NDArray64 = np.array([3, 95, 114], dtype=np.float64) / 255

# seegruen
seegruen20: NDArray64 = np.array([113, 209, 204], dtype=np.float64) / 255
seegruen35: NDArray64 = np.array([84, 191, 183], dtype=np.float64) / 255
seegruen65: NDArray64 = np.array([10, 163, 152], dtype=np.float64) / 255
seegruen80: NDArray64 = np.array([10, 144, 134], dtype=np.float64) / 255
seegruen100: NDArray64 = np.array([6, 126, 121], dtype=np.float64) / 255

# karpfenblau
karpfenblau20: NDArray64 = np.array([180, 188, 214], dtype=np.float64) / 255
karpfenblau35: NDArray64 = np.array([130, 144, 187], dtype=np.float64) / 255
karpfenblau65: NDArray64 = np.array([88, 107, 164], dtype=np.float64) / 255
karpfenblau80: NDArray64 = np.array([62, 84, 150], dtype=np.float64) / 255
karpfenblau100: NDArray64 = np.array([50, 67, 118], dtype=np.float64) / 255

# pinky
pinky20: NDArray64 = np.array([243, 191, 203], dtype=np.float64) / 255
pinky35: NDArray64 = np.array([236, 160, 178], dtype=np.float64) / 255
pinky65: NDArray64 = np.array([230, 128, 152], dtype=np.float64) / 255
pinky80: NDArray64 = np.array([224, 96, 126], dtype=np.float64) / 255
pinky100: NDArray64 = np.array([202, 74, 104], dtype=np.float64) / 255

# bordeaux
bordeaux20: NDArray64 = np.array([210, 166, 180], dtype=np.float64) / 255
bordeaux35: NDArray64 = np.array([188, 122, 143], dtype=np.float64) / 255
bordeaux65: NDArray64 = np.array([165, 77, 105], dtype=np.float64) / 255
bordeaux80: NDArray64 = np.array([142, 32, 67], dtype=np.float64) / 255
bordeaux100: NDArray64 = np.array([119, 20, 52], dtype=np.float64) / 255

# sonstige
schwarz: NDArray64 = np.array([0, 0, 0], dtype=np.float64) / 255
weiss: NDArray64 = np.array([255, 255, 255], dtype=np.float64) / 255
gruen: NDArray64 = np.array([124, 202, 137], dtype=np.float64) / 255
gelb: NDArray64 = np.array([239, 220, 96], dtype=np.float64) / 255
rot: NDArray64 = np.array([208, 21, 86], dtype=np.float64) / 255

# endregion

# region palettes

# color palettes (ordered dark -> light). Shape: (5, 3)
seeblau_palette: NDArray64 = np.concatenate(
    [
        seeblau100,
        seeblau80,
        seeblau65,
        seeblau35,
        seeblau20,
    ]
).reshape(5, 3)

peach_palette: NDArray64 = np.concatenate(
    [
        peach100,
        peach80,
        peach65,
        peach35,
        peach20,
    ]
).reshape(5, 3)

seegrau_palette: NDArray64 = np.concatenate(
    [
        seegrau100,
        seegrau80,
        seegrau65,
        seegrau35,
        seegrau20,
    ]
).reshape(5, 3)

petrol_palette: NDArray64 = np.concatenate(
    [
        petrol100,
        petrol80,
        petrol65,
        petrol35,
        petrol20,
    ]
).reshape(5, 3)

seegruen_palette: NDArray64 = np.concatenate(
    [
        seegruen100,
        seegruen80,
        seegruen65,
        seegruen35,
        seegruen20,
    ]
).reshape(5, 3)

karpfenblau_palette: NDArray64 = np.concatenate(
    [
        karpfenblau100,
        karpfenblau80,
        karpfenblau65,
        karpfenblau35,
        karpfenblau20,
    ]
).reshape(5, 3)

pinky_palette: NDArray64 = np.concatenate(
    [
        pinky100,
        pinky80,
        pinky65,
        pinky35,
        pinky20,
    ]
).reshape(5, 3)

bordeaux_palette: NDArray64 = np.concatenate(
    [
        bordeaux100,
        bordeaux80,
        bordeaux65,
        bordeaux35,
        bordeaux20,
    ]
).reshape(5, 3)

# endregion

# region GLOBAL parameter

PALETTES: Dict[str, NDArray64] = {
    "seeblau": seeblau_palette,
    "peach": peach_palette,
    "seegrau": seegrau_palette,
    "petrol": petrol_palette,
    "seegruen": seegruen_palette,
    "karpfenblau": karpfenblau_palette,
    "pinky": pinky_palette,
    "bordeaux": bordeaux_palette,
}

SHADE_TO_INDEX: Dict[str, int] = {
    "100": 0,
    "80": 1,
    "65": 2,
    "35": 3,
    "20": 4,
}

PERMUTATIONS: Dict[str, tuple[int, int, int]] = {
    "standard": (0, 1, 2),
    "mint": (0, 2, 1),
    "lila": (1, 0, 2),
    "lime": (1, 2, 0),
    "magenta": (2, 0, 1),
    "amber": (2, 1, 0),
}

# endregion

# region private functions


def _get_palette(name: str) -> NDArray64:
    """Return a (5, 3) palette (dark -> light) by name.

    Parameters
    ----------
    name:
        Palette key, e.g. "seeblau", "peach", "seegrau".

    Returns
    -------
    palette:
        Array of shape (5, 3) with RGB values in [0, 1]. The ordering is
        dark -> light and corresponds to ["100", "80", "65", "35", "20"].

    Raises
    ------
    KeyError
        If `name` is unknown.
    """

    key = name.strip().lower()
    if key not in PALETTES:
        raise KeyError(
            f"Unknown palette '{name}'.",
            f"Available: {sorted(PALETTES)}",
        )
    return PALETTES[key]


def _permuta_rgb(
    rgb: NDArray64,
    permutation: str | tuple[int, int, int] = "standard",
) -> NDArray64:
    """Permute RGB channels.

    Parameters
    ----------
    rgb:
        Array of shape (3,) or (..., 3) in sRGB1.
    permutation:
        Either a key in `PERMUTATIONS` (e.g. "standard", "magenta") or an
        explicit index tuple such as (2, 0, 1).

    Returns
    -------
    rgb_perm:
        Array with the same shape as `rgb`.

    Raises
    ------
    KeyError
        If a string `permutation` key is unknown.
    ValueError
        If an explicit permutation does not have length 3.
    """

    if isinstance(permutation, str):
        key = permutation.strip().lower()
        if key not in PERMUTATIONS:
            raise KeyError(
                f"Unknown permutation '{permutation}'.",
                f"Available: {sorted(PERMUTATIONS)}",
            )
        idx = PERMUTATIONS[key]
    else:
        idx = permutation

    if len(idx) != 3:
        raise ValueError("permutation must have length 3")

    return rgb[..., idx]


def _in_gamut(
    rgb: NDArray64,
) -> NDArray[np.bool_] | bool:
    """Check whether RGB values lie within the sRGB gamut [0, 1].

    Parameters
    ----------
    rgb:
        Array of shape (3,) or (..., 3) containing RGB values.

    Returns
    -------
    ok:
        If `rgb` has shape (3,), returns a bool. Otherwise returns a boolean
        mask with shape rgb.shape[:-1].
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    ok = (rgb >= 0.0) & (rgb <= 1.0)
    if rgb.ndim == 1:
        return bool(np.all(ok))
    return np.all(ok, axis=-1)


def _gamut_limit_cam02ucs_jab(
    jab: NDArray64,
    iters: int = 18,
) -> tuple[NDArray64, NDArray64]:
    """Convert CAM02-UCS J'a'b' to sRGB while enforcing sRGB gamut.

    Chroma (a', b') is reduced per sample via a vectorized binary search.
    Lightness J' is kept fixed.

    Parameters
    ----------
    jab:
        Array of shape (n, 3) in CAM02-UCS (j', a', b').
    iters:
        Number of binary-search iterations.

    Returns
    -------
    rgb:
        Array of shape (n, 3) in sRGB1.
    gamut_scale:
        Array of shape (n,) with scaling factors in [0, 1] applied to (a', b').

    Raises
    ------
    ValueError
        If `jab` does not have shape (n, 3).

    Notes
    -----
    This function keeps j' fixed and only scales (a', b') towards zero until
    the corresponding sRGB values lie within [0, 1]. A neutral-chroma fallback
    is applied for rare numerical edge cases.
    """
    jab = np.asarray(jab, dtype=np.float64)
    if jab.ndim != 2 or jab.shape[1] != 3:
        raise ValueError("jab must have shape (n, 3)")

    target_ab = jab[:, 1:].copy()
    lo = np.zeros(jab.shape[0], dtype=np.float64)
    hi = np.ones(jab.shape[0], dtype=np.float64)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        jab_try = jab.copy()
        jab_try[:, 1:] = mid[:, None] * target_ab
        rgb_try = cspace_convert(jab_try, "CAM02-UCS", "sRGB1")
        ok = _in_gamut(rgb_try)
        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    jab_best = jab.copy()
    jab_best[:, 1:] = lo[:, None] * target_ab
    rgb = cspace_convert(jab_best, "CAM02-UCS", "sRGB1")

    bad_mask = np.logical_not(_in_gamut(rgb))
    if np.any(bad_mask):
        jab_fallback = jab[bad_mask].copy()
        jab_fallback[:, 1:] = 0.0
        rgb[bad_mask] = np.clip(
            cspace_convert(jab_fallback, "CAM02-UCS", "sRGB1"),
            0.0,
            1.0,
        )

    return rgb, lo


def _allocate_segment_counts(
    distances: NDArray64,
    n: int,
) -> NDArray[np.int64]:
    """Allocate per-segment sample counts that sum to `n`.

    Each segment receives at least 2 samples. Counts are proportional to the
    provided segment distances and then adjusted so that sum(counts) == n.

    Parameters
    ----------
    distances:
        1D array of non-negative segment distances with shape (m,).
    n:
        Total number of samples across all segments.

    Returns
    -------
    counts:
        Integer array of shape (m,) with sum(counts) == n and counts[i] >= 2
        for all i (unless m == 0, in which case an empty array is returned).

    Raises
    ------
    ValueError
        If `distances` is not 1D.
    ValueError
        If `n` is smaller than 2.

    Notes
    -----
    - If the total distance sum is non-positive, counts are distributed
      approximately uniformly (still enforcing the >=2 constraint).
    - The final correction step forces exact equality sum(counts) == n.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 1:
        raise ValueError("distances must be 1D")
    if n < 2:
        raise ValueError("n must be >= 2")

    m = distances.size
    if m == 0:
        return np.zeros(0, dtype=np.int64)

    dsum = float(np.sum(distances))
    if dsum <= 0.0:
        counts = np.full(m, max(2, n // m), dtype=np.int64)
    else:
        frac = distances / dsum
        counts = np.maximum(2, np.round(frac * n).astype(np.int64))

    diff = int(np.sum(counts) - n)
    if diff > 0:
        order = np.argsort(-counts)
        k = 0
        while diff > 0 and k < order.size:
            i = int(order[k])
            if counts[i] > 2:
                counts[i] -= 1
                diff -= 1
            else:
                k += 1
    elif diff < 0:
        order = np.argsort(-distances)
        k = 0
        while diff < 0:
            i = int(order[k % order.size])
            counts[i] += 1
            diff += 1
            k += 1

    if int(np.sum(counts)) != n:
        counts[0] += n - int(np.sum(counts))

    return counts


def _as_rgba(c: str | NDArray64) -> tuple[float, float, float, float]:
    """Convert a color specification to an RGBA tuple in [0, 1].

    Parameters
    ----------
    c:
        Either a Matplotlib color specification string (passed to `to_rgba`)
        or an array-like with at least 3 components interpreted as RGB. If a
        fourth component is present, it is interpreted as alpha.

    Returns
    -------
    rgba:
        Tuple (r, g, b, a) with floats clipped to [0, 1]. If no alpha is
        provided, alpha defaults to 1.

    Raises
    ------
    ValueError
        If `c` is array-like but has fewer than 3 components.
    """
    if isinstance(c, str):
        r, g, b, a = to_rgba(c)
        return (float(r), float(g), float(b), float(a))
    arr = np.asarray(c, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        raise ValueError("color must have at least 3 components")
    if arr.size == 3:
        r, g, b = arr
        a = 1.0
    else:
        r, g, b, a = arr[:4]
    r, g, b, a = np.clip([r, g, b, a], 0.0, 1.0)
    return (float(r), float(g), float(b), float(a))


# endregion


def get_color(
    palette: str = "seeblau",
    shade: int | str = "100",
    permutation: str | Sequence[int] = "standard",
    alpha: float | None = None,
    fake_alpha: float | None = None,
    background: NDArray64 = np.array([1, 1, 1], dtype=np.float64),
) -> NDArray64:
    """Return a single RGB swatch from a named palette.

    Parameters
    ----------
    palette:
        Palette key.
    shade:
        Either an index 0..4 (0=100, 4=20) or one of {"100", "80", "65",
        "35", "20"}.
    permutation:
        Channel permutation applied to the returned swatch.

    Returns
    -------
    rgb:
        Array of shape (3,) with values in [0, 1].

    Raises
    ------
    KeyError
        If `palette` is unknown.
    ValueError
        If `shade` is not recognized.
    IndexError
        If `shade` is outside [0, 4].
    KeyError
        If a string `permutation` key is unknown.
    ValueError
        If an explicit permutation does not have length 3.
    """
    pal = _get_palette(palette)

    if isinstance(shade, str):
        s = shade.strip()
        if s not in SHADE_TO_INDEX:
            raise ValueError(
                f"shade must be one of {sorted(SHADE_TO_INDEX)},",
                " or an int [0,4]",
            )
        i = SHADE_TO_INDEX[s]
    else:
        i = int(shade)

    if i < 0 or i > 4:
        raise IndexError("shade index must be in [0, 4]")

    rgb = _permuta_rgb(pal[i], permutation)

    # Alpha logic (RGBA always)
    a = 1.0 if alpha is None else float(alpha)
    a = float(np.clip(a, 0.0, 1.0))

    if fake_alpha is not None:
        fa = float(np.clip(float(fake_alpha), 0.0, 1.0))
        bg = np.asarray(background, dtype=np.float64).reshape(-1)
        if bg.size < 3:
            raise ValueError("background must have at least 3 components")
        bg3 = np.clip(bg[:3], 0.0, 1.0)

        rgb = (1.0 - fa) * bg3 + fa * rgb
        a = 1.0  # baked-in transparency -> return opaque

    return np.asarray([rgb[0], rgb[1], rgb[2], a], dtype=np.float64)


def get_colors(
    index: int = 0,
    palettes: str | Sequence[str] = "all",
    shades: str | Sequence[int | str] = "all",
    permutations: str | Sequence[str | tuple[int, int, int]] = "standard",
    alpha: float | None = None,
    fake_alpha: float | None = None,
    background: NDArray64 = weiss,
) -> NDArray64:
    """Return a single RGBA color from concatenated named palettes.

    Parameters
    ----------
    index:
        Index into the constructed palette. Indexing wraps.
    palettes:
        "all", a palette name, or a list of palette names.
    shades:
        "all" or an explicit ordered list of shade selectors.
        You may pass indices 0..4 and/or strings {"100","80","65","35","20"}.
        The order is respected.
    permutations:
        Either a single permutation applied to all palettes, a single channel
        permutation (e.g. (2, 0, 1)) applied to all, or a per-palette list.
    alpha:
        True alpha channel in [0, 1]. If None, defaults to 1.
    fake_alpha:
        If set, pre-blend RGB into `background` by `fake_alpha` and return an
        opaque RGBA (alpha=1).
    background:
        Background RGB used for `fake_alpha`.

    Returns
    -------
    rgba:
        Array of shape (4,) in [0, 1].
    """

    # Resolve palette list
    if isinstance(palettes, str):
        pal_key = palettes.strip().lower()
        if pal_key == "all":
            pal_list = list(PALETTES.keys())
        else:
            pal_list = [pal_key]
    else:
        pal_list = [str(p).strip().lower() for p in palettes]

    if len(pal_list) == 0:
        raise ValueError("palettes resolves to an empty list")

    # Resolve permutations: scalar / channel perm / per-palette.
    if isinstance(permutations, str):
        perm_list = [permutations] * len(pal_list)
    else:
        try:
            perm_seq = list(permutations)
        except TypeError as exc:
            raise TypeError(
                "permutations must be a str, a (3,) channel perm,",
                "or a sequence",
            ) from exc

        is_chan_perm = (
            len(perm_seq) == 3
            and all(isinstance(x, (int, np.integer)) for x in perm_seq)
            and set(int(x) for x in perm_seq) == {0, 1, 2}
        )

        if is_chan_perm:
            perm_list = [tuple(int(x) for x in perm_seq)] * len(pal_list)
        else:
            perm_list = perm_seq
            if len(perm_list) != len(pal_list):
                raise ValueError(
                    "permutations must be a scalar or match palettes length"
                )

    # Resolve shade indices (order preserved)
    if isinstance(shades, str):
        shade_key = shades.strip().lower()
        if shade_key != "all":
            raise ValueError(
                "shades must be 'all' or a sequence of indices/labels",
            )
        shade_idx = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    else:
        idx_list: list[int] = []
        for s in shades:
            if isinstance(s, str):
                ss = s.strip()
                if ss not in SHADE_TO_INDEX:
                    raise ValueError(
                        f"Unknown shade '{s}'.",
                        f"Use one of {sorted(SHADE_TO_INDEX)} or 0..4",
                    )
                idx_list.append(int(SHADE_TO_INDEX[ss]))
            else:
                idx_list.append(int(s))
        shade_idx = np.asarray(idx_list, dtype=np.int64)

    if np.any((shade_idx < 0) | (shade_idx > 4)):
        raise IndexError("shade indices must be in [0, 4]")

    # Build concatenated palette (keep duplicates)
    blocks: list[NDArray64] = []
    for pal_name, perm in zip(pal_list, perm_list):
        pal = _get_palette(pal_name)  # (5,3)
        sel = pal[shade_idx]  # (k,3)
        sel = _permuta_rgb(sel, perm)  # (k,3)
        blocks.append(np.asarray(sel, dtype=np.float64))

    palette_rgb = np.concatenate(blocks, axis=0)
    if palette_rgb.shape[0] == 0:
        raise ValueError("constructed palette is empty")

    rgb = palette_rgb[index % palette_rgb.shape[0]]

    # Alpha logic (RGBA always)
    a = 1.0 if alpha is None else float(alpha)
    a = float(np.clip(a, 0.0, 1.0))

    if fake_alpha is not None:
        fa = float(np.clip(float(fake_alpha), 0.0, 1.0))
        bg = np.asarray(background, dtype=np.float64).reshape(-1)
        if bg.size < 3:
            raise ValueError("background must have at least 3 components")
        bg3 = np.clip(bg[:3], 0.0, 1.0)

        rgb = (1.0 - fa) * bg3 + fa * rgb
        a = 1.0  # baked-in transparency -> return opaque

    return np.asarray([rgb[0], rgb[1], rgb[2], a], dtype=np.float64)


def cmap_tinted_black_to_white(
    base_rgb: NDArray64,
    n: int = 256,
    strength: float = 0.35,
    gamma: float = 1.0,
    alpha: float = 1.0,
    name: str = "black_to_white",
    return_debug: bool = False,
) -> ListedColormap | tuple[ListedColormap, Dict[str, NDArray64]]:
    """Black-to-white colormap with a symmetric CAM02-UCS tint.

    The colormap is constructed to be perceptually linear in CAM02-UCS
    lightness j' (optionally shaped by `gamma`). A color tint derived
    from `base_rgb` is applied with an envelope f(t)=4 t (1-t), so the
    tint vanishes at black and white.

    Parameters
    ----------
    base_rgb:
        Tint reference color in sRGB1, shape (3,).
    n:
        Number of samples in the colormap.
    strength:
        Tint strength multiplier.
    gamma:
        Lightness shaping exponent. gamma=1 gives linear J'.
    alpha:
        Constant alpha channel in [0, 1].
    name:
        Colormap name.
    return_debug:
        If True, return (cmap, debug_dict).

    Returns
    -------
    cmap:
        Matplotlib ListedColormap with shape (n, 4).
    debug:
        Only returned if `return_debug` is True. Contains intermediate arrays
        such as t, J', the envelope f, the per-sample gamut scaling, and the
        effective tint strength.

    Notes
    -----
    Gamut safety is enforced by reducing chroma (a', b') per sample until the
    corresponding sRGB values lie within [0, 1]. This can slightly reduce tint
    strength near high lightness where the sRGB gamut is tight.
    """

    base_rgb: NDArray64 = np.clip(base_rgb, 0, 1)

    # CAM02-UCS Setup: default viewing conditions (sRGB/D65, average surround)
    # colorspacious uses a dict for jCh-like spaces; for UCS it's simpler:
    # "sRGB1" <-> "CAM02-UCS"
    base_jab = cspace_convert(base_rgb[None, :], "sRGB1", "CAM02-UCS")[0]
    _, a0, b0 = base_jab

    t = np.linspace(0.0, 1.0, n)

    # perzeptuelle Lightness: linear in j' (optional gamma shaping)
    j = 100.0 * np.clip(t**gamma, 0, 1)

    # Hülle für den Farbstich (0 an den Enden)
    f = 4.0 * t * (1.0 - t)
    # quasi sowas wie 1-x^2

    # Ziel-Parameter in CAM02-UCS
    jab = np.zeros((n, 3), dtype=float)
    jab[:, 0] = j
    jab[:, 1] = strength * f * a0
    jab[:, 2] = strength * f * b0

    # Gamut-sicher machen: pro Sample Chroma runterregeln, bis RGB gültig
    target_ab = jab[:, 1:].copy()  # shape (n, 2)

    lo = np.zeros(n, dtype=np.float64)
    hi = np.ones(n, dtype=np.float64)

    # Vectorized binary search on the scaling of (a', b')
    for _ in range(18):  # ~2^-18 precision is sufficient
        mid = 0.5 * (lo + hi)  # shape (n,)

        jab_try = jab.copy()
        jab_try[:, 1:] = mid[:, None] * target_ab

        rgb_try = cspace_convert(jab_try, "CAM02-UCS", "sRGB1")
        ok = _in_gamut(rgb_try)  # shape (n,)

        lo = np.where(ok, mid, lo)
        hi = np.where(ok, hi, mid)

    # Final RGB at the largest in-gamut scaling found
    jab_best = jab.copy()
    jab_best[:, 1:] = lo[:, None] * target_ab
    rgb = cspace_convert(jab_best, "CAM02-UCS", "sRGB1")

    # Fallback for rare numerical edge cases: force neutral chroma
    bad_mask = np.logical_not(_in_gamut(rgb))

    # Diagnostics: effective strength envelope and gamut scaling.
    strength_eff = strength * lo * f

    if np.any(bad_mask):
        jab_fallback = jab[bad_mask].copy()
        jab_fallback[:, 1:] = 0.0
        rgb[bad_mask] = np.clip(
            cspace_convert(jab_fallback, "CAM02-UCS", "sRGB1"), 0.0, 1.0
        )
    rgba = np.concatenate([rgb, np.full((n, 1), float(alpha))], axis=1)
    cmap = ListedColormap(rgba, name=name)

    if not return_debug:
        return cmap

    debug = {
        "t": t,
        "j": j,
        "f": f,
        "gamut_scale": lo,
        "strength_eff": strength_eff,
        "a0": float(a0),
        "b0": float(b0),
        "base_jab": base_jab,
    }
    return cmap, debug


def get_cmap(
    palette: str = "seeblau",
    shade: int | str = "100",
    permutation: str | Sequence[int] = "standard",
    n: int = 256,
    strength: float = 1.0,
    gamma: float = 1.0,
    alpha: float = 1.0,
    name: str | None = None,
    under: str | NDArray64 | None = None,
    over: str | NDArray64 | None = None,
    bad: str | NDArray64 | None = None,
) -> ListedColormap:
    """Convenience wrapper for `cmap_tinted_black_to_white` using a palette.

    Parameters
    ----------
    palette:
        Palette key.
    shade:
        Shade selector for the swatch. Either an index 0..4 (0=100, 4=20) or
        one of {"100", "80", "65", "35", "20"}.
    permutation:
        Channel permutation applied to the swatch. Either a key in
        `PERMUTATIONS` (e.g. "standard", "magenta") or an explicit permutation
        such as (2, 0, 1).
    n:
        Number of colormap samples.
    strength:
        Tint strength multiplier passed to `cmap_tinted_black_to_white`.
    gamma:
        Lightness shaping exponent passed to `cmap_tinted_black_to_white`.
    alpha:
        Constant alpha channel passed to `cmap_tinted_black_to_white`.
    name:
        Optional colormap name. If None, a name is constructed from inputs.
    under, over, bad:
        Colors used for out-of-range and invalid values.
        If None, defaults are used:
        - under: first colormap color (min)
        - over: last colormap color (max)
        - bad: `rot`

    Returns
    -------
    cmap:
        Matplotlib ListedColormap.
    """
    base_rgb = get_color(palette, shade, permutation)
    if name is None:
        name = f"cmap_{palette}_{shade}_s{strength:g}_g{gamma:g}"
    cmap = cmap_tinted_black_to_white(
        base_rgb,
        n=n,
        strength=strength,
        gamma=gamma,
        alpha=alpha,
        name=name,
        return_debug=False,
    )

    cols = np.asarray(cmap.colors, dtype=np.float64)
    under_rgba = _as_rgba(under) if under is not None else tuple(cols[0])
    over_rgba = _as_rgba(over) if over is not None else tuple(cols[-1])
    bad_rgba = _as_rgba(bad) if bad is not None else "red"

    cmap.set_under(under_rgba)
    cmap.set_over(over_rgba)
    cmap.set_bad(bad_rgba)
    return cmap


def cmap_tinted_grey(
    rgb_points: Sequence[NDArray64] = (seeblau100, peach100),
    n: int = 256,
    j: float | None = None,
    alpha: float = 1.0,
    enforce_gamut: bool = True,
    name: str = "grey",
    return_debug: bool = False,
) -> ListedColormap | tuple[ListedColormap, Dict[str, NDArray64]]:
    """Interpolate between RGB colors at constant CAM02-UCS lightness J'.

    The input colors are converted from sRGB1 to CAM02-UCS. A piecewise-linear
    path is constructed in the (a', b') plane between the input points while
    keeping J' fixed. The resulting CAM02-UCS path is then converted back to
    sRGB1 to obtain a colormap.

    Parameters
    ----------
    rgb_points:
        Sequence of colors in sRGB1. Interpreted as an array of shape (m, 3)
        with values in [0, 1]. At least two points are required.
    n:
        Total number of samples in the colormap.
    j:
        Target constant lightness J'. If None, uses the mean J' of the input
        points (after conversion to CAM02-UCS).
    alpha:
        Constant alpha channel in [0, 1].
    enforce_gamut:
        If True, reduce chroma (a', b') per sample so that the converted sRGB
        values lie within [0, 1].
    name:
        Colormap name.
    return_debug:
        If True, return (cmap, debug_dict).

    Returns
    -------
    cmap:
        Matplotlib ListedColormap with `n` samples and RGBA entries.
    debug:
        Only returned if `return_debug` is True. Contains intermediate arrays:
        - jab_points: CAM02-UCS coordinates of the input points
        - jab_path: CAM02-UCS coordinates of the interpolated path
        - j_target: scalar target J' as an array of shape (1,)
        - gamut_scale: per-sample chroma scaling factors in [0, 1]
        - counts: per-segment sample counts (float array for convenience)

    Raises
    ------
    ValueError
        If `rgb_points` cannot be interpreted as shape (m, 3).
    ValueError
        If fewer than two points are provided.

    Notes
    -----
    Endpoints are projected onto the constant-j' plane in CAM02-UCS. As a
    consequence, the resulting RGB endpoints generally do not exactly match
    the input colors unless they already share the target j'.
    """

    pts = np.asarray(rgb_points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("rgb_points must have shape (m, 3)")
    if pts.shape[0] < 2:
        raise ValueError("rgb_points must contain at least 2 colors")

    pts = np.clip(pts, 0.0, 1.0)
    jab_pts = cspace_convert(pts, "sRGB1", "CAM02-UCS")

    j0 = float(np.mean(jab_pts[:, 0])) if j is None else float(j)
    ab = jab_pts[:, 1:]

    seg = ab[1:] - ab[:-1]
    dist = np.sqrt(np.sum(seg**2, axis=1))
    counts = _allocate_segment_counts(dist, n)

    # piecewise-linear ab path
    ab_path = []
    for i in range(ab.shape[0] - 1):
        k = int(counts[i])
        u = np.linspace(0.0, 1.0, k, dtype=np.float64)
        seg_ab = (1.0 - u)[:, None] * ab[i] + u[:, None] * ab[i + 1]
        if i > 0:
            seg_ab = seg_ab[1:]  # avoid duplicate knot
        ab_path.append(seg_ab)

    ab_full = np.concatenate(ab_path, axis=0)
    jab = np.zeros((ab_full.shape[0], 3), dtype=np.float64)
    jab[:, 0] = j0
    jab[:, 1:] = ab_full

    if enforce_gamut:
        rgb, gamut_scale = _gamut_limit_cam02ucs_jab(jab)
    else:
        rgb = np.clip(cspace_convert(jab, "CAM02-UCS", "sRGB1"), 0.0, 1.0)
        gamut_scale = np.ones(rgb.shape[0], dtype=np.float64)

    rgba = np.concatenate(
        [rgb, np.full((rgb.shape[0], 1), float(alpha))],
        axis=1,
    )
    rgba = ListedColormap(rgba, name=name)

    if not return_debug:
        return rgba

    debug = {
        "jab_points": jab_pts,
        "jab_path": jab,
        "j_target": np.array([j0], dtype=np.float64),
        "gamut_scale": gamut_scale,
        "counts": counts.astype(np.float64),
    }
    return rgba, debug


def get_cmap_multi(
    palettes: Sequence[str],
    shades: Sequence[int | str] | int | str = "100",
    permutation: Sequence[int | str] | int | str = "standard",
    n: int = 256,
    j: float | None = None,
    alpha: float = 1.0,
    enforce_gamut: bool = True,
    name: str | None = None,
    under: str | NDArray64 | None = None,
    over: str | NDArray64 | None = None,
    bad: str | NDArray64 | None = None,
) -> ListedColormap:
    """Build a constant-lightness (constant j') colormap from palette names.

    Parameters
    ----------
    palettes:
        Sequence of palette names, e.g. ["seeblau", "seegrau", "bordeaux"].
    shades:
        Either a single shade (e.g. "100" or 0..4) applied to all palettes,
        or a sequence of the same length as `palettes`.
    n:
        Number of colormap samples.
    j:
        Target constant lightness j'. If None, uses the mean j' of the selected
        swatches.
    alpha:
        Constant alpha channel.
    enforce_gamut:
        If True, locally reduce chroma so the path remains within sRGB.
    name:
        Optional colormap name.
    under, over, bad:
        Colors used for out-of-range and invalid values.
        If None, defaults are used:
        - under: first colormap color (min)
        - over: last colormap color (max)
        - bad: `rot`

    Notes
    -----
    Endpoints are projected onto constant j' in CAM02-UCS, so the resulting RGB
    endpoints generally do not exactly match the selected swatches.
    """
    pal_list = list(palettes)
    if len(pal_list) < 2:
        raise ValueError("palettes must contain at least 2 entries")

    if isinstance(shades, (str, int)):
        shade_list = [shades] * len(pal_list)
    else:
        shade_list = list(shades)
        if len(shade_list) != len(pal_list):
            raise ValueError(
                "shades must be a scalar or the same length as palettes.",
            )

    if isinstance(permutation, (str, int)):
        perm_list = [permutation] * len(pal_list)
    else:
        perm_list = list(permutation)
        if len(perm_list) != len(pal_list):
            raise ValueError(
                "perm.s must be a scalar or the same length as palettes.",
            )

    rgb_points = np.stack(
        [
            get_color(pal, shade, perm)
            for pal, shade, perm in zip(
                pal_list,
                shade_list,
                perm_list,
            )
        ],
        axis=0,
    ).astype(np.float64)

    if name is None:
        parts = [
            f"{pal}:{shade}:{perm}"
            for pal, shade, perm in zip(
                pal_list,
                shade_list,
                perm_list,
            )
        ]
        name = "constj_" + "_".join(parts)

    cmap = cmap_tinted_grey(
        rgb_points=rgb_points,
        n=n,
        j=j,
        alpha=alpha,
        enforce_gamut=enforce_gamut,
        name=name,
        return_debug=False,
    )

    cols = np.asarray(cmap.colors, dtype=np.float64)
    under_rgba = _as_rgba(under) if under is not None else tuple(cols[0])
    over_rgba = _as_rgba(over) if over is not None else tuple(cols[-1])
    bad_rgba = _as_rgba(bad) if bad is not None else "red"

    cmap.set_under(under_rgba)
    cmap.set_over(over_rgba)
    cmap.set_bad(bad_rgba)
    return cmap
