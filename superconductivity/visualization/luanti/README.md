

# Luanti (Minetest) visualization

This subpackage turns 2D scalar fields (NumPy arrays) into Luanti/Minetest
worlds. The resulting worlds render your data as a colored heightmap terrain.
Each dataset can ship its own palette (colormap), so different worlds can use
different colormaps without touching global game mods.

Core use cases:

- Export a NumPy array to `map.u16le` + `map.json` (+ optional palette PNG).
- Build one Luanti world per exported dataset.
- Deploy worlds/mods into the local Luanti/Minetest directory (macOS).
- Clean remote worlds by prefix/glob.

The canonical workflow is shown in `sim visual.ipynb`.

## What gets generated

For each exported dataset `title` under a dataset base directory `dataset`:

- `./<dataset>/datasets/<title>/map.u16le`
  uint16 little-endian, row-major (Ny×Nx) heightmap.

- `./<dataset>/datasets/<title>/map.json`
  Metadata (shape, scaling ranges, palette filename, …).

- `./<dataset>/datasets/<title>/<palette>.png` (optional)
  256-color palette texture (e.g. `cmap.png`), referenced in `map.json`.

When building worlds:

- `./<dataset>/worlds/<worldname>/`
  A Luanti world directory per dataset. `map.u16le` and `map.json` are copied
  into the world root. If `map.json` references a palette PNG, it is shipped as
  a per-world worldmod:

  `./<dataset>/worlds/<worldname>/worldmods/measurement_palette/textures/<png>`

This matters: Luanti resolves `palette = "..."` as a texture name from loaded
mods/texture packs, not as a filesystem path. A worldmod provides the palette
texture in a per-world way.

## Requirements

- Luanti/Minetest installed locally.
- On macOS, the deploy target is:

  `~/Library/Application Support/minetest/`

- Python dependencies used by export:

  - NumPy
  - Matplotlib
  - Pillow (PIL)

## Public API

You typically import from the package root:

```python
from superconductivity.visualization.luanti import (
    export_dataset,
    build_worlds,
    clean_worlds,
)
```

## Exporting datasets

### `export_dataset(...)`

Exports a 2D array `z` to `map.u16le` + `map.json`, and (optionally) a palette
PNG into `./<dataset>/datasets/<title>/`.

Typical usage (mirrors the notebook pattern where `dataset="visuals"`):

```python
export_dataset(
    z=dG_err_G0,
    title="0 V_off",
    dataset="visuals",
    zlim=(0.5, 1.5),          # scaling knob, not a hard clip
    colormap="viridis",       # or a ListedColormap
    palette_name="cmap",      # writes cmap.png
    hmin=0,
    hmax=80,
    xlen=200,
    ylen=200,
)
```

Notes:

- `zlim=(z_min, z_max)` defines the scaling range for mapping `z -> height`.
  Values outside the range are not clipped in the input; they map to heights
  below/above `hmin/hmax` and are clamped only to uint16 bounds.
- `colormap` writes a 256×1 palette PNG and stores the filename in `map.json`.
- `xlen/ylen` remap your input data to a regular grid (nearest-neighbor) before
  exporting. This is useful when your raw data grid is irregular or you want a
  consistent resolution.

## Building worlds

### `build_worlds(...)`

Creates a world per dataset found in `./<dataset>/datasets/*/`.

Notebook-equivalent call:

```python
build_worlds(prefix="TB (13.6GHz) ", dataset="visuals")
```

What it does:

1) Reads `./visuals/datasets/*/map.u16le` and `map.json`.
2) Creates one world per dataset under `./visuals/worlds/<worldname>/`.
3) Copies `map.u16le` and `map.json` into that world root.
4) If `map.json["palette"]` is set, ships it as a per-world worldmod
   `measurement_palette` so the palette can be resolved by Luanti.

Deploy behavior:

- By default, `build_worlds(..., deploy=True)` calls `deploy.sh` after building.
- Deploy refuses to run if Luanti/Minetest is still running.

Build without deploying:

```python
build_worlds(prefix="TB (13.6GHz) ", dataset="visuals", deploy=False)
```

## Cleaning remote worlds

### `clean_worlds(...)`

Deletes worlds in the remote Luanti/Minetest worlds directory based on a rule.

Notebook example:

```python
clean_worlds(matching_rule="test*")
```

Matching rules:

- If the rule contains glob chars (`*`, `?`, `[`), it is treated as a glob.
- Otherwise it is treated as a prefix (e.g. `"my "` deletes worlds starting
  with `"my "`).

Safety:

- Refuses to run if `luanti` or `minetest` appears to be running.

Preview without deleting:

```python
targets = clean_worlds("TB (13.6GHz) *", dry_run=True)
```

## Deploy script behavior

`deploy.sh` copies:

- mods from the script directory’s `mods/` folder
- worlds from `./<dataset>/worlds/` by default (execution directory)

It supports:

- `--prefix` / `--managed-world-prefix` (which remote worlds to remove)
- `--keep-existing-worlds` (do not delete managed worlds before copying)
- `--dataset-dir DIR` (worlds read from `DIR/worlds`)
- `--src-worlds-dir DIR` (explicit worlds directory; overrides `--dataset-dir`)

Typical manual deploy (from the directory that contains `visuals/worlds/`):

```bash
bash deploy.sh --prefix "TB (13.6GHz) " --dataset-dir visuals
```

## Tuning colors and height scaling in-world

The terrain mod (`mods/measurement_terrain/init.lua`) reads:

- `map.u16le` and `map.json` from the world root
- the palette texture name from `meta["palette"]` (or a default)

Color scaling (contrast) can be decoupled from the geometry export:

- Geometry uses the exported heights (from `export_dataset`).
- Color mapping uses `cmap_hmin/cmap_hmax` (if present in `map.json`), otherwise
  falls back to `hmin/hmax`.

This gives you two separate knobs:

- `zlim` in Python: controls how z maps to block heights (geometry scaling).
- `cmap_hmin/cmap_hmax` in JSON: controls how heights map to palette indices
  (visual contrast), without changing the geometry.

## Interaction / “don’t destroy my terrain”

The measurement terrain node is configured as non-diggable to prevent
accidental persistent modifications.

If you want an “editable session but reset on reload” workflow, that requires
actively re-writing mapblocks on join/start (heavier and not the default).

## End-to-end workflow (as in `sim visual.ipynb`)

```python
dataset = "visuals"

export_dataset(z=dG_err_G0, title="0 V_off", dataset=dataset)
export_dataset(z=I_exp_nA,  title="1 IV exp", dataset=dataset)
export_dataset(z=G_exp_G0,  title="2 dIdV exp", dataset=dataset)

# Build + deploy worlds
build_worlds(prefix="TB (13.6GHz) ", dataset=dataset)

# Optional: delete old remote worlds by pattern
clean_worlds(matching_rule="TB (13.6GHz) *")
```