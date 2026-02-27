"""Luanti/Minetest world builder and deploy helpers.

This module builds Luanti worlds from exported datasets located in
`<this module>/datasets/`. Each dataset directory is expected to contain at
least:

- `map.u16le` : uint16 little-endian heightmap (row-major)
- `map.json`  : metadata sidecar (may include a `palette` filename)

The builder creates one world per dataset under `<this module>/worlds/` based
on the template file `worlds/_template/world.mt.template`.

If `map.json` references a palette texture (e.g. `cmap.png`), it is shipped as
a per-world texture-only worldmod `measurement_palette` so each world can use
its own colormap without modifying global mods.
"""

import json
import shutil
import subprocess
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
WORLDS = ROOT / "worlds"
TEMPLATE = WORLDS / "_template" / "world.mt.template"


def slug(name: str) -> str:
    """Sanitize a dataset name into a filesystem-friendly slug.

    Parameters
    ----------
    name
        Input name to sanitize.

    Returns
    -------
    slug
        Name where any character not in `[A-Za-z0-9_-]` is replaced by `_`.
    """
    return "".join(c if (c.isalnum() or c in "_-") else "_" for c in name)


def build_worlds(
    prefix: str = "",
    *,
    remove_existing_worlds: bool = True,
    deploy: bool = True,
    deploy_script: str | Path = "deploy.sh",
) -> List[str]:
    """Build one Luanti world per dataset and optionally deploy them.

    Worlds are created under `<this module>/worlds/` with names derived from
    the dataset directory names under `<this module>/datasets/`.

    Parameters
    ----------
    prefix
        Prefix prepended to each generated world name.
    remove_existing_worlds
        If True, delete an existing local world directory before rebuilding.
        If False, keep existing worlds and skip rebuilding them.
    deploy
        If True, run the deploy script after building.
    deploy_script
        Path to the deploy script (relative to this module directory or
        absolute).

    Returns
    -------
    built
        List of generated world names.

    Raises
    ------
    FileNotFoundError
        If the world template file is missing.
    """
    if not TEMPLATE.exists():
        raise FileNotFoundError(f"Missing template: {TEMPLATE}")

    world_mt = TEMPLATE.read_text()
    built: List[str] = []

    for ds_dir in sorted(DATASETS.iterdir()):
        if not ds_dir.is_dir():
            continue

        u16 = ds_dir / "map.u16le"
        js = ds_dir / "map.json"

        if not (u16.exists() and js.exists()):
            continue

        wname = prefix + slug(ds_dir.name)
        wdir = WORLDS / wname
        measdir = wdir

        if wdir.exists():
            if remove_existing_worlds:
                shutil.rmtree(wdir)
            else:
                print(
                    "skipping existing world (remove_existing_worlds=False):",
                    wname,
                )
                continue

        measdir.mkdir(parents=True, exist_ok=True)
        (wdir / "world.mt").write_text(world_mt)

        shutil.copy2(u16, measdir / "map.u16le")
        shutil.copy2(js, measdir / "map.json")

        # If the dataset defines a palette texture, ship it as a per-world mod
        # each world can have its own colormap without touching global mods.
        meta = json.loads(js.read_text())

        palette = meta.get("palette")
        if palette:
            pal_src = ds_dir / palette
            if pal_src.exists():
                # Ship palette as a dedicated texture-only worldmod.
                # This avoids shadowing the real `measurement_terrain` mod.
                pal_mod = wdir / "worldmods" / "measurement_palette"
                pal_tex = pal_mod / "textures"
                pal_tex.mkdir(parents=True, exist_ok=True)

                (pal_mod / "init.lua").write_text(
                    "-- Texture-only worldmod for per-world palette shipping.\n"
                )
                (pal_mod / "mod.conf").write_text(
                    "name = measurement_palette\n"
                    "description = Per-world palette texture for measurement_terrain\n"
                )

                shutil.copy2(pal_src, pal_tex / palette)
            else:
                print(f"warning: palette referenced but missing: {pal_src}")

        print(f"built world: {wname}")
        built.append(wname)

    if deploy:
        try:
            run_deploy_script(deploy_script, prefix=prefix)
        except subprocess.CalledProcessError:
            pass

    return built


def run_deploy_script(
    script: str | Path = "deploy.sh",
    *,
    prefix: str = "my ",
) -> None:
    """Run the deploy shell script.

    Parameters
    ----------
    script
        Path to the deploy script (relative to this module directory or
        absolute).
    prefix
        Prefix passed to the deploy script via `--prefix`.

    Raises
    ------
    FileNotFoundError
        If the deploy script does not exist.
    subprocess.CalledProcessError
        If the deploy script exits with a non-zero status.
    """
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = ROOT / script_path
    script_path = script_path.resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Deploy script not found: {script_path}")

    subprocess.run(["bash", str(script_path), "--prefix", prefix], check=True)
