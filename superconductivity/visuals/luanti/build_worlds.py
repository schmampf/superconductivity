"""Luanti/Minetest world builder and deploy helpers.

This module builds Luanti worlds from exported datasets located in
`./datasets/`. Each dataset directory is expected to contain at
least:

- `map.u16le` : uint16 little-endian heightmap (row-major)
- `map.json`  : metadata sidecar (may include a `palette` filename)

The builder creates one world per dataset under `<dataset>/worlds/` based
on the template file `world.mt.template`.

In addition, each generated world is copied into `<dataset>/worlds/` (where
`dataset` is the `dataset` argument interpreted relative to the current
working directory). This makes the dataset folder self-contained for
shipping.

If `map.json` references a palette texture (e.g. `cmap.png`), it is shipped as
a per-world texture-only worldmod `measurement_palette` so each world can use
its own colormap without modifying global mods.
"""

import fnmatch
import json
import shutil
import subprocess
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "world.mt.template"


DEFAULT_REMOTE_WORLDS_DIR = (
    Path.home() / "Library" / "Application Support" / "minetest" / "worlds"
)


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


def clean_worlds(
    matching_rule: str = "my ",
    *,
    worlds_dir: str | Path | None = None,
    dry_run: bool = False,
) -> List[Path]:
    """Delete remote Luanti/Minetest worlds matching a rule.

    The match rule is applied to the world directory name (basename). If
    `matching_rule` contains glob characters (`*`, `?`, `[`), it is treated as a
    glob. Otherwise it is treated as a prefix.

    Parameters
    ----------
    matching_rule
        Prefix or glob used to select worlds for deletion. Default is "my ".
    worlds_dir
        Directory containing worlds. If None, uses the default macOS path:
        `~/Library/Application Support/minetest/worlds`.
    dry_run
        If True, do not delete anything; just return the would-be targets.

    Returns
    -------
    deleted
        List of world directories that were deleted (or would be deleted when
        `dry_run=True`).

    Raises
    ------
    FileNotFoundError
        If the worlds directory does not exist.
    RuntimeError
        If Luanti/Minetest appears to be running.
    """
    wdir = Path(worlds_dir) if worlds_dir is not None else DEFAULT_REMOTE_WORLDS_DIR
    if not wdir.exists():
        raise FileNotFoundError(f"Worlds directory not found: {wdir}")

    # Safety: refuse to delete while the game is running.
    if (
        subprocess.run(
            ["pgrep", "-x", "luanti"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
        or subprocess.run(
            ["pgrep", "-x", "minetest"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    ):
        raise RuntimeError("Luanti/Minetest is running. Quit it first.")

    rule = matching_rule
    is_glob = any(ch in rule for ch in ("*", "?", "["))

    targets: List[Path] = []
    for p in sorted(wdir.iterdir()):
        if not p.is_dir():
            continue
        name = p.name
        if is_glob:
            if fnmatch.fnmatch(name, rule):
                targets.append(p)
        else:
            if name.startswith(rule):
                targets.append(p)

    if dry_run:
        return targets

    for p in targets:
        shutil.rmtree(p)

    return targets


def build_worlds(
    prefix: str = "",
    dataset: str = "",
    *,
    remove_existing_worlds: bool = True,
    deploy: bool = True,
    deploy_script: str | Path = "deploy.sh",
) -> List[str]:
    """Build one Luanti world per dataset and optionally deploy them.

    Worlds are created under `<dataset>/worlds/` with names derived from
    the dataset directory names under `./datasets/`.

    In addition, each generated world is copied into `<dataset>/worlds/` (where
    `dataset` is the `dataset` argument interpreted relative to the current
    working directory). This makes the dataset folder self-contained for
    shipping.

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

    # Datasets are discovered relative to the current working directory.
    # This matches `export_dataset(..., out_dir=".")` usage.
    datasets_dir = Path.cwd() / dataset / "datasets"
    if not datasets_dir.exists():
        raise FileNotFoundError(f"Missing datasets directory: {datasets_dir}")

    # Additionally export the built worlds into the dataset base folder so the
    # dataset directory is self-contained for shipping.
    export_worlds_dir = Path.cwd() / dataset / "worlds"
    export_worlds_dir.mkdir(parents=True, exist_ok=True)

    # Build worlds directly into the execution directory (dataset base).
    worlds_dir = export_worlds_dir

    for ds_dir in sorted(datasets_dir.iterdir()):
        if not ds_dir.is_dir():
            continue

        u16 = ds_dir / "map.u16le"
        js = ds_dir / "map.json"

        if not (u16.exists() and js.exists()):
            continue

        wname = prefix + slug(ds_dir.name)
        wdir = worlds_dir / wname
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
            run_deploy_script(
                deploy_script,
                prefix=prefix,
                dataset_dir=dataset,
            )
        except subprocess.CalledProcessError:
            pass

    return built


def run_deploy_script(
    script: str | Path = "deploy.sh",
    *,
    prefix: str = "my ",
    dataset_dir: str = "",
) -> None:
    """Run the deploy shell script.

    Parameters
    ----------
    script
        Path to the deploy script (relative to this module directory or
        absolute).
    prefix
        Prefix passed to the deploy script via `--prefix`.
    dataset_dir
        Dataset directory relative to the current working directory. This is
        forwarded to deploy.sh via `--dataset-dir` so it can find
        `dataset_dir/worlds`.

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

    cmd = ["bash", str(script_path), "--prefix", prefix]
    if dataset_dir:
        cmd += ["--dataset-dir", dataset_dir]
    else:
        # Worlds are expected in ./worlds when no dataset dir is used.
        cmd += ["--src-worlds-dir", str(Path.cwd() / "worlds")]

    subprocess.run(cmd, check=True)
