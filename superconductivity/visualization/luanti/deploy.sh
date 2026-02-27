#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_MODS="${SRC_ROOT}/mods"
SRC_WORLDS="${SRC_ROOT}/worlds"

DST_ROOT="${HOME}/Library/Application Support/minetest"
DST_MODS="${DST_ROOT}/mods"
DST_WORLDS="${DST_ROOT}/worlds"

mkdir -p "${DST_MODS}" "${DST_WORLDS}"

# ---- CLI options
# Default behavior matches existing script: remove previously deployed worlds
# that start with the managed prefix.
KEEP_EXISTING_WORLDS=0
MANAGED_WORLD_PREFIX="my "

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-existing-worlds)
      KEEP_EXISTING_WORLDS=1
      shift
      ;;
    --managed-world-prefix|--prefix)
      shift
      if [[ $# -eq 0 ]]; then
        echo "Missing value for --managed-world-prefix" >&2
        exit 2
      fi
      MANAGED_WORLD_PREFIX="$1"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Usage: deploy.sh [OPTIONS]

Options:
  --keep-existing-worlds         Do not delete existing remote worlds (managed prefix).
  --managed-world-prefix PREFIX  Prefix used to identify worlds managed by this script.
  --prefix PREFIX                Alias for --managed-world-prefix.
                                Default: "my "
  -h, --help                     Show this help.
EOF
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
done

# Abort if Luanti/Minetest is running (prevents disk I/O corruption)
if pgrep -x "luanti" >/dev/null || pgrep -x "minetest" >/dev/null; then
  echo "Luanti is running. Quit it before deploying."
  exit 1
fi

# Remove previously deployed worlds (managed prefix) to avoid stale copies.
# This keeps unrelated worlds intact.
if [[ ${KEEP_EXISTING_WORLDS} -eq 0 ]]; then
  if [[ -d "${DST_WORLDS}" ]]; then
    for w in "${DST_WORLDS}/${MANAGED_WORLD_PREFIX}"*; do
      [[ -d "${w}" ]] || continue
      rm -rf "${w}"
    done
  fi
else
  echo "Keeping existing remote worlds (skip deletion for prefix: ${MANAGED_WORLD_PREFIX})"
fi

# Remove previously deployed copies of the mods we are about to ship.
# This avoids stale files if a mod is renamed/removed locally.
if [[ -d "${SRC_MODS}" ]]; then
  for m in "${SRC_MODS}"/*; do
    [[ -d "${m}" ]] || continue
    name="$(basename "${m}")"
    rm -rf "${DST_MODS}/${name}"
  done
fi

# Copy mod(s)
for m in "${SRC_MODS}"/*; do
  [[ -d "${m}" ]] || continue
  name="$(basename "${m}")"
  rm -rf "${DST_MODS}/${name}"
  cp -R "${m}" "${DST_MODS}/${name}"
done

# Copy worlds (skip _template)
for w in "${SRC_WORLDS}"/*; do
  [[ -d "${w}" ]] || continue
  [[ "$(basename "${w}")" == "_template" ]] && continue
  name="$(basename "${w}")"
  rm -rf "${DST_WORLDS}/${name}"
  cp -R "${w}" "${DST_WORLDS}/${name}"
done

echo "Deployed mods and worlds to:"
echo "  ${DST_ROOT}"