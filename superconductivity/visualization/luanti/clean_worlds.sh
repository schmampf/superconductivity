#!/usr/bin/env bash
set -euo pipefail

# Delete worlds in the Luanti/Minetest worlds directory by match rule.
#
# Examples:
#   ./clean_worlds.sh --match 'my *' --dry-run
#   ./clean_worlds.sh --match 'meas__*' --yes
#   ./clean_worlds.sh --regex '^my .*' --yes
#   ./clean_worlds.sh --worlds-dir "/custom/path/worlds" --match 'tmp_*' --yes

WORLDS_DIR="${HOME}/Library/Application Support/minetest/worlds"
MATCH_GLOB=""
REGEX=""
DRY_RUN=0
YES=0

usage() {
  cat <<'EOF'
Usage: clean_worlds.sh [OPTIONS]

Options:
  --worlds-dir DIR     Worlds directory. Default:
                       ~/Library/Application Support/minetest/worlds
  --match GLOB         Glob pattern to match world directory names
                       (e.g. 'my *', 'meas__*').
  --regex REGEX        Regex (bash/grep -E) to match world directory names.
                       If set, --match is ignored.
  --dry-run            Print what would be deleted; do not delete.
  --yes                Do not ask for confirmation.
  -h, --help           Show help.

Notes:
- Exactly one of --match or --regex should be provided.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --worlds-dir)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --worlds-dir" >&2; exit 2; }
      WORLDS_DIR="$1"
      shift
      ;;
    --match)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --match" >&2; exit 2; }
      MATCH_GLOB="$1"
      shift
      ;;
    --regex)
      shift
      [[ $# -gt 0 ]] || { echo "Missing value for --regex" >&2; exit 2; }
      REGEX="$1"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --yes)
      YES=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MATCH_GLOB}" && -z "${REGEX}" ]]; then
  echo "error: provide --match or --regex" >&2
  usage >&2
  exit 2
fi
if [[ -n "${MATCH_GLOB}" && -n "${REGEX}" ]]; then
  echo "error: use only one of --match or --regex" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "${WORLDS_DIR}" ]]; then
  echo "error: worlds directory not found: ${WORLDS_DIR}" >&2
  exit 2
fi

# Refuse to run if Luanti/Minetest is running (same safety as deploy.sh).
if pgrep -x "luanti" >/dev/null || pgrep -x "minetest" >/dev/null; then
  echo "Luanti is running. Quit it before deleting worlds." >&2
  exit 1
fi

targets=()

if [[ -n "${REGEX}" ]]; then
  # List directories and filter by regex.
  while IFS= read -r name; do
    [[ -d "${WORLDS_DIR}/${name}" ]] || continue
    if echo "${name}" | grep -E -q "${REGEX}"; then
      targets+=("${WORLDS_DIR}/${name}")
    fi
  done < <(ls -1 "${WORLDS_DIR}")
else
  # Glob match.
  shopt -s nullglob
  for w in "${WORLDS_DIR}/${MATCH_GLOB}"; do
    [[ -d "${w}" ]] || continue
    targets+=("${w}")
  done
  shopt -u nullglob
fi

if [[ ${#targets[@]} -eq 0 ]]; then
  echo "No matching worlds in: ${WORLDS_DIR}"
  exit 0
fi

echo "Worlds directory: ${WORLDS_DIR}"
echo "Matched ${#targets[@]} world(s):"
for t in "${targets[@]}"; do
  echo "  - $(basename "${t}")"
done

if [[ ${DRY_RUN} -eq 1 ]]; then
  echo "dry-run: nothing deleted."
  exit 0
fi

if [[ ${YES} -eq 0 ]]; then
  echo
  read -r -p "Delete these worlds? Type 'yes' to confirm: " ans
  if [[ "${ans}" != "yes" ]]; then
    echo "Aborted."
    exit 1
  fi
fi

for t in "${targets[@]}"; do
  rm -rf "${t}"
done

echo "Deleted ${#targets[@]} world(s)."