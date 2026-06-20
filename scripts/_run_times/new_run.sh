#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# new_run.sh — deterministic per-date GPU-slot allocator for experiment runs.
#
# Scheduling is scoped to a SINGLE date directory (scripts/_run_times/<date>/).
# Date directories are independent; this script never looks across them.
#
# 8-GPU-server grouping convention (see CLAUDE.md "Run-time GPU-slot grouping"):
#   * 4-GPU experiments pair two-per-server: X.1 -> GPU 0-3, X.2 -> GPU 4-7
#   * 8-GPU experiments take a whole server: X   -> GPU 0-7
#
# Given the semantic experiment script, this tool:
#   1. computes the next free slot from existing files in the date dir
#      (4-GPU jobs backfill the lowest open half; 8-GPU jobs take a fresh server),
#   2. patches that experiment's YAML gpu_ids to match the slot, and
#   3. writes a thin wrapper scripts/_run_times/<date>/<slot>-<desc>.sh that
#      delegates to the semantic script (GPU assignment lives in the YAML).
#
# Usage:
#   scripts/_run_times/new_run.sh --script scripts/<family>/run_<...>.sh \
#       [--date YYYY_MM_DD] [--gpus 4|8] [--desc <name>] [--dry-run]
#
#   --script   semantic experiment script (repo-relative or absolute)   [required]
#   --date     date directory name (default: today, YYYY_MM_DD)
#   --gpus     4 (half server) or 8 (whole server)                       [default 4]
#   --desc     wrapper description (default: derived from script name)
#   --dry-run  print the plan only; write nothing, patch nothing
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts/_run_times
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

DATE=""
SEMANTIC=""
GPUS=4
DESC=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --date)    DATE="$2"; shift 2 ;;
    --script)  SEMANTIC="$2"; shift 2 ;;
    --gpus)    GPUS="$2"; shift 2 ;;
    --desc)    DESC="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) grep '^#' "$0" | sed 's/^#\{1,\} \{0,1\}//'; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ -n "$DATE" ]] || DATE="$(date +%Y_%m_%d)"
[[ -n "$SEMANTIC" ]] || { echo "ERROR: --script is required" >&2; exit 2; }
[[ "$GPUS" == "4" || "$GPUS" == "8" ]] || { echo "ERROR: --gpus must be 4 or 8" >&2; exit 2; }

# Resolve semantic script path (accept absolute or repo-relative).
if [[ "$SEMANTIC" = /* ]]; then SEM_ABS="$SEMANTIC"; else SEM_ABS="${REPO_ROOT}/${SEMANTIC}"; fi
SEM_REL="${SEM_ABS#"${REPO_ROOT}"/}"
if [[ "$DRY_RUN" -eq 0 ]]; then
  [[ -f "$SEM_ABS" ]] || { echo "ERROR: semantic script not found: $SEM_ABS" >&2; exit 2; }
fi

# Derive desc from semantic script basename if not given:
#   run_B_xxx_train_sample_eval.sh -> B_xxx
if [[ -z "$DESC" ]]; then
  base="$(basename "$SEM_ABS" .sh)"
  base="${base#run_}"
  base="${base%_train_sample_eval}"
  DESC="$base"
fi

DATE_DIR="${SCRIPT_DIR}/${DATE}"

# --- Compute next slot (within this date dir only) ---------------------------
declare -A HALF1 HALF2 FULL
MAXMAJOR=0
if [[ -d "$DATE_DIR" ]]; then
  shopt -s nullglob
  for f in "$DATE_DIR"/*.sh; do
    tok="$(basename "$f")"; tok="${tok%%-*}"
    if [[ "$tok" =~ ^([0-9]+)\.([0-9]+)$ ]]; then
      maj="${BASH_REMATCH[1]}"; half="${BASH_REMATCH[2]}"
      [[ "$half" == "1" ]] && HALF1[$maj]=1
      [[ "$half" == "2" ]] && HALF2[$maj]=1
      (( maj > MAXMAJOR )) && MAXMAJOR=$maj
    elif [[ "$tok" =~ ^([0-9]+)$ ]]; then
      maj="${BASH_REMATCH[1]}"; FULL[$maj]=1
      (( maj > MAXMAJOR )) && MAXMAJOR=$maj
    fi
  done
  shopt -u nullglob
fi

SLOT=""; GPU_LIST=""
if [[ "$GPUS" == "8" ]]; then
  SLOT="$(( MAXMAJOR + 1 ))"
  GPU_LIST="0, 1, 2, 3, 4, 5, 6, 7"
else
  maj=1
  while : ; do
    full="${FULL[$maj]:-0}"; h1="${HALF1[$maj]:-0}"; h2="${HALF2[$maj]:-0}"
    if [[ "$full" == "1" ]]; then maj=$(( maj + 1 )); continue; fi   # closed by an 8-GPU job
    if [[ "$h1" != "1" ]]; then SLOT="${maj}.1"; GPU_LIST="0, 1, 2, 3"; break; fi
    if [[ "$h2" != "1" ]]; then SLOT="${maj}.2"; GPU_LIST="4, 5, 6, 7"; break; fi
    maj=$(( maj + 1 ))
  done
fi

WRAPPER="${DATE_DIR}/${SLOT}-${DESC}.sh"

# Find the experiment YAML from the semantic script's CONFIG= line.
# Guard with grep -q so a script lacking CONFIG= doesn't trip set -e + pipefail
# (an empty grep would make the pipeline exit non-zero and abort before the
# graceful "config not resolved" warning below); grep -m1 avoids a head/SIGPIPE.
CONFIG_REL=""
if [[ -f "$SEM_ABS" ]] && grep -qE '^[[:space:]]*CONFIG=' "$SEM_ABS"; then
  CONFIG_REL="$(grep -m1 -E '^[[:space:]]*CONFIG=' "$SEM_ABS" \
    | sed -E 's/^[[:space:]]*CONFIG=//; s/^"//; s/"$//; s/^'\''//; s/'\''$//')"
fi

echo "date dir : scripts/_run_times/${DATE}/"
echo "slot     : ${SLOT}   (gpu_ids: [${GPU_LIST}])"
echo "wrapper  : ${WRAPPER#"${REPO_ROOT}"/}"
echo "semantic : ${SEM_REL}"
echo "config   : ${CONFIG_REL:-<not found>}"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "(dry-run: nothing written, no YAML patched)"
  exit 0
fi

# --- Patch YAML gpu_ids ------------------------------------------------------
if [[ -n "$CONFIG_REL" && -f "${REPO_ROOT}/${CONFIG_REL}" ]]; then
  CFG_ABS="${REPO_ROOT}/${CONFIG_REL}"
  # Anchor to the TOP-LEVEL key only (column 0) — this is the one training reads
  # via cfg.get("gpu_ids"); never rewrite a nested/indented gpu_ids: under some
  # sub-config.
  if grep -qE '^gpu_ids:' "$CFG_ABS"; then
    sed -i -E "s|^gpu_ids:.*|gpu_ids: [${GPU_LIST}]|" "$CFG_ABS"
    echo "patched  : gpu_ids in ${CONFIG_REL} -> [${GPU_LIST}]"
  else
    echo "WARNING  : no gpu_ids line in ${CONFIG_REL}; left unchanged" >&2
  fi
else
  echo "WARNING  : config not resolved from semantic script; gpu_ids not patched" >&2
fi

# --- Write wrapper -----------------------------------------------------------
mkdir -p "$DATE_DIR"
cat > "$WRAPPER" <<EOF
#!/usr/bin/env bash
# Auto-generated by scripts/_run_times/new_run.sh
# Date group: ${DATE}   Slot: ${SLOT}   GPUs: [${GPU_LIST}]
# Delegates to the semantic experiment script; GPU assignment lives in its YAML gpu_ids.
set -euo pipefail
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="\$(cd "\${HERE}/../../.." && pwd)"
exec bash "\${REPO_ROOT}/${SEM_REL}"
EOF
chmod +x "$WRAPPER"
echo "created  : ${WRAPPER#"${REPO_ROOT}"/}"
