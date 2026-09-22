#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# new_run.sh — write a run-time wrapper that claims its GPUs at launch time.
#
# GPUs are NOT bound when the wrapper is written.  The wrapper asks
# claim_gpus.sh for as many idle GPUs as the experiment needs and picks them in
# id order, so the same script runs on a 4-GPU box and an 8-GPU box alike.  The
# old scheme -- slot X.1 means GPU 0-1, X.2 means GPU 2-3 ... -- assumed every
# machine had 8 GPUs and is gone, together with the slot numbers.
#
# Given the semantic experiment script, this tool:
#   1. writes the experiment YAML's gpu_ids as the first N ids (a placeholder
#      that fixes the DDP world size and keeps the script runnable on its own),
#   2. writes scripts/_run_times/<date>/<desc>.sh, which claims N idle GPUs and
#      passes them to the semantic script via PROMOE_GPU_IDS_OVERRIDE.
#
# Usage:
#   scripts/_run_times/new_run.sh --script scripts/<family>/run_<...>.sh \
#       --gpus N [--date YYYY_MM_DD] [--desc <name>] [--dry-run]
#
#   --script   semantic experiment script (repo-relative or absolute)  [required]
#   --gpus     GPUs this experiment uses for DDP (positive integer)     [default 4]
#   --date     date directory name (default: today, YYYY_MM_DD)
#   --desc     wrapper name (default: derived from the script name)
#   --dry-run  print the plan only; write nothing, patch nothing
#
# Every arm of one comparison must use the SAME --gpus: per-rank quantities
# (routing-contrastive class centers, pooled expert representations, EC-BC
# selection) change with the world size, so arms at different GPU counts are not
# comparable.
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
[[ "$GPUS" =~ ^[0-9]+$ && "$GPUS" -ge 1 ]] || { echo "ERROR: --gpus must be a positive integer" >&2; exit 2; }

# Resolve semantic script path (accept absolute or repo-relative).
if [[ "$SEMANTIC" = /* ]]; then SEM_ABS="$SEMANTIC"; else SEM_ABS="${REPO_ROOT}/${SEMANTIC}"; fi
SEM_REL="${SEM_ABS#"${REPO_ROOT}"/}"
# The wrapper resolves the semantic script as ${REPO_ROOT}/${SEM_REL}, so a
# script outside the repo would be pasted onto REPO_ROOT and never resolve.
[[ "$SEM_REL" != /* ]] || { echo "ERROR: semantic script must live inside the repo: $SEM_ABS" >&2; exit 2; }
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
WRAPPER="${DATE_DIR}/${DESC}.sh"

# Placeholder gpu_ids: the first N ids.  The wrapper overrides them at launch;
# this value only fixes the world size and keeps the semantic script usable on
# its own.
GPU_LIST="$(seq -s ', ' 0 $(( GPUS - 1 )))"

# Find the experiment YAML from the semantic script's CONFIG= line.
CONFIG_REL=""
if [[ -f "$SEM_ABS" ]] && grep -qE '^[[:space:]]*CONFIG=' "$SEM_ABS"; then
  CONFIG_REL="$(grep -m1 -E '^[[:space:]]*CONFIG=' "$SEM_ABS" \
    | sed -E 's/^[[:space:]]*CONFIG=//; s/^"//; s/"$//; s/^'\''//; s/'\''$//')"
fi

echo "date dir : scripts/_run_times/${DATE}/"
echo "gpus     : ${GPUS} (claimed at launch, in id order; placeholder gpu_ids: [${GPU_LIST}])"
echo "wrapper  : ${WRAPPER#"${REPO_ROOT}"/}"
echo "semantic : ${SEM_REL}"
echo "config   : ${CONFIG_REL:-<not found>}"

if [[ -e "$WRAPPER" && "$DRY_RUN" -eq 0 ]]; then
  echo "ERROR: wrapper already exists: ${WRAPPER#"${REPO_ROOT}"/}" >&2
  echo "       pass a different --desc, or remove it first." >&2
  exit 2
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "(dry-run: nothing written, no YAML patched)"
  exit 0
fi

# --- Patch YAML gpu_ids ------------------------------------------------------
if [[ -n "$CONFIG_REL" && -f "${REPO_ROOT}/${CONFIG_REL}" ]]; then
  CFG_ABS="${REPO_ROOT}/${CONFIG_REL}"
  # Anchor to the TOP-LEVEL key only (column 0) — this is the one training reads
  # via cfg.get("gpu_ids"); never rewrite a nested/indented gpu_ids.
  if grep -qE '^gpu_ids:' "$CFG_ABS"; then
    sed -i -E "s|^gpu_ids:.*|gpu_ids: [${GPU_LIST}]  # placeholder: the wrapper claims idle GPUs at launch|" "$CFG_ABS"
    echo "patched  : gpu_ids in ${CONFIG_REL} -> [${GPU_LIST}] (${GPUS} GPUs)"
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
# Date group: ${DATE}   GPUs needed: ${GPUS}
# Claims ${GPUS} idle GPU(s) in id order and hands them to the semantic script,
# so this runs unchanged on a 4-GPU or an 8-GPU machine.  A GPU counts as idle
# when its used memory is under PROMOE_GPU_IDLE_MAX_MIB (100 by default, which
# ignores the few MiB of driver residue an unused card reports) and no live
# claim holds it.  Set PROMOE_GPU_IDS_OVERRIDE yourself to pick GPUs by hand.
set -euo pipefail
HERE="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="\$(cd "\${HERE}/../../.." && pwd)"
source "\${REPO_ROOT}/scripts/_run_times/claim_gpus.sh"

if [[ -z "\${PROMOE_GPU_IDS_OVERRIDE:-}" ]]; then
    # Call it directly, never via \$( ): in a subshell the EXIT trap would
    # release the claim before the trainer ever starts.
    promoe_claim_gpus ${GPUS}
    export PROMOE_GPU_IDS_OVERRIDE="\${PROMOE_CLAIMED_GPUS}"
fi
echo "[\$(date '+%H:%M:%S')] ${DESC}: using GPU(s) \${PROMOE_GPU_IDS_OVERRIDE}"
exec bash "\${REPO_ROOT}/${SEM_REL}"
EOF
chmod +x "$WRAPPER"
echo "created  : ${WRAPPER#"${REPO_ROOT}"/}"
