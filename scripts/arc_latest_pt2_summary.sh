#!/usr/bin/env bash
set -euo pipefail

# Prints a compact JSON summary for the most recent pt2 loop run.
# By default prefers the macro (scale) summary if present.
#
# Usage:
#   scripts/arc_latest_pt2_summary.sh [limit]
# Examples:
#   scripts/arc_latest_pt2_summary.sh 100
#   scripts/arc_latest_pt2_summary.sh 0

LIMIT="${1:-}"
ROOT="${ROOT:-artifacts}"

pick_latest() {
  local pattern="$1"
  # shellcheck disable=SC2086
  ls -1t ${pattern} 2>/dev/null | head -n 1 || true
}

if [ -n "${LIMIT}" ]; then
  MACRO_PATTERN="${ROOT}/arc_pt2_v141_train${LIMIT}_*_macros_try1/summary.json"
  BASE_PATTERN="${ROOT}/arc_pt2_v141_train${LIMIT}_*_try1/summary.json"
else
  MACRO_PATTERN="${ROOT}/arc_pt2_v141_train*_macros_try1/summary.json"
  BASE_PATTERN="${ROOT}/arc_pt2_v141_train*_try1/summary.json"
fi

SUM_MACRO="$(pick_latest "${MACRO_PATTERN}")"
SUM_BASE="$(pick_latest "${BASE_PATTERN}")"

mtime() {
  local p="$1"
  if [ -z "${p}" ]; then
    echo 0
    return
  fi
  # macOS stat
  stat -f %m "${p}" 2>/dev/null || echo 0
}

MT_MACRO="$(mtime "${SUM_MACRO}")"
MT_BASE="$(mtime "${SUM_BASE}")"

SUM=""
if [ "${MT_MACRO}" -ge "${MT_BASE}" ]; then
  SUM="${SUM_MACRO}"
else
  SUM="${SUM_BASE}"
fi

if [ -z "${SUM}" ]; then
  echo "no_summary_found (looked for: ${MACRO_PATTERN} and ${BASE_PATTERN})" >&2
  exit 2
fi

echo "${SUM}" >&2
if [[ "${SUM}" != *"_macros_try1/summary.json" ]]; then
  echo "[note] macro summary not found yet for newest run; showing base summary" >&2

  # If a macro run directory exists but has not produced a summary yet, show live progress to avoid
  # misreading "HUR=0" as hierarchy regression while the macro stage is still running.
  if [ -n "${LIMIT}" ]; then
    MACRO_DIR_PATTERN="${ROOT}/arc_pt2_v141_train${LIMIT}_*_macros_try1"
  else
    MACRO_DIR_PATTERN="${ROOT}/arc_pt2_v141_train*_macros_try1"
  fi

  MACRO_DIR="$(ls -1td ${MACRO_DIR_PATTERN} 2>/dev/null | head -n 1 || true)"
  if [ -n "${MACRO_DIR}" ] && [ ! -f "${MACRO_DIR}/summary.json" ] && [ -f "${MACRO_DIR}/progress.log" ]; then
    # Heuristic is only for operator convenience: we never change solver behavior based on this,
    # we only surface whether the macro stage is actively progressing.
    PROG_MTIME="$(stat -f %m "${MACRO_DIR}/progress.log" 2>/dev/null || echo 0)"
    NOW="$(date +%s)"
    AGE="$(( NOW - PROG_MTIME ))"
    if [ "${AGE}" -ge 300 ]; then
      echo "[note] macro run incomplete (no summary yet): ${MACRO_DIR} (last progress update ${AGE}s ago)" >&2
    else
      echo "[note] macro run in progress (no summary yet): ${MACRO_DIR} (last progress update ${AGE}s ago)" >&2
    fi
    echo "[note] macro progress tail:" >&2
    tail -n 3 "${MACRO_DIR}/progress.log" >&2 || true
  fi
fi
python3 scripts/arc_print_summary_v141.py "${SUM}"
