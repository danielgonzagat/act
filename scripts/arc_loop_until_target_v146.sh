#!/usr/bin/env bash
set -euo pipefail

# ARC loop runner (ACT vs ARC pt2):
# - Runs the pt2 induction+scale loop until a target solve rate is reached (or Ctrl-C).
# - Uses only global config (no per-task tuning). Each iter is WORM.
#
# Environment variables (optional):
#   LIMIT=100
#   TARGET=0.70
#   JOBS=8
#   MAX_DEPTH=4
#   MAX_PROGRAMS=2000
#   TASK_TIMEOUT_S=90
#   MACRO_BANK_IN=... (optional)
#   CONCEPT_BANK_IN=... (optional)
#   STOP_HUR=0.0 (optional; informational stop condition if >0)
#   STOP_MULTISTEP=0 (optional)

LIMIT="${LIMIT:-100}"
TARGET="${TARGET:-0.70}"
JOBS="${JOBS:-8}"
MAX_DEPTH="${MAX_DEPTH:-4}"
MAX_PROGRAMS="${MAX_PROGRAMS:-2000}"
TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-0}"
SCALE_TASK_TIMEOUT_S="${SCALE_TASK_TIMEOUT_S:-${TASK_TIMEOUT_S}}"

MACRO_BANK_IN="${MACRO_BANK_IN:-}"
CONCEPT_BANK_IN="${CONCEPT_BANK_IN:-}"

STOP_HUR="${STOP_HUR:-0.0}"
STOP_MULTISTEP="${STOP_MULTISTEP:-0}"

ARGS=(--limit "${LIMIT}" --iters 0 --jobs "${JOBS}" --pressure 1 --tries 1)
ARGS+=(--task_timeout_s "${TASK_TIMEOUT_S}" --max_depth "${MAX_DEPTH}" --max_programs "${MAX_PROGRAMS}")
ARGS+=(--stop_rate_kmax "${TARGET}")

if [ "${STOP_HUR}" != "0.0" ]; then
  ARGS+=(--stop_hur "${STOP_HUR}")
fi
if [ "${STOP_MULTISTEP}" != "0" ]; then
  ARGS+=(--stop_multistep_solved "${STOP_MULTISTEP}")
fi
if [ -n "${MACRO_BANK_IN}" ]; then
  ARGS+=(--macro_bank_in "${MACRO_BANK_IN}")
fi
if [ -n "${CONCEPT_BANK_IN}" ]; then
  ARGS+=(--concept_bank_in "${CONCEPT_BANK_IN}")
fi

export PYTHONPYCACHEPREFIX="${PYTHONPYCACHEPREFIX:-$PWD/.pycache}"
export SCALE_TASK_TIMEOUT_S

python3 scripts/arc_loop_pt2_iter_v146.py "${ARGS[@]}"
