#!/usr/bin/env bash
set -euo pipefail

# ARC loop helper (ACT vs ARC pt2):
# - Runs ARC training split with 8 cores by default
# - Writes a run dir + a diag report
# - Prints a tail command for live progress

LIMIT="${1:-100}"
ARC_ROOT="${ARC_ROOT:-ARC-AGI}"
JOBS="${JOBS:-8}"
SEED="${SEED:-0}"
TRIES="${TRIES:-1}"
TASK_TIMEOUT_S="${TASK_TIMEOUT_S:-0}"
# Scale/measurement timeout (macro run) can be set independently from the induction run timeout.
# Default keeps behavior unchanged (same as TASK_TIMEOUT_S).
SCALE_TASK_TIMEOUT_S="${SCALE_TASK_TIMEOUT_S:-${TASK_TIMEOUT_S}}"
# Harness watchdog (NOT a solver timeout): if no tasks complete for this many seconds,
# the harness finalizes the run and marks remaining tasks as SEARCH_BUDGET_EXCEEDED.
NO_PROGRESS_TIMEOUT_S="${NO_PROGRESS_TIMEOUT_S:-0}"
SCALE_NO_PROGRESS_TIMEOUT_S="${SCALE_NO_PROGRESS_TIMEOUT_S:-${NO_PROGRESS_TIMEOUT_S}}"
PRESSURE="${PRESSURE:-0}"
SCALE_USE_PRESSURE_ARGS="${SCALE_USE_PRESSURE_ARGS:-0}"
REQUIRE_CONCEPT_CALL="${REQUIRE_CONCEPT_CALL:-1}"
MACRO_TRY_ON_FAIL_ONLY="${MACRO_TRY_ON_FAIL_ONLY:-1}"
# Macro stage can run with a different policy. Default is 1 (baseline-first) to avoid
# macro-first branching explosions during operator bootstrapping.
SCALE_MACRO_TRY_ON_FAIL_ONLY="${SCALE_MACRO_TRY_ON_FAIL_ONLY:-1}"
MACRO_PROPOSE_MAX_DEPTH="${MACRO_PROPOSE_MAX_DEPTH:-0}"
SCALE_MACRO_PROPOSE_MAX_DEPTH="${SCALE_MACRO_PROPOSE_MAX_DEPTH:-0}"
MACRO_MAX_TEMPLATES="${MACRO_MAX_TEMPLATES:-24}"
MACRO_MAX_INSTANTIATIONS="${MACRO_MAX_INSTANTIATIONS:-10}"
MACRO_MAX_BRANCH_PER_OP="${MACRO_MAX_BRANCH_PER_OP:-10}"
# Reachability pruning is a performance optimization. For operator discovery under tight caps,
# disabling it can expose "unreachable-under-caps" prefixes in trace_programs for mining.
DISABLE_REACHABILITY_PRUNING="${DISABLE_REACHABILITY_PRUNING:-0}"
SCALE_DISABLE_REACHABILITY_PRUNING="${SCALE_DISABLE_REACHABILITY_PRUNING:-0}"
MACRO_BANK_IN="${MACRO_BANK_IN:-}"
CONCEPTS="${CONCEPTS:-1}"
CONCEPT_BANK_IN="${CONCEPT_BANK_IN:-}"
DEEP="${DEEP:-0}"
DEEP_LIMIT="${DEEP_LIMIT:-24}"
DEEP_MAX_DEPTH="${DEEP_MAX_DEPTH:-6}"
DEEP_MAX_PROGRAMS="${DEEP_MAX_PROGRAMS:-12000}"
DEEP_TIMEOUT_S="${DEEP_TIMEOUT_S:-600}"
DEEP_SEED_K="${DEEP_SEED_K:-4}"
POINT_PATCH="${POINT_PATCH:-0}"
POINT_PATCH_MAX_POINTS="${POINT_PATCH_MAX_POINTS:-12}"
MAX_DEPTH="${MAX_DEPTH:-4}"
MAX_PROGRAMS="${MAX_PROGRAMS:-2000}"
SOLUTION_COST_SLACK_BITS="${SOLUTION_COST_SLACK_BITS:-16}"
MACROS="${MACROS:-1}"
MACRO_MIN_SUPPORT="${MACRO_MIN_SUPPORT:-1}"
MACRO_MAX_MACROS="${MACRO_MAX_MACROS:-64}"
MACRO_MIN_LEN="${MACRO_MIN_LEN:-2}"
MACRO_MAX_LEN="${MACRO_MAX_LEN:-5}"
# Also mine operators from SOLVED tasks that Ω marked as born-from-failure (e.g., shallow-suppressed successes).
MACRO_INCLUDE_SOLVED_FROM_FAILURE="${MACRO_INCLUDE_SOLVED_FROM_FAILURE:-0}"
MERGED_MIN_SUPPORT="${MERGED_MIN_SUPPORT:-1}"
MACRO_TRACE_MAX_PROGRAMS_PER_TASK="${MACRO_TRACE_MAX_PROGRAMS_PER_TASK:-40}"
MACRO_TRACE_MAX_LOSS_SHAPE="${MACRO_TRACE_MAX_LOSS_SHAPE:-2}"
MACRO_TRACE_MAX_LOSS_CELLS="${MACRO_TRACE_MAX_LOSS_CELLS:-120}"
CONCEPT_MIN_SUPPORT="${CONCEPT_MIN_SUPPORT:-1}"
CONCEPT_MAX_CONCEPTS="${CONCEPT_MAX_CONCEPTS:-128}"
# Concept mining controls (v146). Defaults preserve existing behavior.
CONCEPT_MIN_LEN="${CONCEPT_MIN_LEN:-1}"
CONCEPT_MAX_LEN="${CONCEPT_MAX_LEN:-3}"
CONCEPT_TRACE_MAX_PROGRAMS_PER_TASK="${CONCEPT_TRACE_MAX_PROGRAMS_PER_TASK:-40}"
CONCEPT_TRACE_MAX_LOSS_SHAPE="${CONCEPT_TRACE_MAX_LOSS_SHAPE:-0}"
CONCEPT_TRACE_MAX_LOSS_CELLS="${CONCEPT_TRACE_MAX_LOSS_CELLS:-120}"
# Concept CSG mining controls. These produce concrete, binderized multi-step closures
# that keep concept_call atomic (no arg branching) and collapse search depth early.
CSG_MIN_LEN="${CSG_MIN_LEN:-3}"
CSG_MAX_LEN="${CSG_MAX_LEN:-12}"
CSG_MIN_SUPPORT="${CSG_MIN_SUPPORT:-3}"
CSG_SUPPORT_SLACK_EVERY="${CSG_SUPPORT_SLACK_EVERY:-2}"
CSG_MAX_TEMPLATES="${CSG_MAX_TEMPLATES:-512}"
CSG_MAX_LOSS_SHAPE="${CSG_MAX_LOSS_SHAPE:-0}"
CSG_MAX_LOSS_CELLS="${CSG_MAX_LOSS_CELLS:-80}"
CSG_PREFIX_ONLY="${CSG_PREFIX_ONLY:-1}"
# v153: window miner (not just prefixes) and strict TRAIN-loss drop from the window start state.
CSG_MAX_CANDIDATES_PER_TASK="${CSG_MAX_CANDIDATES_PER_TASK:-3}"
CSG_MIN_LOSS_DROP="${CSG_MIN_LOSS_DROP:-1}"
CSG_REQUIRE_LAST_WRITES_GRID="${CSG_REQUIRE_LAST_WRITES_GRID:-1}"
# Optional: enable deterministic trace→CSG induction retry pass inside the solver (learn mode only).
# This can collapse deep near-miss pipelines into atomic concept_call steps and improve mining yield.
TRACE_CSG_INDUCTION="${TRACE_CSG_INDUCTION:-0}"
TRACE_CSG_FIRST_PASS_FRAC="${TRACE_CSG_FIRST_PASS_FRAC:-0.55}"
# Concept bank merge cap (v146). Defaults preserve existing behavior.
CONCEPT_BANK_MAX_LEN="${CONCEPT_BANK_MAX_LEN:-3}"
# Keep merged min support low to bootstrap concept emergence; concept calls remain fail-closed per-task.
CONCEPT_MERGED_MIN_SUPPORT="${CONCEPT_MERGED_MIN_SUPPORT:-1}"
CONCEPT_BANK_MAX_CONCEPTS="${CONCEPT_BANK_MAX_CONCEPTS:-128}"

OMEGA="${OMEGA:-0}"
OMEGA_STATE_IN="${OMEGA_STATE_IN:-}"
INDUCTION_LOG_OUT="${INDUCTION_LOG_OUT:-}"
OMEGA_REQUIRE_CONCEPT_CALL_AFTER_RUNS="${OMEGA_REQUIRE_CONCEPT_CALL_AFTER_RUNS:-0}"
OMEGA_REQUIRE_PROMOTED_CONCEPT_CALL_AFTER_RUNS="${OMEGA_REQUIRE_PROMOTED_CONCEPT_CALL_AFTER_RUNS:-0}"
OMEGA_COMBINE_BASE_MACRO="${OMEGA_COMBINE_BASE_MACRO:-1}"

TS="$(date +%Y%m%d_%H%M%S)"
OUT_BASE="artifacts/arc_pt2_v141_train${LIMIT}_${TS}"
PROGRESS_JSONL="artifacts/arc_pt2_v141_train${LIMIT}_${TS}_progress.jsonl"

echo "[arc_loop] out_base=${OUT_BASE}"
echo "[arc_loop] arc_root=${ARC_ROOT}"
echo "[arc_loop] tail: tail -f ${OUT_BASE}_try1/progress.log"
echo "[arc_loop] progress: tail -f ${PROGRESS_JSONL}"
echo

if [ "${TRIES}" != "1" ] && [ "${TASK_TIMEOUT_S}" != "0" ]; then
  echo "[arc_loop] ERROR: TASK_TIMEOUT_S requires TRIES=1 (fast iteration mode)"
  exit 2
fi
if [ "${TRIES}" != "1" ] && [ "${SCALE_TASK_TIMEOUT_S}" != "0" ]; then
  echo "[arc_loop] ERROR: SCALE_TASK_TIMEOUT_S requires TRIES=1 (fast iteration mode)"
  exit 2
fi

EXTRA_ARGS=()
if [ "${POINT_PATCH}" != "0" ]; then
  EXTRA_ARGS+=(--enable_point_patch_repair --point_patch_max_points "${POINT_PATCH_MAX_POINTS}")
fi

PRESSURE_ARGS=()
if [ "${PRESSURE}" != "0" ]; then
  # Pressure run (ACT vs ARC pt2): disable non-persistent closure routes and force abstraction.
  # Note: keep residual stage enabled so the solver can emit FAIL/MISSING_CONCEPT with a
  # deterministic residual_signature + concept_template (used by concept induction + Ω v2).
  PRESSURE_ARGS+=(--disable_repair_stage --macro_try_on_fail_only 0 --abstraction_pressure)
else
  PRESSURE_ARGS+=(--macro_try_on_fail_only "${MACRO_TRY_ON_FAIL_ONLY}")
fi

# Scale-mode args (used after updating concept/macro banks to measure headline solve-rate).
# When PRESSURE=1, we intentionally drop --abstraction_pressure for the score run; pressure is for induction,
# scale is for performance measurement.
SCALE_ARGS=()
SCALE_ARGS+=(--macro_try_on_fail_only "${SCALE_MACRO_TRY_ON_FAIL_ONLY}")
if [ "${PRESSURE}" != "0" ] && [ "${SCALE_USE_PRESSURE_ARGS}" != "0" ]; then
  # Optional: keep the macro run under the same ontological pressure regime.
  # This avoids long-tail stalls from the full repair stage while we are still mining/validating operators.
  SCALE_ARGS+=(--disable_repair_stage --abstraction_pressure)
fi
# Contract safety: when concept_call is required, the macro/scale stage must still
# surface concept_call (fail-closed). Without abstraction pressure, the solver can
# produce primitive-only programs that are rejected by --require_concept_call.
if [ "${REQUIRE_CONCEPT_CALL}" != "0" ]; then
  SCALE_ARGS+=(--abstraction_pressure)
fi

CONCEPT_ARGS=()
if [ "${CONCEPTS}" != "0" ] && [ -n "${CONCEPT_BANK_IN}" ]; then
  if [ ! -f "${CONCEPT_BANK_IN}" ]; then
    echo "[arc_loop] ERROR: CONCEPT_BANK_IN not found: ${CONCEPT_BANK_IN}"
    exit 2
  fi
  CONCEPT_ARGS+=(--concept_templates "${CONCEPT_BANK_IN}")
fi

MACRO_ARGS_BASE=()
if [ "${MACROS}" != "0" ] && [ -n "${MACRO_BANK_IN}" ]; then
  if [ ! -f "${MACRO_BANK_IN}" ]; then
    echo "[arc_loop] ERROR: MACRO_BANK_IN not found: ${MACRO_BANK_IN}"
    exit 2
  fi
  # Use the current macro bank during the LEARN/base run as well.
  # Rationale: macro_call steps can be flattened into primitive pipelines for CSG mining,
  # and can also help reach informative near-miss traces under tight budgets.
  MACRO_ARGS_BASE+=(--macro_templates "${MACRO_BANK_IN}")
fi

RUN_BASE_CMD=(python3 scripts/run_arc_scalpel_v141.py)
RUN_BASE_CMD+=(--mode learn)
RUN_BASE_CMD+=(--arc_root "${ARC_ROOT}")
RUN_BASE_CMD+=(--split training)
RUN_BASE_CMD+=(--limit "${LIMIT}")
RUN_BASE_CMD+=(--seed "${SEED}")
RUN_BASE_CMD+=(--tries "${TRIES}")
RUN_BASE_CMD+=(--jobs "${JOBS}")
RUN_BASE_CMD+=(--require_concept_call "${REQUIRE_CONCEPT_CALL}")
RUN_BASE_CMD+=(--task_timeout_s "${TASK_TIMEOUT_S}")
RUN_BASE_CMD+=(--no_progress_timeout_s "${NO_PROGRESS_TIMEOUT_S}")
RUN_BASE_CMD+=(--max_depth "${MAX_DEPTH}")
RUN_BASE_CMD+=(--max_programs "${MAX_PROGRAMS}")
RUN_BASE_CMD+=(--solution_cost_slack_bits "${SOLUTION_COST_SLACK_BITS}")
RUN_BASE_CMD+=(--macro_propose_max_depth "${MACRO_PROPOSE_MAX_DEPTH}")
RUN_BASE_CMD+=(--macro_max_templates "${MACRO_MAX_TEMPLATES}")
RUN_BASE_CMD+=(--macro_max_instantiations "${MACRO_MAX_INSTANTIATIONS}")
RUN_BASE_CMD+=(--macro_max_branch_per_op "${MACRO_MAX_BRANCH_PER_OP}")
if [ "${DISABLE_REACHABILITY_PRUNING}" != "0" ]; then
  RUN_BASE_CMD+=(--disable_reachability_pruning)
fi
if [ "${TRACE_CSG_INDUCTION}" != "0" ]; then
  RUN_BASE_CMD+=(--trace_csg_induction 1 --trace_csg_first_pass_frac "${TRACE_CSG_FIRST_PASS_FRAC}")
fi
if [ "${#EXTRA_ARGS[@]}" != "0" ]; then
  RUN_BASE_CMD+=("${EXTRA_ARGS[@]}")
fi
if [ "${#PRESSURE_ARGS[@]}" != "0" ]; then
  RUN_BASE_CMD+=("${PRESSURE_ARGS[@]}")
fi
if [ "${#CONCEPT_ARGS[@]}" != "0" ]; then
  RUN_BASE_CMD+=("${CONCEPT_ARGS[@]}")
fi
if [ "${#MACRO_ARGS_BASE[@]}" != "0" ]; then
  RUN_BASE_CMD+=("${MACRO_ARGS_BASE[@]}")
fi
if [ "${OMEGA}" != "0" ]; then
  RUN_BASE_CMD+=(--omega)
  if [ -n "${OMEGA_STATE_IN}" ]; then
    if [ ! -f "${OMEGA_STATE_IN}" ]; then
      echo "[arc_loop] ERROR: OMEGA_STATE_IN not found: ${OMEGA_STATE_IN}"
      exit 2
    fi
    RUN_BASE_CMD+=(--omega_state_in "${OMEGA_STATE_IN}")
  fi
fi
RUN_BASE_CMD+=(--out_base "${OUT_BASE}")
"${RUN_BASE_CMD[@]}"

python3 scripts/arc_diag_v142_from_v141_run.py \
  --run_dir "${OUT_BASE}_try1" \
  --out_path "artifacts/arc_diag_v142_${TS}.md" \
  --out_json "artifacts/arc_diag_v142_${TS}.json"

python3 scripts/arc_append_progress_v1.py \
  --progress "${PROGRESS_JSONL}" \
  --stage "base" \
  --run_dir "${OUT_BASE}_try1" \
  --concept_bank "${CONCEPT_BANK_IN}" \
  --macro_bank "${MACRO_BANK_IN}"

echo
echo "[arc_loop] summary: ${OUT_BASE}_try1/summary.json"
echo "[arc_loop] diag: artifacts/arc_diag_v142_${TS}.md"

if [ "${CONCEPTS}" != "0" ]; then
  if [ -n "${CONCEPT_BANK_IN}" ]; then
    if [ ! -f "${CONCEPT_BANK_IN}" ]; then
      echo "[arc_loop] ERROR: CONCEPT_BANK_IN not found: ${CONCEPT_BANK_IN}"
      exit 2
    fi
    CONCEPT_BANK_BUMPED="artifacts/arc_concept_bank_v146_${TS}_bumped.jsonl"
    echo
    echo "[arc_loop] bumping concept support: ${CONCEPT_BANK_BUMPED}"
    python3 scripts/arc_bump_concept_support_v146.py \
      --concept_bank_in "${CONCEPT_BANK_IN}" \
      --run_dir "${OUT_BASE}_try1" \
      --out "${CONCEPT_BANK_BUMPED}"
    CONCEPT_BANK_IN="${CONCEPT_BANK_BUMPED}"
    echo "[arc_loop] next CONCEPT_BANK_IN (bumped): ${CONCEPT_BANK_BUMPED}"
  fi

  CONCEPTS_OUT="artifacts/arc_concept_templates_v146_${TS}.jsonl"
  if [ -z "${INDUCTION_LOG_OUT}" ]; then
    INDUCTION_LOG_OUT="artifacts/arc_concept_inductions_v146_${TS}.jsonl"
  fi
  echo
  echo "[arc_loop] mining concepts: ${CONCEPTS_OUT}"
  python3 scripts/arc_induce_concept_templates_v146.py \
    --run_dir "${OUT_BASE}_try1" \
    --out "${CONCEPTS_OUT}" \
    --induction_log "${INDUCTION_LOG_OUT}" \
    --min_support "${CONCEPT_MIN_SUPPORT}" \
    --min_len "${CONCEPT_MIN_LEN}" \
    --max_len "${CONCEPT_MAX_LEN}" \
    --trace_max_programs_per_task "${CONCEPT_TRACE_MAX_PROGRAMS_PER_TASK}" \
    --trace_max_loss_shape "${CONCEPT_TRACE_MAX_LOSS_SHAPE}" \
    --trace_max_loss_cells "${CONCEPT_TRACE_MAX_LOSS_CELLS}" \
    --max_concepts "${CONCEPT_MAX_CONCEPTS}"
  echo "[arc_loop] concept induction log: ${INDUCTION_LOG_OUT}"

  CSG_OUT_V152="artifacts/arc_concept_csg_templates_v152_${TS}.jsonl"
  echo
  echo "[arc_loop] mining concept CSGs (v152 prefixes): ${CSG_OUT_V152}"
  python3 scripts/arc_induce_concept_csg_templates_v152.py \
    --tasks_jsonl "${OUT_BASE}_try1/input/arc_tasks_canonical_v141.jsonl" \
    --traces "${OUT_BASE}_try1/trace_candidates.jsonl" \
    --out "${CSG_OUT_V152}" \
    --min_len "${CSG_MIN_LEN}" \
    --max_len "${CSG_MAX_LEN}" \
    --min_support "${CSG_MIN_SUPPORT}" \
    --support_slack_every "${CSG_SUPPORT_SLACK_EVERY}" \
    --max_templates "${CSG_MAX_TEMPLATES}" \
    --max_loss_shape "${CSG_MAX_LOSS_SHAPE}" \
    --max_loss_cells "${CSG_MAX_LOSS_CELLS}" \
    --max_candidates_per_task "${CSG_MAX_CANDIDATES_PER_TASK}" \
    --min_loss_drop "${CSG_MIN_LOSS_DROP}" \
    --require_last_writes_grid "${CSG_REQUIRE_LAST_WRITES_GRID}"

  CSG_OUT_V153="artifacts/arc_concept_csg_templates_v153_${TS}.jsonl"
  echo
  echo "[arc_loop] mining concept CSGs (v153 windows): ${CSG_OUT_V153}"
  python3 scripts/arc_induce_concept_csg_templates_v153.py \
    --tasks_jsonl "${OUT_BASE}_try1/input/arc_tasks_canonical_v141.jsonl" \
    --traces "${OUT_BASE}_try1/trace_candidates.jsonl" \
    --out "${CSG_OUT_V153}" \
    --min_len "${CSG_MIN_LEN}" \
    --max_len "${CSG_MAX_LEN}" \
    --min_support "${CSG_MIN_SUPPORT}" \
    --support_slack_every "${CSG_SUPPORT_SLACK_EVERY}" \
    --max_templates "${CSG_MAX_TEMPLATES}" \
    --max_loss_shape "${CSG_MAX_LOSS_SHAPE}" \
    --max_loss_cells "${CSG_MAX_LOSS_CELLS}" \
    --max_candidates_per_task "${CSG_MAX_CANDIDATES_PER_TASK}" \
    --min_loss_drop "${CSG_MIN_LOSS_DROP}" \
    --require_last_writes_grid "${CSG_REQUIRE_LAST_WRITES_GRID}"

  CONCEPT_BANK_OUT="artifacts/arc_concept_bank_v150_${TS}.jsonl"
  MERGE_CONCEPT_INS=()
  if [ -n "${CONCEPT_BANK_IN}" ]; then
    MERGE_CONCEPT_INS+=(--in "${CONCEPT_BANK_IN}")
  fi
  MERGE_CONCEPT_INS+=(--in "${CONCEPTS_OUT}")
  MERGE_CONCEPT_INS+=(--in "${CSG_OUT_V152}")
  MERGE_CONCEPT_INS+=(--in "${CSG_OUT_V153}")

  echo
  echo "[arc_loop] updating concept bank: ${CONCEPT_BANK_OUT}"
  python3 scripts/arc_merge_concept_bank_v150.py \
    "${MERGE_CONCEPT_INS[@]}" \
    --out "${CONCEPT_BANK_OUT}"

  CONCEPT_BANK_IN="${CONCEPT_BANK_OUT}"
  echo "[arc_loop] next CONCEPT_BANK_IN: ${CONCEPT_BANK_OUT}"
fi

if [ "${OMEGA}" != "0" ]; then
  if [ "${MACROS}" != "0" ] && [ "${OMEGA_COMBINE_BASE_MACRO}" != "0" ]; then
    echo
    echo "[arc_loop] Ω combine base+macro enabled; deferring Ω update until after macro run"
  else
    echo
    OMEGA_OUT="artifacts/omega_state_v2_${TS}_base.json"
    PREV_ARGS=()
    if [ -n "${OMEGA_STATE_IN}" ]; then
      PREV_ARGS+=(--prev_state "${OMEGA_STATE_IN}")
    fi
    CONCEPT_ARGS_OMEGA=()
    if [ -n "${CONCEPT_BANK_IN}" ]; then
      if [ ! -f "${CONCEPT_BANK_IN}" ]; then
        echo "[arc_loop] ERROR: CONCEPT_BANK_IN not found for Ω update: ${CONCEPT_BANK_IN}"
        exit 2
      fi
      CONCEPT_ARGS_OMEGA+=(--concept_bank "${CONCEPT_BANK_IN}")
    fi
    MACRO_ARGS_OMEGA=()
    if [ -n "${MACRO_BANK_IN}" ]; then
      if [ ! -f "${MACRO_BANK_IN}" ]; then
        echo "[arc_loop] ERROR: MACRO_BANK_IN not found for Ω update: ${MACRO_BANK_IN}"
        exit 2
      fi
      MACRO_ARGS_OMEGA+=(--macro_bank "${MACRO_BANK_IN}")
    fi
    INDUCTION_ARGS_OMEGA=()
    if [ -n "${INDUCTION_LOG_OUT}" ]; then
      if [ ! -f "${INDUCTION_LOG_OUT}" ]; then
        echo "[arc_loop] ERROR: INDUCTION_LOG_OUT not found for Ω update: ${INDUCTION_LOG_OUT}"
        exit 2
      fi
      INDUCTION_ARGS_OMEGA+=(--induction_log "${INDUCTION_LOG_OUT}")
    fi
    python3 scripts/omega_update_v2.py \
      "${PREV_ARGS[@]}" \
      --run_dir "${OUT_BASE}_try1" \
      "${CONCEPT_ARGS_OMEGA[@]}" \
      "${MACRO_ARGS_OMEGA[@]}" \
      "${INDUCTION_ARGS_OMEGA[@]}" \
      --base_max_depth "${MAX_DEPTH}" \
      --base_max_programs "${MAX_PROGRAMS}" \
      --require_concept_call_after_runs "${OMEGA_REQUIRE_CONCEPT_CALL_AFTER_RUNS}" \
      --require_promoted_concept_call_after_runs "${OMEGA_REQUIRE_PROMOTED_CONCEPT_CALL_AFTER_RUNS}" \
      --out "${OMEGA_OUT}"
    OMEGA_STATE_IN="${OMEGA_OUT}"
    echo "[arc_loop] next OMEGA_STATE_IN: ${OMEGA_OUT}"

    python3 scripts/arc_append_progress_v1.py \
      --progress "${PROGRESS_JSONL}" \
      --stage "omega_base" \
      --run_dir "${OUT_BASE}_try1" \
      --omega_state "${OMEGA_STATE_IN}" \
      --concept_bank "${CONCEPT_BANK_IN}" \
      --macro_bank "${MACRO_BANK_IN}"
  fi
fi

if [ "${MACROS}" != "0" ]; then
  MACROS_OUT="artifacts/arc_operator_templates_v147_${TS}.jsonl"
  echo
  echo "[arc_loop] mining operators (born-from-failure): ${MACROS_OUT}"

  MINE_OP_ARGS=()
  if [ "${MACRO_INCLUDE_SOLVED_FROM_FAILURE}" != "0" ]; then
    MINE_OP_ARGS+=(--include_solved_from_failure)
  fi

  python3 scripts/arc_mine_operator_templates_v147.py \
    --run_dir "${OUT_BASE}_try1" \
    --out "${MACROS_OUT}" \
    --min_len "${MACRO_MIN_LEN}" \
    --max_len "${MACRO_MAX_LEN}" \
    --min_support "${MACRO_MIN_SUPPORT}" \
    --max_operators "${MACRO_MAX_MACROS}" \
    --trace_max_programs_per_task "${MACRO_TRACE_MAX_PROGRAMS_PER_TASK}" \
    --trace_max_loss_shape "${MACRO_TRACE_MAX_LOSS_SHAPE}" \
    --trace_max_loss_cells "${MACRO_TRACE_MAX_LOSS_CELLS}" \
    "${MINE_OP_ARGS[@]}"

  # Maintain a persistent macro bank across iterations (CSG-as-object): merge the previous bank
  # (if any) with freshly mined macros into a new WORM bank file, then run with the bank.
  MACRO_BANK_OUT="artifacts/arc_operator_bank_v147_${TS}.jsonl"
  MERGE_INS=()
  if [ -n "${MACRO_BANK_IN}" ]; then
    if [ ! -f "${MACRO_BANK_IN}" ]; then
      echo "[arc_loop] ERROR: MACRO_BANK_IN not found: ${MACRO_BANK_IN}"
      exit 2
    fi
    MERGE_INS+=(--in "${MACRO_BANK_IN}")
  fi
  MERGE_INS+=(--in "${MACROS_OUT}")

  echo
  echo "[arc_loop] updating operator bank: ${MACRO_BANK_OUT}"
  python3 scripts/arc_merge_operator_templates_v147.py \
    "${MERGE_INS[@]}" \
    --out "${MACRO_BANK_OUT}" \
    --min_support "${MERGED_MIN_SUPPORT}" \
    --min_len "${MACRO_MIN_LEN}" \
    --max_len "${MACRO_MAX_LEN}" \
    --max_operators "${MACRO_MAX_MACROS}"

  MACROS_EFFECTIVE="${MACRO_BANK_OUT}"
  echo "[arc_loop] next MACRO_BANK_IN: ${MACRO_BANK_OUT}"

  python3 scripts/arc_append_progress_v1.py \
    --progress "${PROGRESS_JSONL}" \
    --stage "operators_mined" \
    --run_dir "${OUT_BASE}_try1" \
    --omega_state "${OMEGA_STATE_IN}" \
    --concept_bank "${CONCEPT_BANK_IN}" \
    --macro_bank "${MACROS_EFFECTIVE}"

  OUT_BASE_MACRO="${OUT_BASE}_macros"
  echo
  echo "[arc_loop] macro run out_base=${OUT_BASE_MACRO}"
  echo "[arc_loop] tail macro: tail -f ${OUT_BASE_MACRO}_try1/progress.log"
  echo

  CONCEPT_ARGS_MACRO=()
  if [ "${CONCEPTS}" != "0" ] && [ -n "${CONCEPT_BANK_IN}" ]; then
    CONCEPT_ARGS_MACRO+=(--concept_templates "${CONCEPT_BANK_IN}")
  fi

  RUN_MACRO_CMD=(python3 scripts/run_arc_scalpel_v141.py)
  RUN_MACRO_CMD+=(--arc_root "${ARC_ROOT}")
  RUN_MACRO_CMD+=(--split training)
  RUN_MACRO_CMD+=(--limit "${LIMIT}")
  RUN_MACRO_CMD+=(--seed "${SEED}")
  RUN_MACRO_CMD+=(--tries "${TRIES}")
  RUN_MACRO_CMD+=(--jobs "${JOBS}")
  RUN_MACRO_CMD+=(--require_concept_call "${REQUIRE_CONCEPT_CALL}")
  RUN_MACRO_CMD+=(--task_timeout_s "${SCALE_TASK_TIMEOUT_S}")
  RUN_MACRO_CMD+=(--no_progress_timeout_s "${SCALE_NO_PROGRESS_TIMEOUT_S}")
  RUN_MACRO_CMD+=(--max_depth "${MAX_DEPTH}")
  RUN_MACRO_CMD+=(--max_programs "${MAX_PROGRAMS}")
  RUN_MACRO_CMD+=(--solution_cost_slack_bits "${SOLUTION_COST_SLACK_BITS}")
  RUN_MACRO_CMD+=(--macro_propose_max_depth "${SCALE_MACRO_PROPOSE_MAX_DEPTH}")
  RUN_MACRO_CMD+=(--macro_max_templates "${MACRO_MAX_TEMPLATES}")
  RUN_MACRO_CMD+=(--macro_max_instantiations "${MACRO_MAX_INSTANTIATIONS}")
  RUN_MACRO_CMD+=(--macro_max_branch_per_op "${MACRO_MAX_BRANCH_PER_OP}")
  if [ "${SCALE_DISABLE_REACHABILITY_PRUNING}" != "0" ]; then
    RUN_MACRO_CMD+=(--disable_reachability_pruning)
  fi
  if [ "${#EXTRA_ARGS[@]}" != "0" ]; then
    RUN_MACRO_CMD+=("${EXTRA_ARGS[@]}")
  fi
  if [ "${#SCALE_ARGS[@]}" != "0" ]; then
    RUN_MACRO_CMD+=("${SCALE_ARGS[@]}")
  fi
  if [ "${#CONCEPT_ARGS_MACRO[@]}" != "0" ]; then
    RUN_MACRO_CMD+=("${CONCEPT_ARGS_MACRO[@]}")
  fi
  if [ "${OMEGA}" != "0" ]; then
    RUN_MACRO_CMD+=(--omega)
    if [ -n "${OMEGA_STATE_IN}" ]; then
      if [ ! -f "${OMEGA_STATE_IN}" ]; then
        echo "[arc_loop] ERROR: OMEGA_STATE_IN not found: ${OMEGA_STATE_IN}"
        exit 2
      fi
      RUN_MACRO_CMD+=(--omega_state_in "${OMEGA_STATE_IN}")
    fi
  fi
  RUN_MACRO_CMD+=(--macro_templates "${MACROS_EFFECTIVE}")
  RUN_MACRO_CMD+=(--out_base "${OUT_BASE_MACRO}")
  "${RUN_MACRO_CMD[@]}"

  python3 scripts/arc_diag_v142_from_v141_run.py \
    --run_dir "${OUT_BASE_MACRO}_try1" \
    --out_path "artifacts/arc_diag_v142_${TS}_macros.md" \
    --out_json "artifacts/arc_diag_v142_${TS}_macros.json"

  echo
  echo "[arc_loop] macro summary: ${OUT_BASE_MACRO}_try1/summary.json"
  echo "[arc_loop] macro diag: artifacts/arc_diag_v142_${TS}_macros.md"

  python3 scripts/arc_append_progress_v1.py \
    --progress "${PROGRESS_JSONL}" \
    --stage "macro_run" \
    --run_dir "${OUT_BASE_MACRO}_try1" \
    --omega_state "${OMEGA_STATE_IN}" \
    --concept_bank "${CONCEPT_BANK_IN}" \
    --macro_bank "${MACROS_EFFECTIVE}"

  if [ "${OMEGA}" != "0" ]; then
    echo
    if [ "${OMEGA_COMBINE_BASE_MACRO}" != "0" ]; then
      COMBINED_EVENTS="${OUT_BASE_MACRO}_try1/omega_events_v2_combined_${TS}.jsonl"
      if [ -f "${COMBINED_EVENTS}" ]; then
        echo "[arc_loop] ERROR: combined events already exists: ${COMBINED_EVENTS}"
        exit 2
      fi
      cat "${OUT_BASE}_try1/omega_events_v2.jsonl" "${OUT_BASE_MACRO}_try1/omega_events_v2.jsonl" > "${COMBINED_EVENTS}"
      echo "[arc_loop] combined omega events: ${COMBINED_EVENTS}"

      OMEGA_OUT2="artifacts/omega_state_v2_${TS}_maxwell.json"
      PREV_ARGS_OMEGA2=()
      if [ -n "${OMEGA_STATE_IN}" ]; then
        PREV_ARGS_OMEGA2+=(--prev_state "${OMEGA_STATE_IN}")
      fi
      CONCEPT_ARGS_OMEGA2=()
      if [ -n "${CONCEPT_BANK_IN}" ]; then
        CONCEPT_ARGS_OMEGA2+=(--concept_bank "${CONCEPT_BANK_IN}")
      fi
      MACRO_ARGS_OMEGA2=()
      if [ -n "${MACROS_EFFECTIVE}" ]; then
        MACRO_ARGS_OMEGA2+=(--macro_bank "${MACROS_EFFECTIVE}")
      fi
      INDUCTION_ARGS_OMEGA2=()
      if [ -n "${INDUCTION_LOG_OUT}" ]; then
        if [ ! -f "${INDUCTION_LOG_OUT}" ]; then
          echo "[arc_loop] ERROR: INDUCTION_LOG_OUT not found for Ω update: ${INDUCTION_LOG_OUT}"
          exit 2
        fi
        INDUCTION_ARGS_OMEGA2+=(--induction_log "${INDUCTION_LOG_OUT}")
      fi

      python3 scripts/omega_update_v2.py \
        "${PREV_ARGS_OMEGA2[@]}" \
        --run_dir "${OUT_BASE_MACRO}_try1" \
        --events_path "${COMBINED_EVENTS}" \
        "${CONCEPT_ARGS_OMEGA2[@]}" \
        "${MACRO_ARGS_OMEGA2[@]}" \
        "${INDUCTION_ARGS_OMEGA2[@]}" \
        --base_max_depth "${MAX_DEPTH}" \
        --base_max_programs "${MAX_PROGRAMS}" \
        --require_concept_call_after_runs "${OMEGA_REQUIRE_CONCEPT_CALL_AFTER_RUNS}" \
        --require_promoted_concept_call_after_runs "${OMEGA_REQUIRE_PROMOTED_CONCEPT_CALL_AFTER_RUNS}" \
        --out "${OMEGA_OUT2}"
      OMEGA_STATE_IN="${OMEGA_OUT2}"
      echo "[arc_loop] next OMEGA_STATE_IN (maxwell): ${OMEGA_OUT2}"

      python3 scripts/arc_append_progress_v1.py \
        --progress "${PROGRESS_JSONL}" \
        --stage "omega_maxwell" \
        --run_dir "${OUT_BASE_MACRO}_try1" \
        --omega_state "${OMEGA_STATE_IN}" \
        --concept_bank "${CONCEPT_BANK_IN}" \
        --macro_bank "${MACROS_EFFECTIVE}"
    else
      OMEGA_OUT2="artifacts/omega_state_v2_${TS}_macro.json"
      CONCEPT_ARGS_OMEGA2=()
      if [ -n "${CONCEPT_BANK_IN}" ]; then
        CONCEPT_ARGS_OMEGA2+=(--concept_bank "${CONCEPT_BANK_IN}")
      fi
      MACRO_ARGS_OMEGA2=()
      if [ -n "${MACROS_EFFECTIVE}" ]; then
        MACRO_ARGS_OMEGA2+=(--macro_bank "${MACROS_EFFECTIVE}")
      fi
      python3 scripts/omega_update_v2.py \
        --prev_state "${OMEGA_STATE_IN}" \
        --run_dir "${OUT_BASE_MACRO}_try1" \
        "${CONCEPT_ARGS_OMEGA2[@]}" \
        "${MACRO_ARGS_OMEGA2[@]}" \
        --base_max_depth "${MAX_DEPTH}" \
        --base_max_programs "${MAX_PROGRAMS}" \
        --require_concept_call_after_runs "${OMEGA_REQUIRE_CONCEPT_CALL_AFTER_RUNS}" \
        --require_promoted_concept_call_after_runs "${OMEGA_REQUIRE_PROMOTED_CONCEPT_CALL_AFTER_RUNS}" \
        --out "${OMEGA_OUT2}"
      OMEGA_STATE_IN="${OMEGA_OUT2}"
      echo "[arc_loop] next OMEGA_STATE_IN (macro): ${OMEGA_OUT2}"

      python3 scripts/arc_append_progress_v1.py \
        --progress "${PROGRESS_JSONL}" \
        --stage "omega_macro" \
        --run_dir "${OUT_BASE_MACRO}_try1" \
        --omega_state "${OMEGA_STATE_IN}" \
        --concept_bank "${CONCEPT_BANK_IN}" \
        --macro_bank "${MACROS_EFFECTIVE}"
    fi
  fi

  if [ "${CONCEPTS}" != "0" ] && [ -n "${CONCEPT_BANK_IN}" ]; then
    # Bump concept support based on actual concept_call usage in the scale (macro) run.
    # This makes the persistent concept bank evolve under performance, not only induction.
    CONCEPT_BANK_BUMPED2="artifacts/arc_concept_bank_v146_${TS}_macro_bumped.jsonl"
    echo
    echo "[arc_loop] bumping concept support (macro run): ${CONCEPT_BANK_BUMPED2}"
    python3 scripts/arc_bump_concept_support_v146.py \
      --concept_bank_in "${CONCEPT_BANK_IN}" \
      --run_dir "${OUT_BASE_MACRO}_try1" \
      --out "${CONCEPT_BANK_BUMPED2}"
    CONCEPT_BANK_IN="${CONCEPT_BANK_BUMPED2}"
    echo "[arc_loop] next CONCEPT_BANK_IN (macro bumped): ${CONCEPT_BANK_BUMPED2}"

    # Mine additional concept templates from the macro run trace (plan-first coverage).
    # Even without abstraction pressure, the macro run can expose useful multi-op closure chains
    # that are not seen in the baseline/pressure run, especially after macro reshaping.
    CONCEPTS_OUT_MACRO="artifacts/arc_concept_templates_v146_${TS}_macros.jsonl"
    echo
    echo "[arc_loop] mining concepts (macro run): ${CONCEPTS_OUT_MACRO}"
    python3 scripts/arc_induce_concept_templates_v146.py \
      --run_dir "${OUT_BASE_MACRO}_try1" \
      --out "${CONCEPTS_OUT_MACRO}" \
      --min_support "${CONCEPT_MIN_SUPPORT}" \
      --max_concepts "${CONCEPT_MAX_CONCEPTS}"

    CSG_OUT_MACRO_V152="artifacts/arc_concept_csg_templates_v152_${TS}_macros.jsonl"
    echo
    echo "[arc_loop] mining concept CSGs (v152 prefixes, macro run): ${CSG_OUT_MACRO_V152}"
    python3 scripts/arc_induce_concept_csg_templates_v152.py \
      --tasks_jsonl "${OUT_BASE_MACRO}_try1/input/arc_tasks_canonical_v141.jsonl" \
      --traces "${OUT_BASE_MACRO}_try1/trace_candidates.jsonl" \
      --out "${CSG_OUT_MACRO_V152}" \
      --min_len "${CSG_MIN_LEN}" \
      --max_len "${CSG_MAX_LEN}" \
      --min_support "${CSG_MIN_SUPPORT}" \
      --support_slack_every "${CSG_SUPPORT_SLACK_EVERY}" \
      --max_templates "${CSG_MAX_TEMPLATES}" \
      --max_loss_shape "${CSG_MAX_LOSS_SHAPE}" \
      --max_loss_cells "${CSG_MAX_LOSS_CELLS}" \
      --max_candidates_per_task "${CSG_MAX_CANDIDATES_PER_TASK}" \
      --min_loss_drop "${CSG_MIN_LOSS_DROP}" \
      --require_last_writes_grid "${CSG_REQUIRE_LAST_WRITES_GRID}"

    CSG_OUT_MACRO_V153="artifacts/arc_concept_csg_templates_v153_${TS}_macros.jsonl"
    echo
    echo "[arc_loop] mining concept CSGs (v153 windows, macro run): ${CSG_OUT_MACRO_V153}"
    python3 scripts/arc_induce_concept_csg_templates_v153.py \
      --tasks_jsonl "${OUT_BASE_MACRO}_try1/input/arc_tasks_canonical_v141.jsonl" \
      --traces "${OUT_BASE_MACRO}_try1/trace_candidates.jsonl" \
      --out "${CSG_OUT_MACRO_V153}" \
      --min_len "${CSG_MIN_LEN}" \
      --max_len "${CSG_MAX_LEN}" \
      --min_support "${CSG_MIN_SUPPORT}" \
      --support_slack_every "${CSG_SUPPORT_SLACK_EVERY}" \
      --max_templates "${CSG_MAX_TEMPLATES}" \
      --max_loss_shape "${CSG_MAX_LOSS_SHAPE}" \
      --max_loss_cells "${CSG_MAX_LOSS_CELLS}" \
      --max_candidates_per_task "${CSG_MAX_CANDIDATES_PER_TASK}" \
      --min_loss_drop "${CSG_MIN_LOSS_DROP}" \
      --require_last_writes_grid "${CSG_REQUIRE_LAST_WRITES_GRID}"

    CONCEPT_BANK_OUT2="artifacts/arc_concept_bank_v150_${TS}_macros.jsonl"
    echo
    echo "[arc_loop] updating concept bank (macro run): ${CONCEPT_BANK_OUT2}"
    python3 scripts/arc_merge_concept_bank_v150.py \
      --in "${CONCEPT_BANK_IN}" \
      --in "${CONCEPTS_OUT_MACRO}" \
      --in "${CSG_OUT_MACRO_V152}" \
      --in "${CSG_OUT_MACRO_V153}" \
      --out "${CONCEPT_BANK_OUT2}"

    CONCEPT_BANK_IN="${CONCEPT_BANK_OUT2}"
    echo "[arc_loop] next CONCEPT_BANK_IN (macro merged): ${CONCEPT_BANK_OUT2}"
  fi
fi

if [ "${DEEP}" != "0" ]; then
  echo
  echo "[arc_loop] deep subset (v144) enabled"

  BASE_FOR_DEEP="${OUT_BASE}_try1"
  if [ "${MACROS}" != "0" ]; then
    BASE_FOR_DEEP="${OUT_BASE_MACRO}_try1"
  fi

  DEEP_OUT_BASE="artifacts/arc_pt2_v144_deep_${TS}"
  echo "[arc_loop] deep out_base=${DEEP_OUT_BASE}"
  echo "[arc_loop] tail deep: tail -f ${DEEP_OUT_BASE}_try1/progress.log"

  python3 scripts/run_arc_deep_subset_v144.py \
    --base_run_dir "${BASE_FOR_DEEP}" \
    --out_base "${DEEP_OUT_BASE}" \
    --jobs "${JOBS}" \
    --limit "${DEEP_LIMIT}" \
    --max_depth "${DEEP_MAX_DEPTH}" \
    --max_programs "${DEEP_MAX_PROGRAMS}" \
    --solution_cost_slack_bits "${SOLUTION_COST_SLACK_BITS}" \
    --timeout_s "${DEEP_TIMEOUT_S}" \
    --seed_k "${DEEP_SEED_K}"

  DEEP_SUMMARY="${DEEP_OUT_BASE}_try1/summary.json"
  if [ ! -f "${DEEP_SUMMARY}" ]; then
    echo "[arc_loop] ERROR: deep summary missing: ${DEEP_SUMMARY}"
    exit 2
  fi
  DEEP_TOTAL="$(python3 - "${DEEP_SUMMARY}" <<'PY'
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    d = json.load(f)
print(int(d.get("tasks_total") or 0))
PY
)"
  if [ "${DEEP_TOTAL}" = "0" ]; then
    echo
    echo "[arc_loop] deep subset selected 0 tasks; skipping deep macro stages"
    echo "[arc_loop] deep summary: ${DEEP_SUMMARY}"
    exit 0
  fi

  DEEP_MACROS_OUT="artifacts/arc_operator_templates_v147_${TS}_deep.jsonl"
  echo
  echo "[arc_loop] mining deep operators (born-from-failure): ${DEEP_MACROS_OUT}"
  python3 scripts/arc_mine_operator_templates_v147.py \
    --run_dir "${DEEP_OUT_BASE}_try1" \
    --origin_run_dir "${BASE_FOR_DEEP}" \
    --out "${DEEP_MACROS_OUT}" \
    --include_solved_from_failure \
    --min_len "${MACRO_MIN_LEN}" \
    --max_len "${MACRO_MAX_LEN}" \
    --min_support "${MACRO_MIN_SUPPORT}" \
    --max_operators "${MACRO_MAX_MACROS}" \
    --trace_max_programs_per_task 40 \
    --trace_max_loss_shape 0 \
    --trace_max_loss_cells "${MACRO_TRACE_MAX_LOSS_CELLS}"

  # Merge macro templates (base + deep) deterministically.
  if [ "${MACROS}" != "0" ]; then
    MERGED_MACROS_OUT="artifacts/arc_operator_bank_v147_${TS}_merged.jsonl"
    echo
    echo "[arc_loop] merging operators: ${MERGED_MACROS_OUT}"
    MERGE_INS=()
    if [ -n "${MACRO_BANK_IN}" ]; then
      MERGE_INS+=(--in "${MACRO_BANK_IN}")
    fi
    MERGE_INS+=(--in "${MACROS_OUT}")
    MERGE_INS+=(--in "${DEEP_MACROS_OUT}")

    python3 scripts/arc_merge_operator_templates_v147.py \
      "${MERGE_INS[@]}" \
      --out "${MERGED_MACROS_OUT}" \
      --min_support "${MERGED_MIN_SUPPORT}" \
      --min_len "${MACRO_MIN_LEN}" \
      --max_len "${MACRO_MAX_LEN}" \
      --max_operators "${MACRO_MAX_MACROS}"
    echo "[arc_loop] next MACRO_BANK_IN (deep merged): ${MERGED_MACROS_OUT}"

    OUT_BASE_MERGED="${OUT_BASE}_macros_deep"
    echo
    echo "[arc_loop] merged macro run out_base=${OUT_BASE_MERGED}"
    echo "[arc_loop] tail merged: tail -f ${OUT_BASE_MERGED}_try1/progress.log"
    echo

    RUN_DEEP_MACRO_CMD=(python3 scripts/run_arc_scalpel_v141.py)
    RUN_DEEP_MACRO_CMD+=(--arc_root "${ARC_ROOT}")
    RUN_DEEP_MACRO_CMD+=(--split training)
    RUN_DEEP_MACRO_CMD+=(--limit "${LIMIT}")
    RUN_DEEP_MACRO_CMD+=(--seed "${SEED}")
    RUN_DEEP_MACRO_CMD+=(--tries "${TRIES}")
    RUN_DEEP_MACRO_CMD+=(--jobs "${JOBS}")
    RUN_DEEP_MACRO_CMD+=(--task_timeout_s "${SCALE_TASK_TIMEOUT_S}")
    RUN_DEEP_MACRO_CMD+=(--no_progress_timeout_s "${SCALE_NO_PROGRESS_TIMEOUT_S}")
    RUN_DEEP_MACRO_CMD+=(--max_depth "${MAX_DEPTH}")
    RUN_DEEP_MACRO_CMD+=(--max_programs "${MAX_PROGRAMS}")
    RUN_DEEP_MACRO_CMD+=(--solution_cost_slack_bits "${SOLUTION_COST_SLACK_BITS}")
    if [ "${#EXTRA_ARGS[@]}" != "0" ]; then
      RUN_DEEP_MACRO_CMD+=("${EXTRA_ARGS[@]}")
    fi
    if [ "${#PRESSURE_ARGS[@]}" != "0" ]; then
      RUN_DEEP_MACRO_CMD+=("${PRESSURE_ARGS[@]}")
    fi
    if [ "${OMEGA}" != "0" ]; then
      RUN_DEEP_MACRO_CMD+=(--omega)
      if [ -n "${OMEGA_STATE_IN}" ]; then
        if [ ! -f "${OMEGA_STATE_IN}" ]; then
          echo "[arc_loop] ERROR: OMEGA_STATE_IN not found: ${OMEGA_STATE_IN}"
          exit 2
        fi
        RUN_DEEP_MACRO_CMD+=(--omega_state_in "${OMEGA_STATE_IN}")
      fi
    fi
    RUN_DEEP_MACRO_CMD+=(--macro_templates "${MERGED_MACROS_OUT}")
    if [ -n "${CONCEPT_BANK_IN}" ]; then
      RUN_DEEP_MACRO_CMD+=(--concept_templates "${CONCEPT_BANK_IN}")
    fi
    RUN_DEEP_MACRO_CMD+=(--out_base "${OUT_BASE_MERGED}")
    "${RUN_DEEP_MACRO_CMD[@]}"

    python3 scripts/arc_diag_v142_from_v141_run.py \
      --run_dir "${OUT_BASE_MERGED}_try1" \
      --out_path "artifacts/arc_diag_v142_${TS}_macros_deep.md" \
      --out_json "artifacts/arc_diag_v142_${TS}_macros_deep.json"

    echo
    echo "[arc_loop] merged macro summary: ${OUT_BASE_MERGED}_try1/summary.json"
    echo "[arc_loop] merged macro diag: artifacts/arc_diag_v142_${TS}_macros_deep.md"

    if [ "${CONCEPTS}" != "0" ] && [ -n "${CONCEPT_BANK_IN}" ]; then
      # Bump concept support based on actual concept_call usage in the merged macro run.
      CONCEPT_BANK_BUMPED3="artifacts/arc_concept_bank_v146_${TS}_macros_deep_bumped.jsonl"
      echo
      echo "[arc_loop] bumping concept support (merged macro run): ${CONCEPT_BANK_BUMPED3}"
      python3 scripts/arc_bump_concept_support_v146.py \
        --concept_bank_in "${CONCEPT_BANK_IN}" \
        --run_dir "${OUT_BASE_MERGED}_try1" \
        --out "${CONCEPT_BANK_BUMPED3}"
      CONCEPT_BANK_IN="${CONCEPT_BANK_BUMPED3}"
      echo "[arc_loop] next CONCEPT_BANK_IN (macros_deep bumped): ${CONCEPT_BANK_BUMPED3}"

      CONCEPTS_OUT_MACRO_DEEP="artifacts/arc_concept_templates_v146_${TS}_macros_deep.jsonl"
      echo
      echo "[arc_loop] mining concepts (merged macro run): ${CONCEPTS_OUT_MACRO_DEEP}"
      python3 scripts/arc_induce_concept_templates_v146.py \
        --run_dir "${OUT_BASE_MERGED}_try1" \
        --out "${CONCEPTS_OUT_MACRO_DEEP}" \
        --min_support "${CONCEPT_MIN_SUPPORT}" \
        --max_concepts "${CONCEPT_MAX_CONCEPTS}"

      CSG_OUT_MACRO_DEEP="artifacts/arc_concept_csg_templates_v153_${TS}_macros_deep.jsonl"
      echo
      echo "[arc_loop] mining concept CSGs (v153, merged macro run): ${CSG_OUT_MACRO_DEEP}"
      python3 scripts/arc_induce_concept_csg_templates_v153.py \
        --tasks_jsonl "${OUT_BASE_MERGED}_try1/input/arc_tasks_canonical_v141.jsonl" \
        --traces "${OUT_BASE_MERGED}_try1/trace_candidates.jsonl" \
        --out "${CSG_OUT_MACRO_DEEP}" \
        --min_len "${CSG_MIN_LEN}" \
        --max_len "${CSG_MAX_LEN}" \
        --min_support "${CSG_MIN_SUPPORT}" \
        --support_slack_every "${CSG_SUPPORT_SLACK_EVERY}" \
        --max_templates "${CSG_MAX_TEMPLATES}" \
        --max_loss_shape "${CSG_MAX_LOSS_SHAPE}" \
        --max_loss_cells "${CSG_MAX_LOSS_CELLS}" \
        --max_candidates_per_task "${CSG_MAX_CANDIDATES_PER_TASK}" \
        --min_loss_drop "${CSG_MIN_LOSS_DROP}" \
        --require_last_writes_grid "${CSG_REQUIRE_LAST_WRITES_GRID}"

      CONCEPT_BANK_OUT3="artifacts/arc_concept_bank_v150_${TS}_macros_deep.jsonl"
      echo
      echo "[arc_loop] updating concept bank (merged macro run): ${CONCEPT_BANK_OUT3}"
      python3 scripts/arc_merge_concept_bank_v150.py \
        --in "${CONCEPT_BANK_IN}" \
        --in "${CONCEPTS_OUT_MACRO_DEEP}" \
        --in "${CSG_OUT_MACRO_DEEP}" \
        --out "${CONCEPT_BANK_OUT3}"

      CONCEPT_BANK_IN="${CONCEPT_BANK_OUT3}"
      echo "[arc_loop] next CONCEPT_BANK_IN (macros_deep merged): ${CONCEPT_BANK_OUT3}"
    fi

    if [ "${OMEGA}" != "0" ]; then
      echo
      OMEGA_OUT3="artifacts/omega_state_v2_${TS}_deep.json"
      CONCEPT_ARGS_OMEGA3=()
      if [ -n "${CONCEPT_BANK_IN}" ]; then
        CONCEPT_ARGS_OMEGA3+=(--concept_bank "${CONCEPT_BANK_IN}")
      fi
      MACRO_ARGS_OMEGA3=()
      if [ -n "${MERGED_MACROS_OUT}" ]; then
        MACRO_ARGS_OMEGA3+=(--macro_bank "${MERGED_MACROS_OUT}")
      fi
      python3 scripts/omega_update_v2.py \
        --prev_state "${OMEGA_STATE_IN}" \
        --run_dir "${OUT_BASE_MERGED}_try1" \
        "${CONCEPT_ARGS_OMEGA3[@]}" \
        "${MACRO_ARGS_OMEGA3[@]}" \
        --base_max_depth "${MAX_DEPTH}" \
        --base_max_programs "${MAX_PROGRAMS}" \
        --require_concept_call_after_runs "${OMEGA_REQUIRE_CONCEPT_CALL_AFTER_RUNS}" \
        --require_promoted_concept_call_after_runs "${OMEGA_REQUIRE_PROMOTED_CONCEPT_CALL_AFTER_RUNS}" \
        --out "${OMEGA_OUT3}"
      OMEGA_STATE_IN="${OMEGA_OUT3}"
      echo "[arc_loop] next OMEGA_STATE_IN (deep): ${OMEGA_OUT3}"
    fi
  else
    echo "[arc_loop] NOTE: DEEP requires MACROS!=0 to run merged macro stage (skipping)"
  fi
fi
