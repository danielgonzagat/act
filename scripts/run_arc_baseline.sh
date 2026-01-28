#!/usr/bin/env bash
set -euo pipefail

# WORM baseline runner (ACT vs ARC pt2)
# - Uses the latest scripts/run_arc_scalpel_v*.py
# - Runs training+evaluation with tries=2 (determinism) and jobs=8
# - Writes simple hashes + reason diagnostics

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

export PYTHONPYCACHEPREFIX="${ROOT}/.pycache"

V="$(ls scripts/run_arc_scalpel_v*.py | sed -E 's/.*_v([0-9]+)\\.py/\\1/' | sort -n | tail -1)"
echo "[baseline] runner=v${V}"

ARC_ROOT="${ARC_ROOT:-ARC-AGI}"
JOBS="${JOBS:-8}"
TRIES="${TRIES:-2}"
SEED="${SEED:-0}"
MAX_PROGRAMS="${MAX_PROGRAMS:-2000}"
MAX_DEPTH="${MAX_DEPTH:-4}"
SOLUTION_COST_SLACK_BITS="${SOLUTION_COST_SLACK_BITS:-16}"
LIMIT="${LIMIT:-20}"  # set LIMIT=0 for full

TS="$(date +%Y%m%d_%H%M%S)"
OUT_TR_BASE="results/arc_baseline_v${V}_training_limit${LIMIT}_${TS}"
OUT_EV_BASE="results/arc_baseline_v${V}_evaluation_limit${LIMIT}_${TS}"

echo "[baseline] ARC_ROOT=${ARC_ROOT}"
echo "[baseline] JOBS=${JOBS} TRIES=${TRIES} SEED=${SEED}"
echo "[baseline] MAX_DEPTH=${MAX_DEPTH} MAX_PROGRAMS=${MAX_PROGRAMS}"
echo "[baseline] LIMIT=${LIMIT}"
echo

python3 "scripts/run_arc_scalpel_v${V}.py" \
  --arc_root "${ARC_ROOT}" \
  --split training \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --tries "${TRIES}" \
  --jobs "${JOBS}" \
  --task_timeout_s 0 \
  --max_depth "${MAX_DEPTH}" \
  --max_programs "${MAX_PROGRAMS}" \
  --solution_cost_slack_bits "${SOLUTION_COST_SLACK_BITS}" \
  --out_base "${OUT_TR_BASE}"

python3 "scripts/run_arc_scalpel_v${V}.py" \
  --arc_root "${ARC_ROOT}" \
  --split evaluation \
  --limit "${LIMIT}" \
  --seed "${SEED}" \
  --tries "${TRIES}" \
  --jobs "${JOBS}" \
  --task_timeout_s 0 \
  --max_depth "${MAX_DEPTH}" \
  --max_programs "${MAX_PROGRAMS}" \
  --solution_cost_slack_bits "${SOLUTION_COST_SLACK_BITS}" \
  --out_base "${OUT_EV_BASE}"

GIT_SHA="$(git rev-parse HEAD 2>/dev/null || echo "NO_GIT")"

for base in "${OUT_TR_BASE}" "${OUT_EV_BASE}"; do
  for t in 1 2; do
    d="${base}_try${t}"
    if [ -d "${d}" ]; then
      printf "%s\n" "${GIT_SHA}" > "${d}/GIT_SHA.txt"
      python3 scripts/diag_reasons.py "${d}/per_task_manifest.jsonl" | tee "${d}/diag_reasons.txt" >/dev/null
      python3 - "${d}" <<'PY'
import hashlib, pathlib, sys
run = pathlib.Path(sys.argv[1])
def sha(p: pathlib.Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()
out = []
for name in ["summary.json", "per_task_manifest.jsonl", "outputs_manifest.json"]:
    p = run / name
    if p.exists():
        out.append(f"{sha(p)}  {name}")
(run / "SHA256SUMS.txt").write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
print("WROTE", run / "SHA256SUMS.txt")
PY
    fi
  done
done

echo
echo "[baseline] DONE"
echo "  TRAIN: ${OUT_TR_BASE}_try1"
echo "  EVAL : ${OUT_EV_BASE}_try1"

