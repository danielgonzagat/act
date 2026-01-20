# STATUS PACK V136 — ARC-AGI-1 OFFICIAL RUN (fix1)

## Scope
- V136 anti-budget-exhaustion: formal reachability pruning + state-dominance pruning (no task_id/path/family logic; deterministic; fail-closed).
- Run `scripts/run_arc_scalpel_v136.py` on ARC-AGI-1 `training` and `evaluation` with `--limit 20 --seed 0`.

## Checklist
- PASS `py_compile`
- PASS `unittest -v`
- PASS training run: `determinism_ok=true`, `isolation_ok=true`, `SEARCH_BUDGET_EXCEEDED=0`
- PASS evaluation run: `determinism_ok=true`, `isolation_ok=true`, `SEARCH_BUDGET_EXCEEDED=0`
- PASS `verify_freeze` (ok=true, warnings=[])

## Commands (copy/paste)
```bash
set -o pipefail
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v136.py \
  --arc_root ARC-AGI --split training --limit 20 --seed 0 \
  --out_base results/run_arc_scalpel_v136_arc_agi1_training_official_fix3 \
  | tee results/smoke_arc_scalpel_v136_arc_agi1_training_official_fix3_try1.json

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v136.py \
  --arc_root ARC-AGI --split evaluation --limit 20 --seed 0 \
  --out_base results/run_arc_scalpel_v136_arc_agi1_evaluation_official_fix3 \
  | tee results/smoke_arc_scalpel_v136_arc_agi1_evaluation_official_fix3_try1.json

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V136_ARC_AGI1_OFFICIAL_RUN_fix1.json \
  | tee results/verify_freeze_v136_arc_agi1_official_run_fix1_try1.json
```

## Results (raw)
### Training (`ARC-AGI/data/training`, limit=20)
- `tasks_total=20, solved=1, unknown=0, failed=19, solve_rate=0.05`
- `failure_counts={"MISSING_OPERATOR":19}`
- `determinism_ok=true, isolation_ok=true`
- `summary_sha256=0144dad1f90be97c7f3e420558c88c89d837d1bcbe7c67954bd01e7e92546cf0`
- `outputs_manifest_sig=5f0ecb1ad585cd8df8cf0cd1d80d40084ff67bdcf180cadbe3c50cfb79610e14`

### Evaluation (`ARC-AGI/data/evaluation`, limit=20)
- `tasks_total=20, solved=0, unknown=0, failed=20, solve_rate=0.0`
- `failure_counts={"MISSING_OPERATOR":20}`
- `determinism_ok=true, isolation_ok=true`
- `summary_sha256=d04c010c75d1362620dbc039f21c8b009d799c5735aecbf84a28299fefde1532`
- `outputs_manifest_sig=6c4c58349c80e9d874781d2a197178cf9f074e83a10809e302418eedb929a9fb`

## Runs (WORM)
- Training: `results/run_arc_scalpel_v136_arc_agi1_training_official_fix3_try1/`, `results/run_arc_scalpel_v136_arc_agi1_training_official_fix3_try2/`
- Evaluation: `results/run_arc_scalpel_v136_arc_agi1_evaluation_official_fix3_try1/`, `results/run_arc_scalpel_v136_arc_agi1_evaluation_official_fix3_try2/`

## verify_freeze
- `results/verify_freeze_v136_arc_agi1_official_run_fix1_try1.json` -> `ok=true, warnings=[]`

## SHA256 (key artifacts)
- `patches/v136_reachability_pruning_fix1.diff`: `1ba535b0b401177162af5ab103142f15ff00a0508b772f7b6b0c98d5b3931e84`
- `results/smoke_arc_scalpel_v136_arc_agi1_training_official_fix3_try1.json`: `3c9e9f7eb55b2983ceeeb1671da4ef4dc6d9282299497a6000f961d2fcda3872`
- `results/smoke_arc_scalpel_v136_arc_agi1_evaluation_official_fix3_try1.json`: `988b0a46548d9b465e94440b9c52ec9cb63cde53144ff8868f6179f51d2ee944`
- Training `results/run_arc_scalpel_v136_arc_agi1_training_official_fix3_try1/summary.json`: `e85328ddb9e5348dd82c016bd561ef6f43ac950aeeefadf88ea1e7d4f70c6df5`
- Training `results/run_arc_scalpel_v136_arc_agi1_training_official_fix3_try1/eval.json`: `25516e4b3c81de451d15d3f574fc19fc02ecd1b089e7435de03e56f47caeaea0`
- Evaluation `results/run_arc_scalpel_v136_arc_agi1_evaluation_official_fix3_try1/summary.json`: `41a8110c630e92cda4ae9123d270da09c0cb56756652710fbcfd22a1c95ddcca`
- Evaluation `results/run_arc_scalpel_v136_arc_agi1_evaluation_official_fix3_try1/eval.json`: `7f36c9b16977c547ce199126c6d82e19809f75d3600b1711db3aad9c45da67f7`

