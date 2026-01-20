# STATUS PACK V135 — ARC-AGI-1 OFFICIAL RUN (fix3)

## Scope
- Run `scripts/run_arc_scalpel_v135.py` on ARC-AGI-1 `training` and `evaluation` splits with `--limit 20 --seed 0`.
- No solver changes (V135 intact). Determinism try1==try2 required.

## Commands (copy/paste)
```bash
set -o pipefail
cd /Users/danielpenin/act

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v135.py \
  --arc_root ARC-AGI --split training --limit 20 --seed 0 \
  --out_base results/run_arc_scalpel_v135_arc_agi1_training_official_fix3 \
  | tee results/smoke_arc_scalpel_v135_arc_agi1_training_official_fix3_try1.json

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v135.py \
  --arc_root ARC-AGI --split evaluation --limit 20 --seed 0 \
  --out_base results/run_arc_scalpel_v135_arc_agi1_evaluation_official_fix3 \
  | tee results/smoke_arc_scalpel_v135_arc_agi1_evaluation_official_fix3_try1.json

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V135_ARC_AGI1_OFFICIAL_RUN_fix3.json \
  | tee results/verify_freeze_v135_arc_agi1_official_run_fix3_try1.json
```

## Results (raw)
### Training (`ARC-AGI/data/training`, limit=20)
- `tasks_total=20, solved=0, unknown=0, failed=20, solve_rate=0.0`
- `failure_counts={"SEARCH_BUDGET_EXCEEDED":20}`
- `determinism_ok=true, isolation_ok=true`
- `summary_sha256=6077d3a60c5ecdd13cd25acc8fea781c4a396b61924f03de99738c1cd8049a70`
- `outputs_manifest_sig=3a21c17ed0fb641e1f9c8ffb7efb524cfca2888d1cdc91335d128ccf8f2718b5`

### Evaluation (`ARC-AGI/data/evaluation`, limit=20)
- `tasks_total=20, solved=0, unknown=0, failed=20, solve_rate=0.0`
- `failure_counts={"MISSING_OPERATOR":1,"SEARCH_BUDGET_EXCEEDED":19}`
- `determinism_ok=true, isolation_ok=true`
- `summary_sha256=f0c742b40fd034e1f9238becfe5e85d69a957ae1a5c297d85946400165d45dd4`
- `outputs_manifest_sig=64816a29aaab8bad10aae8d48646727b37453f77b9ab81bd73003396188acd81`

## Runs (WORM)
- Training: `results/run_arc_scalpel_v135_arc_agi1_training_official_fix3_try1/`, `results/run_arc_scalpel_v135_arc_agi1_training_official_fix3_try2/`
- Evaluation: `results/run_arc_scalpel_v135_arc_agi1_evaluation_official_fix3_try1/`, `results/run_arc_scalpel_v135_arc_agi1_evaluation_official_fix3_try2/`

## verify_freeze
- `results/verify_freeze_v135_arc_agi1_official_run_fix3_try1.json` -> `ok=true, warnings=[]`

## SHA256 (key artifacts)
- `patches/v135_overlay_self_translate.diff`: `222ddb3fc4a647a9ce0af5eceb9086173dc7c556f9b29a981b267241032eab74`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V135_ARC_AGI1_OFFICIAL_RUN_fix3.json`: `cfc90dea61fe75c0c74f5546e559a784f9fec1fbec5d2060efdd5aebc749dd38`
- `results/verify_freeze_v135_arc_agi1_official_run_fix3_try1.json`: `e75fde9b469a1404c8165da678e56ee8839e5d32ed7b936167a81bf700c565ab`
- `results/smoke_arc_scalpel_v135_arc_agi1_training_official_fix3_try1.json`: `a15002d1550b1b1b553f61c07d715566b3f70fc1f25333efc55c62246c25eafc`
- `results/smoke_arc_scalpel_v135_arc_agi1_evaluation_official_fix3_try1.json`: `32d16fd6042f11aa923886309e92f26b099a5f3bd96034818494ce0ab1457e6e`
- Training `results/run_arc_scalpel_v135_arc_agi1_training_official_fix3_try1/summary.json`: `cfa93a0acd51b9be28e6b919dff3349bf82abb347f97915dbe7a6a1f6ec606a8`
- Training `results/run_arc_scalpel_v135_arc_agi1_training_official_fix3_try1/eval.json`: `999be92a456e43e02939f762210d8b0dc3896a808d38c4f13723822b6c65fd0f`
- Evaluation `results/run_arc_scalpel_v135_arc_agi1_evaluation_official_fix3_try1/summary.json`: `ef17945b744584d9094247a5f9d6c3304a2fcd6d7caaafb82194e2dd60847d87`
- Evaluation `results/run_arc_scalpel_v135_arc_agi1_evaluation_official_fix3_try1/eval.json`: `8e4450da0b75bc98a4beb9a97cc398e2960b65beeb6f6c5164ba40b530b46793`

