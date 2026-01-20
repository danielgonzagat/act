# STATUS_PACK_V138_ARC_AGI1_OFFICIAL_RUN_fix1

## Summary
- Adds ARC protocol-correct scoring to the scalpel runner: test outputs are used only for post-hoc scoring (never for inference), and each test input is allowed up to 2 guesses (ARC-style).
- Keeps solver fail-closed: ambiguous minimal solutions remain `UNKNOWN`; the solver exposes a deterministic `predicted_grids` list so the runner can score multiple attempts without “choosing”.
- Preserves determinism (try1==try2 by `summary_sha256` + `outputs_manifest_sig`) and isolation (repo snapshot before==after excluding run_dir).

## Checklist
- PASS `py_compile`
- PASS `unittest -v`
- PASS scalpel sample+synth (determinism_ok=true, isolation_ok=true)
- PASS ARC-AGI smoke (training/evaluation, limit=20) (determinism_ok=true, isolation_ok=true)

## Results
- sample (data/arc_v129_sample): tasks_total=11 solved=10 unknown=0 failed=1 failure_counts={"MISSING_OPERATOR":1}
- synth (data/gridworld_v129_synth): tasks_total=6 solved=6 unknown=0 failed=0 failure_counts={}
- ARC-AGI smoke (limit=20 seed=0 max_programs=4000):
  - training: tasks_total=20 solved=1 unknown=0 failed=19 failure_counts={"MISSING_OPERATOR":10,"SEARCH_BUDGET_EXCEEDED":9}
- ARC-AGI smoke (limit=20 seed=0 max_programs=500):
  - training: tasks_total=20 solved=0 unknown=0 failed=20 failure_counts={"SEARCH_BUDGET_EXCEEDED":20}
  - evaluation: tasks_total=20 solved=0 unknown=0 failed=20 failure_counts={"MISSING_OPERATOR":1,"SEARCH_BUDGET_EXCEEDED":19}

## Commands
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v138.py --arc_root data/arc_v129_sample --split sample --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v138_arc_v129_sample_fix3 | tee results/smoke_arc_scalpel_v138_arc_v129_sample_fix3_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v138.py --arc_root data/gridworld_v129_synth --split synth --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v138_gridworld_v129_synth_fix2 | tee results/smoke_arc_scalpel_v138_gridworld_v129_synth_fix2_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v138.py --arc_root ARC-AGI --split training --limit 20 --seed 0 --out_base results/run_arc_scalpel_v138_arc_agi1_training_official_fix1 | tee results/smoke_arc_scalpel_v138_arc_agi1_training_official_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v138.py --arc_root ARC-AGI --split training --limit 20 --seed 0 --max_programs 500 --out_base results/run_arc_scalpel_v138_arc_agi1_training_official_fix2 | tee results/smoke_arc_scalpel_v138_arc_agi1_training_official_fix2_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v138.py --arc_root ARC-AGI --split evaluation --limit 20 --seed 0 --max_programs 500 --out_base results/run_arc_scalpel_v138_arc_agi1_evaluation_official_fix3 | tee results/smoke_arc_scalpel_v138_arc_agi1_evaluation_official_fix3_try1.json`

## Artifacts (sha256)
- patch: `patches/v138_arc_protocol_scoring_fix1.diff` sha256=a10c4a74f4b42152222db4687f48e8a45e6e25c7d52f987900506e9c664c218f
- pre-diag report: `results/diag/ARC_DIAG_REPORT_v138_PRE_fix1_try1.md` sha256=9ea6bb133b26400985ecdfda42f12732eae5ad5ecc38119eacecd8e1f90e3b27
- `results/smoke_arc_scalpel_v138_arc_v129_sample_fix3_try1.json` sha256=1c7509c161dc4ec8d9c84fc6423bf39305e2bc4912663c88a28e7a7b2895ff12
- `results/smoke_arc_scalpel_v138_gridworld_v129_synth_fix2_try1.json` sha256=f6279ad4394810b9a4844d05f38422a5db80797a74b2cecf0679a0d331d6fbaa
- `results/smoke_arc_scalpel_v138_arc_agi1_training_official_fix1_try1.json` sha256=f3aba41763b03b36b70fa8292071d2d8d02e67d933b86acff666995c702020a9
- `results/smoke_arc_scalpel_v138_arc_agi1_training_official_fix2_try1.json` sha256=4ea86fe2056c52af2fac14c6cf450ef459d236a4480aaafb6a084b799067ff86
- `results/smoke_arc_scalpel_v138_arc_agi1_evaluation_official_fix3_try1.json` sha256=8bf9d86bb9ee0e3a2db3514a08636f852a92a84935d9b4f67837c5c391cffd2c

## Run dirs
- `results/run_arc_scalpel_v138_arc_v129_sample_fix3_try1`
- `results/run_arc_scalpel_v138_arc_v129_sample_fix3_try2`
- `results/run_arc_scalpel_v138_gridworld_v129_synth_fix2_try1`
- `results/run_arc_scalpel_v138_gridworld_v129_synth_fix2_try2`
- `results/run_arc_scalpel_v138_arc_agi1_training_official_fix1_try1`
- `results/run_arc_scalpel_v138_arc_agi1_training_official_fix1_try2`
- `results/run_arc_scalpel_v138_arc_agi1_training_official_fix2_try1`
- `results/run_arc_scalpel_v138_arc_agi1_training_official_fix2_try2`
- `results/run_arc_scalpel_v138_arc_agi1_evaluation_official_fix3_try1`
- `results/run_arc_scalpel_v138_arc_agi1_evaluation_official_fix3_try2`
