# STATUS_PACK_V137_ARC_AGI1_OFFICIAL_RUN_fix1

## Summary
- Adds `propagate_color_translate` (grid propagation along a fixed shift) as a deterministic, general grid operator + direct inference in `arc_solver_v137`.
- Fixes a reachability/pruning bug in V137 (do not prune at depth==max_depth) and adds unit coverage.

## Operator
- name: `propagate_color_translate`
- signature: `(GRID) -> GRID`
- semantics: Iteratively fill pad-cells when the source cell at (r-dy, c-dx) equals color; bounded to h+w+1 iterations.
- invariants: Deterministic, No RNG, No task_id/path branching, Grid values remain in 0..9

## Checklist
- PASS `py_compile`
- PASS `unittest -v`
- PASS scalpel sample+synth (determinism_ok=true, isolation_ok=true)
- PASS ARC-AGI smoke (training/evaluation, limit=20)
- PASS ARC-AGI full (training/evaluation, limit=0)

## Results
- sample: tasks_total=11 solved=11 unknown=0 failed=0 solve_rate=1.0000 failure_counts={}
- synth: tasks_total=6 solved=6 unknown=0 failed=0 solve_rate=1.0000 failure_counts={}
- training_smoke: tasks_total=20 solved=1 unknown=0 failed=19 solve_rate=0.0500 failure_counts={"MISSING_OPERATOR": 14, "SEARCH_BUDGET_EXCEEDED": 5}
- evaluation_smoke: tasks_total=20 solved=0 unknown=0 failed=20 solve_rate=0.0000 failure_counts={"MISSING_OPERATOR": 12, "SEARCH_BUDGET_EXCEEDED": 8}
- training_full: tasks_total=400 solved=17 unknown=0 failed=383 solve_rate=0.0425 failure_counts={"MISSING_OPERATOR": 284, "SEARCH_BUDGET_EXCEEDED": 99}
- evaluation_full: tasks_total=400 solved=2 unknown=0 failed=398 solve_rate=0.0050 failure_counts={"MISSING_OPERATOR": 253, "SEARCH_BUDGET_EXCEEDED": 145}

## Commands
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v137.py --arc_root data/arc_v129_sample --split sample --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v137_arc_v129_sample_fix1 | tee results/smoke_arc_scalpel_v137_arc_v129_sample_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v137.py --arc_root data/gridworld_v129_synth --split synth --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v137_gridworld_v129_synth_fix1 | tee results/smoke_arc_scalpel_v137_gridworld_v129_synth_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v137.py --arc_root ARC-AGI --split training --limit 20 --seed 0 --out_base results/run_arc_scalpel_v137_arc_agi1_training_official_smoke_fix1 | tee results/smoke_arc_scalpel_v137_arc_agi1_training_official_smoke_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v137.py --arc_root ARC-AGI --split evaluation --limit 20 --seed 0 --out_base results/run_arc_scalpel_v137_arc_agi1_evaluation_official_smoke_fix1 | tee results/smoke_arc_scalpel_v137_arc_agi1_evaluation_official_smoke_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v137.py --arc_root ARC-AGI --split training --limit 0 --seed 0 --out_base results/run_arc_scalpel_v137_arc_agi1_training_official_full_fix1 | tee results/smoke_arc_scalpel_v137_arc_agi1_training_official_full_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v137.py --arc_root ARC-AGI --split evaluation --limit 0 --seed 0 --out_base results/run_arc_scalpel_v137_arc_agi1_evaluation_official_full_fix1 | tee results/smoke_arc_scalpel_v137_arc_agi1_evaluation_official_full_fix1_try1.json`

## Artifacts (sha256)
- patch: `patches/v137_propagate_color_translate_fix1.diff` sha256=5fa6a7c1ef54bb5c4d2b2e526ae064595846227e50ae7e449e758601306600cc
- diag training v136: `results/diag/ARC_DIAG_V136_TRAINING_fix3_try1.md` sha256=b83e10a2dfea02cb2ea8797596c38405f8f3d1833edeed769a0d32db04ff803e
- diag evaluation v136: `results/diag/ARC_DIAG_V136_EVALUATION_fix3_try1.md` sha256=837e5d01041b92db4a547056064a5c930e8989e0fa25f6981d3f42cfb49dfcba
- `results/smoke_arc_scalpel_v137_arc_v129_sample_fix1_try1.json` sha256=52ed0340be3caa09f13d3341ec9e74da5f5e5dc6977b69c15f2652a790a01962
- `results/smoke_arc_scalpel_v137_gridworld_v129_synth_fix1_try1.json` sha256=97c4d520697e45b65218538f151f061c164b8c13737dc65bc1862680c6bbb2bf
- `results/smoke_arc_scalpel_v137_arc_agi1_training_official_smoke_fix1_try1.json` sha256=fa8d3b936ae5d3d7278cfdbf37a5bc873d25c6df4f9ac890e65b6321fef1867e
- `results/smoke_arc_scalpel_v137_arc_agi1_evaluation_official_smoke_fix1_try1.json` sha256=886651cbe382d784d2e0c0c84318003fb8ec59353177840daaad5ed26f8aa49d
- `results/smoke_arc_scalpel_v137_arc_agi1_training_official_full_fix1_try1.json` sha256=18964f80eb1c98c8831f5c6436aa827401e330a0856b9bdd42fdcb95fc2cd4e2
- `results/smoke_arc_scalpel_v137_arc_agi1_evaluation_official_full_fix1_try1.json` sha256=c6491e207a7ad3614a4d49052d6e25b51fa3286bd3971d4c6f3ff34c2a292b7a

