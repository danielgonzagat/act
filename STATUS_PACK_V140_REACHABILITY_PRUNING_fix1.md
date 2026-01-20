# STATUS_PACK_V140_REACHABILITY_PRUNING_fix1

## Summary
- Adds abstract reachability pruning (shape + palette + min-steps-to-modify) to reduce futile search expansion without task-specific logic.
- Fixes `new_canvas` argument mismatch (`bg` -> `color`) in the v140 solver proposal layer.
- Adds a deterministic diagnostic script to cluster V139 training-full failures and justify the pruning delta.

## Commands
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v140.py --arc_root data/arc_v129_sample --split sample --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v140_arc_v129_sample_fix1 | tee results/smoke_arc_scalpel_v140_arc_v129_sample_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v140.py --arc_root data/gridworld_v129_synth --split synth --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v140_gridworld_v129_synth_fix1 | tee results/smoke_arc_scalpel_v140_gridworld_v129_synth_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v140.py --arc_root ARC-AGI --split training --limit 0 --seed 0 --max_programs 500 --out_base results/run_arc_scalpel_v140_arc_agi1_training_full_fix2 | tee results/smoke_arc_scalpel_v140_arc_agi1_training_full_fix2_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v140.py --arc_root ARC-AGI --split evaluation --limit 0 --seed 0 --max_programs 500 --out_base results/run_arc_scalpel_v140_arc_agi1_evaluation_full_fix1 | tee results/smoke_arc_scalpel_v140_arc_agi1_evaluation_full_fix1_try1.json`
- `(diff -u /dev/null atos_core/arc_loader_v140.py; diff -u /dev/null atos_core/arc_solver_v140.py; diff -u /dev/null scripts/run_arc_scalpel_v140.py; diff -u /dev/null scripts/arc_diag_v140_from_v139_training_full.py; diff -u /dev/null tests/test_arc_solver_v140.py) > patches/v140_reachability_pruning_fix1.diff`

## Results (try1==try2)
- sample_fix1: tasks_total=11 solved@1=10 solved@2=10 solved@3=10 failure_counts={"MISSING_OPERATOR": 1} summary_sha256=057ef94425a8e8210407a4bdea0d3b5ce62e491503bd0fcbad0385f035fdc598 outputs_manifest_sig=7d8cd90d22ceaef28f11f8adfe9c4897412105c1236f60243d25dc7b6e0c2a98 isolation_ok=true
- synth_fix1: tasks_total=6 solved@1=6 solved@2=6 solved@3=6 failure_counts={} summary_sha256=f0700082f8a2643bda59ca724786de8efda2d3623263b3c0962d1e7354b7fe34 outputs_manifest_sig=5168b18957ce8accd7a5b033e2a4d9acaa636d3313ffa2a385b66fb9ec99b325 isolation_ok=true
- training_full_fix2: tasks_total=400 solved@1=15 solved@2=15 solved@3=15 failure_counts={"MISSING_OPERATOR":193,"SEARCH_BUDGET_EXCEEDED":191,"TEST_OUTPUT_MISMATCH":1} summary_sha256=d90d75d1913658835afd94beeada93428eac29e65876fbfebd3844f0a05feb1b outputs_manifest_sig=7e154dcdf19b002d9e343318bdd6391e07c66bd1e87524cbeafa3c4f0c4d3b81 isolation_ok=true
- evaluation_full_fix1: tasks_total=400 solved@1=2 solved@2=2 solved@3=2 failure_counts={"MISSING_OPERATOR":140,"SEARCH_BUDGET_EXCEEDED":258} summary_sha256=112fb51bf7bbca08738e7bf9e53daa5ab791805948e0b3d572e96a8143db71bc outputs_manifest_sig=1a4b2bb1d3139facb40b3be30fbfa43d2f6ab9180ef0b90a1e180a24d05a801d isolation_ok=true

## Files Added (V140)
- `atos_core/arc_loader_v140.py`
- `atos_core/arc_solver_v140.py`
- `scripts/arc_diag_v140_from_v139_training_full.py`
- `scripts/run_arc_scalpel_v140.py`
- `tests/test_arc_solver_v140.py`

## SHA256 (selected)
- `patches/v140_reachability_pruning_fix1.diff` sha256=7233ed34bb064f396107398059a797f72930929827d7c2b4167adfd88882cdb8
- `results/diag/ARC_DIAG_V140_FROM_V139_TRAINING_FULL_fix1_try1.md` sha256=0dfbb4db8aeabeed410ff1be453ea2d441e0cc27f6127d089ee5bddea722e51e
- `results/smoke_arc_scalpel_v140_arc_agi1_training_full_fix2_try1.json` sha256=8be4ddc25e7249f2ee6c26caa0c210f5af6dd53d3a07693d4d7c684ae9a4888b
- `results/smoke_arc_scalpel_v140_arc_agi1_evaluation_full_fix1_try1.json` sha256=91ca20341d627cf7607a3d689d90f53f50c07fb09943bc662f6bc47f1855e8af
