# STATUS PACK V134 — REPEAT_GRID_PRUNING

## Checklist (PASS/FAIL)
- PASS `py_compile`
- PASS `unittest -v`
- PASS ARC scalpel `data/arc_v129_sample` (determinism_ok=true, isolation_ok=true, SEARCH_BUDGET_EXCEEDED=0)
- PASS ARC scalpel `data/gridworld_v129_synth` (determinism_ok=true, isolation_ok=true, SEARCH_BUDGET_EXCEEDED=0)
- PASS `verify_freeze` (ok=true, warnings=[])

## Contratos Extraídos Dos Anexos
- `/Users/danielpenin/Desktop/Projeto ACT.rtf` (sha256=`f8e42eff53046b4892d277ca3f6b1f400b48331254c800949a584f029a65d2f0`)
- `/Users/danielpenin/Desktop/ACT resolvendo ARC.rtf` (sha256=`985630b412dcf1f3d2f4def6c2901d9ffd7ed86bbd4b6233f3e0779706c1de3c`)
- Determinismo total: ordenação estável + tie-break determinístico; sem RNG/clock implícito.
- WORM: outputs write-once; destino existente => FAIL.
- Fail-closed: ambiguidade real => `UNKNOWN` (sem chute).
- Anti-hack: proibido qualquer branch por `task_id`/filename/path/família.
- Isolamento: nenhuma escrita fora do `run_dir` do scalpel (prova por snapshot before/after).

## Comandos Rodados (copy/paste)
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v134.py --arc_root data/arc_v129_sample --split sample --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v134_arc_v129_sample_fix1 | tee results/smoke_arc_scalpel_v134_arc_v129_sample_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v134.py --arc_root data/gridworld_v129_synth --split synth --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v134_gridworld_v129_synth_fix1 | tee results/smoke_arc_scalpel_v134_gridworld_v129_synth_fix1_try1.json`
- `(diff -u /dev/null atos_core/arc_loader_v134.py; diff -u /dev/null atos_core/arc_ops_v134.py; diff -u /dev/null atos_core/arc_solver_v134.py; diff -u /dev/null scripts/arc_diag_v134_pre.py; diff -u /dev/null scripts/run_arc_scalpel_v134.py; diff -u /dev/null tests/test_arc_solver_v134.py) > patches/v134_repeat_grid_pruning.diff`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze LEDGER_ATOLANG_V0_2_30_BASELINE_V134_REPEAT_GRID_PRUNING.json | tee results/verify_freeze_v134_try1.json`

## Resultados

### `data/arc_v129_sample`
- `tasks_total=11 solved=10 unknown=0 failed=1 solve_rate=0.9090909090909091`
- `failure_counts={"MISSING_OPERATOR":1}`
- `determinism_ok=true isolation_ok=true`
- `try1.summary_sha256=a0d313364aceb83e5b936795aabca047e6e3c70850e4acc88c149932bf6e4365`
- `try1.eval_sha256=a62e1b7f3cdeeb65b50d7e30842b72da12869e8a0c7b8fc752dd969d3b790ace`

### `data/gridworld_v129_synth`
- `tasks_total=6 solved=6 unknown=0 failed=0 solve_rate=1.0`
- `failure_counts={}`
- `determinism_ok=true isolation_ok=true`
- `try1.summary_sha256=b43ddf2adc1282cbac721289a1b480fba15c867827e3c28a399579636d8a2740`
- `try1.eval_sha256=35e100e9b3ddb145338a43f4753e8e38aeacb53733368a30ebaee1fb9e056d7f`

## Artefatos + SHA256 (principais)
- `patches/v134_repeat_grid_pruning.diff` sha256=`88f2ffb1bb56cf3741a91d2ce4c7c4a6c1699ccee4d942fb720300e937941d8e`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V134_REPEAT_GRID_PRUNING.json` sha256=`65f6345ebeffd5ffcc6ec9a9a4a25ce655f9d73d4273953aaadaa3dd403eddbb`
- `results/verify_freeze_v134_try1.json` sha256=`817a0e3bc8767270b37e8c43a578ed4bdfa11571803599f85fc175499e0bf43d`
- `results/smoke_arc_scalpel_v134_arc_v129_sample_fix1_try1.json` sha256=`ab8a97f00b1b302cb03cd702c78995777abd1ecfa8632ba101e25c7117bf18c5`
- `results/smoke_arc_scalpel_v134_gridworld_v129_synth_fix1_try1.json` sha256=`d073a9d859f76acf72a6687177e25e3e2e59be348c6c2601ccabba7eb0c90f83`
- `results/run_arc_scalpel_v134_arc_v129_sample_fix1_try1/summary.json` sha256=`9bcdf2452b8b67eee24b77ee536ef62cde9ab312e3389cf916c4dcd1fac809f9`
- `results/run_arc_scalpel_v134_arc_v129_sample_fix1_try1/eval.json` sha256=`a62e1b7f3cdeeb65b50d7e30842b72da12869e8a0c7b8fc752dd969d3b790ace`
- `results/run_arc_scalpel_v134_arc_v129_sample_fix1_try1/isolation_check_v134.json` sha256=`3fa8ade79e84a50951ac18555bc860cfcbe7fd77b4f3e97d1f656cf8f966d502`
- `results/run_arc_scalpel_v134_arc_v129_sample_fix1_try1/outputs_manifest.json` sha256=`c2b4353bd18fffd787726319f4741950ca25f6ec37d0e4d65db62f76b6bdf99a`
- `results/run_arc_scalpel_v134_gridworld_v129_synth_fix1_try1/summary.json` sha256=`888fb0e6dfea891d659456a3dd1435cf6de062aeab9a9fd81d9c8350b4d6f65d`
- `results/run_arc_scalpel_v134_gridworld_v129_synth_fix1_try1/eval.json` sha256=`35e100e9b3ddb145338a43f4753e8e38aeacb53733368a30ebaee1fb9e056d7f`
- `results/run_arc_scalpel_v134_gridworld_v129_synth_fix1_try1/isolation_check_v134.json` sha256=`3fa8ade79e84a50951ac18555bc860cfcbe7fd77b4f3e97d1f656cf8f966d502`
- `results/run_arc_scalpel_v134_gridworld_v129_synth_fix1_try1/outputs_manifest.json` sha256=`8a8f737fa2bc67ca80b9fb0ef0148f1d968a1f01976fca3f7b44aa815d87d089`

## Run Dirs
- `results/run_arc_scalpel_v134_arc_v129_sample_fix1_try1`
- `results/run_arc_scalpel_v134_arc_v129_sample_fix1_try2`
- `results/run_arc_scalpel_v134_gridworld_v129_synth_fix1_try1`
- `results/run_arc_scalpel_v134_gridworld_v129_synth_fix1_try2`

## GO/NO-GO
- GO

