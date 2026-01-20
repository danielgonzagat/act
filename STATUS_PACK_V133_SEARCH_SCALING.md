# STATUS PACK V133 — SEARCH SCALING

## Checklist (PASS/FAIL)
- PASS `py_compile`
- PASS `unittest -v`
- PASS ARC scalpel `data/arc_v129_sample` (determinism_ok=true, isolation_ok=true)
- PASS ARC scalpel `data/gridworld_v129_synth` (determinism_ok=true, isolation_ok=true)
- PASS `verify_freeze` (ok=true, warnings=[])

## Contratos Extraídos Dos Anexos
- `/Users/danielpenin/Desktop/Projeto ACT.rtf` (sha256=`f8e42eff53046b4892d277ca3f6b1f400b48331254c800949a584f029a65d2f0`)
- `/Users/danielpenin/Desktop/ACT resolvendo ARC.rtf` (sha256=`06e121795273439b0ddffed6b7bea58dd24270ebd36450365dbcb8142c2db59d`)
- Determinismo total: ordenação estável + tie-break determinístico; sem RNG/clock implícito.
- WORM: outputs write-once; destino existente => FAIL.
- Fail-closed: ambiguidade real => `UNKNOWN` (sem chute).
- Anti-hack: proibido qualquer branch por `task_id`/filename/path/família.
- Isolamento: nenhuma escrita fora do `run_dir` do scalpel (prova por snapshot before/after).

## Comandos Rodados (copy/paste)
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v133.py --arc_root data/arc_v129_sample --split sample --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v133_arc_v129_sample_fix2 | tee results/smoke_arc_scalpel_v133_arc_v129_sample_fix2_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v133.py --arc_root data/gridworld_v129_synth --split synth --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v133_gridworld_v129_synth_fix2 | tee results/smoke_arc_scalpel_v133_gridworld_v129_synth_fix2_try1.json`
- `(diff -u /dev/null atos_core/arc_loader_v133.py; diff -u /dev/null atos_core/arc_solver_v133.py; diff -u /dev/null scripts/run_arc_scalpel_v133.py; diff -u /dev/null tests/test_arc_solver_v133.py) > patches/v133_search_scaling.diff`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze LEDGER_ATOLANG_V0_2_30_BASELINE_V133_SEARCH_SCALING.json | tee results/verify_freeze_v133_try1.json`

## Resultados

### `data/arc_v129_sample`
- `tasks_total=11 solved=4 unknown=0 failed=7 solve_rate=0.36363636363636365`
- `failure_counts={"MISSING_OPERATOR":5,"SEARCH_BUDGET_EXCEEDED":2}`
- `determinism_ok=true isolation_ok=true`
- `try1.summary_sha256=ebf01208f463d4d9e635b63149c1e9045746c5f663dc8ee6d3d732af283c331e`
- `try1.eval_sha256=55dfc044592d13634c1afad6e0efd979a243031d5910419dbe08becaa4e80bf7`

### `data/gridworld_v129_synth`
- `tasks_total=6 solved=3 unknown=0 failed=3 solve_rate=0.5`
- `failure_counts={"MISSING_OPERATOR":1,"SEARCH_BUDGET_EXCEEDED":2}`
- `determinism_ok=true isolation_ok=true`
- `try1.summary_sha256=7bfb841cc689d4435eac5a6d1ef7dd936f72825eb8bb2fefb87594c21a3fe6f8`
- `try1.eval_sha256=d23c1c18dbbe6e2ef26fccd7c434ccb464b9b8cc313c27d5dd96dbb637393a42`

## Artefatos + SHA256 (principais)
- `patches/v133_search_scaling.diff` sha256=`c6dff5095dd6541791668ab34bf15456e5dd732c8a4f38863bb1609ebf6d7809`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V133_SEARCH_SCALING.json` sha256=`67efdb5347fc61dcf28261b6de435f7f61c7be778b16f29b1148d2ae99b052ae`
- `results/verify_freeze_v133_try1.json` sha256=`923db6652db442d735e6e69aac19e7351320133925e4306a3a7a9455c87beba2`
- `results/smoke_arc_scalpel_v133_arc_v129_sample_fix2_try1.json` sha256=`0479313bf8bd37fff4f5e418fda26f808991c3b0af68183347014dac3e328bd9`
- `results/smoke_arc_scalpel_v133_gridworld_v129_synth_fix2_try1.json` sha256=`f6486bd7b22d1e5d7bff16578163369be47180471db5e0fb5a82b018739d870f`
- `results/run_arc_scalpel_v133_arc_v129_sample_fix2_try1/summary.json` sha256=`2140fcb2c6dbf739b04b4161a05da9205042edb4e9f718eca4709fffcf06276b`
- `results/run_arc_scalpel_v133_arc_v129_sample_fix2_try1/eval.json` sha256=`55dfc044592d13634c1afad6e0efd979a243031d5910419dbe08becaa4e80bf7`
- `results/run_arc_scalpel_v133_arc_v129_sample_fix2_try1/isolation_check_v133.json` sha256=`5157ae2898f6dff7cef2611376bb706099e8719e94818fd43d3caed022f83e62`
- `results/run_arc_scalpel_v133_arc_v129_sample_fix2_try1/outputs_manifest.json` sha256=`27bd01d8607aac070a31a0b355aceff34f8fbef16994dafd85f6bd45b130f63a`
- `results/run_arc_scalpel_v133_gridworld_v129_synth_fix2_try1/summary.json` sha256=`e55c47d848c29937ef27fd21a6879d8ec674d73d970ddf7b5ff153a7816aa037`
- `results/run_arc_scalpel_v133_gridworld_v129_synth_fix2_try1/eval.json` sha256=`d23c1c18dbbe6e2ef26fccd7c434ccb464b9b8cc313c27d5dd96dbb637393a42`
- `results/run_arc_scalpel_v133_gridworld_v129_synth_fix2_try1/isolation_check_v133.json` sha256=`5157ae2898f6dff7cef2611376bb706099e8719e94818fd43d3caed022f83e62`
- `results/run_arc_scalpel_v133_gridworld_v129_synth_fix2_try1/outputs_manifest.json` sha256=`4bc743e10054a32b9da866dcf80a6ce047c2a69efa9e13e5016da7e86cc5b93d`

## Run Dirs
- `results/run_arc_scalpel_v133_arc_v129_sample_fix2_try1`
- `results/run_arc_scalpel_v133_arc_v129_sample_fix2_try2`
- `results/run_arc_scalpel_v133_gridworld_v129_synth_fix2_try1`
- `results/run_arc_scalpel_v133_gridworld_v129_synth_fix2_try2`

## GO/NO-GO
- GO

