# STATUS PACK V135 — OVERLAY_SELF_TRANSLATE

## Checklist (PASS/FAIL)
- PASS `py_compile`
- PASS `unittest -v`
- PASS PRE-DIAG `results/ARC_DIAG_REPORT_v135_PRE_fix1.md`
- PASS ARC scalpel `data/arc_v129_sample` (determinism_ok=true, isolation_ok=true, SEARCH_BUDGET_EXCEEDED=0)
- PASS ARC scalpel `data/gridworld_v129_synth` (determinism_ok=true, isolation_ok=true, SEARCH_BUDGET_EXCEEDED=0)
- PASS `verify_freeze` (ok=true, warnings=[])

## Contratos Extraídos Dos Anexos
- `/Users/danielpenin/Desktop/Projeto ACT.rtf` (sha256=`f8e42eff53046b4892d277ca3f6b1f400b48331254c800949a584f029a65d2f0`)
- `/Users/danielpenin/Desktop/ACT resolvendo ARC.rtf` (sha256=`6a531d0a2a6b64ab257c7ca44495c1dc1e6b38e8a0ba436e9cc5a199702c663c`)
- Determinismo total: ordenação estável + tie-break determinístico; sem RNG/clock implícito.
- WORM: outputs write-once; destino existente => FAIL.
- Fail-closed: ambiguidade real => `UNKNOWN` (sem chute).
- Anti-hack: proibido qualquer branch por `task_id`/filename/path/família.
- Isolamento: nenhuma escrita fora do `run_dir` do scalpel (prova por snapshot before/after).

## Comandos Rodados (copy/paste)
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/arc_diag_v135_pre.py --run_dir results/run_arc_scalpel_v134_arc_v129_sample_fix1_try1 --run_dir results/run_arc_scalpel_v134_gridworld_v129_synth_fix1_try1 --out_path results/ARC_DIAG_REPORT_v135_PRE_fix1.md`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v135.py --arc_root data/arc_v129_sample --split sample --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v135_arc_v129_sample_fix1 | tee results/smoke_arc_scalpel_v135_arc_v129_sample_fix1_try1.json`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v135.py --arc_root data/gridworld_v129_synth --split synth --limit 999999 --seed 0 --out_base results/run_arc_scalpel_v135_gridworld_v129_synth_fix1 | tee results/smoke_arc_scalpel_v135_gridworld_v129_synth_fix1_try1.json`
- `(diff -u /dev/null scripts/arc_diag_v135_pre.py; diff -u /dev/null atos_core/arc_loader_v135.py; diff -u /dev/null atos_core/arc_ops_v135.py; diff -u /dev/null atos_core/arc_solver_v135.py; diff -u /dev/null scripts/run_arc_scalpel_v135.py; diff -u /dev/null tests/test_arc_solver_v135.py) > patches/v135_overlay_self_translate.diff`
- `set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze LEDGER_ATOLANG_V0_2_30_BASELINE_V135_OVERLAY_SELF_TRANSLATE.json | tee results/verify_freeze_v135_try1.json`

## Resultados

### `data/arc_v129_sample`
- `tasks_total=11 solved=11 unknown=0 failed=0 solve_rate=1.0`
- `failure_counts={}`
- `determinism_ok=true isolation_ok=true`
- `try1.summary_sha256=fe1968b83bcb8c31b34c5193032b1a0d3a976caddd36a934bc49e6be34ee4b63`
- `try1.eval_sha256=3ce5cf8434e6a1fd5c16d9eefc52538cdeb51c7bc9097f710529f877455fa245`

### `data/gridworld_v129_synth`
- `tasks_total=6 solved=6 unknown=0 failed=0 solve_rate=1.0`
- `failure_counts={}`
- `determinism_ok=true isolation_ok=true`
- `try1.summary_sha256=47bb4ddca673716b3d6a27a2afd0132cf37870867358e7282c60337bf62abe28`
- `try1.eval_sha256=8fc95738aa0fbdbf460304796467ff3ad711f9a0a729b130fac748a5d627131b`

## Artefatos + SHA256 (principais)
- `patches/v135_overlay_self_translate.diff` sha256=`222ddb3fc4a647a9ce0af5eceb9086173dc7c556f9b29a981b267241032eab74`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V135_OVERLAY_SELF_TRANSLATE.json` sha256=`4a5aa4df694d68215709df159d8a3d27a29dfd9a56143b374f67560dfb0b679d`
- `results/verify_freeze_v135_try1.json` sha256=`fbac0dc118e48eec98367f5950948172549acd97890974469dc361b64e588a53`
- `results/ARC_DIAG_REPORT_v135_PRE_fix1.md` sha256=`1687ce5e5165124bd980b813bdb23921f9e06c033badabbd22aef3a92d09b588`
- `results/smoke_arc_scalpel_v135_arc_v129_sample_fix1_try1.json` sha256=`47811a321881242ab2563b3c48f9f7224e3f6c8dd7b72601e927223ab904d158`
- `results/smoke_arc_scalpel_v135_gridworld_v129_synth_fix1_try1.json` sha256=`ad566c3359a92890d86b4784f2f56cc9e6d5e89edd792807bc14c3173c6d283f`
- `results/run_arc_scalpel_v135_arc_v129_sample_fix1_try1/summary.json` sha256=`299fe89aa870f2eee8c2aab30ae0b6f8ea4d05a961c4790ce71d82150ca2e43e`
- `results/run_arc_scalpel_v135_arc_v129_sample_fix1_try1/eval.json` sha256=`3ce5cf8434e6a1fd5c16d9eefc52538cdeb51c7bc9097f710529f877455fa245`
- `results/run_arc_scalpel_v135_arc_v129_sample_fix1_try1/isolation_check_v135.json` sha256=`df0828926f67d3da5d6ce7a5dbb6fd6c7fbfc46b46b6f7794b231d29d4cd24c5`
- `results/run_arc_scalpel_v135_arc_v129_sample_fix1_try1/outputs_manifest.json` sha256=`82a560e2f8932aa6ffe106b3e079a8dc45055a9be27c6b1a5e2933a7eb71de2b`
- `results/run_arc_scalpel_v135_gridworld_v129_synth_fix1_try1/summary.json` sha256=`7c1ccf526e1162da2a2a500d6fbddedcd1cf540ec9fa69b901162c849ba323cf`
- `results/run_arc_scalpel_v135_gridworld_v129_synth_fix1_try1/eval.json` sha256=`8fc95738aa0fbdbf460304796467ff3ad711f9a0a729b130fac748a5d627131b`
- `results/run_arc_scalpel_v135_gridworld_v129_synth_fix1_try1/isolation_check_v135.json` sha256=`df0828926f67d3da5d6ce7a5dbb6fd6c7fbfc46b46b6f7794b231d29d4cd24c5`
- `results/run_arc_scalpel_v135_gridworld_v129_synth_fix1_try1/outputs_manifest.json` sha256=`4a65ccd58183e43660f57259233b1ab8a3a813fd6e5638efd2c0536d08c3bac7`

## Run Dirs
- `results/run_arc_scalpel_v135_arc_v129_sample_fix1_try1`
- `results/run_arc_scalpel_v135_arc_v129_sample_fix1_try2`
- `results/run_arc_scalpel_v135_gridworld_v129_synth_fix1_try1`
- `results/run_arc_scalpel_v135_gridworld_v129_synth_fix1_try2`

## GO/NO-GO
- GO

