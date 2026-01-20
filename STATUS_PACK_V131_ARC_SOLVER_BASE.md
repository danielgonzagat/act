# STATUS PACK V131 — ARC_SOLVER_BASE (GO)

## Resumo
V131 adiciona um solver ARC v1 determinístico por busca composicional (programas de primitivas gerais), com custo/MDL explícito, FAIL-CLOSED (AMBIGUOUS_RULE → UNKNOWN), trace auditável e harness WORM (try1/try2 com hashes idênticos + isolation_check).

## Checklist (PASS/FAIL)
- PASS `py_compile` (`atos_core/*.py`, `scripts/*.py`, `tests/*.py`)
- PASS `unittest -v`
- PASS `scripts/run_arc_scalpel_v131.py` (try1==try2: `summary_sha256` e `outputs_manifest_sig` idênticos)
- PASS `isolation_check_v131.ok == true` (zero escrita fora do run_dir)
- PASS `verify_freeze` (`ok=true`, `warnings=[]`)

## Arquivos adicionados (V131)
- `atos_core/arc_loader_v131.py`
- `atos_core/arc_solver_v131.py`
- `scripts/run_arc_scalpel_v131.py`
- `tests/test_arc_solver_v131.py`

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v
set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v131.py --out_base results/run_arc_v131_seed0 --seed 0 --limit 999999 | tee results/smoke_arc_v131_try1.json
(diff -u /dev/null atos_core/arc_loader_v131.py; diff -u /dev/null atos_core/arc_solver_v131.py; diff -u /dev/null scripts/run_arc_scalpel_v131.py; diff -u /dev/null tests/test_arc_solver_v131.py) > patches/v131_arc_solver_base.diff
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze LEDGER_ATOLANG_V0_2_30_BASELINE_V131_ARC_SOLVER_BASE.json | tee results/verify_freeze_v131_try1.json
```

## Métricas (subset `data/arc_v124_sample`, split=sample)
- `tasks_total=4`, `solved=2`, `unknown=1`, `failed=1`, `solve_rate=0.5`
- `failure_counts`: `AMBIGUOUS_RULE=1`, `SEARCH_BUDGET_EXCEEDED=1`

## Determinismo (try1 == try2)
- `summary_sha256`: `ba82dbb010c47d320afc984f32649c4023a812bf8bc9003d32696cede2850c29`
- `eval_sha256` (`results/run_arc_v131_seed0_try1/eval.json`): `c6f973168324510ab27c89c52bd4d3d233696f17ac688f337fa9c1f9dca101f6`
- `outputs_manifest_sig`: `5d8741e96a981677cfdce242dc464aeaf462b08432e22a24ec23b0ff51dc5efd`
- `isolation_check_v131.ok`: `true`

## Artefatos + SHA256
- `patches/v131_arc_solver_base.diff`: `859de303db1973a8a4fe7b8cacafadc8f141e3e0a5210fb28972a5ba28648ec6`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V131_ARC_SOLVER_BASE.json`: `328ef311784a646a6906f2f86622fed0fc54703bbea5667ef8f625cae9269f19`
- `results/verify_freeze_v131_try1.json`: `79dab4d893c988f699fd1d2d84744676115c66559805f483e544a672da758834`
- `results/smoke_arc_v131_try1.json`: `a1030db77311e27d47775457e0a22ce0a90e9d2606a9a17860d35ec94961153f`
- `results/run_arc_v131_seed0_try1/summary.json`: `f9db305d9e2366f16910fd053d8fb1f977f32659d497a1a59448618a945dd790`
- `results/run_arc_v131_seed0_try1/eval.json`: `c6f973168324510ab27c89c52bd4d3d233696f17ac688f337fa9c1f9dca101f6`
- `results/run_arc_v131_seed0_try1/isolation_check_v131.json`: `c00eb01b099295f94e55e0904f21490550f6f6f1686d40c797d7b58c4f40992f`
- `results/run_arc_v131_seed0_try1/outputs_manifest.json`: `8534334c03af4ad1f353e54b5ec44650a0cd48249ef3f354ab31a83e696227fd`
- `results/run_arc_v131_seed0_try1/ARC_DIAG_REPORT_v131.md`: `65b956b7e3a8d47c602fbcbd7b40c9b0fdd80e80c6203348bb69c751033a0abf`

## Runs
- try1: `results/run_arc_v131_seed0_try1`
- try2: `results/run_arc_v131_seed0_try2`

