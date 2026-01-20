# STATUS PACK V132 — V132_ARC_OBJECT_CONCEPTS — GO

## Resumo
- Adiciona um solver ARC “object-centric” (V132) com **estado tipado** (grid/objset/obj/bbox/patch) e operadores estruturais (`cc4`, `select_obj`, `paint_rect`, `draw_rect_border`, `paste`) + reutiliza ops grid→grid já existentes (rotate/reflect/translate/map_colors etc.).
- Introduz `propose_step_variants_v132(...)` para **domínios de parâmetros pequenos e determinísticos** derivados apenas de `train_pairs` (shapes/paleta/corners/mode/delta_bbox/bboxes), sem dependência de `task_id`.
- Fornece harness WORM (`scripts/run_arc_scalpel_v132.py`) com **try1/try2 determinísticos**, `outputs_manifest.json` e `isolation_check_v132.json` (falha se qualquer arquivo fora do run_dir mudar).

## CONTRATOS EXTRAÍDOS DOS ANEXOS (Projeto ACT.rtf / ACT resolvendo ARC.rtf)
- `Determinismo`: ordenação total, tie-break por hashes estáveis, zero dependência de wall-clock ou RNG implícito.
- `WORM`: outputs write-once (falha se o destino existir); logs/artefatos imutáveis e hasháveis.
- `Fail-closed`: sem ambiguidade “resolvida no chute”; múltiplos programas minimais divergentes → `UNKNOWN`.
- `Sem atalhos`: proibido branching por `task_id`/filename/path/família; proibido lookup de padrões/embeddings/treino offline.
- `Auditoria`: todo resultado precisa de trace/artefatos verificáveis; qualquer “capacidade” deve ser explicitável como objeto/ato e replayável.
- `Isolamento`: runs de diagnóstico não podem contaminar o repo (nada persiste fora do `run_dir` WORM).

## Checklist (PASS/FAIL)
- PASS `py_compile` (atos_core + scripts + tests)
- PASS `unittest -v`
- PASS ARC scalpel V132 em `data/arc_v129_sample` (try1==try2; determinism_ok=true; isolation_ok=true)
- PASS ARC scalpel V132 em `data/gridworld_v129_synth` (try1==try2; determinism_ok=true; isolation_ok=true)
- PASS `verify_freeze` do ledger (ok=true, warnings=[])

## Comandos rodados (copy/paste)
```bash
cd /Users/danielpenin/act
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v132.py \
  --arc_root data/arc_v129_sample --out_base results/run_arc_scalpel_v132_arc_v129_sample_fix1 --seed 0 --limit 999 \
| tee results/smoke_arc_scalpel_v132_arc_v129_sample_fix1_try1.json

set -o pipefail && PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/run_arc_scalpel_v132.py \
  --arc_root data/gridworld_v129_synth --out_base results/run_arc_scalpel_v132_gridworld_v129_synth_fix1 --seed 0 --limit 999 \
| tee results/smoke_arc_scalpel_v132_gridworld_v129_synth_fix1_try1.json

(diff -u /dev/null atos_core/arc_loader_v132.py; diff -u /dev/null atos_core/arc_objects_v132.py; diff -u /dev/null atos_core/arc_ops_v132.py; diff -u /dev/null atos_core/arc_solver_v132.py; diff -u /dev/null scripts/run_arc_scalpel_v132.py; diff -u /dev/null tests/test_arc_solver_v132.py) > patches/v132_arc_object_concepts.diff

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V132_ARC_OBJECT_CONCEPTS.json \
| tee results/verify_freeze_v132_try1.json
```

## Resultados (métricas)
### `data/arc_v129_sample`
- tasks_total=11 solved=4 unknown=0 failed=7 solve_rate=0.36363636363636365
- failure_counts (try1): `SEARCH_BUDGET_EXCEEDED=7`
- determinismo: `summary_sha256=751fc438e656ec20fde2594d88041aff6777a8247fd67f4ecd5be2f9b179d47e`, `outputs_manifest_sig=71e777158fc4d2ed6e0ba3948fed66563f4260715aa0c20d3094508db383381a`

### `data/gridworld_v129_synth`
- tasks_total=6 solved=3 unknown=0 failed=3 solve_rate=0.5
- failure_counts (try1): `SEARCH_BUDGET_EXCEEDED=3`
- determinismo: `summary_sha256=24dda312e710a8de7acf2d42e1ffad4b4cf85a25a321ebbba6257f4f901d8ea3`, `outputs_manifest_sig=b47421290ecc15303834c84afd6840a9aa4c1340571e2477ba91f4b42bb59a71`

## Artefatos WORM (paths + sha256)
- `patches/v132_arc_object_concepts.diff` sha256=`0451858c973168eed3fc2ccaa4a6c3af2e1c2b9441937de9da3144f16b0e30e3`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V132_ARC_OBJECT_CONCEPTS.json` sha256=`ddca2be97e8531c6a22df2cc5f1672f77a8eb581bbef9c8aa768704121f927bd`
- `results/verify_freeze_v132_try1.json` sha256=`959b1b932a49d626a6e836823d9cd8b0956e72b02ae392ae6087cf0a4dbaaf55`
- `results/smoke_arc_scalpel_v132_arc_v129_sample_fix1_try1.json` sha256=`be9f102ac42b984e830df231b0bee7da315fd79ad11786c61bb96abbb9728d95`
- `results/smoke_arc_scalpel_v132_gridworld_v129_synth_fix1_try1.json` sha256=`19b5d616bc1c529b86cbc85d79315fe7193960d583374c9c53ceffaccde19d93`

## Run dirs
- `results/run_arc_scalpel_v132_arc_v129_sample_fix1_try1/`
- `results/run_arc_scalpel_v132_arc_v129_sample_fix1_try2/`
- `results/run_arc_scalpel_v132_gridworld_v129_synth_fix1_try1/`
- `results/run_arc_scalpel_v132_gridworld_v129_synth_fix1_try2/`

## Arquivos novos/modificados (V132)
- `atos_core/arc_loader_v132.py`
- `atos_core/arc_objects_v132.py`
- `atos_core/arc_ops_v132.py`
- `atos_core/arc_solver_v132.py`
- `scripts/run_arc_scalpel_v132.py`
- `tests/test_arc_solver_v132.py`
- `patches/v132_arc_object_concepts.diff`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V132_ARC_OBJECT_CONCEPTS.json`

## GO/NO-GO
- GO (determinismo try1==try2 + isolamento + WORM + verify_freeze warnings=[]).

