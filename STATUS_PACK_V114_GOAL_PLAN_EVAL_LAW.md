# STATUS PACK V114 — GOAL→PLAN→EVAL(SATISFIES) como lei + FAIL_EVENT como ATO — GO/NO-GO

## Resumo
- Adiciona um gate V114 que torna **auditável/verificável** a lei: para cada user turn existe exatamente 1 **GOAL ATO**, um **PLAN ATO** e um **EVAL ATO** com `satisfies=true`; caso contrário grava **FAIL_EVENT** como ATO no MindGraph.
- Implementa wrapper `run_conversation_v114` (sem tocar no runtime V110) que roda o V110 e em seguida constrói `mind_graph_v114/` + `goal_plan_eval_summary_v114.json`.
- Regressão Family7 v113 (tasks reais do histórico) passa 20/20 sob runtime V114, determinístico try1==try2.

## Checklist (PASS/FAIL)
- PASS `py_compile` (atos_core + scripts).
- PASS `unittest -v` (inclui `tests/test_goal_plan_eval_gate_v114.py`).
- PASS smoke V114 GOAL/PLAN/EVAL law (fix2): determinismo try1==try2 + negativos (render blocked) + replan (FAIL_EVENT em tentativa).
- PASS regressão Family7 v113 sob runtime V114 (fix1): `tasks_ok==tasks_total==20` e determinismo try1==try2 por `eval_sha256`.
- PASS gating externo intacto (negativos not_allowed / invalid_reason_code + exatamente 1 task com 1 acesso in-cycle).

## Arquivos novos (código)
- `atos_core/goal_plan_eval_gate_v114.py:1` (verificador/gerador do MindGraph v114 + FAIL_EVENT ATO).
- `atos_core/conversation_loop_v114.py:1` (wrapper do `run_conversation_v110` + gate V114).
- `scripts/run_family7_dla_v114.py:1` (runner Family7 v114 usando `run_conversation_v114`).
- `scripts/smoke_v114_family7_real_history_stress.py:1` (smoke determinístico try1/try2 para Family7 com runtime V114).
- `scripts/smoke_v114_goal_plan_eval_law.py:1` (smoke determinístico try1/try2 com casos positivos/negativos/replan).
- `tests/test_goal_plan_eval_gate_v114.py:1` (unit tests do gate V114).

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V114 (GOAL/PLAN/EVAL law) — fix2 (fix1 já existe e falhou por determinismo via paths)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v114_goal_plan_eval_law.py \
  --out_base results/run_smoke_v114_goal_plan_eval_law_fix2 --seed 0 \
  | tee results/smoke_v114_goal_plan_eval_law_fix2_try1.json

# Regressão Family7 v113 (tasks reais do histórico) sob runtime V114 — fix1
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v114_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v114_family7_real_history_stress_fix1 \
  --seed 0 | tee results/smoke_v114_family7_real_history_stress_fix1_try1.json

# Patch diff (WORM)
(diff -u /dev/null atos_core/goal_plan_eval_gate_v114.py; \
 diff -u /dev/null atos_core/conversation_loop_v114.py; \
 diff -u /dev/null scripts/run_family7_dla_v114.py; \
 diff -u /dev/null scripts/smoke_v114_family7_real_history_stress.py; \
 diff -u /dev/null scripts/smoke_v114_goal_plan_eval_law.py; \
 diff -u /dev/null tests/test_goal_plan_eval_gate_v114.py) > patches/v114_goal_plan_eval_law.diff
```

## Métricas principais
- Smoke V114 law (fix2):
  - `summary_sha256 = ffc3e4e1cf2634553970de7752a3d9dc50dafe04a44ff6b124a4bdc9d877cbb6`
  - determinismo: `eval_sha256 try1 == try2 == 473c6e86ff161a60265d0e15cd66f6664622d26451b760cc553ac85bf211871b`
- Family7 v113 sob runtime V114 (fix1):
  - `tasks_total=20`, `tasks_ok=20`
  - determinismo: `eval_sha256 try1 == try2 == 4632a4ba9b101fa39d5eb9e03d771c6b281627f17d65a6d488cc26f1af0171e2`
  - gating externo in-cycle: exatamente 1 task com `external_world_events_total==1`

## Artefatos (paths) + SHA256
- Patch diff: `patches/v114_goal_plan_eval_law.diff` = `91a3252be808df79f4199cfe91f94e14274ccf8d40c0502d6e128bd31cff805e`
- Smoke V114 law (fix2):
  - stdout tee: `results/smoke_v114_goal_plan_eval_law_fix2_try1.json` = `92ae0f9d09e577e7672d00eed9ca94910e5245f7e33441ecafda1be46e24cab4`
  - try1 eval: `results/run_smoke_v114_goal_plan_eval_law_fix2_try1/eval.json` = `473c6e86ff161a60265d0e15cd66f6664622d26451b760cc553ac85bf211871b`
  - try1 summary: `results/run_smoke_v114_goal_plan_eval_law_fix2_try1/summary.json` = `eaa2495673952f1522799b474ac5e55d4ee11581067167a7568c55ff69784c2b`
  - (exemplo artefatos da lei V114):  
    - `results/run_smoke_v114_goal_plan_eval_law_fix2_try1/case_00_positive/goal_plan_eval_summary_v114.json` = `9ab7d490cf2fa5243d3d2af59358ed1793b9aae2a1e3c771d0193137c20033e8`  
    - `results/run_smoke_v114_goal_plan_eval_law_fix2_try1/case_00_positive/mind_graph_v114/mind_nodes.jsonl` = `976f1a326925136ac539f2e29bb0ba6bd987cb423c45575c1269ccaf93539ba3`  
    - `results/run_smoke_v114_goal_plan_eval_law_fix2_try1/case_00_positive/mind_graph_v114/mind_edges.jsonl` = `3f3874e78f64177014fb8849a6d7d0eeb8796ca0b08625252f9fdcc463380fc3`
- Smoke Family7 v114 (fix1):
  - stdout tee: `results/smoke_v114_family7_real_history_stress_fix1_try1.json` = `a380fd1250ac5a953bece60d818dccf228770e74f069dd6245bcd7f2ceb1d9f2`
  - try1 eval: `results/run_smoke_v114_family7_real_history_stress_fix1_try1/eval.json` = `4632a4ba9b101fa39d5eb9e03d771c6b281627f17d65a6d488cc26f1af0171e2`
  - try1 summary: `results/run_smoke_v114_family7_real_history_stress_fix1_try1/summary.json` = `8e0f6a6cf7e9a2c382ff671797863ce4c343cf4cd0ea278a4bd44c127a88e688`

## GO/NO-GO
- GO. (Fix2 no smoke da lei por WORM/determinismo: o fix1 falhou porque o `eval.json` incluía paths variáveis.)

