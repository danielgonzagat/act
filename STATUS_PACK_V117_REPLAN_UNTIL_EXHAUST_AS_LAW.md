# STATUS PACK V117 — REPLAN UNTIL EXHAUST AS LAW

## Resumo
- Implementa loop determinístico de replanning `replan_until_satisfies_v117(...)` com razões explícitas: `ok`, `exhausted_plans`, `plan_search_budget_exhausted`, `duplicate_plan_candidate`.
- Adiciona wrapper `run_conversation_v117(...)` que roda múltiplas tentativas completas do runtime V116 (seeds `seed+i`) e **só retorna sucesso quando encontra uma tentativa com `final_response_v116.ok==true`**; caso contrário emite FAIL determinístico por budget.
- Gera evidência WORM: `replan_trace_v117.json` + `mind_graph_v117/` com `FAIL_EVENT_V117` (ATO) e arestas `DERIVED_FROM` para GOAL/PLAN/TURN.
- Regressão Family7 (histórico real v113) roda sob V117 com 20/20 e determinismo try1==try2, mantendo gating do ExternalWorld e negativos determinísticos.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Smoke V117 (replan law): determinismo try1==try2
- PASS Smoke V117 (Family7 v113): `tasks_ok==20`, determinismo try1==try2, `external_world_events_total==1` em exatamente 1 task, negativos de gating PASS
- PASS WORM: outputs V117 em novos paths (fix1 para smoke replan por colisão prévia)

## Arquivos novos (V117)
- `atos_core/conversation_loop_v117.py:1` — wrapper runtime com “replan por tentativas” + `replan_trace_v117.json` + `mind_graph_v117/`.
- `atos_core/replan_law_v117.py:1` — loop determinístico de replanning + trace write-once.
- `scripts/run_family7_dla_v117.py:1` — runner Family7 usando `run_conversation_v117`.
- `scripts/smoke_v117_family7_real_history_stress.py:1` — smoke determinístico try1/try2 + negativos de gating do ExternalWorld.
- `scripts/smoke_v117_replan_law.py:1` — smoke determinístico do módulo `replan_law_v117` (3 casos).
- `tests/test_replan_law_v117.py:1` — unit tests do replanning loop.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v117_replan_law.py \
  --out_base results/run_smoke_v117_replan_law_fix1 --seed 0 \
| tee results/smoke_v117_replan_law_fix1_try1.json

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v117_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v117_family7_real_history_stress_fix1 --seed 0 \
| tee results/smoke_v117_family7_real_history_stress_fix1_try1.json
```

## Determinismo (try1 == try2)
- Smoke `replan_law`:
  - `summary_sha256`: `65e0bf39d379ab636aa0f3591fc24c99b86860dea998b588c61cb4af7b24d6e8`
  - `eval_sha256` (de `results/run_smoke_v117_replan_law_fix1_try1/summary.json`): `d9b53a238d5257cd438b9fbab91d716bd7b9d92cd5081479ec9296f2eb79ebba`
- Smoke Family7 v113:
  - `summary_sha256`: `12a3e59ca908238535bc673d46438aa0f9f28e50889783dd5ce0e30712135a0e`
  - `eval_sha256` (de `results/run_smoke_v117_family7_real_history_stress_fix1_try1/summary.json`): `0673cf9dea2eddfecd354db58c1ce6db118f8c85f0e27259d0c5ed5186598f59`

## Artefatos (paths)
- Smoke replan law: `results/run_smoke_v117_replan_law_fix1_try1/`, `results/run_smoke_v117_replan_law_fix1_try2/`
- Smoke Family7: `results/run_smoke_v117_family7_real_history_stress_fix1_try1/`, `results/run_smoke_v117_family7_real_history_stress_fix1_try2/`
- Patch diff: `patches/v117_replan_until_exhaust_as_law.diff`

