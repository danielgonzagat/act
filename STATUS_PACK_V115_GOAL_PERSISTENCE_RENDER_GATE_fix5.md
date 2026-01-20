# STATUS PACK V115 (fix5) — GOAL persistente + render gate “lei física” (final_response) + Family7 v113 sob runtime V115 — GO/NO-GO

## Resumo
- Implementa **V115 como wrapper de runtime** que torna a lei **GOAL→PLAN→EVAL(SATISFIES=true)** um **gate de saída**: a resposta “oficial” do turno fica em `final_response_v115.json` e vira `FAIL: <reason>` quando a lei falha.
- Adiciona checks mínimos contra “placebo goal/plan” e registra falhas como **FAIL_EVENT ATO** (MindGraph v71) com evidência/arestas (GOAL/PLAN/turn).
- Atualiza o runner Family7 (tasks reais do histórico) para rodar **V115 law + fluency v112 + gating externo** com múltiplas tentativas determinísticas (seed schedule), mantendo `tasks_ok==20/20` e determinismo try1==try2.

## Checklist (PASS/FAIL)
- PASS `py_compile` (atos_core + scripts).
- PASS `unittest -v` (inclui `tests/test_goal_plan_eval_gate_v115.py`).
- PASS smoke V115 “law + persistência + render gate” (fix5): determinismo try1==try2 + negativos (missing satisfies / goal placebo).
- PASS regressão Family7 v113 sob runtime V115 (fix5): `tasks_ok==tasks_total==20` e determinismo try1==try2 por `eval_sha256`.
- PASS gating externo intacto (negativos not_allowed / invalid_reason_code + exatamente 1 task com `external_world_events_total==1`).

## Arquivos novos (código)
- `atos_core/conversation_loop_v115.py:1` (wrapper `run_conversation_v110` + gate + `final_response_v115.json`).
- `atos_core/goal_persistence_v115.py:1` (lifecycle derivável + snapshot + FAIL renderer).
- `atos_core/goal_plan_eval_gate_v115.py:1` (lei V115 + nontrivial checks + FAIL_EVENT ATO + MindGraph v115).
- `scripts/run_family7_dla_v115.py:1` (runner Family7 v115 com tentativas determinísticas + ledger per-task).
- `scripts/smoke_v115_goal_persistence_render_gate.py:1` (smoke determinístico try1/try2 com casos positivo/negativos).
- `scripts/smoke_v115_family7_real_history_stress.py:1` (smoke determinístico try1/try2 para Family7 com runtime V115).
- `tests/test_goal_plan_eval_gate_v115.py:1` (unit tests do gate V115).

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V115 (lei + persistência + render gate) — fix5
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v115_goal_persistence_render_gate.py \
  --out_base results/run_smoke_v115_goal_persistence_render_gate_fix5 --seed 0 \
  | tee results/smoke_v115_goal_persistence_render_gate_fix5_try1.json

# Regressão Family7 v113 (tasks reais do histórico) sob runtime V115 — fix5
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v115_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v115_family7_real_history_stress_fix5 --seed 0 \
  | tee results/smoke_v115_family7_real_history_stress_fix5_try1.json

# Patch diff (WORM)
(diff -u /dev/null atos_core/conversation_loop_v115.py; \
 diff -u /dev/null atos_core/goal_persistence_v115.py; \
 diff -u /dev/null atos_core/goal_plan_eval_gate_v115.py; \
 diff -u /dev/null scripts/run_family7_dla_v115.py; \
 diff -u /dev/null scripts/smoke_v115_family7_real_history_stress.py; \
 diff -u /dev/null scripts/smoke_v115_goal_persistence_render_gate.py; \
 diff -u /dev/null tests/test_goal_plan_eval_gate_v115.py) \
 > patches/v115_goal_persistence_render_gate_fix5.diff
```

## Métricas principais
- Smoke V115 render gate (fix5):
  - `summary_sha256 = 132fbf63239bf6bf8a13e92f52f335c5bc7bfe666e40526748f80181d81f8d37`
  - determinismo: `eval_sha256 try1 == try2 == 9c94de68677b8c234c6778a66c122976945b22c1a253fab153ee199582903aca`
- Smoke Family7 v113 sob runtime V115 (fix5):
  - `tasks_total=20`, `tasks_ok=20`
  - determinismo: `eval_sha256 try1 == try2 == 0cf0e6c036ada5de80b01c6c3ca7825ffe9321777fc7d741c4186da978b57176`
  - gating externo in-cycle: exatamente 1 task com `external_world_events_total==1`

## Artefatos (paths) + SHA256
- Patch diff: `patches/v115_goal_persistence_render_gate_fix5.diff` = `a28180981d0f32fdeb3596177d8a6d63bfbbd6461bdedbc551e028f587a7be13`
- Smoke V115 render gate (fix5):
  - stdout tee: `results/smoke_v115_goal_persistence_render_gate_fix5_try1.json` = `c44683fdc3fb121f13c94deb98e68309e6f60a738037c134f5374a3fe9b9f3e3`
  - try1 eval: `results/run_smoke_v115_goal_persistence_render_gate_fix5_try1/eval.json` = `9c94de68677b8c234c6778a66c122976945b22c1a253fab153ee199582903aca`
  - try1 summary: `results/run_smoke_v115_goal_persistence_render_gate_fix5_try1/summary.json` = `969bcb8e486af37e858687cc60ca960159c6902dc97ef01ab0fb8d69967e5db6`
- Smoke Family7 v115 (fix5):
  - stdout tee: `results/smoke_v115_family7_real_history_stress_fix5_try1.json` = `2566efaed5e31d9233f07bc3a206d522919d0bd720da7b46fc4fdf7c1084c613`
  - try1 eval: `results/run_smoke_v115_family7_real_history_stress_fix5_try1/eval.json` = `0cf0e6c036ada5de80b01c6c3ca7825ffe9321777fc7d741c4186da978b57176`
  - try1 summary: `results/run_smoke_v115_family7_real_history_stress_fix5_try1/summary.json` = `8e87ea4aa164b92753d12f8d54577e78c6023eed27cda4a2730ed78fa48cc6d4`
  - try1 fail catalog: `results/run_smoke_v115_family7_real_history_stress_fix5_try1/fail_catalog_v115.json` = `c58b3aa299f25390f7ff146569c8326f1ba833b210ecda1506e5fda4a8e2d022`
- Family7 tasks file (input): `tasks/family7_dla_v113_seed0_fix2.jsonl` = `72a91a0d4bb15b31ac4812e6b25d9c84ac9854f91b91ec493bf60face49773b1`

## GO/NO-GO
- GO (fix5 por WORM: execuções anteriores fix1–fix4 foram consumidas por falhas intermediárias antes do runner ficar alinhado com V113/V114 em unresolved_reference e criação de out_dir).

