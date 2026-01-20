# STATUS PACK V116 — “Dialogue Survival as Law” (final_response_v116 + FAIL_EVENT ATO) — GO/NO-GO

## Resumo
- Implementa **V116 como wrapper sobre V115**: além da lei **GOAL→PLAN→EVAL(SATISFIES=true)**, agora **só é permitido “ok” se `dialogue_survival_ok==true`** (derivado determinística e auditavelmente de sinais já existentes: **fluency v112 + unresolved_reference final (flow v108) + contradição (semantic v109)**).
- Quando `dialogue_survival_ok==false`, o wrapper **força FAIL** (escreve `final_response_v116.json` com `FAIL: <reason>`) e grava um **FAIL_EVENT_V116** como **ATO** em `mind_graph_v116/`, com evidência apontando para `dialogue_survival_summary_v116.json`.
- Adiciona runner/smoke Family7 v116 para executar as 20 tasks reais do histórico sob runtime V116 (determinismo try1==try2), mantendo gating externo intacto (negativos e 1 acesso in-cycle).

## Checklist (PASS/FAIL)
- PASS `py_compile` (atos_core + scripts).
- PASS `unittest -v` (inclui `tests/test_dialogue_survival_gate_v116.py`).
- PASS smoke V116 “dialogue_survival render gate”: determinismo try1==try2 + negativos (fluency fail / unresolved_reference).
- PASS regressão Family7 v113 (tasks reais do histórico) sob runtime V116: `tasks_ok==tasks_total==20` e determinismo try1==try2 por `eval_sha256`.
- PASS gating externo intacto (negativos not_allowed / invalid_reason_code + exatamente 1 task com `external_world_events_total==1`).

## Arquivos novos (código)
- `atos_core/dialogue_survival_gate_v116.py:1` (gate determinístico + `dialogue_survival_summary_v116.json` write-once).
- `atos_core/conversation_loop_v116.py:1` (wrapper `run_conversation_v115` + `final_response_v116.json` + `mind_graph_v116/` + FAIL_EVENT_V116).
- `scripts/smoke_v116_dialogue_survival_render_gate.py:1` (smoke determinístico try1/try2 com casos positivo/negativos).
- `scripts/run_family7_dla_v116.py:1` (runner Family7 v116: usa runtime V116 + schedule determinístico + gating externo).
- `scripts/smoke_v116_family7_real_history_stress.py:1` (smoke determinístico try1/try2 para Family7 sob runtime V116).
- `tests/test_dialogue_survival_gate_v116.py:1` (unit tests da função pura `decide_dialogue_survival_v116`).

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V116 (dialogue_survival como lei) — try1/try2 interno
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v116_dialogue_survival_render_gate.py --seed 0 \
  | tee results/smoke_v116_dialogue_survival_render_gate_try1.json

# Regressão Family7 v113 (tasks reais do histórico) sob runtime V116 — fix1 por WORM
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v116_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v116_family7_real_history_stress_fix1 --seed 0 \
  | tee results/smoke_v116_family7_real_history_stress_fix1_try1.json

# Patch diff (WORM)
(diff -u /dev/null atos_core/conversation_loop_v116.py; \
 diff -u /dev/null atos_core/dialogue_survival_gate_v116.py; \
 diff -u /dev/null scripts/run_family7_dla_v116.py; \
 diff -u /dev/null scripts/smoke_v116_dialogue_survival_render_gate.py; \
 diff -u /dev/null scripts/smoke_v116_family7_real_history_stress.py; \
 diff -u /dev/null tests/test_dialogue_survival_gate_v116.py) \
 > patches/v116_dialogue_survival_as_law.diff
```

## Métricas principais
- Smoke V116 dialogue_survival render gate:
  - `summary_sha256 = e71b5a295813b493f790fdec85a6958cddb0b49ed6127d5b470962868dea12b7`
  - determinismo: `eval_sha256 try1 == try2 == 9dc3ffedb8dbdacbe92ee0d3a8119dc0bb9450ef68a603f7d494df3b805e9f80`
- Smoke Family7 v116 (tasks reais do histórico):
  - `tasks_total=20`, `tasks_ok=20`
  - determinismo: `eval_sha256 try1 == try2 == 2ad7fe3db5260d6a89ca8b37d76e90d3531765ac5ac76fc72cd809ee9854efda`
  - `summary_sha256 = 06418b0c1920482ca7efe3e23da7fb09b0eb1f46c8d47232684a2e3cc0e72d7b`
  - gating externo in-cycle: exatamente 1 task com `external_world_events_total==1` (negativos stable).

## Artefatos (paths) + SHA256
- Patch diff: `patches/v116_dialogue_survival_as_law.diff` = `ca2189b4e86ee81439b463eb8ce9342301a5e6ab56050ecc9f5bd76c50926be7`
- Smoke V116 dialogue_survival render gate:
  - stdout tee: `results/smoke_v116_dialogue_survival_render_gate_try1.json` = `6d8e55ec9a844302b3c037990ac110635df7b3049f7d0a2690bc42a259cb27d6`
  - try1 eval: `results/run_smoke_v116_dialogue_survival_render_gate_try1/eval.json` = `9dc3ffedb8dbdacbe92ee0d3a8119dc0bb9450ef68a603f7d494df3b805e9f80`
  - try1 summary: `results/run_smoke_v116_dialogue_survival_render_gate_try1/summary.json` = `c3af1db52179f53164fccbbc8cf8643572a409068c2f86688ce877b29c80fd21`
- Smoke Family7 v116:
  - stdout tee: `results/smoke_v116_family7_real_history_stress_fix1_try1.json` = `1007563eeb52dc56eeb7664543159205a573aa1fe9575ccd736af5318cd82dd2`
  - try1 eval: `results/run_smoke_v116_family7_real_history_stress_fix1_try1/eval.json` = `2ad7fe3db5260d6a89ca8b37d76e90d3531765ac5ac76fc72cd809ee9854efda`
  - try1 summary: `results/run_smoke_v116_family7_real_history_stress_fix1_try1/summary.json` = `309b963e53d0b0ef1e93c44f6ad4597812199459fa7bb9d5e1f85ce094f23fd2`
- Family7 tasks file (input): `tasks/family7_dla_v113_seed0_fix2.jsonl` = `72a91a0d4bb15b31ac4812e6b25d9c84ac9854f91b91ec493bf60face49773b1`

## GO/NO-GO
- GO

