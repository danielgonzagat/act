# STATUS PACK V120 — GOAL PERSISTENCE AS LAW (fix1)

## Resumo
- Adiciona a lei V120 de persistência de goal: qualquer `GOAL_DONE` com `cause_type=="system"` precisa provar horizonte via marcador visível `<step>/<total>` no texto do assistente e não pode fechar antes do horizonte configurado.
- Integra essa lei como gate hard em `atos_core/conversation_loop_v120.py`: só aceita um attempt se passar V115 (goal-plan-eval), V116 (dialogue survival), `fluency_contract_v118` e V120 (goal persistence).
- Parametriza (retrocompatível) o horizonte de autopilot do V110 via `goal_autopilot_total_steps` para permitir smokes de 100+ turns sem editar código.
- Smoke determinístico (try1==try2) com negative tamper que corrompe o marcador de progresso e exige falha estável.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v` (39 tests)
- PASS smoke `scripts/smoke_v120_goal_survival_100_turns.py` determinístico (try1==try2)
- PASS negative tamper: `goal_done_before_horizon_v120`

## Arquivos de código (novos)
- `atos_core/goal_persistence_law_v120.py:1` (verificador V120 + `goal_persistence_summary_v120.json`)
- `atos_core/conversation_loop_v120.py:1` (wrapper com gate hard: V115 + V116 + fluency v118 + V120)
- `scripts/smoke_v120_goal_survival_100_turns.py:1` (smoke 100 turns + determinismo + tamper)
- `tests/test_goal_persistence_law_v120.py:1` (unit tests do verificador V120)

## Arquivos alterados (retrocompatíveis)
- `atos_core/conversation_loop_v110.py:988` (parâmetro opcional `goal_autopilot_total_steps` default=60; sem regressão de comportamento)

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v120_goal_survival_100_turns.py \
  --out_base results/run_smoke_v120_goal_survival_100_turns_fix3 --seed 0 --total_steps 100 \
| tee results/smoke_v120_goal_survival_100_turns_fix3_try1.json

# Patch diff (ordem determinística)
(
  diff -u patches/v120_base/atos_core/conversation_loop_v110.py atos_core/conversation_loop_v110.py; \
  diff -u /dev/null atos_core/conversation_loop_v120.py; \
  diff -u /dev/null atos_core/goal_persistence_law_v120.py; \
  diff -u /dev/null scripts/smoke_v120_goal_survival_100_turns.py; \
  diff -u /dev/null tests/test_goal_persistence_law_v120.py \
) > patches/v120_goal_persistence_as_law_fix1.diff
```

## Determinismo (evidência)
- Smoke (fix3): try1==try2
  - `summary_sha256` = `f8b47bcbe2dc8f92ac7a6cd07dcc09ba81b3a46281ae29b47c8815ff90c2746f`
  - `results/run_smoke_v120_goal_survival_100_turns_fix3_try1/eval.json` sha256 == `4205de167f30728f33365befe823c0698211eb4e613ce6945d68ed76af7fd184`
  - `results/run_smoke_v120_goal_survival_100_turns_fix3_try2/eval.json` sha256 == `4205de167f30728f33365befe823c0698211eb4e613ce6945d68ed76af7fd184`

## Artefatos WORM + SHA256
- Patch diff: `patches/v120_goal_persistence_as_law_fix1.diff` = `b77ccd704334afabfe8c474a32c17c45036691a76e895b5974ddfa2695f5358a`
- Smoke stdout: `results/smoke_v120_goal_survival_100_turns_fix3_try1.json` = `272110b9c7874cb67f5742c38f4869a2fe48a91b6e3c320feef065fa2b94e366`
- Smoke try1 eval: `results/run_smoke_v120_goal_survival_100_turns_fix3_try1/eval.json` = `4205de167f30728f33365befe823c0698211eb4e613ce6945d68ed76af7fd184`
- Smoke try1 smoke_summary: `results/run_smoke_v120_goal_survival_100_turns_fix3_try1/smoke_summary.json` = `22866c3f1fb69a402c32b278f011ad61ba82f4e4680bacae6af9f2798f740d3c`
- Smoke try1 summary: `results/run_smoke_v120_goal_survival_100_turns_fix3_try1/summary.json` = `701081368ab1c699851a45afc2eabf76b20a653e15ad90d7af7d1d355b5d3986`
- Goal persistence summary: `results/run_smoke_v120_goal_survival_100_turns_fix3_try1/goal_persistence_summary_v120.json` = `72b045bb721dc3c9cd26903f9b02dfb40556b188cc9bfa1e07ca019d142dc127`
- Final response: `results/run_smoke_v120_goal_survival_100_turns_fix3_try1/final_response_v120.json` = `51d993586b08f6073907fd454faae6ba64e4c51a05915917cdfe079e65a31edc`

## GO / NO-GO
- GO: a persistência de goal passa a ser gate hard e tem prova determinística (marker `<step>/<total>`), com smoke determinístico + tamper fail-closed.

