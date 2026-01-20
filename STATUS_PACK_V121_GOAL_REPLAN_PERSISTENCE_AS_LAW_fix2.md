# STATUS PACK V121 — GOAL REPLAN PERSISTENCE AS LAW (fix2)

## Resumo
- Adiciona a lei V121 “goal não pode morrer no FAIL”: se `chosen_ok=false` e ainda existem candidatos ranqueados dentro do budget, isso vira violação determinística (`goal_died_on_fail_v121`).
- Exige prova de exaustão: ao declarar exaustão, `attempted_actions[*].eval_id` deve existir em `objective_evals.jsonl`, senão falha (`exhaustion_without_proof_v121`).
- Integra V121 como gate hard em `atos_core/conversation_loop_v121.py` (tentativas determinísticas por seed), preservando determinismo/WORM.
- Regressão Family7 v113 (histórico real) sob runtime V121: 20/20 e determinismo try1==try2.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS smoke `scripts/smoke_v121_goal_replan_until_exhausted.py` determinístico (try1==try2)
- PASS negative tamper (smoke V121): `exhaustion_without_proof_v121`
- PASS smoke `scripts/smoke_v121_family7_real_history_stress.py` determinístico (try1==try2)
- PASS Family7 v113 sob V121: `tasks_ok==20/20`
- PASS negativos do ExternalWorld gating (no smoke Family7): `external_world_access_not_allowed` e `invalid_reason_code`

## Arquivos de código (novos)
- `atos_core/goal_replan_persistence_law_v121.py:1` (verificador + `goal_replan_persistence_summary_v121.json`)
- `atos_core/conversation_loop_v121.py:1` (wrapper com gate hard V121 no loop de tentativas)
- `scripts/smoke_v121_goal_replan_until_exhausted.py:1` (smoke do verificador + tamper)
- `scripts/run_family7_dla_v121.py:1` (runner Family7 usando `run_conversation_v121`)
- `scripts/smoke_v121_family7_real_history_stress.py:1` (smoke Family7 v113 sob runtime V121)
- `tests/test_goal_replan_persistence_law_v121.py:1` (unit tests do verificador V121)

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v121_goal_replan_until_exhausted.py \
  --out_base results/run_smoke_v121_goal_replan_until_exhausted_fix2 --seed 0 \
| tee results/smoke_v121_goal_replan_until_exhausted_fix2_try1.json

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v121_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v121_family7_real_history_stress_fix2 --seed 0 \
| tee results/smoke_v121_family7_real_history_stress_fix2_try1.json

# Patch diff (ordem determinística; somente V121)
(
  diff -u /dev/null atos_core/goal_replan_persistence_law_v121.py; \
  diff -u /dev/null atos_core/conversation_loop_v121.py; \
  diff -u /dev/null scripts/smoke_v121_goal_replan_until_exhausted.py; \
  diff -u /dev/null scripts/smoke_v121_family7_real_history_stress.py; \
  diff -u /dev/null scripts/run_family7_dla_v121.py; \
  diff -u /dev/null tests/test_goal_replan_persistence_law_v121.py \
) > patches/v121_goal_replan_persistence_as_law_fix2.diff
```

## Determinismo (evidência)
- Smoke V121 (verificador):
  - `summary_sha256` = `9ef174af226b11a51feeaad58e601996f9de53f16407c2d0be1776b67793468a`
- Smoke Family7 V121:
  - `summary_sha256` = `adff4cbfb4fea2fd5472c970f91d4bb5dd30bb58289c8e60e7f26cc5f7ba1f34`
  - `results/run_smoke_v121_family7_real_history_stress_fix2_try1/eval.json` sha256 == `9e70354065327ec3f68ea7028c3253695a981e23effb74a2c992263d090ef290`
  - `results/run_smoke_v121_family7_real_history_stress_fix2_try2/eval.json` sha256 == `9e70354065327ec3f68ea7028c3253695a981e23effb74a2c992263d090ef290`

## Artefatos WORM + SHA256
- Patch diff: `patches/v121_goal_replan_persistence_as_law_fix2.diff` = `0bd33d8577edf0152cc417f9d0cb62660978a2b24e03e4f2835028ea90d1ca7d`
- Smoke stdout (goal replan): `results/smoke_v121_goal_replan_until_exhausted_fix2_try1.json` = `6d97b836038b962ba8d3864937438ee9e84235134d25bec64a159563423ebff1`
- Smoke try1 eval (goal replan): `results/run_smoke_v121_goal_replan_until_exhausted_fix2_try1/eval.json` = `5cf1834d0bf07cb917814e65d0db2586ba8f98dc6fae1c27e351e4e355b0ba87`
- Smoke try1 smoke_summary (goal replan): `results/run_smoke_v121_goal_replan_until_exhausted_fix2_try1/smoke_summary.json` = `b70157e9aca6db435d83edc56ca8e2cffbcfe553f6a38418ba75b8ff624b329e`
- Smoke stdout (Family7): `results/smoke_v121_family7_real_history_stress_fix2_try1.json` = `7cfabf9d74a33bf97ddbec76c0c880fe936ee6d2cfca8f1a60b1e7b61de5e8b4`
- Family7 try1 eval: `results/run_smoke_v121_family7_real_history_stress_fix2_try1/eval.json` = `9e70354065327ec3f68ea7028c3253695a981e23effb74a2c992263d090ef290`
- Family7 try1 summary: `results/run_smoke_v121_family7_real_history_stress_fix2_try1/summary.json` = `3b1f2ce1741c5792d139e5e05f6f14001e838b5c4709434c0c2e71d4dbd8f613`

## GO / NO-GO
- GO: a lei V121 impede FAIL “cedo” quando ainda há candidatos ranqueados sob budget, e exige prova auditável de exaustão; Family7 v113 passa 20/20 com determinismo try1==try2.

