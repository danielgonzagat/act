# STATUS PACK V112 — V112_FAMILY7_STRESS_FLUENCY_SURVIVAL (fix3)

## Resumo
- Implementa **Fluency-as-survival V112** como contrato determinístico (diversidade + higiene de “não sei” + anti-loop n-gram) aplicado no runner Family7.
- Fecha o deadlock determinístico do V110 em `OPÇÕES: ... Escolha A/B/C` sob usuário minimalista (`ok/continua/...`) via regra explícita: **ack → escolha padrão “A”** (auditável e determinística).
- Eleva Family7 DL‑A para **stress determinístico**: `tasks_total=20`, incluindo `STRESS_200=2` tasks longas; gera `fail_catalog_v112.json` para triagem.
- Exercita **gating hard do ExternalDialogueWorld**: negativos determinísticos + exatamente 1 acesso em-run, somente após `progress_blocked` por falha hard.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS `tasks_total >= 20` e `stress_200_total >= 2`
- PASS Smoke V112 determinístico (try1==try2: `eval_sha256` idêntico + `summary_sha256` idêntico)
- PASS `tasks_ok == tasks_total` (20/20)
- PASS `stress_200_ok == 2/2`
- PASS External world gating:
  - negativo: `external_world_access_not_allowed`
  - negativo: `invalid_reason_code`
  - em-run: exatamente 1 task com `external_world_events_total==1`
- PASS `scripts/verify_freeze.py --freeze LEDGER_...V112...json` (`ok=true`, `warnings=[]`)

## Arquivos (V112)
- `atos_core/external_world_gating_v112.py:1` — wrapper determinístico de gating para ExternalDialogueWorld (fail-closed).
- `atos_core/fluency_survival_v112.py:1` — contrato V112 (inclui refinamento determinístico para evitar falso-positivo de “não sei” em texto citado) + métricas n-gram/scaffold.
- `scripts/gen_family7_dla_from_history_v112.py:1` — gerador determinístico de tasks (20 tasks; 2× STRESS_200) + injection_plan.
- `scripts/run_family7_dla_v112.py:1` — runner determinístico (multi-attempt seeds) + 1 acesso externo permitido em cenário controlado + `fail_catalog_v112.json`.
- `scripts/smoke_v112_family7_stress_dialogue_survival.py:1` — smoke determinístico (try1/try2) + negativos gating + assert `tasks_ok==tasks_total`.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Unit tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V112 (try1/try2 interno; WORM)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v112_family7_stress_dialogue_survival.py \
  --tasks tasks/family7_dla_v112_seed0_fix1.jsonl \
  --out_base results/run_smoke_v112_family7_stress_fix4 \
  --seed 0 | tee results/smoke_v112_family7_stress_fix4_try1.json

# Patch diff (WORM)
(diff -u /dev/null atos_core/external_world_gating_v112.py; \
 diff -u /dev/null atos_core/fluency_survival_v112.py; \
 diff -u /dev/null scripts/gen_family7_dla_from_history_v112.py; \
 diff -u /dev/null scripts/run_family7_dla_v112.py; \
 diff -u /dev/null scripts/smoke_v112_family7_stress_dialogue_survival.py) \
 > patches/v112_fluency_survival_family7_stress_fix2.diff

# verify_freeze
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V112_FAMILY7_STRESS_FLUENCY_SURVIVAL_fix2.json \
  | tee results/verify_freeze_v112_fix2_try1.json
```

## Determinismo (try1 == try2)
- Smoke stdout `summary_sha256`: `65ca03e69b592b078b459ce5d5434afe498c7087adb268ffad22ee14634bf4f2`
- `results/run_smoke_v112_family7_stress_fix4_try1/eval.json#sha256 == results/run_smoke_v112_family7_stress_fix4_try2/eval.json#sha256`:
  - `eval_sha256=7afbfd6c1048bfea1caaaddd8f4b0786d113d9158b7621819e0b5d683f3e4f78`
- `tasks_ok=20` em ambos tries.

## External world gating (prova)
- Negativos (esperados):
  - `external_world_access_not_allowed`
  - `invalid_reason_code`
- Em-run: 1 evento em `results/run_smoke_v112_family7_stress_fix4_try1/task_000/attempt_001/external_world_events.jsonl` (reason_code=`validator_failed_fluency_contract`).

## Artefatos (paths)
- Smoke:
  - `results/run_smoke_v112_family7_stress_fix4_try1/`
  - `results/run_smoke_v112_family7_stress_fix4_try2/`
  - `results/smoke_v112_family7_stress_fix4_try1.json`
- Tasks:
  - `tasks/family7_dla_v112_seed0_fix1.jsonl`
  - `tasks/family7_dla_v112_seed0_fix1_manifest.json`
- Patch: `patches/v112_fluency_survival_family7_stress_fix2.diff`
- Ledger: `LEDGER_ATOLANG_V0_2_30_BASELINE_V112_FAMILY7_STRESS_FLUENCY_SURVIVAL_fix2.json`
- verify_freeze: `results/verify_freeze_v112_fix2_try1.json`

## SHA256 principais
- `patches/v112_fluency_survival_family7_stress_fix2.diff`: `356358e461fe579e1900d0995fd0f36c17653978f3bbfa8ffc884b65c60ae392`
- `STATUS_PACK_V112_fix3.md`: (ver ledger)
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V112_FAMILY7_STRESS_FLUENCY_SURVIVAL_fix2.json`: `842345ff46aaa51aa159f788e048a0ca8fe90ccd1b1724893cffdae6a61a2726`
- `results/verify_freeze_v112_fix2_try1.json`: `941f8476e5e041346a66908364aeb505ec1d68057b21e14df441f1ac762dd7a5`
- `results/smoke_v112_family7_stress_fix4_try1.json`: `41ec5892b8f52879b3416ba3aa613538fe5938171b6a3c7c489e296f54e78a7e`
- `tasks/family7_dla_v112_seed0_fix1.jsonl`: `c696de923183940859e6a4dba23778258e0ddc5f19368c0486ecb5de529dd5b7`
- `results/run_smoke_v112_family7_stress_fix4_try1/eval.json`: `7afbfd6c1048bfea1caaaddd8f4b0786d113d9158b7621819e0b5d683f3e4f78`

## GO/NO-GO
GO — Smoke ampliado 20/20 determinístico + stress_200 2/2 + gating external world exercitado e fail-closed nos negativos + verify_freeze ok (warnings=[]).

