# STATUS PACK V112 — V112_FAMILY7_STRESS_FLUENCY_SURVIVAL (fix2)

## Resumo
- Adiciona “fluency-as-survival” V112: contrato determinístico de fluência (diversidade + higiene de “não sei” + n-grams) + métricas auditáveis.
- Fecha deadlock determinístico do V110 em “OPÇÕES A/B/C” sob replies minimalistas (ok/continua/etc.) via mapeamento explícito (ack → escolha padrão “A”) no runner V112 (somente para Family7).
- Eleva Family7 DL‑A para regime “stress”: 20 tasks determinísticas (inclui 2× STRESS_200), com fail_catalog reprodutível.
- Exercita gating hard do ExternalDialogueWorld (negativos determinísticos + 1 acesso permitido somente após progress_blocked).

## Checklist (PASS/FAIL)
- PASS `py_compile` (atos_core + scripts)
- PASS `python -m unittest -v`
- PASS Family7 v112 tasks: `tasks_total=20` e `stress_200>=2`
- PASS Smoke V112 determinístico `try1 == try2` (eval_sha256 idêntico; summary_sha256 idêntico)
- PASS `tasks_ok == tasks_total` (20/20)
- PASS `stress_200_ok == 2/2`
- PASS External world gating:
  - negativo: `external_world_access_not_allowed`
  - negativo: `invalid_reason_code`
  - em-run: exatamente 1 task com `external_world_events_total==1`

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
```

## Métricas (Smoke V112)
- `tasks_total=20`, `tasks_ok=20`
- `stress_200_ok=2/2`
- `external_world_in_cycle_calls_total=1` (apenas após falha hard determinística no attempt0 do task marcado `allow_external_world_once=true`)
- `summary_sha256=65ca03e69b592b078b459ce5d5434afe498c7087adb268ffad22ee14634bf4f2`
- `eval_sha256=7afbfd6c1048bfea1caaaddd8f4b0786d113d9158b7621819e0b5d683f3e4f78` (try1==try2)

## Arquivos adicionados/modificados (V112)
- `atos_core/external_world_gating_v112.py:1` — wrapper determinístico de gating para acesso ao ExternalDialogueWorld (negativos: not_allowed/invalid_reason).
- `atos_core/fluency_survival_v112.py:1` — contrato V112 + métricas n-gram/scaffold + plano determinístico de re-run seeds.
- `scripts/gen_family7_dla_from_history_v112.py:1` — gerador determinístico de tasks (20 tasks; 2× STRESS_200).
- `scripts/run_family7_dla_v112.py:1` — runner determinístico (multi-attempt seeds) + 1 acesso externo permitido em cenário controlado + fail_catalog.
- `scripts/smoke_v112_family7_stress_dialogue_survival.py:1` — smoke determinístico (try1/try2) + negativos gating + asserts (inclui `tasks_ok==tasks_total`).

## Artefatos (paths)
- Runs:
  - `results/run_smoke_v112_family7_stress_fix4_try1/`
  - `results/run_smoke_v112_family7_stress_fix4_try2/`
- Smoke stdout (tee): `results/smoke_v112_family7_stress_fix4_try1.json`
- Tasks: `tasks/family7_dla_v112_seed0_fix1.jsonl`, `tasks/family7_dla_v112_seed0_fix1_manifest.json`
- Fail catalog: `results/run_smoke_v112_family7_stress_fix4_try1/fail_catalog_v112.json`

## GO/NO-GO
GO (Smoke V112 20/20 determinístico + gating do external world exercitado e fail-closed nos negativos).

