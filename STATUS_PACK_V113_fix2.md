# STATUS PACK V113 — REAL HISTORY ExternalDialogueWorld + FAMILY7 DL‑A v113 (fix2) — GO/NO‑GO

## Resumo
- Canonicaliza o export `conversations.json` do ChatGPT em `external_world/dialogue_history_canonical_v113.jsonl` (ordem temporal global por `create_time`, determinística) + `external_world/dialogue_history_canonical_v113_manifest.json`.
- Implementa API read‑only do mundo externo (`atos_core/external_dialogue_world_v113.py`) e gating determinístico (`atos_core/external_world_gating_v113.py`) com reasons enumerados e falhas negativas estáveis.
- Gera `tasks/family7_dla_v113_seed0_fix2.jsonl` (20 tasks, 2×STRESS_200, 1×STRESS_500) a partir do mundo canônico v113, com injeções adversariais determinísticas.
- Roda smoke V113 com try1/try2 e prova determinismo (eval sha256 idêntico) + prova de gating (1 task com 1 acesso; negativos not_allowed/invalid_reason_code).

## Checklist (PASS/FAIL)
- PASS `py_compile` (atos_core + scripts).
- PASS `unittest -v`.
- PASS canonicalização v113 gerada (jsonl + manifest) e SHA256 registrado.
- PASS generator v113: `tasks_total=20`, `stress_200_total=2`, `stress_500_total=1`.
- PASS smoke v113: `tasks_ok==tasks_total` e determinismo try1==try2 por `eval_sha256`.
- PASS gating (negativos): `external_world_access_not_allowed` e `invalid_reason_code`.
- PASS external world in‑cycle: exatamente 1 task com `external_world_events_total==1`.

## Arquivos novos (código)
- `scripts/canonicalize_chatgpt_export_v113.py:1` (canonicalização do export → jsonl v113 + manifest).
- `atos_core/external_dialogue_world_v113.py:1` (API read‑only: `FETCH_TURN`, `OBSERVE_RANGE`, `SEARCH`).
- `atos_core/external_world_gating_v113.py:1` (gating determinístico + events compatíveis com ledger v111).
- `scripts/gen_family7_dla_from_history_v113.py:1` (gerador Family7 v113 a partir do mundo v113).
- `scripts/run_family7_dla_v113.py:1` (runner determinístico + fluency survival v112 + 1 acesso externo sob gating).
- `scripts/smoke_v113_family7_real_history_stress.py:1` (smoke try1/try2 + asserts + negativos de gating).

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Canonicalização do export (auto‑detecta caminhos padrão se --input não for fornecido)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/canonicalize_chatgpt_export_v113.py --out external_world

# Tasks v113 (fix2)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/gen_family7_dla_from_history_v113.py \
  --world_manifest external_world/dialogue_history_canonical_v113_manifest.json \
  --seed 0 --out tasks/family7_dla_v113_seed0_fix2.jsonl --tasks_total 20 --stress_200 2 --stress_500 1

# Smoke v113 (try1/try2 interno)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v113_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v113_family7_real_history_stress_fix2 \
  --seed 0 | tee results/smoke_v113_family7_real_history_stress_fix2_try1.json
```

## Métricas principais (smoke)
- `tasks_total=20`, `tasks_ok=20`
- `stress_200_total=2`, `stress_500_total=1`
- determinismo: `eval_sha256 try1 == try2 == 792cc75e9b6626a93571dcd8f86aa8d6c02ddc57677178d08474cc8e2e5f024d`
- `summary_sha256 (smoke) = 2916bce5c6a630dbcfef57653fdb961b2154bde2519602ce869a62a76eab4c03`
- gating in‑cycle: 1 task com `external_world_events_total==1`

## Artefatos (paths) + SHA256
- Patch diff: `patches/v113_real_history_external_world_family7_stress_fix2.diff` = `4debec24d8589d846b23488c9602531ed8748f98c245d1616da312f5fab3a795`
- World v113:
  - `external_world/dialogue_history_canonical_v113.jsonl` = `4a5d9d61a858f9f9079a179041a05d908037783c95266130afc22586dcbf7c28`
  - `external_world/dialogue_history_canonical_v113_manifest.json` = `0a8e4276a27f6dc7cd97d155628fc221d7a5a688a0bb09b88025dbde6353c1b7`
- Tasks v113:
  - `tasks/family7_dla_v113_seed0_fix2.jsonl` = `72a91a0d4bb15b31ac4812e6b25d9c84ac9854f91b91ec493bf60face49773b1`
  - `tasks/family7_dla_v113_seed0_fix2.jsonl_manifest.json` = `e3c28b45c150c648ecc2c17aca1525737b30ce198eb97f62d7f2799cf2b972a0`
- Smoke stdout (tee): `results/smoke_v113_family7_real_history_stress_fix2_try1.json` = `1fe588fc317c37e2b6057d594f9f04c85270519620d7aef5cd25d67747f59301`
- Smoke run try1:
  - `results/run_smoke_v113_family7_real_history_stress_fix2_try1/eval.json` = `792cc75e9b6626a93571dcd8f86aa8d6c02ddc57677178d08474cc8e2e5f024d`
  - `results/run_smoke_v113_family7_real_history_stress_fix2_try1/summary.json` = `2bc504e652f46c8b902220a016b2aacccec4d9acf973dd69de987882988045c1`
  - `results/run_smoke_v113_family7_real_history_stress_fix2_try1/fail_catalog_v113.json` = `9a6f4e1eec02e270393c138d29f43f36e7834df270346884d118dbc128d1af30`

## GO/NO‑GO
- GO. (Fix2: fix1 falhou por `allow_external_world_once=false` em todas as tasks; corrigido no gerador e re‑executado WORM.)

