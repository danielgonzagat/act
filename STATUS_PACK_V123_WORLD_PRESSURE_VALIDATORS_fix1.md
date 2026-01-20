# STATUS PACK V123 — WORLD_PRESSURE_VALIDATORS (fix1)

## Resumo
- Introduz “pressão seletiva do mundo” como gate hard (V123): IAC (intenção→ação→consequência) + anti‑regressão (falha repetida exige evidência/causal_diff) + reuso obrigatório (quando o mundo indica alternativa) + exaustão reforçada com o mundo.
- Implementa wrapper `run_conversation_v123` sobre `run_conversation_v121` que consulta o ExternalWorld V122 **somente quando progress está bloqueado** (deny‑by‑default preservado pelo gate V122).
- Adiciona runner/smokes determinísticos para Family7 (tasks v122) sob runtime V123 e mantém regressão v113 intacta.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS smoke V123 validators: determinismo try1==try2
- PASS smoke Family7 v122 sob V123: `tasks_ok==20/20` + determinismo try1==try2
- PASS regressão Family7 v113 (runner v113 intacto): `tasks_ok==20/20` + determinismo try1==try2

## Arquivos de código (V123)
- `atos_core/world_pressure_validators_v123.py:1` (reason codes + consult WORM via gate V122 + validadores IAC/anti‑regressão/reuso/exaustão)
- `atos_core/conversation_loop_v123.py:1` (wrapper runtime + gates hard V123 + artefatos `final_response_v123.json`/`world_pressure_summary_v123.json`)
- `scripts/run_family7_dla_v123.py:1` (runner Family7 sob runtime V123)
- `scripts/smoke_v123_world_pressure_validators.py:1` (smoke dos validadores + negativos deny/invalid_reason_code)
- `scripts/smoke_v123_family7_real_history_stress.py:1` (smoke determinístico try1/try2 para tasks v122 sob V123)
- `tests/test_world_pressure_validators_v123.py:1` (unit tests V123)

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v123_world_pressure_validators.py \
  --out_base results/run_smoke_v123_world_pressure_validators_fix1 \
  --seed 0 \
| tee results/smoke_v123_world_pressure_validators_fix1_try1.json

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v123_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v122_seed0.jsonl \
  --out_base results/run_smoke_v123_family7_real_history_stress_fix2 \
  --seed 0 \
| tee results/smoke_v123_family7_real_history_stress_fix2_try1.json

set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v113_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v113_regression_v123_fix1 \
  --seed 0 \
| tee results/smoke_v113_regression_v123_fix1_try1.json
```

## Determinismo (evidência)
- Smoke V123 validators: `results/smoke_v123_world_pressure_validators_fix1_try1.json`
  - `summary_sha256` = `ac2b64387df5bbfc2beb5b0b20da27def33b88ac0adcc032af45fb44ecd25bda`
- Smoke Family7 v122 sob V123: `results/smoke_v123_family7_real_history_stress_fix2_try1.json`
  - `summary_sha256` = `79951396b82ad6f35bf2f3f099d413e69ef46beff34f6823e6c93f928df6118c`
  - `results/run_smoke_v123_family7_real_history_stress_fix2_try1/eval.json` sha256 == `6f560afcd2052b767d23883a9e0cd45179e78c93d4ea5f3116be2503a5c79dd6`
  - `results/run_smoke_v123_family7_real_history_stress_fix2_try2/eval.json` sha256 == `6f560afcd2052b767d23883a9e0cd45179e78c93d4ea5f3116be2503a5c79dd6`
- Regressão v113 (runner v113): `results/smoke_v113_regression_v123_fix1_try1.json`
  - `summary_sha256` = `2916bce5c6a630dbcfef57653fdb961b2154bde2519602ce869a62a76eab4c03`

## Artefatos WORM + SHA256 (principais)
- Patch diff: `patches/v123_world_pressure_validators_fix1.diff` = `be42112a43f7163692fd2b86402c003c81551e1f2ac7a9d266858134ce35a008`
- Smoke stdout: `results/smoke_v123_world_pressure_validators_fix1_try1.json` = `c11ab37f93eeb50a84792cd121048b2ad1d240f752ff030d700570ab7a4aff71`
- Smoke stdout: `results/smoke_v123_family7_real_history_stress_fix2_try1.json` = `0248883d00c8d08abddf4a013094a2071d803d2aa18983f334600c2684b1bfe8`
- Tasks v122: `tasks/family7_dla_v122_seed0.jsonl` = `8148f7761ae776fc492c4f0b4a2a52879c2db8a1dbdd1a9484716de719ecfffd`
- Tasks v113: `tasks/family7_dla_v113_seed0_fix2.jsonl` = `72a91a0d4bb15b31ac4812e6b25d9c84ac9854f91b91ec493bf60face49773b1`

## GO/NO-GO
- GO: gates e smokes passam, determinismo try1==try2 comprovado, e V123 não introduz acesso implícito ao ExternalWorld (consulta só ocorre sob bloqueio de progresso via gate V122).

