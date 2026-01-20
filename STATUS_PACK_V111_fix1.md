# STATUS PACK V111 — V111_EXTERNAL_DIALOGUE_WORLD + FAMILY7_DL-A + FLUENCY-AS-SURVIVAL (fix1)

## Resumo
- Adiciona **ExternalDialogueWorld V111** (mundo externo read-only) com pipeline determinístico:
  - `conversations.json` → `dialogue_history_canonical_v111.jsonl` + `offsets_v111.bin` + `conversations_index_v111.json` + `world_manifest_v111.json`.
  - API de acesso determinístico por offset (`fetch_turn`, `observe_range`, `search_text`).
- Adiciona **External World Ledger V111** (WORM) para acesso gated ao mundo externo:
  - `external_world_events.jsonl` (cadeia `prev_event_sig/event_sig`) + snapshot `external_world_registry_snapshot_v111.json`.
  - Gating hard por `reason_code` enumerado + modo “repair/blocked” (sem “muleta” em cenário normal).
- Adiciona **FAMILY 7 — DL‑A** (diálogo longo adversarial) usando o histórico como fonte de caos (somente turnos reais do usuário + injeções determinísticas), sem usar respostas históricas como alvo.
- Adiciona **Fluency contract V111** como condição binária por tarefa (sem LLM/embeddings; métricas determinísticas).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Build world determinístico + verify: `results/verify_external_world_v111.json` (`ok=true`)
- PASS Smoke gating (fix2): `external_world_events_total==1`, chain ok, determinismo ok; negative tamper → `external_world_event_sig_mismatch`
- PASS Family7 smoke (fix5): `tasks_ok==tasks_total==2`, determinismo ok; **external world unused** (`events_total==0`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v111.py --run_dir results/run_smoke_v111_family7_dla_fix5_try1`

## Arquivos novos (V111)
- `atos_core/external_dialogue_world_v111.py:1` — loader + fetch por offsets + observe/search determinísticos.
- `atos_core/external_world_ledger_v111.py:1` — ledger WORM do mundo externo (event_sig chain + snapshot + chain hash).
- `atos_core/fluency_contract_v111.py:1` — contrato determinístico de fluência (binário + métricas).
- `atos_core/intent_grammar_v111.py:1` — comandos/intents do V111 (inclui atos gated do mundo externo).
- `atos_core/conversation_loop_v111.py:1` — wrapper do loop (integra gating/ledger V111 quando acionado).
- `atos_core/conversation_v111.py:1` — verificação wrapper (inclui no-hybridization check e mundo externo).
- `scripts/build_external_dialogue_world_v111.py:1` — build world (streaming JSON; WORM copy; manifest).
- `scripts/verify_external_world_v111.py:1` — verificador do world (manifest + offsets + amostras).
- `scripts/gen_family7_dla_from_history_v111.py:1` — gera tasks Family7 DL‑A determinísticas (seed 0).
- `scripts/run_family7_dla_v111.py:1` — runner determinístico (try1==try2; freeze_manifest v111 por tarefa).
- `scripts/smoke_v111_external_world_gating.py:1` — smoke gating + negative tamper.
- `scripts/smoke_v111_family7_dla_dialogue_survival.py:1` — smoke Family7 DL‑A (2 tasks; determinismo).
- `scripts/verify_conversation_chain_v111.py:1` — verificador CLI (wrapper de V110 por tarefa + checks V111).

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Build world (path real do export no host)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/build_external_dialogue_world_v111.py \
  --input /Users/danielpenin/Desktop/HISTORICO/conversations.json \
  --out external_world

# Verify world
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_external_world_v111.py \
  --manifest external_world/manifests/world_manifest_v111.json \
  | tee results/verify_external_world_v111.json

# Generate Family7 tasks (fix1)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/gen_family7_dla_from_history_v111.py \
  --world_manifest external_world/manifests/world_manifest_v111.json \
  --seed 0 --out tasks/family7_dla_v111_seed0_fix1.jsonl

# Smoke gating (fix2)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v111_external_world_gating.py \
  --out_base results/run_smoke_v111_external_world_gating_fix2 \
  --seed 0 \
  --world_manifest external_world/manifests/world_manifest_v111.json \
  | tee results/smoke_v111_external_world_gating_fix2_try1.json

# Smoke Family7 (fix5)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v111_family7_dla_dialogue_survival.py \
  --tasks tasks/family7_dla_v111_seed0_fix1.jsonl \
  --out_base results/run_smoke_v111_family7_dla_fix5 \
  --seed 0 \
  | tee results/smoke_v111_family7_dla_fix5_try1.json

# Verify chain (try1)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v111.py \
  --run_dir results/run_smoke_v111_family7_dla_fix5_try1 \
  | tee results/verify_conversation_chain_v111_family7_fix5_try1.json
```

## Determinismo (try1 == try2)
- Smoke gating (fix2): `external_world_chain_hash_v111=0650c6aaf8ff151bcea2b0db3a5695113cb1099ea73c7427db731f68ba90c2cc` e `summary_sha256=e21e271159a06f3bad5bb33482cafe9b0fcb22ef7ad919c054f511970c7c1283`.
- Smoke Family7 (fix5): `eval_sha256=c294ce51fb462caa7cc9c34a3437f538a6c48380da04cb4189783cd44e5f8aea`, `try1.summary_sha256==try2.summary_sha256==bf5c1ce0a9a61c6ae4b9d93e1f3b668520d19762e677f4edd954cc28f9f5a8e3`, `summary_sha256=10a28c6f17cb7957f52a11c5a07a4798b427980f848cef930684cb6a14a7f799`.

## Negative tamper
- PASS (esperado): corromper `external_world_events.jsonl` no smoke gating ⇒ `external_world_event_sig_mismatch`.

## SHA256 principais
- `patches/v111_external_dialogue_world_family7_dla_fluency_survival_fix1.diff`: `dc24128da125ece21d89c087dbaf66a1e65a56e78b083866486907f390728095`
- `external_world/manifests/world_manifest_v111.json`: `6157ef077b2ff83207a351755cb50e20820300da372b57fefeb07d3cd6a91d01`
- `tasks/family7_dla_v111_seed0_fix1.jsonl`: `8c3a2bd932d72b3037c13c7c283abc7136d78325ab19465695bd71b954d40325`
- `results/smoke_v111_external_world_gating_fix2_try1.json`: `61dbb8d8b954914479eb2a7f5a6ad2c8df28dedb90704aab7f59cdfe21367f7f`
- `results/smoke_v111_family7_dla_fix5_try1.json`: `300fdee45f0cbf0e5a5f89f0b8f33ab75dce5783db559f6e9f6076e3cfe601c7`
- `results/verify_external_world_v111.json`: `260461b6302280dbee0f5a613e02f2a5960d59759234602a2580f2d9699d9adf`
- `results/verify_conversation_chain_v111_family7_fix5_try1.json`: `077f60bf0c20bd25729fc6af88ed4e48b0c8597f5f667b434d0f8c16f30ce2a3`

## GO/NO-GO
- **GO** — world build/verify determinísticos; gating hard demonstrado + negative tamper; Family7 DL‑A smoke passa com external world unused; fluency contract aplicado; determinismo try1==try2; verificador V111 passa.
