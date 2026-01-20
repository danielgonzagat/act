# STATUS PACK V122 — EXTERNAL WORLD UNIFIED (Conversations.json + Projeto ACT.rtf) + Family7 v122 (fix1)

## Resumo
- Constrói um ExternalWorld unificado e determinístico em `external_world_v122/` a partir de:
  - `~/Desktop/HISTORICO/conversations.json` (histórico) → `dialogue_history_canonical_v122.jsonl` (1 linha = 1 turno)
  - `~/Desktop/Projeto ACT.rtf` (engenharia) → `engineering_doc_plain_v122.txt` + `engineering_doc_chunks_v122.jsonl` (chunks determinísticos)
- Adiciona API read-only (`ew_load_and_verify`/`search`/`fetch`) + gate hard fail-closed com reasons enumerados; toda consulta gera `external_world_events_v122.jsonl` + evidences.
- Gera `tasks/family7_dla_v122_seed0.jsonl` (20 tasks) a partir do histórico canônico, sem embeddings/clustering/LLM, e roda smoke determinístico try1==try2.
- Mantém regressão Family7 v113 (histórico real) 20/20 (smoke determinístico) e mantém negativos do gating (not_allowed/invalid_reason_code).

## Checklist (PASS/FAIL)
- PASS build world determinístico (smoke V122 build determinism)
- PASS gate deny-by-default + invalid_reason_code (smoke V122)
- PASS gate allow-only-when-forced + manifest tamper (smoke V122)
- PASS Family7 v122: `tasks_ok==20/20` e determinismo try1==try2
- PASS regressão Family7 v113: `tasks_ok==20/20` e determinismo try1==try2
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`

## Entradas (Desktop) + SHA256
- `/Users/danielpenin/Desktop/HISTORICO/conversations.json` = `cc3637ad1bc0f954dd879110a96cebb0800a8c67111dea989a5e2f894b950cb9`
- `/Users/danielpenin/Desktop/Projeto ACT.rtf` = `a8db26927b05b8b28550f45ae4eea5ea44a22e6d3c9ae05aa8ff713f85154387`

## Arquivos de código (novos)
- `scripts/build_external_world_v122.py:1` (builder streaming + manifest WORM)
- `atos_core/external_world_v122.py:1` (API read-only + verify manifest + anti-copy signal)
- `atos_core/external_world_gate_v122.py:1` (gate hard + ledger/evidence WORM + chain hash)
- `scripts/build_family7_dla_tasks_v122.py:1` (gera `tasks/family7_dla_v122_seed0.jsonl`)
- `scripts/smoke_v122_external_world_build_deterministic.py:1`
- `scripts/smoke_v122_external_world_gate_deny.py:1`
- `scripts/smoke_v122_external_world_gate_allow.py:1`
- `scripts/smoke_v122_family7_real_history_stress.py:1`
- `tests/test_external_world_v122_build.py:1`
- `tests/test_external_world_v122_gate.py:1`

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Inputs (desktop)
find "$HOME/Desktop" -maxdepth 3 -iname "Conversations.json" -print
find "$HOME/Desktop" -maxdepth 2 -iname "Projeto ACT.rtf" -print
sha256sum "$HOME/Desktop/HISTORICO/conversations.json" "$HOME/Desktop/Projeto ACT.rtf"

# Build ExternalWorld v122
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/build_external_world_v122.py \
  --conversations_input "$HOME/Desktop/HISTORICO/conversations.json" \
  --rtf_input "$HOME/Desktop/Projeto ACT.rtf" \
  --out external_world_v122

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke: deterministic build (content hash)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v122_external_world_build_deterministic.py \
  --conversations_input "$HOME/Desktop/HISTORICO/conversations.json" \
  --rtf_input "$HOME/Desktop/Projeto ACT.rtf" \
  --out1 external_world_v122 \
  --out2 external_world_v122_try3 \
| tee results/smoke_v122_external_world_build_deterministic_fix2_try1.json

# Smoke: gate deny-by-default
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v122_external_world_gate_deny.py \
  --manifest external_world_v122/manifest_v122.json \
| tee results/smoke_v122_external_world_gate_deny_fix1_try1.json

# Smoke: gate allow-only-when-forced + manifest tamper
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v122_external_world_gate_allow.py \
  --manifest external_world_v122/manifest_v122.json \
  --out_base results/run_smoke_v122_external_world_gate_allow_fix1 \
  --seed 0 \
| tee results/smoke_v122_external_world_gate_allow_fix1_try1.json

# Build tasks v122
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/build_family7_dla_tasks_v122.py \
  --world_manifest external_world_v122/manifest_v122.json \
  --seed 0 \
  --out tasks/family7_dla_v122_seed0.jsonl

# Smoke: Family7 v122 (try1/try2 determinismo interno)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v122_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v122_seed0.jsonl \
  --out_base results/run_smoke_v122_family7_real_history_stress_fix1 \
  --seed 0 \
| tee results/smoke_v122_family7_real_history_stress_fix1_try1.json

# Regressão: Family7 v113 intacto (try1/try2 determinismo interno)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v113_family7_real_history_stress.py \
  --tasks tasks/family7_dla_v113_seed0_fix2.jsonl \
  --out_base results/run_smoke_v122_regression_v113_family7 \
  --seed 0 \
| tee results/smoke_v122_regression_v113_family7_try1.json
```

## Determinismo (evidência)
- Build determinism (V122): `results/smoke_v122_external_world_build_deterministic_fix2_try1.json`
  - `summary_sha256` = `6f489fe1485ecf2d4e28941881f4312d6575c43e50b302b194b9e6fb6b84c57a`
- Gate allow (V122): `results/smoke_v122_external_world_gate_allow_fix1_try1.json`
  - `summary_sha256` = `40a716c84479b3686a9019e1d0502a21f9b2922e667bca283be5840b49bdfece`
- Family7 v122: `results/smoke_v122_family7_real_history_stress_fix1_try1.json`
  - `summary_sha256` = `ac0b60ae854713ff1e5b53f154b03acfb4c4100d5243b160241a68538baf8284`
  - `results/run_smoke_v122_family7_real_history_stress_fix1_try1/eval.json` sha256 == `d2c86ca7e7f0674870b1d73bfdde1f7c9471311e5225c926789212f91dcae265`
  - `results/run_smoke_v122_family7_real_history_stress_fix1_try2/eval.json` sha256 == `d2c86ca7e7f0674870b1d73bfdde1f7c9471311e5225c926789212f91dcae265`
- Regressão v113: `results/smoke_v122_regression_v113_family7_try1.json`
  - `summary_sha256` = `2916bce5c6a630dbcfef57653fdb961b2154bde2519602ce869a62a76eab4c03`

## Artefatos WORM + SHA256
- Patch diff: `patches/v122_external_world_unified_family7_fix1.diff` = `21e47052560bd0a09a1ae3d4b4e7d915ba4489459bcb6358c879c3dab88ac16b`
- World manifest: `external_world_v122/manifest_v122.json` = `401ac845d0238d07ebe459d8cb0341eb42f13f1c249fc35c06cef35493f7d527`
- World canonical turns: `external_world_v122/dialogue_history_canonical_v122.jsonl` = `c273fdf26093b6cd1634478495d2326727f2c081ec023ae5ac607acdb53a551e`
- Engineering plain text: `external_world_v122/engineering_doc_plain_v122.txt` = `ed4b02049e8caf4333b9302fe49c0b1034b167e4f44f45a2ddc16379aac0acb9`
- Engineering chunks: `external_world_v122/engineering_doc_chunks_v122.jsonl` = `51483fa43e86a9d2b8562520aa0768f82dca578a0e615926b915599d00921dd8`
- Tasks v122: `tasks/family7_dla_v122_seed0.jsonl` = `8148f7761ae776fc492c4f0b4a2a52879c2db8a1dbdd1a9484716de719ecfffd`
- Tasks v122 manifest: `tasks/family7_dla_v122_seed0.jsonl.manifest.json` = `b75f6e8a249ce522abbdd0b3f441379d12c062312944d1f8727fdc0b0620e2d0`
- Smoke stdout (build determinism): `results/smoke_v122_external_world_build_deterministic_fix2_try1.json` = `c7a555394e379c22877eff07d44345412de1ba39dc397d2c80f01eae5a156f30`
- Smoke stdout (gate deny): `results/smoke_v122_external_world_gate_deny_fix1_try1.json` = `633e0dd62166d9a0e0cef3bfa52bb19bdf0181193eee261115f4d22cdb55a202`
- Smoke stdout (gate allow): `results/smoke_v122_external_world_gate_allow_fix1_try1.json` = `fbe1691b7c8389b44349d708d53ae61c60f468abfb2a0ad515aaf70b77af4351`
- Gate allow try1 events: `results/run_smoke_v122_external_world_gate_allow_fix1_try1/external_world_events_v122.jsonl` = `93c6163a3ab3ce947b8c3542889d7a7e2644732d88db8cc1e8dc5c7cc0e2cb48`
- Gate allow try1 evidence: `results/run_smoke_v122_external_world_gate_allow_fix1_try1/external_world_evidence_v122.jsonl` = `54970051e4ab96e1481041568463b292625603cc209e79543ee4442ef2566e8c`
- Gate allow try1 snapshot: `results/run_smoke_v122_external_world_gate_allow_fix1_try1/external_world_registry_snapshot_v122.json` = `ee5f962173221216ce145bf054c35a76b829be9e977dcc72617ef8fe3410eeff`
- Smoke stdout (Family7 v122): `results/smoke_v122_family7_real_history_stress_fix1_try1.json` = `e8853908fd4449c4e24bf98ff740afad75a4e3e33926ac74b41a27a2964cddb2`
- Family7 v122 try1 eval: `results/run_smoke_v122_family7_real_history_stress_fix1_try1/eval.json` = `d2c86ca7e7f0674870b1d73bfdde1f7c9471311e5225c926789212f91dcae265`
- Family7 v122 try1 summary: `results/run_smoke_v122_family7_real_history_stress_fix1_try1/summary.json` = `71e94309c20fc1b53d650c3fc9b7eb9e36d13c81e83e3f30ec2a82deb2b350e3`

## GO / NO-GO
- GO: ExternalWorld unificado (histórico + doc) com manifest e hashes determinísticos, gate fail-closed com reasons enumerados + evidences WORM, tasks Family7 v122 geradas do mundo canônico e smoke 20/20 com determinismo try1==try2, e regressão Family7 v113 intacta.

