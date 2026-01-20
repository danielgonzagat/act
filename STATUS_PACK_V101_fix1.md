# STATUS PACK V101 — V101_BINDINGS_REFERENCE_RESOLUTION (fix1)

## Resumo
- Adiciona **Binding Ledger V101** (`binding_events.jsonl` + `binding_snapshot.json`) WORM hash-chained, com `event_sig` encadeado por evento e `binding_chain_hash`.
- Implementa **resolução determinística de referência** (pronomes/dêiticos) com fail-closed: `RESOLVED|AMBIGUOUS|MISS`, gerando eventos `BIND_RESOLVE|BIND_AMBIGUOUS|BIND_MISS`.
- Adiciona **introspecção**: `bindings`, `explain_binding <id>`, `trace_ref <turn_id>`.
- Integra no loop conversacional V101 com logs/manifest/summary e verificador V101.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS Regressões smoke V90–V100 (out_base `results/run_smoke_*_regression_v101_fix1`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Smoke V101 determinístico (try1==try2; inclui `binding_chain_hash`)
- PASS Verificador V101 ok no try1
- PASS Negative tamper: `binding_events[0].event_sig="0"*64` ⇒ falha com reason `binding_event_sig_mismatch`

## Arquivos novos (V101)
- `atos_core/binding_ledger_v101.py:1` — eventos/assinatura (`event_sig`) + snapshot/replay.
- `atos_core/bindings_v101.py:1` — modelo de binding + resolução determinística (RESOLVED/AMBIGUOUS/MISS).
- `atos_core/intent_grammar_v101.py:1` — intents/raw intercepts V101 (plan create/shorten/priority + bindings/explain/trace_ref).
- `atos_core/conversation_v101.py:1` — renderers + `verify_conversation_chain_v101`.
- `atos_core/conversation_loop_v101.py:1` — loop V101 com ledger de bindings + snapshot + hash no summary/manifest.
- `scripts/verify_conversation_chain_v101.py:1` — verificador CLI V101.
- `scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py:1` — smoke V101 (try1/try2 + tamper).

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (V90–V100)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v101_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v101_fix1 --seed 0

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V101 (try1/try2 interno; WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py \
  --out_base results/run_smoke_v101_bindings_reference_resolution_fix1 --seed 0 \
  | tee results/smoke_v101_bindings_reference_resolution_fix1_try1.json

# Verificador V101 (recomendado)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v101.py --run_dir \
  results/run_smoke_v101_bindings_reference_resolution_fix1_try1

# Patch diff (WORM) — somente arquivos V101
(diff -u /dev/null atos_core/binding_ledger_v101.py; \
 diff -u /dev/null atos_core/bindings_v101.py; \
 diff -u /dev/null atos_core/conversation_loop_v101.py; \
 diff -u /dev/null atos_core/conversation_v101.py; \
 diff -u /dev/null atos_core/intent_grammar_v101.py; \
 diff -u /dev/null scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v101.py) \
 > patches/v101_bindings_reference_resolution_fix1.diff
```

## Determinismo (try1 == try2)
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `d393af530f9700c5f70cdc401a650b740c2d57b7686d362d1a21696487c5b09c`
- `state_chain_hash`: `947bc9951b7be20ae54baab37db1d4c81ae5d224c8a8d5c3370e72c97dfe4303`
- `parse_chain_hash`: `14e41f573b2d0a1373e350b565c97442eba5f6bedd7b47e80dbb0c5dbd08fb26`
- `learned_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `plan_chain_hash`: `7078c6f1c0a52483b3f8ed9129213b2fea8067d9d6982711adfaaa7196ac1b58`
- `memory_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `belief_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `evidence_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `goal_chain_hash`: `467039b1276e97c0e646f01e404dda6c017a8bbe560bf7a628b2898a5fc5879b`
- `discourse_chain_hash`: `53aa76aede310fd79b90a239d775b6070c167b6071e8ddea046fbc9301bcec9d`
- `fragment_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `binding_chain_hash`: `41a03c4ae80675c6bc89ef56658ae9a887071ffabf50609a2852881a73309931`
- `ledger_hash` (sha256 do `freeze_manifest_v101.json`): `41808c4a7c18926efe817ac6d37371b54cc332a3813644cedac1122d6f72f9bd`
- `summary_sha256`: `10c186885947bfc06bb8553da0b00d7a3d98393f4b4f2db418989418fa865285`

## Negative tamper
- PASS: `binding_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v101` falha com reason `binding_event_sig_mismatch`.

## Smoke (eventos)
- `binding_events` (try1): `BIND_CREATE=3`, `BIND_RESOLVE=2`, `BIND_AMBIGUOUS=1`

## Artefatos (paths)
- Smoke V101 try1: `results/run_smoke_v101_bindings_reference_resolution_fix1_try1/`
- Smoke V101 try2: `results/run_smoke_v101_bindings_reference_resolution_fix1_try2/`
- Stdout tee: `results/smoke_v101_bindings_reference_resolution_fix1_try1.json`
- Patch diff: `patches/v101_bindings_reference_resolution_fix1.diff`

