# STATUS PACK V98 — V98_OPERATIONAL_TRUTH_AGENCY — GO (fix1)

## Resumo
- Adiciona **Evidence Ledger V98** (`evidence_events.jsonl`, hash-chained) e comandos raw: `evidence:`, `evidences`, `why <key>`, `versions`, `dossier|regulatory|compliance`.
- Implementa **revisão de crença dirigida por evidência**: `evidence: k=v` sempre grava evidência e, se necessário, revisa crença via `RETRACT(old)+ADD(new)` no mesmo step, com `cause_evidence_id`.
- Estende **ActionPlan** para carregar `evidence_read_ids/evidence_write_ids` e `provenance.cause_evidence_ids`, e atualiza **EXPLAIN** para mostrar TOP‑K + efeitos (memory/belief/evidence) + proveniência, sem racionalização.
- Verificador V98: wrapper sobre V97, valida `evidence_events.jsonl` (event/item sig) + consistência `belief_events.cause_evidence_*` ↔ evidência + cross-check do texto de `evidences`/`why` contra renderers determinísticos.

## Checklist
- PASS: `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS: regressões V90–V97 (smokes) + `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS: smoke V98 try1==try2 (hashes idênticos, inclui `evidence_chain_hash`)
- PASS: política de verdade operacional:
  - `belief: project = IAAA` + `evidence: project = IAAA_v2` ⇒ `beliefs` mostra apenas `IAAA_v2`
  - `why project` referencia `CAUSE: evidence_id=...` e inclui `WAS/NOW`
- PASS: live learning (TEACH) continua funcionando e aparece em `plan.provenance.learned_rule_ids`
- PASS: `versions` e `dossier` são determinísticos e verificáveis via renderers
- PASS: negative tamper: corrupção `evidence_events[0].event_sig="0"*64` ⇒ fail-closed com reason determinístico `evidence_event_sig_mismatch`

## Arquivos novos (V98)
- `atos_core/intent_grammar_v98.py:1`
- `atos_core/conversation_v98.py:1`
- `atos_core/conversation_loop_v98.py:1`
- `scripts/verify_conversation_chain_v98.py:1`
- `scripts/smoke_v98_operational_truth_agency_dialogue_csv.py:1`
- `patches/v98_operational_truth_agency_fix1.diff:1`
- `STATUS_PACK_V98_fix1.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (V90–V97)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v98_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V98 (try1/try2 dentro do script; WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py \
  --out_base results/run_smoke_v98_operational_truth_agency_fix1 --seed 0 \
  | tee results/smoke_v98_operational_truth_agency_fix1_try1.json

# Patch diff (WORM) — somente arquivos V98
(diff -u /dev/null atos_core/conversation_loop_v98.py; \
 diff -u /dev/null atos_core/conversation_v98.py; \
 diff -u /dev/null atos_core/intent_grammar_v98.py; \
 diff -u /dev/null scripts/smoke_v98_operational_truth_agency_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v98.py) \
 > patches/v98_operational_truth_agency_fix1.diff
```

## Determinismo (try1 == try2)
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `5ec30955a1ae16e015b04e5adac6c698f79edadd9b3ab27b507d97e7c053fc4b`
- `state_chain_hash`: `ba497ae7fcafe55de46e5c00a99316a0d546382b3e96fae2d877d78cb50dab84`
- `parse_chain_hash`: `c4baacc5d2cb545a09e290c2bcd2ca268a442876d5303671bbfa27e3b1615df1`
- `learned_chain_hash`: `d07c83af7f1c0078100f5058074cf34e1497bcd4bff1657ef9e39e44590bd03f`
- `plan_chain_hash`: `32319f53be0d4b719aaac674e42ed2a6ee940eac40524da9a694214ed464f947`
- `memory_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `belief_chain_hash`: `2bfb56576a41b9fe636fb8bb91a663359792fe7ff045223d2d621ca2fd5b7968`
- `evidence_chain_hash`: `cffc2d9bb7229127e73fb6bd1306f919959e92b0d19e0fa4cb0b29b0363eed99`
- `ledger_hash` (sha256 do `freeze_manifest_v98.json`): `b6ae3277a30dbe81e6bf0dba8b658abbd06e5f7429b81e433045ab5f31453d45`
- `summary_sha256`: `358a9c113fcda0bbb4cf23c0b1d510896a3911e1994e6ac44f2cc87f123c99c7`

## Negative tamper
- PASS: `evidence_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v98` falha com reason `evidence_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V98 try1: `results/run_smoke_v98_operational_truth_agency_fix1_try1/`
- Smoke V98 try2: `results/run_smoke_v98_operational_truth_agency_fix1_try2/`
- Stdout tee: `results/smoke_v98_operational_truth_agency_fix1_try1.json`
- Patch diff: `patches/v98_operational_truth_agency_fix1.diff`

