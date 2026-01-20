# STATUS PACK V99 — V99_CONTINUOUS_AGENCY_GOAL_LEDGER — GO (fix1)

## Resumo
- Adiciona **Goal Ledger V99** (`goal_events.jsonl`, hash-chained) + snapshot derivado `goal_ledger_snapshot.json` e comandos raw: `goal:`, `goals`, `done <goal_id>`, `next`, `auto <n>`.
- Implementa **agência contínua observável**: `next/auto` executam ticks determinísticos que escrevem eventos no goal ledger e registram efeitos no `ActionPlan` (campos `goal_read_ids/goal_write_ids` + `provenance.goal_ids`).
- Estende **EXPLAIN** para incluir **efeitos em goals** (`GOAL_EFFECTS`) sem racionalização (somente campos do plan).
- Verificador V99: wrapper sobre V98, valida `goal_events.jsonl` (event_sig + prev_event_sig chain) + snapshot derivado + cross-check do texto de `goals/next/auto` contra renderers determinísticos.

## Checklist
- PASS: `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS: regressões V90–V98 (smokes) + `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS: smoke V99 try1==try2 (hashes idênticos, inclui `goal_chain_hash`)
- PASS: comandos V99 no smoke:
  - `goal:` cria goal_id e `goals` lista esse goal_id
  - `next` e `auto 2` geram subgoals (via `goal_write_ids` no ActionPlan) e escrevem `goal_events.jsonl`
- PASS: live learning (V97): `teach: statusz => beliefs` cria 1 learned rule e `statusz` usa `matched_rule_id` + `plan.provenance.learned_rule_ids`
- PASS: verdade operacional (V98): `belief: project = IAAA` + `evidence: project = IAAA_v2` ⇒ `beliefs` mostra apenas `IAAA_v2`; `why project` inclui `WAS/NOW/CAUSE`
- PASS: negative tamper V99: corrupção `goal_events[0].event_sig="0"*64` ⇒ fail-closed com reason determinístico `goal_event_sig_mismatch`

## Arquivos novos (V99)
- `atos_core/goal_ledger_v99.py:1`
- `atos_core/intent_grammar_v99.py:1`
- `atos_core/conversation_v99.py:1`
- `atos_core/conversation_loop_v99.py:1`
- `scripts/verify_conversation_chain_v99.py:1`
- `scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py:1`
- `patches/v99_continuous_agency_goal_ledger_fix1.diff:1`
- `STATUS_PACK_V99_fix1.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (V90–V98)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v99_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V99 (try1/try2 dentro do script; WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py \
  --out_base results/run_smoke_v99_continuous_agency_goal_ledger_fix1 --seed 0 \
  | tee results/smoke_v99_continuous_agency_goal_ledger_fix1_try1.json

# (Recomendado) Verificar invariantes/chain no run_dir try1
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v99.py --run_dir \
  results/run_smoke_v99_continuous_agency_goal_ledger_fix1_try1

# Patch diff (WORM) — somente arquivos V99
(diff -u /dev/null atos_core/conversation_loop_v99.py; \
 diff -u /dev/null atos_core/conversation_v99.py; \
 diff -u /dev/null atos_core/goal_ledger_v99.py; \
 diff -u /dev/null atos_core/intent_grammar_v99.py; \
 diff -u /dev/null scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v99.py) \
 > patches/v99_continuous_agency_goal_ledger_fix1.diff
```

## Determinismo (try1 == try2)
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `42cad9f271bca4d38e6b2693716b2616c17806a454c238b6086faa52500ba7b9`
- `state_chain_hash`: `3cc5a79ed0acc6ac443e4d53c6dbdfb334c7e9a27a9bc6754bc7bbb1e99efec1`
- `parse_chain_hash`: `f1e256a48d3ed55d9ccbc5ff30144f5965dfc9de3cdb75a8c071825aa14ba9ea`
- `learned_chain_hash`: `d07c83af7f1c0078100f5058074cf34e1497bcd4bff1657ef9e39e44590bd03f`
- `plan_chain_hash`: `db6d7a79d7b301effc59ee7aa9b40efe4e979d9ab629490b63cdfe883cce3ab0`
- `memory_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `belief_chain_hash`: `1b73923ac66ba6f5ce869acd8bd84473814b284393cd2856df5dcb0d4645607e`
- `evidence_chain_hash`: `63d16f7d7d34bad03e425c3cafd86d2ba5870d780dd03fea7c0c4ca811cba219`
- `goal_chain_hash`: `ebf9b511adc18104aad3ac9ac22bd65ef5d3e2d61c60feecb8fa6d950d0fc2f3`
- `ledger_hash` (sha256 do `freeze_manifest_v99.json`): `c126f93c2e3701a390dc2248cc1191e81d49721a83223f2d50ce6ddcfdff69cf`
- `summary_sha256`: `7f5750b24745f6ef0d2ddde0b248cdf506ea20fc59693b0477ba377d7cd4e0d9`

## Negative tamper
- PASS: `goal_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v99` falha com reason `goal_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V99 try1: `results/run_smoke_v99_continuous_agency_goal_ledger_fix1_try1/`
- Smoke V99 try2: `results/run_smoke_v99_continuous_agency_goal_ledger_fix1_try2/`
- Stdout tee: `results/smoke_v99_continuous_agency_goal_ledger_fix1_try1.json`
- Patch diff: `patches/v99_continuous_agency_goal_ledger_fix1.diff`

