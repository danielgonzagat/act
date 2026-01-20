# STATUS PACK V105 — V105_CONTINUOUS_AGENCY_GOAL_PLAN_LOOP (fix3)

## Resumo
- Adiciona **Agency Ledger V105** (`agency_events.jsonl`) WORM (hash-chain externo + cadeia interna `prev_event_sig/event_sig`) e snapshot derivado (`agency_registry_snapshot_v105.json`).
- Implementa **Agency Engine V105** (1 decisão por turno) conectado a **Goal Ledger V99** + **Plan Engine/Ledger V104** + renderização via **V102 templates** (somente para pergunta curta/continuação humana).
- Adiciona introspecção V105 (raw intercept): `agency`, `explain_agency <turn_index>`, `trace_agency <turn_index>`.
- Adiciona verificador V105 (wrapper do V104) com checks de:
  - cadeia interna de `agency_events` + snapshot,
  - render determinístico de `agency/explain_agency/trace_agency` a partir do prefixo do ledger,
  - negative tamper (fail-closed).

> Nota WORM: tentativas anteriores consumiram `fix1` e `fix2` (smokes abortados); este baseline fecha em `fix3`.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Regressões smoke V90–V104 (out_base `results/run_smoke_*_regression_v105_fix3`)
- PASS Smoke V105 determinístico (try1==try2; inclui `agency_chain_hash_v105`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v105.py --run_dir results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1`
- PASS Negative tamper: `agency_events[0].event_sig="0"*64` ⇒ falha com reason `agency_event_sig_mismatch`

## Arquivos novos (V105)
- `atos_core/agency_ledger_v105.py:1` — `AgencyEventV105` + `event_sig` + `agency_chain_hash_v105` + snapshot + renderers.
- `atos_core/intent_grammar_v105.py:1` — raw intercept `agency/explain_agency/trace_agency` (fail-closed).
- `atos_core/conversation_v105.py:1` — `verify_conversation_chain_v105` (wrapper V104) + invariantes do agency ledger.
- `atos_core/conversation_loop_v105.py:1` — loop V105 + `agency_events.jsonl` + snapshot + hashes no manifest/summary.
- `scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py:1` — smoke V105 (try1/try2 + tamper).
- `scripts/verify_conversation_chain_v105.py:1` — verificador CLI V105.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V104 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py --out_base results/run_smoke_v103_regression_v105_fix3 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py --out_base results/run_smoke_v104_regression_v105_fix3 --seed 0

# Smoke V105 (try1/try2 interno; WORM) + tee
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py \
  --out_base results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3 --seed 0 \
  | tee results/smoke_v105_continuous_agency_goal_plan_loop_fix3_try1.json

# Verificador V105
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v105.py --run_dir \
  results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1

# Patch diff (WORM) — somente arquivos V105
(diff -u /dev/null atos_core/agency_ledger_v105.py; \
 diff -u /dev/null atos_core/conversation_loop_v105.py; \
 diff -u /dev/null atos_core/conversation_v105.py; \
 diff -u /dev/null atos_core/intent_grammar_v105.py; \
 diff -u /dev/null scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v105.py) \
 > patches/v105_continuous_agency_goal_plan_loop_fix3.diff
```

## Determinismo (try1 == try2) — Smoke V105
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `974c51da3ec97495d7a69ba040f541ed53b21215bc3b207b7a07b53cdb85d61c`
- `state_chain_hash`: `22e4d4980f41251eb2091687eaac7597c19f1dad8f495554100d2403c92709e8`
- `parse_chain_hash`: `4fc13f4c2400dd6c795470116e10cc1a46de2636cd2ad8125461f0fe10df537c`
- `plan_chain_hash` (action_plans): `7fd64587535a657a59c7514f12c19e6fa94bcb1e27111ed70948237d21d0b6bd`
- `plan_events_chain_hash_v104`: `bd3d428e3f3926f16aa8e9df713f5bfb2eb631086506c820c8e44d534601b56b`
- `agency_chain_hash_v105`: `49fcdc978dc5145bea21374db84c022606416aacd58609a303a808e73a9de053`
- `ledger_hash` (sha256 do `freeze_manifest_v105.json`): `6950639a0c1cdc6e1591f9aeec12b569dcdba54f16fb17e4c37d79137adc7681`
- `summary_sha256`: `e6b0cfd8d0f69f255b11edb5570834411b87924778272193508beb2f7c3dbc9c`

## Negative tamper
- PASS: `agency_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v105` falha com reason `agency_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V105 try1: `results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1/`
- Smoke V105 try2: `results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try2/`
- Stdout tee: `results/smoke_v105_continuous_agency_goal_plan_loop_fix3_try1.json`
- Patch diff: `patches/v105_continuous_agency_goal_plan_loop_fix3.diff`

## SHA256 principais
- `patches/v105_continuous_agency_goal_plan_loop_fix3.diff`: `6294da5f6e462e2de9ffc8ae6e6f06485ce459cf9a03117e8df0f3de3d721d44`
- `results/smoke_v105_continuous_agency_goal_plan_loop_fix3_try1.json`: `06f7471dab64d6814507890df6dbda2399fdecbcc249dfaee98390aaa36783ce`
- `results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1/summary.json`: `27256d7e6e73f4887d01ce5716bd5e47dc05d108a991140dbbb65638d0799b17`
- `results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1/agency_events.jsonl`: `17c61735fb4e298626658a1ef5875a4ece805add29c60a777e6bbe9ec7effb3a`
- `results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1/agency_registry_snapshot_v105.json`: `1fcfff9161e60329997b54496c0cfedd27f1ae529cb6c9806f82f686ede44f6f`
- `results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1/freeze_manifest_v105.json`: `6950639a0c1cdc6e1591f9aeec12b569dcdba54f16fb17e4c37d79137adc7681`
- `results/run_smoke_v105_continuous_agency_goal_plan_loop_fix3_try1/verify_chain_v105.json`: `c902a1955e00fb14fe2209a2bfb587f0590562b2b27fb94589df7a3e8f5626f9`

## GO/NO-GO
- **GO** — smoke determinístico (try1==try2), negative tamper ok, regressões ok, e verificador V105 passa.

