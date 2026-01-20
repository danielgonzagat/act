# STATUS PACK V103 — V103_CONCEPT_SUBGRAPHS_TOC_MDL_ENGINE (fix1)

## Resumo
- Adiciona **Concept Engine V103**: conceitos como objetos vivos com CSG (estrutura imutável) + CSV (estado derivado por eventos) e indução determinística por MDL.
- Introduz **Concept Ledger V103** (`concept_events.jsonl`) com `prev_event_sig/event_sig` (cadeia interna) + `concept_chain_hash` em `summary.json` e `freeze_manifest_v103.json`.
- Implementa **TEACH_CONCEPT** (+/− exemplos), **INDUCE_CONCEPT** automático (>=2 positivos e >=1 negativo), **match** e **ToC PASS/FAIL** (distância de domínio determinística), com **prune** por thresholds.
- Adiciona introspecção V103: `concepts`, `explain_concept <id|name>`, `trace_concepts <turn_id|turn_index>`.
- Adiciona verificador V103 (wrapper do V102) com checks de `concept_events` + snapshot derivado `concept_library_snapshot_v103.json` + negative tamper.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS Regressões smoke V90–V102 (out_base `results/run_smoke_*_regression_v103_fix1`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Smoke V103 determinístico (try1==try2; inclui `concept_chain_hash`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v103.py --run_dir results/run_smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv_fix1_try1`
- PASS Negative tamper: `concept_events[0].event_sig="0"*64` ⇒ falha com reason `concept_event_sig_mismatch`

## Arquivos novos/modificados (V103)
- `atos_core/concept_model_v103.py:1` — CSG canônico + hashing + builder `make_csg_rule_v103`.
- `atos_core/concept_engine_v103.py:1` — extração de features discretas + indução MDL determinística + match + ToC distance.
- `atos_core/concept_ledger_v103.py:1` — `ConceptEventV103` + `event_sig` + `concept_chain_hash`.
- `atos_core/concept_registry_v103.py:1` — fold/replay do ledger → registry + snapshot + render views.
- `atos_core/intent_grammar_v103.py:1` — raw intercepts `teach_concept`, `concepts`, `explain_concept`, `trace_concepts`.
- `atos_core/conversation_v103.py:1` — `verify_conversation_chain_v103` (wrapper V102) + invariantes V103.
- `atos_core/conversation_loop_v103.py:1` — loop V103 + `concept_events.jsonl` + snapshot + hashes no manifest/summary.
- `scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py:1` — smoke V103 (try1/try2 + tamper).
- `scripts/verify_conversation_chain_v103.py:1` — verificador CLI V103.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V102 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v103_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v103_fix1 --seed 0

# Smoke V103 (try1/try2 interno; WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py \
  --out_base results/run_smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv_fix1 --seed 0 \
  | tee results/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv_fix1_try1.json

# Verificador V103 (recomendado)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v103.py --run_dir \
  results/run_smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv_fix1_try1

# Patch diff (WORM) — somente arquivos V103
(diff -u /dev/null atos_core/concept_engine_v103.py; \
 diff -u /dev/null atos_core/concept_ledger_v103.py; \
 diff -u /dev/null atos_core/concept_model_v103.py; \
 diff -u /dev/null atos_core/concept_registry_v103.py; \
 diff -u /dev/null atos_core/conversation_loop_v103.py; \
 diff -u /dev/null atos_core/conversation_v103.py; \
 diff -u /dev/null atos_core/intent_grammar_v103.py; \
 diff -u /dev/null scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v103.py) \
 > patches/v103_concept_subgraphs_toc_mdl_engine_fix1.diff
```

## Determinismo (try1 == try2)
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `f8d1c3f316c198eb1017f4405dd50b2528bd7429de0258ab65ca094e8c6ebb69`
- `state_chain_hash`: `07aa8d36441b352c88a405c9080d3e72b1aea524164d2973292c2d767989c08d`
- `parse_chain_hash`: `50d01dac4f3dc1f9a271fd7ebccd79be893d6d1bccdd8729ca709a1595a97b4a`
- `learned_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `plan_chain_hash`: `68a07bedfddbd83f7840b832bcf87ac8a023c76563d40c379731aa015db453ba`
- `binding_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `style_chain_hash`: `a1a958e6dc53718f0f0e3cb5ba78ef1e28c339d41f75ed2ea8de58c0e09f9ff2`
- `concept_chain_hash`: `e262b909da04de2b685a8e5050421cb96264d0df994b8ab72bbe162d55c6a279`
- `ledger_hash` (sha256 do `freeze_manifest_v103.json`): `02a613c42eaedac2f9821a75917146ca681aaa6aa5fa06935c2ce54a5fb2deac`
- `summary_sha256`: `0776ba840c86e020a55393c965b009659132decd8c2edbe2c5a771519443b9e5`

## Negative tamper
- PASS: `concept_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v103` falha com reason `concept_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V103 try1: `results/run_smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv_fix1_try1/`
- Smoke V103 try2: `results/run_smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv_fix1_try2/`
- Stdout tee: `results/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv_fix1_try1.json`
- Patch diff: `patches/v103_concept_subgraphs_toc_mdl_engine_fix1.diff`

