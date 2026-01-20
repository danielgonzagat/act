# STATUS PACK V109 — V109_SEMANTIC_CONCEPT_SURVIVAL (fix7)

## Resumo
- Adiciona **Semantic Survival Layer V109** (wrapper do V108):
  - **Semantic Ledger V109** WORM: `semantic_events.jsonl` (hash-chain externo + cadeia interna `prev_event_sig/event_sig`) + snapshot derivado `semantic_registry_snapshot_v109.json`.
  - **Gating semântico** (invariantes V109): falha semântica observável ⇒ `progress_allowed_v109=false` + repair obrigatório (bloqueia goal/plan/agency).
  - Introspecção raw (fail‑closed): `semantics`, `explain_semantics <turn_index>`, `trace_semantics <turn_index>`.
  - Verificador V109 (wrapper do V108) com negative tamper determinístico.
- Smoke V109 prova (curto e incontestável):
  - **Cenário A**: ensinar 1 vez (2 positivos + 1 negativo) ⇒ conceito vira objeto e é aplicado em outro domínio (ToC PASS).
  - **Cenário B**: contradição (“prometo X” vs “não vou X”) ⇒ repair + gating (`progress_allowed_v109=false`).

> Nota WORM: existem tentativas anteriores `fix1..fix6` em `results/` (falhas/iterações). O fechamento está em `fix7` (sem overwrite).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Regressões smoke V90–V108 (out_base `results/run_smoke_*_regression_v109_fix7`)
- PASS Smoke V109 determinístico (try1==try2; inclui `semantic_chain_hash_v109`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v109.py --run_dir results/run_smoke_v109_semantic_concept_survival_fix7_try1`
- PASS Negative tamper: `semantic_events[0].event_sig="0"*64` ⇒ falha com reason `semantic_event_sig_mismatch`

## Arquivos novos (V109)
- `atos_core/conversation_loop_v109.py:1` — wrapper V108; executa Semantics Engine; persiste `semantic_events.jsonl` + snapshot + hashes; aplica gating V109.
- `atos_core/conversation_v109.py:1` — `verify_conversation_chain_v109` (wrapper V108) + checks de `semantic_*` + invariantes V109 + negative tamper.
- `atos_core/intent_grammar_v109.py:1` — raw intercept `semantics/explain_semantics/trace_semantics` (fail‑closed).
- `atos_core/semantics_engine_v109.py:1` — detector/flags/score/repair V109 (ensino/reuso de conceito + contradição/commitments).
- `atos_core/semantics_ledger_v109.py:1` — schema do ledger, renderers e fold/replay de snapshot + `semantic_chain_hash_v109`.
- `scripts/smoke_v109_semantic_concept_survival_dialogue_csv.py:1` — smoke V109 (try1/try2 + negative tamper).
- `scripts/verify_conversation_chain_v109.py:1` — verificador CLI V109.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V108 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py --out_base results/run_smoke_v103_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py --out_base results/run_smoke_v104_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py --out_base results/run_smoke_v105_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv.py --out_base results/run_smoke_v106_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v107_pragmatics_intent_dialogue_state_survival_dialogue_csv.py --out_base results/run_smoke_v107_regression_v109_fix7 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v108_flow_discourse_fluency_memory_survival_dialogue_csv.py --out_base results/run_smoke_v108_regression_v109_fix7 --seed 0

# Smoke V109 (try1/try2 interno; WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v109_semantic_concept_survival_dialogue_csv.py \\
  --out_base results/run_smoke_v109_semantic_concept_survival_fix7 --seed 0 \\
  | tee results/smoke_v109_semantic_concept_survival_fix7_try1.json

# Verificador V109 (PASS)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v109.py --run_dir \\
  results/run_smoke_v109_semantic_concept_survival_fix7_try1

# Verificador V109 — negative tamper (espera FAIL com reason semantic_event_sig_mismatch)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v109.py --run_dir \\
  results/run_smoke_v109_semantic_concept_survival_fix7_try1_tamper

# Patch diff (WORM) — somente arquivos V109 (ordem lexicográfica)
(diff -u /dev/null atos_core/conversation_loop_v109.py; \\
 diff -u /dev/null atos_core/conversation_v109.py; \\
 diff -u /dev/null atos_core/intent_grammar_v109.py; \\
 diff -u /dev/null atos_core/semantics_engine_v109.py; \\
 diff -u /dev/null atos_core/semantics_ledger_v109.py; \\
 diff -u /dev/null scripts/smoke_v109_semantic_concept_survival_dialogue_csv.py; \\
 diff -u /dev/null scripts/verify_conversation_chain_v109.py) \\
 > patches/v109_semantic_concept_survival_fix7.diff
```

## Determinismo (try1 == try2) — Smoke V109 (fix7)
- `seed`: `0`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `083dd7e6ab77032eae7cae890793285b2b1ccb73885d89238e0d87332482cd9d`
- `state_chain_hash`: `05f8ea84d56c3cff2dacf70aa5ed302efbb4a29e7eef374eb90d0469918fc6c2`
- `parse_chain_hash`: `d7ebb22984043b83ff132b95d7531455656dda0910457263f21a39a2e9a37ef4`
- `plan_chain_hash` (action_plans): `d07ccae87a1f73571f3e84f992bb70743caa01cd8cb3f37d0db985faa71e99a7`
- `dialogue_chain_hash_v106`: `0fcb6931393a5c249112f1e86fd67deb67c00718ac9686fdf65623675cf7f664`
- `pragmatics_chain_hash_v107`: `1a21a8872a961becaaf98b763eabb72751e346c880bbbbf7faeec603936346eb`
- `flow_chain_hash_v108`: `b760130aa14975ad02cabcb20a517e9fddb3b508750555150e90af74ed1e8577`
- `semantic_chain_hash_v109`: `403fc6864b5263e65c3fa7dee749e4a82a0e2017cc02f2477c49c92772046e9f`
- `ledger_hash` (sha256 do `freeze_manifest_v109.json`): `671ce932711b71a5060cef5b9ccc4e51eb2efecb9bde417b5c3d87abb09b904e`
- Smoke stdout: `summary_sha256=bc174c52110f0404418ad9d611558a12dffcb0c6f2f7d33e0cec0c2f290eee8e` e `eval_sha256=889063f25565d8acf1217eac77322a0c1d773ea41f896c2aa443a388353171f0`

## Negative tamper
- PASS: `semantic_events[0].event_sig="0"*64` ⇒ `verify_conversation_chain_v109` falha com reason `semantic_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V109 try1: `results/run_smoke_v109_semantic_concept_survival_fix7_try1/`
- Smoke V109 try2: `results/run_smoke_v109_semantic_concept_survival_fix7_try2/`
- Tamper try1: `results/run_smoke_v109_semantic_concept_survival_fix7_try1_tamper/`
- Stdout tee: `results/smoke_v109_semantic_concept_survival_fix7_try1.json`
- Patch diff: `patches/v109_semantic_concept_survival_fix7.diff`

## SHA256 principais (artefatos V109)
- `patches/v109_semantic_concept_survival_fix7.diff`: `bc166769346e3166e59e2adc30c4ab4f7f2da21eb08d288bdebda285b7cb1067`
- `results/smoke_v109_semantic_concept_survival_fix7_try1.json`: `88f39e8b7b367d3b12219d9eb9b6a245792b045e06857c63b598af372319d6e9`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/summary.json`: `86727294c051b058fc9a20056aba8149c114145d6be076ad19afc44a565f385e`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/semantic_events.jsonl`: `8220358bdbaf454f167ca82f45a2df368528d36d09366ad4c24002b37110132b`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/semantic_registry_snapshot_v109.json`: `52250f0fcbc71a894a49367ab3936f70d334118bcf7db1a179407792c5a9019d`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/eval.json`: `889063f25565d8acf1217eac77322a0c1d773ea41f896c2aa443a388353171f0`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/smoke_summary.json`: `124aadf32091183a86c0fa5eb06a4ed2f5ac6ed35acc6d47d226fcdb4e0dd371`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/freeze_manifest_v109.json`: `671ce932711b71a5060cef5b9ccc4e51eb2efecb9bde417b5c3d87abb09b904e`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/verify_chain_v109.json`: `6490fa3944aac47cdd9ecc8da3cb7b43478a2a040bcd1a60815db4661c36077a`
- `results/run_smoke_v109_semantic_concept_survival_fix7_try1/transcript.jsonl`: `568fc1d3ed41fa2d264442b81f27c1c6bcaa5f7009e2938368f0aad652208fdd`

## GO/NO-GO
- **GO** — regressões V90–V108 + testes passam; smoke V109 determinístico (try1==try2); Cenário A (conceito ensinado → reuso + ToC PASS) e Cenário B (contradição → repair + gating) exercitados; verificador V109 passa e negative tamper falha com reason correto.

