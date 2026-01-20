# STATUS PACK V110 — V110_CONTINUOUS_AGENCY_LONG_DIALOGUE_SURVIVAL (fix6)

## Resumo
- Adiciona **Executive/Agency Survival Layer V110** (wrapper do V109) para fechar o escape “goal aberto + sistema passivo”.
  - **Executive Ledger V110** WORM: `executive_events.jsonl` (hash-chain externo + cadeia interna `prev_event_sig/event_sig`) + snapshot derivado `executive_registry_snapshot_v110.json`.
  - **Invariantes V110 (hard gates)**:
    - **S6**: proibido passividade com goal aberto (sem avançar / sem pedir info necessária / sem fechar goal).
    - **S7**: detector determinístico de stall (2 turnos seguidos sem avanço nem pergunta necessária ⇒ repair forçado).
    - **S8**: bound anti “over-clarify” em pontos do smoke onde a referência já deveria estar resolvida.
  - Introspecção raw (fail‑closed): `executive`, `explain_executive <turn_index>`, `trace_executive <turn_index>`.
- Smoke V110 demonstra 2 cenários:
  - **Cenário A**: autopilot em diálogo longo (60 turns) com usuário minimalista (“ok/continua/isso”) ⇒ goal progride e fecha sem depender do humano dirigir.
  - **Cenário B**: stall controlado ⇒ invariantes disparam repair, depois progresso retoma e goal continua.

> Nota WORM: existem tentativas anteriores `fix1..fix5` em `results/`. O fechamento final está em `fix6` (sem overwrite).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Regressões smoke V90–V109 (out_base `results/run_smoke_*_regression_v110_fix6`)
- PASS Smoke V110 determinístico (try1==try2; inclui `executive_chain_hash_v110`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v110.py --run_dir results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1`
- PASS Negative tamper: `executive_events[0].event_sig="0"*64` ⇒ falha com reason `executive_event_sig_mismatch`

## Arquivos novos (V110)
- `atos_core/conversation_loop_v110.py:1` — wrapper V109; integra Executive Engine; persiste `executive_events.jsonl` + snapshot + hashes; aplica gates S6/S7/S8.
- `atos_core/conversation_v110.py:1` — `verify_conversation_chain_v110` (wrapper V109) + checks de `executive_*` + invariantes V110 + negative tamper.
- `atos_core/executive_engine_v110.py:1` — engine determinístico (score/flags/repair) para “no passivity with open goal”.
- `atos_core/executive_ledger_v110.py:1` — schema/append/verify/fold de `executive_events.jsonl` + `executive_chain_hash_v110`.
- `atos_core/intent_grammar_v110.py:1` — raw intercept `executive/explain_executive/trace_executive` (fail‑closed).
- `scripts/smoke_v110_continuous_agency_long_dialogue_survival_dialogue_csv.py:1` — smoke V110 (try1/try2 + negative tamper).
- `scripts/verify_conversation_chain_v110.py:1` — verificador CLI V110.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V109 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py --out_base results/run_smoke_v103_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py --out_base results/run_smoke_v104_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py --out_base results/run_smoke_v105_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv.py --out_base results/run_smoke_v106_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v107_pragmatics_intent_dialogue_state_survival_dialogue_csv.py --out_base results/run_smoke_v107_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v108_flow_discourse_fluency_memory_survival_dialogue_csv.py --out_base results/run_smoke_v108_regression_v110_fix6 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v109_semantic_concept_survival_dialogue_csv.py --out_base results/run_smoke_v109_regression_v110_fix6 --seed 0

# Smoke V110 (try1/try2 interno; WORM)
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v110_continuous_agency_long_dialogue_survival_dialogue_csv.py \
  --out_base results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6 --seed 0 \
  | tee results/smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1.json

# Verificador V110 (PASS)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v110.py --run_dir \
  results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1

# Verificador V110 — negative tamper (espera FAIL com reason executive_event_sig_mismatch)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v110.py --run_dir \
  results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1_tamper

# Patch diff (WORM) — somente arquivos V110 (ordem lexicográfica)
(diff -u /dev/null atos_core/conversation_loop_v110.py; \
 diff -u /dev/null atos_core/conversation_v110.py; \
 diff -u /dev/null atos_core/executive_engine_v110.py; \
 diff -u /dev/null atos_core/executive_ledger_v110.py; \
 diff -u /dev/null atos_core/intent_grammar_v110.py; \
 diff -u /dev/null scripts/smoke_v110_continuous_agency_long_dialogue_survival_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v110.py) \
 > patches/v110_continuous_agency_long_dialogue_survival_fix6.diff
```

## Determinismo (try1 == try2) — Smoke V110 (fix6)
- `seed`: `0`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `4c158da6f9c5ebb3a2ae7f5cb0ce8ac4b439b530cf865a221a84d7fa7b1553f5`
- `state_chain_hash`: `fe9c987fa07dad1e7f57aec9408f459427f3e7d3a6dafdbf150e35d24ec3e19e`
- `parse_chain_hash`: `08c01958e4aa19e4192ec531292f137afbc69862136fab794f57b3b65b34e4c8`
- `plan_chain_hash` (action_plans): `578a670073fa83d961dbd6fc51edf2443a94430c3a87e3a9f8ed58f033d239cc`
- `plan_events_chain_hash_v104`: `6f192f86bd8b7a1553244047df3644425b5994f2dbc36ad01f52684add78e52e`
- `agency_chain_hash_v105`: `89b06b5bd4fdafea79d45aae1f5af11c0dbaec98d0cd467d4f470ebda7e9998d`
- `dialogue_chain_hash_v106`: `73ccc8a162ee643d08a956c174dce05763d1caea68c07cdf9239d3ff77594142`
- `pragmatics_chain_hash_v107`: `77596977a7f34cccca3e60b2852383b3ee5e3b54aa6e2a1a9283318979678a05`
- `flow_chain_hash_v108`: `24b1e0b66a08e44b040ca708bf5de8e7cfd66c67ae32d118543678bedae38556`
- `semantic_chain_hash_v109`: `9d02d605e7d885caf20ba7e71f694f32a8b1870800570c6b6dbba72acfaf02c3`
- `executive_chain_hash_v110`: `17b8d0217bde79aa368e4517b698b5d7a4ffb9d02165f7072f184ccc62bea9d2`
- `ledger_hash` (sha256 do `freeze_manifest_v110.json`): `665e816732158f4191673bd5a7444d98f44e15780200ab3776fc209d58dc268a`
- Smoke stdout: `summary_sha256=1307b8e71d641cdc5260541a9c800625164b4bbc5746e8cc6895784c7e12edaa` e `eval_sha256=79ce03cb1c04d72953d8c33bc014c188ade1fce8d26cc6a767ddac9850ed2263`
- Métricas-chave (eval):
  - `goal1_exec_events_total`: `60`
  - `stall_events_total`: `1`
  - `progress_resume_events_total`: `4`

## Negative tamper
- PASS: `executive_events[0].event_sig="0"*64` ⇒ `verify_conversation_chain_v110` falha com reason `executive_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V110 try1: `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/`
- Smoke V110 try2: `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try2/`
- Tamper try1: `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1_tamper/`
- Stdout tee: `results/smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1.json`
- Patch diff: `patches/v110_continuous_agency_long_dialogue_survival_fix6.diff`

## SHA256 principais (artefatos V110)
- `patches/v110_continuous_agency_long_dialogue_survival_fix6.diff`: `b5ba4aa02b2683417551b35f3cd0559b2aab8107f42c6ba4c820f6df6b4523bd`
- `results/smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1.json`: `14e851664df5c58ac58454f85fe56bd44b31f8143b5733eb2dced95e92dd5896`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/summary.json`: `443745ff2d5128a126cd7e739f506dfb46833ebd56ffc422cfd7ae9560afc0ae`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/executive_events.jsonl`: `3fcd4c50864c90585ae145cde105dc6217dcf7e397f08fda3d6592be68ce3873`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/executive_registry_snapshot_v110.json`: `345219928e18a662136a3f28d26fae4f540e5d3252a81b42433f877fc13b0179`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/eval.json`: `79ce03cb1c04d72953d8c33bc014c188ade1fce8d26cc6a767ddac9850ed2263`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/smoke_summary.json`: `bbde52c5860a76bee4303cd30b671d17f424109135694c40d9e0edf853250b15`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/freeze_manifest_v110.json`: `665e816732158f4191673bd5a7444d98f44e15780200ab3776fc209d58dc268a`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/verify_chain_v110.json`: `46a13dfa13cc346d300c0c3f770d56bd565b59875c744b94570af5dcdb077579`
- `results/run_smoke_v110_continuous_agency_long_dialogue_survival_fix6_try1/transcript.jsonl`: `f1e716065444b6475f04ba4eb50475adf48e7a2acb30d67d26fb1db260661d5e`

## GO/NO-GO
- **GO** — regressões V90–V109 + testes passam; smoke V110 determinístico (try1==try2); autopilot (60 turns) mantém goal e fecha sem direção humana; stall controlado força repair e retoma; verificador V110 passa e negative tamper falha com reason correto.

