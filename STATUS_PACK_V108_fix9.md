# STATUS PACK V108 — V108_FLOW_DISCOURSE_FLUENCY_MEMORY_SURVIVAL (fix9)

## Resumo
- Adiciona **FLOW Ledger V108** (`flow_events.jsonl`) WORM (hash-chain externo + cadeia interna `prev_event_sig/event_sig`) e snapshot derivado (`flow_registry_snapshot_v108.json`), com `flow_chain_hash_v108` incluído em `summary.json` + `freeze_manifest_v108.json`.
- Implementa **Flow Engine V108** determinístico:
  - `DiscourseAct` planner (OPEN/ACK/ANSWER/CLARIFY/REPAIR/PIVOT_SOFT/SUMMARY/NEXT_STEP/CLOSE),
  - críticos determinísticos de fluência/fluxo/memória conversacional (ex.: `UNRESOLVED_REFERENCE`, `ABRUPT_TOPIC_SHIFT`, `REPETITION_LOOP`),
  - **DSC (Deterministic Soft Choice)** para escolher entre candidatos elite sem softmax/LLM.
- Introduz **invariantes de sobrevivência V108**:
  - **S4**: `flow_score_v108 < 70` ou flags críticas ⇒ `progress_allowed_v108=false` e `repair_action_v108!=NONE` (bloqueia progresso de goal/plan/agency).
  - **S5**: referência não resolvida / pendências envelhecidas ⇒ bloqueia progresso e força repair (`CLARIFY_REFERENCE`/`SUMMARY_CONFIRM`).
- Adiciona introspecção raw V108 (fail-closed): `flow`, `explain_flow <turn_index>`, `trace_flow <turn_index>`.
- Adiciona verificador V108 (wrapper do V107) com checks de:
  - cadeia interna de `flow_events`,
  - snapshot reproduzível por fold/replay,
  - invariantes S4/S5,
  - render determinístico de `flow/explain_flow/trace_flow`,
  - negative tamper (fail‑closed).

> Nota WORM: existem runs anteriores `fix1..fix8` em `results/` (tentativas/falhas). O fechamento está em `fix9` (sem overwrite).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Regressões smoke V90–V107 (out_base `results/run_smoke_*_regression_v108_fix9`)
- PASS Smoke V108 determinístico (try1==try2; inclui `flow_chain_hash_v108`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v108.py --run_dir results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1`
- PASS Negative tamper: `flow_events[0].event_sig="0"*64` ⇒ falha com reason `flow_event_sig_mismatch`

## Arquivos novos (V108)
- `atos_core/conversation_loop_v108.py:1` — wrapper V107; aplica DSC/flow score; persiste `flow_events.jsonl` + snapshot + hashes; aplica gating S4/S5.
- `atos_core/conversation_v108.py:1` — `verify_conversation_chain_v108` (wrapper V107) + checks `flow` + invariantes S4/S5.
- `atos_core/flow_engine_v108.py:1` — planner/critics/score/seleção determinística + decisão de repair.
- `atos_core/flow_ledger_v108.py:1` — schema do ledger, renderers e fold/replay de snapshot + `flow_chain_hash_v108`.
- `atos_core/intent_grammar_v108.py:1` — raw intercept `flow/explain_flow/trace_flow` (fail‑closed).
- `scripts/smoke_v108_flow_discourse_fluency_memory_survival_dialogue_csv.py:1` — smoke V108 (try1/try2 + tamper).
- `scripts/verify_conversation_chain_v108.py:1` — verificador CLI V108.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V107 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py --out_base results/run_smoke_v103_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py --out_base results/run_smoke_v104_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py --out_base results/run_smoke_v105_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv.py --out_base results/run_smoke_v106_regression_v108_fix9 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v107_pragmatics_intent_dialogue_state_survival_dialogue_csv.py --out_base results/run_smoke_v107_regression_v108_fix9 --seed 0

# Smoke V108 (try1/try2 interno; WORM) + tee
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v108_flow_discourse_fluency_memory_survival_dialogue_csv.py \
  --out_base results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9 --seed 0 \
  | tee results/smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1.json

# Verificador V108 (PASS)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v108.py --run_dir \
  results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1

# Verificador V108 — negative tamper (espera FAIL com reason flow_event_sig_mismatch)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v108.py --run_dir \
  results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1_tamper

# Patch diff (WORM) — somente arquivos V108 (ordem lexicográfica)
(diff -u /dev/null atos_core/conversation_loop_v108.py; \
 diff -u /dev/null atos_core/conversation_v108.py; \
 diff -u /dev/null atos_core/flow_engine_v108.py; \
 diff -u /dev/null atos_core/flow_ledger_v108.py; \
 diff -u /dev/null atos_core/intent_grammar_v108.py; \
 diff -u /dev/null scripts/smoke_v108_flow_discourse_fluency_memory_survival_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v108.py) \
 > patches/v108_flow_discourse_fluency_memory_survival_fix9.diff
```

## Determinismo (try1 == try2) — Smoke V108 (fix9)
- `seed`: `0`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `508335a7ac3ba1c3d72fd75b0dfe28dca801475dbeabbaa9baabb46571bca7b5`
- `state_chain_hash`: `a1815ef738197b806e904f9d5e9e107af679e9dc3ff7e857b44d7ab1723daf7b`
- `parse_chain_hash`: `337ec71f605db0c76f5e25ba8f259d1cdce7dd1249ca34028aaa20ba7094d3e9`
- `plan_chain_hash` (action_plans): `3585a0df6cd7a221ff2dde52e17c908942003e85d8beae66f53e44d04a8bec01`
- `plan_events_chain_hash_v104`: `3deaf5d1bdd889bcd3a2e950693d291632ac9850a96179dd266c37d6d07cd07e`
- `agency_chain_hash_v105`: `65172fd6b7039023d878cb32ca42559125e1721493f6a76d5936991e9749ea06`
- `style_chain_hash`: `d3de7848866d899bb91f35ef686fad5b424e48756b4426a0ef200bcfcc261c0a`
- `dialogue_chain_hash_v106`: `8263ec1bc727be66be32688257c93e41c31448fa254bbebd4d8651800006d1a1`
- `pragmatics_chain_hash_v107`: `9f8b86733f93a9690199dd8088e4c531ae344f894d63cce5d60b98c4de07a7ed`
- `flow_chain_hash_v108`: `009506b8026afdf1132ed29e22ce9610e1fb556bd632f8e72fd90047cc5027a4`
- `ledger_hash` (sha256 do `freeze_manifest_v108.json`): `bb46ed0d7f8ecafb38d28f61abcfafe322d7be2e38026aba110243e70ab520cc`
- Smoke stdout: `summary_sha256=534b94052d65fd56ebbbc4434a87c8a32d69828f471ec0f9c8ba20a2347936b8` e `eval_sha256=8876f6deed11f38e118decb8277cb1cfc590fdf376a587f2fc7afed2b244af90`

## Negative tamper
- PASS: `flow_events[0].event_sig="0"*64` ⇒ `verify_conversation_chain_v108` falha com reason `flow_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V108 try1: `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/`
- Smoke V108 try2: `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try2/`
- Tamper try1: `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1_tamper/`
- Stdout tee: `results/smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1.json`
- Patch diff: `patches/v108_flow_discourse_fluency_memory_survival_fix9.diff`

## SHA256 principais (artefatos V108)
- `patches/v108_flow_discourse_fluency_memory_survival_fix9.diff`: `2f7304546d469e6c438e555107c01c715aa16cbd4492dcdc33aa2ef1a4b434ae`
- `results/smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1.json`: `60f7ee81bf06fe8eccb3a4f68e8a95df10a59fbd15aaca1e05906d5a8f56f66c`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/summary.json`: `7741facd32060fa2e4550415c132c1019a26ec4e36aade35b0d6274e25a2c85b`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/flow_events.jsonl`: `114ea50d03a95c3d36211d9d29ba4b25ac754832d50490899230d94770a52701`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/flow_registry_snapshot_v108.json`: `6d006b5b12e793685bc1d5d203b2977d14ae0b7b1abc4170cfc57315e058dce8`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/eval.json`: `8876f6deed11f38e118decb8277cb1cfc590fdf376a587f2fc7afed2b244af90`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/smoke_summary.json`: `350d42dd6f63ec4925f99301f97101e3404ca8008d501b23ff722d4c5a373f2c`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/freeze_manifest_v108.json`: `bb46ed0d7f8ecafb38d28f61abcfafe322d7be2e38026aba110243e70ab520cc`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/verify_chain_v108.json`: `a68c3c7ec4761ed2c1b291282a9786403a195681d0d483ac811000d56b3a6f92`
- `results/run_smoke_v108_flow_discourse_fluency_memory_survival_fix9_try1/transcript.jsonl`: `866cf9c1fc5fb060c770e185415e3070e6d01b236effb4267bce44c41b8463fc`

## GO/NO-GO
- **GO** — regressões V90–V107 + testes passam; smoke V108 determinístico (try1==try2); invariantes S4/S5 exercitadas (block + repair); verificador V108 passa e negative tamper falha com reason correto.

