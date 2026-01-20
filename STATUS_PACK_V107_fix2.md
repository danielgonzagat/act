# STATUS PACK V107 — V107_PRAGMATICS_INTENT_DIALOGUE_STATE_SURVIVAL (fix2)

## Resumo
- Adiciona **Pragmatics Ledger V107** (`pragmatics_events.jsonl`) WORM (hash-chain externo + cadeia interna `prev_event_sig/event_sig`) e snapshot derivado (`pragmatics_registry_snapshot_v107.json`).
- Implementa **Pragmatics Engine V107** determinístico: IntentActs (inclui `PROMISE_ATTEMPT` e `UNKNOWN`), controlador de regime (MED) e memória estruturada (MDE: `pending_questions`, `commitments`, `topic_stack`), com **Soft Diversity Forcing (SDF)** (anti-loop de repetição).
- Introduz **PragmaticsSurvivalMetric V107** (0..100 + flags) e aplica **invariantes de sobrevivência**:
  - **S1/S2**: não permite 2 turnos consecutivos degradados (`coherence_v106 < 60` ou `pragmatics_v107 < 60`) sem `repair_action`.
  - **S3 (progress gating)**: se pragmática degradar (flags críticas), bloqueia progresso (goal/plan/agency) e força repair (clarify/refuse/repair) como única saída.
- Integra V107 na seleção de resposta: score do candidato = `min(coherence_score_v106, pragmatics_score_v107)` com tie-break determinístico; registra candidatos (hash+scores+flags+repetition) no ledger.
- Adiciona introspecção V107 (raw intercept, fail‑closed): `pragmatics`, `explain_pragmatics <turn_index>`, `trace_pragmatics <turn_index>`.
- Adiciona verificador V107 (wrapper do V106) com checks de:
  - cadeia interna de `pragmatics_events`,
  - snapshot reproduzível por fold/replay,
  - invariantes S1/S2/S3,
  - render determinístico de `pragmatics/explain_pragmatics/trace_pragmatics`,
  - negative tamper (fail‑closed).

> Nota WORM: `fix1` já existia em `results/` (runs e stdout). A entrega fechada usa `fix2` (sem overwrite).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Regressões smoke V90–V106 (out_base `results/run_smoke_*_regression_v107_fix2`)
- PASS Smoke V107 determinístico (try1==try2; inclui `pragmatics_chain_hash_v107`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v107.py --run_dir results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1`
- PASS Negative tamper: `pragmatics_events[0].event_sig="0"*64` ⇒ falha com reason `pragmatics_event_sig_mismatch`

## Arquivos novos (V107)
- `atos_core/conversation_loop_v107.py:1` — wrapper V106 + persistência `pragmatics_events.jsonl` + snapshot + hashes no manifest/summary + integração na seleção e no gating S3.
- `atos_core/conversation_v107.py:1` — `verify_conversation_chain_v107` (wrapper V106) + checks pragmáticos + invariantes S1/S2/S3.
- `atos_core/intent_grammar_v107.py:1` — raw intercept `pragmatics/explain_pragmatics/trace_pragmatics` (fail‑closed).
- `atos_core/pragmatics_engine_v107.py:1` — IntentActs + MED + MDE + SDF + score/flags determinísticos.
- `atos_core/pragmatics_ledger_v107.py:1` — schema/assinatura `PragmaticsEventV107`, renderers, snapshot e `pragmatics_chain_hash_v107`.
- `scripts/smoke_v107_pragmatics_intent_dialogue_state_survival_dialogue_csv.py:1` — smoke V107 (try1/try2 + tamper).
- `scripts/verify_conversation_chain_v107.py:1` — verificador CLI V107.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V106 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py --out_base results/run_smoke_v103_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py --out_base results/run_smoke_v104_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py --out_base results/run_smoke_v105_regression_v107_fix2 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv.py --out_base results/run_smoke_v106_regression_v107_fix2 --seed 0

# Smoke V107 (try1/try2 interno; WORM) + tee
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v107_pragmatics_intent_dialogue_state_survival_dialogue_csv.py \
  --out_base results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2 --seed 0 \
  | tee results/smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1.json

# Verificador V107
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v107.py --run_dir \
  results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1

# Patch diff (WORM) — somente arquivos V107
(diff -u /dev/null atos_core/conversation_loop_v107.py; \
 diff -u /dev/null atos_core/conversation_v107.py; \
 diff -u /dev/null atos_core/intent_grammar_v107.py; \
 diff -u /dev/null atos_core/pragmatics_engine_v107.py; \
 diff -u /dev/null atos_core/pragmatics_ledger_v107.py; \
 diff -u /dev/null scripts/smoke_v107_pragmatics_intent_dialogue_state_survival_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v107.py) \
 > patches/v107_pragmatics_intent_dialogue_state_survival_fix2.diff
```

## Determinismo (try1 == try2) — Smoke V107 (fix2)
- `seed`: `0`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `769979d8c4046fe7eb2c615bf6b0216723df1cf96144a33533286f6f0834a3de`
- `state_chain_hash`: `89b20eb20b7284bcf481a968a2f8d2c874ef1cc16bd99af7948dbc0f8250356b`
- `parse_chain_hash`: `ac572a1bef7eb2a6e0cf90f85dc88e369bafc759120d08e431e9d435f3458baa`
- `plan_chain_hash` (action_plans): `0802e1b5cee9de3db9949b26fa4fd45e7324c388c27884f09e7405bd500d066a`
- `plan_events_chain_hash_v104`: `41ceb902dc7363fe6e78d25d9d208f9bba5d219c26d8fe22a5cb2428893dc3e9`
- `agency_chain_hash_v105`: `9479b84a1c22cf17ec0b462b66893eac03da8f3a60e61bc045d96ecac77d570b`
- `style_chain_hash`: `bde6d5c7a934512a962ad25b146f0f7fa04dbfee9a7de6d4d33c0202c5aa9e09`
- `dialogue_chain_hash_v106`: `47be357fd043e7379dc8345b5d36f7ac066288b5c6ab6c471da11f272486878c`
- `pragmatics_chain_hash_v107`: `18dccf3fe80dca66b07bb072be32940d25d27d606204ebac0c7e311224b72811`
- `ledger_hash` (sha256 do `freeze_manifest_v107.json`): `96b98f055352aebd555fa1878bd96c57d6a05e940eac64fd33f8a566b3aeff9f`
- `summary_sha256`: `eeabe552252f57a8402e4b31be3fde99e72f49b8bee133e0fdbde09b90163896`

## Negative tamper
- PASS: `pragmatics_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v107` falha com reason `pragmatics_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V107 try1: `results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1/`
- Smoke V107 try2: `results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try2/`
- Stdout tee: `results/smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1.json`
- Patch diff: `patches/v107_pragmatics_intent_dialogue_state_survival_fix2.diff`

## SHA256 principais
- `patches/v107_pragmatics_intent_dialogue_state_survival_fix2.diff`: `9756f81aff67669d4f617909b7a0fb1bcf083a44636174f8d799e4991bd523d3`
- `results/smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1.json`: `1f0c35f80b18fd5e778f0b44df7a0a59431fea2c08b8af17749b9486190081c4`
- `results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1/summary.json`: `e12aade50b5d572d407445b637b57811635a35473bcca4dba36d8b4ff9ef1857`
- `results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1/pragmatics_events.jsonl`: `c72b54eb274d18ceb5857206e8a9eae2196ab367d4d7a8b1c297ee2682e310a7`
- `results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1/pragmatics_registry_snapshot_v107.json`: `671f69ee3454d96a5642a75d7989a0aa608107a46d7f5999bace964ea447ec9a`
- `results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1/freeze_manifest_v107.json`: `96b98f055352aebd555fa1878bd96c57d6a05e940eac64fd33f8a566b3aeff9f`
- `results/run_smoke_v107_pragmatics_intent_dialogue_state_survival_fix2_try1/verify_chain_v107.json`: `71e8e57d003f984e5c0a72b0346fb915ccfd000a00ff552c2d0194ac2e2664bd`

## GO/NO-GO
- **GO** — smoke determinístico (try1==try2), invariantes S1/S2/S3 aplicadas (progress gating + repair), negative tamper ok, regressões ok e verificador V107 passa.

