# STATUS PACK V104 — V104_PLANNING_OPTIONS_REASON_LEDGER (fix1)

## Resumo
- Adiciona **Plan Engine V104**: geração determinística de opções **A/B/C** por turno (`plan_kind`, efeitos discretos, score inteiro, tie-break estável).
- Adiciona **Plan Ledger V104** (`plan_events.jsonl`) WORM (hash-chain externo + cadeia interna `prev_event_sig/event_sig`) e snapshot derivado (`plan_registry_snapshot_v104.json`).
- Integra o **Plan Engine V104** ao loop de conversa V104 (wrapper do V103): usa **concept hits V103** como features explícitas e registra a decisão por turno em `plan_events.jsonl`.
- Adiciona introspecção V104 (raw intercept): `plans`, `explain_plan <turn_index>`, `trace_plans <turn_index>`.
- Adiciona verificador V104 (wrapper do V103) com checks de:
  - cadeia interna de `plan_events` + snapshot,
  - render determinístico de `plans/explain_plan/trace_plans` a partir do prefixo do ledger,
  - negative tamper (fail-closed).

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Regressões smoke V90–V103 (out_base `results/run_smoke_*_regression_v104_fix1`)
- PASS Smoke V104 determinístico (try1==try2; inclui `plan_events_chain_hash_v104`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v104.py --run_dir results/run_smoke_v104_planning_options_reason_ledger_fix1_run2_try1`
- PASS Negative tamper: `plan_events[0].event_sig="0"*64` ⇒ falha com reason `plan_event_sig_mismatch`

## Arquivos novos/modificados (V104)
- `atos_core/plan_ledger_v104.py:1` — `PlanEventV104` + `event_sig/event_id` + `plan_chain_hash_v104`.
- `atos_core/plan_engine_v104.py:1` — geração de candidatos A/B/C + score determinístico e `active_plan_state_v104`.
- `atos_core/plan_registry_v104.py:1` — fold/replay + snapshot + renderers (`plans/explain_plan/trace_plans`).
- `atos_core/intent_grammar_v104.py:1` — raw intercept `plans/explain_plan/trace_plans` (fail-closed).
- `atos_core/conversation_v104.py:1` — `verify_conversation_chain_v104` (wrapper V103) + invariantes do plan ledger.
- `atos_core/conversation_loop_v104.py:1` — loop V104 + `plan_events.jsonl` + snapshot + hashes no manifest/summary.
- `scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py:1` — smoke V104 (try1/try2 + tamper).
- `scripts/verify_conversation_chain_v104.py:1` — verificador CLI V104.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V103 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v104_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py --out_base results/run_smoke_v103_regression_v104_fix1 --seed 0

# Smoke V104 (try1/try2 interno; WORM) + tee
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py \
  --out_base results/run_smoke_v104_planning_options_reason_ledger_fix1_run2 --seed 0 \
  | tee results/smoke_v104_planning_options_reason_ledger_fix1_run2_try1.json

# Verificador V104
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v104.py --run_dir \
  results/run_smoke_v104_planning_options_reason_ledger_fix1_run2_try1

# Patch diff (WORM) — somente arquivos V104
(diff -u /dev/null atos_core/conversation_loop_v104.py; \
 diff -u /dev/null atos_core/conversation_v104.py; \
 diff -u /dev/null atos_core/intent_grammar_v104.py; \
 diff -u /dev/null atos_core/plan_engine_v104.py; \
 diff -u /dev/null atos_core/plan_ledger_v104.py; \
 diff -u /dev/null atos_core/plan_registry_v104.py; \
 diff -u /dev/null scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v104.py) \
 > patches/v104_planning_options_reason_ledger_fix1.diff
```

## Determinismo (try1 == try2) — Smoke V104
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `5215de10cbd3791419ea624fe8c3ac7abca7894b9ce80ec3871448fdcc2b4ea0`
- `state_chain_hash`: `621f553649122af031c656a159aeb087f0354c35120aacf6dcf066e44edaaa34`
- `parse_chain_hash`: `c095f913c233a02390b4f3b75b0c27cd05bdd96fc3d9b82e60bcc94d0dcd2d39`
- `plan_chain_hash` (action_plans): `a1ca0252caaf5ce81a968de9227cf76a4969766d67aeef34bc695c781d528d1f`
- `plan_events_chain_hash_v104`: `224fbb612789e591f5f4470d93665f9f477b12f7eb3aa9f90c82c862527989b1`
- `binding_chain_hash`: `1d9dcf1fad9e83149bf34afbf3018963f1ab2cf1764302b2fc18967b8f94c27b`
- `style_chain_hash`: `ef05f6fd4a25a0fa724c07bdd538e4317848d13d2f05457aa46f6fd94c1fa56e`
- `concept_chain_hash`: `e7256d8f77fbf199e02b39c7b3930fbd2b577e1f8c307484f878904e973b9cbb`
- `ledger_hash` (sha256 do `freeze_manifest_v104.json`): `ba330dc40a4bd54e00f011ea88c056a1b7e17a4f9d97299e69c8df4aaaebc7b1`
- `summary_sha256`: `02d154b49a90e8455cb2bd81ce91ba90d38e0063c4cf5d6c50cc6292520a11ab`

## Negative tamper
- PASS: `plan_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v104` falha com reason `plan_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V104 try1: `results/run_smoke_v104_planning_options_reason_ledger_fix1_run2_try1/`
- Smoke V104 try2: `results/run_smoke_v104_planning_options_reason_ledger_fix1_run2_try2/`
- Stdout tee: `results/smoke_v104_planning_options_reason_ledger_fix1_run2_try1.json`
- Patch diff: `patches/v104_planning_options_reason_ledger_fix1.diff`

## GO/NO-GO
- **GO** — smoke determinístico, negative tamper ok, regressões ok, e verificador V104 passa.

