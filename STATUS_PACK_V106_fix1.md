# STATUS PACK V106 — V106_DIALOGUE_COHERENCE_SURVIVAL_LEDGER (fix1)

## Resumo
- Adiciona **Dialogue Ledger V106** (`dialogue_events.jsonl`) WORM (hash-chain externo + cadeia interna `prev_event_sig/event_sig`) e snapshot derivado (`dialogue_registry_snapshot_v106.json`).
- Implementa **DialogueCoherenceMetric V106** (componentes 0..100 + flags determinísticos) e integra na seleção de resposta por turno com **regra de sobrevivência**: não permite 2 turnos consecutivos com `coherence_score < 60` sem `repair_action`.
- Integra V106 com **V102 templates/critics** (geração TOP‑K candidatos) e registra `candidates[]` (hash + métricas + flags) no ledger.
- Adiciona introspecção V106 (raw intercept, fail‑closed): `dialogue`, `explain_dialogue <turn_index>`, `trace_dialogue <turn_index>`.
- Adiciona verificador V106 (wrapper do V105) com checks de:
  - cadeia interna de `dialogue_events`,
  - snapshot reproduzível por fold/replay,
  - render determinístico de `dialogue/explain_dialogue/trace_dialogue`,
  - invariant de sobrevivência (repair obrigatório quando degradar),
  - negative tamper (fail‑closed).

> Nota WORM: um smoke anterior consumiu `fix1` (stdout tee ficou vazio). O smoke fechado roda em `fix2` (runs) e mantém determinismo try1==try2.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Regressões smoke V90–V105 (out_base `results/run_smoke_*_regression_v106_fix1`)
- PASS Smoke V106 determinístico (try1==try2; inclui `dialogue_chain_hash_v106`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v106.py --run_dir results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1`
- PASS Negative tamper: `dialogue_events[0].event_sig="0"*64` ⇒ falha com reason `dialogue_event_sig_mismatch`

## Arquivos novos (V106)
- `atos_core/dialogue_ledger_v106.py:1` — schema/assinatura `DialogueEventV106`, renderers `dialogue/explain_dialogue/trace_dialogue`, snapshot e `dialogue_chain_hash_v106`.
- `atos_core/dialogue_engine_v106.py:1` — métricas/flags determinísticas + seleção de candidato + regra de sobrevivência (repair).
- `atos_core/intent_grammar_v106.py:1` — raw intercept `dialogue/explain_dialogue/trace_dialogue` (fail‑closed).
- `atos_core/conversation_v106.py:1` — `verify_conversation_chain_v106` (wrapper V105) + invariantes do dialogue ledger.
- `atos_core/conversation_loop_v106.py:1` — loop V106 + persistência `dialogue_events.jsonl` + snapshot + hashes no manifest/summary.
- `scripts/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv.py:1` — smoke V106 (try1/try2 + tamper).
- `scripts/verify_conversation_chain_v106.py:1` — verificador CLI V106.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Regressões smoke V90–V105 (WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py --out_base results/run_smoke_v100_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v101_bindings_reference_resolution_dialogue_csv.py --out_base results/run_smoke_v101_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py --out_base results/run_smoke_v102_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v103_concept_subgraphs_toc_mdl_dialogue_csv.py --out_base results/run_smoke_v103_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v104_planning_options_reason_ledger_dialogue_csv.py --out_base results/run_smoke_v104_regression_v106_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v105_continuous_agency_goal_plan_loop_dialogue_csv.py --out_base results/run_smoke_v105_regression_v106_fix1 --seed 0

# Smoke V106 (try1/try2 interno; WORM) + tee
set -o pipefail
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv.py \
  --out_base results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2 --seed 0 \
  | tee results/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1.json

# Verificador V106
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v106.py --run_dir \
  results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1

# Patch diff (WORM) — somente arquivos V106
(diff -u /dev/null atos_core/conversation_loop_v106.py; \
 diff -u /dev/null atos_core/conversation_v106.py; \
 diff -u /dev/null atos_core/dialogue_engine_v106.py; \
 diff -u /dev/null atos_core/dialogue_ledger_v106.py; \
 diff -u /dev/null atos_core/intent_grammar_v106.py; \
 diff -u /dev/null scripts/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v106.py) \
 > patches/v106_dialogue_coherence_survival_ledger_fix1.diff
```

## Determinismo (try1 == try2) — Smoke V106 (fix2)
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `c0cbfcfb8ca757be057fdb542f9d2c8fc2f1a28042f4d870b887486d91819de8`
- `state_chain_hash`: `e27bf5045c12a4b67d771f396c7c8591fc73e9c9f15b56e6a150b3b42407d661`
- `parse_chain_hash`: `30638fba4ccbda30e391642dda6a961853caf30429491a6ba204bc93a9d5de02`
- `plan_chain_hash` (action_plans): `b1b5d8ae9cb2d137a69ac29e60b35ffd9182b76533daccc75ecbe5c6f711313b`
- `plan_events_chain_hash_v104`: `2e5c15a017ec2b6846e434ce912944d1d3d4968b0f557e0073973c538a06c257`
- `agency_chain_hash_v105`: `5c050a22c9918e304ee8b93971e000f9d07e6039563b695aaa04de04310868e9`
- `style_chain_hash`: `60c1126c3420751bdb738f2a990e37101b8c5da0f1ea96b016e39e4c39cd39ae`
- `dialogue_chain_hash_v106`: `e38b09f1ca20b428376d4658a1caa6919ebfc7fc39e8c5bc4d7e157098892bd0`
- `repairs_total`: `6`
- `ledger_hash` (sha256 do `freeze_manifest_v106.json`): `959c7220480a565e7b5474e30338c09559c63f4caa3c9c44550a85dc4b085b16`
- `summary_sha256`: `5fd7d63648d0e2e2d31c0746d58ac871eba152de92e558d2a220de44e189c775`

## Negative tamper
- PASS: `dialogue_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v106` falha com reason `dialogue_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V106 try1: `results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1/`
- Smoke V106 try2: `results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try2/`
- Stdout tee: `results/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1.json`
- Patch diff: `patches/v106_dialogue_coherence_survival_ledger_fix1.diff`

## SHA256 principais
- `patches/v106_dialogue_coherence_survival_ledger_fix1.diff`: `e07ab3b14b940aca36ac16dada0a2902fa93d9cf18e66f6b7275b6ee3453583b`
- `results/smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1.json`: `faa8c8c67bade3de7ee48db3b2457bfcbc913fe9e5da0ae28b6b3de2ca5beace`
- `results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1/summary.json`: `82b78a2b9857ed66c99069076bd426884f3ab7346dfc0f0082fe5a4294721bbc`
- `results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1/dialogue_events.jsonl`: `e6ebc739113bdafcabcb578bdeba2d6f3e0f5e1a644cc38793111858014b0207`
- `results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1/dialogue_registry_snapshot_v106.json`: `7d7151a0bfe31453c808aea74c00042844350427a4bd584ae77e31e8410396b0`
- `results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1/freeze_manifest_v106.json`: `959c7220480a565e7b5474e30338c09559c63f4caa3c9c44550a85dc4b085b16`
- `results/run_smoke_v106_dialogue_coherence_survival_ledger_dialogue_csv_fix2_try1/verify_chain_v106.json`: `93f11d0398d44417d9e2efbfc641f2c2070511fa15ee8270a63937bb593171d4`

## GO/NO-GO
- **GO** — smoke determinístico (try1==try2), regra de sobrevivência aplicada, negative tamper ok, regressões ok e verificador V106 passa.

