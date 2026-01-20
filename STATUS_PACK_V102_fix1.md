# STATUS PACK V102 — V102_STYLE_PROFILE_TEMPLATES_SOFT_SELECTION_CRITICS (fix1)

## Resumo
- Adiciona **Style Profile V102** (memória linguística estruturada) com atualização determinística por sinais explícitos do usuário e renderer `style_profile`.
- Implementa **Discourse Templates V102** + **críticos de estilo/fluência** + **seleção determinística** (TOP‑K candidatos, scoring e escolha auditável).
- Adiciona **Style Ledger V102** (`style_events.jsonl`) hash‑chained, snapshot derivado (`template_library_snapshot_v102.json`) e verificador V102.
- Adiciona introspecção V102: `style_profile`, `templates`, `explain_style <turn_id>`, `trace_style <turn_id>`.

## Checklist (PASS/FAIL)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS Regressões smoke V90–V101 (out_base `results/run_smoke_*_regression_v102_fix1`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS Smoke V102 determinístico (try1==try2; inclui `style_chain_hash`)
- PASS `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v102.py --run_dir results/run_smoke_v102_style_profile_templates_soft_selection_fix3_try1`
- PASS Negative tamper: `style_events[0].event_sig="0"*64` ⇒ falha com reason `style_event_sig_mismatch`

## Arquivos novos (V102)
- `atos_core/style_profile_v102.py:1` — modelo + update rules + renderer.
- `atos_core/discourse_templates_v102.py:1` — template library determinística + constraints.
- `atos_core/style_critics_v102.py:1` — críticos determinísticos + score.
- `atos_core/style_selector_v102.py:1` — build TOP‑K + seleção + explain.
- `atos_core/style_ledger_v102.py:1` — eventos style + fold stats + snapshot + chain hash.
- `atos_core/intent_grammar_v102.py:1` — intents/raw intercepts V102 (style_profile/templates/explain_style/trace_style).
- `atos_core/conversation_v102.py:1` — `verify_conversation_chain_v102` (wrapper de V101) + invariantes V102.
- `atos_core/conversation_loop_v102.py:1` — loop V102 + `style_events.jsonl` + snapshot + hashes no manifest/summary.
- `scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py:1` — smoke V102 (try1/try2 + tamper).
- `scripts/verify_conversation_chain_v102.py:1` — verificador CLI V102.

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Tests
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V102 (try1/try2 interno; WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py \
  --out_base results/run_smoke_v102_style_profile_templates_soft_selection_fix3 --seed 0 \
  | tee results/smoke_v102_style_profile_templates_soft_selection_fix3_try1.json

# Verificador V102 (recomendado)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v102.py --run_dir \
  results/run_smoke_v102_style_profile_templates_soft_selection_fix3_try1

# Patch diff (WORM) — somente arquivos V102
(diff -u /dev/null atos_core/conversation_loop_v102.py; \
 diff -u /dev/null atos_core/conversation_v102.py; \
 diff -u /dev/null atos_core/discourse_templates_v102.py; \
 diff -u /dev/null atos_core/intent_grammar_v102.py; \
 diff -u /dev/null atos_core/style_critics_v102.py; \
 diff -u /dev/null atos_core/style_ledger_v102.py; \
 diff -u /dev/null atos_core/style_profile_v102.py; \
 diff -u /dev/null atos_core/style_selector_v102.py; \
 diff -u /dev/null scripts/smoke_v102_style_profile_templates_soft_selection_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v102.py) \
 > patches/v102_style_profile_templates_soft_selection_critics_fix1.diff
```

## Determinismo (try1 == try2)
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `2fc17cf3d22fbeffc65edbf1ee765c3918095182cd4cf0bb895fd5b7a00e9454`
- `state_chain_hash`: `ad234a552cfa5698509de1183c60ea6cc7b71d61f5cf70c0b92ee025b7958cd3`
- `parse_chain_hash`: `adb0d49d8c1ec5df150d6bb2a9d4821f3dc86cfb9cd62e2715ecc6ecb8d39f11`
- `learned_chain_hash`: `b3ed9ea1aae301f9ceeb8d650060d9c665df29b7ec38adfb904c29daa36b2036`
- `plan_chain_hash`: `856e123c769b2e1d857fd33f8f56f711b37d6b2eb8e28ff19f40bdb8ad87d925`
- `memory_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `belief_chain_hash`: `e46079ce99f8d3ca34986818edce9ceaa4d2cbcac8d3613382c2dc0493f61640`
- `evidence_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `goal_chain_hash`: `34c359ed49a30c889c553d89a7dc4f65520fede9aa79252e8ab0ef5da7700cb8`
- `discourse_chain_hash`: `e37270d8d60c1f29663e8f806ffddcd4b784479227677c8e334ba1e93b637ee0`
- `fragment_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `binding_chain_hash`: `336ce49df8a7de7830e616bc02024f06b1f3c9bc74eb574df70e6866b1d2c76a`
- `style_chain_hash`: `333cc8b1cd2b683a304a82f4f73587f3e61d49c91ff3fcbfa1524a74a5a5d075`
- `ledger_hash` (sha256 do `freeze_manifest_v102.json`): `bf7cc47ab55041e82cc8b264110a3e4367a6f72916cc645377d822cbc348e5d7`
- `summary_sha256`: `eb9977bd0f418fd28429f650a5ff6ead27e4a4555855ea0f8f82bc1688c1e488`

## Negative tamper
- PASS: `style_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v102` falha com reason `style_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V102 try1: `results/run_smoke_v102_style_profile_templates_soft_selection_fix3_try1/`
- Smoke V102 try2: `results/run_smoke_v102_style_profile_templates_soft_selection_fix3_try2/`
- Stdout tee: `results/smoke_v102_style_profile_templates_soft_selection_fix3_try1.json`
- Patch diff: `patches/v102_style_profile_templates_soft_selection_critics_fix1.diff`
