# STATUS PACK V96 — V96_BELIEF_LEDGER_DIALOGUE_CSV — GO (fix1)

## Resumo
- Adiciona **Belief Ledger V96** (crenças `key=value`) como log WORM hash-chained: `belief_events.jsonl` com eventos `ADD`/`RETRACT` e itens `belief_item_v96` com `belief_id`/`belief_sig` determinísticos.
- Adiciona comandos raw determinísticos (accent-fold, fail-closed) em V96:
  - `belief:` / `crenca:` (ADD; não sobrescreve)
  - `revise:` / `revisar:` (revisão = `RETRACT(old)` + `ADD(new)` no mesmo step)
  - `beliefs` / `crencas` (listagem determinística)
  - `forget belief <key>` / `esquece crenca <key>` (retract por key)
  - Mantém NOTE/RECALL/FORGET last (memória) como no V95.
- Integra crenças ao diálogo:
  - `GET <k>` lê crença ativa se `k` não existe em `vars_map` (resposta determinística `k=<belief_value>`).
  - `EXPLAIN` usa `render_explain_text_v96` (extensão do v94) e inclui campos de memória/crenças lidos/escritos no ActionPlan.
- Verificador V96 (`verify_conversation_chain_v96`) cobre invariantes V95 + replay do belief ledger + cross-check do texto de `beliefs` via renderer determinístico + negative tamper test.

## Checklist
- PASS: `py_compile` (atos_core + scripts)
- PASS: regressões V90–V95 (smokes) ok
- PASS: `python3 -m unittest -v` ok
- PASS: smoke V96 try1==try2 (hashes idênticos, incluindo `belief_chain_hash`)
- PASS: `belief:` adiciona crença e `beliefs` renderiza determinístico
- PASS: `revise:` troca valor com trilha (`RETRACT`+`ADD`) e `beliefs` não mostra valor antigo
- PASS: `forget belief <key>` remove crença ativa (fail-closed em `missing_key`)
- PASS: integração `GET` lê crença ativa quando var ausente
- PASS: negative test V96 (corrupção `belief_event_sig`) → fail-closed com reason determinístico `belief_event_sig_mismatch`

## Arquivos novos (V96)
- `atos_core/intent_grammar_v96.py:1`
- `atos_core/conversation_v96.py:1`
- `atos_core/conversation_loop_v96.py:1`
- `scripts/verify_conversation_chain_v96.py:1`
- `scripts/smoke_v96_belief_ledger_dialogue_csv.py:1`
- `patches/v96_belief_ledger_dialogue_csv_fix1.diff:1`
- `STATUS_PACK_V96_fix1.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (V90–V95)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v96_fix1 --seed 0 | tee results/smoke_v90_regression_v96_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v96_fix1 --seed 0 | tee results/smoke_v91_regression_v96_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v96_fix1 --seed 0 | tee results/smoke_v92_regression_v96_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v96_fix1 --seed 0 | tee results/smoke_v93_regression_v96_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v96_fix1 --seed 0 | tee results/smoke_v94_regression_v96_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v96_fix1 --seed 0 | tee results/smoke_v95_regression_v96_fix1_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V96 (try1/try2 dentro do script)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py \
  --out_base results/run_smoke_v96_belief_ledger_dialogue_csv_fix2 --seed 0 \
  | tee results/smoke_v96_belief_ledger_dialogue_csv_fix2_try1.json

# Patch diff (WORM) — somente arquivos V96
(diff -u /dev/null atos_core/conversation_loop_v96.py; \
 diff -u /dev/null atos_core/conversation_v96.py; \
 diff -u /dev/null atos_core/intent_grammar_v96.py; \
 diff -u /dev/null scripts/smoke_v96_belief_ledger_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v96.py) \
 > patches/v96_belief_ledger_dialogue_csv_fix1.diff
```

## Determinismo (try1 == try2)
Do smoke V96: `results/smoke_v96_belief_ledger_dialogue_csv_fix2_try1.json`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `00228c02cd0e81c398dc5f912b64026daf1a6a2f6defb030287cd33464ef3aed`
- `state_chain_hash`: `f53fd41fa8f810909242896013049e754e5a5232944611b5d2d1b23c311c1e98`
- `parse_chain_hash`: `b867052c9a2251b2f5f3458bb8eeeaf5794e12ef5ae096b92f1bd3959e9a9551`
- `learned_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `plan_chain_hash`: `720c12f75207ee595a6e4f6d634d924c639d366178b726b29aa25e60565075de`
- `memory_chain_hash`: `58dbed0bd0db8f2e2fcb6d4f389ad0516c473c99244bfb13d09ce5efe268d37d`
- `belief_chain_hash`: `e2ccd2fe52eff6e88c7f6d7c93c4a287f5dbbec8edea597d676c0b1f0e3b1c15`
- `ledger_hash` (sha256 do `freeze_manifest_v96.json`): `eecaae55442910a63909265fc33cfd65c44023b2875a486b1f17fcaabacbbd2b`
- `summary_sha256`: `42ad42a6e698ab90775826346e730083589b71a27b34706736df1f7aa27c149c`

## Negative test V96 (fail-closed)
- Corrupção: `belief_events[0].event_sig="0"*64` → reason=`belief_event_sig_mismatch`

## Artefatos principais (sha256)
- `patches/v96_belief_ledger_dialogue_csv_fix1.diff`: `e9e4589f64672d30e0f99786d8c3e6c928d90e3464949db70e1fc23057b102ae`
- `results/smoke_v96_belief_ledger_dialogue_csv_fix2_try1.json`: `ef2d0c4c3362cfb85d7487aff81b3225394693e5ab46d4f8b11e9d2d30290973`
- `results/run_smoke_v96_belief_ledger_dialogue_csv_fix2_try1/freeze_manifest_v96.json`: `eecaae55442910a63909265fc33cfd65c44023b2875a486b1f17fcaabacbbd2b`
- `results/run_smoke_v96_belief_ledger_dialogue_csv_fix2_try1/verify_chain_v96.json`: `676dc4ae8ad7cabee63b63cf60dcc8f69259593e36826508e65154b36574d83e`
- `results/run_smoke_v96_belief_ledger_dialogue_csv_fix2_try1/belief_events.jsonl`: `e3639a72a8b5bf9544bc21dff87525d44a904b24c2a81cdba96f13b2737219ae`
- `results/run_smoke_v96_belief_ledger_dialogue_csv_fix2_try1/memory_events.jsonl`: `69c1269628cc6a86771fefb52aafba1b86aa45691ce4e8b63108880e33607091`
- `results/run_smoke_v96_belief_ledger_dialogue_csv_fix2_try1/action_plans.jsonl`: `f3120c4df539a0c7bd7662925dcb2fd816dac638da1882069421e1af798ab5ae`
- `results/run_smoke_v96_belief_ledger_dialogue_csv_fix2_try1/summary.json`: `4bd1f41e65bbf0d46a675c964fc2edac45edb8b1c1bed53cc17113c9826cc097`

## GO/NO-GO
GO — V96 adiciona belief ledger WORM (ADD/RETRACT) com revisão explícita (revise=RETRACT+ADD), integra GET/EXPLAIN com auditoria (ActionPlanV96), verificação robusta e determinismo try1==try2, mantendo regressões V90–V95 e unittest verdes.

