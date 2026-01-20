# STATUS PACK V91 — V91_INTENT_GRAMMAR_DIALOGUE_CSV — GO

## Resumo (o que foi implementado)
- **IntentGrammar V91 determinística (EN+PT)**: regras explícitas (`intent_rule_v91`) com `rule_sig`, tie-break determinístico (especificidade → comprimento → `rule_id`).
- **Parser V91 fail-closed**: normalização (lowercase + accent fold + pontuação simples) + tokenização + números por tabela fixa; saída com `parse_sig` e logs WORM (`intent_parses.jsonl`).
- **Loop conversacional V91**: usa objetivos V90 (Objective CSV) + ações V90 (concept_csv) e escolhe `COMM_*` via parse; estado só avança quando apropriado (não avança em `COMM_ASK_CLARIFY`/`COMM_CONFIRM`).
- **Verifier V91**: valida hash-chains de todos JSONL + vínculo Turn↔Parse (`turn.refs.parse_sig`) + regra “não cria estado em clarificação”.
- **Smoke V91**: 12 turns (EN+PT), 2 clarificações (missing key + missing slot), 1 unknown (fail-closed via `COMM_CORRECT`), determinismo try1==try2.

## Checklist (PASS/FAIL)
- PASS: Step 0 re-sync (py_compile + verify_freeze V90 + smoke V90 regression)
- PASS: Smoke V91 try1/try2 determinístico (hashes idênticos)
- PASS: `verify_chain_v91.json` ok=true (hash-chains + invariantes)
- PASS: Negative test V91 (corrupção `turn.refs.parse_sig` → reason=`turn_parse_sig_mismatch`)
- PASS: Regressões: smoke V90 (novamente) + `python3 -m unittest -v`
- PASS: Ledger V91 + `verify_freeze` ok=true, `warnings=[]`

## Arquivos criados (somente novos)
- `atos_core/intent_grammar_v91.py:1`
- `atos_core/conversation_v91.py:1`
- `atos_core/conversation_loop_v91.py:1`
- `scripts/verify_conversation_chain_v91.py:1`
- `scripts/smoke_v91_intent_grammar_dialogue_csv.py:1`
- `patches/v91_intent_grammar_dialogue_csv_fix1.diff:1`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V91_INTENT_GRAMMAR_DIALOGUE_CSV.json:1`
- `results/verify_freeze_v91_try1.json:1`
- `STATUS_PACK_V91.md:1`

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

# STEP 0 — re-sync (read-only)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json
test ! -e results/smoke_v90_regression_try1.json && \
  PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py \
    --out_base results/run_smoke_v90_regression --seed 0 | tee results/smoke_v90_regression_try1.json

# STEP 5 — smoke V91 (try1/try2 dentro do script)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py \
  --out_base results/run_smoke_v91_intent_grammar_dialogue_csv_fix1 --seed 0 \
  | tee results/smoke_v91_intent_grammar_dialogue_csv_fix1_try1.json

# STEP 6 — regressões mínimas
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json
test ! -e results/smoke_v90_regression2_try1.json && \
  PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py \
    --out_base results/run_smoke_v90_regression2 --seed 0 | tee results/smoke_v90_regression2_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# STEP 7 — diff + freeze
(diff -u /dev/null atos_core/conversation_loop_v91.py; \
 diff -u /dev/null atos_core/conversation_v91.py; \
 diff -u /dev/null atos_core/intent_grammar_v91.py; \
 diff -u /dev/null scripts/smoke_v91_intent_grammar_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v91.py) \
 > patches/v91_intent_grammar_dialogue_csv_fix1.diff

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V91_INTENT_GRAMMAR_DIALOGUE_CSV.json \
  | tee results/verify_freeze_v91_try1.json
```

## Métricas chave (smoke V91 try1)
- `turns_total=24`, `user_turns_total=12`
- `parses_total=12`, `parses_ok=10`, `unknowns=1`
- `clarifications=2` (missing key + missing slot)
- `states_total=10` (não cria state em clarificação/confirm)

## Prova de determinismo (try1 == try2)
Smoke stdout: `results/smoke_v91_intent_grammar_dialogue_csv_fix1_try1.json`
- `store_hash`: `22c9af432c461322ddfb7a5120d2231c365f0f921e5a89708b4df973c452bb51`
- `transcript_hash`: `b7e27d35f36779a13d0b96c3e44d0831fd192856731e38dbfcc3c87bbbd8d86e`
- `state_chain_hash`: `13434b1915624cf79a5ac34f55b8df3e17af4de841808571301e6cfe018a0d15`
- `parse_chain_hash`: `1216125afa8668208fe1b63f619c8d4fc64b39d09ae1ef99e9567b014be483ee`
- `ledger_hash` (sha256 do `freeze_manifest_v91.json`): `72c2909f80313c62db6940d00837d29174e898faca0e4e49a6f780baa4bf18f2`
- `summary_sha256`: `996d02309367be0d9c873047d0222b28691e4bcd7fb56b27692a89d1ba2f4d7a`

## Negative test (fail-closed)
- Corrupção em memória: `turn.refs.parse_sig = "0"*64` no primeiro user turn → verificador falha com `reason="turn_parse_sig_mismatch"`.

## Artefatos + SHA256 (principais)
- Patch: `patches/v91_intent_grammar_dialogue_csv_fix1.diff` → `0304711ae2f098438b6f3dccde31d319dd1b2aa627f8a85a38469cd67604d46c`
- Smoke stdout (tee): `results/smoke_v91_intent_grammar_dialogue_csv_fix1_try1.json` → `2b563b37141d534c561a9d27874910f138c19059777749328ca07bd75cef6853`
- Ledger: `LEDGER_ATOLANG_V0_2_30_BASELINE_V91_INTENT_GRAMMAR_DIALOGUE_CSV.json` → `3e9c638a0ec18e34840e6a8c1c66a2e72701cff35834de1c87246472cfaf255b`
- verify_freeze: `results/verify_freeze_v91_try1.json` → `b314613964147f8649f2cabf9679cb06fb14a1fdf05ef39843de3da1a4b3adec`

**Try1 core: `results/run_smoke_v91_intent_grammar_dialogue_csv_fix1_try1/`**
- `store.jsonl` → `87367b9208556ba2ac26c9208371e338bcc9221e388a07f5014775dd417b44bd`
- `intent_grammar_snapshot.json` → `49171fbf2abee4ebb85c74bcc2b52263f3f142fe6c48ac74a0b65cf288052d99`
- `conversation_turns.jsonl` → `1e473f2fe2dfc796c107f447f744343fb075b064581375bd181c48c31f0754ea`
- `intent_parses.jsonl` → `8bbd383661d3acd43432c801f0c7a7319b620cbfd91fa1f22bc1ef98da0ad5ee`
- `conversation_states.jsonl` → `529a4cab5cea578bb722ca6eb5c9098787aec74558823937b6aeb92a2dfd057f`
- `dialogue_trials.jsonl` → `5ee6718492c6983c2d170eb34d6300d387ae5d92c3085910b02bc1ac734a1b2e`
- `objective_evals.jsonl` → `f467ef823ca9f10c9a054cb46f74be07965ee1f952b0f94957b17068c49dc973`
- `transcript.jsonl` → `7d0074ab4219dfde0bac911bbec116f5bb348b3c6cf1285a6cb3c7618e9e0083`
- `verify_chain_v91.json` → `e0ef241f6512c5b06785ec95853622a077c7e33d4446d734c8ec73ab24a19bd8`
- `freeze_manifest_v91.json` → `72c2909f80313c62db6940d00837d29174e898faca0e4e49a6f780baa4bf18f2`
- `summary.json` → `8e3e79ae64d854fcac2306d274cea35d4e82c6895ae89629f1ab16eed2c5f297`

**Try2 core (idêntico por hash):** `results/run_smoke_v91_intent_grammar_dialogue_csv_fix1_try2/`

**Regressão V90 (smokes):**
- `results/smoke_v90_regression_try1.json` → `4e90fa94dd7229046c7e6cee960ae4cfdf5099d07ccddf6912cfd90bb3423e0a`
- `results/smoke_v90_regression2_try1.json` → `f954fa3e4f43d496cdf8b4548266bc20225bb165219bf6746d2f36a46effbb96`

## Decisão final
**GO** — V91 passa smoke determinístico (try1==try2), parser+logs WORM, verificador+negative test fail-closed, regressão V90 intacta, ledger + verify_freeze ok (warnings=[]).

