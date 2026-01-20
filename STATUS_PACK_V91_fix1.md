# STATUS PACK V91 — V91_INTENT_GRAMMAR_DIALOGUE_CSV — GO (fix1)

## Resumo
- Intent grammar determinística (EN+PT) como **estado explícito**: regras `intent_rule_v91` com `rule_sig` e tie-break determinístico.
- Parser determinístico + logs WORM: `intent_parses.jsonl` hash-chained + `parse_sig` referenciado em `turn_v91.refs`.
- Loop conversacional V91: objetivos como Objective CSV (V90) + ações como `concept_csv` (V90), com fallback fail-closed (sem alucinação).
- Verificação V91: valida hash-chain de JSONL + vínculo Turn↔Parse + regra “não cria estado em clarificação/confirm”.

## Checklist
- PASS: `py_compile` (atos_core + scripts)
- PASS: `verify_freeze` V90 ok=true, warnings=[]
- PASS: smoke V90 (regressão) ok=true (WORM)
- PASS: smoke V91 try1==try2 (hashes idênticos)
- PASS: negative test V91 (corrupção parse_sig) → fail-closed com reason determinístico
- PASS: `verify_freeze` V91 (ok=true, warnings=[])

## Arquivos novos (V91)
- `atos_core/intent_grammar_v91.py:1`
- `atos_core/conversation_v91.py:1`
- `atos_core/conversation_loop_v91.py:1`
- `scripts/verify_conversation_chain_v91.py:1`
- `scripts/smoke_v91_intent_grammar_dialogue_csv.py:1`
- `patches/v91_intent_grammar_dialogue_csv_fix1.diff:1`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V91_INTENT_GRAMMAR_DIALOGUE_CSV_fix1.json:1`
- `STATUS_PACK_V91_fix1.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py \
  --out_base results/run_smoke_v90_regression2 --seed 0 | tee results/smoke_v90_regression2_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py \
  --out_base results/run_smoke_v91_intent_grammar_dialogue_csv_fix1 --seed 0 \
  | tee results/smoke_v91_intent_grammar_dialogue_csv_fix1_try1.json

(diff -u /dev/null atos_core/conversation_loop_v91.py; \
 diff -u /dev/null atos_core/conversation_v91.py; \
 diff -u /dev/null atos_core/intent_grammar_v91.py; \
 diff -u /dev/null scripts/smoke_v91_intent_grammar_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v91.py) \
 > patches/v91_intent_grammar_dialogue_csv_fix1.diff

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V91_INTENT_GRAMMAR_DIALOGUE_CSV_fix1.json \
  | tee results/verify_freeze_v91_fix1_try1.json
```

## Determinismo (try1 == try2)
Do smoke V91: `results/smoke_v91_intent_grammar_dialogue_csv_fix1_try1.json`
- `store_hash`: `22c9af432c461322ddfb7a5120d2231c365f0f921e5a89708b4df973c452bb51`
- `transcript_hash`: `b7e27d35f36779a13d0b96c3e44d0831fd192856731e38dbfcc3c87bbbd8d86e`
- `state_chain_hash`: `13434b1915624cf79a5ac34f55b8df3e17af4de841808571301e6cfe018a0d15`
- `parse_chain_hash`: `1216125afa8668208fe1b63f619c8d4fc64b39d09ae1ef99e9567b014be483ee`
- `ledger_hash` (sha256 do `freeze_manifest_v91.json`): `72c2909f80313c62db6940d00837d29174e898faca0e4e49a6f780baa4bf18f2`
- `summary_sha256`: `996d02309367be0d9c873047d0222b28691e4bcd7fb56b27692a89d1ba2f4d7a`

## Negative test V91 (fail-closed)
- Corrupção: `turn.refs.parse_sig="0"*64` → reason=`turn_parse_sig_mismatch`

## Artefatos principais (sha256)
- `patches/v91_intent_grammar_dialogue_csv_fix1.diff`: `0304711ae2f098438b6f3dccde31d319dd1b2aa627f8a85a38469cd67604d46c`
- `results/smoke_v91_intent_grammar_dialogue_csv_fix1_try1.json`: `2b563b37141d534c561a9d27874910f138c19059777749328ca07bd75cef6853`
- `results/run_smoke_v91_intent_grammar_dialogue_csv_fix1_try1/freeze_manifest_v91.json`: `72c2909f80313c62db6940d00837d29174e898faca0e4e49a6f780baa4bf18f2`
- `results/run_smoke_v91_intent_grammar_dialogue_csv_fix1_try1/verify_chain_v91.json`: `e0ef241f6512c5b06785ec95853622a077c7e33d4446d734c8ec73ab24a19bd8`
- `results/smoke_v90_regression2_try1.json`: `f954fa3e4f43d496cdf8b4548266bc20225bb165219bf6746d2f36a46effbb96`

## GO/NO-GO
GO — determinismo try1==try2, logs WORM + verify_chain ok, negative test fail-closed, verify_freeze ok e V90 sem regressão.

