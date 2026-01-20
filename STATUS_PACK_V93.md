# STATUS PACK V93 — V93_TEACHABLE_INTENT_GRAMMAR_DIALOGUE_CSV — GO (fix1)

## Resumo
- Adiciona `TEACH/ENSINE` (intercept raw, sem heurística por texto solta): `teach: <lhs> => <rhs>` cria uma **IntentRule learned** (lit-only, sem slots) com `rule_id`/`rule_sig` determinísticos e provenance explícita.
- Integra learned rules no parsing (V92 parser) sem quebrar V90–V92: regras ativas entram no conjunto de parse e mudam comportamento imediatamente na conversa.
- Persistência auditável: `learned_intent_rules.jsonl` WORM hash-chained + invariantes no `verify_conversation_chain_v93` (TEACH aceita → cria state e learned event; TEACH rejeita → não cria state nem learned event).

## Checklist
- PASS: `py_compile` (atos_core + scripts)
- PASS: `verify_freeze` V90 ok=true, warnings=[]
- PASS: smoke V90 (regressão) ok=true (WORM)
- PASS: smoke V91 (regressão) ok=true (WORM)
- PASS: smoke V92 (regressão) ok=true (WORM)
- PASS: `python3 -m unittest -v` (suite curta) ok
- PASS: smoke V93 try1==try2 (hashes idênticos)
- PASS: TEACH aceito altera comportamento (`show all vars` deixa de ser `COMM_CORRECT` e vira `SUMMARY`)
- PASS: TEACH inválido (`rhs` com slots) é rejeitado e **não atualiza state**
- PASS: TEACH ambíguo é rejeitado com `COMM_CONFIRM` e **não atualiza state**
- PASS: negative test V93 (corrupção `learned_rule.rule_sig`) → fail-closed com reason determinístico

## Arquivos novos (V93)
- `atos_core/intent_grammar_v93.py:1`
- `atos_core/conversation_v93.py:1`
- `atos_core/conversation_loop_v93.py:1`
- `scripts/verify_conversation_chain_v93.py:1`
- `scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py:1`
- `patches/v93_teachable_intent_grammar_dialogue_csv_fix1.diff:1`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V93_TEACHABLE_INTENT_GRAMMAR_DIALOGUE_CSV_fix1.json:1`
- `STATUS_PACK_V93.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py \
  --out_base results/run_smoke_v90_regression_v93 --seed 0 | tee results/smoke_v90_regression_v93_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py \
  --out_base results/run_smoke_v91_intent_grammar_dialogue_csv_regression_v93 --seed 0 \
  | tee results/smoke_v91_regression_v93_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py \
  --out_base results/run_smoke_v92_compound_intents_dialogue_csv_regression_v93 --seed 0 \
  | tee results/smoke_v92_regression_v93_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py \
  --out_base results/run_smoke_v93_teachable_intent_grammar_dialogue_csv_fix2 --seed 0 \
  | tee results/smoke_v93_teachable_intent_grammar_dialogue_csv_fix2_try1.json

(diff -u /dev/null atos_core/conversation_loop_v93.py; \
 diff -u /dev/null atos_core/conversation_v93.py; \
 diff -u /dev/null atos_core/intent_grammar_v93.py; \
 diff -u /dev/null scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v93.py) \
 > patches/v93_teachable_intent_grammar_dialogue_csv_fix1.diff

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V93_TEACHABLE_INTENT_GRAMMAR_DIALOGUE_CSV_fix1.json \
  | tee results/verify_freeze_v93_fix1_try1.json
```

## Determinismo (try1 == try2)
Do smoke V93: `results/smoke_v93_teachable_intent_grammar_dialogue_csv_fix2_try1.json`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `d68e51d2088e0ac77b9e2f03dafa3c80af2f0c99c59d88405012fc267b26469b`
- `state_chain_hash`: `fbe9ab9f87b8e5ff064b7bd91df63083272946948a3427ac21b7e4681e69ad11`
- `parse_chain_hash`: `3c4b45a19268ab5c8adcc6c6060e438d1d1cddd84ea680e8685b5be6893412ce`
- `learned_chain_hash`: `d0247cf2f13cda7148dc6f1063c4ed41c6433c03959a176d9319d96f9d044bd1`
- `ledger_hash` (sha256 do `freeze_manifest_v93.json`): `084067154ee484cd31c74c0c8b7f8ae7bd99de34a381e2fb677ab8d280a4a3cd`
- `summary_sha256`: `92686b48d4c5c19b0aed02e7e4d278cfc46c4d2f991f47b52e4dc0ef58f479cb`

## Negative test V93 (fail-closed)
- Corrupção: `learned_rule.rule_sig="0"*64` → reason=`rule_sig_mismatch`

## Artefatos principais (sha256)
- `patches/v93_teachable_intent_grammar_dialogue_csv_fix1.diff`: `5bf46487b9b46ad7e881572c6d8a08b1d5b0af8d08e684f328ed6ba0a1ed5f2e`
- `results/smoke_v93_teachable_intent_grammar_dialogue_csv_fix2_try1.json`: `1df7f989c2aee686a10a53011b6aa331f11716721653031b1c9d54830ceb53c7`
- `results/run_smoke_v93_teachable_intent_grammar_dialogue_csv_fix2_try1/freeze_manifest_v93.json`: `084067154ee484cd31c74c0c8b7f8ae7bd99de34a381e2fb677ab8d280a4a3cd`
- `results/run_smoke_v93_teachable_intent_grammar_dialogue_csv_fix2_try1/verify_chain_v93.json`: `98a132e21bf8e196e9b6c47310753be53f7d60a782df492605bc6b3126a374c8`
- `results/smoke_v90_regression_v93_try1.json`: `b415d241c770da63207d1d1e6d9fb4582bc868c409c802e2daf55a69bd861363`
- `results/smoke_v91_regression_v93_try1.json`: `eb59768fb70084fc936555c43e6fdb92b79f18cce439a03f42e8b4a0b6792fcd`
- `results/smoke_v92_regression_v93_try1.json`: `80dfd2d330c37731a55afbb1e6d04f23627bc89b42da60d36bf0b31380ead98d`

## GO/NO-GO
GO — V93 adiciona aprendizado auditável (aliases lit-only) com TEACH/ENSINE, logs WORM hash-chained + verify_chain V93, determinismo try1==try2, e regressões V90–V92 + unittest ok.

