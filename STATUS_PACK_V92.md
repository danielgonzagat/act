# STATUS PACK V92 — V92_COMPOUND_INTENT_GRAMMAR_DIALOGUE_CSV — GO (fix1)

## Resumo
- Intent grammar + parser V92 (EN+PT) com **fluência controlada**: tokens de cortesia ignoráveis (whitelist) apenas fora do span casado e sinônimo seguro `vars→variables`, com tie-break determinístico e ambiguidade fail-closed.
- Suporte a **compound commands** (split seguro por `;` e `\\n`) com política **all-or-nothing**: se qualquer segmento falhar/missing → não executa nenhum.
- Loop conversacional V92 (sem quebrar V90/V91): logs WORM hash-chained (`intent_parses.jsonl` com payload compound) + verificador V92 que valida cadeias JSONL, Turn↔Parse e regra “clarificação/confirm não cria estado”.

## Checklist
- PASS: `py_compile` (atos_core + scripts)
- PASS: `verify_freeze` V90 ok=true, warnings=[]
- PASS: smoke V90 (regressão) ok=true (WORM)
- PASS: smoke V91 (regressão) ok=true (WORM)
- PASS: smoke V92 try1==try2 (hashes idênticos)
- PASS: V92 “courtesy tokens” parseiam (ex.: `summary please`, `resumo por favor`, `por favor defina x como 4`)
- PASS: V92 compound OK agrega output determinístico (linhas) e executa tudo
- PASS: V92 fail-closed `blorp please` → COMM_CORRECT
- PASS: V92 ALL-OR-NOTHING `set x to 4; set z` → não executa nada + COMM_ASK_CLARIFY
- PASS: V92 `set x please` trata `please` como ignorável (missing v) e NÃO como valor
- PASS: negative test V92 (corrupção `turn.refs.parse_sig`) → fail-closed com reason determinístico

## Arquivos novos (V92)
- `atos_core/intent_grammar_v92.py:1`
- `atos_core/conversation_v92.py:1`
- `atos_core/conversation_loop_v92.py:1`
- `scripts/verify_conversation_chain_v92.py:1`
- `scripts/smoke_v92_compound_intents_dialogue_csv.py:1`
- `patches/v92_compound_intent_grammar_dialogue_csv_fix1.diff:1`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V92_COMPOUND_INTENT_GRAMMAR_DIALOGUE_CSV_fix1.json:1`
- `STATUS_PACK_V92.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py \
  --out_base results/run_smoke_v90_regression_v92 --seed 0 | tee results/smoke_v90_regression_v92_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py \
  --out_base results/run_smoke_v91_intent_grammar_dialogue_csv_regression_v92 --seed 0 \
  | tee results/smoke_v91_regression_v92_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py \
  --out_base results/run_smoke_v92_compound_intents_dialogue_csv_fix2 --seed 0 \
  | tee results/smoke_v92_compound_intents_dialogue_csv_fix2_try1.json

(diff -u /dev/null atos_core/conversation_loop_v92.py; \
 diff -u /dev/null atos_core/conversation_v92.py; \
 diff -u /dev/null atos_core/intent_grammar_v92.py; \
 diff -u /dev/null scripts/smoke_v92_compound_intents_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v92.py) \
 > patches/v92_compound_intent_grammar_dialogue_csv_fix1.diff

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V92_COMPOUND_INTENT_GRAMMAR_DIALOGUE_CSV_fix1.json \
  | tee results/verify_freeze_v92_fix1_try1.json
```

## Determinismo (try1 == try2)
Do smoke V92: `results/smoke_v92_compound_intents_dialogue_csv_fix2_try1.json`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `c0f877a9ebd8d23302f66cd238c6d514eb5dc9b0e9babdfc4be095965b1ae7ab`
- `state_chain_hash`: `f87fc8395ee59f1af19edec35cf38db08e55d288c5b5a63f881029a6987d65ec`
- `parse_chain_hash`: `39147f455b638e760ee410300c9604475af2b9a8f865ecbd501dcebd0363109e`
- `ledger_hash` (sha256 do `freeze_manifest_v92.json`): `836803df124fd4e4c8b5a2f1555b8cf4d54e04ba4cb9ddc572f4d97d9b757174`
- `summary_sha256`: `cf00ff2aa73b83f9add213580c8dda85c9984318f1567a69453b453e53101c96`

## Negative test V92 (fail-closed)
- Corrupção: `turn.refs.parse_sig="0"*64` → reason=`turn_parse_sig_mismatch`

## Artefatos principais (sha256)
- `patches/v92_compound_intent_grammar_dialogue_csv_fix1.diff`: `98e251522dd9533f72d93a397f70154d0ba9aab4c84b2e9737db51e2acdc58c1`
- `results/smoke_v92_compound_intents_dialogue_csv_fix2_try1.json`: `fc3860b0d826ef76ccfc2c734225e7760a2a75e9bcb4c44ce19efb5af63f9cc9`
- `results/run_smoke_v92_compound_intents_dialogue_csv_fix2_try1/freeze_manifest_v92.json`: `836803df124fd4e4c8b5a2f1555b8cf4d54e04ba4cb9ddc572f4d97d9b757174`
- `results/run_smoke_v92_compound_intents_dialogue_csv_fix2_try1/verify_chain_v92.json`: `33a8f38f705ea547469f6f1d3f2c1a973dc705928b3a1afdaf2c1b679b52d9c7`
- `results/smoke_v90_regression_v92_try1.json`: `bb31e7c348d20965952f6ccaf32c31e47bc9b9fb9c18a1edb3579c24acf05771`
- `results/smoke_v91_regression_v92_try1.json`: `59897b21b001136910d26d471727a981341c3575f9ef1af9654a54131bb2963d`

## GO/NO-GO
GO — V92 adiciona cortesias + compound commands com all-or-nothing, com determinismo try1==try2, logs WORM hash-chained + verify_chain ok, negative test fail-closed, e regressões V90/V91 ok.

