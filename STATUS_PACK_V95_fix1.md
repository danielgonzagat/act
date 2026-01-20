# STATUS PACK V95 — V95_MEMORY_LEDGER_DIALOGUE_CSV — GO (fix1)

## Resumo
- Adiciona **V95 NOTE/RECALL/FORGET** (intercept raw, bilíngue, fail-closed) com novos intents `INTENT_NOTE/INTENT_RECALL/INTENT_FORGET`.
- Adiciona **Memory Ledger V95** como log WORM hash-chained: `memory_events.jsonl` com eventos `ADD`/`RETRACT` e itens `memory_item_v95` com `memory_id`/`memory_sig` determinísticos.
- Integra memória ao estado: `ConversationStateV95.bindings.memory_active` + `memory_active_count` (deriváveis por replay do ledger).
- Integra memória ao decision trace: **ActionPlanV95** inclui `memory_read_ids` e `memory_write_event_ids` para auditoria de “o que leu/escreveu”.
- Verificador V95 (`verify_conversation_chain_v95`) cobre invariantes V94 + replay de memória + cross-check do texto de RECALL via renderer determinístico.

## Checklist
- PASS: `py_compile` (atos_core + scripts)
- PASS: regressões V90–V94 (smokes) ok
- PASS: `python3 -m unittest -v` ok
- PASS: smoke V95 try1==try2 (hashes idênticos, incluindo `memory_chain_hash`)
- PASS: NOTE cria evento `ADD` e atualiza `memory_active`
- PASS: RECALL renderiza somente `memory_active` (`MEMORY:` / `MEMORY: (empty)`)
- PASS: FORGET last gera `RETRACT` e remove da memória ativa (fail-closed em `no_active_memory`)
- PASS: negative test V95 (corrupção `event_sig`) → fail-closed com reason determinístico `event_sig_mismatch`

## Arquivos novos (V95)
- `atos_core/intent_grammar_v95.py:1`
- `atos_core/conversation_v95.py:1`
- `atos_core/conversation_loop_v95.py:1`
- `scripts/verify_conversation_chain_v95.py:1`
- `scripts/smoke_v95_memory_ledger_dialogue_csv.py:1`
- `patches/v95_memory_ledger_dialogue_csv_fix1.diff:1`
- `STATUS_PACK_V95_fix1.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (V90–V94)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v95 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v95 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v95 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v95 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v95 --seed 0

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V95 (try1/try2 dentro do script)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py \
  --out_base results/run_smoke_v95_memory_ledger_dialogue_csv_fix1 --seed 0 \
  | tee results/smoke_v95_memory_ledger_dialogue_csv_fix1_try1.json

# Patch diff (WORM) — somente arquivos V95
(diff -u /dev/null atos_core/conversation_loop_v95.py; \
 diff -u /dev/null atos_core/conversation_v95.py; \
 diff -u /dev/null atos_core/intent_grammar_v95.py; \
 diff -u /dev/null scripts/smoke_v95_memory_ledger_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v95.py) \
 > patches/v95_memory_ledger_dialogue_csv_fix1.diff
```

## Determinismo (try1 == try2)
Do smoke V95: `results/smoke_v95_memory_ledger_dialogue_csv_fix1_try1.json`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `7c1d219221e5b645db5d7ba8edc8bfd8cde718132d8fcfd1ce4d76c00f4270c5`
- `state_chain_hash`: `b71ac44951e4c0e37ada4d7bcbac81d7f76d4a0e5616c3e05a90c40e77455000`
- `parse_chain_hash`: `11e87fcc502ef0b1fe7475b77572876205b80e6e966fa055c82e2f7f924a3f2e`
- `learned_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `plan_chain_hash`: `e4b8fb3b59e4138ecee67a9a2d465f81b4d7ad04c206c0174874cdee4cfe4e25`
- `memory_chain_hash`: `dffad9e2d145a8b92c0d0e2b982677c1dc543a43d0bf759124d8c6cd66ae6e9b`
- `ledger_hash` (sha256 do `freeze_manifest_v95.json`): `ac44b848f65ede48f782429354b0af230fd35d21c018365201f46c6f337c525b`
- `summary_sha256`: `81d3f725bc034afe95c4c06f563c066126a4089a2a32af2179ec3ca4eb5ca6e0`

## Negative test V95 (fail-closed)
- Corrupção: `memory_events[0].event_sig="0"*64` → reason=`event_sig_mismatch`

## Artefatos principais (sha256)
- `patches/v95_memory_ledger_dialogue_csv_fix1.diff`: `a854b6f96693e42a9505e9a7d838f1995cf009e374b17b9c59a3abe50fb51a89`
- `results/smoke_v95_memory_ledger_dialogue_csv_fix1_try1.json`: `c1dc5439c090f093d6f1f2a8b8089068f96e459389f54df025dcc14cfb34aa91`
- `results/run_smoke_v95_memory_ledger_dialogue_csv_fix1_try1/freeze_manifest_v95.json`: `ac44b848f65ede48f782429354b0af230fd35d21c018365201f46c6f337c525b`
- `results/run_smoke_v95_memory_ledger_dialogue_csv_fix1_try1/verify_chain_v95.json`: `28d674399f5b4776642f90debc64802dfd6ec0a712a5e8f58ed94b06a6d3b717`
- `results/run_smoke_v95_memory_ledger_dialogue_csv_fix1_try1/memory_events.jsonl`: `ce1a4d8ba6bbed269c2bed3630e90f5cf907c87de40c5dbea997dc761e5ddb47`
- `results/run_smoke_v95_memory_ledger_dialogue_csv_fix1_try1/action_plans.jsonl`: `d676a33dbf7507efc754a13337d3c6ea06816ada0582d1b6c3c42b36c2c2133b`
- `results/run_smoke_v95_memory_ledger_dialogue_csv_fix1_try1/summary.json`: `dc868ff7f0ffc0b31cf3b7d297d84eb0e7345d9d304349ca01eb61b53b83cc77`

## GO/NO-GO
GO — V95 adiciona memória persistente (NOTE/RECALL/FORGET) como ledger WORM hash-chained, integrada ao estado e ao decision-trace, com determinismo try1==try2 e negative test fail-closed, mantendo regressões V90–V94 ok.

