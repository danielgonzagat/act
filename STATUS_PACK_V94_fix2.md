# STATUS PACK V94 — V94_ACTION_PLANNING_EXPLAIN_DIALOGUE_CSV — GO (fix2)

## Resumo
- Adiciona **ActionPlan/DecisionTrace V94** como objeto explícito e auditável por user turn, persistido em `action_plans.jsonl` (WORM hash-chained) com `plan_sig`/`plan_id` determinísticos.
- Adiciona **EXPLAIN/EXPLIQUE (V94)** via intercept raw (fail-closed): responde com explicação determinística baseada no último plan anterior que não seja EXPLAIN.
- Verificador V94 (`verify_conversation_chain_v94`) cobre invariantes de plans (rank, tentativas, chosen=first-ok, cross-check com refs do assistant) + hash-chains de todos JSONL do run.

## Checklist
- PASS: `py_compile` (atos_core + scripts)
- PASS: regressões V90–V93 (smokes + unittest) ok
- PASS: smoke V94 try1==try2 (hashes idênticos)
- PASS: EXPLAIN explica o turno anterior (não “explain do explain”)
- PASS: `action_plans.jsonl` (1 plan por user turn) + `plan_chain_hash` estável
- PASS: negative test V94 (corrupção `plan_sig`) → fail-closed com reason determinístico

## Arquivos novos (V94)
- `atos_core/intent_grammar_v94.py:1`
- `atos_core/conversation_v94.py:1`
- `atos_core/conversation_loop_v94.py:1`
- `scripts/verify_conversation_chain_v94.py:1`
- `scripts/smoke_v94_action_planning_explain_dialogue_csv.py:1`
- `patches/v94_action_planning_explain_dialogue_csv_fix2.diff:1`
- `STATUS_PACK_V94_fix2.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (já executadas nesta rodada)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py \
  --out_base results/run_smoke_v90_regression_v94 --seed 0 | tee results/smoke_v90_regression_v94_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py \
  --out_base results/run_smoke_v91_intent_grammar_dialogue_csv_regression_v94 --seed 0 \
  | tee results/smoke_v91_regression_v94_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py \
  --out_base results/run_smoke_v92_compound_intents_dialogue_csv_regression_v94 --seed 0 \
  | tee results/smoke_v92_regression_v94_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py \
  --out_base results/run_smoke_v93_teachable_intent_grammar_dialogue_csv_regression_v94 --seed 0 \
  | tee results/smoke_v93_regression_v94_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V94
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py \
  --out_base results/run_smoke_v94_action_planning_explain_dialogue_csv_fix4 --seed 0 \
  | tee results/smoke_v94_action_planning_explain_dialogue_csv_fix4_try1.json

# Patch diff (WORM)
(diff -u /dev/null atos_core/conversation_loop_v94.py; \
 diff -u /dev/null atos_core/conversation_v94.py; \
 diff -u /dev/null atos_core/intent_grammar_v94.py; \
 diff -u /dev/null scripts/smoke_v94_action_planning_explain_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v94.py) \
 > patches/v94_action_planning_explain_dialogue_csv_fix2.diff
```

## Determinismo (try1 == try2)
Do smoke V94: `results/smoke_v94_action_planning_explain_dialogue_csv_fix4_try1.json`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `d0657b1fdd5830afecdebabdbaa703f9738626a561a1d10dec4f2584585679a6`
- `state_chain_hash`: `bd145a7b0952c3d4b02311c6a0302dae1f8d5f18e8e39964b3dd43833d5b062c`
- `parse_chain_hash`: `5e641db79ee800ef97149b2077dc59bc5629415952b64f7f982f9b4e5d150021`
- `learned_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `plan_chain_hash`: `da9346640c9ca4ad21cbda059f39c80ccd9511df4b57a5400801819ca56429c4`
- `ledger_hash` (sha256 do `freeze_manifest_v94.json`): `d367534acdba1a55e7b4db853616662241f5a6280930746f42b21a175d23fa06`
- `summary_sha256`: `f16fdbdb33dc78caf1242a9529643430663ae4300717cfd6032e3074ac426094`

## Negative test V94 (fail-closed)
- Corrupção: `plan_sig="0"*64` → reason=`plan_sig_mismatch`

## Artefatos principais (sha256)
- `patches/v94_action_planning_explain_dialogue_csv_fix2.diff`: `69a07141293dfa5e18bdd4a5a0806f3afb199f6715735103c8783cee39b3308c`
- `results/smoke_v94_action_planning_explain_dialogue_csv_fix4_try1.json`: `0e42a42212afe0f71f5bb7430187778d3e355f9021160f960a8b34984b831575`
- `results/run_smoke_v94_action_planning_explain_dialogue_csv_fix4_try1/freeze_manifest_v94.json`: `d367534acdba1a55e7b4db853616662241f5a6280930746f42b21a175d23fa06`
- `results/run_smoke_v94_action_planning_explain_dialogue_csv_fix4_try1/verify_chain_v94.json`: `197b2b0b992c66d58bc6f8c54b3fa0b51288104526d5987068c13162a76fc600`
- `results/smoke_v90_regression_v94_try1.json`: `6868ce18f48378c83022012b74f5f15442f5a3b43e57c0d5064fea0abfb104b2`
- `results/smoke_v91_regression_v94_try1.json`: `bf4dd2156bd1b01064a566510975c1b9270a0a08b676d3af6ee0e8304020bbac`
- `results/smoke_v92_regression_v94_try1.json`: `d4e8d33386493a57e068e47c1738aafa11842efaa3fc30e3e6531bee9863329d`
- `results/smoke_v93_regression_v94_try1.json`: `9cfefbffa8c2cd4de49b57d31bdcddd5a0f229e054b20d0086fe19aea8da2b56`

## GO/NO-GO
GO — V94 adiciona decision-trace explícito (`action_plans.jsonl`) + EXPLAIN determinístico baseado em logs, com determinismo try1==try2 e negative test fail-closed, mantendo regressões V90–V93 ok.

