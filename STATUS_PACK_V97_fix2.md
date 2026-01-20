# STATUS PACK V97 — V97_SELF_EXPLAIN_LIVE_LEARNING — GO (fix2)

> Nota WORM: houve uma tentativa inicial “fix1” que falhou por bug de implementação (criou `results/run_smoke_v97_self_explain_live_learning_fix1_try1/` e um `tee` vazio). Como WORM proíbe sobrescrita/reuso, o baseline V97 foi fechado como **fix2**.

## Resumo
- Adiciona comando **SYSTEM/SISTEMA/MANUAL/ABOUT** (intercept raw, determinístico) que retorna um texto de auditoria e grava snapshot WORM `system_spec_snapshot.json`.
- Demonstra **aprendizado ao vivo** via `TEACH`: regra aprendida `statusz => beliefs`, usada imediatamente no turno seguinte; regra fica em `learned_intent_rules.jsonl` (hash-chained) e é referenciada no `ActionPlan` como proveniência.
- Fortalece **EXPLAIN** com renderer V97 (`render_explain_text_v97`) que mostra TOP‑K alternativas + score determinístico + tentativas + efeitos (memory/belief) e proveniência de regra aprendida, sem “racionalização”.
- Verificador V97: wrapper sobre V96 que exige que, quando o parse usar regra aprendida, isso apareça em `plan.provenance.learned_rule_ids`; e o script `verify_conversation_chain_v97.py` valida `system_spec_snapshot.json` contra `freeze_manifest_v97.json`.

## Checklist
- PASS: `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS: regressões V90–V96 (smokes) + `python3 -m unittest -v`
- PASS: smoke V97 try1==try2 (hashes idênticos, incluindo `system_spec_sha256`)
- PASS: `system_spec_snapshot.json` existe e `SYSTEM` responde exatamente `render_system_text_v97(snapshot)`
- PASS: live learning: `teach: statusz => beliefs` cria 1 learned rule e `statusz` usa essa regra (parse + plan provenance)
- PASS: `EXPLAIN` responde exatamente `render_explain_text_v97(last_plan_before_explain)`
- PASS: negative tamper: corrupção `learned_rule.rule_sig="0"*64` → fail-closed com reason determinístico `rule_sig_mismatch`

## Arquivos novos (V97)
- `atos_core/intent_grammar_v97.py:1`
- `atos_core/conversation_v97.py:1`
- `atos_core/conversation_loop_v97.py:1`
- `scripts/verify_conversation_chain_v97.py:1`
- `scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py:1`
- `patches/v97_self_explain_live_learning_fix2.diff:1`
- `STATUS_PACK_V97_fix2.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (V90–V96)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v97_fix1 --seed 0 | tee results/smoke_v90_regression_v97_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v97_fix1 --seed 0 | tee results/smoke_v91_regression_v97_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v97_fix1 --seed 0 | tee results/smoke_v92_regression_v97_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v97_fix1 --seed 0 | tee results/smoke_v93_regression_v97_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v97_fix1 --seed 0 | tee results/smoke_v94_regression_v97_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v97_fix1 --seed 0 | tee results/smoke_v95_regression_v97_fix1_try1.json
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v97_fix1 --seed 0 | tee results/smoke_v96_regression_v97_fix1_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V97 (try1/try2 dentro do script; WORM)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py \
  --out_base results/run_smoke_v97_self_explain_live_learning_fix2 --seed 0 \
  | tee results/smoke_v97_self_explain_live_learning_fix2_try1.json

# Patch diff (WORM) — somente arquivos V97
(diff -u /dev/null atos_core/conversation_loop_v97.py; \
 diff -u /dev/null atos_core/conversation_v97.py; \
 diff -u /dev/null atos_core/intent_grammar_v97.py; \
 diff -u /dev/null scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v97.py) \
 > patches/v97_self_explain_live_learning_fix2.diff
```

## Determinismo (try1 == try2)
Do smoke V97: `results/smoke_v97_self_explain_live_learning_fix2_try1.json`
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `system_spec_sha256`: `6d91ed7978557a4d16145e0f8a84a0e9e2629c76213cd3ea869acf6281584090`
- `transcript_hash`: `63fb548aef047bd7344e10055b603535501928760cac09f99ab9da41402b9496`
- `state_chain_hash`: `53f9c74a61a1e5cf0e0390ab07b7171c0928dace989dbc7a025c0bf316cb68a9`
- `parse_chain_hash`: `1a3174495a4657655e66ecce8ba4f17253ec7d9b7f08c6637a1e45f4284670a0`
- `learned_chain_hash`: `d07c83af7f1c0078100f5058074cf34e1497bcd4bff1657ef9e39e44590bd03f`
- `plan_chain_hash`: `9a390e6f96301238c0cf75a11b6e890b5d09478e0cddf1c92066e214d61780c8`
- `memory_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `belief_chain_hash`: `ebd960049569846ea5d6d7dcdfd5f9daafb597f1bb90d71d99820ceb6c545e51`
- `ledger_hash` (sha256 do `freeze_manifest_v97.json`): `7cf9405e27393859b5df96d492d9be0b70535e4bf9aa899a62f23dc24e86d0c8`
- `summary_sha256`: `fcc8b796fbccbb901e974cd7d67058626164fc984ef404c5663d45053cb05b88`

## Negative test V97 (fail-closed)
- Corrupção: `learned_events[0].learned_rule.rule_sig="0"*64` → reason=`rule_sig_mismatch`

## Artefatos principais (sha256)
- `patches/v97_self_explain_live_learning_fix2.diff`: `4a3cc06ad64fde3f1ffee6e3389a9ff3a7013e447d4067bf85d26be948e87e61`
- `results/smoke_v97_self_explain_live_learning_fix2_try1.json`: `a7706487cdb7e628196180afbd2a2f6a8be78198f1ae1b8ce6e738e16782552f`
- `results/run_smoke_v97_self_explain_live_learning_fix2_try1/system_spec_snapshot.json`: `6d91ed7978557a4d16145e0f8a84a0e9e2629c76213cd3ea869acf6281584090`
- `results/run_smoke_v97_self_explain_live_learning_fix2_try1/learned_intent_rules.jsonl`: `6a2c307db2d77c0f26639958bd218ec3923316b5e000c1f2ea12f1102406476c`
- `results/run_smoke_v97_self_explain_live_learning_fix2_try1/action_plans.jsonl`: `dc56037cc5b66d0ada1ef05a2a0c49286d13f7958da9e39966d6646783a95161`
- `results/run_smoke_v97_self_explain_live_learning_fix2_try1/freeze_manifest_v97.json`: `7cf9405e27393859b5df96d492d9be0b70535e4bf9aa899a62f23dc24e86d0c8`
- `results/run_smoke_v97_self_explain_live_learning_fix2_try1/verify_chain_v97.json`: `cdd9366ffc5f6c70e40570ce630b3cfbd9d13a16ca8c15daf0c25241dc367325`
- `results/run_smoke_v97_self_explain_live_learning_fix2_try1/summary.json`: `eee583a0dfe131e40ce9fc76deac968715c2aa92c6237ca8d22a4457888ec33e`

## GO/NO-GO
GO — V97 adiciona `SYSTEM` (self-explain auditável) + demo de aprendizado ao vivo (TEACH → uso imediato, WORM) + EXPLAIN com top‑k/proveniência/efeitos, mantendo determinismo try1==try2 e regressões V90–V96 + unittest verdes.

