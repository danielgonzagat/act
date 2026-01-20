# STATUS PACK V100 — V100_FLUENCY_DISCOURSE_CONCEPTS_DOSSIER — GO (fix1)

## Resumo
- Adiciona **Discourse Ledger V100** (`discourse_events.jsonl`, hash-chained + `event_sig` por evento) com **candidatos TOP‑K (>=3)**, métricas determinísticas e seleção do melhor.
- Implementa **fluency como pressão estrutural**: geração de variantes determinísticas, scoring explícito (`fluency_metrics_v100` + `fluency_score_v100`) e seleção auditável via ledger + `EXPLAIN` v100.
- Introduz **conceitos vivos no domínio do discurso** via **fragmentos** (`fragment_events.jsonl` + `fragment_library_snapshot.json`) com **promoção determinística** por uso/score.
- Adiciona `discourse` (estado discursivo estruturado) e `dossier` determinístico/verificável, além de verificador V100 com **no‑hybridization** (fail‑closed se detectar deps proibidas).

## Checklist
- PASS: `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py`
- PASS: regressões V90–V99 (smokes) + `PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v`
- PASS: smoke V100 try1==try2 (inclui `discourse_chain_hash` e `fragment_chain_hash`)
- PASS: `discourse_events.jsonl` com `candidates_topk` (>=3) + seleção determinística do best‑by‑score
- PASS: fragmento promovido (`frag_v100_prefix_answer_v0`) com evidência (>=3 USE) em `fragment_library_snapshot.json`
- PASS: negative tamper V100: corrupção `discourse_events[0].event_sig="0"*64` ⇒ fail‑closed reason `discourse_event_sig_mismatch`
- PASS: no‑hybridization (verificador V100) — sem dependências proibidas detectadas

## Arquivos novos (V100)
- `atos_core/fragment_library_v100.py:1`
- `atos_core/discourse_ledger_v100.py:1`
- `atos_core/intent_grammar_v100.py:1`
- `atos_core/conversation_v100.py:1`
- `atos_core/conversation_loop_v100.py:1`
- `scripts/verify_conversation_chain_v100.py:1`
- `scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py:1`
- `patches/v100_fluency_discourse_concepts_dossier_fix1.diff:1`
- `STATUS_PACK_V100_fix1.md:1`

## Comandos (copy/paste)
```bash
cd /Users/danielpenin/act

# Compile
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py

# Regressões (V90–V99)
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py --out_base results/run_smoke_v90_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v91_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v91_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v92_compound_intents_dialogue_csv.py --out_base results/run_smoke_v92_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v93_teachable_intent_grammar_dialogue_csv.py --out_base results/run_smoke_v93_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v94_action_planning_explain_dialogue_csv.py --out_base results/run_smoke_v94_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v95_memory_ledger_dialogue_csv.py --out_base results/run_smoke_v95_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v96_belief_ledger_dialogue_csv.py --out_base results/run_smoke_v96_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v98_operational_truth_agency_dialogue_csv.py --out_base results/run_smoke_v98_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v99_continuous_agency_goal_ledger_dialogue_csv.py --out_base results/run_smoke_v99_regression_v100_fix1 --seed 0
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

# Smoke V100 (try1/try2 dentro do script; WORM)
# Nota WORM: este fix usa out_base *_fix3 porque tentativas *_fix1/*_fix2 já existiam em results/.
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py \
  --out_base results/run_smoke_v100_fluency_discourse_concepts_dossier_fix3 --seed 0 \
  | tee results/smoke_v100_fluency_discourse_concepts_dossier_fix3_try1.json

# (Recomendado) Verificador V100 no run_dir try1
PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_conversation_chain_v100.py --run_dir \
  results/run_smoke_v100_fluency_discourse_concepts_dossier_fix3_try1

# Patch diff (WORM) — somente arquivos V100
(diff -u /dev/null atos_core/conversation_loop_v100.py; \
 diff -u /dev/null atos_core/conversation_v100.py; \
 diff -u /dev/null atos_core/discourse_ledger_v100.py; \
 diff -u /dev/null atos_core/fragment_library_v100.py; \
 diff -u /dev/null atos_core/intent_grammar_v100.py; \
 diff -u /dev/null scripts/smoke_v100_fluency_discourse_concepts_dossier_dialogue_csv.py; \
 diff -u /dev/null scripts/verify_conversation_chain_v100.py) \
 > patches/v100_fluency_discourse_concepts_dossier_fix1.diff
```

## Determinismo (try1 == try2)
- `store_hash`: `b1117e16b3400824573da82ded3159a565d8efa09f66521b44834ab81a6f3ed3`
- `transcript_hash`: `dd50f719d27b294a01fb18f9deeea2f43017ead98fa7d9eddcc63c867f343732`
- `state_chain_hash`: `70f4bf574c4f180d18acf499e0eb102dd1e2598752df3dd53e0094431609d3dd`
- `parse_chain_hash`: `a699665446817fad26a03ff6c089521889d621e44b10eb3c5aba61ab819f8217`
- `learned_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `plan_chain_hash`: `7266821a6ca05d3c8566c009737d6a23b287510d250ca292024d8027f407dbb8`
- `memory_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `belief_chain_hash`: `2922523f03674e45dac92d458261c83ac07baa7236b5e99cb5235474f494430c`
- `evidence_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `goal_chain_hash`: `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`
- `discourse_chain_hash`: `47085e48e0fdaf21b977368b2487acc02152d4e32d5119cfe1a0974efc32d03f`
- `fragment_chain_hash`: `b6d6160288f8a8d76bb8945c0aedd125a6e452e899249c2cabe472063d426086`
- `ledger_hash` (sha256 do `freeze_manifest_v100.json`): `15572050103ff12ecec2dd47fcf7e00d59fcfbc691b35377dab211071536e69c`
- `summary_sha256`: `6533063f69eba52b50de4dd392d0f9b373bb71cd3e09137d50161e37e681f667`

## Negative tamper
- PASS: `discourse_events[0].event_sig = "0"*64` ⇒ `verify_conversation_chain_v100` falha com reason `discourse_event_sig_mismatch`.

## Artefatos (paths)
- Smoke V100 try1: `results/run_smoke_v100_fluency_discourse_concepts_dossier_fix3_try1/`
- Smoke V100 try2: `results/run_smoke_v100_fluency_discourse_concepts_dossier_fix3_try2/`
- Stdout tee: `results/smoke_v100_fluency_discourse_concepts_dossier_fix3_try1.json`
- Patch diff: `patches/v100_fluency_discourse_concepts_dossier_fix1.diff`

