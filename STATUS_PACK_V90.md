# STATUS PACK V90 — V90_CONVERSATION_STATE_DIALOGUE_CSV — GO

## Resumo (o que foi implementado)
- **ConversationStateCSV (WORM)**: estado conversacional imutável por turno (cadeia `prev_state_id`), com `state_sig` e verificador determinístico.
- **TurnCSV (WORM)**: cada mensagem vira um `turn_v90` com `text_sig` e referências a objetivo/ação.
- **Diálogo como ação (CSV)**: ações linguísticas são `concept_csv` executadas pelo mesmo runtime (`EngineV80`) e avaliadas por **Objective CSV**.
- **Objetivos comunicativos** (CSV): `COMM_*` implementados como Objective CSV por igualdade determinística de `__output` vs `expected`.
- **Anti-alucinação estrutural (harness determinístico)**: se falta dado (ex.: `GET z` sem `z`) → `COMM_ASK_CLARIFY` (pergunta), sem inventar.
- **Logs WORM hash-chained** por run (`*.jsonl` com `prev_hash`/`entry_hash`) + `verify_chain_v90.json` (cadeia + invariantes).

## Checklist (PASS/FAIL)
- PASS: `py_compile` (atos_core + scripts + tests)
- PASS: Smoke V90 (try1/try2) determinístico (`store_hash`, `transcript_hash`, `state_chain_hash`, `ledger_hash` e `summary_sha256` idênticos)
- PASS: `verify_conversation_chain_v90` ok (try1/try2)
- PASS: `verify_chained_jsonl_v90` ok para turns/states/trials/evals/transcript
- PASS: Negative test (corrupção em memória → verificador falha com reason determinístico)
- PASS: Regressões `unittest` (12 testes)
- PASS: `verify_freeze` ok=true, `warnings=[]`

## Arquivos criados/modificados
**Novos (código):**
- `atos_core/conversation_v90.py`
- `atos_core/conversation_objectives_v90.py`
- `atos_core/conversation_actions_v90.py`
- `atos_core/conversation_loop_v90.py`
- `scripts/smoke_v90_conversation_dialogue_csv.py`

**Novos (governança/WORM):**
- `patches/v90_conversation_state_dialogue_csv_fix1.diff`
- `LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json`
- `results/verify_freeze_v90_try1.json`
- `STATUS_PACK_V90.md`

## Comandos executados (copy/paste)
```bash
cd /Users/danielpenin/act

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py tests/*.py

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v90_conversation_dialogue_csv.py \
  --out_base results/run_smoke_v90_conversation_dialogue_csv_fix2 \
  --seed 0 | tee results/smoke_v90_conversation_dialogue_csv_fix2_try1.json

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m unittest -v

(diff -u /dev/null atos_core/conversation_actions_v90.py; \
 diff -u /dev/null atos_core/conversation_loop_v90.py; \
 diff -u /dev/null atos_core/conversation_objectives_v90.py; \
 diff -u /dev/null atos_core/conversation_v90.py; \
 diff -u /dev/null scripts/smoke_v90_conversation_dialogue_csv.py) \
 > patches/v90_conversation_state_dialogue_csv_fix1.diff

PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/verify_freeze.py --freeze \
  LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json \
  | tee results/verify_freeze_v90_try1.json
```

## Prova de determinismo (try1 == try2)
**Smoke (seed=0)**: `results/smoke_v90_conversation_dialogue_csv_fix2_try1.json`
- `store_hash`: `db91c825e7db04beadf7aaf59903033b47738ac235fdb048a5df8b27c7cc4294`
- `transcript_hash`: `0c6ddff5ff8e40202e27bfd14e86460b7940dbcc8e76a571729bfece28e82f57`
- `state_chain_hash`: `4e64ff0119188fd4166c6c49ca96fd31649fecfa61d38d6d8322b0f52f42b76f`
- `ledger_hash` (sha256 do `freeze_manifest_v90.json`): `34792cf1c6d0f0f2220a712c2ba0a1fa513b774b7e6bcd20341b170a52eaaf5b`
- `summary_sha256`: `5ada9ce43591788f623d91c8b1840e7902b762b31939a5ea0eb4d49c1091786a`

**Runs:**
- try1: `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/`
- try2: `results/run_smoke_v90_conversation_dialogue_csv_fix2_try2/`

## Negative test (verificador falha closed)
- Corrupção em memória: `state_index=999` no estado genesis → `verify_conversation_chain_v90` retorna `ok=false` com `reason="state_index_not_incrementing"`.

## Artefatos (paths + SHA256 principais)
- Patch: `patches/v90_conversation_state_dialogue_csv_fix1.diff`
  - `ba61ef9203c11091a2c5d436a0649f06f35123dc1f5992bfc1e235ff8f3d1037`
- Ledger: `LEDGER_ATOLANG_V0_2_30_BASELINE_V90_CONVERSATION_STATE_DIALOGUE_CSV.json`
  - `5e4e9b3047065bf24e4165def185e73fb5738155bb6ece7cbbd13c3193424eb5`
- Verify freeze: `results/verify_freeze_v90_try1.json`
  - `97de8c579e4e7809cdef6c87e30bd946ad3f447e7b9e34670a74328feaa16aa4`
- Smoke stdout: `results/smoke_v90_conversation_dialogue_csv_fix2_try1.json`
  - `ccd8880a1c311926ae4d0b2a13d591f3b5efc81f176de416a428cf31ac044061`

**Try1 core (todos determinísticos e idênticos no try2):**
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/store.jsonl`
  - `25a3ca8b413bef8f734baa3efe8584238b0993adc9b2539d6bb9fa65b3277d14`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/conversation_turns.jsonl`
  - `32125d411d3c95bf6b1675acab91dba3619909254200849238eea00bd6855e88`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/conversation_states.jsonl`
  - `af3fb047debf6f2de207f5a07ead511e2bae6f91733cce41c6b69c5ed9dbdbbd`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/dialogue_trials.jsonl`
  - `1a0bf11f0abda2127eb1a36766a772ae775ee9071e23ef72e0df45a36307c394`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/objective_evals.jsonl`
  - `cbea0ebf91247ba86314d86a49931662739a05c29ffd5aa5ad624ff7d8f1f8f7`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/transcript.jsonl`
  - `56c291faed5230c147ec961ab1e342ac8bd794b6759e6d7362a24275d950ea69`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/verify_chain_v90.json`
  - `e64688a659f86c8340f17233598ba3be0f88a5ed41ac30e0bc26e9a78e290ac6`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/freeze_manifest_v90.json`
  - `34792cf1c6d0f0f2220a712c2ba0a1fa513b774b7e6bcd20341b170a52eaaf5b`
- `results/run_smoke_v90_conversation_dialogue_csv_fix2_try1/summary.json`
  - `3e5c7a8fb5b5a4ca680f3171b403b728033291e0a065d3f6e2ca049cd382ca20`

## Limitações atuais (intencionais) + caminho claro p/ V91
- **Domínio do diálogo** nesta V90 é um harness determinístico (DSL `SET/GET/ADD/SUMMARY/END`) para provar:
  estado explícito + escolha de objetivo + fala como ação + autoavaliação + anti-alucinação.
- **Objetivos comunicativos** estão implementados por igualdade exata (`__output == expected`) para manter verificabilidade forte.
- Próximo passo (V91): generalizar o parser determinístico/gramática de intents e expandir o conjunto de objetivos/açōes mantendo:
  1) objetivos como Objective CSV, 2) ações como concept_csv, 3) provas WORM + chains, 4) fallback fail-closed.

## Decisão final
**GO** — Smoke determinístico, cadeia verificada (ok), negative test fail-closed, regressões ok, ledger + verify_freeze limpos.

