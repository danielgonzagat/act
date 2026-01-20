# ActCore / AtoLang v0.1 (discreto, explícito, sem pesos)

Este repositório implementa um protótipo mínimo (v0.1) de um “sistema cognitivo” cujo substrato é **somente uma base de ATOS (JSON)**.  
Não há rede neural, não há pesos treináveis, não há gradiente.

## Invariantes
- **Aprender = editar atos** (ADD / MERGE / PRUNE / REWRITE). Nada de SGD/Adam/embeddings treináveis.
- **Determinístico** (seed fixa) e **CPU-only**.
- `created_at` e `ledger.time` são derivados do `step` (não wall-clock) para reprodutibilidade.
- **Contagens (evidência) são aprendizado permitido** e vivem em `act.evidence.table` (source of truth).
- **Runtime pode ter cache derivado**, mas nada que influencie previsão fora dos atos.
- **Tudo que melhora desempenho vira estado explícito** nos próprios atos (`acts.jsonl`) + log WORM (`ledger.jsonl`).
- **Guardrails de fluência**: penalidade explícita contra repetição e ciclos de n-grams.

## O que é um ATO
Um ato é um objeto JSON com: `id, version, created_at, kind, match, program, evidence, cost, deps`.  
O “bytecode” (AtoLang v0.1) é uma lista de instruções serializáveis (ex.: `MATCH_NGRAM`, `EMIT_CANDIDATES_TOPK`, `APPLY_PENALTY`, ...).

## Como rodar

Modos:
- `--mode demo`: permite **prefill** de tabelas ao criar novos atos (útil para demo de compressão).
- `--mode pure`: **proíbe prefill**; atos novos começam com `evidence.table = {}` e só acumulam contagens via `observe()` (evidência KA).

## Budget (PURE)
Atos com `allow_new_contexts/allow_new_tokens` usam budget explícito no próprio ato:
- `evidence.max_contexts`, `evidence.max_next_per_ctx`, `evidence.evict_policy="fifo"`
- aplicado no runtime em `atos_core/engine.py:157` (evicção determinística)

## Fluência (seleção de patch)
Seleção estrutural usa um score explícito:
`score = gain_bits - λ*(repeat3 + loop_rate + whitespace_ratio + (1-unique_reply_rate) + most_common_reply_frac)`
com `λ = TrainConfig.fluency_lambda`, logado em `report.json`.

Treino (gera `acts.jsonl`, `ledger.jsonl`, `report.json`):
```bash
python3 scripts/train.py --data data/sample_text.txt --steps 200000 --seed 0 --out results/run1 --mode demo
python3 scripts/train.py --data data/sample_text.txt --steps 200000 --seed 0 --out results/run_pure_200k --mode pure
```

Avaliação (gera `transcripts.jsonl` e imprime um resumo):
```bash
python3 scripts/eval.py --run results/run1
```

Geração:
```bash
python3 scripts/generate.py --run results/run1 --prompt "Oi, quem é você?" --max_new_tokens 200
```

## Artefatos (por run)
- `results/<run>/acts.jsonl`: base completa de atos (o “modelo”).
- `results/<run>/ledger.jsonl`: ledger WORM encadeado por hash (patches estruturais + hash do snapshot por janela).
- `results/<run>/report.json`: métricas por janela (a cada `--window`).
- `results/<run>/transcripts.jsonl`: 20 prompts multi-turn com respostas geradas.
- `results/<run>/snapshots/stepXXXXXX_acts.jsonl`: snapshot determinístico do estado completo por janela.
# act
# act
# act
# act
