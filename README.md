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
- aplicado no runtime em `atos_core/engine.py` (evicção determinística; em `--mode pure` os n-grams usam `evict_policy="count_lex"` por padrão)

## Fluência (seleção de patch)
Seleção estrutural usa um score explícito:
`score = gain_bits - λ*(repeat3 + loop_rate + whitespace_ratio + (1-unique_reply_rate) + most_common_reply_frac)`
com `λ = TrainConfig.fluency_lambda`, logado em `report.json`.

Parâmetros úteis:
- `--fluency_lambda_schedule linear_warmup` + `--fluency_warmup_frac` (rampa de λ ao longo do treino; evita deadlocks no começo).
- `--gain_horizon_steps` (horizonte usado para estimar ganho MDL de um patch; útil se você vai treinar em “segmentos” mas quer que a seleção pense no plano longo).
- `--holdout_frac` + `--holdout_eval_windows` (pressão de generalização: NLL estático em um slice determinístico fora do treino).
- `--selection_mode survival` (seleção por gargalo existencial: otimiza o pior termo do `SYSTEM_SURVIVAL`, incluindo holdout/utility/fluência).
- `--skill_suite_pack sota_v4` (pack determinístico maior; utility vira gargalo AND por categoria no `SYSTEM_SURVIVAL`, incluindo `clarify`/`consistency`, **memória longa** (store+recall) e **composição de conceitos**: tarefas `plan_validator` exigem `concept_calls_max_depth>=1`).
- `--skill_suite_pack sota_v5` (como `sota_v4`, mas exige **hierarquia mais profunda**: tarefas `plan_validator` exigem `concept_calls_max_depth>=2`).
- `--agency_suite_enabled` (suite de agência: planner+executor determinísticos; vira gargalo `utility:agency`).
- `--concept_csv_mining_enabled` (INDUCE_CONCEPT): minera/promove atos `concept_csv` (conceitos vivos invocáveis) via traços determinísticos; controle com `--concept_csv_budget`, `--concept_csv_mining_top_k`, `--concept_csv_mining_max_new_per_window`.
- `--concept_csv_composed_enabled` (INDUCE_CONCEPT v74): minera/promove conceitos **compostos** (`CSV_CALL`) a partir de traços multi‑passo.
- `--survival_concept_no_add_windows` + `--survival_concept_reuse_stall_windows` + `--survival_concept_hard_fail_windows` (lei existencial: em plateau + reuso saturado, se não criar novos conceitos `concept_csv`, o treino falha determinísticamente).
- `--survival_concept_composed_no_add_windows` + `--survival_concept_composed_rate_stall_windows` + `--survival_concept_composed_hard_fail_windows` (lei existencial: se o pack exigir composição e o sistema não induzir conceitos compostos, falha determinísticamente).
- `--survival_concept_deep_rate_stall_windows` + `--survival_concept_deep_hard_fail_windows` (lei existencial: se o pack exigir profundidade>=2 e não emergir uso de conceitos profundos, falha determinísticamente).

Treino (gera `acts.jsonl`, `ledger.jsonl`, `report.json`):
```bash
python3 scripts/train.py --data data/sample_text.txt --steps 200000 --seed 0 --out results/run1 --mode demo
python3 scripts/train.py --data data/sample_text.txt --steps 200000 --seed 0 --out results/run_pure_200k --mode pure
```

Avaliação (gera `transcripts.jsonl` e imprime um resumo):
```bash
python3 scripts/eval.py --run results/run1
```

## Treino com Hugging Face (dataset fechado)

Gerar um corpus `.txt` (streaming + determinístico) a partir de um dataset/model no Hub:
```bash
python3 scripts/prepare_hf_corpus.py \
  --source https://huggingface.co/thegoodfellas/tgf-flan-t5-base-ptbr \
  --out data/raw/hf_corpus/tgf_flan_ptbr__train.txt \
  --target_bytes 20000000 \
  --seed 0
```

Treino end-to-end (prepara o corpus e treina no mesmo comando):
```bash
python3 scripts/train_hf_ka.py \
  --source https://huggingface.co/thegoodfellas/tgf-flan-t5-base-ptbr \
  --out artifacts/run_tgf_flan_ptbr_pure_200k \
  --steps 200000 \
  --gain_horizon_steps 200000 \
  --window 10000 \
  --mode pure \
  --selection_mode survival \
  --holdout_frac 0.05 --holdout_eval_windows 3 \
  --concept_csv_mining_enabled \
  --concept_csv_composed_enabled \
  --agency_suite_enabled \
  --utility_weight 1.0 --skill_suite_pack sota_v4 \
  --target_bytes 20000000 \
  --seed 0
```

Nota: `thegoodfellas/tgf-flan-t5-base-ptbr` é um **model repo**; o script resolve automaticamente o dataset via `datasets:` do Model Card (ex.: `thegoodfellas/mc4-pt-cleaned`).

Sweep de passos (500 → 1k → 2k → 5k), útil para inspecionar snapshots/metrics por escala:
```bash
python3 scripts/train_hf_sweep.py \
  --source https://huggingface.co/thegoodfellas/tgf-flan-t5-base-ptbr \
  --out_root artifacts/hf_sweep_tgf \
  --steps_list 500,1000,2000,5000 \
  --gain_horizon_steps 200000 \
  --mode pure \
  --target_bytes 20000000 \
  --seed 0
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

## Demos / entrega
- Checklist e comandos: `docs/DELIVERY_CHECKLIST.md`
- Linguagem interna (concept_csv): `docs/INTERNAL_LANGUAGE.md`
- Reprodutibilidade/freeze: `docs/REPRODUCIBILITY.md`

Scripts:
```bash
python3 scripts/demo_sota_pack.py --run <RUN_DIR> --pack sota_v8 --show_transcripts 3
python3 scripts/inspect_concept_csv.py --run <RUN_DIR> --concept_id <CONCEPT_ID>
python3 scripts/freeze_run.py --run <RUN_DIR>
```

## ARC (ACT vs ARC pt2)

Ferramentas principais (CPU-only, determinístico, WORM):
- Runner: `scripts/run_arc_scalpel_v141.py`
- Loop pt2 (pressão → mineração → bancos → score): `scripts/arc_loop_pt2.sh`
- Loop iterativo/infinito (WORM): `scripts/arc_loop_pt2_iter_v146.py`

Exemplos:
```bash
# Loop infinito (iters=0) até bater solve_rate_kmax ≥ 0.70 no subset (para throughput: tries=1 + timeout)
python3 scripts/arc_loop_pt2_iter_v146.py \
  --limit 100 --iters 0 --jobs 8 --pressure 1 --tries 1 --task_timeout_s 90 \
  --max_depth 4 --max_programs 2000 --stop_rate_kmax 0.70 \
  --concept_bank_in <concept_bank.jsonl> --macro_bank_in <macro_bank.jsonl>

# Run “headline” (tries=2, sem timeout) em ARC-AGI-1 training/evaluation (400/400) com bancos fixos
python3 scripts/run_arc_scalpel_v141.py \
  --arc_root ARC-AGI --split training --limit 0 --seed 0 --tries 2 --jobs 8 \
  --max_depth 4 --max_programs 2000 \
  --concept_templates <concept_bank.jsonl> \
  --macro_templates <macro_bank.jsonl> \
  --out_base results/arc_agi1_training_full_$(date +%Y%m%d_%H%M%S)

python3 scripts/run_arc_scalpel_v141.py \
  --arc_root ARC-AGI --split evaluation --limit 0 --seed 0 --tries 2 --jobs 8 \
  --max_depth 4 --max_programs 2000 \
  --concept_templates <concept_bank.jsonl> \
  --macro_templates <macro_bank.jsonl> \
  --out_base results/arc_agi1_evaluation_full_$(date +%Y%m%d_%H%M%S)
```
