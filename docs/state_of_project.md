# Estado do Projeto ACT (pós‑V62)

## 1) O que já construímos (componentes rodáveis)
- **Núcleo ATOS**: `atos_core/act.py`, `atos_core/store.py`, `atos_core/ledger.py` (WORM/hash‑chain), `atos_core/engine.py` (geração + execução determinística de atos).
- **Ética + incerteza (invariantes fail‑closed)**: `atos_core/ethics.py`, `atos_core/uncertainty.py` integrados em load/add/promotion/execute.
- **Conceitos/CSV + Goals (first‑class)**:
  - `kind="concept_csv"` com execução determinística via `Engine.execute_concept_csv(...)` (trace de instruções `t:"INS"`).
  - `kind="goal"` com execução determinística via `Engine.execute_goal(...)`.
- **Mining + promoção WORM**:
  - Miner determinístico de CSV: `atos_core/csv_miner.py` (n‑grams de primitivas a partir de traces INS).
  - Proof‑Carrying Concept (PCC): `atos_core/proof.py` (`certificate_v1` e `certificate_v2` com `call_deps` + verificação + tamper detection).
- **Gate / custo de scan (eficiência)**:
  - Gate compare+fallback determinístico por token e builders “competitor‑aware” (scripts em `scripts/`).
  - Promoção do gate table como ACT: `kind="gate_table_ctxsig"` + uso opcional via `EngineConfig.use_gate_table_act`.
- **Suites e telemetria**:
  - `atos_core/suite.py` com chat suite + goal shadow (scheduler determinístico) + logs WORM.
  - Scripts de smoke/pipelines (v50…v62) em `scripts/`.

## 2) O que já provamos (com evidência WORM)
- **Determinismo + invariância**: chat suite preserva `sha256(full_text)` com goal shadow ativado (telemetria pura).
- **WORM auditável**: promoção append‑only com ledger hash‑chained e freeze com `verify_chain=true` (ex.: `LEDGER_ATOLANG_*_V62_*_TRY2.json`).
- **Mining real do chat**: traces do goal‑shadow (`goal_shadow_trace.jsonl` com `INS`) → miner → multi‑promoção sob budget → reexecução from‑store com `mismatch_goals==0`.
- **PCC v2 (composição)**: wrapper com `CSV_CALL` verifica `call_deps` e falha sob tamper (`callee_program_sha256_mismatch`).
- **Segurança estrutural**: ética e disciplina IR/IC aplicadas em load/add/promotion/execute (fail‑closed).
- **Eficiência**: gate híbrido (K=2) validado com economia real de scan e divergência 0 em seeds/modes (v56–v58), e promovido para dentro do store (v57).

## 3) Onde estamos no roadmap do atrator (o gargalo atual)
- **Já resolvido**: eficiência/gating e infraestrutura de governança (WORM, prova, hashes, invariância).
- **Em andamento**: transformar experiência real (traces) em **objetos cognitivos reutilizáveis** (CSV) com prova forte e promoção automática.
- **Gargalo atual**: **agência/planejamento** (composição de conceitos sob objetivos), com auditoria e sem mudar o caminho padrão de geração.

## 4) O progresso está no foco certo? (justificativa técnica)
Sim: o projeto já tem (i) ontologia única (atos), (ii) mecanismos de evolução (mineração+promoção), (iii) prova verificável (PCC), e (iv) invariantes (ética/uncertainty). O próximo salto lógico é um loop agentivo determinístico que produza traces mineráveis e force composição/transferência — sem depender de “fluência” como sinal.

## 5) Top 3 gargalos críticos (ordem) + estratégia incremental
1) **Planejamento determinístico sobre conceitos/goals (search + orçamento)**  
   - Estratégia: loop agentivo com plano explícito + execução + validação + traces mineráveis; mining/promoção a partir de agent traces; reexecução from‑store para invariância.
2) **Transferência/composição em escala (ToC, merge/split, lifecycle)**  
   - Estratégia: ampliar PCC (deps transitivas quando necessário), ToC como critério de sobrevivência, budget/eviction por ΔMDL real + reuse + falhas; testes “cross‑task” determinísticos.
3) **Sinal de utilidade robusto (avaliadores determinísticos mais ricos)**  
   - Estratégia: expandir suites determinísticas (estado multi‑turn, formato, aritmética multi‑etapa, memória) e plugar em seleção/promoção (peso inicialmente shadow).

