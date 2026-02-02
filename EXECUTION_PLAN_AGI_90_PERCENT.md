# EXECUTION_PLAN_AGI_90_PERCENT.md

## 📋 Plano para Concluir AGI com >90% ARC

**Data:** 2026-02-02
**Versão Base:** V158
**Testes:** 517 passando

---

## Status dos Módulos

| Módulo | Status | Arquivo |
|--------|--------|---------|
| Full Training Pipeline | ✅ Implementado | `full_training_pipeline_v158.py` |
| Failure-Driven Miner | ✅ Implementado | `failure_driven_miner_v146.py` |
| Cross-Task Miner | ✅ Implementado | `cross_task_miner_v147.py` |
| Solver Concept Gate | ✅ Implementado | `solver_concept_gate_v145.py` |
| Meta-KA | ✅ Implementado | `meta_ka_v156.py` |
| Convergence Loop | ✅ Implementado | `convergence_loop_v157.py` |
| Cognitive Authority | ✅ Implementado | `cognitive_authority_v155.py` |
| Parallel Solver | ✅ Implementado | `arc_parallel_solver_v158.py` |
| Arc Solver V141 | ✅ Implementado | `arc_solver_v141.py` |

---

## Fase 1 — Preparação e Treino Completo

### 1.1 Completar full_training_pipeline_v158.py
- [x] Carregar 400 tarefas ARC-AGI-1
- [x] Integrar mineração de falhas
- [x] Freezing conceitual por geração  
- [x] Log no ledger

### 1.2 Ativar Módulos Avançados
- [x] `failure_driven_miner_v146.py` - Mineração orientada a falhas
- [x] `cross_task_miner_v147.py` - Mineração cruzada entre tarefas
- [x] `solver_concept_gate_v145.py` - Gate de conceito obrigatório
- [x] `meta_ka_v156.py` - Meta conhecimento e ajuste

### 1.3 Loop de Convergência
- [x] `convergence_loop_v157.py` - Script de loop automático

---

## Fase 2 — Redes Neurais Auxiliares

### 2.1 NNs como Ferramentas
- [x] Wrapper com fallback manual (em `cognitive_authority_v155.py`)
- [x] Log de uso de NN (via ledger)
- [x] Prova de opcionalidade implementada

### 2.2 Garantir Independência
- [x] Verificado: execução robusta sem NN
- [x] Validação conceitual de inferências
- [x] Nenhuma decisão automática pela NN

---

## Fase 3 — Loop de Treino até Domínio Total

### 3.1 Execução do Loop
```bash
# Quick test (50 tarefas, 3 iterações)
python -m atos_core.full_training_pipeline_v158 --quick-test

# Full training (400 tarefas, 10 iterações)
python -m atos_core.full_training_pipeline_v158 \
    --tasks 400 \
    --iterations 10 \
    --depth 6 \
    --programs 5000 \
    --workers 8
```

### 3.2 Critério de Parada
- Target: **≥90% ARC-AGI-1**
- Target: **≥90% ARC-AGI-2**
- Conceitos emergindo e sendo reutilizados
- NN comprovadamente opcional

### 3.3 Validação ARC-AGI-2
```bash
python -m scripts.solver_benchmark_v154 \
    --dataset arc2 \
    --depth 6 \
    --programs 5000
```

---

## Fase 4 — Aferição Final e Publicação

### 4.1 Exportar Artefato
- [ ] Snapshot do modelo cognitivo
- [ ] Exportar ledger conceitual
- [ ] Pipeline reproduzível congelado

### 4.2 Documentos Finais
- [ ] `FER_V158_CONVERGENCE_ARCHITECTURE.md`
- [ ] `STATUS_PACK_V160_FINAL_AGI_VERIFIED.md`
- [ ] `FER_FINAL_EVAL_REPORT.md`

### 4.3 Checklist de Publicação
- [ ] Execução CPU-only verificada
- [ ] Desempenho ≥90% nos dois benchmarks
- [ ] Decisões explicadas e versionadas
- [ ] Replay do raciocínio possível

---

## Comandos Úteis

```bash
# Rodar testes
python -m pytest tests/ -v

# Benchmark rápido (10 tarefas)
python -c "from atos_core.quick_benchmark_v154 import quick_benchmark; quick_benchmark(10)"

# Verificar autoridade hierárquica
python -c "from atos_core.cognitive_authority_v155 import verify_authority_hierarchy; print(verify_authority_hierarchy())"

# Provar opcionalidade NN
python -c "
from atos_core.convergence_loop_v157 import ConvergenceLoop
loop = ConvergenceLoop()
print(loop.prove_nn_optionality())
"
```

---

## Métricas Atuais

| Métrica | Valor |
|---------|-------|
| Baseline | ~11% |
| Testes | 517 |
| Operadores | 69 |
| Conceitos | Em emergência |
| Meta Target | ≥90% |

---

## Próximo Passo Imediato

1. Rodar `full_training_pipeline_v158.py` com 100 tarefas
2. Observar taxa de conceitos emergindo
3. Ajustar `solver_depth` e `programs` conforme necessário
4. Iterar até platô ou target

---

**Veredito Atual:** `AGI_NÃO_CONFIRMADA` (baseline 11%)

**Caminho para AGI:** Conceitos devem emergir, profundidade deve crescer, reutilização deve aumentar. Loop contínuo até atingir 90%.
