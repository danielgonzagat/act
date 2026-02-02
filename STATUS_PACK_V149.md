# STATUS_PACK_V149 - Lei de Diversidade Conceitual

## REGIME: PÓS-EMERGÊNCIA CONFIRMADA (V147→V148→V149)

V147 confirmou emergência. V148 criou infraestrutura. V149 garante saúde do ecossistema.

---

## PROBLEMA IDENTIFICADO

Com emergência confirmada (1 conceito sobreviveu), surge novo risco:

**E se esse conceito dominar tudo?**

- AGI ≠ um conceito bom
- AGI = ecossistema conceitual **diverso e competitivo**

Sem diversidade, o sistema:
1. Overfita em um tipo de padrão
2. Falha em tasks que requerem outros conceitos
3. Colapsa se o conceito dominante for removido

---

## SOLUÇÃO: diversity_law_v149.py (~550 linhas)

### Componentes Principais

#### 1. ConceptEcosystem - Rastreamento Cross-Task
```python
ecosystem = ConceptEcosystem()
ecosystem.total_tasks = 100

# Registra uso de conceitos
ecosystem.record_usage("task_1", "concept_a", depth=1)
ecosystem.record_usage("task_2", "concept_a", depth=0)
ecosystem.record_usage("task_3", "concept_b", depth=2)

# Métricas disponíveis
print(ecosystem.concepts["concept_a"].usage_rate(100))  # % tasks usando
print(ecosystem.concepts["concept_a"].avg_depth())  # profundidade média
print(ecosystem.get_diversity_score())  # 0-1, maior = mais saudável
```

#### 2. Dominance Detection - Max 40% Usage
```python
config = DiversityConfig()
config.max_concept_usage_pct = 40.0  # Nenhum conceito pode ter >40% usage

dominant = ecosystem.get_dominant_concepts(config)
# ["concept_a"] se concept_a for usado em >40% das tasks
```

#### 3. Repeated Call Penalty
```python
config.max_repeated_calls = 3  # Máximo 3 chamadas do mesmo conceito por solução
config.repeated_call_penalty = 0.1  # 10% penalidade por chamada extra

# Solução ["c1", "c1", "c1", "c1", "c1"] → VIOLATION (excede limit)
```

#### 4. Lineage Competition
```python
# Conceitos são agrupados em "linhagens" (famílias)
ecosystem.assign_lineage("color_swap", "lineage_color")
ecosystem.assign_lineage("color_fill", "lineage_color")
ecosystem.assign_lineage("rotate_90", "lineage_geometry")

# Tournament seleciona os melhores de cada linhagem
results = run_lineage_tournament(ecosystem, config)
# Conceitos fracos são marcados para eliminação
```

#### 5. Similarity Detection
```python
# Conceitos muito similares (>90%) são detectados
pairs = find_similar_concepts(concepts, threshold=0.9)
# [("concept_a", "concept_b", 0.95)] → muito similares, considerar merge
```

### API Principal

```python
from atos_core.diversity_law_v149 import (
    ConceptEcosystem,
    DiversityConfig,
    enforce_diversity_law,
    should_trigger_alternative_mining,
)

# Verificar se solução satisfaz lei de diversidade
passed, audit = enforce_diversity_law(
    solution_concepts=["concept_a", "concept_b"],
    ecosystem=ecosystem,
)

# Verificar se mining alternativo é necessário
request = should_trigger_alternative_mining(ecosystem, config)
if request:
    print(f"Mining necessário: {request.trigger_reason}")
    # CONCEPT_DOMINANCE, INSUFFICIENT_LINEAGES, ou LOW_DIVERSITY_SCORE
```

---

## CONFIGURAÇÃO DEFAULT

| Parâmetro | Valor | Significado |
|-----------|-------|-------------|
| `max_concept_usage_pct` | 40% | Nenhum conceito pode dominar >40% das tasks |
| `max_repeated_calls` | 3 | Máx 3 chamadas do mesmo conceito/solução |
| `repeated_call_penalty` | 0.1 | 10% penalidade por chamada extra |
| `min_concept_families` | 3 | Mínimo 3 famílias conceituais |
| `similarity_threshold` | 0.9 | Conceitos >90% similares são candidatos a merge |
| `min_diversity_score` | 0.5 | Score mínimo para passar no gate |
| `min_lineages` | 2 | Mínimo 2 linhagens competindo |
| `lineage_tournament_size` | 3 | Top 3 de cada linhagem sobrevivem |

---

## TESTES

39 novos testes cobrindo:
- ConceptUsageStats (usage rate, depth tracking)
- ConceptEcosystem (recording, lineages, serialization)
- ConceptSimilarity (signatures, Jaccard similarity)
- DiversityCheck (violations, warnings, scoring)
- AlternativeMining (trigger detection)
- LineageTournament (competition mechanism)
- EnforceDiversityLaw (main entry point)

---

## MÉTRICAS V149

| Métrica | Valor |
|---------|-------|
| Tests Passando | 417 (+39 sobre V148) |
| Operadores | 82 |
| Conceitos Sobreviventes (V147) | 1 |
| Max Usage % Permitido | 40% |
| Min Lineages Requeridas | 2 |
| Penalidade Repetição | 10%/call |

---

## ARQUIVOS V149

| Arquivo | Linhas | Propósito |
|---------|--------|-----------|
| `atos_core/diversity_law_v149.py` | ~550 | Lei de Diversidade Conceitual |
| `tests/test_diversity_law_v149.py` | ~350 | 39 testes para a lei |
| `STATUS_PACK_V149.md` | Este | Documentação V149 |

---

## PRÓXIMOS PASSOS

1. **Integrar diversity_law com inevitability_gate** - Enforcement automático no gate
2. **Escalar ARC-AGI-1 Training→Eval** - Pipeline completo com 16 cores
3. **Testes de Robustez Anti-Regressão** - Sistema DEVE falhar sem conceitos
4. **Transferência Fora do ARC** - Provar generalização em outro domínio

---

## CONEXÃO COM LEIS COGNITIVAS

A Lei de Diversidade complementa as 7 leis de sobrevivência do gate:

| Lei Original | Complemento da Diversidade |
|--------------|---------------------------|
| depth > 0 | Profundidade varia por linhagem |
| reuse ≥ 2 | Reuse não pode concentrar em 1 conceito |
| collapse_factor | Diversidade impede monopolização do colapso |
| world_hits | Diferentes conceitos resolvem diferentes tasks |
| world_pressure | Pressão distribuída entre linhagens |

**A diversidade é a 8ª lei implícita: a emergência precisa ser múltipla.**
