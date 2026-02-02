# STATUS PACK V144: AGI REGIME

**Data**: 2025-01-26  
**Esquema**: V0.2.30 → BASELINE_V144  
**Tipo**: Mandatory Concept Creation via Survival Laws  

---

## 🎯 OBJETIVO ALCANÇADO

**Implementar o "AGI Regime" que torna impossível resolver tarefas sem conceitos/planejamento.**

### O Problema Identificado

**Citação da Auditoria AGI**:
> "Enquanto o sistema pode continuar funcionando sem criar conceitos, AGI não acontece. O sistema ainda permite sobrevivência sem isso. O sistema precisa de pressão existencial real: tornar impossível resolver tarefas sem conceitos/planejamento."

### A Solução

**7 LEIS DE SOBREVIVÊNCIA** (Survival Laws) que transformam conceitos de opcionais em obrigatórios:

1. **LAW_CONCEPT**: Não há fallback global. Se `concept_policy_required=True` e nenhum conceito executado → FAIL.
2. **LAW_DEPTH**: Conceitos rasos (depth < min) não satisfazem → FAIL.
3. **LAW_COMPOSITION**: Requer cadeias CSV_CALL, não só primitivos.
4. **LAW_REUSE**: Conceitos devem ser reutilizados em múltiplos contextos.
5. **LAW_PROOF**: PCC + hashes obrigatórios para promoção.
6. **LAW_UTILITY**: Utility é o bottleneck do loss.
7. **LAW_BUDGET**: Search explode sem conceitos → FAIL.

---

## 📦 COMPONENTES

### 1. `agi_regime_v144.py` (~925 linhas)

**Implementação das 7 Leis de Sobrevivência**

```python
@dataclass
class AGIRegimeConfig:
    """Configuração do regime AGI."""
    enable_law_concept: bool = True
    enable_law_depth: bool = True
    enable_law_composition: bool = True
    enable_law_reuse: bool = True
    enable_law_proof: bool = True
    enable_law_utility: bool = True
    enable_law_budget: bool = True
    
    default_min_depth: int = 1
    default_min_csv_calls: int = 1
    default_search_budget: int = 1000
```

**Lifecycle de Conceitos**:
```
CANDIDATE → PROMOTED → QUARANTINED → PRUNED
```

**Funções principais**:
- `validate_law_concept()`: Força execução de conceito quando `concept_policy_required=True`
- `validate_law_depth()`: Verifica profundidade mínima do conceito
- `validate_law_composition()`: Valida número de CSV_CALLs
- `validate_law_budget()`: Limita steps de busca
- `validate_survival_laws()`: Entry point para validação completa
- `compute_regime_loss()`: Loss infinito se qualquer lei falhar
- `create_agi_regime_tasks()`: **O SWITCH** que liga o regime

**ConceptLifecycleManager**:
- Registra conceitos candidatos
- Rastreia uso em múltiplos contextos
- Promove conceitos com reuso cross-context
- Quarentena conceitos com falhas consecutivas
- Poda conceitos sem utilidade

### 2. `agi_loop_v144.py` (~783 linhas)

**Closed-Loop AGI Engine**

```python
class LoopPhase(str, Enum):
    PLAN = "plan"
    EXECUTE = "execute"
    VALIDATE = "validate"
    REPAIR = "repair"
    MINE = "mine"
    PROMOTE = "promote"
    RERUN = "rerun"
    COMPLETE = "complete"
    FAILED = "failed"
```

**Fluxo do Loop**:
```
1. PLAN    → Gera plano de execução (deve incluir conceitos)
2. EXECUTE → Executa task com concept_executor
3. VALIDATE → Valida contra Survival Laws
4. REPAIR   → Tenta corrigir falhas (força criação de conceitos)
5. MINE     → Extrai padrões compostos de CSV_CALL
6. PROMOTE  → Promove conceitos com utilidade provada
7. RERUN    → Re-executa com novos conceitos
```

**Características**:
- Determinístico: Same task → same result
- Fail-closed: Se leis falharem e repair impossível → FAIL
- Mining integrado no loop (não post-process)
- Transfer learning em batches via conceitos compartilhados

### 3. `csv_composed_miner_v144.py` (~583 linhas)

**Mineração de Conceitos Hierárquicos**

```python
@dataclass
class CallSubgraph:
    """Subgrafo de CSV_CALL."""
    nodes: Tuple[CallNode, ...]
    edges: Tuple[CallEdge, ...]
    root_concept_id: str
    max_depth: int
    trace_id: str
    context_id: str
    family_id: str
```

**Pipeline de Mining**:
1. `extract_call_subgraphs()`: Extrai subgrafos de eventos CSV_CALL
2. `mine_composed_concepts()`: Identifica padrões frequentes
3. `materialize_composed_concept_act()`: Cria Acts concept_csv
4. `run_composed_mining_pipeline()`: Pipeline completo

**Critérios de Promoção**:
- Frequência mínima em traces
- Uso em múltiplos contextos
- Uso em múltiplas famílias de tasks
- Score de utilidade mínimo

### 4. `test_agi_regime_v144.py` (~600 linhas)

**Suite de Testes Completa**

- ✅ 26 testes, todos passando
- Cobertura de todas as 7 leis
- Testes de lifecycle de conceitos
- Testes de integration AGI loop
- Testes WORM-compliant ledger

---

## 🔬 VALIDAÇÃO EXPERIMENTAL

### Testes de Regressão

```bash
$ python -m unittest discover -s tests -p 'test_*.py'
Ran 268 tests in 2.537s

OK
```

**Todos os testes passaram**: 
- 242 testes antigos continuam funcionando
- 26 novos testes V144 validam as 7 leis

### Teste Manual das Leis

**LAW_CONCEPT - Obrigatoriedade de Conceitos**:
```python
# FAIL: task requer conceito mas não executou
trace = {"concept_executor": {"used": False}}
task = {"concept_policy_required": True}
result = validate_law_concept(trace=trace, task=task)
# result.passed = False ✅
```

**LAW_DEPTH - Profundidade Mínima**:
```python
# FAIL: conceito raso demais
trace = {"concept_executor": {"max_depth": 1}}
task = {"concept_min_depth": 2}
result = validate_law_depth(trace=trace, task=task, min_depth=2)
# result.passed = False ✅
```

**LAW_COMPOSITION - CSV_CALL Chains**:
```python
# FAIL: sem composição
trace = {"concept_executor": {"calls_total": 0}}
task = {"concept_min_csv_calls": 2}
result = validate_law_composition(trace=trace, task=task)
# result.passed = False ✅
```

**LAW_BUDGET - Search Collapse**:
```python
# FAIL: search explodiu
trace = {"search_steps": 150}
task = {"search_budget": 100}
result = validate_law_budget(trace=trace, task=task)
# result.passed = False ✅
```

---

## 🏗️ ARQUITETURA

### Hierarquia de Decisão

```
┌────────────────────────────────────────┐
│     create_agi_regime_tasks()          │  ← O SWITCH
│  (regime_level: bootstrap|full)        │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│     apply_regime_to_tasks()            │
│  - Adiciona concept_policy_required    │
│  - Seta concept_min_depth              │
│  - Seta concept_min_csv_calls          │
│  - Seta search_budget                  │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│          run_agi_loop()                │
│  Plan → Execute → Validate → ...       │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│    validate_survival_laws()            │
│  - Valida todas as 7 leis              │
│  - Retorna RegimeValidationResult      │
└────────────────┬───────────────────────┘
                 │
                 ▼
┌────────────────────────────────────────┐
│      compute_regime_loss()             │
│  - Se qualquer lei falhou → loss=∞     │
│  - Senão → utility bottleneck loss     │
└────────────────────────────────────────┘
```

### Fluxo de Dados

```
Task (sem regime) 
    ↓
create_agi_regime_tasks(regime_level="full")
    ↓
Task com {concept_policy_required=True, concept_min_depth=2, ...}
    ↓
run_agi_loop()
    ↓
ConceptLifecycleManager rastreia todos os conceitos
    ↓
validate_survival_laws() verifica todas as leis
    ↓
Se FAIL → phase_repair() tenta corrigir
    ↓
Se repair impossível → exit_reason=REGIME_VIOLATION
    ↓
Loss infinito → pressure para criar conceitos
```

---

## 🔐 GARANTIAS WORM

### Ledger Entries

**Regime Validation**:
```json
{
  "schema_version": 144,
  "kind": "regime_validation_ledger_entry_v144",
  "task_id": "...",
  "step": 100,
  "timestamp": "2025-01-26T...",
  "result": {
    "passed": false,
    "laws_checked": 7,
    "laws_passed": 5,
    "laws_failed": 2,
    "failure_reasons": [
      "law_concept: concept_not_executed",
      "law_depth: insufficient_depth"
    ]
  },
  "prev_hash": "...",
  "entry_hash": "..."
}
```

**Loop Result**:
```json
{
  "schema_version": 144,
  "kind": "agi_loop_ledger_entry_v144",
  "timestamp": "...",
  "step": 200,
  "result": {
    "success": false,
    "exit_reason": "regime_violation",
    "iterations_completed": 5,
    "concepts_mined": 3,
    "concepts_promoted": 0,
    "final_state": {...}
  },
  "entry_hash": "..."
}
```

---

## 📊 MÉTRICAS DE IMPACTO

### Antes do V144 (V143)

- Tasks podiam passar sem conceitos
- Concepts eram opcionais
- Fallback global sempre disponível
- Mining como post-process

### Depois do V144

- **Tasks utility DEVEM usar conceitos** (LAW_CONCEPT)
- **Conceitos profundos obrigatórios** (LAW_DEPTH ≥ 2)
- **Composição obrigatória** (LAW_COMPOSITION)
- **Mining integrado no loop**
- **Loss infinito se leis falharem**

### Expectativa

**"Se o sistema não planificar, falhará"** ← Agora é verdade!

- Tasks sem conceitos → FAIL
- Search sem conceitos → FAIL (budget explode)
- Conceitos rasos → FAIL
- Sem composição → FAIL

**Pressure existencial real** que força emergência de AGI.

---

## 🚀 ROADMAP PÓS-V144

### Próximos Passos

1. **V145**: Integrar com `parallel_solver_v143.py`
   - Aplicar regime em ARC tasks
   - Validar pressure em problemas reais

2. **V146**: World Pressure Integration
   - Combinar Survival Laws + World Pressure
   - Constraint propagation via conceitos

3. **V147**: Meta-Learning Loop
   - Conceitos que criam conceitos
   - Self-modification via mining

4. **V148**: Full AGI Demo
   - End-to-end no ARC Evaluation Set
   - Proof of concept emergence

---

## 📝 CHANGELOG

### Adicionado

- ✅ `agi_regime_v144.py`: 7 Survival Laws
- ✅ `agi_loop_v144.py`: Closed-loop engine
- ✅ `csv_composed_miner_v144.py`: Hierarchical concept mining
- ✅ `test_agi_regime_v144.py`: 26 tests completos
- ✅ ConceptLifecycleManager (ICS)
- ✅ `create_agi_regime_tasks()` - O SWITCH

### Modificado

- ✅ SCHEMA_VERSION: 143 → 144

### Corrigido

- ✅ Bug em `register_concept()` call signature
- ✅ Bug em `phase_validate()` com API incorreta
- ✅ Bug em `phase_repair()` acessando atributos inexistentes
- ✅ Bug em `ComposedMiningResult.to_dict()` com `int(list)`

---

## 🧪 REPRODUZIBILIDADE

### Setup

```bash
cd /workspaces/act
```

### Rodar Testes V144

```bash
python -m unittest tests.test_agi_regime_v144 -v
# 26 tests, todos passam
```

### Rodar Suite Completa

```bash
python -m unittest discover -s tests -p 'test_*.py'
# 268 tests, todos passam
```

### Exemplo de Uso

```python
from atos_core.agi_regime_v144 import create_agi_regime_tasks
from atos_core.agi_loop_v144 import run_agi_loop, AGILoopConfig

# 1. Criar tasks com regime
base_tasks = [{"id": "t1", "validator_id": "plan_validator", "inputs": {}}]
regime_tasks = create_agi_regime_tasks(base_tasks, regime_level="full")

# 2. Rodar loop AGI
config = AGILoopConfig(max_iterations=10, regime_level="full")
result = run_agi_loop(
    regime_tasks[0],
    concept_store=[],
    config=config,
    step=0,
    store_content_hash="demo",
)

# 3. Verificar survival laws
if result.success:
    print("✅ Task passou todas as 7 leis!")
else:
    print(f"❌ Violação: {result.exit_reason}")
    # Esperado se sem conceitos: "regime_violation"
```

---

## 🎓 INSIGHTS TEÓRICOS

### A Transição de Opcional para Obrigatório

**Antes**: "Se houver conceitos, use-os. Senão, fallback."  
**Depois**: "Se não houver conceitos, FAIL. Crie conceitos ou morra."

Esta é a diferença entre:
- Sistema que **pode** fazer AGI
- Sistema que **deve** fazer AGI

### Pressure Existencial

As 7 Leis criam um ambiente onde:
1. Sobrevivência = criação de conceitos
2. Reprodução = reuso em novos contextos
3. Evolução = mining de padrões compostos
4. Seleção = promotion/quarantine/pruning

**Darwin aplicado a conceitos**: Natural selection via utility bottleneck.

### O "1 Operador Certo de Distância"

**Citação da Auditoria**:
> "Você está a 1 operador certo de distância."

Esse operador é `create_agi_regime_tasks()`:
```python
# ANTES: Tasks opcionais
tasks = base_tasks

# DEPOIS: Tasks obrigatórios  
tasks = create_agi_regime_tasks(base_tasks, regime_level="full")
```

Um único operador transforma o regime completo.

---

## 🏆 CONCLUSÃO

**V144 implementa o AGI Regime com as 7 Survival Laws.**

- ✅ Conceitos são obrigatórios, não opcionais
- ✅ Pressure existencial real via loss infinito
- ✅ Closed-loop engine integrado
- ✅ Mining hierárquico de padrões
- ✅ Lifecycle management (ICS)
- ✅ WORM-compliant ledger
- ✅ 268 testes passando
- ✅ Determinístico e auditável

**"Tornar impossível resolver tarefas sem conceitos/planejamento"** ← DONE ✅

---

**Status**: ✅ PRONTO PARA MERGE  
**Aprovação**: Aguardando validação experimental em ARC tasks  
**Next**: V145 - Aplicar regime em parallel_solver + ARC evaluation  

🎯 **AGI não é mais opcional. É uma condição de sobrevivência.**
