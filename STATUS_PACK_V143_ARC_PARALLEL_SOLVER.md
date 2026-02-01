# STATUS_PACK_V143 - ARC Parallel Solver com Avaliação Real

**Data**: 2025-01-28  
**Versão**: V143  
**Predecessor**: V142 (parallel_planner_v142.py - skeleton paralelo)  
**Objetivo**: Integrar worker search com avaliação real via arc_solver_v141

---

## 1. Resumo Executivo

V143 implementa o primeiro solver paralelo completo para ARC-AGI com avaliação real.
Conecta a infraestrutura determinística de V142 com a lógica de apply/eval de arc_solver_v141.

### Mudanças Principais
- **arc_parallel_solver_v143.py**: Solver paralelo com ProcessPoolExecutor
- **Best-first search**: Heapq com prioridade (loss_shape, loss_cells, cost_bits, depth, sig)
- **Distribuição de operadores**: Workers recebem fatias contíguas de OP_DEFS_V141
- **Caching local**: apply_cache e eval_cache por worker
- **Paleta de cores**: Inclui cores de input E output para transformações

---

## 2. Arquitetura

### 2.1 Estruturas de Dados

```
ProgramStepV143(op_id: str, args: FrozenDict)
    └── Passo imutável, hashable

ProgramV143(steps: Tuple[ProgramStepV143, ...])
    └── Programa completo, imutável

EvalResultV143(loss_shape: int, loss_cells: int, cost_bits: int, 
               train_final_states: List[GridV124], test_out: Optional[GridV124])
    └── Resultado de avaliação com métricas
```

### 2.2 Fluxo de Execução

```
solve_arc_task_parallel_v143()
    │
    ├─► Inicializar config e serializar dados
    │
    ├─► ProcessPoolExecutor(max_workers=N)
    │       │
    │       ├─► _worker_search_v143(worker_0, ops[0:K])
    │       ├─► _worker_search_v143(worker_1, ops[K:2K])
    │       └─► ...
    │
    ├─► Coletar resultados de cada worker
    │
    └─► Selecionar melhor solução (menor loss, cost_bits)
```

### 2.3 Worker Search Algorithm

```
_worker_search_v143():
    frontier = [(priority, program)]  # heapq best-first
    
    while frontier and len(explored) < max_programs:
        _, prog = heappop(frontier)
        
        eval_result = _eval_program_v143(prog, train_pairs, caches)
        
        if loss_shape == 0 and loss_cells == 0:
            return SOLVED
        
        if depth < max_depth:
            for next_step in _propose_next_steps_v143(...):
                new_prog = prog + (next_step,)
                heappush(frontier, (new_priority, new_prog))
    
    return best_found
```

---

## 3. Funções Chave

### 3.1 Avaliação Real

```python
def _apply_step_v143(step: ProgramStepV143, state: GridV124, cache: Dict) -> GridV124:
    """Aplica operador usando arc_ops_v141.apply_op_v141"""
    key = (step, state.data_tuple)
    if key in cache:
        return cache[key]
    result = apply_op_v141(op_id=step.op_id, args=dict(step.args), state=state)
    cache[key] = result
    return result

def _eval_program_v143(prog: ProgramV143, train_pairs, test_in, caches) -> EvalResultV143:
    """Avalia programa em todos os train pairs, retorna loss agregado"""
    # Aplica sequência de passos a cada input
    # Compara com output esperado
    # Retorna (loss_shape, loss_cells, cost_bits)
```

### 3.2 Proposta de Próximos Passos

```python
def _propose_next_steps_v143(
    operator_slice: Tuple[int, int],
    all_operator_ids: List[str],
    current_states: List[GridV124],
    palette_out: Set[int]
) -> List[ProgramStepV143]:
    """Gera candidatos baseado em:
    - Operadores na fatia do worker
    - Slots disponíveis no grid atual (linhas, colunas)
    - Cores disponíveis (input + output palette)
    """
```

**Correção importante**: A paleta de cores agora inclui cores do estado atual (input transformado), não apenas do output esperado. Isso permite operações como `replace_color(from_color=input_color, to_color=output_color)`.

---

## 4. Correções de Bugs

### 4.1 replace_color Args
```python
# ERRADO:
{"old": c1, "new": c2}

# CORRETO (conforme arc_ops_v132.py):
{"from_color": int(c1), "to_color": int(c2)}
```

### 4.2 step_cost_bits_v141 Signature
```python
# ERRADO:
step_cost_bits_v141(step.op_id, step.args)

# CORRETO (keyword args):
step_cost_bits_v141(op_id=str(step.op_id), args=dict(step.args))
```

### 4.3 Color Palette
```python
# ERRADO (só output):
all_colors = set(palette_out)

# CORRETO (input + output):
state_colors = set()
for st in train_final_states:
    for row in st.grid:
        for c in row:
            state_colors.add(int(c))
all_colors = set(palette_out) | state_colors
```

---

## 5. Testes

### 5.1 Resultados
```
$ python3 -m unittest tests.test_arc_parallel_solver_v143 -v
Ran 21 tests in 0.214s
OK
```

### 5.2 Testes de Integração Real

| Teste | Task | Resultado | Programa Encontrado |
|-------|------|-----------|---------------------|
| test_solve_simple_rotate90 | Rotação 90° | SOLVED | `[{"op_id": "rotate90", "args": {}}]` |
| test_solve_replace_color | Blue→Gray | SOLVED | `[{"op_id": "replace_color", "args": {"from_color": 1, "to_color": 5}}]` |

### 5.3 Suite Completa
```
$ python3 -m unittest discover -s tests -p "test_*.py" -v
Ran 242 tests in 2.474s
OK
```

---

## 6. Limitações Conhecidas

1. **Cache não compartilhado**: Cada worker tem seu próprio cache. Não há sharing via multiprocessing.Manager ainda.

2. **Distribuição por índice**: Workers recebem fatias contíguas de operadores. Se um operador crítico (ex: rotate90) estiver em fatia de worker lento, pode haver delay.

3. **Sem beam width**: Busca explora todos os candidatos até max_programs. Beam search limitaria memória.

4. **Sem hash consing**: Programas idênticos em workers diferentes não são detectados.

---

## 7. Próximos Passos (Prioridade)

### P1: Shared Cache (Incremental)
```python
# Usar Manager para compartilhar apply_cache
manager = multiprocessing.Manager()
shared_cache = manager.dict()
```

### P2: Result Aggregation
- Agregar melhores parciais de cada worker
- Implementar checkpoint periódico

### P3: Beam Search
- Limitar frontier size por beam_width
- Podar candidatos com loss muito alto

### P4: Teste em ARC-AGI Real
- Executar em tasks do dataset oficial
- Medir taxa de solução e tempo

---

## 8. Arquivos Criados

| Arquivo | Linhas | Propósito |
|---------|--------|-----------|
| `atos_core/arc_parallel_solver_v143.py` | ~1000 | Solver paralelo com avaliação real |
| `tests/test_arc_parallel_solver_v143.py` | ~600 | 21 testes unitários |

---

## 9. Dependências

```
arc_parallel_solver_v143.py
    ├── arc_solver_v141.py (solve_arc_task_v141 - não usado diretamente)
    ├── arc_ops_v141.py (apply_op_v141, step_cost_bits_v141, OP_DEFS_V141)
    ├── arc_types_v124.py (GridV124)
    ├── parallel_planner_v142.py (WorkerLogEntryV142, LogSeverityV142)
    └── multiprocessing (ProcessPoolExecutor)
```

---

## 10. Conformidade WORM

- Hash chain via `write_parallel_solve_to_ledger_v143()`
- Cada entrada inclui: config, resultado, trace, hash anterior
- Append-only no arquivo de ledger

---

## 11. Métricas de Busca

```python
ParallelSolverResultV143 inclui:
    solved: bool
    program: Optional[List[Dict]]
    test_output: Optional[List[List[int]]]
    search_stats: Dict[str, Any]
        - total_programs_explored
        - workers_used
        - best_loss_per_worker
        - runtime_seconds
```

---

## 12. Exemplo de Uso

```python
from atos_core.arc_parallel_solver_v143 import (
    solve_arc_task_parallel_v143,
    ParallelSolverConfigV143,
)

config = ParallelSolverConfigV143(
    num_workers=4,
    max_programs_per_worker=1000,
    max_depth=5,
    seed=42,
)

result = solve_arc_task_parallel_v143(
    train_pairs=[(in_grid, out_grid), ...],
    test_in=test_grid,
    config=config,
)

if result.solved:
    print(f"Programa: {result.program}")
    print(f"Saída: {result.test_output}")
```

---

**Status**: ✅ COMPLETO - Pronto para testes em escala
