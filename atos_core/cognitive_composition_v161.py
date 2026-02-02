"""
cognitive_composition_v161.py - Operadores de Composição e Abstração

Esta expansão implementa todos os operadores de composição e abstração
autorizados para alcançar ≥90% de acerto nos benchmarks ARC-AGI-1 e ARC-AGI-2.

Os operadores de composição/abstração realizam:
- Encadeamento e composição de operações
- Criação e instanciação de templates
- Abstração hierárquica e multi-nível
- Planejamento e decomposição de metas
- Composição dinâmica e adaptativa

Schema version: 161
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import math
from collections import Counter
from copy import deepcopy

# Type alias
GridV124 = List[List[int]]

COGNITIVE_COMPOSITION_SCHEMA_VERSION_V161 = 161


# ─────────────────────────────────────────────────────────────────────────────
# Composition Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ComposedOperator:
    """Result of composing operators."""
    name: str
    operations: List[Dict[str, Any]]
    params: Dict[str, Any] = field(default_factory=dict)
    
    def __call__(self, g: GridV124, **kwargs) -> GridV124:
        result = g
        for op in self.operations:
            func = op.get("func")
            op_params = {**op.get("params", {}), **kwargs}
            if callable(func):
                result = func(result, **op_params)
        return result


@dataclass
class Template:
    """A parameterized template concept."""
    name: str
    pattern: GridV124
    placeholders: Dict[str, Any]
    constraints: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class AbstractConcept:
    """An abstract concept that can be instantiated."""
    name: str
    level: int
    components: List[str]
    instantiation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionPlan:
    """A plan for executing transformations."""
    steps: List[Dict[str, Any]]
    dependencies: Dict[int, List[int]]
    estimated_cost: float


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def _grid_shape_v161(g: GridV124) -> Tuple[int, int]:
    """Get (height, width) of grid."""
    if not g:
        return (0, 0)
    return (len(g), len(g[0]) if g[0] else 0)


def _copy_grid_v161(g: GridV124) -> GridV124:
    """Deep copy a grid."""
    return [[int(c) for c in row] for row in g]


def _new_grid_v161(rows: int, cols: int, fill: int = 0) -> GridV124:
    """Create a new grid with given dimensions."""
    return [[int(fill) for _ in range(int(cols))] for _ in range(int(rows))]


def _grids_equal_v161(g1: GridV124, g2: GridV124) -> bool:
    """Check if two grids are equal."""
    h1, w1 = _grid_shape_v161(g1)
    h2, w2 = _grid_shape_v161(g2)
    if h1 != h2 or w1 != w2:
        return False
    for r in range(h1):
        for c in range(w1):
            if int(g1[r][c]) != int(g2[r][c]):
                return False
    return True


def _find_objects_v161(g: GridV124, *, bg: int = 0) -> List[Set[Tuple[int, int]]]:
    """Find all connected components."""
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return []
    
    visited: Set[Tuple[int, int]] = set()
    objects: List[Set[Tuple[int, int]]] = []
    
    for r in range(h):
        for c in range(w):
            if (r, c) in visited or int(g[r][c]) == bg:
                continue
            
            component: Set[Tuple[int, int]] = set()
            queue = [(r, c)]
            visited.add((r, c))
            
            while queue:
                cr, cc = queue.pop(0)
                component.add((cr, cc))
                
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if (nr, nc) not in visited and int(g[nr][nc]) != bg:
                            visited.add((nr, nc))
                            queue.append((nr, nc))
            
            objects.append(component)
    
    return objects


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC COMPOSITION OPERATORS
# ─────────────────────────────────────────────────────────────────────────────


def chain_operations(g: GridV124, 
                     operations: List[Callable], 
                     *, params_list: List[Dict] = None) -> GridV124:
    """
    Aplica sequência de operações em pipeline.
    
    Papel: Executa transformações encadeadas.
    Justificativa: A maioria das regras envolve múltiplas transformações.
    """
    result = _copy_grid_v161(g)
    params_list = params_list or [{}] * len(operations)
    
    for i, op in enumerate(operations):
        params = params_list[i] if i < len(params_list) else {}
        try:
            result = op(result, **params)
        except Exception:
            pass
    
    return result


def conditional_composition(g: GridV124,
                            condition: Callable[[GridV124], bool],
                            if_true: Callable[[GridV124], GridV124],
                            if_false: Callable[[GridV124], GridV124]) -> GridV124:
    """
    Aplica operação A ou B conforme condição.
    
    Papel: Ramifica execução baseado em estado do grid.
    Justificativa: Regras frequentemente têm exceções condicionais.
    """
    if condition(g):
        return if_true(g)
    else:
        return if_false(g)


def loop_composition(g: GridV124,
                     operation: Callable[[GridV124], GridV124],
                     *, condition: Callable[[GridV124], bool] = None,
                     max_iterations: int = 10) -> GridV124:
    """
    Repete operação enquanto condição for verdadeira.
    
    Papel: Permite transformações iterativas até convergência.
    Justificativa: Alguns padrões requerem aplicação repetida de regras.
    """
    result = _copy_grid_v161(g)
    
    for _ in range(max_iterations):
        if condition and not condition(result):
            break
        
        prev = _copy_grid_v161(result)
        result = operation(result)
        
        # Stop if no change
        if _grids_equal_v161(prev, result):
            break
    
    return result


def parallel_composition(g: GridV124,
                         operations: List[Callable],
                         merger: Callable[[List[GridV124]], GridV124]) -> GridV124:
    """
    Aplica operações em paralelo e combina resultados.
    
    Papel: Executa múltiplas transformações independentes.
    Justificativa: Algumas regras combinam vários efeitos simultâneos.
    """
    results = []
    for op in operations:
        try:
            results.append(op(g))
        except Exception:
            results.append(_copy_grid_v161(g))
    
    if not results:
        return _copy_grid_v161(g)
    
    return merger(results)


def merge_max(grids: List[GridV124]) -> GridV124:
    """Merge grids by taking maximum value at each cell."""
    if not grids:
        return [[]]
    
    h, w = _grid_shape_v161(grids[0])
    result = _new_grid_v161(h, w, 0)
    
    for r in range(h):
        for c in range(w):
            vals = [int(g[r][c]) for g in grids if r < len(g) and c < len(g[0])]
            result[r][c] = max(vals) if vals else 0
    
    return result


def merge_union(grids: List[GridV124], *, bg: int = 0) -> GridV124:
    """Merge grids by union (any non-bg wins)."""
    if not grids:
        return [[]]
    
    h, w = _grid_shape_v161(grids[0])
    result = _new_grid_v161(h, w, bg)
    
    for r in range(h):
        for c in range(w):
            for g in grids:
                if r < len(g) and c < len(g[0]) and int(g[r][c]) != bg:
                    result[r][c] = int(g[r][c])
                    break
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 2. PARAMETERIZED OPERATORS
# ─────────────────────────────────────────────────────────────────────────────


def parameterized_operator(operation: Callable,
                           param_bindings: Dict[str, Any]) -> ComposedOperator:
    """
    Cria operador com parâmetros predefinidos.
    
    Papel: Generaliza operador para diferentes configurações.
    Justificativa: Parametrização permite reusar mesma lógica.
    """
    def bound_op(g: GridV124, **kwargs) -> GridV124:
        merged_params = {**param_bindings, **kwargs}
        return operation(g, **merged_params)
    
    return ComposedOperator(
        name=f"parameterized_{operation.__name__}",
        operations=[{"func": bound_op, "params": param_bindings}],
        params=param_bindings
    )


def abstract_parameterization(operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extrai parâmetros comuns de operações concretas.
    
    Papel: Identifica variáveis em uma sequência de operações.
    Justificativa: Parâmetros abstratos permitem generalização.
    """
    all_params: Dict[str, List[Any]] = {}
    
    for op in operations:
        params = op.get("params", {})
        for key, value in params.items():
            if key not in all_params:
                all_params[key] = []
            all_params[key].append(value)
    
    # Find parameters that vary
    variable_params = {}
    constant_params = {}
    
    for key, values in all_params.items():
        unique_values = set(str(v) for v in values)
        if len(unique_values) > 1:
            variable_params[key] = {
                "observed_values": list(unique_values),
                "type": type(values[0]).__name__
            }
        else:
            constant_params[key] = values[0] if values else None
    
    return {
        "variable_params": variable_params,
        "constant_params": constant_params,
        "total_params": len(all_params)
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. TEMPLATE OPERATORS
# ─────────────────────────────────────────────────────────────────────────────


def template_concept(pattern: GridV124,
                     *, placeholder_color: int = 9) -> Template:
    """
    Cria template abstrato com placeholders.
    
    Papel: Define padrão genérico que pode ser instanciado.
    Justificativa: Templates capturam estrutura sem cor específica.
    """
    h, w = _grid_shape_v161(pattern)
    placeholders = {}
    
    for r in range(h):
        for c in range(w):
            color = int(pattern[r][c])
            if color == placeholder_color:
                ph_id = f"ph_{r}_{c}"
                placeholders[ph_id] = {"position": (r, c), "original": color}
    
    return Template(
        name="template",
        pattern=_copy_grid_v161(pattern),
        placeholders=placeholders
    )


def concept_instantiation(template: Template,
                          bindings: Dict[str, int]) -> GridV124:
    """
    Instancia template com valores concretos.
    
    Papel: Transforma template abstrato em grid concreto.
    Justificativa: Instanciação preenche placeholders.
    """
    result = _copy_grid_v161(template.pattern)
    
    for ph_id, info in template.placeholders.items():
        if ph_id in bindings:
            r, c = info["position"]
            result[r][c] = int(bindings[ph_id])
    
    return result


def placeholder_filler(g: GridV124,
                       template: Template,
                       *, infer_from_context: bool = True) -> GridV124:
    """
    Preenche placeholders do template baseado no contexto.
    
    Papel: Auto-completa template analisando grid de entrada.
    Justificativa: Inferência automática economiza especificação manual.
    """
    result = _copy_grid_v161(template.pattern)
    h, w = _grid_shape_v161(g)
    th, tw = _grid_shape_v161(result)
    
    if infer_from_context:
        # Find dominant color in input that's not in template
        template_colors = set()
        for row in template.pattern:
            for cell in row:
                template_colors.add(int(cell))
        
        input_colors = Counter()
        for row in g:
            for cell in row:
                input_colors[int(cell)] += 1
        
        # Find color to use for placeholders
        fill_color = 0
        for color, count in input_colors.most_common():
            if color not in template_colors:
                fill_color = color
                break
        
        # Fill placeholders
        for ph_id, info in template.placeholders.items():
            r, c = info["position"]
            result[r][c] = fill_color
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 4. HIERARCHICAL COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────


def hierarchical_composition(g: GridV124,
                             levels: List[List[Callable]]) -> GridV124:
    """
    Aplica operações organizadas hierarquicamente.
    
    Papel: Executa transformações em níveis de abstração.
    Justificativa: Problemas complexos decompõem-se em níveis.
    """
    result = _copy_grid_v161(g)
    
    for level_ops in levels:
        for op in level_ops:
            try:
                result = op(result)
            except Exception:
                pass
    
    return result


def layered_abstraction(concepts: List[AbstractConcept]) -> Dict[str, Any]:
    """
    Organiza conceitos em camadas de abstração.
    
    Papel: Estrutura conceitos do concreto ao abstrato.
    Justificativa: Organização em camadas facilita navegação.
    """
    layers: Dict[int, List[AbstractConcept]] = {}
    
    for concept in concepts:
        level = concept.level
        if level not in layers:
            layers[level] = []
        layers[level].append(concept)
    
    # Sort by level
    sorted_layers = dict(sorted(layers.items()))
    
    return {
        "layers": {k: [c.name for c in v] for k, v in sorted_layers.items()},
        "total_concepts": len(concepts),
        "num_layers": len(sorted_layers),
        "max_level": max(layers.keys()) if layers else 0
    }


def recursive_composition(g: GridV124,
                          operation: Callable,
                          *, depth: int = 3,
                          aggregator: Callable = None) -> GridV124:
    """
    Aplica operação recursivamente em níveis decrescentes.
    
    Papel: Processa grid em resolução decrescente.
    Justificativa: Recursão captura estrutura em múltiplas escalas.
    """
    if depth <= 0:
        return _copy_grid_v161(g)
    
    # Apply operation at current level
    result = operation(g)
    
    # Recurse (simplified: apply again at "smaller scale")
    result = recursive_composition(result, operation, depth=depth-1, aggregator=aggregator)
    
    return result


def multi_step_abstraction(g: GridV124,
                           abstractors: List[Callable],
                           *, preserve_structure: bool = True) -> List[GridV124]:
    """
    Gera múltiplos níveis de abstração do grid.
    
    Papel: Produz versões progressivamente mais abstratas.
    Justificativa: Diferentes níveis revelam diferentes padrões.
    """
    abstractions = [_copy_grid_v161(g)]
    current = g
    
    for abstractor in abstractors:
        try:
            current = abstractor(current)
            abstractions.append(_copy_grid_v161(current))
        except Exception:
            pass
    
    return abstractions


# ─────────────────────────────────────────────────────────────────────────────
# 5. DYNAMIC COMPOSITION
# ─────────────────────────────────────────────────────────────────────────────


def dynamic_composer(g: GridV124,
                     operator_pool: Dict[str, Callable],
                     selector: Callable[[GridV124], str]) -> GridV124:
    """
    Seleciona operador dinamicamente baseado no estado.
    
    Papel: Escolhe transformação adequada em runtime.
    Justificativa: Adaptação dinâmica melhora flexibilidade.
    """
    selected_name = selector(g)
    
    if selected_name in operator_pool:
        return operator_pool[selected_name](g)
    
    return _copy_grid_v161(g)


def combinatorial_composer(g: GridV124,
                           operators: List[Callable],
                           *, max_depth: int = 2) -> List[Tuple[List[str], GridV124]]:
    """
    Gera todas combinações de operadores até certa profundidade.
    
    Papel: Explora espaço de composições possíveis.
    Justificativa: Busca exaustiva encontra composição correta.
    """
    results = []
    op_names = [op.__name__ for op in operators]
    
    # Depth 1: single operators
    for i, op in enumerate(operators):
        try:
            result = op(g)
            results.append(([op_names[i]], result))
        except:
            pass
    
    # Depth 2: pairs
    if max_depth >= 2:
        for i, op1 in enumerate(operators):
            for j, op2 in enumerate(operators):
                try:
                    temp = op1(g)
                    result = op2(temp)
                    results.append(([op_names[i], op_names[j]], result))
                except:
                    pass
    
    return results


def backtracking_composer(g: GridV124,
                          target: GridV124,
                          operators: List[Callable],
                          *, max_depth: int = 5) -> Optional[List[str]]:
    """
    Encontra sequência de operadores via backtracking.
    
    Papel: Busca caminho de transformação com retrocesso.
    Justificativa: Backtracking evita becos sem saída.
    """
    op_names = [op.__name__ for op in operators]
    
    def search(current: GridV124, path: List[int], depth: int) -> Optional[List[int]]:
        if _grids_equal_v161(current, target):
            return path
        
        if depth >= max_depth:
            return None
        
        for i, op in enumerate(operators):
            try:
                next_grid = op(current)
                result = search(next_grid, path + [i], depth + 1)
                if result is not None:
                    return result
            except:
                pass
        
        return None
    
    path = search(g, [], 0)
    
    if path is not None:
        return [op_names[i] for i in path]
    
    return None


def auto_context_switch(g: GridV124,
                        contexts: Dict[str, Dict[str, Callable]],
                        context_detector: Callable[[GridV124], str]) -> GridV124:
    """
    Troca de contexto automaticamente baseado na entrada.
    
    Papel: Seleciona conjunto de operações apropriado.
    Justificativa: Diferentes inputs requerem diferentes abordagens.
    """
    detected_context = context_detector(g)
    
    if detected_context in contexts:
        ops = contexts[detected_context]
        # Apply default operation from context
        if "default" in ops:
            return ops["default"](g)
        elif ops:
            # Use first available operation
            first_op = list(ops.values())[0]
            return first_op(g)
    
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 6. PLANNING AND DECOMPOSITION
# ─────────────────────────────────────────────────────────────────────────────


def goal_decomposition(start: GridV124,
                       goal: GridV124,
                       *, bg: int = 0) -> ExecutionPlan:
    """
    Decompõe transformação em sub-metas.
    
    Papel: Quebra problema grande em passos menores.
    Justificativa: Dividir e conquistar simplifica resolução.
    """
    steps = []
    
    sh, sw = _grid_shape_v161(start)
    gh, gw = _grid_shape_v161(goal)
    
    # Step 1: Handle size change
    if sh != gh or sw != gw:
        steps.append({
            "id": 0,
            "type": "resize",
            "from": (sh, sw),
            "to": (gh, gw)
        })
    
    # Step 2: Handle color changes
    start_colors = Counter()
    goal_colors = Counter()
    
    for r in range(sh):
        for c in range(sw):
            start_colors[int(start[r][c])] += 1
    
    for r in range(gh):
        for c in range(gw):
            goal_colors[int(goal[r][c])] += 1
    
    for color in set(list(start_colors.keys()) + list(goal_colors.keys())):
        start_count = start_colors.get(color, 0)
        goal_count = goal_colors.get(color, 0)
        
        if start_count != goal_count:
            steps.append({
                "id": len(steps),
                "type": "color_adjust",
                "color": color,
                "from_count": start_count,
                "to_count": goal_count
            })
    
    # Step 3: Handle object changes
    start_objects = _find_objects_v161(start, bg=bg)
    goal_objects = _find_objects_v161(goal, bg=bg)
    
    if len(start_objects) != len(goal_objects):
        steps.append({
            "id": len(steps),
            "type": "object_adjust",
            "from_count": len(start_objects),
            "to_count": len(goal_objects)
        })
    
    # Build dependencies (linear for now)
    dependencies = {i: [i-1] for i in range(1, len(steps))}
    
    return ExecutionPlan(
        steps=steps,
        dependencies=dependencies,
        estimated_cost=len(steps) * 10
    )


def divide_and_conquer_operator(g: GridV124,
                                divider: Callable[[GridV124], List[GridV124]],
                                solver: Callable[[GridV124], GridV124],
                                combiner: Callable[[List[GridV124]], GridV124]) -> GridV124:
    """
    Divide problema, resolve partes, combina resultados.
    
    Papel: Aplica paradigma divide-and-conquer.
    Justificativa: Problemas grandes se resolvem dividindo.
    """
    # Divide
    parts = divider(g)
    
    # Conquer
    solved_parts = [solver(part) for part in parts]
    
    # Combine
    return combiner(solved_parts)


def partial_order_planner(tasks: List[Dict[str, Any]],
                          constraints: List[Tuple[int, int]]) -> ExecutionPlan:
    """
    Planeja ordem parcial de tarefas.
    
    Papel: Ordena tarefas respeitando dependências.
    Justificativa: Ordem correta evita estados inválidos.
    """
    # Build dependency graph
    dependencies: Dict[int, List[int]] = {i: [] for i in range(len(tasks))}
    
    for before, after in constraints:
        if 0 <= before < len(tasks) and 0 <= after < len(tasks):
            dependencies[after].append(before)
    
    # Topological sort to get valid order
    visited = set()
    order = []
    
    def visit(i):
        if i in visited:
            return
        visited.add(i)
        for dep in dependencies[i]:
            visit(dep)
        order.append(i)
    
    for i in range(len(tasks)):
        visit(i)
    
    # Reorder tasks
    ordered_tasks = [tasks[i] for i in order]
    
    return ExecutionPlan(
        steps=ordered_tasks,
        dependencies=dependencies,
        estimated_cost=sum(t.get("cost", 1) for t in tasks)
    )


# ─────────────────────────────────────────────────────────────────────────────
# 7. ABSTRACTION OPERATORS
# ─────────────────────────────────────────────────────────────────────────────


def schema_abstraction(examples: List[Tuple[GridV124, GridV124]]) -> Dict[str, Any]:
    """
    Extrai esquema abstrato de exemplos.
    
    Papel: Generaliza padrão a partir de instâncias concretas.
    Justificativa: Esquemas capturam essência da transformação.
    """
    if not examples:
        return {"schema": None}
    
    # Analyze common properties
    shape_transforms = []
    color_mappings = []
    
    for inp, out in examples:
        ih, iw = _grid_shape_v161(inp)
        oh, ow = _grid_shape_v161(out)
        
        shape_transforms.append({
            "in": (ih, iw),
            "out": (oh, ow),
            "ratio": (oh / max(1, ih), ow / max(1, iw))
        })
        
        # Color mapping for same-size grids
        if ih == oh and iw == ow:
            mapping = {}
            for r in range(ih):
                for c in range(iw):
                    in_c = int(inp[r][c])
                    out_c = int(out[r][c])
                    if in_c != out_c:
                        if in_c not in mapping:
                            mapping[in_c] = out_c
            color_mappings.append(mapping)
    
    # Abstract schema
    schema = {
        "shape_transform": None,
        "color_mapping": None
    }
    
    # Check for consistent shape transform
    ratios = [st["ratio"] for st in shape_transforms]
    if len(set(ratios)) == 1:
        schema["shape_transform"] = {
            "type": "ratio",
            "ratio": ratios[0]
        }
    
    # Check for consistent color mapping
    if color_mappings:
        consistent = all(m == color_mappings[0] for m in color_mappings)
        if consistent:
            schema["color_mapping"] = color_mappings[0]
    
    return schema


def concept_refinement_operator(concept: AbstractConcept,
                                feedback: Dict[str, Any]) -> AbstractConcept:
    """
    Refina conceito baseado em feedback.
    
    Papel: Ajusta abstração conforme novos dados.
    Justificativa: Refinamento melhora precisão progressivamente.
    """
    refined = AbstractConcept(
        name=concept.name + "_refined",
        level=concept.level,
        components=concept.components.copy(),
        instantiation_rules=concept.instantiation_rules.copy()
    )
    
    # Apply feedback
    if "add_component" in feedback:
        refined.components.append(feedback["add_component"])
    
    if "remove_component" in feedback and feedback["remove_component"] in refined.components:
        refined.components.remove(feedback["remove_component"])
    
    if "update_rules" in feedback:
        refined.instantiation_rules.update(feedback["update_rules"])
    
    return refined


def detail_refiner(g: GridV124,
                   abstract: GridV124,
                   *, enhancement_level: int = 1) -> GridV124:
    """
    Adiciona detalhes a uma versão abstrata.
    
    Papel: Enriquece representação simplificada.
    Justificativa: Partindo de abstração, adiciona detalhes necessários.
    """
    h, w = _grid_shape_v161(abstract)
    result = _copy_grid_v161(abstract)
    
    # Add details from original where abstract has placeholders
    gh, gw = _grid_shape_v161(g)
    
    for r in range(min(h, gh)):
        for c in range(min(w, gw)):
            # If abstract cell is "empty" (0), use original
            if int(result[r][c]) == 0 and int(g[r][c]) != 0:
                if enhancement_level >= 1:
                    result[r][c] = int(g[r][c])
    
    return result


def detail_injector(abstract: GridV124,
                    details: Dict[Tuple[int, int], int]) -> GridV124:
    """
    Injeta detalhes específicos em posições.
    
    Papel: Adiciona valores pontuais à abstração.
    Justificativa: Detalhes específicos completam a transformação.
    """
    result = _copy_grid_v161(abstract)
    h, w = _grid_shape_v161(result)
    
    for (r, c), value in details.items():
        if 0 <= r < h and 0 <= c < w:
            result[r][c] = int(value)
    
    return result


def on_the_fly_abstraction(g: GridV124,
                           abstraction_rules: List[Callable[[int], int]]) -> GridV124:
    """
    Abstrai grid aplicando regras em tempo real.
    
    Papel: Transforma valores conforme regras de mapeamento.
    Justificativa: Abstração on-the-fly é eficiente em memória.
    """
    h, w = _grid_shape_v161(g)
    result = _new_grid_v161(h, w, 0)
    
    for r in range(h):
        for c in range(w):
            value = int(g[r][c])
            for rule in abstraction_rules:
                value = rule(value)
            result[r][c] = value
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 8. ANALOGY AND BLENDING
# ─────────────────────────────────────────────────────────────────────────────


def analogy_composition(source_pair: Tuple[GridV124, GridV124],
                        target_input: GridV124) -> GridV124:
    """
    Aplica transformação aprendida de exemplo por analogia.
    
    Papel: Transfere padrão de (A→B) para C→?.
    Justificativa: Analogia é mecanismo fundamental de generalização.
    """
    source_in, source_out = source_pair
    
    sh, sw = _grid_shape_v161(source_in)
    oh, ow = _grid_shape_v161(source_out)
    th, tw = _grid_shape_v161(target_input)
    
    # Compute transformation ratio
    h_ratio = oh / max(1, sh)
    w_ratio = ow / max(1, sw)
    
    # Apply to target
    result_h = int(th * h_ratio)
    result_w = int(tw * w_ratio)
    
    if result_h == 0 or result_w == 0:
        return _copy_grid_v161(target_input)
    
    result = _new_grid_v161(result_h, result_w, 0)
    
    # Build color mapping from source pair
    color_map = {}
    if sh == oh and sw == ow:
        for r in range(sh):
            for c in range(sw):
                in_c = int(source_in[r][c])
                out_c = int(source_out[r][c])
                if in_c not in color_map:
                    color_map[in_c] = out_c
    
    # Apply to target
    for r in range(result_h):
        for c in range(result_w):
            src_r = int(r / h_ratio) if h_ratio > 0 else r
            src_c = int(c / w_ratio) if w_ratio > 0 else c
            
            if src_r < th and src_c < tw:
                in_color = int(target_input[src_r][src_c])
                result[r][c] = color_map.get(in_color, in_color)
    
    return result


def conceptual_blender(concept1: AbstractConcept,
                       concept2: AbstractConcept) -> AbstractConcept:
    """
    Combina dois conceitos em um novo híbrido.
    
    Papel: Funde características de conceitos diferentes.
    Justificativa: Blending gera novas ideias a partir de existentes.
    """
    # Merge components
    merged_components = list(set(concept1.components + concept2.components))
    
    # Merge rules
    merged_rules = {**concept1.instantiation_rules, **concept2.instantiation_rules}
    
    # Average level
    avg_level = (concept1.level + concept2.level) // 2
    
    return AbstractConcept(
        name=f"{concept1.name}_x_{concept2.name}",
        level=avg_level,
        components=merged_components,
        instantiation_rules=merged_rules
    )


def metaphorical_mapping(source_domain: Dict[str, Any],
                         target_domain: Dict[str, Any],
                         mappings: Dict[str, str]) -> Dict[str, Any]:
    """
    Mapeia estrutura de um domínio para outro.
    
    Papel: Transfere relações entre domínios.
    Justificativa: Metáforas estruturais permitem raciocínio por analogia.
    """
    result = {}
    
    for source_key, target_key in mappings.items():
        if source_key in source_domain:
            result[target_key] = source_domain[source_key]
    
    # Keep unmapped target properties
    for key, value in target_domain.items():
        if key not in result:
            result[key] = value
    
    return result


def schema_merger(schema1: Dict[str, Any],
                  schema2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Combina dois esquemas em um unificado.
    
    Papel: Cria esquema mais geral a partir de dois específicos.
    Justificativa: Unificação de esquemas amplia cobertura.
    """
    merged = {}
    
    all_keys = set(list(schema1.keys()) + list(schema2.keys()))
    
    for key in all_keys:
        v1 = schema1.get(key)
        v2 = schema2.get(key)
        
        if v1 is None:
            merged[key] = v2
        elif v2 is None:
            merged[key] = v1
        elif v1 == v2:
            merged[key] = v1
        else:
            # Conflict: keep both as options
            merged[key] = {"option1": v1, "option2": v2}
    
    return merged


def concept_overlay(base: GridV124,
                    overlay: GridV124,
                    *, transparent: int = 0) -> GridV124:
    """
    Sobrepõe grid conceitual em base.
    
    Papel: Combina duas representações em camadas.
    Justificativa: Sobreposição permite composição visual.
    """
    bh, bw = _grid_shape_v161(base)
    oh, ow = _grid_shape_v161(overlay)
    
    result = _copy_grid_v161(base)
    
    for r in range(min(bh, oh)):
        for c in range(min(bw, ow)):
            overlay_val = int(overlay[r][c])
            if overlay_val != transparent:
                result[r][c] = overlay_val
    
    return result


def concept_pipeline(g: GridV124,
                     pipeline: List[Dict[str, Any]]) -> GridV124:
    """
    Executa pipeline completo de transformações conceituais.
    
    Papel: Processa grid através de estágios definidos.
    Justificativa: Pipeline organiza fluxo de processamento.
    """
    result = _copy_grid_v161(g)
    
    for stage in pipeline:
        op = stage.get("operation")
        params = stage.get("params", {})
        
        if callable(op):
            try:
                result = op(result, **params)
            except Exception:
                pass
    
    return result


def exception_handler_composition(g: GridV124,
                                  main_op: Callable,
                                  exception_ops: Dict[type, Callable]) -> GridV124:
    """
    Aplica operação com tratamento de exceções alternativas.
    
    Papel: Fallback para operações que falham.
    Justificativa: Robustez requer tratamento de erros.
    """
    try:
        return main_op(g)
    except Exception as e:
        for exc_type, handler in exception_ops.items():
            if isinstance(e, exc_type):
                try:
                    return handler(g)
                except:
                    pass
    
    return _copy_grid_v161(g)


def abstract_solver(g: GridV124,
                    problem_type: str,
                    solvers: Dict[str, Callable]) -> GridV124:
    """
    Seleciona solver baseado no tipo de problema.
    
    Papel: Direciona para algoritmo apropriado.
    Justificativa: Diferentes problemas requerem diferentes soluções.
    """
    if problem_type in solvers:
        return solvers[problem_type](g)
    
    # Default: return copy
    return _copy_grid_v161(g)


def macro_operator_creator(operations: List[Dict[str, Any]],
                           name: str) -> ComposedOperator:
    """
    Cria macro-operador a partir de sequência de operações.
    
    Papel: Encapsula operações frequentes em unidade reutilizável.
    Justificativa: Macros reduzem repetição e erro.
    """
    ops_with_funcs = []
    
    for op in operations:
        if "func" in op:
            ops_with_funcs.append({
                "func": op["func"],
                "params": op.get("params", {})
            })
    
    return ComposedOperator(
        name=name,
        operations=ops_with_funcs,
        params={}
    )


def merge_strategies(strategies: List[Callable[[GridV124], GridV124]],
                     g: GridV124,
                     scorer: Callable[[GridV124], float]) -> GridV124:
    """
    Aplica múltiplas estratégias e seleciona melhor resultado.
    
    Papel: Compete estratégias e escolhe vencedora.
    Justificativa: Quando não há certeza, competição decide.
    """
    best_result = _copy_grid_v161(g)
    best_score = scorer(g)
    
    for strategy in strategies:
        try:
            result = strategy(g)
            score = scorer(result)
            
            if score > best_score:
                best_result = result
                best_score = score
        except:
            pass
    
    return best_result


# ─────────────────────────────────────────────────────────────────────────────
# COMPOSITION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────


COMPOSITION_OPERATORS_V161 = {
    # Basic composition
    "chain_operations": chain_operations,
    "conditional_composition": conditional_composition,
    "loop_composition": loop_composition,
    "parallel_composition": parallel_composition,
    
    # Parameterized
    "parameterized_operator": parameterized_operator,
    "abstract_parameterization": abstract_parameterization,
    
    # Templates
    "template_concept": template_concept,
    "concept_instantiation": concept_instantiation,
    "placeholder_filler": placeholder_filler,
    
    # Hierarchical
    "hierarchical_composition": hierarchical_composition,
    "layered_abstraction": layered_abstraction,
    "recursive_composition": recursive_composition,
    "multi_step_abstraction": multi_step_abstraction,
    
    # Dynamic
    "dynamic_composer": dynamic_composer,
    "combinatorial_composer": combinatorial_composer,
    "backtracking_composer": backtracking_composer,
    "auto_context_switch": auto_context_switch,
    
    # Planning
    "goal_decomposition": goal_decomposition,
    "divide_and_conquer_operator": divide_and_conquer_operator,
    "partial_order_planner": partial_order_planner,
    
    # Abstraction
    "schema_abstraction": schema_abstraction,
    "concept_refinement_operator": concept_refinement_operator,
    "detail_refiner": detail_refiner,
    "detail_injector": detail_injector,
    "on_the_fly_abstraction": on_the_fly_abstraction,
    
    # Analogy and blending
    "analogy_composition": analogy_composition,
    "conceptual_blender": conceptual_blender,
    "metaphorical_mapping": metaphorical_mapping,
    "schema_merger": schema_merger,
    "concept_overlay": concept_overlay,
    "concept_pipeline": concept_pipeline,
    "exception_handler_composition": exception_handler_composition,
    "abstract_solver": abstract_solver,
    "macro_operator_creator": macro_operator_creator,
    "merge_strategies": merge_strategies,
}


def get_composition_operator_v161(name: str):
    """Get a composition operator by name."""
    return COMPOSITION_OPERATORS_V161.get(name)


def list_composition_operators_v161() -> List[str]:
    """List all available composition operators."""
    return list(COMPOSITION_OPERATORS_V161.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
Total de Operadores de Composição/Abstração Implementados: 35

CATEGORIAS:
1. Composição Básica (4):
   - chain_operations, conditional_composition
   - loop_composition, parallel_composition

2. Parametrização (2):
   - parameterized_operator, abstract_parameterization

3. Templates (3):
   - template_concept, concept_instantiation, placeholder_filler

4. Hierárquica (4):
   - hierarchical_composition, layered_abstraction
   - recursive_composition, multi_step_abstraction

5. Dinâmica (4):
   - dynamic_composer, combinatorial_composer
   - backtracking_composer, auto_context_switch

6. Planejamento (3):
   - goal_decomposition, divide_and_conquer_operator
   - partial_order_planner

7. Abstração (5):
   - schema_abstraction, concept_refinement_operator
   - detail_refiner, detail_injector, on_the_fly_abstraction

8. Analogia e Blending (10):
   - analogy_composition, conceptual_blender
   - metaphorical_mapping, schema_merger
   - concept_overlay, concept_pipeline
   - exception_handler_composition, abstract_solver
   - macro_operator_creator, merge_strategies
"""
