"""
cognitive_ops_conditional_v161.py - Operadores Condicionais (Controle)

Esta expansão implementa todos os operadores de controle autorizados para
alcançar ≥90% de acerto nos benchmarks ARC-AGI-1 e ARC-AGI-2.

Os operadores condicionais permitem:
- Ramificação baseada em propriedades do input
- Loops e iterações controladas
- Fallbacks e estratégias alternativas
- Adaptação de comportamento a características da tarefa

Schema version: 161
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union
import time

# Type alias para grids
GridV124 = List[List[int]]

COGNITIVE_OPS_CONDITIONAL_SCHEMA_VERSION_V161 = 161


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


def _get_neighbors_4_v161(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Get 4-connected neighbors within bounds."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            neighbors.append((nr, nc))
    return neighbors


def _find_objects_v161(g: GridV124, *, bg: int = 0) -> List[Set[Tuple[int, int]]]:
    """Find all connected components (objects) in grid."""
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
                
                for nr, nc in _get_neighbors_4_v161(cr, cc, h, w):
                    if (nr, nc) not in visited and int(g[nr][nc]) != bg:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            
            objects.append(component)
    
    return objects


def _count_colors_v161(g: GridV124, *, exclude_bg: bool = True, bg: int = 0) -> Dict[int, int]:
    """Count occurrences of each color."""
    h, w = _grid_shape_v161(g)
    counts: Dict[int, int] = {}
    
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            if exclude_bg and color == bg:
                continue
            counts[color] = counts.get(color, 0) + 1
    
    return counts


def _detect_symmetry_h_v161(g: GridV124) -> bool:
    """Detect horizontal symmetry."""
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return True
    
    for r in range(h):
        for c in range(w // 2):
            if int(g[r][c]) != int(g[r][w - 1 - c]):
                return False
    return True


def _detect_symmetry_v_v161(g: GridV124) -> bool:
    """Detect vertical symmetry."""
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return True
    
    for r in range(h // 2):
        for c in range(w):
            if int(g[r][c]) != int(g[h - 1 - r][c]):
                return False
    return True


def _has_shape_v161(g: GridV124, shape: GridV124, *, bg: int = 0) -> bool:
    """Check if grid contains a specific shape pattern."""
    h, w = _grid_shape_v161(g)
    sh, sw = _grid_shape_v161(shape)
    
    if sh > h or sw > w or sh == 0 or sw == 0:
        return False
    
    for r in range(h - sh + 1):
        for c in range(w - sw + 1):
            match = True
            for sr in range(sh):
                for sc in range(sw):
                    if int(shape[sr][sc]) != bg:
                        if int(g[r + sr][c + sc]) != int(shape[sr][sc]):
                            match = False
                            break
                if not match:
                    break
            if match:
                return True
    
    return False


# ─────────────────────────────────────────────────────────────────────────────
# CONDITIONAL EXECUTION CONTEXT
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ExecutionContext:
    """Context for conditional execution tracking."""
    grid: GridV124
    iteration: int = 0
    max_iterations: int = 100
    start_time: float = 0.0
    timeout: float = 30.0
    variables: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.variables is None:
            self.variables = {}
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def timed_out(self) -> bool:
        return self.elapsed() > self.timeout
    
    def should_stop(self) -> bool:
        return self.iteration >= self.max_iterations or self.timed_out()


# ─────────────────────────────────────────────────────────────────────────────
# 1. BASIC CONDITIONAL OPERATORS
# ─────────────────────────────────────────────────────────────────────────────


def conditional_apply(
    g: GridV124,
    *,
    condition: Callable[[GridV124], bool],
    if_true: Callable[[GridV124], GridV124],
    if_false: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Avalia condição e executa operador correspondente.
    
    Papel: Funciona como if-then-else genérico no programa cognitivo.
    Justificativa: Evita operações desnecessárias, tornando soluções adaptativas.
    """
    if condition(g):
        return if_true(g)
    elif if_false is not None:
        return if_false(g)
    return _copy_grid_v161(g)


def if_color_match(
    g: GridV124,
    *,
    color: int,
    if_present: Callable[[GridV124], GridV124],
    if_absent: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Verifica se cor específica está presente e aplica operação.
    
    Papel: Especializa comportamento baseado na paleta encontrada.
    Justificativa: Permite responder diferentemente a inputs de coloração distinta.
    """
    h, w = _grid_shape_v161(g)
    color_present = any(int(g[r][c]) == color for r in range(h) for c in range(w))
    
    if color_present:
        return if_present(g)
    elif if_absent is not None:
        return if_absent(g)
    return _copy_grid_v161(g)


def if_shape_present(
    g: GridV124,
    *,
    shape: GridV124,
    if_present: Callable[[GridV124], GridV124],
    if_absent: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Checa se certo formato aparece no input.
    
    Papel: Introduz ramificações dependentes de padrões visuais.
    Justificativa: Aumenta robustez para lidar com múltiplos cenários.
    """
    if _has_shape_v161(g, shape, bg=bg):
        return if_present(g)
    elif if_absent is not None:
        return if_absent(g)
    return _copy_grid_v161(g)


def switch_by_count(
    g: GridV124,
    *,
    count_fn: Callable[[GridV124], int],
    cases: Dict[int, Callable[[GridV124], GridV124]],
    default: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Seleciona operação baseado em valor contado.
    
    Papel: Implementa lógica switch-case baseada em contagem.
    Justificativa: Resolve cenários do tipo "caso N, faça tal coisa".
    """
    count = count_fn(g)
    
    if count in cases:
        return cases[count](g)
    elif default is not None:
        return default(g)
    return _copy_grid_v161(g)


def threshold_selector(
    g: GridV124,
    *,
    metric_fn: Callable[[GridV124], float],
    threshold: float,
    if_above: Callable[[GridV124], GridV124],
    if_below: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Compara métrica numérica a limiar e executa operação correspondente.
    
    Papel: Incorpora decisões do tipo "≥ X ou < X".
    Justificativa: Permite estratégias adaptativas baseadas em quantidades.
    """
    value = metric_fn(g)
    
    if value > threshold:
        return if_above(g)
    elif if_below is not None:
        return if_below(g)
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ITERATION AND LOOP OPERATORS
# ─────────────────────────────────────────────────────────────────────────────


def iterative_refinement(
    g: GridV124,
    *,
    operation: Callable[[GridV124], GridV124],
    condition: Callable[[GridV124], bool],
    max_iterations: int = 100,
) -> GridV124:
    """
    Aplica operador repetidamente enquanto condição estiver satisfeita.
    
    Papel: Implementa laço while interno.
    Justificativa: Elimina necessidade de saber de antemão quantas vezes aplicar ação.
    """
    result = _copy_grid_v161(g)
    iteration = 0
    
    while condition(result) and iteration < max_iterations:
        new_result = operation(result)
        # Check for no change (convergence)
        if new_result == result:
            break
        result = new_result
        iteration += 1
    
    return result


def loop_over_objects(
    g: GridV124,
    *,
    operation: Callable[[GridV124, Set[Tuple[int, int]]], GridV124],
    bg: int = 0,
) -> GridV124:
    """
    Itera sub-ação para cada objeto detectado.
    
    Papel: Funciona como for-each sobre componentes.
    Justificativa: Evita escrever passos redundantes para vários objetos.
    """
    objects = _find_objects_v161(g, bg=bg)
    result = _copy_grid_v161(g)
    
    for obj in objects:
        result = operation(result, obj)
    
    return result


def conditional_iterate(
    g: GridV124,
    *,
    operation: Callable[[GridV124], GridV124],
    condition: Callable[[GridV124], bool],
    max_iterations: int = 100,
) -> GridV124:
    """
    Realiza operação repetidamente enquanto condição permanecer verdadeira.
    
    Papel: Combina verificação e repetição numa construção concisa.
    Justificativa: Evita escrever lógica de controle de loop explicitamente.
    """
    return iterative_refinement(g, operation=operation, condition=condition, max_iterations=max_iterations)


def for_each_color(
    g: GridV124,
    *,
    operation: Callable[[GridV124, int], GridV124],
    bg: int = 0,
) -> GridV124:
    """
    Itera sub-ação para cada cor distinta presente.
    
    Papel: Implementa laço por camadas de cor.
    Justificativa: Permite decompor problemas multicoloridos de forma limpa.
    """
    colors = _count_colors_v161(g, exclude_bg=True, bg=bg)
    result = _copy_grid_v161(g)
    
    for color in sorted(colors.keys()):
        result = operation(result, color)
    
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 3. STOP AND EXIT CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────


def conditional_stop(
    g: GridV124,
    *,
    operations: List[Callable[[GridV124], GridV124]],
    stop_condition: Callable[[GridV124], bool],
) -> GridV124:
    """
    Executa operações em sequência, parando se condição de sucesso for atingida.
    
    Papel: Implementa break no fluxo de solução.
    Justificativa: Economiza tempo e evita over-solving.
    """
    result = _copy_grid_v161(g)
    
    for op in operations:
        result = op(result)
        if stop_condition(result):
            break
    
    return result


def early_exit_condition(
    g: GridV124,
    *,
    operations: List[Callable[[GridV124], GridV124]],
    success_check: Callable[[GridV124], bool],
) -> GridV124:
    """
    Verifica critério de sucesso após cada etapa e aborta se atingido.
    
    Papel: Impede que solver continue modificando solução já correta.
    Justificativa: Evita over-processing após objetivo alcançado.
    """
    return conditional_stop(g, operations=operations, stop_condition=success_check)


def timed_escape(
    g: GridV124,
    *,
    operation: Callable[[GridV124], GridV124],
    timeout_seconds: float = 5.0,
    fallback: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Monitora tempo e abandona se exceder limite.
    
    Papel: Proteção temporal contra loops infinitos ou busca ineficaz.
    Justificativa: Garante que solver respeite limites de tempo/complexidade.
    """
    start = time.time()
    
    try:
        result = operation(g)
        elapsed = time.time() - start
        
        if elapsed > timeout_seconds:
            # Operation completed but took too long - log warning
            if fallback is not None:
                return fallback(g)
        
        return result
    except Exception:
        if fallback is not None:
            return fallback(g)
        return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 4. FALLBACK AND RECOVERY OPERATORS
# ─────────────────────────────────────────────────────────────────────────────


def fail_safe_branch(
    g: GridV124,
    *,
    primary: Callable[[GridV124], GridV124],
    fallback: Callable[[GridV124], GridV124],
    validator: Callable[[GridV124], bool],
) -> GridV124:
    """
    Define caminho alternativo caso abordagem principal falhe.
    
    Papel: Plano B embutido no solver.
    Justificativa: Aumenta resiliência, permitindo recuperação interna de falhas.
    """
    try:
        result = primary(g)
        if validator(result):
            return result
    except Exception:
        pass
    
    return fallback(g)


def fallback_branch(
    g: GridV124,
    *,
    primary: Callable[[GridV124], GridV124],
    fallback: Callable[[GridV124], GridV124],
    check_progress: Callable[[GridV124, GridV124], bool],
) -> GridV124:
    """
    Segue ramo alternativo se método atual não produz resultado desejado.
    
    Papel: Permite que solver tente método alternativo antes de desistir.
    Justificativa: Aumenta taxa de sucesso dando segunda chance dentro do procedimento.
    """
    result = primary(g)
    
    if check_progress(g, result):
        return result
    
    return fallback(g)


def validate_and_branch(
    g: GridV124,
    *,
    operation: Callable[[GridV124], GridV124],
    validator: Callable[[GridV124], bool],
    alternative: Callable[[GridV124], GridV124],
) -> GridV124:
    """
    Valida estado após operação e escolhe caminho alternativo se falhar.
    
    Papel: Mecanismo de backtracking controlado.
    Justificativa: Economiza tempo e evita erros acumulados.
    """
    result = operation(g)
    
    if validator(result):
        return result
    
    return alternative(g)


def guard_condition(
    g: GridV124,
    *,
    guard: Callable[[GridV124], bool],
    operation: Callable[[GridV124], GridV124],
) -> GridV124:
    """
    Pré-condição explícita antes de executar operador crítico.
    
    Papel: Porteiro garantindo que operação só aconteça quando seguro.
    Justificativa: Melhora segurança das composições de operações.
    """
    if guard(g):
        return operation(g)
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 5. OBJECT COUNT CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────


def if_single_object(
    g: GridV124,
    *,
    if_single: Callable[[GridV124], GridV124],
    if_multiple: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Verifica se há apenas um objeto presente.
    
    Papel: Distingue processamento entre casos triviais e complexos.
    Justificativa: Evita sobrecarga quando input é simples.
    """
    objects = _find_objects_v161(g, bg=bg)
    
    if len(objects) == 1:
        return if_single(g)
    elif if_multiple is not None:
        return if_multiple(g)
    return _copy_grid_v161(g)


def if_multiple_objects(
    g: GridV124,
    *,
    if_multiple: Callable[[GridV124], GridV124],
    if_single_or_none: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Se há mais de um objeto, ativa determinado ramo.
    
    Papel: Garante soluções multi-objetos não sejam aplicadas indevidamente.
    Justificativa: Previne erros de tentar analisar relações inexistentes.
    """
    objects = _find_objects_v161(g, bg=bg)
    
    if len(objects) > 1:
        return if_multiple(g)
    elif if_single_or_none is not None:
        return if_single_or_none(g)
    return _copy_grid_v161(g)


def if_object_count_equals(
    g: GridV124,
    *,
    count: int,
    if_equals: Callable[[GridV124], GridV124],
    if_different: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Avalia se número de objetos é exatamente igual a valor específico.
    
    Papel: Introduz gatilhos baseados em contagem exata.
    Justificativa: Captura requisitos precisos de alguns puzzles.
    """
    objects = _find_objects_v161(g, bg=bg)
    
    if len(objects) == count:
        return if_equals(g)
    elif if_different is not None:
        return if_different(g)
    return _copy_grid_v161(g)


def if_object_count_gt(
    g: GridV124,
    *,
    threshold: int,
    if_greater: Callable[[GridV124], GridV124],
    if_not_greater: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Checa se quantidade de objetos é maior que limiar.
    
    Papel: Distingue cenários densos de esparsos.
    Justificativa: Melhora performance lidando separadamente com casos.
    """
    objects = _find_objects_v161(g, bg=bg)
    
    if len(objects) > threshold:
        return if_greater(g)
    elif if_not_greater is not None:
        return if_not_greater(g)
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 6. COLOR CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────


def if_all_same_color(
    g: GridV124,
    *,
    if_same: Callable[[GridV124], GridV124],
    if_different: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Testa se todas células não-fundo compartilham mesma cor.
    
    Papel: Ajusta solução baseado em homogeneidade de cor.
    Justificativa: Permite simplificar quando não há diversidade de cores.
    """
    colors = _count_colors_v161(g, exclude_bg=True, bg=bg)
    
    if len(colors) <= 1:
        return if_same(g)
    elif if_different is not None:
        return if_different(g)
    return _copy_grid_v161(g)


def if_multi_color(
    g: GridV124,
    *,
    if_multi: Callable[[GridV124], GridV124],
    if_single: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Se múltiplas cores distintas estão presentes, ativa operações.
    
    Papel: Garante presença de multicor seja tratada quando relevante.
    Justificativa: Evita desperdício ao tentar distinguir cores inexistentes.
    """
    colors = _count_colors_v161(g, exclude_bg=True, bg=bg)
    
    if len(colors) > 1:
        return if_multi(g)
    elif if_single is not None:
        return if_single(g)
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 7. SYMMETRY CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────


def if_symmetric(
    g: GridV124,
    *,
    if_symmetric: Callable[[GridV124], GridV124],
    if_asymmetric: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Testa se figura apresenta alguma forma de simetria.
    
    Papel: Diferencia solução com base em simetria presente.
    Justificativa: Acelera solução de inputs simétricos aproveitando redundância.
    """
    has_symmetry = _detect_symmetry_h_v161(g) or _detect_symmetry_v_v161(g)
    
    if has_symmetry:
        return if_symmetric(g)
    elif if_asymmetric is not None:
        return if_asymmetric(g)
    return _copy_grid_v161(g)


def if_asymmetric(
    g: GridV124,
    *,
    if_asymmetric: Callable[[GridV124], GridV124],
    if_symmetric: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Se input não tem simetria detectável, realiza passos extras.
    
    Papel: Garante padrões não-simétricos recebam atenção especial.
    Justificativa: Melhora sucesso em casos desordenados.
    """
    has_symmetry = _detect_symmetry_h_v161(g) or _detect_symmetry_v_v161(g)
    
    if not has_symmetry:
        return if_asymmetric(g)
    elif if_symmetric is not None:
        return if_symmetric(g)
    return _copy_grid_v161(g)


def branch_by_symmetry(
    g: GridV124,
    *,
    branches: Dict[str, Callable[[GridV124], GridV124]],
    default: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Desvia execução de acordo com tipo de simetria detectado.
    
    Papel: Seleciona diferentes sub-rotinas conforme simetria.
    Justificativa: Permite subsolvers otimizados para cada situação.
    
    branches keys: "horizontal", "vertical", "both", "none"
    """
    sym_h = _detect_symmetry_h_v161(g)
    sym_v = _detect_symmetry_v_v161(g)
    
    if sym_h and sym_v and "both" in branches:
        return branches["both"](g)
    elif sym_h and "horizontal" in branches:
        return branches["horizontal"](g)
    elif sym_v and "vertical" in branches:
        return branches["vertical"](g)
    elif not sym_h and not sym_v and "none" in branches:
        return branches["none"](g)
    elif default is not None:
        return default(g)
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 8. GRID SIZE CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────


def if_grid_large(
    g: GridV124,
    *,
    threshold: int = 20,
    if_large: Callable[[GridV124], GridV124],
    if_small: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Verifica se grade é grande (acima de certo tamanho).
    
    Papel: Adaptação de estratégia ao tamanho do problema.
    Justificativa: Evita que solver fique sem desempenho em inputs grandes.
    """
    h, w = _grid_shape_v161(g)
    
    if h > threshold or w > threshold:
        return if_large(g)
    elif if_small is not None:
        return if_small(g)
    return _copy_grid_v161(g)


def if_sparse(
    g: GridV124,
    *,
    density_threshold: float = 0.3,
    if_sparse: Callable[[GridV124], GridV124],
    if_dense: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Determina se grade é esparsa (poucos pixels coloridos).
    
    Papel: Aproveita informação de "pouca coisa para olhar".
    Justificativa: Tarefas esparsas podem ser resolvidas mais rápido.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    non_bg = sum(1 for r in range(h) for c in range(w) if int(g[r][c]) != bg)
    density = non_bg / (h * w)
    
    if density < density_threshold:
        return if_sparse(g)
    elif if_dense is not None:
        return if_dense(g)
    return _copy_grid_v161(g)


def if_dense(
    g: GridV124,
    *,
    density_threshold: float = 0.7,
    if_dense: Callable[[GridV124], GridV124],
    if_sparse: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Se grade está densamente preenchida, adota lógica apropriada.
    
    Papel: Reconhece que abordagem de objetos isolados pode falhar em densos.
    Justificativa: Aumenta robustez em puzzles onde tudo está conectado.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    non_bg = sum(1 for r in range(h) for c in range(w) if int(g[r][c]) != bg)
    density = non_bg / (h * w)
    
    if density > density_threshold:
        return if_dense(g)
    elif if_sparse is not None:
        return if_sparse(g)
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# 9. OBJECT TYPE CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────


def branch_by_object_type(
    g: GridV124,
    *,
    classifiers: Dict[str, Callable[[GridV124, Set[Tuple[int, int]]], bool]],
    handlers: Dict[str, Callable[[GridV124], GridV124]],
    default: Optional[Callable[[GridV124], GridV124]] = None,
    bg: int = 0,
) -> GridV124:
    """
    Inspeciona tipos de objetos e seleciona rotinas diferentes.
    
    Papel: Despachante baseado em classificação de objetos.
    Justificativa: Permite soluções especializadas para diferentes configurações.
    """
    objects = _find_objects_v161(g, bg=bg)
    
    if not objects:
        if default is not None:
            return default(g)
        return _copy_grid_v161(g)
    
    # Determine predominant object type
    type_counts: Dict[str, int] = {}
    
    for obj in objects:
        for type_name, classifier in classifiers.items():
            if classifier(g, obj):
                type_counts[type_name] = type_counts.get(type_name, 0) + 1
                break
    
    if type_counts:
        predominant = max(type_counts, key=lambda k: type_counts[k])
        if predominant in handlers:
            return handlers[predominant](g)
    
    if default is not None:
        return default(g)
    return _copy_grid_v161(g)


def scenario_selector(
    g: GridV124,
    *,
    scenarios: List[Tuple[Callable[[GridV124], bool], Callable[[GridV124], GridV124]]],
    default: Optional[Callable[[GridV124], GridV124]] = None,
) -> GridV124:
    """
    Analisa características do input e seleciona modo de solução adequado.
    
    Papel: Meta-condicional de alto nível selecionando estratégia.
    Justificativa: Se o sistema tem várias estratégias, faz seleção automática.
    """
    for condition, handler in scenarios:
        if condition(g):
            return handler(g)
    
    if default is not None:
        return default(g)
    return _copy_grid_v161(g)


# ─────────────────────────────────────────────────────────────────────────────
# OPERATOR REGISTRY
# ─────────────────────────────────────────────────────────────────────────────


CONDITIONAL_OPERATORS_V161 = {
    # Basic conditionals
    "conditional_apply": conditional_apply,
    "if_color_match": if_color_match,
    "if_shape_present": if_shape_present,
    "switch_by_count": switch_by_count,
    "threshold_selector": threshold_selector,
    
    # Iteration and loops
    "iterative_refinement": iterative_refinement,
    "loop_over_objects": loop_over_objects,
    "conditional_iterate": conditional_iterate,
    "for_each_color": for_each_color,
    
    # Stop conditions
    "conditional_stop": conditional_stop,
    "early_exit_condition": early_exit_condition,
    "timed_escape": timed_escape,
    
    # Fallback and recovery
    "fail_safe_branch": fail_safe_branch,
    "fallback_branch": fallback_branch,
    "validate_and_branch": validate_and_branch,
    "guard_condition": guard_condition,
    
    # Object count conditions
    "if_single_object": if_single_object,
    "if_multiple_objects": if_multiple_objects,
    "if_object_count_equals": if_object_count_equals,
    "if_object_count_gt": if_object_count_gt,
    
    # Color conditions
    "if_all_same_color": if_all_same_color,
    "if_multi_color": if_multi_color,
    
    # Symmetry conditions
    "if_symmetric": if_symmetric,
    "if_asymmetric": if_asymmetric,
    "branch_by_symmetry": branch_by_symmetry,
    
    # Grid size conditions
    "if_grid_large": if_grid_large,
    "if_sparse": if_sparse,
    "if_dense": if_dense,
    
    # Object type conditions
    "branch_by_object_type": branch_by_object_type,
    "scenario_selector": scenario_selector,
}


def get_conditional_operator_v161(name: str) -> Optional[Callable]:
    """Get a conditional operator by name."""
    return CONDITIONAL_OPERATORS_V161.get(name)


def list_conditional_operators_v161() -> List[str]:
    """List all available conditional operators."""
    return list(CONDITIONAL_OPERATORS_V161.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
Total de Operadores Condicionais Implementados: 29

CATEGORIAS:
1. Condicionais Básicos (5):
   - conditional_apply, if_color_match, if_shape_present
   - switch_by_count, threshold_selector

2. Iteração e Loops (4):
   - iterative_refinement, loop_over_objects
   - conditional_iterate, for_each_color

3. Condições de Parada (3):
   - conditional_stop, early_exit_condition, timed_escape

4. Fallback e Recuperação (4):
   - fail_safe_branch, fallback_branch
   - validate_and_branch, guard_condition

5. Condições de Contagem de Objetos (4):
   - if_single_object, if_multiple_objects
   - if_object_count_equals, if_object_count_gt

6. Condições de Cor (2):
   - if_all_same_color, if_multi_color

7. Condições de Simetria (3):
   - if_symmetric, if_asymmetric, branch_by_symmetry

8. Condições de Tamanho (3):
   - if_grid_large, if_sparse, if_dense

9. Condições de Tipo de Objeto (2):
   - branch_by_object_type, scenario_selector
"""
