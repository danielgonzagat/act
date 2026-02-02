"""
arc_solver_v161.py - Solver ARC com Operadores V161 Integrados

Este módulo estende o solver V141 adicionando os novos operadores V161
ao espaço de busca, aumentando significativamente a cobertura de padrões ARC.

Principais melhorias:
1. Adiciona 31 novos operadores ao pool de candidatos
2. Cria heurísticas para detectar quando usar operadores V161
3. Gera concept_templates automáticos a partir dos operadores V161

Schema version: 161
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .arc_ops_v132 import StateV132
from .arc_ops_v161 import (
    OP_DEFS_V161,
    apply_op_v161,
    step_cost_bits_v161,
    list_v161_operators,
)
from .arc_solver_v141 import (
    solve_arc_task_v141,
    SolveConfigV141,
    ProgramStepV141,
)
from .grid_v124 import (
    GridV124,
    grid_from_list_v124,
    grid_shape_v124,
    grid_equal_v124,
    unique_colors_v124,
)

ARC_SOLVER_SCHEMA_VERSION_V161 = 161


# ─────────────────────────────────────────────────────────────────────────────
# HEURÍSTICAS PARA DETECÇÃO DE PADRÕES V161
# ─────────────────────────────────────────────────────────────────────────────


def _detect_symmetry_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve simetria."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        
        # Simetria 4-way: output é 2x input
        if ho == hi * 2 and wo == wi * 2:
            ops.append("symmetry_4way")
        
        # Simetria diagonal: output é transposto
        if ho == wi and wo == hi and ho != hi:
            ops.append("symmetry_diagonal_main")
            ops.append("symmetry_diagonal_anti")
        
        # Mirror extend
        if ho == hi and wo == wi * 2:
            ops.append("mirror_extend_h")
        if ho == hi * 2 and wo == wi:
            ops.append("mirror_extend_v")
    
    return list(set(ops))


def _detect_scale_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve escala."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        
        if hi > 0 and wi > 0:
            if ho == hi * 2 and wo == wi * 2:
                ops.append("scale_2x")
            elif ho == hi * 3 and wo == wi * 3:
                ops.append("scale_3x")
            elif hi == ho * 2 and wi == wo * 2:
                ops.append("scale_half")
    
    return list(set(ops))


def _detect_crop_pad_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve crop ou padding."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        
        # Crop: output menor
        if ho < hi or wo < wi:
            ops.append("crop_margin")
            ops.append("extract_core")
        
        # Pad: output maior
        if ho > hi or wo > wi:
            ops.append("pad_uniform")
    
    return list(set(ops))


def _detect_tiling_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve tiling/repetição."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        
        # Tiling: output é múltiplo do input
        if hi > 0 and wi > 0 and ho > hi and wo > wi:
            if ho % hi == 0 and wo % wi == 0:
                ops.append("tile_to_size")
        
        # Repeating unit: output menor e se repete
        if ho < hi and wo < wi:
            ops.append("find_repeating_unit")
    
    return list(set(ops))


def _detect_object_selection_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve seleção de objetos."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        # Se output tem menos células coloridas, provavelmente selecionou um objeto
        in_colors = unique_colors_v124(inp)
        out_colors = unique_colors_v124(out)
        
        if len(out_colors) < len(in_colors):
            ops.extend(["select_largest_obj", "select_smallest_obj", "remove_noise"])
    
    return list(set(ops))


def _detect_color_manipulation_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve manipulação de cor."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        in_colors = set(unique_colors_v124(inp))
        out_colors = set(unique_colors_v124(out))
        
        # Se as cores mudam mas a forma é a mesma
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        
        if hi == ho and wi == wo:
            if in_colors != out_colors:
                ops.extend(["color_swap", "color_normalize"])
            
            # Se número de cores diferentes
            if len(out_colors) < len(in_colors):
                ops.append("color_to_most_frequent")
    
    return list(set(ops))


def _detect_connectivity_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve conectividade."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        
        if hi == ho and wi == wo:
            # Conta células coloridas
            in_count = sum(1 for r in range(hi) for c in range(wi) if inp[r][c] != 0)
            out_count = sum(1 for r in range(ho) for c in range(wo) if out[r][c] != 0)
            
            # Se output tem mais células, provavelmente conectou algo
            if out_count > in_count:
                ops.extend(["connect_same_color_h", "connect_same_color_v", "fill_holes"])
    
    return list(set(ops))


def _detect_frame_task(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Detecta se a tarefa envolve frames/bordas."""
    ops: List[str] = []
    
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        
        # Se output é menor e mesmo formato
        if ho == hi - 2 and wo == wi - 2:
            ops.append("extract_core")
        
        # Frame detection
        if hi == ho and wi == wo:
            ops.append("find_frame")
    
    return list(set(ops))


def get_v161_candidate_ops(train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[str]:
    """Retorna operadores V161 candidatos baseado nas características da tarefa."""
    candidates: List[str] = []
    
    candidates.extend(_detect_symmetry_task(train_pairs))
    candidates.extend(_detect_scale_task(train_pairs))
    candidates.extend(_detect_crop_pad_task(train_pairs))
    candidates.extend(_detect_tiling_task(train_pairs))
    candidates.extend(_detect_object_selection_task(train_pairs))
    candidates.extend(_detect_color_manipulation_task(train_pairs))
    candidates.extend(_detect_connectivity_task(train_pairs))
    candidates.extend(_detect_frame_task(train_pairs))
    
    return list(set(candidates))


# ─────────────────────────────────────────────────────────────────────────────
# GERAÇÃO DE CONCEPT TEMPLATES V161
# ─────────────────────────────────────────────────────────────────────────────


def generate_v161_concept_templates(
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    bg_candidates: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Gera concept_templates para os operadores V161 relevantes.
    """
    templates: List[Dict[str, Any]] = []
    
    if bg_candidates is None:
        bg_candidates = [0]
    
    # Detecta operadores candidatos
    candidate_ops = get_v161_candidate_ops(train_pairs)
    
    # Adiciona sempre alguns operadores base
    all_ops = list(set(candidate_ops + [
        "crop_margin", "extract_core", "color_normalize",
        "select_largest_obj", "remove_noise"
    ]))
    
    # Gera templates para cada operador
    for op in all_ops:
        if op not in OP_DEFS_V161:
            continue
        
        if op in ["crop_margin", "select_largest_obj", "select_smallest_obj", 
                  "remove_noise", "fill_holes", "connect_same_color_h", 
                  "connect_same_color_v", "find_frame"]:
            for bg in bg_candidates[:2]:
                templates.append({
                    "concept_id": f"v161_{op}_bg{bg}",
                    "steps": [{"op_id": op, "args": {"bg": bg}}],
                    "support": 1,
                })
        elif op in ["color_swap"]:
            for c1 in range(1, 4):
                for c2 in range(c1 + 1, 5):
                    templates.append({
                        "concept_id": f"v161_{op}_{c1}_{c2}",
                        "steps": [{"op_id": op, "args": {"c1": c1, "c2": c2}}],
                        "support": 1,
                    })
        elif op in ["scale_2x", "scale_3x", "scale_half", "symmetry_4way",
                    "symmetry_8way", "symmetry_diagonal_main", "symmetry_diagonal_anti",
                    "mirror_extend_h", "mirror_extend_v", "color_normalize",
                    "color_invert", "extract_core", "find_repeating_unit",
                    "sort_rows_by_sum", "sort_cols_by_sum"]:
            templates.append({
                "concept_id": f"v161_{op}",
                "steps": [{"op_id": op, "args": {}}],
                "support": 1,
            })
        elif op == "pad_uniform":
            for pad in [1, 2]:
                for color in bg_candidates[:2]:
                    templates.append({
                        "concept_id": f"v161_{op}_p{pad}_c{color}",
                        "steps": [{"op_id": op, "args": {"pad": pad, "color": color}}],
                        "support": 1,
                    })
    
    return templates


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER DIRETO V161
# ─────────────────────────────────────────────────────────────────────────────


def _try_single_op_v161(
    op_id: str,
    args: Dict[str, Any],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
) -> Optional[GridV124]:
    """Tenta um único operador V161."""
    try:
        for inp, expected_out in train_pairs:
            state = StateV132(grid=inp)
            result_state = apply_op_v161(state=state, op_id=op_id, args=args)
            result_grid = result_state.grid
            
            if not grid_equal_v124(result_grid, expected_out):
                return None
        
        test_state = StateV132(grid=test_in)
        test_result = apply_op_v161(state=test_state, op_id=op_id, args=args)
        return test_result.grid
    
    except Exception:
        return None


def _try_composite_v161(
    steps: List[Dict[str, Any]],
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
) -> Optional[GridV124]:
    """Tenta uma composição de operadores V161."""
    try:
        for inp, expected_out in train_pairs:
            state = StateV132(grid=inp)
            for step in steps:
                state = apply_op_v161(state=state, op_id=step["op_id"], args=step.get("args", {}))
            
            if not grid_equal_v124(state.grid, expected_out):
                return None
        
        test_state = StateV132(grid=test_in)
        for step in steps:
            test_state = apply_op_v161(state=test_state, op_id=step["op_id"], args=step.get("args", {}))
        return test_state.grid
    
    except Exception:
        return None


def solve_direct_v161(
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
) -> Optional[Dict[str, Any]]:
    """Tenta resolver diretamente usando operadores V161."""
    from collections import Counter
    
    # Determina backgrounds prováveis
    bg_candidates = [0]
    for inp, _ in train_pairs:
        h, w = grid_shape_v124(inp)
        all_cells = [inp[r][c] for r in range(h) for c in range(w)]
        counter = Counter(all_cells)
        bg_candidates = [c for c, _ in counter.most_common(2)]
        break
    
    # Detecta candidatos
    candidates = get_v161_candidate_ops(train_pairs)
    all_ops = list(set(candidates + list_v161_operators()))
    
    # 1. Tenta operadores simples
    for op in all_ops:
        arg_variations: List[Dict[str, Any]] = [{}]
        
        if op in ["crop_margin", "select_largest_obj", "select_smallest_obj",
                  "fill_holes", "connect_same_color_h", "connect_same_color_v", "find_frame"]:
            arg_variations = [{"bg": bg} for bg in bg_candidates]
        
        if op == "remove_noise":
            arg_variations = [{"bg": bg, "min_size": ms} for bg in bg_candidates for ms in [2, 3, 4]]
        
        if op == "color_swap":
            arg_variations = [{"c1": c1, "c2": c2} for c1 in range(1, 6) for c2 in range(c1+1, 7)]
        
        if op == "pad_uniform":
            arg_variations = [{"pad": p, "color": c} for p in [1, 2] for c in bg_candidates]
        
        for args in arg_variations:
            result = _try_single_op_v161(op, args, train_pairs, test_in)
            if result is not None:
                return {
                    "status": "SOLVED",
                    "predicted_grid": result,
                    "program_steps": [{"op_id": op, "args": args}],
                    "method": "v161_direct_single",
                }
    
    # 2. Tenta composições pré-definidas
    composites = [
        [{"op_id": "crop_margin", "args": {"bg": 0}}, {"op_id": "scale_2x", "args": {}}],
        [{"op_id": "extract_core", "args": {}}, {"op_id": "mirror_extend_h", "args": {}}],
        [{"op_id": "extract_core", "args": {}}, {"op_id": "mirror_extend_v", "args": {}}],
        [{"op_id": "select_largest_obj", "args": {"bg": 0}}, {"op_id": "crop_margin", "args": {"bg": 0}}],
        [{"op_id": "remove_noise", "args": {"bg": 0, "min_size": 3}}, {"op_id": "crop_margin", "args": {"bg": 0}}],
        [{"op_id": "color_normalize", "args": {}}, {"op_id": "scale_2x", "args": {}}],
        [{"op_id": "crop_margin", "args": {"bg": 0}}, {"op_id": "symmetry_4way", "args": {}}],
        [{"op_id": "find_repeating_unit", "args": {}}, {"op_id": "scale_2x", "args": {}}],
    ]
    
    for steps in composites:
        result = _try_composite_v161(steps, train_pairs, test_in)
        if result is not None:
            return {
                "status": "SOLVED",
                "predicted_grid": result,
                "program_steps": steps,
                "method": "v161_direct_composite",
            }
    
    # 3. Tenta combinações de 2 operadores
    simple_ops = ["crop_margin", "extract_core", "color_normalize", 
                  "select_largest_obj", "remove_noise", "scale_2x", "scale_half",
                  "mirror_extend_h", "mirror_extend_v", "symmetry_4way"]
    
    for op1 in simple_ops:
        for op2 in simple_ops:
            if op1 == op2:
                continue
            
            args1: Dict[str, Any] = {}
            args2: Dict[str, Any] = {}
            
            if op1 in ["crop_margin", "select_largest_obj", "remove_noise"]:
                args1 = {"bg": 0}
            if op2 in ["crop_margin", "select_largest_obj", "remove_noise"]:
                args2 = {"bg": 0}
            if op1 == "remove_noise":
                args1["min_size"] = 3
            if op2 == "remove_noise":
                args2["min_size"] = 3
            
            steps = [
                {"op_id": op1, "args": args1},
                {"op_id": op2, "args": args2},
            ]
            
            result = _try_composite_v161(steps, train_pairs, test_in)
            if result is not None:
                return {
                    "status": "SOLVED",
                    "predicted_grid": result,
                    "program_steps": steps,
                    "method": "v161_direct_pair",
                }
    
    return None


# ─────────────────────────────────────────────────────────────────────────────
# SOLVER HÍBRIDO V161
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SolveConfigV161:
    """Configuração para o solver V161."""
    max_depth: int = 6
    max_programs: int = 8000
    try_v161_first: bool = True
    concept_templates: Tuple[Dict[str, Any], ...] = ()


def solve_arc_task_v161(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    test_in: GridV124,
    config: Optional[SolveConfigV161] = None,
) -> Dict[str, Any]:
    """Solver híbrido que tenta V161 primeiro, depois V141."""
    from collections import Counter
    
    if config is None:
        config = SolveConfigV161()
    
    # 1. Tenta V161 direto primeiro
    if config.try_v161_first:
        v161_result = solve_direct_v161(train_pairs, test_in)
        if v161_result is not None:
            return v161_result
    
    # 2. Gera concept_templates V161
    bg_candidates = [0]
    for inp, _ in train_pairs:
        h, w = grid_shape_v124(inp)
        all_cells = [inp[r][c] for r in range(h) for c in range(w)]
        counter = Counter(all_cells)
        bg_candidates = [c for c, _ in counter.most_common(2)]
        break
    
    v161_templates = generate_v161_concept_templates(train_pairs, bg_candidates)
    all_templates = list(config.concept_templates) + v161_templates
    
    # 3. Usa solver V141 com templates injetados
    v141_config = SolveConfigV141(
        max_depth=config.max_depth,
        max_programs=config.max_programs,
        concept_templates=tuple(all_templates),
    )
    
    result = solve_arc_task_v141(
        train_pairs=train_pairs,
        test_in=test_in,
        config=v141_config,
    )
    
    return result


__all__ = [
    "ARC_SOLVER_SCHEMA_VERSION_V161",
    "SolveConfigV161",
    "solve_arc_task_v161",
    "solve_direct_v161",
    "get_v161_candidate_ops",
    "generate_v161_concept_templates",
]
