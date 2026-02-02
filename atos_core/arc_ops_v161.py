"""
arc_ops_v161.py - Operadores ARC Estendidos V161

Este módulo estende arc_ops_v141 com novos operadores de alto impacto
baseados na especificação dos atos cognitivos V161.

Novos operadores adicionados:
- Simetrias (diagonal, 4-way, 8-way)
- Padrões (checker, stripe, gradient)
- Composições de cor (swap, invert, normalize)
- Transformações espaciais (scale, crop_margin, pad_uniform)
- Detecção de estrutura (find_frame, extract_core, find_repeating_unit)

Schema version: 161
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional, Set, Tuple

from .arc_objects_v132 import BBoxV132, ObjectSetV132, ObjectV132
from .arc_ops_v132 import OpDefV132, StateV132
from .arc_ops_v141 import (
    OP_DEFS_V141, 
    apply_op_v141, 
    step_cost_bits_v141,
    _check_color_v141,
    connected_components4_nonbg_multicolor_v141,
)
from .grid_v124 import (
    GridV124,
    grid_from_list_v124,
    grid_shape_v124,
    reflect_h_v124,
    reflect_v_v124,
    rotate180_v124,
    rotate270_v124,
    rotate90_v124,
    translate_v124,
    grid_equal_v124,
)

ARC_OPS_SCHEMA_VERSION_V161 = 161


# ─────────────────────────────────────────────────────────────────────────────
# OPERADOR DEFINITIONS V161
# ─────────────────────────────────────────────────────────────────────────────

OP_DEFS_V161 = dict(OP_DEFS_V141)

# Simetrias
OP_DEFS_V161["symmetry_diagonal_main"] = OpDefV132(
    op_id="symmetry_diagonal_main", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["symmetry_diagonal_anti"] = OpDefV132(
    op_id="symmetry_diagonal_anti", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["symmetry_4way"] = OpDefV132(
    op_id="symmetry_4way", reads=("grid",), writes=("grid",), base_cost_bits=18
)
OP_DEFS_V161["symmetry_8way"] = OpDefV132(
    op_id="symmetry_8way", reads=("grid",), writes=("grid",), base_cost_bits=20
)

# Padrões
OP_DEFS_V161["fill_checker"] = OpDefV132(
    op_id="fill_checker", reads=("grid",), writes=("grid",), base_cost_bits=16
)
OP_DEFS_V161["fill_stripes_h"] = OpDefV132(
    op_id="fill_stripes_h", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["fill_stripes_v"] = OpDefV132(
    op_id="fill_stripes_v", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["fill_gradient_h"] = OpDefV132(
    op_id="fill_gradient_h", reads=("grid",), writes=("grid",), base_cost_bits=18
)
OP_DEFS_V161["fill_gradient_v"] = OpDefV132(
    op_id="fill_gradient_v", reads=("grid",), writes=("grid",), base_cost_bits=18
)

# Manipulação de Cor
OP_DEFS_V161["color_swap"] = OpDefV132(
    op_id="color_swap", reads=("grid",), writes=("grid",), base_cost_bits=12
)
OP_DEFS_V161["color_invert"] = OpDefV132(
    op_id="color_invert", reads=("grid",), writes=("grid",), base_cost_bits=10
)
OP_DEFS_V161["color_normalize"] = OpDefV132(
    op_id="color_normalize", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["color_to_most_frequent"] = OpDefV132(
    op_id="color_to_most_frequent", reads=("grid",), writes=("grid",), base_cost_bits=16
)
OP_DEFS_V161["color_unique_to_bg"] = OpDefV132(
    op_id="color_unique_to_bg", reads=("grid",), writes=("grid",), base_cost_bits=16
)

# Transformações Espaciais
OP_DEFS_V161["scale_2x"] = OpDefV132(
    op_id="scale_2x", reads=("grid",), writes=("grid",), base_cost_bits=12
)
OP_DEFS_V161["scale_3x"] = OpDefV132(
    op_id="scale_3x", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["scale_half"] = OpDefV132(
    op_id="scale_half", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["crop_margin"] = OpDefV132(
    op_id="crop_margin", reads=("grid",), writes=("grid",), base_cost_bits=12
)
OP_DEFS_V161["pad_uniform"] = OpDefV132(
    op_id="pad_uniform", reads=("grid",), writes=("grid",), base_cost_bits=14
)

# Detecção de Estrutura
OP_DEFS_V161["find_frame"] = OpDefV132(
    op_id="find_frame", reads=("grid",), writes=("grid",), base_cost_bits=18
)
OP_DEFS_V161["extract_core"] = OpDefV132(
    op_id="extract_core", reads=("grid",), writes=("grid",), base_cost_bits=16
)
OP_DEFS_V161["find_repeating_unit"] = OpDefV132(
    op_id="find_repeating_unit", reads=("grid",), writes=("grid",), base_cost_bits=22
)
OP_DEFS_V161["tile_to_size"] = OpDefV132(
    op_id="tile_to_size", reads=("grid",), writes=("grid",), base_cost_bits=18
)

# Operações em Objetos
OP_DEFS_V161["select_largest_obj"] = OpDefV132(
    op_id="select_largest_obj", reads=("grid",), writes=("grid",), base_cost_bits=18
)
OP_DEFS_V161["select_smallest_obj"] = OpDefV132(
    op_id="select_smallest_obj", reads=("grid",), writes=("grid",), base_cost_bits=18
)
OP_DEFS_V161["remove_noise"] = OpDefV132(
    op_id="remove_noise", reads=("grid",), writes=("grid",), base_cost_bits=16
)
OP_DEFS_V161["fill_holes"] = OpDefV132(
    op_id="fill_holes", reads=("grid",), writes=("grid",), base_cost_bits=18
)

# Contagem e Análise
OP_DEFS_V161["count_colors_grid"] = OpDefV132(
    op_id="count_colors_grid", reads=("grid",), writes=("grid",), base_cost_bits=20
)
OP_DEFS_V161["sort_rows_by_sum"] = OpDefV132(
    op_id="sort_rows_by_sum", reads=("grid",), writes=("grid",), base_cost_bits=18
)
OP_DEFS_V161["sort_cols_by_sum"] = OpDefV132(
    op_id="sort_cols_by_sum", reads=("grid",), writes=("grid",), base_cost_bits=18
)

# Conectividade
OP_DEFS_V161["connect_same_color_h"] = OpDefV132(
    op_id="connect_same_color_h", reads=("grid",), writes=("grid",), base_cost_bits=16
)
OP_DEFS_V161["connect_same_color_v"] = OpDefV132(
    op_id="connect_same_color_v", reads=("grid",), writes=("grid",), base_cost_bits=16
)
OP_DEFS_V161["connect_same_color_diag"] = OpDefV132(
    op_id="connect_same_color_diag", reads=("grid",), writes=("grid",), base_cost_bits=18
)

# Máscaras Avançadas
OP_DEFS_V161["mask_largest_component"] = OpDefV132(
    op_id="mask_largest_component", reads=("grid",), writes=("patch",), base_cost_bits=18
)
OP_DEFS_V161["mask_border_cells"] = OpDefV132(
    op_id="mask_border_cells", reads=("grid",), writes=("patch",), base_cost_bits=14
)
OP_DEFS_V161["mask_corner_cells"] = OpDefV132(
    op_id="mask_corner_cells", reads=("grid",), writes=("patch",), base_cost_bits=14
)

# Cópias e Espelhamentos
OP_DEFS_V161["copy_quadrant"] = OpDefV132(
    op_id="copy_quadrant", reads=("grid",), writes=("grid",), base_cost_bits=16
)
OP_DEFS_V161["mirror_extend_h"] = OpDefV132(
    op_id="mirror_extend_h", reads=("grid",), writes=("grid",), base_cost_bits=14
)
OP_DEFS_V161["mirror_extend_v"] = OpDefV132(
    op_id="mirror_extend_v", reads=("grid",), writes=("grid",), base_cost_bits=14
)


# ─────────────────────────────────────────────────────────────────────────────
# IMPLEMENTAÇÕES DOS OPERADORES V161
# ─────────────────────────────────────────────────────────────────────────────


def _get_color_counts(g: GridV124) -> Dict[int, int]:
    """Conta ocorrências de cada cor na grade."""
    counts: Dict[int, int] = {}
    h, w = grid_shape_v124(g)
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            counts[v] = counts.get(v, 0) + 1
    return counts


def _most_frequent_color(g: GridV124, exclude_bg: bool = True, bg: int = 0) -> int:
    """Retorna a cor mais frequente."""
    counts = _get_color_counts(g)
    if exclude_bg and bg in counts:
        del counts[bg]
    if not counts:
        return bg
    return max(counts.keys(), key=lambda c: (counts[c], -c))


def _apply_symmetry_diagonal_main(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Reflete pela diagonal principal (r,c) -> (c,r)."""
    g = state.grid
    h, w = grid_shape_v124(g)
    size = max(h, w)
    out = [[0 for _ in range(size)] for _ in range(size)]
    for r in range(h):
        for c in range(w):
            out[c][r] = int(g[r][c])
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_symmetry_diagonal_anti(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Reflete pela diagonal anti (r,c) -> (w-1-c, h-1-r)."""
    g = state.grid
    h, w = grid_shape_v124(g)
    out = [[0 for _ in range(h)] for _ in range(w)]
    for r in range(h):
        for c in range(w):
            out[w - 1 - c][h - 1 - r] = int(g[r][c])
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_symmetry_4way(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Aplica simetria 4-way: combina reflexões H e V."""
    g = state.grid
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return state
    
    # Duplica a grade em ambas direções
    out_h = h * 2
    out_w = w * 2
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            # Quadrante superior esquerdo (original)
            out[r][c] = v
            # Quadrante superior direito (reflexão H)
            out[r][out_w - 1 - c] = v
            # Quadrante inferior esquerdo (reflexão V)
            out[out_h - 1 - r][c] = v
            # Quadrante inferior direito (reflexão HV)
            out[out_h - 1 - r][out_w - 1 - c] = v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_symmetry_8way(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Aplica simetria 8-way (todas as simetrias possíveis)."""
    g = state.grid
    h, w = grid_shape_v124(g)
    if h == 0 or w == 0:
        return state
    
    # Para simetria 8-way, precisa ser quadrada
    size = max(h, w)
    out = [[0 for _ in range(size * 2)] for _ in range(size * 2)]
    
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            # 8 posições simétricas
            positions = [
                (r, c),
                (r, size * 2 - 1 - c),
                (size * 2 - 1 - r, c),
                (size * 2 - 1 - r, size * 2 - 1 - c),
                (c, r),
                (c, size * 2 - 1 - r),
                (size * 2 - 1 - c, r),
                (size * 2 - 1 - c, size * 2 - 1 - r),
            ]
            for pr, pc in positions:
                if 0 <= pr < size * 2 and 0 <= pc < size * 2:
                    out[pr][pc] = v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_fill_checker(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Preenche com padrão de tabuleiro."""
    g = state.grid
    h, w = grid_shape_v124(g)
    c1 = int(args.get("c1", 0))
    c2 = int(args.get("c2", 1))
    
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            out[r][c] = c1 if (r + c) % 2 == 0 else c2
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_fill_stripes_h(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Preenche com listras horizontais."""
    g = state.grid
    h, w = grid_shape_v124(g)
    c1 = int(args.get("c1", 0))
    c2 = int(args.get("c2", 1))
    
    out = [[c1 if r % 2 == 0 else c2 for c in range(w)] for r in range(h)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_fill_stripes_v(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Preenche com listras verticais."""
    g = state.grid
    h, w = grid_shape_v124(g)
    c1 = int(args.get("c1", 0))
    c2 = int(args.get("c2", 1))
    
    out = [[c1 if c % 2 == 0 else c2 for c in range(w)] for r in range(h)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_color_swap(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Troca duas cores."""
    g = state.grid
    h, w = grid_shape_v124(g)
    c1 = int(args.get("c1", 0))
    c2 = int(args.get("c2", 1))
    
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            if v == c1:
                out[r][c] = c2
            elif v == c2:
                out[r][c] = c1
            else:
                out[r][c] = v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_color_invert(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Inverte cores (9 - cor)."""
    g = state.grid
    h, w = grid_shape_v124(g)
    max_color = int(args.get("max_color", 9))
    
    out = [[max_color - int(g[r][c]) for c in range(w)] for r in range(h)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_color_normalize(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Normaliza cores para sequência 0,1,2,..."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    # Encontra cores únicas em ordem de aparecimento
    seen: Dict[int, int] = {}
    next_color = 0
    out = [[0 for _ in range(w)] for _ in range(h)]
    
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            if v not in seen:
                seen[v] = next_color
                next_color += 1
            out[r][c] = seen[v]
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_color_to_most_frequent(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Substitui cor especificada pela mais frequente."""
    g = state.grid
    h, w = grid_shape_v124(g)
    target = int(args.get("target", 0))
    bg = int(args.get("bg", 0))
    
    most_freq = _most_frequent_color(g, exclude_bg=True, bg=bg)
    
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            out[r][c] = most_freq if v == target else v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_scale_2x(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Escala a grade 2x."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    out = [[0 for _ in range(w * 2)] for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            out[r * 2][c * 2] = v
            out[r * 2][c * 2 + 1] = v
            out[r * 2 + 1][c * 2] = v
            out[r * 2 + 1][c * 2 + 1] = v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_scale_3x(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Escala a grade 3x."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    out = [[0 for _ in range(w * 3)] for _ in range(h * 3)]
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            for dr in range(3):
                for dc in range(3):
                    out[r * 3 + dr][c * 3 + dc] = v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_scale_half(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Reduz a grade pela metade (usa moda de cada bloco 2x2)."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    new_h = h // 2
    new_w = w // 2
    if new_h == 0 or new_w == 0:
        return state
    
    out = [[0 for _ in range(new_w)] for _ in range(new_h)]
    for r in range(new_h):
        for c in range(new_w):
            # Pega os 4 pixels do bloco 2x2
            colors = [
                int(g[r * 2][c * 2]),
                int(g[r * 2][c * 2 + 1]) if c * 2 + 1 < w else 0,
                int(g[r * 2 + 1][c * 2]) if r * 2 + 1 < h else 0,
                int(g[r * 2 + 1][c * 2 + 1]) if r * 2 + 1 < h and c * 2 + 1 < w else 0,
            ]
            # Usa a moda
            counts: Dict[int, int] = {}
            for v in colors:
                counts[v] = counts.get(v, 0) + 1
            out[r][c] = max(counts.keys(), key=lambda x: (counts[x], -x))
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_crop_margin(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Remove margens de uma cor específica."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    
    # Encontra bounding box do conteúdo não-bg
    r_min, r_max = h, 0
    c_min, c_max = w, 0
    
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) != bg:
                r_min = min(r_min, r)
                r_max = max(r_max, r)
                c_min = min(c_min, c)
                c_max = max(c_max, c)
    
    if r_min > r_max or c_min > c_max:
        return state
    
    out = [[int(g[r][c]) for c in range(c_min, c_max + 1)] for r in range(r_min, r_max + 1)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_pad_uniform(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Adiciona padding uniforme."""
    g = state.grid
    h, w = grid_shape_v124(g)
    pad = int(args.get("pad", 1))
    color = int(args.get("color", 0))
    
    out_h = h + 2 * pad
    out_w = w + 2 * pad
    out = [[color for _ in range(out_w)] for _ in range(out_h)]
    
    for r in range(h):
        for c in range(w):
            out[r + pad][c + pad] = int(g[r][c])
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_find_frame(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Extrai a moldura (borda) da grade."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    
    out = [[bg for _ in range(w)] for _ in range(h)]
    
    # Primeira e última linha
    for c in range(w):
        out[0][c] = int(g[0][c])
        out[h - 1][c] = int(g[h - 1][c])
    
    # Primeira e última coluna
    for r in range(h):
        out[r][0] = int(g[r][0])
        out[r][w - 1] = int(g[r][w - 1])
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_extract_core(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Extrai o núcleo (remove borda de 1 pixel)."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    if h <= 2 or w <= 2:
        return state
    
    out = [[int(g[r][c]) for c in range(1, w - 1)] for r in range(1, h - 1)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_find_repeating_unit(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Encontra a menor unidade que se repete na grade."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    # Tenta diferentes tamanhos de unidade
    for uh in range(1, h // 2 + 1):
        if h % uh != 0:
            continue
        for uw in range(1, w // 2 + 1):
            if w % uw != 0:
                continue
            
            # Verifica se a unidade se repete
            unit = [[int(g[r][c]) for c in range(uw)] for r in range(uh)]
            is_repeating = True
            
            for r in range(h):
                if not is_repeating:
                    break
                for c in range(w):
                    if int(g[r][c]) != unit[r % uh][c % uw]:
                        is_repeating = False
                        break
            
            if is_repeating:
                return replace(state, grid=grid_from_list_v124(unit), objset=None, obj=None, bbox=None, patch=None)
    
    return state


def _apply_tile_to_size(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Replica a grade para atingir tamanho específico."""
    g = state.grid
    h, w = grid_shape_v124(g)
    target_h = int(args.get("target_h", h * 2))
    target_w = int(args.get("target_w", w * 2))
    
    if h == 0 or w == 0:
        return state
    
    out = [[int(g[r % h][c % w]) for c in range(target_w)] for r in range(target_h)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_select_largest_obj(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Seleciona apenas o maior objeto."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    
    objset = connected_components4_nonbg_multicolor_v141(g, bg=bg)
    if not objset.objects:
        return state
    
    # Encontra o maior objeto
    largest = max(objset.objects, key=lambda o: o.area)
    
    out = [[bg for _ in range(w)] for _ in range(h)]
    for r, c in largest.cells:
        out[r][c] = int(g[r][c])
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_select_smallest_obj(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Seleciona apenas o menor objeto."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    
    objset = connected_components4_nonbg_multicolor_v141(g, bg=bg)
    if not objset.objects:
        return state
    
    smallest = min(objset.objects, key=lambda o: o.area)
    
    out = [[bg for _ in range(w)] for _ in range(h)]
    for r, c in smallest.cells:
        out[r][c] = int(g[r][c])
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_remove_noise(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Remove objetos pequenos (ruído)."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    min_size = int(args.get("min_size", 3))
    
    objset = connected_components4_nonbg_multicolor_v141(g, bg=bg)
    
    out = [[bg for _ in range(w)] for _ in range(h)]
    for obj in objset.objects:
        if obj.area >= min_size:
            for r, c in obj.cells:
                out[r][c] = int(g[r][c])
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_fill_holes(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Preenche buracos em objetos."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    fill_color = int(args.get("fill_color", 1))
    
    # Encontra células de fundo que não estão conectadas à borda
    visited: Set[Tuple[int, int]] = set()
    border_connected: Set[Tuple[int, int]] = set()
    
    # BFS a partir de todas as células de borda que são bg
    queue: List[Tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r == 0 or r == h - 1 or c == 0 or c == w - 1) and int(g[r][c]) == bg:
                if (r, c) not in visited:
                    queue.append((r, c))
                    visited.add((r, c))
    
    while queue:
        r, c = queue.pop(0)
        border_connected.add((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and int(g[nr][nc]) == bg:
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == bg and (r, c) not in border_connected:
                out[r][c] = fill_color
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_connect_same_color_h(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Conecta células da mesma cor horizontalmente."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    
    out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    
    for r in range(h):
        for c1 in range(w):
            if int(g[r][c1]) == bg:
                continue
            for c2 in range(c1 + 1, w):
                if int(g[r][c2]) == int(g[r][c1]):
                    # Preenche entre c1 e c2
                    for c in range(c1, c2 + 1):
                        out[r][c] = int(g[r][c1])
                    break
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_connect_same_color_v(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Conecta células da mesma cor verticalmente."""
    g = state.grid
    h, w = grid_shape_v124(g)
    bg = int(args.get("bg", 0))
    
    out = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    
    for c in range(w):
        for r1 in range(h):
            if int(g[r1][c]) == bg:
                continue
            for r2 in range(r1 + 1, h):
                if int(g[r2][c]) == int(g[r1][c]):
                    for r in range(r1, r2 + 1):
                        out[r][c] = int(g[r1][c])
                    break
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_mirror_extend_h(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Estende a grade com espelho horizontal."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    out = [[0 for _ in range(w * 2)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            out[r][c] = v
            out[r][w * 2 - 1 - c] = v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_mirror_extend_v(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Estende a grade com espelho vertical."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    out = [[0 for _ in range(w)] for _ in range(h * 2)]
    for r in range(h):
        for c in range(w):
            v = int(g[r][c])
            out[r][c] = v
            out[h * 2 - 1 - r][c] = v
    
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_copy_quadrant(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Copia um quadrante para toda a grade."""
    g = state.grid
    h, w = grid_shape_v124(g)
    quadrant = str(args.get("quadrant", "tl"))  # tl, tr, bl, br
    
    qh = h // 2
    qw = w // 2
    if qh == 0 or qw == 0:
        return state
    
    # Extrai o quadrante
    if quadrant == "tl":
        q = [[int(g[r][c]) for c in range(qw)] for r in range(qh)]
    elif quadrant == "tr":
        q = [[int(g[r][c]) for c in range(qw, w)] for r in range(qh)]
    elif quadrant == "bl":
        q = [[int(g[r][c]) for c in range(qw)] for r in range(qh, h)]
    else:  # br
        q = [[int(g[r][c]) for c in range(qw, w)] for r in range(qh, h)]
    
    # Replica para toda a grade
    out = [[q[r % len(q)][c % len(q[0])] for c in range(w)] for r in range(h)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


def _apply_sort_rows_by_sum(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Ordena linhas pela soma de seus valores."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    rows = [[int(g[r][c]) for c in range(w)] for r in range(h)]
    rows.sort(key=lambda row: sum(row))
    
    return replace(state, grid=grid_from_list_v124(rows), objset=None, obj=None, bbox=None, patch=None)


def _apply_sort_cols_by_sum(state: StateV132, args: Dict[str, Any]) -> StateV132:
    """Ordena colunas pela soma de seus valores."""
    g = state.grid
    h, w = grid_shape_v124(g)
    
    cols = [[int(g[r][c]) for r in range(h)] for c in range(w)]
    cols.sort(key=lambda col: sum(col))
    
    out = [[cols[c][r] for c in range(w)] for r in range(h)]
    return replace(state, grid=grid_from_list_v124(out), objset=None, obj=None, bbox=None, patch=None)


# ─────────────────────────────────────────────────────────────────────────────
# DISPATCHER PRINCIPAL V161
# ─────────────────────────────────────────────────────────────────────────────

_OP_HANDLERS_V161 = {
    "symmetry_diagonal_main": _apply_symmetry_diagonal_main,
    "symmetry_diagonal_anti": _apply_symmetry_diagonal_anti,
    "symmetry_4way": _apply_symmetry_4way,
    "symmetry_8way": _apply_symmetry_8way,
    "fill_checker": _apply_fill_checker,
    "fill_stripes_h": _apply_fill_stripes_h,
    "fill_stripes_v": _apply_fill_stripes_v,
    "color_swap": _apply_color_swap,
    "color_invert": _apply_color_invert,
    "color_normalize": _apply_color_normalize,
    "color_to_most_frequent": _apply_color_to_most_frequent,
    "scale_2x": _apply_scale_2x,
    "scale_3x": _apply_scale_3x,
    "scale_half": _apply_scale_half,
    "crop_margin": _apply_crop_margin,
    "pad_uniform": _apply_pad_uniform,
    "find_frame": _apply_find_frame,
    "extract_core": _apply_extract_core,
    "find_repeating_unit": _apply_find_repeating_unit,
    "tile_to_size": _apply_tile_to_size,
    "select_largest_obj": _apply_select_largest_obj,
    "select_smallest_obj": _apply_select_smallest_obj,
    "remove_noise": _apply_remove_noise,
    "fill_holes": _apply_fill_holes,
    "connect_same_color_h": _apply_connect_same_color_h,
    "connect_same_color_v": _apply_connect_same_color_v,
    "mirror_extend_h": _apply_mirror_extend_h,
    "mirror_extend_v": _apply_mirror_extend_v,
    "copy_quadrant": _apply_copy_quadrant,
    "sort_rows_by_sum": _apply_sort_rows_by_sum,
    "sort_cols_by_sum": _apply_sort_cols_by_sum,
}


def step_cost_bits_v161(*, op_id: str, args: Dict[str, Any]) -> int:
    """Calcula custo em bits de um passo V161."""
    op = str(op_id)
    if op in OP_DEFS_V161 and op not in OP_DEFS_V141:
        od = OP_DEFS_V161[op]
        base = int(od.base_cost_bits)
        extra = 0
        for key in args:
            if key in ("bg", "c1", "c2", "color", "fill_color", "target"):
                extra += 4  # cores 0-9
            elif key in ("pad", "min_size"):
                extra += 3
            elif key in ("quadrant", "mode"):
                extra += 2
            elif key in ("target_h", "target_w"):
                extra += 5
        return int(base + extra)
    return step_cost_bits_v141(op_id=op_id, args=args)


def apply_op_v161(*, state: StateV132, op_id: str, args: Dict[str, Any]) -> StateV132:
    """Aplica operador V161 ou delega para V141."""
    op = str(op_id)
    handler = _OP_HANDLERS_V161.get(op)
    if handler is not None:
        return handler(state, args)
    return apply_op_v141(state=state, op_id=op_id, args=args)


def list_v161_operators() -> List[str]:
    """Lista todos os operadores V161 novos."""
    return [op for op in OP_DEFS_V161 if op not in OP_DEFS_V141]


def get_v161_op_def(op_id: str) -> Optional[OpDefV132]:
    """Obtém definição de operador V161."""
    return OP_DEFS_V161.get(op_id)


# ─────────────────────────────────────────────────────────────────────────────
# EXPORTS
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "ARC_OPS_SCHEMA_VERSION_V161",
    "OP_DEFS_V161",
    "apply_op_v161",
    "step_cost_bits_v161",
    "list_v161_operators",
    "get_v161_op_def",
]
