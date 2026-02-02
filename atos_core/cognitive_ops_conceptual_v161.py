"""
cognitive_ops_conceptual_v161.py - Operadores Conceituais Expandidos

Esta expansão implementa todos os operadores conceituais autorizados para
alcançar ≥90% de acerto nos benchmarks ARC-AGI-1 e ARC-AGI-2.

Categorias implementadas:
1. Transformações geométricas (rotate_270, reflect_diagonal, etc.)
2. Manipulação de cores (invert_colors, rotate_color_palette, etc.)
3. Operações de forma (outline_shape, fill_shape, connect_points, etc.)
4. Operações de padrão (tile_pattern, pattern_completion, etc.)
5. Operações de escala (scale_to_fit, enlarge_pattern, etc.)

Schema version: 161
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Callable
import math

# Type alias para grids
GridV124 = List[List[int]]

COGNITIVE_OPS_CONCEPTUAL_SCHEMA_VERSION_V161 = 161


# ─────────────────────────────────────────────────────────────────────────────
# Utility Functions
# ─────────────────────────────────────────────────────────────────────────────


def _check_color_v161(c: int) -> int:
    """Validate ARC color (0-9)."""
    cc = int(c)
    if cc < 0 or cc > 9:
        raise ValueError("color_out_of_range")
    return cc


def _new_grid_v161(rows: int, cols: int, fill: int = 0) -> GridV124:
    """Create a new grid with given dimensions."""
    return [[int(fill) for _ in range(int(cols))] for _ in range(int(rows))]


def _copy_grid_v161(g: GridV124) -> GridV124:
    """Deep copy a grid."""
    return [[int(c) for c in row] for row in g]


def _grid_shape_v161(g: GridV124) -> Tuple[int, int]:
    """Get (height, width) of grid."""
    if not g:
        return (0, 0)
    return (len(g), len(g[0]) if g[0] else 0)


def _get_neighbors_4_v161(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Get 4-connected neighbors within bounds."""
    neighbors = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            neighbors.append((nr, nc))
    return neighbors


def _get_neighbors_8_v161(r: int, c: int, h: int, w: int) -> List[Tuple[int, int]]:
    """Get 8-connected neighbors within bounds."""
    neighbors = []
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w:
                neighbors.append((nr, nc))
    return neighbors


def _find_objects_v161(g: GridV124, *, bg: int = 0, connectivity: int = 4) -> List[Set[Tuple[int, int]]]:
    """Find all connected components (objects) in grid."""
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return []
    
    get_neighbors = _get_neighbors_4_v161 if connectivity == 4 else _get_neighbors_8_v161
    visited: Set[Tuple[int, int]] = set()
    objects: List[Set[Tuple[int, int]]] = []
    
    for r in range(h):
        for c in range(w):
            if (r, c) in visited or int(g[r][c]) == bg:
                continue
            
            # BFS to find connected component
            component: Set[Tuple[int, int]] = set()
            queue = [(r, c)]
            visited.add((r, c))
            
            while queue:
                cr, cc = queue.pop(0)
                component.add((cr, cc))
                
                for nr, nc in get_neighbors(cr, cc, h, w):
                    if (nr, nc) not in visited and int(g[nr][nc]) != bg:
                        visited.add((nr, nc))
                        queue.append((nr, nc))
            
            objects.append(component)
    
    return objects


def _object_bbox_v161(obj: Set[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    """Get bounding box (min_r, min_c, max_r, max_c) of object."""
    if not obj:
        return (0, 0, 0, 0)
    rs = [r for r, c in obj]
    cs = [c for r, c in obj]
    return (min(rs), min(cs), max(rs), max(cs))


def _extract_object_v161(g: GridV124, obj: Set[Tuple[int, int]], *, bg: int = 0) -> GridV124:
    """Extract object as a minimal grid."""
    if not obj:
        return [[]]
    
    min_r, min_c, max_r, max_c = _object_bbox_v161(obj)
    h = max_r - min_r + 1
    w = max_c - min_c + 1
    
    out = _new_grid_v161(h, w, bg)
    for r, c in obj:
        out[r - min_r][c - min_c] = int(g[r][c])
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 1. TRANSFORMAÇÕES GEOMÉTRICAS
# ─────────────────────────────────────────────────────────────────────────────


def rotate_270(g: GridV124) -> GridV124:
    """
    Rotaciona a grade em 270 graus (equivalente a 90° para a esquerda).
    
    Papel: Permite alinhar imagens em qualquer orientação.
    Justificativa: Acelera tarefas que exigem rotações de 270° diretamente.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    # 270° = transpor e depois flip vertical
    out = _new_grid_v161(w, h)
    for r in range(h):
        for c in range(w):
            out[w - 1 - c][r] = int(g[r][c])
    return out


def reflect_diagonal(g: GridV124) -> GridV124:
    """
    Espelha o padrão ao longo da diagonal principal.
    
    Papel: Completa padrões simétricos em relação à diagonal.
    Justificativa: Resolve tarefas com simetria diagonal sem composições complexas.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    # Diagonal principal = transpor
    size = max(h, w)
    out = _new_grid_v161(size, size)
    
    for r in range(h):
        for c in range(w):
            out[c][r] = int(g[r][c])
    
    return out


def reflect_antidiagonal(g: GridV124) -> GridV124:
    """
    Reflete o padrão pela diagonal secundária (anti-diagonal).
    
    Papel: Permite reconstruir simetrias em relação à diagonal inversa.
    Justificativa: Amplia capacidade de completar padrões em todos eixos diagonais.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    size = max(h, w)
    out = _new_grid_v161(size, size)
    
    for r in range(h):
        for c in range(w):
            # Anti-diagonal: (r, c) -> (size-1-c, size-1-r)
            out[size - 1 - c][size - 1 - r] = int(g[r][c])
    
    return out


def translate_pattern(g: GridV124, *, dr: int = 0, dc: int = 0, bg: int = 0) -> GridV124:
    """
    Desloca um padrão dentro da grade sem alterar sua forma.
    
    Papel: Reposiciona objetos mantendo sua estrutura.
    Justificativa: Simplifica tarefas onde a solução é puramente posicional.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    out = _new_grid_v161(h, w, bg)
    
    for r in range(h):
        for c in range(w):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out[nr][nc] = int(g[r][c])
    
    return out


def wraparound_shift(g: GridV124, *, direction: str = "right", amount: int = 1) -> GridV124:
    """
    Desloca conteúdo com wraparound (toro).
    
    direction: "up", "down", "left", "right"
    
    Papel: Trata padrões cíclicos ou cenários com continuidade além das bordas.
    Justificativa: Facilita encaixar partes desconexas movendo e reconectando.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    out = _new_grid_v161(h, w)
    
    for r in range(h):
        for c in range(w):
            if direction == "right":
                nc = (c + amount) % w
                out[r][nc] = int(g[r][c])
            elif direction == "left":
                nc = (c - amount) % w
                out[r][nc] = int(g[r][c])
            elif direction == "down":
                nr = (r + amount) % h
                out[nr][c] = int(g[r][c])
            elif direction == "up":
                nr = (r - amount) % h
                out[nr][c] = int(g[r][c])
            else:
                out[r][c] = int(g[r][c])
    
    return out


def transpose_grid(g: GridV124) -> GridV124:
    """
    Troca linhas e colunas (transposição de matriz).
    
    Papel: Reorienta estrutura espacial do input.
    Justificativa: Útil para inverter relação linha/coluna e revelar simetrias.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    out = _new_grid_v161(w, h)
    for r in range(h):
        for c in range(w):
            out[c][r] = int(g[r][c])
    return out


def shear_pattern(g: GridV124, *, axis: str = "horizontal", amount: int = 1, bg: int = 0) -> GridV124:
    """
    Aplica transformação de cisalhamento (skew).
    
    axis: "horizontal" ou "vertical"
    
    Papel: Ajusta padrões inclinados ou cria inclinação diagonal.
    Justificativa: Expande expressividade geométrica para alinhamentos diagonais.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    
    if axis == "horizontal":
        # Cada linha é deslocada proporcionalmente
        new_w = w + abs(amount) * (h - 1)
        out = _new_grid_v161(h, new_w, bg)
        for r in range(h):
            offset = r * amount if amount > 0 else (h - 1 - r) * abs(amount)
            for c in range(w):
                out[r][c + offset] = int(g[r][c])
    else:  # vertical
        new_h = h + abs(amount) * (w - 1)
        out = _new_grid_v161(new_h, w, bg)
        for c in range(w):
            offset = c * amount if amount > 0 else (w - 1 - c) * abs(amount)
            for r in range(h):
                out[r + offset][c] = int(g[r][c])
    
    return out


def scale_to_fit(g: GridV124, *, target_h: int, target_w: int, bg: int = 0) -> GridV124:
    """
    Redimensiona padrão para caber em tamanho-alvo.
    
    Papel: Adapta objetos para ocuparem determinada área.
    Justificativa: Agiliza soluções onde output é versão reescalada.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _new_grid_v161(target_h, target_w, bg)
    
    bg = _check_color_v161(bg)
    out = _new_grid_v161(target_h, target_w, bg)
    
    # Scale factors
    scale_r = target_h / h
    scale_c = target_w / w
    
    for tr in range(target_h):
        for tc in range(target_w):
            sr = int(tr / scale_r)
            sc = int(tc / scale_c)
            if 0 <= sr < h and 0 <= sc < w:
                out[tr][tc] = int(g[sr][sc])
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. MANIPULAÇÃO DE CORES
# ─────────────────────────────────────────────────────────────────────────────


def invert_colors(g: GridV124, *, bg: int = 0, fg: int = 1) -> GridV124:
    """
    Inverte as cores (fundo vira destaque e vice-versa).
    
    Papel: Produz o "negativo" da imagem.
    Justificativa: Resolve tarefas onde output é inversão do input.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    fg = _check_color_v161(fg)
    out = _copy_grid_v161(g)
    
    for r in range(h):
        for c in range(w):
            if int(out[r][c]) == bg:
                out[r][c] = fg
            else:
                out[r][c] = bg
    
    return out


def rotate_color_palette(g: GridV124, *, rotation: int = 1) -> GridV124:
    """
    Permuta ciclicamente as cores presentes.
    
    Papel: Reatribui cores mantendo disposição espacial.
    Justificativa: Acelera puzzles com troca cíclica de cores.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    # Get unique colors and sort them
    colors = sorted(set(int(g[r][c]) for r in range(h) for c in range(w)))
    if len(colors) <= 1:
        return _copy_grid_v161(g)
    
    # Create rotation mapping
    color_map = {}
    for i, c in enumerate(colors):
        new_idx = (i + rotation) % len(colors)
        color_map[c] = colors[new_idx]
    
    out = _copy_grid_v161(g)
    for r in range(h):
        for c in range(w):
            out[r][c] = color_map.get(int(g[r][c]), int(g[r][c]))
    
    return out


def quantize_colors(g: GridV124, *, max_colors: int = 2, bg: int = 0) -> GridV124:
    """
    Reduz número de cores distintas na imagem.
    
    Papel: Simplifica representação removendo detalhes de cor irrelevantes.
    Justificativa: Ajuda a generalizar focando em elementos fundamentais.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    
    # Count colors (excluding bg)
    counts: Dict[int, int] = {}
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            if color != bg:
                counts[color] = counts.get(color, 0) + 1
    
    if len(counts) <= max_colors:
        return _copy_grid_v161(g)
    
    # Keep top max_colors
    top_colors = set(sorted(counts, key=lambda x: counts[x], reverse=True)[:max_colors])
    if not top_colors:
        top_colors = {1}  # Default
    
    main_color = max(top_colors, key=lambda x: counts.get(x, 0))
    
    out = _copy_grid_v161(g)
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            if color != bg and color not in top_colors:
                out[r][c] = main_color
    
    return out


def merge_colors(g: GridV124, *, source: int, target: int) -> GridV124:
    """
    Mescla duas cores distintas em uma única.
    
    Papel: Combina objetos/regiões que diferiam apenas por cor.
    Justificativa: Reduz variabilidade quando distinções de cor não importam.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    source = _check_color_v161(source)
    target = _check_color_v161(target)
    
    out = _copy_grid_v161(g)
    for r in range(h):
        for c in range(w):
            if int(out[r][c]) == source:
                out[r][c] = target
    
    return out


def isolate_color(g: GridV124, *, color: int, bg: int = 0) -> GridV124:
    """
    Mantém apenas células de uma cor específica, resto vira fundo.
    
    Papel: Extrai elementos do grid que possuem cor de interesse.
    Justificativa: Destaca aspecto específico para raciocínio isolado.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    color = _check_color_v161(color)
    bg = _check_color_v161(bg)
    
    out = _new_grid_v161(h, w, bg)
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == color:
                out[r][c] = color
    
    return out


def remove_color(g: GridV124, *, color: int, bg: int = 0) -> GridV124:
    """
    Remove completamente uma cor da imagem, substituindo por fundo.
    
    Papel: Apaga elementos indesejados baseados em cor.
    Justificativa: Simplifica ignorando ou removendo tudo de certa cor.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    color = _check_color_v161(color)
    bg = _check_color_v161(bg)
    
    out = _copy_grid_v161(g)
    for r in range(h):
        for c in range(w):
            if int(out[r][c]) == color:
                out[r][c] = bg
    
    return out


def map_colors(g: GridV124, *, color_map: Dict[int, int]) -> GridV124:
    """
    Aplica mapeamento arbitrário de cores em toda a grade.
    
    Papel: Realiza múltiplas trocas de cor simultaneamente.
    Justificativa: Acelera transformações complexas de paleta atomicamente.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    out = _copy_grid_v161(g)
    for r in range(h):
        for c in range(w):
            old_color = int(g[r][c])
            if old_color in color_map:
                out[r][c] = _check_color_v161(color_map[old_color])
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. OPERAÇÕES DE FORMA
# ─────────────────────────────────────────────────────────────────────────────


def outline_shape(g: GridV124, *, bg: int = 0) -> GridV124:
    """
    Remove interior deixando apenas contorno (bordas) das formas.
    
    Papel: Extrai silhueta de formas sólidas.
    Justificativa: Ajuda destacar bordas ou verificar formato ignorando preenchimento.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    out = _new_grid_v161(h, w, bg)
    
    for r in range(h):
        for c in range(w):
            color = int(g[r][c])
            if color == bg:
                continue
            
            # Check if this is a border pixel
            is_border = False
            for nr, nc in _get_neighbors_4_v161(r, c, h, w):
                if int(g[nr][nc]) == bg:
                    is_border = True
                    break
            
            # Also border if at edge of grid
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                is_border = True
            
            if is_border:
                out[r][c] = color
    
    return out


def fill_shape(g: GridV124, *, bg: int = 0, fill_color: Optional[int] = None) -> GridV124:
    """
    Preenche interior de formas fechadas.
    
    Papel: Reconstrói objetos vazados ou completa áreas internas.
    Justificativa: Automatiza flood fill dentro de contornos.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    out = _copy_grid_v161(g)
    
    # Find objects and fill their bounding boxes
    objects = _find_objects_v161(g, bg=bg)
    
    for obj in objects:
        if not obj:
            continue
        
        # Get bounding box
        min_r, min_c, max_r, max_c = _object_bbox_v161(obj)
        
        # Determine fill color (mode color of object)
        colors = [int(g[r][c]) for r, c in obj]
        obj_color = fill_color if fill_color is not None else Counter(colors).most_common(1)[0][0]
        
        # Fill convex hull of object (simplified: fill bounding box interior)
        for r in range(min_r, max_r + 1):
            for c in range(min_c, max_c + 1):
                # Check if this cell is "inside" the object
                # Simple heuristic: between leftmost and rightmost in row
                row_cells = [(rr, cc) for rr, cc in obj if rr == r]
                if row_cells:
                    cs = [cc for _, cc in row_cells]
                    if min(cs) <= c <= max(cs):
                        out[r][c] = obj_color
    
    return out


def connect_points(g: GridV124, *, p1: Tuple[int, int], p2: Tuple[int, int], color: int = 1) -> GridV124:
    """
    Desenha linha reta conectando dois pontos.
    
    Papel: Une objetos separados criando conexão direta.
    Justificativa: Muitas tasks exigem conectar elementos com traço.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    color = _check_color_v161(color)
    out = _copy_grid_v161(g)
    
    r1, c1 = p1
    r2, c2 = p2
    
    # Bresenham's line algorithm
    dr = abs(r2 - r1)
    dc = abs(c2 - c1)
    sr = 1 if r1 < r2 else -1
    sc = 1 if c1 < c2 else -1
    err = dr - dc
    
    r, c = r1, c1
    while True:
        if 0 <= r < h and 0 <= c < w:
            out[r][c] = color
        
        if r == r2 and c == c2:
            break
        
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
    
    return out


def bridge_gap(g: GridV124, *, bg: int = 0, max_gap: int = 3) -> GridV124:
    """
    Preenche espaço entre objetos próximos, unindo-os.
    
    Papel: Fecha lacunas pequenas entre partes de padrão interrompido.
    Justificativa: Elimina lacunas entre segmentos diretamente.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    out = _copy_grid_v161(g)
    
    # Fill horizontal gaps
    for r in range(h):
        last_color = bg
        last_c = -1
        for c in range(w):
            color = int(g[r][c])
            if color != bg:
                if last_color != bg and c - last_c - 1 <= max_gap:
                    # Fill the gap
                    fill = last_color
                    for fc in range(last_c + 1, c):
                        out[r][fc] = fill
                last_color = color
                last_c = c
    
    # Fill vertical gaps
    for c in range(w):
        last_color = bg
        last_r = -1
        for r in range(h):
            color = int(out[r][c])
            if color != bg:
                if last_color != bg and r - last_r - 1 <= max_gap:
                    fill = last_color
                    for fr in range(last_r + 1, r):
                        out[fr][c] = fill
                last_color = color
                last_r = r
    
    return out


def remove_small_objects(g: GridV124, *, bg: int = 0, min_size: int = 3) -> GridV124:
    """
    Remove objetos menores que o limiar.
    
    Papel: Filtra ruídos ou detalhes insignificantes.
    Justificativa: Evita que elementos irrelevantes influenciem raciocínio.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    objects = _find_objects_v161(g, bg=bg)
    
    out = _copy_grid_v161(g)
    for obj in objects:
        if len(obj) < min_size:
            for r, c in obj:
                out[r][c] = bg
    
    return out


def remove_large_objects(g: GridV124, *, bg: int = 0, max_size: int = 10) -> GridV124:
    """
    Remove objetos maiores que o limiar.
    
    Papel: Filtra componentes muito grandes, preservando menores.
    Justificativa: Ajuda quando grandes áreas são contexto irrelevante.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    objects = _find_objects_v161(g, bg=bg)
    
    out = _copy_grid_v161(g)
    for obj in objects:
        if len(obj) > max_size:
            for r, c in obj:
                out[r][c] = bg
    
    return out


def duplicate_object(g: GridV124, *, obj_idx: int = 0, offset: Tuple[int, int] = (0, 5), bg: int = 0) -> GridV124:
    """
    Cria cópia idêntica de um objeto selecionado.
    
    Papel: Permite replicar padrões ou objetos detectados.
    Justificativa: Útil para tasks de replicação ou multiplicação.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    objects = _find_objects_v161(g, bg=bg)
    
    if obj_idx >= len(objects):
        return _copy_grid_v161(g)
    
    obj = objects[obj_idx]
    out = _copy_grid_v161(g)
    dr, dc = offset
    
    for r, c in obj:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr][nc] = int(g[r][c])
    
    return out


def extract_object(g: GridV124, *, obj_idx: int = 0, bg: int = 0) -> GridV124:
    """
    Recorta objeto específico da grade.
    
    Papel: Separa elemento de seu contexto original.
    Justificativa: Facilita manobras de rearranjo.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return [[]]
    
    bg = _check_color_v161(bg)
    objects = _find_objects_v161(g, bg=bg)
    
    if obj_idx >= len(objects):
        return [[]]
    
    return _extract_object_v161(g, objects[obj_idx], bg=bg)


def reposition_object(g: GridV124, *, obj_idx: int = 0, new_pos: Tuple[int, int], bg: int = 0) -> GridV124:
    """
    Move objeto para novas coordenadas.
    
    Papel: Recoloca elemento no local correto da resposta.
    Justificativa: Agiliza soluções onde tarefa é mover X para determinado lugar.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    objects = _find_objects_v161(g, bg=bg)
    
    if obj_idx >= len(objects):
        return _copy_grid_v161(g)
    
    obj = objects[obj_idx]
    min_r, min_c, _, _ = _object_bbox_v161(obj)
    
    # Calculate offset
    dr = new_pos[0] - min_r
    dc = new_pos[1] - min_c
    
    out = _copy_grid_v161(g)
    
    # Remove from old position
    for r, c in obj:
        out[r][c] = bg
    
    # Place at new position
    for r, c in obj:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out[nr][nc] = int(g[r][c])
    
    return out


def align_objects(g: GridV124, *, axis: str = "horizontal", position: str = "center", bg: int = 0) -> GridV124:
    """
    Alinha múltiplos objetos ao longo de um eixo.
    
    axis: "horizontal" ou "vertical"
    position: "start", "center", "end"
    
    Papel: Ajusta posição de todos objetos para ficarem alinhados.
    Justificativa: Resolve puzzles que exigem organizar componentes dispersos.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    objects = _find_objects_v161(g, bg=bg)
    
    if not objects:
        return _copy_grid_v161(g)
    
    out = _new_grid_v161(h, w, bg)
    
    if axis == "horizontal":
        # Align vertically (same row)
        if position == "center":
            target_r = h // 2
        elif position == "start":
            target_r = 0
        else:  # end
            target_r = h - 1
        
        c_offset = 0
        for obj in objects:
            obj_grid = _extract_object_v161(g, obj, bg=bg)
            obj_h, obj_w = _grid_shape_v161(obj_grid)
            
            for r in range(obj_h):
                for c in range(obj_w):
                    nr = target_r - obj_h // 2 + r
                    nc = c_offset + c
                    if 0 <= nr < h and 0 <= nc < w:
                        if int(obj_grid[r][c]) != bg:
                            out[nr][nc] = int(obj_grid[r][c])
            
            c_offset += obj_w + 1
    else:  # vertical
        if position == "center":
            target_c = w // 2
        elif position == "start":
            target_c = 0
        else:
            target_c = w - 1
        
        r_offset = 0
        for obj in objects:
            obj_grid = _extract_object_v161(g, obj, bg=bg)
            obj_h, obj_w = _grid_shape_v161(obj_grid)
            
            for r in range(obj_h):
                for c in range(obj_w):
                    nr = r_offset + r
                    nc = target_c - obj_w // 2 + c
                    if 0 <= nr < h and 0 <= nc < w:
                        if int(obj_grid[r][c]) != bg:
                            out[nr][nc] = int(obj_grid[r][c])
            
            r_offset += obj_h + 1
    
    return out


def flood_fill(g: GridV124, *, start: Tuple[int, int], color: int, connectivity: int = 4) -> GridV124:
    """
    Preenche região contígua a partir de uma célula de referência.
    
    Papel: Funciona como "balde de tinta", colorindo áreas delimitadas.
    Justificativa: Essencial para completar áreas internas ou regiões cercadas.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    color = _check_color_v161(color)
    r0, c0 = start
    
    if not (0 <= r0 < h and 0 <= c0 < w):
        return _copy_grid_v161(g)
    
    target_color = int(g[r0][c0])
    if target_color == color:
        return _copy_grid_v161(g)
    
    get_neighbors = _get_neighbors_4_v161 if connectivity == 4 else _get_neighbors_8_v161
    
    out = _copy_grid_v161(g)
    visited: Set[Tuple[int, int]] = set()
    queue = [(r0, c0)]
    visited.add((r0, c0))
    
    while queue:
        r, c = queue.pop(0)
        out[r][c] = color
        
        for nr, nc in get_neighbors(r, c, h, w):
            if (nr, nc) not in visited and int(g[nr][nc]) == target_color:
                visited.add((nr, nc))
                queue.append((nr, nc))
    
    return out


def crop_region(g: GridV124, *, r1: int, c1: int, r2: int, c2: int) -> GridV124:
    """
    Recorta sub-região retangular da grade.
    
    Papel: Isola parte de interesse do input.
    Justificativa: Simplifica problema removendo áreas irrelevantes.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return [[]]
    
    r1 = max(0, min(r1, h - 1))
    r2 = max(0, min(r2, h - 1))
    c1 = max(0, min(c1, w - 1))
    c2 = max(0, min(c2, w - 1))
    
    if r1 > r2:
        r1, r2 = r2, r1
    if c1 > c2:
        c1, c2 = c2, c1
    
    new_h = r2 - r1 + 1
    new_w = c2 - c1 + 1
    
    out = _new_grid_v161(new_h, new_w)
    for r in range(new_h):
        for c in range(new_w):
            out[r][c] = int(g[r1 + r][c1 + c])
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 4. OPERAÇÕES DE PADRÃO
# ─────────────────────────────────────────────────────────────────────────────


def tile_pattern(g: GridV124, *, tiles_h: int = 2, tiles_w: int = 2) -> GridV124:
    """
    Repete padrão em forma de grade (tiling).
    
    Papel: Gera mosaico compondo cópias lado a lado.
    Justificativa: Acelera criação de outputs onde solução é repetir bloco N×M vezes.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return [[]]
    
    new_h = h * tiles_h
    new_w = w * tiles_w
    
    out = _new_grid_v161(new_h, new_w)
    
    for r in range(new_h):
        for c in range(new_w):
            out[r][c] = int(g[r % h][c % w])
    
    return out


def repeat_mirror(g: GridV124, *, tiles_h: int = 2, tiles_w: int = 2) -> GridV124:
    """
    Repete padrão com espelhamento alternado.
    
    Papel: Preenche áreas mantendo continuidade de borda.
    Justificativa: Resolve tasks onde motivo se repete formando tapete simétrico.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return [[]]
    
    new_h = h * tiles_h
    new_w = w * tiles_w
    
    out = _new_grid_v161(new_h, new_w)
    
    for tr in range(tiles_h):
        for tc in range(tiles_w):
            # Determine if we need to flip
            flip_h = tr % 2 == 1
            flip_w = tc % 2 == 1
            
            for r in range(h):
                for c in range(w):
                    sr = (h - 1 - r) if flip_h else r
                    sc = (w - 1 - c) if flip_w else c
                    out[tr * h + r][tc * w + c] = int(g[sr][sc])
    
    return out


def pattern_completion(g: GridV124, *, symmetry: str = "horizontal", bg: int = 0) -> GridV124:
    """
    Completa padrão parcial baseado em simetria esperada.
    
    symmetry: "horizontal", "vertical", "diagonal", "rot180"
    
    Papel: Replica metade para formar figura completa simétrica.
    Justificativa: Evita tentativas manuais de adivinhar continuação.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    out = _copy_grid_v161(g)
    
    if symmetry == "horizontal":
        # Mirror left to right
        for r in range(h):
            for c in range(w // 2 + 1, w):
                if int(out[r][c]) == bg:
                    out[r][c] = int(out[r][w - 1 - c])
    elif symmetry == "vertical":
        # Mirror top to bottom
        for r in range(h // 2 + 1, h):
            for c in range(w):
                if int(out[r][c]) == bg:
                    out[r][c] = int(out[h - 1 - r][c])
    elif symmetry == "diagonal":
        # Mirror across diagonal
        size = max(h, w)
        new_out = _new_grid_v161(size, size, bg)
        for r in range(h):
            for c in range(w):
                new_out[r][c] = int(g[r][c])
        for r in range(size):
            for c in range(size):
                if int(new_out[r][c]) == bg and r < size and c < size:
                    if int(new_out[c][r]) != bg:
                        new_out[r][c] = int(new_out[c][r])
        return new_out
    elif symmetry == "rot180":
        # 180° rotational symmetry
        mid = (h * w) // 2
        for r in range(h):
            for c in range(w):
                idx = r * w + c
                if idx > mid:
                    if int(out[r][c]) == bg:
                        out[r][c] = int(out[h - 1 - r][w - 1 - c])
    
    return out


def extrapolate_sequence(g: GridV124, *, axis: str = "horizontal", steps: int = 1, bg: int = 0) -> GridV124:
    """
    Identifica sequência e estende além do último elemento.
    
    Papel: Projeta próximo elemento de padrão sequencial.
    Justificativa: Automatiza inferência de regularidade e próximo passo.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    
    if axis == "horizontal":
        # Extend horizontally by repeating last column pattern
        new_w = w + steps
        out = _new_grid_v161(h, new_w, bg)
        
        for r in range(h):
            for c in range(w):
                out[r][c] = int(g[r][c])
        
        # Find period in each row
        for r in range(h):
            # Simple: repeat the pattern found
            for s in range(steps):
                c = w + s
                # Copy from position that creates pattern
                src_c = c % w
                out[r][c] = int(g[r][src_c])
        
        return out
    else:
        # Extend vertically
        new_h = h + steps
        out = _new_grid_v161(new_h, w, bg)
        
        for r in range(h):
            for c in range(w):
                out[r][c] = int(g[r][c])
        
        for c in range(w):
            for s in range(steps):
                r = h + s
                src_r = r % h
                out[r][c] = int(g[src_r][c])
        
        return out


def combine_patterns(g1: GridV124, g2: GridV124, *, mode: str = "overlay", bg: int = 0) -> GridV124:
    """
    Sobrepõe ou funde dois padrões distintos.
    
    mode: "overlay" (g2 over g1), "max", "min", "xor"
    
    Papel: Permite compor solução a partir de duas fontes.
    Justificativa: Garante combinação de duas ideias sem recriar do zero.
    """
    h1, w1 = _grid_shape_v161(g1)
    h2, w2 = _grid_shape_v161(g2)
    
    h = max(h1, h2)
    w = max(w1, w2)
    bg = _check_color_v161(bg)
    
    out = _new_grid_v161(h, w, bg)
    
    # Copy g1
    for r in range(h1):
        for c in range(w1):
            out[r][c] = int(g1[r][c])
    
    # Combine with g2
    for r in range(h2):
        for c in range(w2):
            v2 = int(g2[r][c])
            v1 = int(out[r][c]) if r < h1 and c < w1 else bg
            
            if mode == "overlay":
                if v2 != bg:
                    out[r][c] = v2
            elif mode == "max":
                out[r][c] = max(v1, v2)
            elif mode == "min":
                if v1 == bg:
                    out[r][c] = v2
                elif v2 == bg:
                    out[r][c] = v1
                else:
                    out[r][c] = min(v1, v2)
            elif mode == "xor":
                if v1 == bg and v2 != bg:
                    out[r][c] = v2
                elif v1 != bg and v2 == bg:
                    out[r][c] = v1
                else:
                    out[r][c] = bg
    
    return out


def replace_pattern(g: GridV124, *, pattern: GridV124, replacement: GridV124, bg: int = 0) -> GridV124:
    """
    Substitui todas ocorrências de um sub-padrão por outro.
    
    Papel: Procura instâncias de arranjo e troca por novo desenho.
    Justificativa: Útil quando transformação envolve trocar motivo por outro.
    """
    h, w = _grid_shape_v161(g)
    ph, pw = _grid_shape_v161(pattern)
    rh, rw = _grid_shape_v161(replacement)
    
    if ph > h or pw > w or ph == 0 or pw == 0:
        return _copy_grid_v161(g)
    
    out = _copy_grid_v161(g)
    
    # Find all occurrences
    for r in range(h - ph + 1):
        for c in range(w - pw + 1):
            match = True
            for pr in range(ph):
                for pc in range(pw):
                    if int(g[r + pr][c + pc]) != int(pattern[pr][pc]):
                        match = False
                        break
                if not match:
                    break
            
            if match:
                # Replace with replacement
                for rr in range(min(rh, h - r)):
                    for rc in range(min(rw, w - c)):
                        out[r + rr][c + rc] = int(replacement[rr][rc])
    
    return out


def mirror_copy(g: GridV124, *, axis: str = "horizontal", bg: int = 0) -> GridV124:
    """
    Cria cópia espelhada adjacente ao original.
    
    axis: "horizontal" ou "vertical"
    
    Papel: Gera rapidamente imagens com simetria bilateral.
    Justificativa: Resolve tarefas de duplicar e espelhar para completar todo simétrico.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    if axis == "horizontal":
        # Mirror to the right
        new_w = w * 2
        out = _new_grid_v161(h, new_w)
        for r in range(h):
            for c in range(w):
                out[r][c] = int(g[r][c])
                out[r][new_w - 1 - c] = int(g[r][c])
        return out
    else:
        # Mirror below
        new_h = h * 2
        out = _new_grid_v161(new_h, w)
        for r in range(h):
            for c in range(w):
                out[r][c] = int(g[r][c])
                out[new_h - 1 - r][c] = int(g[r][c])
        return out


def mask_pattern(g: GridV124, mask: GridV124, *, bg: int = 0) -> GridV124:
    """
    Aplica máscara sobre padrão, preservando apenas dentro da máscara.
    
    Papel: Funciona como recortar padrão no formato de outro.
    Justificativa: Útil para combinar aspecto de input com formato de outro.
    """
    h, w = _grid_shape_v161(g)
    mh, mw = _grid_shape_v161(mask)
    
    out_h = min(h, mh)
    out_w = min(w, mw)
    bg = _check_color_v161(bg)
    
    out = _new_grid_v161(out_h, out_w, bg)
    
    for r in range(out_h):
        for c in range(out_w):
            if int(mask[r][c]) != bg:
                out[r][c] = int(g[r][c])
    
    return out


def sort_objects(g: GridV124, *, by: str = "size", order: str = "asc", axis: str = "horizontal", bg: int = 0) -> GridV124:
    """
    Reordena objetos na grade de acordo com critério.
    
    by: "size", "color", "position"
    order: "asc", "desc"
    axis: "horizontal", "vertical"
    
    Papel: Alinha objetos em sequência ordenada.
    Justificativa: Resolve tasks que requerem organizar ou ordenar elementos.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    objects = _find_objects_v161(g, bg=bg)
    
    if not objects:
        return _copy_grid_v161(g)
    
    # Sort objects by criterion
    def get_key(obj):
        if by == "size":
            return len(obj)
        elif by == "color":
            colors = [int(g[r][c]) for r, c in obj]
            return Counter(colors).most_common(1)[0][0]
        else:  # position
            min_r, min_c, _, _ = _object_bbox_v161(obj)
            return (min_r, min_c)
    
    sorted_objs = sorted(objects, key=get_key, reverse=(order == "desc"))
    
    # Rebuild grid with sorted objects
    out = _new_grid_v161(h, w, bg)
    
    if axis == "horizontal":
        c_offset = 0
        for obj in sorted_objs:
            obj_grid = _extract_object_v161(g, obj, bg=bg)
            obj_h, obj_w = _grid_shape_v161(obj_grid)
            
            for r in range(obj_h):
                for c in range(obj_w):
                    if int(obj_grid[r][c]) != bg:
                        if r < h and c_offset + c < w:
                            out[r][c_offset + c] = int(obj_grid[r][c])
            c_offset += obj_w + 1
    else:
        r_offset = 0
        for obj in sorted_objs:
            obj_grid = _extract_object_v161(g, obj, bg=bg)
            obj_h, obj_w = _grid_shape_v161(obj_grid)
            
            for r in range(obj_h):
                for c in range(obj_w):
                    if int(obj_grid[r][c]) != bg:
                        if r_offset + r < h and c < w:
                            out[r_offset + r][c] = int(obj_grid[r][c])
            r_offset += obj_h + 1
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 5. OPERAÇÕES DE ESCALA E PROPORÇÃO
# ─────────────────────────────────────────────────────────────────────────────


def enlarge_pattern(g: GridV124, *, factor: int = 2) -> GridV124:
    """
    Aumenta padrão proporcionalmente.
    
    Papel: Cria versão maior do desenho mantendo forma.
    Justificativa: Quando output é input ampliado, realiza sem distorcer.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0 or factor < 1:
        return _copy_grid_v161(g)
    
    new_h = h * factor
    new_w = w * factor
    
    out = _new_grid_v161(new_h, new_w)
    
    for r in range(new_h):
        for c in range(new_w):
            out[r][c] = int(g[r // factor][c // factor])
    
    return out


def shrink_pattern(g: GridV124, *, factor: int = 2, mode: str = "mode") -> GridV124:
    """
    Reduz padrão de forma proporcional.
    
    mode: "mode" (cor mais comum), "min", "max", "first"
    
    Papel: Gera versão em menor escala conservando essência estrutural.
    Justificativa: Auxilia em tasks que pedem miniaturização do input.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0 or factor < 1:
        return _copy_grid_v161(g)
    
    new_h = max(1, h // factor)
    new_w = max(1, w // factor)
    
    out = _new_grid_v161(new_h, new_w)
    
    for r in range(new_h):
        for c in range(new_w):
            # Collect values in block
            values = []
            for dr in range(factor):
                for dc in range(factor):
                    sr, sc = r * factor + dr, c * factor + dc
                    if sr < h and sc < w:
                        values.append(int(g[sr][sc]))
            
            if not values:
                continue
            
            if mode == "mode":
                out[r][c] = Counter(values).most_common(1)[0][0]
            elif mode == "min":
                out[r][c] = min(values)
            elif mode == "max":
                out[r][c] = max(values)
            else:
                out[r][c] = values[0]
    
    return out


def fill_by_count(g: GridV124, *, count_color: int = 1, fill_color: int = 2, bg: int = 0) -> GridV124:
    """
    Preenche região com número de elementos igual a uma contagem.
    
    Papel: Traduz contagem quantitativa em ação construtiva.
    Justificativa: Acelera tasks onde output codifica quantidade derivada do input.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    count_color = _check_color_v161(count_color)
    fill_color = _check_color_v161(fill_color)
    bg = _check_color_v161(bg)
    
    # Count occurrences of count_color
    count = sum(1 for r in range(h) for c in range(w) if int(g[r][c]) == count_color)
    
    out = _copy_grid_v161(g)
    
    # Fill 'count' cells with fill_color (in bg cells)
    filled = 0
    for r in range(h):
        for c in range(w):
            if filled >= count:
                break
            if int(out[r][c]) == bg:
                out[r][c] = fill_color
                filled += 1
        if filled >= count:
            break
    
    return out


def scale_by_count(g: GridV124, *, count_color: int = 1, target_obj: int = 0, bg: int = 0) -> GridV124:
    """
    Escala objeto proporcionalmente a uma contagem.
    
    Papel: Liga magnitude de propriedade do input ao tamanho do output.
    Justificativa: Permite saídas proporcionais a medidas do input.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    count_color = _check_color_v161(count_color)
    bg = _check_color_v161(bg)
    
    # Count
    count = sum(1 for r in range(h) for c in range(w) if int(g[r][c]) == count_color)
    if count == 0:
        count = 1
    
    objects = _find_objects_v161(g, bg=bg)
    if target_obj >= len(objects):
        return _copy_grid_v161(g)
    
    obj = objects[target_obj]
    obj_grid = _extract_object_v161(g, obj, bg=bg)
    
    # Scale by count
    return enlarge_pattern(obj_grid, factor=count)


def duplicate_by_count(g: GridV124, *, count_color: int = 1, target_obj: int = 0, bg: int = 0) -> GridV124:
    """
    Duplica objeto N vezes, onde N é uma contagem.
    
    Papel: Reproduz elementos na quantidade exata requerida.
    Justificativa: Automatiza respostas tipo "repita X tantas vezes quanto Y".
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    count_color = _check_color_v161(count_color)
    bg = _check_color_v161(bg)
    
    # Count
    count = sum(1 for r in range(h) for c in range(w) if int(g[r][c]) == count_color)
    if count == 0:
        return _copy_grid_v161(g)
    
    objects = _find_objects_v161(g, bg=bg)
    if target_obj >= len(objects):
        return _copy_grid_v161(g)
    
    obj = objects[target_obj]
    obj_grid = _extract_object_v161(g, obj, bg=bg)
    oh, ow = _grid_shape_v161(obj_grid)
    
    # Create output with 'count' copies horizontally
    new_w = ow * count + (count - 1)  # Add spacing
    out = _new_grid_v161(oh, new_w, bg)
    
    for i in range(count):
        offset = i * (ow + 1)
        for r in range(oh):
            for c in range(ow):
                if offset + c < new_w:
                    out[r][offset + c] = int(obj_grid[r][c])
    
    return out


def radial_pattern(g: GridV124, *, copies: int = 4, center: Optional[Tuple[int, int]] = None, bg: int = 0) -> GridV124:
    """
    Posiciona múltiplas cópias rotacionadas ao redor de um ponto central.
    
    Papel: Gera simetrias radiais ou repetições angulares.
    Justificativa: Atende tasks onde objeto deve aparecer em várias orientações.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0 or copies < 1:
        return _copy_grid_v161(g)
    
    bg = _check_color_v161(bg)
    
    if center is None:
        center = (h // 2, w // 2)
    
    # Create larger canvas for radial pattern
    max_dim = max(h, w) * 3
    out = _new_grid_v161(max_dim, max_dim, bg)
    
    angle_step = 360 / copies
    
    for i in range(copies):
        angle = i * angle_step
        # For simplicity, we use 90° increments
        rotations = int(angle / 90) % 4
        
        rotated = g
        for _ in range(rotations):
            rotated = rotate_270(rotated)
        
        # Place at offset from center
        rh, rw = _grid_shape_v161(rotated)
        offset_r = center[0] + max_dim // 2 - rh // 2
        offset_c = center[1] + max_dim // 2 - rw // 2 + i * (rw + 1)
        
        for r in range(rh):
            for c in range(rw):
                if 0 <= offset_r + r < max_dim and 0 <= offset_c + c < max_dim:
                    if int(rotated[r][c]) != bg:
                        out[offset_r + r][offset_c + c] = int(rotated[r][c])
    
    # Crop to content
    min_r, max_r, min_c, max_c = max_dim, 0, max_dim, 0
    for r in range(max_dim):
        for c in range(max_dim):
            if int(out[r][c]) != bg:
                min_r = min(min_r, r)
                max_r = max(max_r, r)
                min_c = min(min_c, c)
                max_c = max(max_c, c)
    
    if max_r >= min_r and max_c >= min_c:
        return crop_region(out, r1=min_r, c1=min_c, r2=max_r, c2=max_c)
    
    return _copy_grid_v161(g)


def stretch_pattern(g: GridV124, *, factor_h: int = 1, factor_w: int = 2) -> GridV124:
    """
    Escala padrão de forma não-uniforme.
    
    Papel: Ajusta proporção de desenho (esticar em um eixo mais que outro).
    Justificativa: Amplia transformações geométricas além de escalas uniformes.
    """
    h, w = _grid_shape_v161(g)
    if h == 0 or w == 0:
        return _copy_grid_v161(g)
    
    new_h = h * max(1, factor_h)
    new_w = w * max(1, factor_w)
    
    out = _new_grid_v161(new_h, new_w)
    
    for r in range(new_h):
        for c in range(new_w):
            sr = r // factor_h if factor_h > 0 else r
            sc = c // factor_w if factor_w > 0 else c
            if sr < h and sc < w:
                out[r][c] = int(g[sr][sc])
    
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OPERATOR REGISTRY
# ─────────────────────────────────────────────────────────────────────────────


CONCEPTUAL_OPERATORS_V161 = {
    # Geometric transformations
    "rotate_270": rotate_270,
    "reflect_diagonal": reflect_diagonal,
    "reflect_antidiagonal": reflect_antidiagonal,
    "translate_pattern": translate_pattern,
    "wraparound_shift": wraparound_shift,
    "transpose_grid": transpose_grid,
    "shear_pattern": shear_pattern,
    "scale_to_fit": scale_to_fit,
    
    # Color manipulation
    "invert_colors": invert_colors,
    "rotate_color_palette": rotate_color_palette,
    "quantize_colors": quantize_colors,
    "merge_colors": merge_colors,
    "isolate_color": isolate_color,
    "remove_color": remove_color,
    "map_colors": map_colors,
    
    # Shape operations
    "outline_shape": outline_shape,
    "fill_shape": fill_shape,
    "connect_points": connect_points,
    "bridge_gap": bridge_gap,
    "remove_small_objects": remove_small_objects,
    "remove_large_objects": remove_large_objects,
    "duplicate_object": duplicate_object,
    "extract_object": extract_object,
    "reposition_object": reposition_object,
    "align_objects": align_objects,
    "flood_fill": flood_fill,
    "crop_region": crop_region,
    
    # Pattern operations
    "tile_pattern": tile_pattern,
    "repeat_mirror": repeat_mirror,
    "pattern_completion": pattern_completion,
    "extrapolate_sequence": extrapolate_sequence,
    "combine_patterns": combine_patterns,
    "replace_pattern": replace_pattern,
    "mirror_copy": mirror_copy,
    "mask_pattern": mask_pattern,
    "sort_objects": sort_objects,
    
    # Scale operations
    "enlarge_pattern": enlarge_pattern,
    "shrink_pattern": shrink_pattern,
    "fill_by_count": fill_by_count,
    "scale_by_count": scale_by_count,
    "duplicate_by_count": duplicate_by_count,
    "radial_pattern": radial_pattern,
    "stretch_pattern": stretch_pattern,
}


def get_conceptual_operator_v161(name: str) -> Optional[Callable]:
    """Get a conceptual operator by name."""
    return CONCEPTUAL_OPERATORS_V161.get(name)


def list_conceptual_operators_v161() -> List[str]:
    """List all available conceptual operators."""
    return list(CONCEPTUAL_OPERATORS_V161.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
Total de Operadores Conceituais Implementados: 44

CATEGORIAS:
1. Transformações Geométricas (8):
   - rotate_270, reflect_diagonal, reflect_antidiagonal
   - translate_pattern, wraparound_shift, transpose_grid
   - shear_pattern, scale_to_fit

2. Manipulação de Cores (7):
   - invert_colors, rotate_color_palette, quantize_colors
   - merge_colors, isolate_color, remove_color, map_colors

3. Operações de Forma (12):
   - outline_shape, fill_shape, connect_points, bridge_gap
   - remove_small_objects, remove_large_objects
   - duplicate_object, extract_object, reposition_object
   - align_objects, flood_fill, crop_region

4. Operações de Padrão (9):
   - tile_pattern, repeat_mirror, pattern_completion
   - extrapolate_sequence, combine_patterns, replace_pattern
   - mirror_copy, mask_pattern, sort_objects

5. Operações de Escala (7):
   - enlarge_pattern, shrink_pattern, fill_by_count
   - scale_by_count, duplicate_by_count, radial_pattern
   - stretch_pattern
"""
