"""
cognitive_miners_v161.py - Mineradores de Conceito

Esta expansão implementa todos os mineradores de conceito autorizados para
alcançar ≥90% de acerto nos benchmarks ARC-AGI-1 e ARC-AGI-2.

Os mineradores de conceito realizam:
- Descoberta de analogias entre exemplos
- Clustering de padrões similares
- Identificação de diferenças sistemáticas
- Generalização e especialização de regras
- Exploração exaustiva de transformações

Schema version: 161
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
import math
from collections import Counter

# Type alias
GridV124 = List[List[int]]

COGNITIVE_MINERS_SCHEMA_VERSION_V161 = 161


# ─────────────────────────────────────────────────────────────────────────────
# Mining Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class MiningResult:
    """Result of a mining operation."""
    concepts: List[Dict[str, Any]]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Analogy:
    """Represents a discovered analogy."""
    source: GridV124
    target: GridV124
    mapping: Dict[str, Any]
    strength: float
    

@dataclass
class Pattern:
    """Represents a discovered pattern."""
    template: GridV124
    occurrences: List[Tuple[int, int]]  # (row, col) positions
    confidence: float
    

@dataclass
class Transformation:
    """Represents a discovered transformation."""
    name: str
    params: Dict[str, Any]
    success_rate: float
    examples: List[Tuple[GridV124, GridV124]]


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


def _grid_diff_v161(g1: GridV124, g2: GridV124) -> List[Tuple[int, int, int, int]]:
    """Find differences between grids: [(r, c, color1, color2), ...]"""
    h1, w1 = _grid_shape_v161(g1)
    h2, w2 = _grid_shape_v161(g2)
    
    diffs = []
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    
    for r in range(max_h):
        for c in range(max_w):
            c1 = int(g1[r][c]) if r < h1 and c < w1 else -1
            c2 = int(g2[r][c]) if r < h2 and c < w2 else -1
            if c1 != c2:
                diffs.append((r, c, c1, c2))
    
    return diffs


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


def _extract_subgrid_v161(g: GridV124, r: int, c: int, h: int, w: int, *, bg: int = 0) -> GridV124:
    """Extract subgrid from position (r, c) with size (h, w)."""
    gh, gw = _grid_shape_v161(g)
    result = [[bg for _ in range(w)] for _ in range(h)]
    
    for i in range(h):
        for j in range(w):
            gr, gc = r + i, c + j
            if 0 <= gr < gh and 0 <= gc < gw:
                result[i][j] = int(g[gr][gc])
    
    return result


def _count_colors_v161(g: GridV124) -> Dict[int, int]:
    """Count color frequencies."""
    counts: Dict[int, int] = {}
    for row in g:
        for cell in row:
            color = int(cell)
            counts[color] = counts.get(color, 0) + 1
    return counts


# ─────────────────────────────────────────────────────────────────────────────
# 1. ANALOGY MINERS
# ─────────────────────────────────────────────────────────────────────────────


def analogy_miner(examples: List[Tuple[GridV124, GridV124]]) -> MiningResult:
    """
    Tenta inferir regras por analogia entre input/output.
    
    Papel: Detecta semelhanças estruturais entre exemplos para formar hipótese.
    Justificativa: A analogia é fundamental para transferência de conhecimento.
    """
    if not examples:
        return MiningResult(concepts=[], confidence=0.0)
    
    analogies = []
    
    for i, (in1, out1) in enumerate(examples):
        for j, (in2, out2) in enumerate(examples):
            if i >= j:
                continue
            
            # Compare input differences
            in_diffs = _grid_diff_v161(in1, in2)
            out_diffs = _grid_diff_v161(out1, out2)
            
            # Look for consistent mappings
            if len(in_diffs) > 0 and len(out_diffs) > 0:
                # Check if color mappings are consistent
                in_color_changes = [(d[2], d[3]) for d in in_diffs if d[2] >= 0 and d[3] >= 0]
                out_color_changes = [(d[2], d[3]) for d in out_diffs if d[2] >= 0 and d[3] >= 0]
                
                if in_color_changes and out_color_changes:
                    analogies.append({
                        "type": "color_transform",
                        "examples": (i, j),
                        "in_changes": in_color_changes[:5],
                        "out_changes": out_color_changes[:5]
                    })
    
    # Compute shape analogies
    for i, (inp, out) in enumerate(examples):
        in_shape = _grid_shape_v161(inp)
        out_shape = _grid_shape_v161(out)
        
        if in_shape != out_shape:
            analogies.append({
                "type": "shape_transform",
                "example": i,
                "in_shape": in_shape,
                "out_shape": out_shape,
                "ratio_h": out_shape[0] / in_shape[0] if in_shape[0] > 0 else 0,
                "ratio_w": out_shape[1] / in_shape[1] if in_shape[1] > 0 else 0
            })
    
    confidence = min(1.0, len(analogies) / max(1, len(examples)))
    
    return MiningResult(
        concepts=analogies,
        confidence=confidence,
        metadata={"num_examples": len(examples)}
    )


def transfer_miner(source_examples: List[Tuple[GridV124, GridV124]], 
                   target_input: GridV124) -> MiningResult:
    """
    Minera regras que se transferem de outros contextos.
    
    Papel: Tenta reutilizar regras aprendidas em exemplos anteriores.
    Justificativa: Transferência de conceitos acelera aprendizado.
    """
    if not source_examples:
        return MiningResult(concepts=[], confidence=0.0)
    
    transferable = []
    target_shape = _grid_shape_v161(target_input)
    target_colors = set(_count_colors_v161(target_input).keys())
    
    for i, (inp, out) in enumerate(source_examples):
        in_shape = _grid_shape_v161(inp)
        out_shape = _grid_shape_v161(out)
        in_colors = set(_count_colors_v161(inp).keys())
        
        # Check shape compatibility
        shape_compatible = (in_shape == target_shape or 
                          (in_shape[0] == target_shape[0]) or
                          (in_shape[1] == target_shape[1]))
        
        # Check color overlap
        color_overlap = len(target_colors & in_colors) / max(1, len(target_colors | in_colors))
        
        if shape_compatible or color_overlap > 0.5:
            transferable.append({
                "source_example": i,
                "shape_compatible": shape_compatible,
                "color_overlap": color_overlap,
                "transform_shape": (in_shape, out_shape)
            })
    
    confidence = len(transferable) / max(1, len(source_examples))
    
    return MiningResult(
        concepts=transferable,
        confidence=confidence,
        metadata={"target_shape": target_shape}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. PATTERN MINERS
# ─────────────────────────────────────────────────────────────────────────────


def pattern_cluster_miner(g: GridV124, *, min_pattern_size: int = 2, bg: int = 0) -> MiningResult:
    """
    Agrupa sub-padrões semelhantes dentro de um grid.
    
    Papel: Identifica repetições ou micro-estruturas comuns.
    Justificativa: Útil para detectar tiles ou motivos repetitivos.
    """
    h, w = _grid_shape_v161(g)
    patterns: Dict[str, List[Tuple[int, int]]] = {}
    
    # Extract all subgrids of size min_pattern_size x min_pattern_size
    for r in range(h - min_pattern_size + 1):
        for c in range(w - min_pattern_size + 1):
            subgrid = _extract_subgrid_v161(g, r, c, min_pattern_size, min_pattern_size, bg=bg)
            key = str(subgrid)
            
            if key not in patterns:
                patterns[key] = []
            patterns[key].append((r, c))
    
    # Filter to patterns that appear multiple times
    clusters = []
    for key, positions in patterns.items():
        if len(positions) >= 2:
            clusters.append({
                "pattern": eval(key),
                "positions": positions,
                "count": len(positions)
            })
    
    # Sort by frequency
    clusters.sort(key=lambda x: -x["count"])
    
    confidence = len(clusters) / max(1, len(patterns))
    
    return MiningResult(
        concepts=clusters[:20],  # Top 20 patterns
        confidence=confidence,
        metadata={"total_patterns": len(patterns)}
    )


def difference_miner(examples: List[Tuple[GridV124, GridV124]]) -> MiningResult:
    """
    Minera diferenças sistemáticas input→output.
    
    Papel: Foca no que muda, para descobrir regras.
    Justificativa: Se a maioria do grid não muda, entender a diferença é chave.
    """
    if not examples:
        return MiningResult(concepts=[], confidence=0.0)
    
    all_diffs = []
    
    for i, (inp, out) in enumerate(examples):
        diffs = _grid_diff_v161(inp, out)
        
        # Analyze diff patterns
        color_changes: Dict[Tuple[int, int], int] = {}
        position_changes: List[Tuple[int, int]] = []
        
        for r, c, old, new in diffs:
            if old >= 0 and new >= 0:
                color_changes[(old, new)] = color_changes.get((old, new), 0) + 1
            position_changes.append((r, c))
        
        all_diffs.append({
            "example": i,
            "num_changes": len(diffs),
            "color_changes": dict(color_changes),
            "changed_region_size": len(position_changes)
        })
    
    # Find common patterns across examples
    common_color_changes = Counter()
    for diff in all_diffs:
        for (old, new), count in diff["color_changes"].items():
            common_color_changes[(old, new)] += 1
    
    concepts = []
    for (old, new), freq in common_color_changes.most_common(10):
        concepts.append({
            "type": "color_replacement",
            "from": old,
            "to": new,
            "frequency": freq / len(examples)
        })
    
    confidence = len(concepts) / max(1, len(examples))
    
    return MiningResult(
        concepts=concepts,
        confidence=min(1.0, confidence),
        metadata={"all_diffs": all_diffs}
    )


def gap_miner(g: GridV124, *, bg: int = 0) -> MiningResult:
    """
    Detecta "lacunas" em padrões que indicam regiões a serem preenchidas.
    
    Papel: Identifica áreas onde falta algo conforme o padrão.
    Justificativa: Padrões com lacunas pedem completamento.
    """
    h, w = _grid_shape_v161(g)
    
    # Find potential gaps - background cells surrounded by non-background
    gaps = []
    
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) == bg:
                # Check neighborhood
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        neighbors.append(int(g[nr][nc]))
                
                # If surrounded mostly by non-bg, it's a gap
                non_bg_count = sum(1 for n in neighbors if n != bg)
                if non_bg_count >= 3:  # 3 out of 4 neighbors are non-bg
                    gaps.append({
                        "position": (r, c),
                        "neighbors": neighbors,
                        "suggested_fill": max(set([n for n in neighbors if n != bg]), 
                                            key=neighbors.count) if neighbors else bg
                    })
    
    confidence = min(1.0, len(gaps) / max(1, h * w * 0.1))
    
    return MiningResult(
        concepts=gaps,
        confidence=confidence,
        metadata={"grid_shape": (h, w)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERALIZATION/SPECIALIZATION MINERS
# ─────────────────────────────────────────────────────────────────────────────


def generalization_miner(examples: List[Tuple[GridV124, GridV124]]) -> MiningResult:
    """
    Tenta generalizar regras a partir de vários exemplos específicos.
    
    Papel: Constrói regra mais abrangente que cubra todos os exemplos.
    Justificativa: Uma regra geral é mais robusta para novos inputs.
    """
    if not examples:
        return MiningResult(concepts=[], confidence=0.0)
    
    generalizations = []
    
    # Analyze shape transformations
    shape_transforms = []
    for inp, out in examples:
        in_shape = _grid_shape_v161(inp)
        out_shape = _grid_shape_v161(out)
        shape_transforms.append((in_shape, out_shape))
    
    # Check for consistent shape rule
    if len(set(shape_transforms)) == 1:
        in_s, out_s = shape_transforms[0]
        generalizations.append({
            "type": "fixed_shape_transform",
            "rule": f"({in_s[0]}, {in_s[1]}) -> ({out_s[0]}, {out_s[1]})",
            "confidence": 1.0
        })
    else:
        # Check for ratio-based rule
        ratios = [(out_s[0] / max(1, in_s[0]), out_s[1] / max(1, in_s[1])) 
                  for in_s, out_s in shape_transforms]
        if len(set(ratios)) == 1:
            generalizations.append({
                "type": "ratio_shape_transform",
                "ratio": ratios[0],
                "confidence": 1.0
            })
    
    # Analyze color mappings
    all_color_maps: List[Dict[int, int]] = []
    for inp, out in examples:
        h, w = _grid_shape_v161(inp)
        color_map: Dict[int, int] = {}
        
        oh, ow = _grid_shape_v161(out)
        if h == oh and w == ow:
            for r in range(h):
                for c in range(w):
                    in_c = int(inp[r][c])
                    out_c = int(out[r][c])
                    if in_c != out_c:
                        if in_c not in color_map:
                            color_map[in_c] = out_c
        
        all_color_maps.append(color_map)
    
    # Find consistent color mappings
    if all_color_maps:
        consistent = {}
        for in_c in range(10):
            out_colors = [m.get(in_c) for m in all_color_maps if in_c in m]
            if out_colors and len(set(out_colors)) == 1:
                consistent[in_c] = out_colors[0]
        
        if consistent:
            generalizations.append({
                "type": "color_mapping",
                "mapping": consistent,
                "confidence": len(consistent) / 10
            })
    
    confidence = len(generalizations) / max(1, 3)  # Expected ~3 generalizations
    
    return MiningResult(
        concepts=generalizations,
        confidence=min(1.0, confidence),
        metadata={"num_examples": len(examples)}
    )


def specialization_miner(general_rule: Dict[str, Any], 
                         examples: List[Tuple[GridV124, GridV124]]) -> MiningResult:
    """
    Refina regra geral para sub-casos específicos.
    
    Papel: Particulariza hipótese conforme detalhes dos exemplos.
    Justificativa: Às vezes regra geral falha em subcasos, necessitando ajustes.
    """
    if not examples or not general_rule:
        return MiningResult(concepts=[], confidence=0.0)
    
    specializations = []
    rule_type = general_rule.get("type", "unknown")
    
    for i, (inp, out) in enumerate(examples):
        # Check if general rule applies perfectly
        in_shape = _grid_shape_v161(inp)
        out_shape = _grid_shape_v161(out)
        
        special_case = {
            "example": i,
            "deviations": []
        }
        
        if rule_type == "ratio_shape_transform":
            expected_ratio = general_rule.get("ratio", (1, 1))
            actual_ratio = (out_shape[0] / max(1, in_shape[0]), 
                          out_shape[1] / max(1, in_shape[1]))
            
            if actual_ratio != expected_ratio:
                special_case["deviations"].append({
                    "type": "shape_ratio_mismatch",
                    "expected": expected_ratio,
                    "actual": actual_ratio
                })
        
        elif rule_type == "color_mapping":
            mapping = general_rule.get("mapping", {})
            h, w = _grid_shape_v161(inp)
            oh, ow = _grid_shape_v161(out)
            
            if h == oh and w == ow:
                for r in range(h):
                    for c in range(w):
                        in_c = int(inp[r][c])
                        out_c = int(out[r][c])
                        expected_out = mapping.get(in_c, in_c)
                        
                        if out_c != expected_out:
                            special_case["deviations"].append({
                                "type": "color_exception",
                                "position": (r, c),
                                "in": in_c,
                                "expected_out": expected_out,
                                "actual_out": out_c
                            })
        
        if special_case["deviations"]:
            specializations.append(special_case)
    
    confidence = 1.0 - (len(specializations) / max(1, len(examples)))
    
    return MiningResult(
        concepts=specializations,
        confidence=confidence,
        metadata={"general_rule": rule_type}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. EXHAUSTIVE/EXPLORATORY MINERS
# ─────────────────────────────────────────────────────────────────────────────


def bruteforce_miner(inp: GridV124, out: GridV124, 
                     operators: Dict[str, Callable]) -> MiningResult:
    """
    Testa exaustivamente operadores em busca de match input→output.
    
    Papel: Encontra qualquer operador que transforme input em output esperado.
    Justificativa: Útil para descobrir regras simples que não são óbvias.
    """
    matches = []
    
    for name, op in operators.items():
        try:
            # Try operator with default params
            result = op(inp)
            
            if _grids_equal_v161(result, out):
                matches.append({
                    "operator": name,
                    "params": {},
                    "match_type": "exact"
                })
        except:
            pass
    
    confidence = 1.0 if matches else 0.0
    
    return MiningResult(
        concepts=matches,
        confidence=confidence,
        metadata={"operators_tested": len(operators)}
    )


def exploratory_miner(g: GridV124, *, depth: int = 2, bg: int = 0) -> MiningResult:
    """
    Explora possíveis transformações não direcionadas.
    
    Papel: Gera pool de transformações candidatas.
    Justificativa: Antes de ter hipótese, exploramos livremente.
    """
    h, w = _grid_shape_v161(g)
    explorations = []
    
    # Explore symmetry properties
    # Horizontal flip
    h_flip = [[g[r][w-1-c] for c in range(w)] for r in range(h)]
    is_h_symmetric = _grids_equal_v161(g, h_flip)
    
    # Vertical flip
    v_flip = [[g[h-1-r][c] for c in range(w)] for r in range(h)]
    is_v_symmetric = _grids_equal_v161(g, v_flip)
    
    explorations.append({
        "property": "symmetry",
        "horizontal": is_h_symmetric,
        "vertical": is_v_symmetric
    })
    
    # Explore color distribution
    colors = _count_colors_v161(g)
    dominant = max(colors, key=lambda k: colors[k]) if colors else 0
    
    explorations.append({
        "property": "color_distribution",
        "colors": colors,
        "dominant": dominant,
        "num_colors": len(colors)
    })
    
    # Explore object count
    objects = _find_objects_v161(g, bg=bg)
    
    explorations.append({
        "property": "objects",
        "count": len(objects),
        "sizes": sorted([len(obj) for obj in objects], reverse=True)[:5]
    })
    
    # Explore periodicity
    periodicities = []
    for period in range(2, min(h, w) // 2 + 1):
        is_periodic_h = all(
            int(g[r][c]) == int(g[r][c % period])
            for r in range(h) for c in range(w)
        )
        is_periodic_v = all(
            int(g[r][c]) == int(g[r % period][c])
            for r in range(h) for c in range(w)
        )
        if is_periodic_h or is_periodic_v:
            periodicities.append({
                "period": period,
                "horizontal": is_periodic_h,
                "vertical": is_periodic_v
            })
    
    if periodicities:
        explorations.append({
            "property": "periodicity",
            "findings": periodicities
        })
    
    confidence = len(explorations) / 4  # Expected 4 exploration types
    
    return MiningResult(
        concepts=explorations,
        confidence=confidence,
        metadata={"depth": depth}
    )


def mutation_miner(g: GridV124, *, max_mutations: int = 10, bg: int = 0) -> MiningResult:
    """
    Propõe pequenas mutações e avalia quais poderiam ser regras.
    
    Papel: Gera variantes do input para entender sensibilidade.
    Justificativa: Perturbações revelam quais mudanças são significativas.
    """
    h, w = _grid_shape_v161(g)
    mutations = []
    colors = set(_count_colors_v161(g).keys())
    
    # Single cell mutations
    for r in range(min(h, 3)):
        for c in range(min(w, 3)):
            original = int(g[r][c])
            for new_color in colors:
                if new_color != original:
                    mutations.append({
                        "type": "single_cell",
                        "position": (r, c),
                        "from": original,
                        "to": new_color
                    })
                    if len(mutations) >= max_mutations:
                        break
            if len(mutations) >= max_mutations:
                break
        if len(mutations) >= max_mutations:
            break
    
    # Color swap mutations
    color_list = list(colors)
    for i, c1 in enumerate(color_list[:3]):
        for c2 in color_list[i+1:3]:
            mutations.append({
                "type": "color_swap",
                "colors": (c1, c2)
            })
    
    confidence = len(mutations) / max_mutations
    
    return MiningResult(
        concepts=mutations[:max_mutations],
        confidence=confidence,
        metadata={"original_colors": list(colors)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. COMPOSITION MINERS
# ─────────────────────────────────────────────────────────────────────────────


def composition_miner(examples: List[Tuple[GridV124, GridV124]],
                      operators: Dict[str, Callable]) -> MiningResult:
    """
    Tenta encontrar composição de operadores existentes que resolva transformação.
    
    Papel: Combina operadores conhecidos para resolver transformação.
    Justificativa: Muitas regras são composições de primitivas.
    """
    if not examples or not operators:
        return MiningResult(concepts=[], confidence=0.0)
    
    compositions = []
    
    # Try single operators first
    for name, op in list(operators.items())[:20]:  # Limit to avoid explosion
        try:
            all_match = True
            for inp, out in examples:
                result = op(inp)
                if not _grids_equal_v161(result, out):
                    all_match = False
                    break
            
            if all_match:
                compositions.append({
                    "sequence": [name],
                    "length": 1
                })
        except:
            pass
    
    # Try pairs (limited)
    if not compositions:
        op_names = list(operators.keys())[:10]
        for i, name1 in enumerate(op_names):
            for name2 in op_names[i:]:
                try:
                    all_match = True
                    for inp, out in examples:
                        temp = operators[name1](inp)
                        result = operators[name2](temp)
                        if not _grids_equal_v161(result, out):
                            all_match = False
                            break
                    
                    if all_match:
                        compositions.append({
                            "sequence": [name1, name2],
                            "length": 2
                        })
                except:
                    pass
    
    confidence = 1.0 if compositions else 0.0
    
    return MiningResult(
        concepts=compositions,
        confidence=confidence,
        metadata={"operators_available": len(operators)}
    )


def parameter_miner(examples: List[Tuple[GridV124, GridV124]],
                    operator: Callable,
                    param_ranges: Dict[str, List[Any]]) -> MiningResult:
    """
    Testa diferentes parâmetros de um operador para achar aquele que satisfaz.
    
    Papel: Encontra parâmetros ideais para um operador conhecido.
    Justificativa: Achar parâmetros corretos pode ser crucial.
    """
    if not examples or not param_ranges:
        return MiningResult(concepts=[], confidence=0.0)
    
    successful_params = []
    
    # Generate parameter combinations (limited)
    param_names = list(param_ranges.keys())
    
    if len(param_names) == 1:
        for val in param_ranges[param_names[0]]:
            try:
                params = {param_names[0]: val}
                all_match = True
                
                for inp, out in examples:
                    result = operator(inp, **params)
                    if not _grids_equal_v161(result, out):
                        all_match = False
                        break
                
                if all_match:
                    successful_params.append(params)
            except:
                pass
    
    confidence = len(successful_params) / max(1, len(param_ranges.get(param_names[0], [1])))
    
    return MiningResult(
        concepts=successful_params,
        confidence=min(1.0, confidence),
        metadata={"param_ranges": {k: len(v) for k, v in param_ranges.items()}}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. GOAL-ORIENTED MINERS
# ─────────────────────────────────────────────────────────────────────────────


def subgoal_miner(inp: GridV124, out: GridV124, *, bg: int = 0) -> MiningResult:
    """
    Decompõe transformação em metas intermediárias.
    
    Papel: Estabelece sub-objetivos para atingir output final.
    Justificativa: Problemas complexos são resolvidos em etapas.
    """
    subgoals = []
    
    h1, w1 = _grid_shape_v161(inp)
    h2, w2 = _grid_shape_v161(out)
    
    # Shape change subgoal
    if h1 != h2 or w1 != w2:
        subgoals.append({
            "type": "resize",
            "from_shape": (h1, w1),
            "to_shape": (h2, w2),
            "priority": 1
        })
    
    # Color change subgoals
    in_colors = set(_count_colors_v161(inp).keys())
    out_colors = set(_count_colors_v161(out).keys())
    
    removed = in_colors - out_colors
    added = out_colors - in_colors
    
    if removed:
        subgoals.append({
            "type": "remove_colors",
            "colors": list(removed),
            "priority": 2
        })
    
    if added:
        subgoals.append({
            "type": "add_colors",
            "colors": list(added),
            "priority": 3
        })
    
    # Object count change
    in_objects = len(_find_objects_v161(inp, bg=bg))
    out_objects = len(_find_objects_v161(out, bg=bg))
    
    if in_objects != out_objects:
        subgoals.append({
            "type": "change_object_count",
            "from": in_objects,
            "to": out_objects,
            "priority": 4
        })
    
    confidence = len(subgoals) / 4  # Expected max 4 subgoals
    
    return MiningResult(
        concepts=subgoals,
        confidence=min(1.0, confidence),
        metadata={"decomposition_depth": len(subgoals)}
    )


def exception_miner(examples: List[Tuple[GridV124, GridV124]],
                    rule: Dict[str, Any]) -> MiningResult:
    """
    Identifica exceções a uma regra principal.
    
    Papel: Registra casos que fogem da regra geral.
    Justificativa: Toda regra tem exceções que devem ser documentadas.
    """
    if not examples:
        return MiningResult(concepts=[], confidence=0.0)
    
    exceptions = []
    
    for i, (inp, out) in enumerate(examples):
        # Check if this example matches expected rule behavior
        # This is a simplified check - would need actual rule application
        
        in_shape = _grid_shape_v161(inp)
        out_shape = _grid_shape_v161(out)
        
        expected_shape = rule.get("expected_shape")
        if expected_shape and out_shape != expected_shape:
            exceptions.append({
                "example": i,
                "type": "shape_exception",
                "expected": expected_shape,
                "actual": out_shape
            })
        
        expected_mapping = rule.get("color_mapping", {})
        if expected_mapping:
            mismatches = []
            h, w = min(_grid_shape_v161(inp), _grid_shape_v161(out))
            
            for r in range(h):
                for c in range(w):
                    in_c = int(inp[r][c])
                    out_c = int(out[r][c])
                    expected_out = expected_mapping.get(in_c, in_c)
                    
                    if out_c != expected_out:
                        mismatches.append((r, c, expected_out, out_c))
            
            if mismatches:
                exceptions.append({
                    "example": i,
                    "type": "color_mapping_exception",
                    "mismatches": mismatches[:5]
                })
    
    # Confidence is higher if fewer exceptions
    confidence = 1.0 - (len(exceptions) / max(1, len(examples)))
    
    return MiningResult(
        concepts=exceptions,
        confidence=confidence,
        metadata={"rule_type": rule.get("type", "unknown")}
    )


def curriculum_miner(examples: List[Tuple[GridV124, GridV124]]) -> MiningResult:
    """
    Ordena exemplos por dificuldade inferida.
    
    Papel: Determina sequência ótima para aprender regras gradualmente.
    Justificativa: Aprender do simples ao complexo facilita generalização.
    """
    if not examples:
        return MiningResult(concepts=[], confidence=0.0)
    
    difficulties = []
    
    for i, (inp, out) in enumerate(examples):
        in_h, in_w = _grid_shape_v161(inp)
        out_h, out_w = _grid_shape_v161(out)
        
        # Compute complexity metrics
        in_colors = len(_count_colors_v161(inp))
        out_colors = len(_count_colors_v161(out))
        in_objects = len(_find_objects_v161(inp))
        out_objects = len(_find_objects_v161(out))
        
        diffs = _grid_diff_v161(inp, out)
        
        # Difficulty score
        difficulty = (
            in_h * in_w +  # Grid size
            in_colors * 2 +  # Color complexity
            in_objects * 3 +  # Object complexity
            len(diffs) * 2 +  # Change complexity
            abs(in_h - out_h) * 5 +  # Shape change penalty
            abs(in_w - out_w) * 5
        )
        
        difficulties.append({
            "example": i,
            "difficulty": difficulty,
            "metrics": {
                "grid_size": in_h * in_w,
                "colors": in_colors,
                "objects": in_objects,
                "changes": len(diffs)
            }
        })
    
    # Sort by difficulty
    difficulties.sort(key=lambda x: x["difficulty"])
    
    # Add order field
    for order, item in enumerate(difficulties):
        item["recommended_order"] = order + 1
    
    confidence = 1.0  # Always produces ordering
    
    return MiningResult(
        concepts=difficulties,
        confidence=confidence,
        metadata={"num_examples": len(examples)}
    )


def inversion_miner(examples: List[Tuple[GridV124, GridV124]]) -> MiningResult:
    """
    Tenta aprender regra reversa (output→input).
    
    Papel: Se soubermos inverter, entendemos melhor a transformação.
    Justificativa: Reversibilidade indica compreensão profunda.
    """
    # Swap input and output to mine reverse transformation
    reversed_examples = [(out, inp) for inp, out in examples]
    
    # Use generalization miner on reversed examples
    reverse_result = generalization_miner(reversed_examples)
    
    # Wrap result
    concepts = [{
        "type": "inverse_rule",
        "forward_to_backward": True,
        "reverse_generalizations": reverse_result.concepts
    }]
    
    return MiningResult(
        concepts=concepts,
        confidence=reverse_result.confidence,
        metadata={"method": "reversed_generalization"}
    )


def reinforcement_miner(examples: List[Tuple[GridV124, GridV124]],
                        feedback: List[Dict[str, Any]]) -> MiningResult:
    """
    Ajusta mineração com base em feedback externo.
    
    Papel: Refina hipóteses conforme acertos/erros passados.
    Justificativa: Feedback direciona aprendizado eficiente.
    """
    if not examples or not feedback:
        return MiningResult(concepts=[], confidence=0.0)
    
    # Analyze feedback to guide mining
    successful_patterns = []
    failed_patterns = []
    
    for fb in feedback:
        if fb.get("success"):
            successful_patterns.append(fb.get("pattern_used"))
        else:
            failed_patterns.append(fb.get("pattern_used"))
    
    # Mine focusing on successful patterns
    reinforced_concepts = []
    
    success_types = Counter([p.get("type") for p in successful_patterns if p])
    fail_types = Counter([p.get("type") for p in failed_patterns if p])
    
    for pattern_type, count in success_types.most_common(5):
        if pattern_type:
            reinforced_concepts.append({
                "type": pattern_type,
                "success_count": count,
                "fail_count": fail_types.get(pattern_type, 0),
                "recommendation": "prefer" if count > fail_types.get(pattern_type, 0) else "avoid"
            })
    
    confidence = len(reinforced_concepts) / max(1, len(success_types))
    
    return MiningResult(
        concepts=reinforced_concepts,
        confidence=min(1.0, confidence),
        metadata={"total_feedback": len(feedback)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# MINER REGISTRY
# ─────────────────────────────────────────────────────────────────────────────


CONCEPT_MINERS_V161 = {
    # Analogy miners
    "analogy_miner": analogy_miner,
    "transfer_miner": transfer_miner,
    
    # Pattern miners
    "pattern_cluster_miner": pattern_cluster_miner,
    "difference_miner": difference_miner,
    "gap_miner": gap_miner,
    
    # Generalization/Specialization
    "generalization_miner": generalization_miner,
    "specialization_miner": specialization_miner,
    
    # Exhaustive/Exploratory
    "bruteforce_miner": bruteforce_miner,
    "exploratory_miner": exploratory_miner,
    "mutation_miner": mutation_miner,
    
    # Composition
    "composition_miner": composition_miner,
    "parameter_miner": parameter_miner,
    
    # Goal-oriented
    "subgoal_miner": subgoal_miner,
    "exception_miner": exception_miner,
    "curriculum_miner": curriculum_miner,
    "inversion_miner": inversion_miner,
    "reinforcement_miner": reinforcement_miner,
}


def get_concept_miner_v161(name: str):
    """Get a concept miner by name."""
    return CONCEPT_MINERS_V161.get(name)


def list_concept_miners_v161() -> List[str]:
    """List all available concept miners."""
    return list(CONCEPT_MINERS_V161.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
Total de Mineradores de Conceito Implementados: 17

CATEGORIAS:
1. Mineradores de Analogia (2):
   - analogy_miner, transfer_miner

2. Mineradores de Padrão (3):
   - pattern_cluster_miner, difference_miner, gap_miner

3. Generalização/Especialização (2):
   - generalization_miner, specialization_miner

4. Exaustivos/Exploratórios (3):
   - bruteforce_miner, exploratory_miner, mutation_miner

5. Composição (2):
   - composition_miner, parameter_miner

6. Orientados a Meta (5):
   - subgoal_miner, exception_miner, curriculum_miner
   - inversion_miner, reinforcement_miner
"""
