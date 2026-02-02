"""
cognitive_compression_v161.py - Ferramentas de Compressão

Esta expansão implementa todas as ferramentas de compressão autorizadas para
alcançar ≥90% de acerto nos benchmarks ARC-AGI-1 e ARC-AGI-2.

As ferramentas de compressão realizam:
- Avaliação de comprimento de descrição mínima (MDL)
- Fusão e simplificação de conceitos
- Compressão de padrões e estados
- Codificação eficiente de features
- Poda de redundâncias

Schema version: 161
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
import math
from collections import Counter

# Type alias
GridV124 = List[List[int]]

COGNITIVE_COMPRESSION_SCHEMA_VERSION_V161 = 161


# ─────────────────────────────────────────────────────────────────────────────
# Compression Result Types
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class CompressionResult:
    """Result of a compression operation."""
    compressed: Any
    original_size: int
    compressed_size: int
    ratio: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConceptDescription:
    """Minimal description of a concept."""
    description: str
    length: int
    components: List[str]
    

@dataclass
class EncodedState:
    """Encoded representation of a grid state."""
    code: List[int]
    dictionary: Dict[str, int]
    reversible: bool


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


def _grid_to_flat_v161(g: GridV124) -> List[int]:
    """Flatten grid to 1D list."""
    return [int(cell) for row in g for cell in row]


def _flat_to_grid_v161(flat: List[int], h: int, w: int) -> GridV124:
    """Unflatten 1D list to grid."""
    return [[int(flat[r * w + c]) for c in range(w)] for r in range(h)]


def _entropy_v161(data: List[int]) -> float:
    """Calculate Shannon entropy of data."""
    if not data:
        return 0.0
    
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    
    for count in counts.values():
        if count > 0:
            p = count / total
            entropy -= p * math.log2(p)
    
    return entropy


def _run_length_encode_v161(data: List[int]) -> List[Tuple[int, int]]:
    """Run-length encode a sequence."""
    if not data:
        return []
    
    encoded = []
    current = data[0]
    count = 1
    
    for val in data[1:]:
        if val == current:
            count += 1
        else:
            encoded.append((current, count))
            current = val
            count = 1
    
    encoded.append((current, count))
    return encoded


def _run_length_decode_v161(encoded: List[Tuple[int, int]]) -> List[int]:
    """Decode run-length encoded sequence."""
    decoded = []
    for val, count in encoded:
        decoded.extend([val] * count)
    return decoded


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
# 1. MDL EVALUATORS
# ─────────────────────────────────────────────────────────────────────────────


def mdl_evaluator(g: GridV124, description: ConceptDescription) -> CompressionResult:
    """
    Calcula comprimento de descrição mínima (Minimum Description Length).
    
    Papel: Quantifica complexidade de uma representação/hipótese.
    Justificativa: Prioriza explicações mais simples (Occam's Razor).
    """
    h, w = _grid_shape_v161(g)
    original_size = h * w  # Raw cell count
    
    # Description length = length of concept + length of exceptions
    description_length = description.length
    
    # Count components
    component_cost = len(description.components) * 2  # Each component has overhead
    
    # Total MDL
    mdl = description_length + component_cost
    
    ratio = mdl / max(1, original_size)
    
    return CompressionResult(
        compressed=description,
        original_size=original_size,
        compressed_size=mdl,
        ratio=ratio,
        metadata={"type": "mdl", "components": len(description.components)}
    )


def description_length_guard(concepts: List[ConceptDescription], 
                             *, max_length: int = 100) -> CompressionResult:
    """
    Evita explosão de comprimento de descrição.
    
    Papel: Filtra conceitos muito longos.
    Justificativa: Conceitos com descrição longa indicam overfitting.
    """
    filtered = [c for c in concepts if c.length <= max_length]
    rejected = [c for c in concepts if c.length > max_length]
    
    original_count = len(concepts)
    filtered_count = len(filtered)
    
    return CompressionResult(
        compressed=filtered,
        original_size=original_count,
        compressed_size=filtered_count,
        ratio=filtered_count / max(1, original_count),
        metadata={
            "max_length": max_length,
            "rejected_count": len(rejected),
            "rejected_lengths": [c.length for c in rejected[:5]]
        }
    )


def entropy_minimizer(g: GridV124) -> CompressionResult:
    """
    Busca representação que minimize entropia.
    
    Papel: Encontra encoding mais compacto via redução de entropia.
    Justificativa: Baixa entropia = alta compressibilidade.
    """
    h, w = _grid_shape_v161(g)
    flat = _grid_to_flat_v161(g)
    
    original_entropy = _entropy_v161(flat)
    original_size = len(flat)
    
    # Try different encodings
    best_encoding = flat
    best_entropy = original_entropy
    best_name = "raw"
    
    # 1. Row-major (already have)
    row_entropy = original_entropy
    
    # 2. Column-major
    col_flat = [int(g[r][c]) for c in range(w) for r in range(h)]
    col_entropy = _entropy_v161(col_flat)
    if col_entropy < best_entropy:
        best_encoding = col_flat
        best_entropy = col_entropy
        best_name = "column_major"
    
    # 3. Diagonal traversal
    diag_flat = []
    for d in range(h + w - 1):
        for r in range(max(0, d - w + 1), min(h, d + 1)):
            c = d - r
            if 0 <= c < w:
                diag_flat.append(int(g[r][c]))
    diag_entropy = _entropy_v161(diag_flat)
    if diag_entropy < best_entropy:
        best_encoding = diag_flat
        best_entropy = diag_entropy
        best_name = "diagonal"
    
    # Compute compression ratio based on entropy bits
    bits_original = original_size * math.ceil(math.log2(max(max(flat) + 1, 2)))
    bits_compressed = int(best_entropy * len(best_encoding)) + 8  # Header overhead
    
    return CompressionResult(
        compressed={"encoding": best_name, "entropy": best_entropy},
        original_size=bits_original,
        compressed_size=bits_compressed,
        ratio=bits_compressed / max(1, bits_original),
        metadata={
            "original_entropy": original_entropy,
            "minimized_entropy": best_entropy,
            "best_encoding": best_name
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. CONCEPT MERGERS/SIMPLIFIERS
# ─────────────────────────────────────────────────────────────────────────────


def concept_merger(concepts: List[Dict[str, Any]], 
                   *, similarity_threshold: float = 0.8) -> CompressionResult:
    """
    Combina conceitos similares em um único.
    
    Papel: Reduz redundância agrupando conceitos parecidos.
    Justificativa: Conceitos quase idênticos podem ser unificados.
    """
    if not concepts:
        return CompressionResult(
            compressed=[],
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    merged = []
    used = set()
    
    for i, c1 in enumerate(concepts):
        if i in used:
            continue
        
        # Find similar concepts
        cluster = [c1]
        for j, c2 in enumerate(concepts[i+1:], i+1):
            if j in used:
                continue
            
            # Simple similarity: shared keys and values
            keys1 = set(c1.keys())
            keys2 = set(c2.keys())
            key_overlap = len(keys1 & keys2) / max(1, len(keys1 | keys2))
            
            if key_overlap >= similarity_threshold:
                cluster.append(c2)
                used.add(j)
        
        # Merge cluster into single concept
        merged_concept = {}
        for c in cluster:
            for k, v in c.items():
                if k not in merged_concept:
                    merged_concept[k] = v
                elif isinstance(v, list) and isinstance(merged_concept[k], list):
                    merged_concept[k] = list(set(merged_concept[k] + v))
        
        merged_concept["_merged_from"] = len(cluster)
        merged.append(merged_concept)
        used.add(i)
    
    return CompressionResult(
        compressed=merged,
        original_size=len(concepts),
        compressed_size=len(merged),
        ratio=len(merged) / max(1, len(concepts)),
        metadata={"threshold": similarity_threshold}
    )


def concept_simplifier(concept: Dict[str, Any], 
                       *, max_depth: int = 3) -> CompressionResult:
    """
    Simplifica estrutura de um conceito.
    
    Papel: Remove detalhes desnecessários de representação.
    Justificativa: Conceitos devem ser compreensíveis e compactos.
    """
    def simplify_recursive(obj, depth=0):
        if depth >= max_depth:
            if isinstance(obj, (list, dict)):
                return f"<{type(obj).__name__}:{len(obj) if isinstance(obj, (list, dict)) else '?'}>"
            return obj
        
        if isinstance(obj, dict):
            return {k: simplify_recursive(v, depth + 1) for k, v in list(obj.items())[:10]}
        elif isinstance(obj, list):
            if len(obj) > 5:
                return [simplify_recursive(x, depth + 1) for x in obj[:5]] + [f"...+{len(obj)-5}"]
            return [simplify_recursive(x, depth + 1) for x in obj]
        return obj
    
    simplified = simplify_recursive(concept)
    
    original_str = str(concept)
    simplified_str = str(simplified)
    
    return CompressionResult(
        compressed=simplified,
        original_size=len(original_str),
        compressed_size=len(simplified_str),
        ratio=len(simplified_str) / max(1, len(original_str)),
        metadata={"max_depth": max_depth}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. PATTERN COMPRESSORS
# ─────────────────────────────────────────────────────────────────────────────


def pattern_compressor(g: GridV124, *, bg: int = 0) -> CompressionResult:
    """
    Comprime grid identificando padrões repetitivos.
    
    Papel: Substitui repetições por referências.
    Justificativa: Muitos grids têm estrutura repetitiva.
    """
    h, w = _grid_shape_v161(g)
    flat = _grid_to_flat_v161(g)
    
    # Run-length encoding
    rle = _run_length_encode_v161(flat)
    
    # Calculate sizes
    original_size = len(flat)
    rle_size = len(rle) * 2  # Each tuple is (value, count)
    
    # Try to find repeating tile patterns
    tile_sizes = [(2, 2), (3, 3), (2, 3), (3, 2)]
    best_tile = None
    best_tile_compression = 0
    
    for th, tw in tile_sizes:
        if h % th == 0 and w % tw == 0:
            # Extract first tile
            tile = [[int(g[r][c]) for c in range(tw)] for r in range(th)]
            
            # Check if tile repeats
            matches = True
            for tr in range(0, h, th):
                for tc in range(0, w, tw):
                    for r in range(th):
                        for c in range(tw):
                            if int(g[tr + r][tc + c]) != tile[r][c]:
                                matches = False
                                break
                        if not matches:
                            break
                    if not matches:
                        break
                if not matches:
                    break
            
            if matches:
                tile_compression = (h * w) - (th * tw + 4)  # tile + position info
                if tile_compression > best_tile_compression:
                    best_tile = {"tile": tile, "size": (th, tw)}
                    best_tile_compression = tile_compression
    
    # Choose best compression
    if best_tile and best_tile_compression > (original_size - rle_size):
        compressed = best_tile
        compressed_size = best_tile["size"][0] * best_tile["size"][1] + 4
        method = "tile"
    else:
        compressed = {"rle": rle}
        compressed_size = rle_size
        method = "rle"
    
    return CompressionResult(
        compressed=compressed,
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=compressed_size / max(1, original_size),
        metadata={"method": method, "grid_shape": (h, w)}
    )


def lossy_pattern_compressor(g: GridV124, *, tolerance: int = 1, bg: int = 0) -> CompressionResult:
    """
    Comprime permitindo pequenos erros de reconstrução.
    
    Papel: Troca precisão por maior compressão.
    Justificativa: Se diferença é tolerável, compressão agressiva é útil.
    """
    h, w = _grid_shape_v161(g)
    
    # Find dominant color
    counts = Counter(_grid_to_flat_v161(g))
    dominant = counts.most_common(1)[0][0]
    
    # Compress by noting only differences from dominant
    differences = []
    for r in range(h):
        for c in range(w):
            if int(g[r][c]) != dominant:
                differences.append((r, c, int(g[r][c])))
    
    # Apply tolerance: merge nearby differences
    if tolerance > 0 and differences:
        merged_diffs = []
        used = set()
        
        for i, (r1, c1, v1) in enumerate(differences):
            if i in used:
                continue
            
            cluster = [(r1, c1, v1)]
            for j, (r2, c2, v2) in enumerate(differences[i+1:], i+1):
                if j in used:
                    continue
                
                # Check if close enough to merge
                if abs(r1 - r2) <= tolerance and abs(c1 - c2) <= tolerance:
                    if v1 == v2:  # Same color
                        cluster.append((r2, c2, v2))
                        used.add(j)
            
            # Store cluster as bounding box
            rs = [r for r, c, v in cluster]
            cs = [c for r, c, v in cluster]
            merged_diffs.append({
                "bbox": (min(rs), min(cs), max(rs), max(cs)),
                "color": v1,
                "cells": len(cluster)
            })
            used.add(i)
        
        differences = merged_diffs
    
    original_size = h * w
    compressed_size = 1 + len(differences) * 4  # dominant + diffs
    
    return CompressionResult(
        compressed={"dominant": dominant, "differences": differences},
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=compressed_size / max(1, original_size),
        metadata={"tolerance": tolerance, "num_differences": len(differences)}
    )


def differential_compressor(g1: GridV124, g2: GridV124) -> CompressionResult:
    """
    Comprime diferenças entre dois grids.
    
    Papel: Armazena apenas delta entre estados.
    Justificativa: Se grids são similares, delta é pequeno.
    """
    h1, w1 = _grid_shape_v161(g1)
    h2, w2 = _grid_shape_v161(g2)
    
    # Handle different sizes
    max_h = max(h1, h2)
    max_w = max(w1, w2)
    
    differences = []
    for r in range(max_h):
        for c in range(max_w):
            v1 = int(g1[r][c]) if r < h1 and c < w1 else -1
            v2 = int(g2[r][c]) if r < h2 and c < w2 else -1
            
            if v1 != v2:
                differences.append((r, c, v1, v2))
    
    original_size = h1 * w1 + h2 * w2
    compressed_size = len(differences) * 4 + 4  # diffs + shape info
    
    return CompressionResult(
        compressed={
            "base_shape": (h1, w1),
            "target_shape": (h2, w2),
            "differences": differences
        },
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=compressed_size / max(1, original_size),
        metadata={"num_differences": len(differences)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 4. STATE ENCODERS
# ─────────────────────────────────────────────────────────────────────────────


def state_encoder(g: GridV124, *, bg: int = 0) -> CompressionResult:
    """
    Codifica estado do grid de forma compacta.
    
    Papel: Transforma grid em representação numericamente manipulável.
    Justificativa: Estados codificados permitem comparação eficiente.
    """
    h, w = _grid_shape_v161(g)
    
    # Build encoding dictionary
    unique_colors = sorted(set(_grid_to_flat_v161(g)))
    color_to_code = {c: i for i, c in enumerate(unique_colors)}
    
    # Encode grid
    encoded = []
    for r in range(h):
        for c in range(w):
            encoded.append(color_to_code[int(g[r][c])])
    
    # Compress encoded
    bits_per_symbol = max(1, math.ceil(math.log2(max(len(unique_colors), 2))))
    
    original_size = h * w * 4  # Assuming 4 bits per color
    compressed_size = len(encoded) * bits_per_symbol + len(unique_colors)  # data + dict
    
    return CompressionResult(
        compressed=EncodedState(
            code=encoded,
            dictionary={str(k): v for k, v in color_to_code.items()},
            reversible=True
        ),
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=compressed_size / max(1, original_size),
        metadata={"num_symbols": len(unique_colors), "bits_per_symbol": bits_per_symbol}
    )


def feature_vectorizer(g: GridV124, *, bg: int = 0) -> CompressionResult:
    """
    Converte grid em vetor de features fixo.
    
    Papel: Extrai características para comparação/ML.
    Justificativa: Vetores de features habilitam métodos estatísticos.
    """
    h, w = _grid_shape_v161(g)
    
    features = []
    
    # Basic features
    features.extend([h, w, h * w])  # Shape
    
    # Color features
    counts = Counter(_grid_to_flat_v161(g))
    for c in range(10):  # ARC uses colors 0-9
        features.append(counts.get(c, 0))
    
    # Object features
    objects = _find_objects_v161(g, bg=bg)
    features.append(len(objects))  # Object count
    
    if objects:
        sizes = [len(obj) for obj in objects]
        features.extend([min(sizes), max(sizes), sum(sizes) // len(sizes)])
    else:
        features.extend([0, 0, 0])
    
    # Symmetry features
    flat = _grid_to_flat_v161(g)
    h_flip = [int(g[r][w-1-c]) for r in range(h) for c in range(w)]
    v_flip = [int(g[h-1-r][c]) for r in range(h) for c in range(w)]
    
    h_sym = sum(1 for a, b in zip(flat, h_flip) if a == b) / max(1, len(flat))
    v_sym = sum(1 for a, b in zip(flat, v_flip) if a == b) / max(1, len(flat))
    
    features.extend([h_sym, v_sym])
    
    # Entropy feature
    features.append(_entropy_v161(flat))
    
    original_size = h * w
    compressed_size = len(features)
    
    return CompressionResult(
        compressed=features,
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=compressed_size / max(1, original_size),
        metadata={"feature_count": len(features)}
    )


def dimensionality_reducer(features: List[float], 
                           *, target_dims: int = 8) -> CompressionResult:
    """
    Reduz dimensão de vetor de features.
    
    Papel: Comprime representação mantendo informação essencial.
    Justificativa: Menos dimensões = processamento mais rápido.
    """
    if not features:
        return CompressionResult(
            compressed=[],
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    original_dims = len(features)
    
    if original_dims <= target_dims:
        return CompressionResult(
            compressed=features,
            original_size=original_dims,
            compressed_size=original_dims,
            ratio=1.0,
            metadata={"method": "identity"}
        )
    
    # Simple reduction: select most variant features + averages
    # In practice, would use PCA or similar
    
    reduced = []
    
    # Take every nth feature to get target_dims
    step = original_dims // target_dims
    for i in range(0, original_dims, step):
        if len(reduced) < target_dims:
            reduced.append(features[i])
    
    # Pad if needed
    while len(reduced) < target_dims:
        reduced.append(sum(features) / len(features))
    
    return CompressionResult(
        compressed=reduced[:target_dims],
        original_size=original_dims,
        compressed_size=target_dims,
        ratio=target_dims / original_dims,
        metadata={"method": "sampling", "step": step}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 5. REDUNDANCY PRUNERS
# ─────────────────────────────────────────────────────────────────────────────


def redundancy_pruner(concepts: List[Dict[str, Any]]) -> CompressionResult:
    """
    Remove conceitos redundantes de um conjunto.
    
    Papel: Elimina conceitos que não acrescentam informação nova.
    Justificativa: Redundância desperdiça memória e processamento.
    """
    if not concepts:
        return CompressionResult(
            compressed=[],
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    # Hash concepts for comparison
    seen = set()
    unique = []
    
    for c in concepts:
        # Create hashable representation
        key = str(sorted(c.items()) if isinstance(c, dict) else c)
        
        if key not in seen:
            seen.add(key)
            unique.append(c)
    
    return CompressionResult(
        compressed=unique,
        original_size=len(concepts),
        compressed_size=len(unique),
        ratio=len(unique) / max(1, len(concepts)),
        metadata={"removed": len(concepts) - len(unique)}
    )


def usage_pruner(concepts: List[Dict[str, Any]], 
                 usage_counts: Dict[str, int],
                 *, min_usage: int = 1) -> CompressionResult:
    """
    Remove conceitos raramente utilizados.
    
    Papel: Mantém apenas conceitos frequentemente acessados.
    Justificativa: Conceitos não utilizados ocupam espaço desnecessário.
    """
    if not concepts:
        return CompressionResult(
            compressed=[],
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    pruned = []
    for c in concepts:
        concept_id = c.get("id", str(c))
        usage = usage_counts.get(concept_id, 0)
        
        if usage >= min_usage:
            pruned.append(c)
    
    return CompressionResult(
        compressed=pruned,
        original_size=len(concepts),
        compressed_size=len(pruned),
        ratio=len(pruned) / max(1, len(concepts)),
        metadata={"min_usage": min_usage, "pruned_count": len(concepts) - len(pruned)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# 6. ARCHIVAL COMPRESSORS
# ─────────────────────────────────────────────────────────────────────────────


def concept_archive_compressor(archive: List[Dict[str, Any]], 
                                *, max_entries: int = 100) -> CompressionResult:
    """
    Comprime arquivo de conceitos antigos.
    
    Papel: Mantém histórico compacto de conceitos aprendidos.
    Justificativa: Armazenamento ilimitado é inviável.
    """
    if not archive:
        return CompressionResult(
            compressed=[],
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    if len(archive) <= max_entries:
        return CompressionResult(
            compressed=archive,
            original_size=len(archive),
            compressed_size=len(archive),
            ratio=1.0,
            metadata={"method": "identity"}
        )
    
    # Keep most recent and most important
    # Importance could be based on usage, but here we'll use recency
    compressed = archive[-max_entries:]
    
    return CompressionResult(
        compressed=compressed,
        original_size=len(archive),
        compressed_size=len(compressed),
        ratio=len(compressed) / len(archive),
        metadata={"method": "recency_based", "evicted": len(archive) - len(compressed)}
    )


def concept_chunker(concepts: List[Dict[str, Any]], 
                    *, chunk_size: int = 10) -> CompressionResult:
    """
    Agrupa conceitos em chunks gerenciáveis.
    
    Papel: Organiza conceitos para acesso eficiente.
    Justificativa: Chunks facilitam carregamento lazy.
    """
    if not concepts:
        return CompressionResult(
            compressed=[],
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    chunks = []
    for i in range(0, len(concepts), chunk_size):
        chunk = concepts[i:i + chunk_size]
        chunks.append({
            "chunk_id": i // chunk_size,
            "start_idx": i,
            "end_idx": min(i + chunk_size, len(concepts)),
            "concepts": chunk,
            "summary": f"{len(chunk)} concepts"
        })
    
    # Compressed size is chunk metadata overhead
    compressed_size = len(chunks) * 4  # Chunk headers
    
    return CompressionResult(
        compressed=chunks,
        original_size=len(concepts),
        compressed_size=len(chunks),
        ratio=len(chunks) / max(1, len(concepts)),
        metadata={"num_chunks": len(chunks), "chunk_size": chunk_size}
    )


def macro_compressor(operations: List[Dict[str, Any]], 
                     *, min_sequence_length: int = 2) -> CompressionResult:
    """
    Comprime sequências repetidas de operações em macros.
    
    Papel: Detecta e substitui padrões operacionais por macros.
    Justificativa: Sequências comuns devem ser nomeadas e reutilizadas.
    """
    if not operations:
        return CompressionResult(
            compressed=[],
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    # Find repeating sequences
    sequences: Dict[str, int] = {}
    
    for length in range(min_sequence_length, min(len(operations) // 2 + 1, 6)):
        for i in range(len(operations) - length + 1):
            seq = tuple(op.get("name", str(op)) for op in operations[i:i + length])
            seq_key = str(seq)
            sequences[seq_key] = sequences.get(seq_key, 0) + 1
    
    # Keep only repeated sequences
    repeated = {k: v for k, v in sequences.items() if v >= 2}
    
    # Create macros
    macros = []
    for i, (seq_key, count) in enumerate(sorted(repeated.items(), 
                                                  key=lambda x: -x[1])[:10]):
        macros.append({
            "macro_id": f"MACRO_{i}",
            "sequence": eval(seq_key),
            "occurrences": count,
            "savings": count * (len(eval(seq_key)) - 1)
        })
    
    # Calculate compression
    total_savings = sum(m["savings"] for m in macros)
    
    return CompressionResult(
        compressed={"operations": operations, "macros": macros},
        original_size=len(operations),
        compressed_size=len(operations) - total_savings,
        ratio=(len(operations) - total_savings) / max(1, len(operations)),
        metadata={"num_macros": len(macros), "total_savings": total_savings}
    )


def memory_compactor(memory: Dict[str, Any], 
                     *, priority_keys: List[str] = None) -> CompressionResult:
    """
    Compacta estrutura de memória removendo entradas de baixa prioridade.
    
    Papel: Mantém memória dentro de limites aceitáveis.
    Justificativa: Memória infinita não é prática.
    """
    if not memory:
        return CompressionResult(
            compressed={},
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    priority_keys = priority_keys or []
    
    # Calculate sizes
    original_size = len(str(memory))
    
    # Keep priority keys
    compacted = {}
    for key in priority_keys:
        if key in memory:
            compacted[key] = memory[key]
    
    # Add remaining keys up to budget
    remaining_budget = original_size // 2
    for key, value in memory.items():
        if key not in compacted:
            value_size = len(str(value))
            if value_size <= remaining_budget:
                compacted[key] = value
                remaining_budget -= value_size
    
    compressed_size = len(str(compacted))
    
    return CompressionResult(
        compressed=compacted,
        original_size=original_size,
        compressed_size=compressed_size,
        ratio=compressed_size / max(1, original_size),
        metadata={
            "original_keys": len(memory),
            "compacted_keys": len(compacted),
            "priority_keys_kept": len([k for k in priority_keys if k in compacted])
        }
    )


def search_space_cache(cache: Dict[str, Any], 
                       *, max_entries: int = 1000) -> CompressionResult:
    """
    Comprime cache de espaço de busca.
    
    Papel: Mantém cache eficiente para acelerações.
    Justificativa: Cache descontrolado esgota memória.
    """
    if not cache:
        return CompressionResult(
            compressed={},
            original_size=0,
            compressed_size=0,
            ratio=1.0
        )
    
    original_size = len(cache)
    
    if original_size <= max_entries:
        return CompressionResult(
            compressed=cache,
            original_size=original_size,
            compressed_size=original_size,
            ratio=1.0,
            metadata={"method": "identity"}
        )
    
    # LRU-like: keep most recent entries
    # (In practice, would track access times)
    keys = list(cache.keys())[-max_entries:]
    compressed = {k: cache[k] for k in keys}
    
    return CompressionResult(
        compressed=compressed,
        original_size=original_size,
        compressed_size=len(compressed),
        ratio=len(compressed) / original_size,
        metadata={"evicted": original_size - len(compressed)}
    )


# ─────────────────────────────────────────────────────────────────────────────
# COMPRESSION REGISTRY
# ─────────────────────────────────────────────────────────────────────────────


COMPRESSION_TOOLS_V161 = {
    # MDL evaluators
    "mdl_evaluator": mdl_evaluator,
    "description_length_guard": description_length_guard,
    "entropy_minimizer": entropy_minimizer,
    
    # Concept mergers/simplifiers
    "concept_merger": concept_merger,
    "concept_simplifier": concept_simplifier,
    
    # Pattern compressors
    "pattern_compressor": pattern_compressor,
    "lossy_pattern_compressor": lossy_pattern_compressor,
    "differential_compressor": differential_compressor,
    
    # State encoders
    "state_encoder": state_encoder,
    "feature_vectorizer": feature_vectorizer,
    "dimensionality_reducer": dimensionality_reducer,
    
    # Redundancy pruners
    "redundancy_pruner": redundancy_pruner,
    "usage_pruner": usage_pruner,
    
    # Archival compressors
    "concept_archive_compressor": concept_archive_compressor,
    "concept_chunker": concept_chunker,
    "macro_compressor": macro_compressor,
    "memory_compactor": memory_compactor,
    "search_space_cache": search_space_cache,
}


def get_compression_tool_v161(name: str):
    """Get a compression tool by name."""
    return COMPRESSION_TOOLS_V161.get(name)


def list_compression_tools_v161() -> List[str]:
    """List all available compression tools."""
    return list(COMPRESSION_TOOLS_V161.keys())


# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────

"""
Total de Ferramentas de Compressão Implementadas: 18

CATEGORIAS:
1. Avaliadores MDL (3):
   - mdl_evaluator, description_length_guard, entropy_minimizer

2. Fusores/Simplificadores de Conceito (2):
   - concept_merger, concept_simplifier

3. Compressores de Padrão (3):
   - pattern_compressor, lossy_pattern_compressor, differential_compressor

4. Codificadores de Estado (3):
   - state_encoder, feature_vectorizer, dimensionality_reducer

5. Podadores de Redundância (2):
   - redundancy_pruner, usage_pruner

6. Compressores de Arquivo (5):
   - concept_archive_compressor, concept_chunker
   - macro_compressor, memory_compactor, search_space_cache
"""
