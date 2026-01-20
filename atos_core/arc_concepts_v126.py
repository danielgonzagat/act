from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .grid_v124 import GridV124, grid_shape_v124, unique_colors_v124

ARC_CONCEPTS_SCHEMA_VERSION_V126 = 126


def _mode_color_v126(g: GridV124) -> int:
    # Deterministic mode with tie-break by smallest color.
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            counts[int(x)] = int(counts.get(int(x), 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


@dataclass(frozen=True)
class ConceptV126:
    concept_kind: str
    params: Tuple[Tuple[str, Any], ...]
    invariants: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"concept_kind": str(self.concept_kind)}
        p: Dict[str, Any] = {}
        for k, v in sorted(self.params, key=lambda kv: str(kv[0])):
            p[str(k)] = v
        d["params"] = p
        d["invariants"] = [str(s) for s in self.invariants]
        return d

    def concept_sig(self) -> str:
        body = {"schema_version": int(ARC_CONCEPTS_SCHEMA_VERSION_V126), "kind": "arc_concept_v126", **self.to_dict()}
        return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class BackgroundConceptV126:
    bg_color: int

    @classmethod
    def propose(cls, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
        # Background candidates from input modes + default 0.
        bgs: List[int] = [0]
        for inp, _ in train_pairs:
            bgs.append(_mode_color_v126(inp))
        bgs.append(_mode_color_v126(test_in))
        bgs_s = sorted(set(int(x) for x in bgs))
        out: List[ConceptV126] = []
        for bg in bgs_s:
            out.append(
                ConceptV126(
                    concept_kind="Background",
                    params=(("bg_color", int(bg)),),
                    invariants=("bg_color in 0..9",),
                )
            )
        return out


@dataclass(frozen=True)
class PaletteConceptV126:
    @classmethod
    def propose(cls, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
        in_colors: set[int] = set()
        out_colors: set[int] = set()
        for inp, out in train_pairs:
            in_colors |= set(int(c) for c in unique_colors_v124(inp))
            out_colors |= set(int(c) for c in unique_colors_v124(out))
        in_colors |= set(int(c) for c in unique_colors_v124(test_in))
        return [
            ConceptV126(
                concept_kind="Palette",
                params=(
                    ("in_colors", [int(c) for c in sorted(in_colors)]),
                    ("out_colors", [int(c) for c in sorted(out_colors)]),
                ),
                invariants=("colors in 0..9",),
            )
        ]


BboxSourceKindV126 = Literal["bbox_nonzero", "bbox_by_color", "bbox_of_selected_object"]


@dataclass(frozen=True)
class BboxSourceConceptV126:
    @classmethod
    def propose(cls, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
        # Propose bbox sources based on input palettes (excluding background candidates handled elsewhere).
        in_colors: set[int] = set()
        for inp, _ in train_pairs:
            in_colors |= set(int(c) for c in unique_colors_v124(inp))
        in_colors |= set(int(c) for c in unique_colors_v124(test_in))
        colors = sorted(in_colors)

        out: List[ConceptV126] = []
        out.append(
            ConceptV126(
                concept_kind="BboxSource",
                params=(("kind", "bbox_nonzero"), ("bg", 0)),
                invariants=("bbox within grid bounds",),
            )
        )
        for c in colors:
            if int(c) == 0:
                continue
            out.append(
                ConceptV126(
                    concept_kind="BboxSource",
                    params=(("kind", "bbox_by_color"), ("color", int(c))),
                    invariants=("color in 0..9", "bbox within grid bounds",),
                )
            )
        for sel in ["largest_area", "smallest_area", "leftmost", "rightmost", "topmost", "bottommost"]:
            for c in colors:
                if int(c) == 0:
                    continue
                out.append(
                    ConceptV126(
                        concept_kind="BboxSource",
                        params=(("kind", "bbox_of_selected_object"), ("selector", str(sel)), ("color", int(c))),
                        invariants=("selector in enum", "color in 0..9", "bbox within grid bounds",),
                    )
                )
        # Deterministic ordering by signature.
        out.sort(key=lambda c: str(c.concept_sig()))
        return out


@dataclass(frozen=True)
class SymmetryConceptV126:
    @classmethod
    def propose(cls, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
        # Placeholder: symmetry detection lives in DSL ops; here we only declare concept kind.
        _ = (train_pairs, test_in)
        return []


@dataclass(frozen=True)
class RelationConceptV126:
    @classmethod
    def propose(cls, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
        _ = (train_pairs, test_in)
        return []


@dataclass(frozen=True)
class ContainerInsideConceptV126:
    @classmethod
    def propose(cls, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
        _ = (train_pairs, test_in)
        return []


@dataclass(frozen=True)
class BorderFillConceptV126:
    @classmethod
    def propose(cls, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
        _ = (train_pairs, test_in)
        return []


def propose_concepts_v126(*, train_pairs: Sequence[Tuple[GridV124, GridV124]], test_in: GridV124) -> List[ConceptV126]:
    """
    Deterministic concept proposal set. This is NOT a solver.
    Only depends on the current task's train pairs and test input.
    """
    out: List[ConceptV126] = []
    out.extend(BackgroundConceptV126.propose(train_pairs=train_pairs, test_in=test_in))
    out.extend(PaletteConceptV126.propose(train_pairs=train_pairs, test_in=test_in))
    out.extend(BboxSourceConceptV126.propose(train_pairs=train_pairs, test_in=test_in))
    out.extend(SymmetryConceptV126.propose(train_pairs=train_pairs, test_in=test_in))
    out.extend(RelationConceptV126.propose(train_pairs=train_pairs, test_in=test_in))
    out.extend(ContainerInsideConceptV126.propose(train_pairs=train_pairs, test_in=test_in))
    out.extend(BorderFillConceptV126.propose(train_pairs=train_pairs, test_in=test_in))
    # stable dedup by concept_sig
    seen: set[str] = set()
    uniq: List[ConceptV126] = []
    for c in out:
        sig = str(c.concept_sig())
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(c)
    uniq.sort(key=lambda c: str(c.concept_sig()))
    return uniq


def concept_trace_v126(concepts: Sequence[ConceptV126]) -> Dict[str, Any]:
    return {
        "schema_version": int(ARC_CONCEPTS_SCHEMA_VERSION_V126),
        "kind": "arc_concept_trace_v126",
        "concepts": [{"sig": str(c.concept_sig()), **c.to_dict()} for c in concepts],
    }

