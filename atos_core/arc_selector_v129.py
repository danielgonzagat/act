from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_delta_v129 import compute_delta_v129
from .arc_dsl_v126 import ObjectSetV126, ObjectV126, connected_components_v126
from .arc_dsl_v129 import SelectorV129, select_object_v129
from .grid_v124 import GridV124, grid_shape_v124, unique_colors_v124

ARC_SELECTOR_SCHEMA_VERSION_V129 = 129


def _cells_intersection_count(mask: GridV124, cells: Sequence[Tuple[int, int]]) -> int:
    if not mask:
        return 0
    h, w = grid_shape_v124(mask)
    n = 0
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        if 0 <= rr < h and 0 <= cc < w and int(mask[rr][cc]) != 0:
            n += 1
    return int(n)


@dataclass(frozen=True)
class SelectorHypothesisV129:
    color_filter: Optional[int]
    selector: SelectorV129
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_SELECTOR_SCHEMA_VERSION_V129),
            "kind": "selector_hypothesis_v129",
            "color_filter": int(self.color_filter) if self.color_filter is not None else None,
            "selector": str(self.selector),
            "evidence": ev,
        }

    def hypothesis_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _connected_components_all_colors_v129(g: GridV124) -> ObjectSetV126:
    objs: List[ObjectV126] = []
    for c in unique_colors_v124(g):
        os = connected_components_v126(g, color=int(c))
        objs.extend(list(os.objects))
    objs.sort(key=lambda o: (o.bbox.to_tuple(), int(o.color), o.cells))
    return ObjectSetV126(objects=tuple(objs))


def _dominant_color_from_counts(counts: Sequence[Tuple[int, int]]) -> Optional[int]:
    if not counts:
        return None
    items = sorted(((int(c), int(n)) for c, n in counts), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


def _target_objects_for_pair_v129(inp: GridV124, out: GridV124) -> Tuple[ObjectSetV126, List[str]]:
    delta = compute_delta_v129(inp, out)
    obj_set = _connected_components_all_colors_v129(inp)
    if not obj_set.objects or not delta.changed_mask:
        return obj_set, []
    inter_counts: List[Tuple[int, int]] = []
    for idx, obj in enumerate(obj_set.objects):
        inter_counts.append((int(_cells_intersection_count(delta.changed_mask, obj.cells)), int(idx)))
    inter_counts.sort(key=lambda t: (-int(t[0]), int(t[1])))
    max_inter = int(inter_counts[0][0]) if inter_counts else 0
    if max_inter <= 0:
        return obj_set, []
    target_sigs: List[str] = []
    for inter, idx in inter_counts:
        if int(inter) != int(max_inter):
            break
        target_sigs.append(str(obj_set.objects[int(idx)].object_sig()))
    target_sigs = sorted(set(str(s) for s in target_sigs))
    return obj_set, target_sigs


def infer_selector_hypotheses_v129(*, train_pairs: Sequence[Tuple[GridV124, GridV124]]) -> List[SelectorHypothesisV129]:
    per_pair: List[Dict[str, Any]] = []
    target_color_intersection: Optional[set[int]] = None

    dominant_in_color: Optional[int] = None
    dominant_out_color: Optional[int] = None

    for i, (inp, out) in enumerate(train_pairs):
        delta = compute_delta_v129(inp, out)
        obj_set, target_sigs = _target_objects_for_pair_v129(inp, out)
        target_colors = set()
        for obj in obj_set.objects:
            if str(obj.object_sig()) in set(target_sigs):
                target_colors.add(int(obj.color))

        if target_color_intersection is None:
            target_color_intersection = set(int(c) for c in target_colors)
        else:
            target_color_intersection &= set(int(c) for c in target_colors)

        di = _dominant_color_from_counts(delta.in_color_counts_in_changed)
        do = _dominant_color_from_counts(delta.out_color_counts_in_changed)
        if i == 0:
            dominant_in_color = di
            dominant_out_color = do
        else:
            if dominant_in_color is not None and di != dominant_in_color:
                dominant_in_color = None
            if dominant_out_color is not None and do != dominant_out_color:
                dominant_out_color = None

        per_pair.append({"obj_set": obj_set, "target_sigs": set(target_sigs), "pair_index": int(i)})

    if target_color_intersection is None:
        target_color_intersection = set()

    color_filters: List[Optional[int]] = [None]
    for c in sorted(target_color_intersection):
        color_filters.append(int(c))
    if dominant_in_color is not None:
        color_filters.append(int(dominant_in_color))
    if dominant_out_color is not None:
        color_filters.append(int(dominant_out_color))
    ints = sorted({int(c) for c in color_filters if c is not None})
    color_filters = [None] + [int(c) for c in ints]

    selector_kinds: List[SelectorV129] = [
        "largest_area",
        "smallest_area",
        "leftmost",
        "rightmost",
        "topmost",
        "bottommost",
        "closest_to_center",
        "closest_to_tl",
        "closest_to_tr",
        "closest_to_bl",
        "closest_to_br",
    ]

    out_hyps: List[SelectorHypothesisV129] = []
    for cf in color_filters:
        for sel in selector_kinds:
            ok = True
            chosen: List[Dict[str, Any]] = []
            for row in per_pair:
                obj_set_all: ObjectSetV126 = row["obj_set"]
                inp_idx = int(row["pair_index"])
                tgt = row["target_sigs"]
                if cf is not None:
                    objs_f = tuple(o for o in obj_set_all.objects if int(o.color) == int(cf))
                    obj_set = ObjectSetV126(objects=objs_f)
                else:
                    obj_set = obj_set_all
                if not obj_set.objects:
                    ok = False
                    break
                chosen_obj = select_object_v129(obj_set, selector=sel)
                chosen_sig = str(chosen_obj.object_sig())
                if chosen_sig not in tgt:
                    ok = False
                    break
                chosen.append({"pair_index": int(inp_idx), "chosen_object_sig": str(chosen_sig)})
            if not ok:
                continue
            evidence_items: List[Tuple[str, Any]] = []
            evidence_items.append(("pairs", list(sorted(chosen, key=lambda d: int(d["pair_index"])))))
            out_hyps.append(
                SelectorHypothesisV129(
                    color_filter=int(cf) if cf is not None else None,
                    selector=sel,
                    evidence=tuple(evidence_items),
                )
            )

    seen: set[str] = set()
    uniq: List[SelectorHypothesisV129] = []
    for h in out_hyps:
        s = str(h.hypothesis_sig())
        if s in seen:
            continue
        seen.add(s)
        uniq.append(h)
    uniq.sort(key=lambda h: str(h.hypothesis_sig()))
    return uniq

