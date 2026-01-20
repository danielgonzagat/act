from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_delta_v129 import DeltaEvidenceV129, compute_delta_v129
from .arc_dsl_v126 import BboxV126
from .arc_dsl_v129 import OP_DEFS_V129, apply_op_v129
from .arc_selector_v129 import SelectorHypothesisV129
from .grid_v124 import (
    GridV124,
    grid_equal_v124,
    grid_shape_v124,
    reflect_h_v124,
    reflect_v_v124,
    rotate180_v124,
    rotate270_v124,
    rotate90_v124,
    translate_v124,
    unique_colors_v124,
)

ARC_INVERSE_SCHEMA_VERSION_V129 = 129


def _mode_color_v129(g: GridV124) -> int:
    counts: Dict[int, int] = {}
    for row in g:
        for x in row:
            counts[int(x)] = int(counts.get(int(x), 0)) + 1
    if not counts:
        return 0
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0])


def _step_cost_bits_v129(step: Dict[str, Any]) -> int:
    op = str(step.get("op_id") or "")
    if op in OP_DEFS_V129:
        base = int(OP_DEFS_V129[op].cost_bits)
    elif op == "map_colors":
        base = 16
    elif op == "translate":
        base = 12
    else:
        base = 24
    extra = 0
    args = step.get("args") if isinstance(step.get("args"), dict) else {}
    for k, v in args.items():
        if k == "mapping" and isinstance(v, dict):
            extra += 8 * int(len(v))
        else:
            extra += 4
    in_vars = step.get("in_vars") if isinstance(step.get("in_vars"), list) else []
    extra += 2 * max(0, int(len(in_vars)) - 1)
    return int(base + extra)


def _program_cost_bits_v129(steps: Sequence[Dict[str, Any]]) -> int:
    return int(sum(_step_cost_bits_v129(s) for s in steps))


def _program_sig_v129(steps: Sequence[Dict[str, Any]]) -> str:
    body = {"schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V129), "kind": "arc_program_steps_v129", "steps": list(steps)}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class BboxExprV129:
    steps: Tuple[Dict[str, Any], ...]
    bbox_var: str
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V129),
            "kind": "bbox_expr_v129",
            "steps": list(self.steps),
            "bbox_var": str(self.bbox_var),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def expr_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class MaskExprV129:
    steps: Tuple[Dict[str, Any], ...]
    mask_var: str
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V129),
            "kind": "mask_expr_v129",
            "steps": list(self.steps),
            "mask_var": str(self.mask_var),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def expr_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class InverseCandidateV129:
    op_name: str
    steps: Tuple[Dict[str, Any], ...]
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V129),
            "kind": "inverse_candidate_v129",
            "op_name": str(self.op_name),
            "steps": list(self.steps),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def candidate_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _apply_steps_v129(*, steps: Sequence[Dict[str, Any]], g_cur: GridV124) -> Any:
    env: Dict[str, Any] = {"gC": g_cur}
    for st in steps:
        op = str(st.get("op_id") or "")
        in_vars = st.get("in_vars") if isinstance(st.get("in_vars"), list) else []
        out_var = str(st.get("out_var") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        ins = [env[str(v)] for v in in_vars]
        if op in OP_DEFS_V129:
            env[out_var] = apply_op_v129(op_id=op, inputs=ins, args=dict(args))
            continue
        if op == "map_colors":
            g = ins[0]
            m_raw = args.get("mapping", {})
            if not isinstance(m_raw, dict):
                raise ValueError("mapping_not_dict")
            m: Dict[int, int] = {int(k): int(v) for k, v in m_raw.items()}
            env[out_var] = tuple(tuple(int(m.get(int(x), int(x))) for x in row) for row in g)
            continue
        if op == "translate":
            g = ins[0]
            env[out_var] = translate_v124(g, dx=int(args["dx"]), dy=int(args["dy"]), pad=int(args.get("pad", 0)))
            continue
        raise ValueError(f"unknown_op:{op}")
    return env[str(steps[-1]["out_var"])] if steps else g_cur


def _apply_steps_bbox_v129(steps: Sequence[Dict[str, Any]], g_cur: GridV124) -> BboxV126:
    out = _apply_steps_v129(steps=steps, g_cur=g_cur)
    if not isinstance(out, BboxV126):
        raise ValueError("bbox_expr_not_bbox")
    return out


def propose_bbox_exprs_v129(
    *,
    bg_candidates: Sequence[int],
    color_candidates: Sequence[int],
    selector_hypotheses: Sequence[SelectorHypothesisV129],
    max_expand_delta: int = 2,
) -> List[BboxExprV129]:
    base: List[BboxExprV129] = []
    for bg in sorted(set(int(x) for x in bg_candidates)):
        st = {"op_id": "bbox_nonzero", "in_vars": ["gC"], "out_var": "bb1", "args": {"bg": int(bg)}}
        cost = _program_cost_bits_v129([st])
        base.append(BboxExprV129(steps=(st,), bbox_var="bb1", cost_bits=int(cost), evidence=(("kind", "bbox_nonzero"), ("bg", int(bg))),))
    for c in sorted(set(int(x) for x in color_candidates)):
        st = {"op_id": "bbox_by_color", "in_vars": ["gC"], "out_var": "bb1", "args": {"color": int(c)}}
        cost = _program_cost_bits_v129([st])
        base.append(BboxExprV129(steps=(st,), bbox_var="bb1", cost_bits=int(cost), evidence=(("kind", "bbox_by_color"), ("color", int(c))),))

    for h in selector_hypotheses:
        cf = h.color_filter
        sel = str(h.selector)
        args_cc: Dict[str, Any] = {}
        if cf is not None:
            args_cc["color"] = int(cf)
        st1 = {"op_id": "connected_components", "in_vars": ["gC"], "out_var": "os1", "args": args_cc}
        st2 = {"op_id": "select_object", "in_vars": ["os1"], "out_var": "o2", "args": {"selector": str(sel)}}
        st3 = {"op_id": "bbox_of_object", "in_vars": ["o2"], "out_var": "bb3", "args": {}}
        steps = [st1, st2, st3]
        cost = _program_cost_bits_v129(steps)
        ev: List[Tuple[str, Any]] = [("kind", "bbox_of_selected_object"), ("selector", str(sel))]
        if cf is not None:
            ev.append(("color_filter", int(cf)))
        base.append(BboxExprV129(steps=tuple(steps), bbox_var="bb3", cost_bits=int(cost), evidence=tuple(ev)))

    expanded: List[BboxExprV129] = []
    for expr in base:
        expanded.append(expr)
        for d in range(1, int(max_expand_delta) + 1):
            st = {"op_id": "bbox_expand", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "bbE", "args": {"delta": int(d)}}
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v129(steps)
            ev = list(expr.evidence) + [("expand_delta", int(d))]
            expanded.append(BboxExprV129(steps=tuple(steps), bbox_var="bbE", cost_bits=int(cost), evidence=tuple(ev)))

    seen: set[str] = set()
    uniq: List[BboxExprV129] = []
    for e in expanded:
        sig = _program_sig_v129(e.steps)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(e)
    uniq.sort(key=lambda e: (int(e.cost_bits), str(e.expr_sig())))
    return uniq


def propose_mask_exprs_v129(
    *,
    color_candidates: Sequence[int],
    bbox_exprs: Sequence[BboxExprV129],
    selector_hypotheses: Sequence[SelectorHypothesisV129],
) -> List[MaskExprV129]:
    base: List[MaskExprV129] = []
    for c in sorted(set(int(x) for x in color_candidates)):
        st = {"op_id": "mask_by_color", "in_vars": ["gC"], "out_var": "m1", "args": {"color": int(c)}}
        cost = _program_cost_bits_v129([st])
        base.append(MaskExprV129(steps=(st,), mask_var="m1", cost_bits=int(cost), evidence=(("kind", "mask_by_color"), ("color", int(c))),))

    for be in bbox_exprs:
        st = {"op_id": "mask_rect", "in_vars": ["gC", str(be.bbox_var)], "out_var": "mR", "args": {}}
        steps = list(be.steps) + [st]
        cost = _program_cost_bits_v129(steps)
        base.append(MaskExprV129(steps=tuple(steps), mask_var="mR", cost_bits=int(cost), evidence=(("kind", "mask_rect"), ("bbox_expr_sig", str(be.expr_sig()))),))

    for h in selector_hypotheses:
        cf = h.color_filter
        sel = str(h.selector)
        args_cc: Dict[str, Any] = {}
        if cf is not None:
            args_cc["color"] = int(cf)
        st1 = {"op_id": "connected_components", "in_vars": ["gC"], "out_var": "os1", "args": args_cc}
        st2 = {"op_id": "select_object", "in_vars": ["os1"], "out_var": "o2", "args": {"selector": str(sel)}}
        st3 = {"op_id": "mask_of_object", "in_vars": ["gC", "o2"], "out_var": "m3", "args": {}}
        steps = [st1, st2, st3]
        cost = _program_cost_bits_v129(steps)
        ev: List[Tuple[str, Any]] = [("kind", "mask_of_selected_object"), ("selector", str(sel))]
        if cf is not None:
            ev.append(("color_filter", int(cf)))
        base.append(MaskExprV129(steps=tuple(steps), mask_var="m3", cost_bits=int(cost), evidence=tuple(ev)))

    seen: set[str] = set()
    uniq: List[MaskExprV129] = []
    for e in base:
        sig = _program_sig_v129(e.steps)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(e)
    uniq.sort(key=lambda e: (int(e.cost_bits), str(e.expr_sig())))
    return uniq


def inverse_propose_map_colors_v129(*, inp: GridV124, out: GridV124) -> List[InverseCandidateV129]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo) or hi == 0 or wi == 0:
        return []
    mapping: Dict[int, int] = {}
    for r in range(hi):
        for c in range(wi):
            a = int(inp[r][c])
            b = int(out[r][c])
            if a in mapping and mapping[a] != b:
                return []
            mapping[a] = b
    m = {str(k): int(mapping[k]) for k in sorted(mapping.keys())}
    st = {"op_id": "map_colors", "in_vars": ["gC"], "out_var": "g1", "args": {"mapping": m}}
    cost = _program_cost_bits_v129([st])
    return [InverseCandidateV129(op_name="map_colors", steps=(st,), cost_bits=int(cost), evidence=(("mapping_size", int(len(m))),))]


def inverse_propose_translate_v129(*, inp: GridV124, out: GridV124, bg_candidates: Sequence[int]) -> List[InverseCandidateV129]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo) or hi == 0 or wi == 0:
        return []
    out_cands: List[InverseCandidateV129] = []
    seen: set[Tuple[int, int, int]] = set()
    from .grid_v124 import bbox_nonzero_v124

    for bg in sorted(set(int(x) for x in bg_candidates)):
        bb_in = bbox_nonzero_v124(inp, bg=int(bg))
        bb_out = bbox_nonzero_v124(out, bg=int(bg))
        if bb_in == (0, 0, 0, 0) or bb_out == (0, 0, 0, 0):
            continue
        dx = int(bb_out[1] - bb_in[1])
        dy = int(bb_out[0] - bb_in[0])
        key = (int(dx), int(dy), int(bg))
        if key in seen:
            continue
        seen.add(key)
        got = translate_v124(inp, dx=int(dx), dy=int(dy), pad=int(bg))
        if not grid_equal_v124(got, out):
            continue
        st = {"op_id": "translate", "in_vars": ["gC"], "out_var": "g1", "args": {"dx": int(dx), "dy": int(dy), "pad": int(bg)}}
        cost = _program_cost_bits_v129([st])
        out_cands.append(InverseCandidateV129(op_name="translate", steps=(st,), cost_bits=int(cost), evidence=(("dx", int(dx)), ("dy", int(dy)), ("pad", int(bg))),))
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_symmetry_v129(*, inp: GridV124, out: GridV124) -> List[InverseCandidateV129]:
    out_cands: List[InverseCandidateV129] = []
    checks: List[Tuple[str, GridV124]] = [
        ("reflect_h", reflect_h_v124(inp)),
        ("reflect_v", reflect_v_v124(inp)),
        ("rotate90", rotate90_v124(inp)),
        ("rotate180", rotate180_v124(inp)),
        ("rotate270", rotate270_v124(inp)),
    ]
    from .arc_dsl_v129 import transpose_v129

    checks.append(("transpose", transpose_v129(inp)))
    for op_id, got in checks:
        if not grid_equal_v124(got, out):
            continue
        st = {"op_id": str(op_id), "in_vars": ["gC"], "out_var": "g1", "args": {}}
        cost = _program_cost_bits_v129([st])
        out_cands.append(InverseCandidateV129(op_name=str(op_id), steps=(st,), cost_bits=int(cost), evidence=(("exact", True),)))
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_scale_up_v129(*, inp: GridV124, out: GridV124) -> List[InverseCandidateV129]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if hi == 0 or wi == 0 or ho == 0 or wo == 0:
        return []
    if ho % hi != 0 or wo % wi != 0:
        return []
    k_h = int(ho // hi)
    k_w = int(wo // wi)
    if k_h != k_w:
        return []
    k = int(k_h)
    if k <= 1:
        return []
    # Verify replication.
    for r in range(hi):
        for c in range(wi):
            v = int(inp[r][c])
            for rr in range(r * k, (r + 1) * k):
                for cc in range(c * k, (c + 1) * k):
                    if int(out[rr][cc]) != v:
                        return []
    st = {"op_id": "scale_up", "in_vars": ["gC"], "out_var": "g1", "args": {"k": int(k)}}
    cost = _program_cost_bits_v129([st])
    return [InverseCandidateV129(op_name="scale_up", steps=(st,), cost_bits=int(cost), evidence=(("k", int(k)),))]


def inverse_propose_tile_v129(*, inp: GridV124, out: GridV124) -> List[InverseCandidateV129]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if hi == 0 or wi == 0 or ho == 0 or wo == 0:
        return []
    if ho % hi != 0 or wo % wi != 0:
        return []
    reps_h = int(ho // hi)
    reps_w = int(wo // wi)
    if reps_h <= 1 and reps_w <= 1:
        return []
    # Verify tiling.
    for r in range(ho):
        for c in range(wo):
            if int(out[r][c]) != int(inp[r % hi][c % wi]):
                return []
    st = {"op_id": "tile", "in_vars": ["gC"], "out_var": "g1", "args": {"reps_h": int(reps_h), "reps_w": int(reps_w)}}
    cost = _program_cost_bits_v129([st])
    return [InverseCandidateV129(op_name="tile", steps=(st,), cost_bits=int(cost), evidence=(("reps_h", int(reps_h)), ("reps_w", int(reps_w))),)]


def inverse_propose_crop_bbox_v129(
    *, inp: GridV124, out: GridV124, delta: DeltaEvidenceV129, bbox_exprs: Sequence[BboxExprV129]
) -> List[InverseCandidateV129]:
    if not delta.shape_changed:
        return []
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if hi == 0 or wi == 0 or ho == 0 or wo == 0:
        return []
    if ho > hi or wo > wi:
        return []
    out_cands: List[InverseCandidateV129] = []
    for expr in bbox_exprs:
        try:
            bb = _apply_steps_bbox_v129(expr.steps, inp)
        except Exception:
            continue
        try:
            patch = _apply_steps_v129(steps=list(expr.steps) + [{"op_id": "crop_bbox", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "g1", "args": {}}], g_cur=inp)
        except Exception:
            continue
        if not isinstance(patch, tuple):
            continue
        if not grid_equal_v124(patch, out):
            continue
        st = {"op_id": "crop_bbox", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "g1", "args": {}}
        steps = list(expr.steps) + [st]
        cost = _program_cost_bits_v129(steps)
        out_cands.append(
            InverseCandidateV129(
                op_name="crop_bbox",
                steps=tuple(steps),
                cost_bits=int(cost),
                evidence=(
                    ("bbox_expr_sig", str(expr.expr_sig())),
                    ("in_shape", [int(hi), int(wi)]),
                    ("out_shape", [int(ho), int(wo)]),
                    ("bbox", list(bb.to_tuple())),
                ),
            )
        )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_fill_rect_v129(
    *, inp: GridV124, out: GridV124, delta: DeltaEvidenceV129, bbox_exprs: Sequence[BboxExprV129]
) -> List[InverseCandidateV129]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    cand_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    cand_colors = sorted(set(int(c) for c in cand_colors))
    target_bbox_t = tuple(int(x) for x in delta.changed_bbox)
    out_cands: List[InverseCandidateV129] = []
    for fill_color in cand_colors:
        for expr in bbox_exprs:
            try:
                bb = _apply_steps_bbox_v129(expr.steps, inp)
            except Exception:
                continue
            if bb.to_tuple() != target_bbox_t:
                continue
            st = {"op_id": "fill_rect", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "gF", "args": {"color": int(fill_color)}}
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v129(steps)
            out_cands.append(
                InverseCandidateV129(op_name="fill_rect", steps=tuple(steps), cost_bits=int(cost), evidence=(("fill_color", int(fill_color)), ("bbox_expr_sig", str(expr.expr_sig()))),)
            )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_draw_rect_border_v129(
    *, inp: GridV124, out: GridV124, delta: DeltaEvidenceV129, bbox_exprs: Sequence[BboxExprV129]
) -> List[InverseCandidateV129]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    cand_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    cand_colors = sorted(set(int(c) for c in cand_colors))
    target_bbox_t = tuple(int(x) for x in delta.changed_bbox)
    out_cands: List[InverseCandidateV129] = []
    for border_color in cand_colors:
        for expr in bbox_exprs:
            try:
                bb = _apply_steps_bbox_v129(expr.steps, inp)
            except Exception:
                continue
            if bb.to_tuple() != target_bbox_t:
                continue
            st = {"op_id": "draw_rect_border", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "gB", "args": {"color": int(border_color), "thickness": 1}}
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v129(steps)
            out_cands.append(
                InverseCandidateV129(op_name="draw_rect_border", steps=tuple(steps), cost_bits=int(cost), evidence=(("border_color", int(border_color)), ("bbox_expr_sig", str(expr.expr_sig())), ("thickness", 1),),)
            )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_paint_mask_v129(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV129,
    mask_exprs: Sequence[MaskExprV129],
    bg_candidates: Sequence[int],
    max_candidates: int = 64,
) -> List[InverseCandidateV129]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    paint_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    paint_colors = sorted(set(int(c) for c in paint_colors))
    only_colors = sorted(set(int(c) for c in delta.in_colors_in_changed))
    bgs = sorted(set(int(x) for x in bg_candidates))

    out_cands: List[InverseCandidateV129] = []
    base_changed = int(delta.changed_cells)
    for me in mask_exprs:
        try:
            mask = _apply_steps_v129(steps=me.steps, g_cur=inp)
        except Exception:
            continue
        if not (isinstance(mask, tuple) and (not mask or isinstance(mask[0], tuple))):
            continue
        overlap = 0
        hi, wi = grid_shape_v124(inp)
        for r in range(hi):
            for c in range(wi):
                if int(mask[r][c]) != 0 and int(delta.changed_mask[r][c]) != 0:
                    overlap += 1
        if overlap <= 0:
            continue

        for pc in paint_colors:
            st = {"op_id": "paint_mask", "in_vars": ["gC", str(me.mask_var)], "out_var": "gP", "args": {"color": int(pc), "mode": "overwrite"}}
            steps = list(me.steps) + [st]
            try:
                got = _apply_steps_v129(steps=steps, g_cur=inp)
            except Exception:
                continue
            if not isinstance(got, tuple):
                continue
            d2 = compute_delta_v129(got, out)
            if int(d2.changed_cells) < base_changed:
                cost = _program_cost_bits_v129(steps)
                out_cands.append(
                    InverseCandidateV129(
                        op_name="paint_mask",
                        steps=tuple(steps),
                        cost_bits=int(cost),
                        evidence=(("paint_color", int(pc)), ("mode", "overwrite"), ("mask_expr_sig", str(me.expr_sig())), ("mask_overlap", int(overlap)),),
                    )
                )

            for bg in bgs:
                st = {"op_id": "paint_mask", "in_vars": ["gC", str(me.mask_var)], "out_var": "gP", "args": {"color": int(pc), "mode": "only_bg", "bg": int(bg)}}
                steps = list(me.steps) + [st]
                try:
                    got = _apply_steps_v129(steps=steps, g_cur=inp)
                except Exception:
                    continue
                if not isinstance(got, tuple):
                    continue
                d2 = compute_delta_v129(got, out)
                if int(d2.changed_cells) < base_changed:
                    cost = _program_cost_bits_v129(steps)
                    out_cands.append(
                        InverseCandidateV129(
                            op_name="paint_mask",
                            steps=tuple(steps),
                            cost_bits=int(cost),
                            evidence=(("paint_color", int(pc)), ("mode", "only_bg"), ("bg", int(bg)), ("mask_expr_sig", str(me.expr_sig())), ("mask_overlap", int(overlap)),),
                        )
                    )

            for oc in only_colors:
                st = {
                    "op_id": "paint_mask",
                    "in_vars": ["gC", str(me.mask_var)],
                    "out_var": "gP",
                    "args": {"color": int(pc), "mode": "only_color", "only_color": int(oc)},
                }
                steps = list(me.steps) + [st]
                try:
                    got = _apply_steps_v129(steps=steps, g_cur=inp)
                except Exception:
                    continue
                if not isinstance(got, tuple):
                    continue
                d2 = compute_delta_v129(got, out)
                if int(d2.changed_cells) < base_changed:
                    cost = _program_cost_bits_v129(steps)
                    out_cands.append(
                        InverseCandidateV129(
                            op_name="paint_mask",
                            steps=tuple(steps),
                            cost_bits=int(cost),
                            evidence=(("paint_color", int(pc)), ("mode", "only_color"), ("only_color", int(oc)), ("mask_expr_sig", str(me.expr_sig())), ("mask_overlap", int(overlap)),),
                        )
                    )

    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands[: int(max_candidates)]


def inverse_propose_paste_v129(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV129,
    bg_candidates: Sequence[int],
    max_candidates: int = 32,
) -> List[InverseCandidateV129]:
    if delta.shape_changed:
        return []
    hi, wi = grid_shape_v124(inp)
    if hi == 0 or wi == 0:
        return []
    bgs = sorted(set(int(x) for x in bg_candidates))
    out_cands: List[InverseCandidateV129] = []

    # Try cropping the input bbox and pasting it at a detected offset in the output.
    from .arc_dsl_v126 import bbox_nonzero_v126, crop_bbox_v126

    for bg in bgs:
        bb = bbox_nonzero_v126(inp, bg=int(bg))
        patch = crop_bbox_v126(inp, bb)
        ph, pw = grid_shape_v124(patch)
        if ph == 0 or pw == 0:
            continue
        # Search for patch occurrences in out by scanning top-left positions, but only inside changed_bbox.
        r0, c0, r1, c1 = tuple(int(x) for x in delta.changed_bbox)
        rr0 = max(0, int(r0 - ph))
        cc0 = max(0, int(c0 - pw))
        rr1 = min(hi, int(r1))
        cc1 = min(wi, int(c1))
        for top in range(rr0, rr1):
            for left in range(cc0, cc1):
                if top + ph > hi or left + pw > wi:
                    continue
                match = True
                for pr in range(ph):
                    for pc in range(pw):
                        if int(out[top + pr][left + pc]) != int(patch[pr][pc]):
                            match = False
                            break
                    if not match:
                        break
                if not match:
                    continue
                st1 = {"op_id": "crop_bbox", "in_vars": ["gC", "bb1"], "out_var": "p1", "args": {}}
                st0 = {"op_id": "bbox_nonzero", "in_vars": ["gC"], "out_var": "bb1", "args": {"bg": int(bg)}}
                st2 = {"op_id": "paste", "in_vars": ["gC", "p1"], "out_var": "gP", "args": {"top": int(top), "left": int(left), "transparent": int(bg)}}
                steps = [st0, st1, st2]
                try:
                    got = _apply_steps_v129(steps=steps, g_cur=inp)
                except Exception:
                    continue
                if not isinstance(got, tuple):
                    continue
                d2 = compute_delta_v129(got, out)
                if int(d2.changed_cells) >= int(delta.changed_cells):
                    continue
                cost = _program_cost_bits_v129(steps)
                out_cands.append(
                    InverseCandidateV129(
                        op_name="paste",
                        steps=tuple(steps),
                        cost_bits=int(cost),
                        evidence=(("bg", int(bg)), ("top", int(top)), ("left", int(left)), ("patch_bbox", bb.to_dict()),),
                    )
                )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands[: int(max_candidates)]


def inverse_propose_overlay_v129(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV129,
    bg_candidates: Sequence[int],
    max_candidates: int = 16,
) -> List[InverseCandidateV129]:
    if delta.shape_changed:
        return []
    hi, wi = grid_shape_v124(inp)
    if hi == 0 or wi == 0:
        return []
    bgs = sorted(set(int(x) for x in bg_candidates))
    out_cands: List[InverseCandidateV129] = []
    # If out == overlay(inp, out, transparent=bg) is tautological; instead require reduction by overlaying a translated copy of inp.
    # Derive candidate offsets from evidence: search for the input's non-bg patch occurring elsewhere in the output.
    from .arc_dsl_v126 import bbox_nonzero_v126, crop_bbox_v126

    seen: set[Tuple[int, int, int]] = set()
    for bg in bgs:
        bb_in = bbox_nonzero_v126(inp, bg=int(bg))
        patch = crop_bbox_v126(inp, bb_in)
        ph, pw = grid_shape_v124(patch)
        if ph == 0 or pw == 0:
            continue

        r0, c0, r1, c1 = tuple(int(x) for x in delta.changed_bbox)
        if (r0, c0, r1, c1) == (0, 0, 0, 0):
            continue
        rr0 = max(0, int(r0 - ph))
        cc0 = max(0, int(c0 - pw))
        rr1 = min(hi, int(r1))
        cc1 = min(wi, int(c1))
        for top in range(rr0, rr1):
            for left in range(cc0, cc1):
                if top + ph > hi or left + pw > wi:
                    continue
                match = True
                for pr in range(ph):
                    for pc in range(pw):
                        if int(out[top + pr][left + pc]) != int(patch[pr][pc]):
                            match = False
                            break
                    if not match:
                        break
                if not match:
                    continue
                dx = int(left - int(bb_in.c0))
                dy = int(top - int(bb_in.r0))
                if dx == 0 and dy == 0:
                    continue
                key = (int(dx), int(dy), int(bg))
                if key in seen:
                    continue
                seen.add(key)
                st1 = {"op_id": "translate", "in_vars": ["gC"], "out_var": "gT", "args": {"dx": int(dx), "dy": int(dy), "pad": int(bg)}}
                st2 = {"op_id": "overlay", "in_vars": ["gC", "gT"], "out_var": "gO", "args": {"transparent": int(bg)}}
                steps = [st1, st2]
                try:
                    got = _apply_steps_v129(steps=steps, g_cur=inp)
                except Exception:
                    continue
                if not (isinstance(got, tuple) and (not got or isinstance(got[0], tuple))):
                    continue
                if not grid_equal_v124(got, out):
                    continue
                cost = _program_cost_bits_v129(steps)
                out_cands.append(
                    InverseCandidateV129(
                        op_name="overlay",
                        steps=tuple(steps),
                        cost_bits=int(cost),
                        evidence=(("bg", int(bg)), ("dx", int(dx)), ("dy", int(dy)), ("patch_bbox", bb_in.to_dict()),),
                    )
                )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands[: int(max_candidates)]


def build_inverse_candidates_v129(
    *,
    inp: GridV124,
    out: GridV124,
    selector_hypotheses: Sequence[SelectorHypothesisV129],
) -> Tuple[List[InverseCandidateV129], Dict[str, Any]]:
    delta = compute_delta_v129(inp, out)
    # Background candidates: mode of grids + corners + explicit 0.
    bgs: List[int] = [0]
    for g in (inp, out):
        cols = unique_colors_v124(g)
        if cols:
            # include most frequent and corners deterministically
            bgs.append(int(_mode_color_v129(g)))
            h, w = grid_shape_v124(g)
            if h > 0 and w > 0:
                bgs.extend([int(g[0][0]), int(g[0][w - 1]), int(g[h - 1][0]), int(g[h - 1][w - 1])])
    bg_candidates = sorted(set(int(x) for x in bgs))

    color_candidates: List[int] = []
    for g in (inp, out):
        color_candidates.extend([int(c) for c in unique_colors_v124(g)])
    if delta.changed_mask:
        color_candidates.extend([int(c) for c in delta.out_colors_in_changed])
        color_candidates.extend([int(c) for c in delta.in_colors_in_changed])
    color_candidates = sorted(set(int(c) for c in color_candidates))

    bbox_exprs = propose_bbox_exprs_v129(bg_candidates=bg_candidates, color_candidates=color_candidates, selector_hypotheses=selector_hypotheses)
    mask_exprs = propose_mask_exprs_v129(color_candidates=color_candidates, bbox_exprs=bbox_exprs, selector_hypotheses=selector_hypotheses)

    cands: List[InverseCandidateV129] = []
    cands.extend(inverse_propose_symmetry_v129(inp=inp, out=out))
    cands.extend(inverse_propose_scale_up_v129(inp=inp, out=out))
    cands.extend(inverse_propose_tile_v129(inp=inp, out=out))
    cands.extend(inverse_propose_crop_bbox_v129(inp=inp, out=out, delta=delta, bbox_exprs=bbox_exprs))
    cands.extend(inverse_propose_map_colors_v129(inp=inp, out=out))
    cands.extend(inverse_propose_translate_v129(inp=inp, out=out, bg_candidates=bg_candidates))
    cands.extend(inverse_propose_fill_rect_v129(inp=inp, out=out, delta=delta, bbox_exprs=bbox_exprs))
    cands.extend(inverse_propose_draw_rect_border_v129(inp=inp, out=out, delta=delta, bbox_exprs=bbox_exprs))
    cands.extend(inverse_propose_paint_mask_v129(inp=inp, out=out, delta=delta, mask_exprs=mask_exprs, bg_candidates=bg_candidates))
    cands.extend(inverse_propose_paste_v129(inp=inp, out=out, delta=delta, bg_candidates=bg_candidates))
    cands.extend(inverse_propose_overlay_v129(inp=inp, out=out, delta=delta, bg_candidates=bg_candidates))

    uniq: Dict[str, InverseCandidateV129] = {}
    for c in cands:
        uniq[str(c.candidate_sig())] = c
    out_cands = list(uniq.values())
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    trace: Dict[str, Any] = {
        "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V129),
        "delta_sig": str(delta.delta_sig()),
        "bg_candidates": [int(x) for x in bg_candidates],
        "color_candidates": [int(x) for x in color_candidates],
        "bbox_exprs_total": int(len(bbox_exprs)),
        "mask_exprs_total": int(len(mask_exprs)),
        "candidates_total": int(len(out_cands)),
    }
    return out_cands, trace
