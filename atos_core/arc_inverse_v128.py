from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_delta_v128 import DeltaEvidenceV128, compute_delta_v128
from .arc_dsl_v126 import BboxV126
from .arc_dsl_v128 import OP_DEFS_V128, apply_op_v128
from .arc_selector_v128 import SelectorHypothesisV128
from .grid_v124 import GridV124, grid_equal_v124, grid_shape_v124, translate_v124, unique_colors_v124

ARC_INVERSE_SCHEMA_VERSION_V128 = 128


def _step_cost_bits_v128(step: Dict[str, Any]) -> int:
    op = str(step.get("op_id") or "")
    if op in OP_DEFS_V128:
        base = int(OP_DEFS_V128[op].cost_bits)
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


def _program_cost_bits_v128(steps: Sequence[Dict[str, Any]]) -> int:
    return int(sum(_step_cost_bits_v128(s) for s in steps))


def _program_sig_v128(steps: Sequence[Dict[str, Any]]) -> str:
    body = {"schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V128), "kind": "arc_program_steps_v128", "steps": list(steps)}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class BboxExprV128:
    steps: Tuple[Dict[str, Any], ...]
    bbox_var: str
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V128),
            "kind": "bbox_expr_v128",
            "steps": list(self.steps),
            "bbox_var": str(self.bbox_var),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def expr_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class MaskExprV128:
    steps: Tuple[Dict[str, Any], ...]
    mask_var: str
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V128),
            "kind": "mask_expr_v128",
            "steps": list(self.steps),
            "mask_var": str(self.mask_var),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def expr_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class InverseCandidateV128:
    op_name: str
    steps: Tuple[Dict[str, Any], ...]
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V128),
            "kind": "inverse_candidate_v128",
            "op_name": str(self.op_name),
            "steps": list(self.steps),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def candidate_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _apply_steps_v128(*, steps: Sequence[Dict[str, Any]], g_cur: GridV124) -> Any:
    env: Dict[str, Any] = {"gC": g_cur}
    for st in steps:
        op = str(st.get("op_id") or "")
        in_vars = st.get("in_vars") if isinstance(st.get("in_vars"), list) else []
        out_var = str(st.get("out_var") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        ins = [env[str(v)] for v in in_vars]
        if op in OP_DEFS_V128:
            env[out_var] = apply_op_v128(op_id=op, inputs=ins, args=dict(args))
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


def _apply_steps_bbox_v128(steps: Sequence[Dict[str, Any]], g_cur: GridV124) -> BboxV126:
    out = _apply_steps_v128(steps=steps, g_cur=g_cur)
    if not isinstance(out, BboxV126):
        raise ValueError("bbox_expr_not_bbox")
    return out


def propose_bbox_exprs_v128(
    *,
    bg_candidates: Sequence[int],
    color_candidates: Sequence[int],
    selector_hypotheses: Sequence[SelectorHypothesisV128],
    max_expand_delta: int = 2,
) -> List[BboxExprV128]:
    base: List[BboxExprV128] = []
    for bg in sorted(set(int(x) for x in bg_candidates)):
        st = {"op_id": "bbox_nonzero", "in_vars": ["gC"], "out_var": "bb1", "args": {"bg": int(bg)}}
        cost = _program_cost_bits_v128([st])
        base.append(BboxExprV128(steps=(st,), bbox_var="bb1", cost_bits=int(cost), evidence=(("kind", "bbox_nonzero"), ("bg", int(bg))),))
    for c in sorted(set(int(x) for x in color_candidates)):
        st = {"op_id": "bbox_by_color", "in_vars": ["gC"], "out_var": "bb1", "args": {"color": int(c)}}
        cost = _program_cost_bits_v128([st])
        base.append(BboxExprV128(steps=(st,), bbox_var="bb1", cost_bits=int(cost), evidence=(("kind", "bbox_by_color"), ("color", int(c))),))

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
        cost = _program_cost_bits_v128(steps)
        ev: List[Tuple[str, Any]] = [("kind", "bbox_of_selected_object"), ("selector", str(sel))]
        if cf is not None:
            ev.append(("color_filter", int(cf)))
        base.append(BboxExprV128(steps=tuple(steps), bbox_var="bb3", cost_bits=int(cost), evidence=tuple(ev)))

    expanded: List[BboxExprV128] = []
    for expr in base:
        expanded.append(expr)
        for d in range(1, int(max_expand_delta) + 1):
            st = {"op_id": "bbox_expand", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "bbE", "args": {"delta": int(d)}}
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v128(steps)
            ev = list(expr.evidence) + [("expand_delta", int(d))]
            expanded.append(BboxExprV128(steps=tuple(steps), bbox_var="bbE", cost_bits=int(cost), evidence=tuple(ev)))

    seen: set[str] = set()
    uniq: List[BboxExprV128] = []
    for e in expanded:
        sig = _program_sig_v128(e.steps)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(e)
    uniq.sort(key=lambda e: (int(e.cost_bits), str(e.expr_sig())))
    return uniq


def propose_mask_exprs_v128(
    *,
    color_candidates: Sequence[int],
    bbox_exprs: Sequence[BboxExprV128],
    selector_hypotheses: Sequence[SelectorHypothesisV128],
) -> List[MaskExprV128]:
    base: List[MaskExprV128] = []
    for c in sorted(set(int(x) for x in color_candidates)):
        st = {"op_id": "mask_by_color", "in_vars": ["gC"], "out_var": "m1", "args": {"color": int(c)}}
        cost = _program_cost_bits_v128([st])
        base.append(MaskExprV128(steps=(st,), mask_var="m1", cost_bits=int(cost), evidence=(("kind", "mask_by_color"), ("color", int(c))),))

    # mask_rect(bbox_expr)
    for be in bbox_exprs:
        st = {"op_id": "mask_rect", "in_vars": ["gC", str(be.bbox_var)], "out_var": "mR", "args": {}}
        steps = list(be.steps) + [st]
        cost = _program_cost_bits_v128(steps)
        base.append(MaskExprV128(steps=tuple(steps), mask_var="mR", cost_bits=int(cost), evidence=(("kind", "mask_rect"), ("bbox_expr_sig", str(be.expr_sig()))),))

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
        cost = _program_cost_bits_v128(steps)
        ev: List[Tuple[str, Any]] = [("kind", "mask_of_selected_object"), ("selector", str(sel))]
        if cf is not None:
            ev.append(("color_filter", int(cf)))
        base.append(MaskExprV128(steps=tuple(steps), mask_var="m3", cost_bits=int(cost), evidence=tuple(ev)))

    seen: set[str] = set()
    uniq: List[MaskExprV128] = []
    for e in base:
        sig = _program_sig_v128(e.steps)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(e)
    uniq.sort(key=lambda e: (int(e.cost_bits), str(e.expr_sig())))
    return uniq


def inverse_propose_map_colors_v128(*, inp: GridV124, out: GridV124) -> List[InverseCandidateV128]:
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
    cost = _program_cost_bits_v128([st])
    return [InverseCandidateV128(op_name="map_colors", steps=(st,), cost_bits=int(cost), evidence=(("mapping_size", int(len(m))),))]


def inverse_propose_translate_v128(*, inp: GridV124, out: GridV124, bg_candidates: Sequence[int]) -> List[InverseCandidateV128]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo) or hi == 0 or wi == 0:
        return []
    out_cands: List[InverseCandidateV128] = []
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
        cost = _program_cost_bits_v128([st])
        out_cands.append(
            InverseCandidateV128(op_name="translate", steps=(st,), cost_bits=int(cost), evidence=(("dx", int(dx)), ("dy", int(dy)), ("pad", int(bg))),)
        )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_fill_rect_v128(
    *, inp: GridV124, out: GridV124, delta: DeltaEvidenceV128, bbox_exprs: Sequence[BboxExprV128]
) -> List[InverseCandidateV128]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    cand_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    cand_colors = sorted(set(int(c) for c in cand_colors))
    target_bbox_t = tuple(int(x) for x in delta.changed_bbox)
    out_cands: List[InverseCandidateV128] = []
    for fill_color in cand_colors:
        for expr in bbox_exprs:
            try:
                bb = _apply_steps_bbox_v128(expr.steps, inp)
            except Exception:
                continue
            if bb.to_tuple() != target_bbox_t:
                continue
            st = {"op_id": "fill_rect", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "gF", "args": {"color": int(fill_color)}}
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v128(steps)
            out_cands.append(
                InverseCandidateV128(op_name="fill_rect", steps=tuple(steps), cost_bits=int(cost), evidence=(("fill_color", int(fill_color)), ("bbox_expr_sig", str(expr.expr_sig()))),)
            )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_draw_rect_border_v128(
    *, inp: GridV124, out: GridV124, delta: DeltaEvidenceV128, bbox_exprs: Sequence[BboxExprV128]
) -> List[InverseCandidateV128]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    cand_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    cand_colors = sorted(set(int(c) for c in cand_colors))
    target_bbox_t = tuple(int(x) for x in delta.changed_bbox)
    out_cands: List[InverseCandidateV128] = []
    for border_color in cand_colors:
        for expr in bbox_exprs:
            try:
                bb = _apply_steps_bbox_v128(expr.steps, inp)
            except Exception:
                continue
            if bb.to_tuple() != target_bbox_t:
                continue
            st = {"op_id": "draw_rect_border", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "gB", "args": {"color": int(border_color), "thickness": 1}}
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v128(steps)
            out_cands.append(
                InverseCandidateV128(op_name="draw_rect_border", steps=tuple(steps), cost_bits=int(cost), evidence=(("border_color", int(border_color)), ("bbox_expr_sig", str(expr.expr_sig())), ("thickness", 1),),)
            )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_paint_mask_v128(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV128,
    mask_exprs: Sequence[MaskExprV128],
    bg_candidates: Sequence[int],
    max_candidates: int = 64,
) -> List[InverseCandidateV128]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    # Candidate target colors: prefer newly introduced colors.
    paint_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    paint_colors = sorted(set(int(c) for c in paint_colors))
    only_colors = sorted(set(int(c) for c in delta.in_colors_in_changed))
    bgs = sorted(set(int(x) for x in bg_candidates))

    out_cands: List[InverseCandidateV128] = []
    base_changed = int(delta.changed_cells)
    for me in mask_exprs:
        # Quick overlap check: mask must hit some changed cell on this pair.
        try:
            mask = _apply_steps_v128(steps=me.steps, g_cur=inp)
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
            # overwrite
            st = {"op_id": "paint_mask", "in_vars": ["gC", str(me.mask_var)], "out_var": "gP", "args": {"color": int(pc), "mode": "overwrite"}}
            steps = list(me.steps) + [st]
            try:
                got = _apply_steps_v128(steps=steps, g_cur=inp)
            except Exception:
                continue
            if not isinstance(got, tuple):
                continue
            d2 = compute_delta_v128(got, out)
            if int(d2.changed_cells) < base_changed:
                cost = _program_cost_bits_v128(steps)
                out_cands.append(
                    InverseCandidateV128(
                        op_name="paint_mask",
                        steps=tuple(steps),
                        cost_bits=int(cost),
                        evidence=(("paint_color", int(pc)), ("mode", "overwrite"), ("mask_expr_sig", str(me.expr_sig())), ("mask_overlap", int(overlap)),),
                    )
                )

            # only_bg
            for bg in bgs:
                st = {"op_id": "paint_mask", "in_vars": ["gC", str(me.mask_var)], "out_var": "gP", "args": {"color": int(pc), "mode": "only_bg", "bg": int(bg)}}
                steps = list(me.steps) + [st]
                try:
                    got = _apply_steps_v128(steps=steps, g_cur=inp)
                except Exception:
                    continue
                if not isinstance(got, tuple):
                    continue
                d2 = compute_delta_v128(got, out)
                if int(d2.changed_cells) < base_changed:
                    cost = _program_cost_bits_v128(steps)
                    out_cands.append(
                        InverseCandidateV128(
                            op_name="paint_mask",
                            steps=tuple(steps),
                            cost_bits=int(cost),
                            evidence=(("paint_color", int(pc)), ("mode", "only_bg"), ("bg", int(bg)), ("mask_expr_sig", str(me.expr_sig())), ("mask_overlap", int(overlap)),),
                        )
                    )

            # only_color
            for oc in only_colors:
                st = {"op_id": "paint_mask", "in_vars": ["gC", str(me.mask_var)], "out_var": "gP", "args": {"color": int(pc), "mode": "only_color", "only_color": int(oc)}}
                steps = list(me.steps) + [st]
                try:
                    got = _apply_steps_v128(steps=steps, g_cur=inp)
                except Exception:
                    continue
                if not isinstance(got, tuple):
                    continue
                d2 = compute_delta_v128(got, out)
                if int(d2.changed_cells) < base_changed:
                    cost = _program_cost_bits_v128(steps)
                    out_cands.append(
                        InverseCandidateV128(
                            op_name="paint_mask",
                            steps=tuple(steps),
                            cost_bits=int(cost),
                            evidence=(("paint_color", int(pc)), ("mode", "only_color"), ("only_color", int(oc)), ("mask_expr_sig", str(me.expr_sig())), ("mask_overlap", int(overlap)),),
                        )
                    )

    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands[: int(max_candidates)]


def _find_patch_occurrences_v128(*, grid: GridV124, patch: GridV124) -> List[Tuple[int, int]]:
    hg, wg = grid_shape_v124(grid)
    hp, wp = grid_shape_v124(patch)
    if hp == 0 or wp == 0 or hg == 0 or wg == 0 or hp > hg or wp > wg:
        return []
    hits: List[Tuple[int, int]] = []
    for top in range(0, hg - hp + 1):
        for left in range(0, wg - wp + 1):
            ok = True
            for r in range(hp):
                for c in range(wp):
                    if int(grid[top + r][left + c]) != int(patch[r][c]):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                hits.append((int(top), int(left)))
    return hits


def inverse_propose_paste_v128(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV128,
    bbox_exprs: Sequence[BboxExprV128],
    bg_candidates: Sequence[int],
    max_candidates: int = 48,
) -> List[InverseCandidateV128]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    bgs = sorted(set(int(x) for x in bg_candidates))
    out_cands: List[InverseCandidateV128] = []
    base_changed = int(delta.changed_cells)
    for expr in bbox_exprs[:24]:
        try:
            bb = _apply_steps_bbox_v128(expr.steps, inp)
        except Exception:
            continue
        # crop_bbox is in v126; still callable via apply_op_v128.
        st_crop = {"op_id": "crop_bbox", "in_vars": ["gC", str(expr.bbox_var)], "out_var": "patch1", "args": {}}
        steps_crop = list(expr.steps) + [st_crop]
        try:
            patch = _apply_steps_v128(steps=steps_crop, g_cur=inp)
        except Exception:
            continue
        if not (isinstance(patch, tuple) and (not patch or isinstance(patch[0], tuple))):
            continue
        hp, wp = grid_shape_v124(patch)
        if hp == 0 or wp == 0:
            continue
        # Find exact matches of the patch in out, deterministic scan.
        hits = _find_patch_occurrences_v128(grid=out, patch=patch)
        if not hits:
            continue
        # Prefer hits that touch the changed bbox.
        r0, c0, r1, c1 = tuple(int(x) for x in delta.changed_bbox)
        scored: List[Tuple[int, int, int]] = []
        for top, left in hits:
            rr0 = int(top)
            cc0 = int(left)
            rr1 = int(top + hp)
            cc1 = int(left + wp)
            inter = max(0, min(rr1, r1) - max(rr0, r0)) * max(0, min(cc1, c1) - max(cc0, c0))
            scored.append((int(-inter), int(top), int(left)))
        scored.sort()
        for _neg_inter, top, left in scored[:6]:
            for bg in bgs:
                st_paste = {"op_id": "paste", "in_vars": ["gC", "patch1"], "out_var": "gP", "args": {"top": int(top), "left": int(left), "transparent": int(bg)}}
                steps = steps_crop + [st_paste]
                try:
                    got = _apply_steps_v128(steps=steps, g_cur=inp)
                except Exception:
                    continue
                if not isinstance(got, tuple):
                    continue
                d2 = compute_delta_v128(got, out)
                if int(d2.changed_cells) < base_changed:
                    cost = _program_cost_bits_v128(steps)
                    out_cands.append(
                        InverseCandidateV128(
                            op_name="paste",
                            steps=tuple(steps),
                            cost_bits=int(cost),
                            evidence=(("bbox_expr_sig", str(expr.expr_sig())), ("dest_top", int(top)), ("dest_left", int(left)), ("transparent", int(bg)), ("patch_shape", {"h": int(hp), "w": int(wp)}),),
                        )
                    )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands[: int(max_candidates)]


def inverse_propose_overlay_v128(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV128,
    bg_candidates: Sequence[int],
    max_candidates: int = 24,
) -> List[InverseCandidateV128]:
    # Minimal overlay abduction: overlay(inp, translate(inp)) or overlay(inp, map_colors(inp)).
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo) or hi == 0 or wi == 0:
        return []
    bgs = sorted(set(int(x) for x in bg_candidates))
    out_cands: List[InverseCandidateV128] = []
    base_changed = int(delta.changed_cells)

    # translate-derived top
    for t in inverse_propose_translate_v128(inp=inp, out=out, bg_candidates=bgs)[:6]:
        # If translate already yields out, overlay would be redundant.
        pass
    # Instead: enumerate small translations inferred from bbox delta, and check overlay equality.
    from .grid_v124 import bbox_nonzero_v124

    for bg in bgs:
        bb_in = bbox_nonzero_v124(inp, bg=int(bg))
        bb_out = bbox_nonzero_v124(out, bg=int(bg))
        if bb_in == (0, 0, 0, 0) or bb_out == (0, 0, 0, 0):
            continue
        dx = int(bb_out[1] - bb_in[1])
        dy = int(bb_out[0] - bb_in[0])
        top_grid = translate_v124(inp, dx=int(dx), dy=int(dy), pad=int(bg))
        # overlay is in v126 via apply_op_v128.
        try:
            got = apply_op_v128(op_id="overlay", inputs=[inp, top_grid], args={"transparent": int(bg)})
        except Exception:
            continue
        if not (isinstance(got, tuple) and (not got or isinstance(got[0], tuple))):
            continue
        if not grid_equal_v124(got, out):
            # Still accept as partial if it reduces delta.
            d2 = compute_delta_v128(got, out)
            if int(d2.changed_cells) >= base_changed:
                continue
        st1 = {"op_id": "translate", "in_vars": ["gC"], "out_var": "gT", "args": {"dx": int(dx), "dy": int(dy), "pad": int(bg)}}
        st2 = {"op_id": "overlay", "in_vars": ["gC", "gT"], "out_var": "gO", "args": {"transparent": int(bg)}}
        steps = [st1, st2]
        cost = _program_cost_bits_v128(steps)
        out_cands.append(
            InverseCandidateV128(
                op_name="overlay",
                steps=tuple(steps),
                cost_bits=int(cost),
                evidence=(("dx", int(dx)), ("dy", int(dy)), ("transparent", int(bg)),),
            )
        )

    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands[: int(max_candidates)]

