from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .arc_delta_v127 import DeltaEvidenceV127
from .arc_dsl_v126 import BboxV126, OP_DEFS_V126, apply_op_v126, bbox_from_tuple_v126
from .grid_v124 import GridV124, grid_equal_v124, grid_shape_v124, translate_v124

ARC_INVERSE_SCHEMA_VERSION_V127 = 127


def _step_cost_bits_v127(step: Dict[str, Any]) -> int:
    op = str(step.get("op_id") or "")
    if op in OP_DEFS_V126:
        base = int(OP_DEFS_V126[op].cost_bits)
    elif op in {"map_colors"}:
        base = 16
    elif op in {"translate"}:
        base = 12
    else:
        base = 20
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


def _program_cost_bits_v127(steps: Sequence[Dict[str, Any]]) -> int:
    return int(sum(_step_cost_bits_v127(s) for s in steps))


def _program_sig_v127(steps: Sequence[Dict[str, Any]]) -> str:
    body = {"schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V127), "kind": "arc_program_steps_v127", "steps": list(steps)}
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


@dataclass(frozen=True)
class BboxExprV127:
    steps: Tuple[Dict[str, Any], ...]
    bbox_var: str
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V127),
            "kind": "bbox_expr_v127",
            "steps": list(self.steps),
            "bbox_var": str(self.bbox_var),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def expr_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


@dataclass(frozen=True)
class InverseCandidateV127:
    op_name: str
    steps: Tuple[Dict[str, Any], ...]
    cost_bits: int
    evidence: Tuple[Tuple[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        ev: Dict[str, Any] = {}
        for k, v in sorted(self.evidence, key=lambda kv: str(kv[0])):
            ev[str(k)] = v
        return {
            "schema_version": int(ARC_INVERSE_SCHEMA_VERSION_V127),
            "kind": "inverse_candidate_v127",
            "op_name": str(self.op_name),
            "steps": list(self.steps),
            "cost_bits": int(self.cost_bits),
            "evidence": ev,
        }

    def candidate_sig(self) -> str:
        return sha256_hex(canonical_json_dumps(self.to_dict()).encode("utf-8"))


def _apply_steps_bbox_v127(steps: Sequence[Dict[str, Any]], inp: GridV124) -> BboxV126:
    env: Dict[str, Any] = {"g0": inp}
    for st in steps:
        op = str(st.get("op_id") or "")
        in_vars = st.get("in_vars") if isinstance(st.get("in_vars"), list) else []
        out_var = str(st.get("out_var") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        ins = [env[str(v)] for v in in_vars]
        if op in OP_DEFS_V126:
            env[out_var] = apply_op_v126(op_id=op, inputs=ins, args=dict(args))
        else:
            raise ValueError(f"unsupported_bbox_expr_op:{op}")
    last = env[str(steps[-1]["out_var"])]
    if not isinstance(last, BboxV126):
        raise ValueError("bbox_expr_not_bbox")
    return last


def propose_bbox_exprs_v127(
    *,
    bg_candidates: Sequence[int],
    color_candidates: Sequence[int],
    selector_hypotheses: Sequence[Dict[str, Any]],
    max_expand_delta: int = 2,
) -> List[BboxExprV127]:
    """
    Build candidate bbox expressions (as step sequences) from generic sources:
    - bbox_nonzero(bg)
    - bbox_by_color(color)
    - bbox_of_selected_object(connected_components(color_filter), selector)
    plus bbox_expand(delta) variants.
    """
    base: List[BboxExprV127] = []
    for bg in sorted(set(int(x) for x in bg_candidates)):
        st = {"op_id": "bbox_nonzero", "in_vars": ["g0"], "out_var": "bb1", "args": {"bg": int(bg)}}
        cost = _program_cost_bits_v127([st])
        base.append(BboxExprV127(steps=(st,), bbox_var="bb1", cost_bits=int(cost), evidence=(("kind", "bbox_nonzero"), ("bg", int(bg))),))
    for c in sorted(set(int(x) for x in color_candidates)):
        st = {"op_id": "bbox_by_color", "in_vars": ["g0"], "out_var": "bb1", "args": {"color": int(c)}}
        cost = _program_cost_bits_v127([st])
        base.append(BboxExprV127(steps=(st,), bbox_var="bb1", cost_bits=int(cost), evidence=(("kind", "bbox_by_color"), ("color", int(c))),))

    for h in selector_hypotheses:
        if not isinstance(h, dict):
            continue
        cf = h.get("color_filter")
        sel = str(h.get("selector") or "")
        args_cc: Dict[str, Any] = {}
        if cf is not None:
            args_cc["color"] = int(cf)
        st1 = {"op_id": "connected_components", "in_vars": ["g0"], "out_var": "os1", "args": args_cc}
        st2 = {"op_id": "select_object", "in_vars": ["os1"], "out_var": "o2", "args": {"selector": str(sel)}}
        st3 = {"op_id": "bbox_of_object", "in_vars": ["o2"], "out_var": "bb3", "args": {}}
        steps = [st1, st2, st3]
        cost = _program_cost_bits_v127(steps)
        ev: List[Tuple[str, Any]] = [("kind", "bbox_of_selected_object"), ("selector", str(sel))]
        if cf is not None:
            ev.append(("color_filter", int(cf)))
        base.append(BboxExprV127(steps=tuple(steps), bbox_var="bb3", cost_bits=int(cost), evidence=tuple(ev)))

    expanded: List[BboxExprV127] = []
    for expr in base:
        expanded.append(expr)
        for d in range(1, int(max_expand_delta) + 1):
            st = {
                "op_id": "bbox_expand",
                "in_vars": ["g0", str(expr.bbox_var)],
                "out_var": "bbE",
                "args": {"delta": int(d)},
            }
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v127(steps)
            ev = list(expr.evidence) + [("expand_delta", int(d))]
            expanded.append(BboxExprV127(steps=tuple(steps), bbox_var="bbE", cost_bits=int(cost), evidence=tuple(ev)))

    # Dedup by signature of steps, stable sort by (cost_bits, expr_sig)
    seen: set[str] = set()
    uniq: List[BboxExprV127] = []
    for e in expanded:
        sig = _program_sig_v127(e.steps)
        if sig in seen:
            continue
        seen.add(sig)
        uniq.append(e)
    uniq.sort(key=lambda e: (int(e.cost_bits), str(e.expr_sig())))
    return uniq


def inverse_propose_map_colors_v127(*, inp: GridV124, out: GridV124) -> List[InverseCandidateV127]:
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
    st = {"op_id": "map_colors", "in_vars": ["g0"], "out_var": "g1", "args": {"mapping": m}}
    cost = _program_cost_bits_v127([st])
    cand = InverseCandidateV127(
        op_name="map_colors",
        steps=(st,),
        cost_bits=int(cost),
        evidence=(("mapping_size", int(len(m))),),
    )
    return [cand]


def inverse_propose_translate_v127(*, inp: GridV124, out: GridV124, bg_candidates: Sequence[int]) -> List[InverseCandidateV127]:
    hi, wi = grid_shape_v124(inp)
    ho, wo = grid_shape_v124(out)
    if (hi, wi) != (ho, wo) or hi == 0 or wi == 0:
        return []
    out_cands: List[InverseCandidateV127] = []
    seen: set[Tuple[int, int, int]] = set()
    for bg in sorted(set(int(x) for x in bg_candidates)):
        # Infer shift via bbox of non-bg mass under this bg hypothesis.
        from .grid_v124 import bbox_nonzero_v124

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
        st = {"op_id": "translate", "in_vars": ["g0"], "out_var": "g1", "args": {"dx": int(dx), "dy": int(dy), "pad": int(bg)}}
        cost = _program_cost_bits_v127([st])
        out_cands.append(
            InverseCandidateV127(
                op_name="translate",
                steps=(st,),
                cost_bits=int(cost),
                evidence=(("dx", int(dx)), ("dy", int(dy)), ("pad", int(bg))),
            )
        )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def _changed_color_singleton_v127(delta: DeltaEvidenceV127) -> Optional[int]:
    cols = list(delta.out_colors_in_changed)
    if len(cols) == 1:
        return int(cols[0])
    return None


def inverse_propose_fill_rect_v127(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV127,
    bbox_exprs: Sequence[BboxExprV127],
) -> List[InverseCandidateV127]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    # When multiple colors change, we still propose fills using candidate colors (prefer newly introduced ones).
    cand_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    cand_colors = sorted(set(int(c) for c in cand_colors))
    target_bbox_t = tuple(int(x) for x in delta.changed_bbox)
    out_cands: List[InverseCandidateV127] = []
    for fill_color in cand_colors:
        for expr in bbox_exprs:
            try:
                bb = _apply_steps_bbox_v127(expr.steps, inp)
            except Exception:
                continue
            if bb.to_tuple() != target_bbox_t:
                continue
            st = {
                "op_id": "fill_rect",
                "in_vars": ["g0", str(expr.bbox_var)],
                "out_var": "gF",
                "args": {"color": int(fill_color)},
            }
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v127(steps)
            ev: List[Tuple[str, Any]] = [("fill_color", int(fill_color)), ("bbox_expr_sig", str(expr.expr_sig()))]
            out_cands.append(
                InverseCandidateV127(op_name="fill_rect", steps=tuple(steps), cost_bits=int(cost), evidence=tuple(ev))
            )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands


def inverse_propose_draw_rect_border_v127(
    *,
    inp: GridV124,
    out: GridV124,
    delta: DeltaEvidenceV127,
    bbox_exprs: Sequence[BboxExprV127],
) -> List[InverseCandidateV127]:
    if delta.shape_changed or not delta.changed_mask:
        return []
    cand_colors = list(delta.palette_added) if delta.palette_added else list(delta.out_colors_in_changed)
    cand_colors = sorted(set(int(c) for c in cand_colors))
    target_bbox_t = tuple(int(x) for x in delta.changed_bbox)
    out_cands: List[InverseCandidateV127] = []
    for border_color in cand_colors:
        for expr in bbox_exprs:
            try:
                bb = _apply_steps_bbox_v127(expr.steps, inp)
            except Exception:
                continue
            if bb.to_tuple() != target_bbox_t:
                continue
            st = {
                "op_id": "draw_rect_border",
                "in_vars": ["g0", str(expr.bbox_var)],
                "out_var": "gB",
                "args": {"color": int(border_color), "thickness": 1},
            }
            steps = list(expr.steps) + [st]
            cost = _program_cost_bits_v127(steps)
            ev: List[Tuple[str, Any]] = [
                ("border_color", int(border_color)),
                ("bbox_expr_sig", str(expr.expr_sig())),
                ("thickness", 1),
            ]
            out_cands.append(
                InverseCandidateV127(op_name="draw_rect_border", steps=tuple(steps), cost_bits=int(cost), evidence=tuple(ev))
            )
    out_cands.sort(key=lambda c: (int(c.cost_bits), str(c.candidate_sig())))
    return out_cands
