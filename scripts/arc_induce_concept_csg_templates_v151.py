#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps
from atos_core.arc_ops_v141 import OP_DEFS_V141
from atos_core.grid_v124 import grid_shape_v124


def _stable_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except Exception:
                continue
            if isinstance(row, dict):
                yield row


def _type_feasible_from_start(op_ids: Sequence[str]) -> bool:
    avail: Set[str] = {"grid", "orig"}
    for op_id in op_ids:
        od = OP_DEFS_V141.get(str(op_id) or "")
        if od is None:
            return False
        reads = tuple(getattr(od, "reads", ()) or ())
        writes = tuple(getattr(od, "writes", ()) or ())
        if any(str(t) not in avail for t in reads):
            return False
        for t in writes:
            avail.add(str(t))
    return True


def _writes_grid(op_id: str) -> bool:
    od = OP_DEFS_V141.get(str(op_id) or "")
    if od is None:
        return False
    writes = tuple(getattr(od, "writes", ()) or ())
    return "grid" in {str(x) for x in writes if str(x)}


def _flatten_trace_steps(trace_steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten meta steps (concept_call / macro_call) to primitive steps.

    trace_candidates.jsonl stores programs as concept_calls around primitives; we mine the
    underlying primitive pipelines.
    """
    flat: List[Dict[str, Any]] = []
    for st in trace_steps:
        if not isinstance(st, dict):
            continue
        op = str(st.get("op_id") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        if not op:
            continue
        if op in {"concept_call", "macro_call"}:
            inner = args.get("steps")
            if isinstance(inner, list):
                flat.extend(_flatten_trace_steps([x for x in inner if isinstance(x, dict)]))
            continue
        flat.append({"op_id": str(op), "args": dict(args)})
    return flat


def _top1(counts: Dict[int, int]) -> Optional[int]:
    if not counts:
        return None
    items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    return int(items[0][0]) if items else None


def _bbox_of_color(g: Sequence[Sequence[int]], *, color: int) -> Optional[Tuple[int, int, int, int]]:
    r0 = c0 = 10**9
    r1 = c1 = -1
    for r, row in enumerate(g):
        for c, x in enumerate(row):
            if int(x) != int(color):
                continue
            r0 = min(int(r0), int(r))
            c0 = min(int(c0), int(c))
            r1 = max(int(r1), int(r))
            c1 = max(int(c1), int(c))
    if r1 < 0:
        return None
    return (int(r0), int(c0), int(r1), int(c1))


@dataclass(frozen=True)
class _BinderVals:
    # Keep in sync with atos_core.arc_solver_v141._resolve_concept_csg_binders_v141.
    bg: int
    in_height: int
    in_width: int
    out_height: Optional[int]
    out_width: Optional[int]
    diff_from_top1: Optional[int]
    diff_from_top2: Optional[int]
    diff_from_top3: Optional[int]
    diff_to_top1: Optional[int]
    diff_to_top2: Optional[int]
    diff_to_top3: Optional[int]
    diff_rmin: Optional[int]
    diff_cmin: Optional[int]
    diff_rmax: Optional[int]
    diff_cmax: Optional[int]
    offset_dy: Optional[int]
    offset_dx: Optional[int]
    in_only_nonbg: Optional[int]
    out_only_nonbg: Optional[int]
    in_top_nonbg: Optional[int]
    out_top_nonbg: Optional[int]


def _compute_binders(
    *, train_pairs: Sequence[Tuple[List[List[int]], List[List[int]]]], test_in: List[List[int]]
) -> _BinderVals:
    bg_counts: Dict[int, int] = {}
    for inp, out in train_pairs:
        for row in inp:
            for x in row:
                bg_counts[int(x)] = int(bg_counts.get(int(x), 0)) + 1
        for row in out:
            for x in row:
                bg_counts[int(x)] = int(bg_counts.get(int(x), 0)) + 1
    for row in test_in:
        for x in row:
            bg_counts[int(x)] = int(bg_counts.get(int(x), 0)) + 1
    items_bg = sorted(((int(c), int(n)) for c, n in bg_counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
    bg = int(items_bg[0][0]) if items_bg else 0
    in_h, in_w = grid_shape_v124(test_in)

    out_h: Optional[int] = None
    out_w: Optional[int] = None
    ok_out = True
    for _, out in train_pairs:
        ho, wo = grid_shape_v124(out)
        if out_h is None:
            out_h = int(ho)
            out_w = int(wo)
        elif int(ho) != int(out_h) or int(wo) != int(out_w):
            ok_out = False
            break
    if not ok_out:
        out_h = None
        out_w = None

    diff_from: Dict[int, int] = {}
    diff_to: Dict[int, int] = {}
    diff_rmin: Optional[int] = None
    diff_cmin: Optional[int] = None
    diff_rmax: Optional[int] = None
    diff_cmax: Optional[int] = None
    ok_diff = True
    for inp, out in train_pairs:
        hi, wi = grid_shape_v124(inp)
        ho, wo = grid_shape_v124(out)
        if (int(hi), int(wi)) != (int(ho), int(wo)):
            ok_diff = False
            break
        rmin = cmin = 10**9
        rmax = cmax = -1
        for r in range(int(hi)):
            for c in range(int(wi)):
                vi = int(inp[int(r)][int(c)])
                vo = int(out[int(r)][int(c)])
                if vi == vo:
                    continue
                rmin = min(int(rmin), int(r))
                cmin = min(int(cmin), int(c))
                rmax = max(int(rmax), int(r))
                cmax = max(int(cmax), int(c))
                diff_from[int(vi)] = int(diff_from.get(int(vi), 0)) + 1
                diff_to[int(vo)] = int(diff_to.get(int(vo), 0)) + 1
        if rmax >= 0:
            if diff_rmin is None:
                diff_rmin, diff_cmin, diff_rmax, diff_cmax = int(rmin), int(cmin), int(rmax), int(cmax)
            elif (
                int(rmin) != int(diff_rmin)
                or int(cmin) != int(diff_cmin)
                or int(rmax) != int(diff_rmax)
                or int(cmax) != int(diff_cmax)
            ):
                diff_rmin = diff_cmin = diff_rmax = diff_cmax = None
    def _topk(counts: Dict[int, int], k: int) -> List[int]:
        if not counts or int(k) <= 0:
            return []
        items = sorted(((int(c), int(n)) for c, n in counts.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
        return [int(c) for c, _n in items[: int(k)]]

    diff_from_top = _topk(diff_from, 3) if ok_diff else []
    diff_to_top = _topk(diff_to, 3) if ok_diff else []
    diff_from_top1 = int(diff_from_top[0]) if len(diff_from_top) >= 1 else None
    diff_from_top2 = int(diff_from_top[1]) if len(diff_from_top) >= 2 else None
    diff_from_top3 = int(diff_from_top[2]) if len(diff_from_top) >= 3 else None
    diff_to_top1 = int(diff_to_top[0]) if len(diff_to_top) >= 1 else None
    diff_to_top2 = int(diff_to_top[1]) if len(diff_to_top) >= 2 else None
    diff_to_top3 = int(diff_to_top[2]) if len(diff_to_top) >= 3 else None

    offset_dy: Optional[int] = None
    offset_dx: Optional[int] = None
    if diff_from_top1 is not None and diff_to_top1 is not None:
        dy0: Optional[int] = None
        dx0: Optional[int] = None
        ok = True
        for inp, out in train_pairs:
            bi = _bbox_of_color(inp, color=int(diff_from_top1))
            bo = _bbox_of_color(out, color=int(diff_to_top1))
            if bi is None or bo is None:
                ok = False
                break
            dy = int(bo[0] - bi[0])
            dx = int(bo[1] - bi[1])
            if dy0 is None:
                dy0 = int(dy)
                dx0 = int(dx)
            elif int(dy) != int(dy0) or int(dx) != int(dx0):
                ok = False
                break
        if ok and dy0 is not None and dx0 is not None:
            offset_dy = int(dy0)
            offset_dx = int(dx0)

    def _unique_colors(g: List[List[int]]) -> List[int]:
        cols: Set[int] = set()
        for row in g:
            for x in row:
                cols.add(int(x))
        return sorted(int(x) for x in cols)

    def _only_nonbg_color(grids: Sequence[List[List[int]]]) -> Optional[int]:
        c0: Optional[int] = None
        for g in grids:
            cols = [int(x) for x in _unique_colors(g) if int(x) != int(bg)]
            cols = sorted(set(int(x) for x in cols))
            if len(cols) != 1:
                return None
            v = int(cols[0])
            if c0 is None:
                c0 = int(v)
            elif int(v) != int(c0):
                return None
        return int(c0) if c0 is not None else None

    def _top1_nonbg(grids: Sequence[List[List[int]]]) -> Optional[int]:
        counts2: Dict[int, int] = {}
        for g in grids:
            for row in g:
                for x in row:
                    v = int(x)
                    if int(v) == int(bg):
                        continue
                    counts2[int(v)] = int(counts2.get(int(v), 0)) + 1
        if not counts2:
            return None
        items = sorted(((int(c), int(n)) for c, n in counts2.items()), key=lambda cn: (-int(cn[1]), int(cn[0])))
        return int(items[0][0]) if items else None

    train_in = [p[0] for p in train_pairs]
    train_out = [p[1] for p in train_pairs]
    in_only_nonbg = _only_nonbg_color(train_in + [test_in])
    out_only_nonbg = _only_nonbg_color(train_out)
    in_top_nonbg = _top1_nonbg(train_in + [test_in])
    out_top_nonbg = _top1_nonbg(train_out)

    return _BinderVals(
        bg=int(bg),
        in_height=int(in_h),
        in_width=int(in_w),
        out_height=int(out_h) if out_h is not None else None,
        out_width=int(out_w) if out_w is not None else None,
        diff_from_top1=int(diff_from_top1) if diff_from_top1 is not None else None,
        diff_from_top2=int(diff_from_top2) if diff_from_top2 is not None else None,
        diff_from_top3=int(diff_from_top3) if diff_from_top3 is not None else None,
        diff_to_top1=int(diff_to_top1) if diff_to_top1 is not None else None,
        diff_to_top2=int(diff_to_top2) if diff_to_top2 is not None else None,
        diff_to_top3=int(diff_to_top3) if diff_to_top3 is not None else None,
        diff_rmin=int(diff_rmin) if diff_rmin is not None else None,
        diff_cmin=int(diff_cmin) if diff_cmin is not None else None,
        diff_rmax=int(diff_rmax) if diff_rmax is not None else None,
        diff_cmax=int(diff_cmax) if diff_cmax is not None else None,
        offset_dy=int(offset_dy) if offset_dy is not None else None,
        offset_dx=int(offset_dx) if offset_dx is not None else None,
        in_only_nonbg=int(in_only_nonbg) if in_only_nonbg is not None else None,
        out_only_nonbg=int(out_only_nonbg) if out_only_nonbg is not None else None,
        in_top_nonbg=int(in_top_nonbg) if in_top_nonbg is not None else None,
        out_top_nonbg=int(out_top_nonbg) if out_top_nonbg is not None else None,
    )


def _maybe_bind_arg(*, op_id: str, key: str, value: Any, b: _BinderVals) -> Any:
    """
    Replace constants by binder expressions when they match deterministic task-derived values.

    This is not per-task hacking: binders are generic semantic slots derived from demonstrations
    and resolved at call time inside the solver.
    """
    op = str(op_id)
    k = str(key)

    # Special-case structured args that should be inferred deterministically, not embedded as constants.
    if op == "map_colors" and k == "mapping" and isinstance(value, dict):
        return {"infer": "color_map_diff"}

    if isinstance(value, str):
        if op == "select_obj" and k == "key" and str(value) in {
            "area",
            "width",
            "height",
            "bbox_area",
            "top",
            "left",
            "bottom",
            "right",
            "color",
            "dist_center",
        }:
            return {"bind": "select_obj_key"}
        if op == "select_obj" and k == "order" and str(value) in {"max", "min"}:
            return {"bind": "select_obj_order"}
        return value

    if not isinstance(value, int):
        return value
    v = int(value)

    # Op/arg-aware bindings (generic).
    if op == "mask_by_color" and k == "color":
        if b.diff_from_top1 is not None and v == int(b.diff_from_top1):
            return {"bind": "diff_from_top1"}
        if b.diff_from_top2 is not None and v == int(b.diff_from_top2):
            return {"bind": "diff_from_top2"}
        if b.diff_from_top3 is not None and v == int(b.diff_from_top3):
            return {"bind": "diff_from_top3"}
    if op == "select_obj" and k == "rank":
        if v in {0, 1, 2}:
            return {"bind": "select_obj_rank"}
    if op in {"paint_mask", "paint_rect", "draw_rect_border"} and k == "color":
        if b.diff_to_top1 is not None and v == int(b.diff_to_top1):
            return {"bind": "diff_to_top1"}
        if b.diff_to_top2 is not None and v == int(b.diff_to_top2):
            return {"bind": "diff_to_top2"}
        if b.diff_to_top3 is not None and v == int(b.diff_to_top3):
            return {"bind": "diff_to_top3"}
    if op == "replace_color" and k in {"from", "from_color"}:
        if b.diff_from_top1 is not None and v == int(b.diff_from_top1):
            return {"bind": "diff_from_top1"}
        if b.diff_from_top2 is not None and v == int(b.diff_from_top2):
            return {"bind": "diff_from_top2"}
        if b.diff_from_top3 is not None and v == int(b.diff_from_top3):
            return {"bind": "diff_from_top3"}
    if op == "replace_color" and k in {"to", "to_color"}:
        if b.diff_to_top1 is not None and v == int(b.diff_to_top1):
            return {"bind": "diff_to_top1"}
        if b.diff_to_top2 is not None and v == int(b.diff_to_top2):
            return {"bind": "diff_to_top2"}
        if b.diff_to_top3 is not None and v == int(b.diff_to_top3):
            return {"bind": "diff_to_top3"}
    if op == "fill_enclosed_region" and k in {"fill", "fill_color"}:
        if b.diff_to_top1 is not None and v == int(b.diff_to_top1):
            return {"bind": "diff_to_top1"}
        if b.diff_to_top2 is not None and v == int(b.diff_to_top2):
            return {"bind": "diff_to_top2"}
        if b.diff_to_top3 is not None and v == int(b.diff_to_top3):
            return {"bind": "diff_to_top3"}
    if op in {"new_canvas", "pad_to"} and k in {"height", "width"}:
        if k == "height" and b.out_height is not None and v == int(b.out_height):
            return {"bind": "out_height"}
        if k == "width" and b.out_width is not None and v == int(b.out_width):
            return {"bind": "out_width"}
    if op in {"pad_to", "new_canvas"} and k in {"pad", "color"} and v == int(b.bg):
        return {"bind": "bg"}
    if op == "paste" and k == "transparent" and v == int(b.bg):
        return {"bind": "bg"}
    if op == "paste" and k == "top":
        if b.diff_rmin is not None and v == int(b.diff_rmin):
            return {"bind": "diff_rmin"}
        if b.diff_rmax is not None and v == int(b.diff_rmax):
            return {"bind": "diff_rmax"}
    if op == "paste" and k == "left":
        if b.diff_cmin is not None and v == int(b.diff_cmin):
            return {"bind": "diff_cmin"}
        if b.diff_cmax is not None and v == int(b.diff_cmax):
            return {"bind": "diff_cmax"}
    if op == "patch_translate" and k in {"dy", "dx"}:
        if k == "dy" and b.offset_dy is not None and v == int(b.offset_dy):
            return {"bind": "offset_dy_from_to_bbox"}
        if k == "dx" and b.offset_dx is not None and v == int(b.offset_dx):
            return {"bind": "offset_dx_from_to_bbox"}

    # Typed-by-key fallback (generic, deterministic). Do not bind across semantic types.
    color_like_keys = {
        "color",
        "bg",
        "pad",
        "transparent",
        "from_color",
        "to_color",
        "from",
        "to",
        "seed_color",
        "paint_color",
        "fill_color",
        "fill",
    }
    if ("color" in k) or (k in color_like_keys):
        if b.diff_from_top1 is not None and v == int(b.diff_from_top1):
            return {"bind": "diff_from_top1"}
        if b.diff_from_top2 is not None and v == int(b.diff_from_top2):
            return {"bind": "diff_from_top2"}
        if b.diff_from_top3 is not None and v == int(b.diff_from_top3):
            return {"bind": "diff_from_top3"}
        if b.diff_to_top1 is not None and v == int(b.diff_to_top1):
            return {"bind": "diff_to_top1"}
        if b.diff_to_top2 is not None and v == int(b.diff_to_top2):
            return {"bind": "diff_to_top2"}
        if b.diff_to_top3 is not None and v == int(b.diff_to_top3):
            return {"bind": "diff_to_top3"}
        if b.in_only_nonbg is not None and v == int(b.in_only_nonbg):
            return {"bind": "in_only_nonbg"}
        if b.out_only_nonbg is not None and v == int(b.out_only_nonbg):
            return {"bind": "out_only_nonbg"}
        if b.in_top_nonbg is not None and v == int(b.in_top_nonbg):
            return {"bind": "in_top_nonbg"}
        if b.out_top_nonbg is not None and v == int(b.out_top_nonbg):
            return {"bind": "out_top_nonbg"}
        if v == int(b.bg):
            return {"bind": "bg"}

    if k == "height":
        if b.out_height is not None and v == int(b.out_height):
            return {"bind": "out_height"}
    if k == "width":
        if b.out_width is not None and v == int(b.out_width):
            return {"bind": "out_width"}

    if k in {"top", "rmin", "bottom", "rmax"}:
        if b.diff_rmin is not None and v == int(b.diff_rmin):
            return {"bind": "diff_rmin"}
        if b.diff_rmax is not None and v == int(b.diff_rmax):
            return {"bind": "diff_rmax"}
    if k in {"left", "cmin", "right", "cmax"}:
        if b.diff_cmin is not None and v == int(b.diff_cmin):
            return {"bind": "diff_cmin"}
        if b.diff_cmax is not None and v == int(b.diff_cmax):
            return {"bind": "diff_cmax"}

    if k == "dy":
        if b.offset_dy is not None and v == int(b.offset_dy):
            return {"bind": "offset_dy_from_to_bbox"}
    if k == "dx":
        if b.offset_dx is not None and v == int(b.offset_dx):
            return {"bind": "offset_dx_from_to_bbox"}

    # If we couldn't bind a small translation/placement scalar, mark it as an inferable slot.
    # This keeps concept_call atomic while allowing generic reuse across tasks/domains where
    # the *role* is stable but the concrete offset/placement differs.
    if k in {"dy", "dx", "top", "left"}:
        vv: Optional[int] = None
        if isinstance(v, int):
            vv = int(v)
        elif isinstance(v, str) and str(v).lstrip("-").isdigit():
            vv = int(str(v))
        if vv is not None:
            if k == "dy" and abs(int(vv)) <= 2:
                return {"infer": "dy_small"}
            if k == "dx" and abs(int(vv)) <= 2:
                return {"infer": "dx_small"}
            if k == "top" and abs(int(vv)) <= 2:
                return {"infer": "top_small"}
            if k == "left" and abs(int(vv)) <= 2:
                return {"infer": "left_small"}

    # If the argument is a color-like scalar and we couldn't bind it to a known deterministic slot,
    # mark it as an inferable semantic parameter. This enables ConceptCSG patterns to generalize
    # across tasks where the *role* is stable but the concrete color differs.
    if ("color" in k) or (k in color_like_keys):
        if op in {"mask_by_color", "bbox_by_color", "mask_and_color"} and k == "color":
            return {"infer": "color_in"}
        if op in {"paint_mask", "paint_rect", "draw_rect_border"} and k == "color":
            return {"infer": "color_out"}
        if op == "fill_enclosed_region" and k in {"fill", "fill_color"}:
            return {"infer": "color_out"}
        if op == "replace_color" and k in {"from", "from_color"}:
            return {"infer": "color_in"}
        if op == "replace_color" and k in {"to", "to_color"}:
            return {"infer": "color_out"}

    return v


def _estimate_cost_bits(steps: Sequence[Dict[str, Any]], *, support: int) -> int:
    base = 0
    for st in steps:
        op = str(st.get("op_id") or "")
        od = OP_DEFS_V141.get(op)
        base += int(getattr(od, "base_cost_bits", 1) or 1) if od is not None else 3
    # Compression: concept_call carries only its template cost, not the inner steps.
    base = int(round(float(base) * 0.45)) + 6
    bonus = 0
    if int(support) > 0:
        bonus = min(8, int((support.bit_length() - 1)))
    return max(1, int(base - bonus))


def _min_support_for_len(*, base_min_support: int, length: int, slack_every: int) -> int:
    """
    Longer CSGs can justify lower support under MDL.
    slack_every=0 disables length-based slack.
    """
    if int(slack_every) <= 0:
        return int(base_min_support)
    # Every `slack_every` extra steps beyond 5 reduces min_support by 1 down to 1.
    extra = max(0, int(length) - 5)
    slack = int(extra // int(slack_every))
    return max(1, int(base_min_support) - int(slack))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=12)
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument(
        "--support_slack_every",
        type=int,
        default=2,
        help="Every N steps above len=5 reduces min_support by 1 (down to 1). 0 disables.",
    )
    ap.add_argument("--max_templates", type=int, default=512)
    ap.add_argument("--max_loss_shape", type=int, default=0)
    ap.add_argument("--max_loss_cells", type=int, default=80)
    ap.add_argument("--prefix_only", action="store_true", help="Mine only prefixes (early-collapse bias).")
    ap.add_argument(
        "--require_last_writes_grid",
        action="store_true",
        help="Only emit patterns whose last op writes grid (default: on).",
    )
    ap.set_defaults(require_last_writes_grid=True)
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir))
    out_path = Path(str(args.out))
    _ensure_absent(out_path)

    tasks_path = run_dir / "input" / "arc_tasks_canonical_v141.jsonl"
    traces_path = run_dir / "trace_candidates.jsonl"
    if not tasks_path.exists():
        raise SystemExit(f"missing:{tasks_path}")
    if not traces_path.exists():
        raise SystemExit(f"missing:{traces_path}")

    task_map: Dict[str, Tuple[List[Tuple[List[List[int]], List[List[int]]]], List[List[int]]]] = {}
    for row in _iter_jsonl(tasks_path):
        tid = str(row.get("task_id") or "")
        tps = row.get("train_pairs")
        qps = row.get("test_pairs")
        if not tid or not isinstance(tps, list) or not tps or not isinstance(qps, list) or not qps:
            continue
        train_pairs: List[Tuple[List[List[int]], List[List[int]]]] = []
        for tp in tps:
            if not isinstance(tp, dict):
                continue
            inp = tp.get("in_grid")
            out = tp.get("out_grid")
            if isinstance(inp, list) and isinstance(out, list):
                train_pairs.append((inp, out))
        # For binder computation we use the first test input (consistent with v141 solver).
        test0 = qps[0] if isinstance(qps[0], dict) else {}
        test_in = test0.get("in_grid")
        if not train_pairs or not isinstance(test_in, list):
            continue
        task_map[tid] = (train_pairs, test_in)

    # signature -> tasks
    support_by_sig: DefaultDict[str, Set[str]] = defaultdict(set)
    steps_by_sig: Dict[str, List[Dict[str, Any]]] = {}
    op_ids_by_sig: Dict[str, List[str]] = {}

    def _emit_pattern(task_id: str, steps: Sequence[Dict[str, Any]]) -> None:
        op_ids = [str(s.get("op_id") or "") for s in steps if str(s.get("op_id") or "")]
        if len(op_ids) < int(args.min_len) or len(op_ids) > int(args.max_len):
            return
        if not _type_feasible_from_start(op_ids):
            return
        if bool(args.require_last_writes_grid) and not _writes_grid(op_ids[-1]):
            return
        sig_obj = {"op_ids": op_ids, "steps": steps}
        sig = _sha256_hex(canonical_json_dumps(sig_obj))
        support_by_sig[sig].add(str(task_id))
        if sig not in steps_by_sig:
            steps_by_sig[sig] = [dict(s) for s in steps]
            op_ids_by_sig[sig] = list(op_ids)

    for row in _iter_jsonl(traces_path):
        if str(row.get("kind") or "") != "arc_trace_candidate_v141":
            continue
        tid = str(row.get("task_id") or "")
        if not tid or tid not in task_map:
            continue
        loss = row.get("loss") if isinstance(row.get("loss"), dict) else {}
        ls = int(loss.get("shape") or 0)
        lc = int(loss.get("cells") or 0)
        if ls > int(args.max_loss_shape) or lc > int(args.max_loss_cells):
            continue
        raw_steps = row.get("steps")
        if not isinstance(raw_steps, list) or not raw_steps:
            continue
        flat = _flatten_trace_steps([s for s in raw_steps if isinstance(s, dict)])
        if not flat:
            continue

        train_pairs, test_in = task_map[tid]
        binders = _compute_binders(train_pairs=train_pairs, test_in=test_in)

        # Binderize per-step args deterministically.
        flat2: List[Dict[str, Any]] = []
        for st in flat:
            op0 = str(st.get("op_id") or "")
            if not op0 or op0 in {"macro_call", "concept_call"}:
                continue
            a0 = st.get("args") if isinstance(st.get("args"), dict) else {}
            a2: Dict[str, Any] = {}
            for k in sorted(a0.keys()):
                a2[str(k)] = _maybe_bind_arg(op_id=str(op0), key=str(k), value=a0[k], b=binders)
            flat2.append({"op_id": str(op0), "args": dict(a2)})

        n = int(len(flat2))
        if n <= 0:
            continue

        if bool(args.prefix_only):
            for L in range(int(args.min_len), min(int(args.max_len), n) + 1):
                _emit_pattern(tid, flat2[:L])
        else:
            for L in range(int(args.min_len), min(int(args.max_len), n) + 1):
                for i in range(0, n - L + 1):
                    _emit_pattern(tid, flat2[i : i + L])

    rows_out: List[Dict[str, Any]] = []
    for sig, tasks in support_by_sig.items():
        steps = steps_by_sig.get(sig) or []
        op_ids = op_ids_by_sig.get(sig) or []
        if not steps or not op_ids:
            continue
        sup = int(len(tasks))
        min_sup = _min_support_for_len(
            base_min_support=int(args.min_support),
            length=int(len(op_ids)),
            slack_every=int(args.support_slack_every),
        )
        if sup < int(min_sup):
            continue

        cost_bits = _estimate_cost_bits(steps, support=sup)
        cid = "csgb_" + str(sig)[:16]
        rows_out.append(
            {
                "kind": "arc_concept_csg_v149",
                "schema_version": 149,
                "concept_id": str(cid),
                "op_ids": list(op_ids),
                "cost_bits": int(cost_bits),
                "support": int(sup),
                "steps": list(steps),
                "signature": {"sha256": str(sig), "prefix_only": bool(args.prefix_only)},
                "support_task_ids": sorted(list(tasks)),
            }
        )

    rows_out.sort(
        key=lambda r: (
            -int(r.get("support") or 0),
            -len(list(r.get("op_ids") or [])),
            int(r.get("cost_bits") or 10),
            str(r.get("concept_id") or ""),
        )
    )
    rows_out = rows_out[: int(max(0, int(args.max_templates)))]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows_out:
            f.write(_stable_json(row) + "\n")

    print(str(out_path))
    print(_stable_json({"templates_total": len(rows_out)}))


if __name__ == "__main__":
    main()
