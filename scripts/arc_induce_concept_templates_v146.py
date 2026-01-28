#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.arc_ops_v132 import StateV132
from atos_core.arc_ops_v141 import apply_op_v141
from atos_core.arc_solver_v141 import (
    ProgramStepV141,
    _apply_program_steps_to_grid_v141,
    _count_cell_mismatch_same_shape_v141,
    _infer_bg_candidates_v141,
    _infer_post_fill_bg_by_periodic_tile_reduce_v141,
    _infer_post_fill_enclosed_bg_reduce_v141,
    _infer_post_map_colors_by_orig_reduce_v141,
    _infer_post_overlay_self_translate_reduce_v141,
    _infer_post_overlay_orig_on_bg_reduce_v141,
    _infer_post_overlay_orig_on_color_reduce_v141,
)
from atos_core.grid_v124 import GridV124, grid_from_list_v124, grid_shape_v124, unique_colors_v124


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _iter_per_task_files(run_dir: Path) -> Iterable[Path]:
    per_task_dir = run_dir / "per_task"
    if not per_task_dir.is_dir():
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")
    for p in sorted(per_task_dir.glob("*.json.json")):
        if p.is_file():
            yield p


def _as_grid_v124(x: Any) -> GridV124:
    if not isinstance(x, list):
        raise ValueError("bad_grid")
    return grid_from_list_v124(x)


def _extract_train_pairs_v124(obj: Dict[str, Any]) -> List[Tuple[GridV124, GridV124]]:
    task = obj.get("task") if isinstance(obj.get("task"), dict) else {}
    pairs = task.get("train_pairs")
    if not isinstance(pairs, list):
        return []
    out: List[Tuple[GridV124, GridV124]] = []
    for row in pairs:
        if not isinstance(row, dict):
            continue
        ig = row.get("in_grid")
        og = row.get("out_grid")
        if not isinstance(ig, list) or not isinstance(og, list):
            continue
        out.append((_as_grid_v124(ig), _as_grid_v124(og)))
    return out


def _extract_trace_programs(obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    solver_results = obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return []
    sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
    trace = sr0.get("trace") if isinstance(sr0.get("trace"), dict) else {}
    tps = trace.get("trace_programs")
    if not isinstance(tps, list):
        return []
    return [r for r in tps if isinstance(r, dict)]


def _extract_concept_template_from_failure(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Fast-path: if the solver already emitted an explicit concept template row for a failure cone,
    use it directly instead of re-mining from trace programs.

    This makes concept induction an explicit operator in the loop (failure -> concept object -> bank),
    while remaining fully deterministic and audit-friendly.
    """
    solver_results = obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return None
    sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
    fr = sr0.get("failure_reason") if isinstance(sr0.get("failure_reason"), dict) else {}
    details = fr.get("details") if isinstance(fr.get("details"), dict) else {}
    tmpl = details.get("concept_template") if isinstance(details.get("concept_template"), dict) else None
    if tmpl is None:
        return None
    sig = tmpl.get("signature") if isinstance(tmpl.get("signature"), dict) else None
    op_ids = tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else None
    if sig is None or op_ids is None or not op_ids:
        return None
    op_ids_s = [str(x) for x in op_ids if str(x)]
    if not op_ids_s:
        return None
    return {
        "kind": "arc_concept_template_v146",
        "schema_version": 146,
        "signature": {str(k): sig.get(k) for k in sorted(sig.keys())},
        "op_ids": list(op_ids_s),
        "cost_bits": int(tmpl.get("cost_bits") or 10),
    }


def _extract_base_steps_from_trace(
    trace_programs: Sequence[Dict[str, Any]],
) -> Optional[Tuple[Tuple[ProgramStepV141, ...], int]]:
    # (cost_bits, depth, loss_cells, program_sig, steps)
    best: Optional[Tuple[int, int, int, str, Tuple[ProgramStepV141, ...]]] = None
    for row in trace_programs:
        loss = row.get("loss")
        if not isinstance(loss, dict):
            continue
        ls = int(loss.get("shape") or 0)
        lc = int(loss.get("cells") or 0)
        if ls != 0 or lc <= 0:
            continue
        steps_raw = row.get("steps")
        if not isinstance(steps_raw, list):
            continue
        steps: List[ProgramStepV141] = []
        ok = True
        for st in steps_raw:
            if not isinstance(st, dict):
                ok = False
                break
            op_id = str(st.get("op_id") or "")
            args = st.get("args") if isinstance(st.get("args"), dict) else {}
            if not op_id:
                ok = False
                break
            steps.append(ProgramStepV141(op_id=op_id, args=dict(args)))
        if not ok:
            continue
        cost_bits = int(row.get("cost_bits") or 0)
        depth = int(row.get("depth") or len(steps))
        sig = str(row.get("program_sig") or "")
        # Failure Cone base_program: pick the cheapest shape-correct (loss_shape==0) program.
        # This matches ACT-vs-ARC pt2: if a cheap program reaches the right shape but can't close
        # without going deep, we treat the residual as missing abstraction (concept), not compute.
        key = (int(cost_bits), int(depth), int(lc), sig)
        rec = (int(cost_bits), int(depth), int(lc), sig, tuple(steps))
        if best is None or key < (best[0], best[1], best[2], best[3]):
            best = rec
    if best is None:
        return None
    base_cells = int(best[2])
    return best[4], int(base_cells)


def _diff_bucket(loss_cells: int) -> str:
    n = int(loss_cells)
    if n == 1:
        return "ONE"
    if n <= 5:
        return "SMALL"
    if n <= 20:
        return "MEDIUM"
    return "LARGE"


def _palette_relation(want: Sequence[int], got: Sequence[int]) -> str:
    a = set(int(x) for x in want)
    b = set(int(x) for x in got)
    if a == b:
        return "EQUAL"
    if a.issubset(b):
        return "SUBSET"
    if b.issubset(a):
        return "SUPERSET"
    # For v1 keep the enum closed: map mixed to EQUAL (safe because inference is fail-closed).
    return "EQUAL"


def _infer_palette_relation_for_base_program(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    base_steps: Tuple[ProgramStepV141, ...],
) -> str:
    apply_cache: Dict[Tuple[Any, ...], Any] = {}
    grid_hash_cache: Dict[GridV124, str] = {}
    metrics: Dict[str, int] = {}

    rel: Optional[str] = None
    for inp, want in train_pairs:
        got = _apply_program_steps_to_grid_v141(
            inp=inp,
            steps=base_steps,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )
        got_cols = sorted(int(c) for c in unique_colors_v124(got))
        want_cols = sorted(int(c) for c in unique_colors_v124(want))
        r = _palette_relation(want_cols, got_cols)
        if rel is None:
            rel = r
        elif rel != r:
            # Keep enum closed and deterministic.
            rel = "EQUAL"
            break
    return str(rel or "EQUAL")


def _concept_id_for(*, signature: Dict[str, Any], op_ids: Sequence[str]) -> str:
    body = {
        "schema_version": 146,
        "kind": "concept_key_v146",
        "signature": {str(k): signature.get(k) for k in sorted(signature.keys())},
        "op_ids": [str(x) for x in op_ids if str(x)],
    }
    return "csg_" + sha256_hex(canonical_json_dumps(body).encode("utf-8"))[:16]


def _min_required_closure_improvement_total(before_total: int) -> int:
    b = int(before_total)
    if b <= 0:
        return 0
    return max(1, int((b + 7) // 8))


def _try_induce_op_for_failure(
    *,
    train_pairs: Sequence[Tuple[GridV124, GridV124]],
    base_steps: Tuple[ProgramStepV141, ...],
) -> List[Tuple[str, str]]:
    """
    Returns [(diff_kind, op_id), ...] for deterministic 1-step closures that can be inferred for base_steps.

    The induced concepts use args=INFER at runtime; we only check that such args exist for this task.

    Important: we return *all* viable closures (small, fixed candidate set), not only the best one.
    This matches ACT-vs-ARC pt2: failure cones should produce a set of reusable concept candidates,
    allowing cross-task support to accumulate naturally (e.g. COPY_FROM_ORIG and COLOR_REMAP can both
    be valid closure families, but apply to different tasks).
    """
    apply_cache: Dict[Tuple[Any, ...], Any] = {}
    grid_hash_cache: Dict[GridV124, str] = {}
    metrics: Dict[str, int] = {}

    bg_candidates = _infer_bg_candidates_v141(grids=[p[0] for p in train_pairs] + [p[1] for p in train_pairs])

    # Baseline residual (Failure Cone): total mismatch of base_steps on train pairs.
    before_total = 0
    for inp0, want0 in train_pairs:
        got0 = _apply_program_steps_to_grid_v141(
            inp=inp0,
            steps=base_steps,
            apply_cache=apply_cache,
            grid_hash_cache=grid_hash_cache,
            metrics=metrics,
        )
        before0 = _count_cell_mismatch_same_shape_v141(got=got0, want=want0)
        if before0 is None:
            return None
        before_total += int(before0)
    if before_total <= 0:
        return []

    def eval_after(op_id: str, args: Dict[str, Any]) -> Optional[int]:
        total = 0
        for inp, want in train_pairs:
            got = _apply_program_steps_to_grid_v141(
                inp=inp,
                steps=base_steps,
                apply_cache=apply_cache,
                grid_hash_cache=grid_hash_cache,
                metrics=metrics,
            )
            before = _count_cell_mismatch_same_shape_v141(got=got, want=want)
            if before is None:
                return None
            try:
                st2 = apply_op_v141(state=StateV132(grid=got, orig=inp), op_id=str(op_id), args=dict(args))
            except Exception:
                return None
            after = _count_cell_mismatch_same_shape_v141(got=st2.grid, want=want)
            if after is None:
                return None
            total += int(after)
        return int(total)

    # Evaluate all viable 1-step closures and pick the best (min total mismatch after).
    # This avoids getting stuck on a weak closure when a stronger one exists.
    cand: List[Tuple[int, str, str]] = []  # (after_total, diff_kind, op_id)

    bg = _infer_post_overlay_orig_on_bg_reduce_v141(
        train_pairs=train_pairs,
        steps=base_steps,
        bg_candidates=bg_candidates,
        apply_cache=apply_cache,
        grid_hash_cache=grid_hash_cache,
        metrics=metrics,
    )
    if bg is not None:
        a = eval_after("overlay_orig_on_bg", {"bg": int(bg)})
        if a is not None:
            cand.append((int(a), "COPY_FROM_ORIG", "overlay_orig_on_bg"))

    cc = _infer_post_overlay_orig_on_color_reduce_v141(
        train_pairs=train_pairs,
        steps=base_steps,
        apply_cache=apply_cache,
        grid_hash_cache=grid_hash_cache,
        metrics=metrics,
    )
    if cc is not None:
        a = eval_after("overlay_orig_on_color", {"color": int(cc)})
        if a is not None:
            cand.append((int(a), "COPY_FROM_ORIG", "overlay_orig_on_color"))

    rows = _infer_post_map_colors_by_orig_reduce_v141(
        train_pairs=train_pairs,
        steps=base_steps,
        apply_cache=apply_cache,
        grid_hash_cache=grid_hash_cache,
        metrics=metrics,
    )
    if rows is not None:
        a = eval_after("map_colors_by_orig", {"mapping": list(rows)})
        if a is not None:
            cand.append((int(a), "COLOR_REMAP", "map_colors_by_orig"))

    bg2 = _infer_post_fill_enclosed_bg_reduce_v141(
        train_pairs=train_pairs,
        steps=base_steps,
        bg_candidates=bg_candidates,
        apply_cache=apply_cache,
        grid_hash_cache=grid_hash_cache,
        metrics=metrics,
    )
    if bg2 is not None:
        args2: Dict[str, Any] = {}
        if isinstance(bg2, dict):
            if "bg" in bg2:
                args2["bg"] = int(bg2.get("bg") or 0)
            if "fill_color" in bg2:
                args2["fill_color"] = int(bg2.get("fill_color") or 0)
        else:
            args2["bg"] = int(bg2)
        a = eval_after("fill_enclosed_bg", dict(args2))
        if a is not None:
            cand.append((int(a), "REGION_FILL", "fill_enclosed_bg"))

    bg3 = _infer_post_fill_bg_by_periodic_tile_reduce_v141(
        train_pairs=train_pairs,
        steps=base_steps,
        bg_candidates=bg_candidates,
        apply_cache=apply_cache,
        grid_hash_cache=grid_hash_cache,
        metrics=metrics,
    )
    if bg3 is not None:
        a = eval_after("fill_bg_by_periodic_tile", {"bg": int(bg3)})
        if a is not None:
            cand.append((int(a), "REGION_FILL", "fill_bg_by_periodic_tile"))

    st_args = _infer_post_overlay_self_translate_reduce_v141(
        train_pairs=train_pairs,
        steps=base_steps,
        bg_candidates=bg_candidates,
        apply_cache=apply_cache,
        grid_hash_cache=grid_hash_cache,
        metrics=metrics,
    )
    if st_args is not None:
        a = eval_after("overlay_self_translate", dict(st_args))
        if a is not None:
            cand.append((int(a), "OFFSET_COPY", "overlay_self_translate"))

    if not cand:
        return []

    dk_rank = {
        "COPY_FROM_ORIG": 0,
        "COLOR_REMAP": 1,
        "OFFSET_COPY": 2,
        "REGION_FILL": 3,
    }
    cand.sort(key=lambda t: (int(t[0]), int(dk_rank.get(str(t[1]), 99)), str(t[1]), str(t[2])))

    out: List[Tuple[str, str]] = []
    for after_total, diff_kind, op_id in cand:
        # Under pt2 "abstraction pressure", any deterministic reduction is enough to
        # propose a concept template; false positives are prevented by per-task
        # fail-closed instantiation inside the solver.
        if int(after_total) >= int(before_total):
            continue
        out.append((str(diff_kind), str(op_id)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--induction_log",
        default="",
        help="Optional WORM jsonl log of per-task induced concept births (for Ω tracking).",
    )
    ap.add_argument("--min_support", type=int, default=1)
    ap.add_argument("--max_concepts", type=int, default=128)
    ap.add_argument("--only_failure_kind", default="", help="Optional filter for scoring.failure_kind")
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    out_path = Path(str(args.out)).resolve()
    _ensure_absent(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    induction_log_path: Optional[Path] = None
    if str(args.induction_log).strip():
        induction_log_path = Path(str(args.induction_log)).resolve()
        _ensure_absent(induction_log_path)
        induction_log_path.parent.mkdir(parents=True, exist_ok=True)

    only_fk = str(args.only_failure_kind or "").strip()

    support: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    payload: Dict[Tuple[str, str], Tuple[Dict[str, Any], List[str], int]] = {}
    induction_rows: List[Dict[str, Any]] = []

    tasks_seen = 0
    failures_seen = 0
    induced_seen = 0

    for p in _iter_per_task_files(run_dir):
        tasks_seen += 1
        obj = _read_json(p)
        if not isinstance(obj, dict):
            continue
        task_obj = obj.get("task") if isinstance(obj.get("task"), dict) else {}
        task_id = str(task_obj.get("task_id") or "")
        scoring = obj.get("scoring") if isinstance(obj.get("scoring"), dict) else {}
        fk = str(scoring.get("failure_kind") or "")
        if only_fk and fk != only_fk:
            continue
        # Ω law: no recurrent failure type can live outside the failure→concept cycle.
        # Include MISSING_OPERATOR here to attempt closure mining from near-miss traces
        # (still fail-closed; promotion/Ω will reject placebo concepts).
        if fk not in {
            "MISSING_ABSTRACTION",
            "MISSING_CONCEPT",
            "SEARCH_BUDGET_EXCEEDED",
            "CONCEPT_DEPTH_TOO_SHALLOW",
            "MISSING_OPERATOR",
        }:
            continue

        # Fast path: when the solver already emitted a concept template row, use it directly.
        tmpl = _extract_concept_template_from_failure(obj)
        if tmpl is not None:
            sig = tmpl.get("signature") if isinstance(tmpl.get("signature"), dict) else None
            op_ids = tmpl.get("op_ids") if isinstance(tmpl.get("op_ids"), list) else None
            if sig is not None and op_ids is not None and op_ids:
                key = (_stable_json(sig), _stable_json(op_ids))
                support[key] += 1
                op_ids_s = [str(x) for x in op_ids if str(x)]
                cost_bits = int(tmpl.get("cost_bits") or 10)
                payload[key] = (dict(sig), list(op_ids_s), int(cost_bits))
                if induction_log_path is not None and task_id and op_ids_s:
                    cid = _concept_id_for(signature=dict(sig), op_ids=op_ids_s)
                    induction_rows.append(
                        {
                            "schema_version": 146,
                            "kind": "arc_concept_induction_v146",
                            "source": "solver_emitted",
                            "rank": 0,
                            "task_id": str(task_id),
                            "concept_id": str(cid),
                            "signature": dict(sig),
                            "op_ids": list(op_ids_s),
                            "cost_bits": int(cost_bits),
                        }
                    )
            failures_seen += 1
            induced_seen += 1
            continue

        train_pairs = _extract_train_pairs_v124(obj)
        if not train_pairs:
            continue
        trace_programs = _extract_trace_programs(obj)
        if not trace_programs:
            continue

        base = _extract_base_steps_from_trace(trace_programs)
        if base is None:
            continue
        base_steps, base_cells = base

        failures_seen += 1
        induced_ops = _try_induce_op_for_failure(train_pairs=train_pairs, base_steps=base_steps)
        if not induced_ops:
            continue
        palette_rel = _infer_palette_relation_for_base_program(train_pairs=train_pairs, base_steps=base_steps)
        for rank_i, (diff_kind, op_id) in enumerate(induced_ops):
            req_depth = 1
            if str(diff_kind) == "MULTI_STAGE":
                db = _diff_bucket(int(base_cells))
                req_depth = 3 if str(db) == "LARGE" else 2
            sig = {
                "shape_ok": True,
                "diff_kind": str(diff_kind),
                "diff_bucket": _diff_bucket(int(base_cells)),
                "palette_relation": str(palette_rel),
                "required_depth": int(req_depth),
            }
            op_ids = [str(op_id)]
            key = (_stable_json(sig), _stable_json(op_ids))
            support[key] += 1
            payload[key] = (dict(sig), list(op_ids), 10)
            if induction_log_path is not None and task_id:
                cid = _concept_id_for(signature=dict(sig), op_ids=op_ids)
                induction_rows.append(
                    {
                        "schema_version": 146,
                        "kind": "arc_concept_induction_v146",
                        "source": "mined",
                        "rank": int(rank_i),
                        "task_id": str(task_id),
                        "concept_id": str(cid),
                        "signature": dict(sig),
                        "op_ids": list(op_ids),
                        "cost_bits": 10,
                    }
                )
            induced_seen += 1

    rows: List[Dict[str, Any]] = []
    for key, n in support.items():
        if int(n) < int(args.min_support):
            continue
        sig, op_ids, cost_bits = payload.get(key) or ({}, [], 10)
        if not sig or not op_ids:
            continue
        if len(op_ids) > 3:
            continue
        cid = _concept_id_for(signature=sig, op_ids=op_ids)
        rows.append(
            {
                "kind": "arc_concept_template_v146",
                "schema_version": 146,
                "concept_id": str(cid),
                "signature": dict(sig),
                "op_ids": list(op_ids),
                "support": int(n),
                # Keep concept call cost low by design; it represents amortized reuse.
                "cost_bits": int(cost_bits),
            }
        )

    rows.sort(
        key=lambda r: (
            -int(r.get("support") or 0),
            str(r.get("concept_id") or ""),
            _stable_json(r),
        )
    )
    rows = rows[: int(max(0, int(args.max_concepts)))]

    out_path.write_text("\n".join(_stable_json(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

    if induction_log_path is not None:
        induction_rows.sort(
            key=lambda r: (
                str(r.get("task_id") or ""),
                int(r.get("rank") or 0),
                str(r.get("concept_id") or ""),
                _stable_json(r),
            )
        )
        induction_log_path.write_text(
            "\n".join(_stable_json(r) for r in induction_rows) + ("\n" if induction_rows else ""),
            encoding="utf-8",
        )
    print(
        _stable_json(
            {
                "ok": True,
                "run_dir": str(run_dir),
                "out": str(out_path),
                "induction_log": str(induction_log_path) if induction_log_path is not None else "",
                "tasks_seen": int(tasks_seen),
                "failures_seen": int(failures_seen),
                "induced_seen": int(induced_seen),
                "concepts_written": int(len(rows)),
                "min_support": int(args.min_support),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
