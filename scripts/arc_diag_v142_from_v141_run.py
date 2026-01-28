#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _grid_shape(grid: Sequence[Sequence[int]]) -> Tuple[int, int]:
    h = int(len(grid))
    w = int(len(grid[0])) if h else 0
    return (h, w)


def _grid_palette(grid: Sequence[Sequence[int]]) -> Set[int]:
    out: Set[int] = set()
    for row in grid:
        for x in row:
            out.add(int(x))
    return out


def _grid_diff_cells_same_shape(inp: Sequence[Sequence[int]], out: Sequence[Sequence[int]]) -> Optional[int]:
    hi, wi = _grid_shape(inp)
    ho, wo = _grid_shape(out)
    if (hi, wi) != (ho, wo):
        return None
    diff = 0
    for r in range(hi):
        for c in range(wi):
            if int(inp[r][c]) != int(out[r][c]):
                diff += 1
    return int(diff)


def _shape_rel_for_pairs(pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]]) -> str:
    if not pairs:
        return "unknown"
    if all(a == b for a, b in pairs):
        return "same"
    if all(b == (a[1], a[0]) for a, b in pairs):
        return "swap_hw"
    ratios: Set[Tuple[int, int]] = set()
    ok = True
    for (hi, wi), (ho, wo) in pairs:
        if hi <= 0 or wi <= 0 or ho <= 0 or wo <= 0:
            ok = False
            break
        if ho % hi != 0 or wo % wi != 0:
            ok = False
            break
        ratios.add((int(ho // hi), int(wo // wi)))
    if ok and len(ratios) == 1:
        ry, rx = list(ratios)[0]
        return f"scale_integer:{int(ry)}x{int(rx)}"
    return "shape_change_mixed"


def _palette_rel(p_in: Set[int], p_out: Set[int]) -> str:
    if p_out.issubset(p_in):
        if p_in == p_out:
            return "equal"
        return "subset"
    if p_in.issubset(p_out):
        return "superset"
    return "other"


def _delta_density_bin(pairs: Sequence[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]]) -> str:
    diffs: List[float] = []
    for inp, out in pairs:
        d = _grid_diff_cells_same_shape(inp, out)
        if d is None:
            continue
        h, w = _grid_shape(inp)
        diffs.append(float(d) / float(max(1, h * w)))
    if not diffs:
        return "n/a"
    avg = sum(diffs) / float(len(diffs))
    if avg <= 0.10:
        return "sparse<=0.10"
    if avg <= 0.30:
        return "local<=0.30"
    return "dense>0.30"


def _cells_bin(n: Optional[int]) -> str:
    if n is None:
        return "n/a"
    nn = int(n)
    if nn <= 0:
        return "0"
    if nn <= 10:
        return "1..10"
    if nn <= 30:
        return "11..30"
    if nn <= 80:
        return "31..80"
    if nn <= 160:
        return "81..160"
    return ">160"


@dataclass(frozen=True)
class TaskFeatV142:
    task_id: str
    solver_status: str
    failure_kind: str
    n_train: int
    shape_rel: str
    palette_rel: str
    delta_density: str
    # From solver trace_programs (best among recorded items).
    min_loss_shape: Optional[int]
    min_loss_cells: Optional[int]
    min_loss_depth: Optional[int]
    min_loss_cost_bits: Optional[int]
    min_shape0_cells: Optional[int]
    min_shape0_depth: Optional[int]
    # Simple operator usage signal from trace programs.
    ops_in_trace_top: List[Tuple[str, int]]

    def cluster_key(self) -> str:
        return _stable_json(
            {
                "failure_kind": self.failure_kind,
                "shape_rel": self.shape_rel,
                "palette_rel": self.palette_rel,
                "delta_density": self.delta_density,
                "min_loss_shape": self.min_loss_shape,
                "min_shape0_cells_bin": _cells_bin(self.min_shape0_cells),
            }
        )


def _iter_per_task_paths(run_dir: Path) -> List[Path]:
    per_task_dir = run_dir / "per_task"
    if not per_task_dir.is_dir():
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")
    names = sorted([p for p in per_task_dir.iterdir() if p.is_file() and p.name.endswith(".json")])
    return names


def _infer_task_id_from_filename(p: Path) -> str:
    # Handles files like 045e512c.json.json and 045e512c.json
    name = p.name
    if name.endswith(".json.json"):
        return name[: -len(".json.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return name


def _task_feats_for_path(p: Path) -> TaskFeatV142:
    d = _read_json(p)

    task_id = _infer_task_id_from_filename(p).split(".")[0]

    scoring = d.get("scoring") if isinstance(d.get("scoring"), dict) else {}
    solver_status = str(scoring.get("solver_status") or "")
    failure_kind = str(scoring.get("failure_kind") or "")

    task = d.get("task") if isinstance(d.get("task"), dict) else {}
    train_pairs_raw = task.get("train_pairs") or []
    train_pairs: List[Tuple[Sequence[Sequence[int]], Sequence[Sequence[int]]]] = []
    pair_shapes: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    pin: Set[int] = set()
    pout: Set[int] = set()
    for pair in train_pairs_raw:
        if not isinstance(pair, dict):
            continue
        inp = pair.get("in_grid")
        out = pair.get("out_grid")
        if not isinstance(inp, list) or not isinstance(out, list):
            continue
        train_pairs.append((inp, out))
        pair_shapes.append((_grid_shape(inp), _grid_shape(out)))
        pin |= _grid_palette(inp)
        pout |= _grid_palette(out)

    shape_rel = _shape_rel_for_pairs(pair_shapes)
    palette_rel = _palette_rel(pin, pout)
    delta_density = _delta_density_bin(train_pairs)

    # Extract loss stats from solver trace programs (top-N only).
    solver_results = d.get("solver_results") if isinstance(d.get("solver_results"), list) else []
    trace_programs: List[Dict[str, Any]] = []
    if solver_results and isinstance(solver_results[0], dict):
        trace = solver_results[0].get("trace") if isinstance(solver_results[0].get("trace"), dict) else {}
        if isinstance(trace.get("trace_programs"), list):
            trace_programs = [tp for tp in trace["trace_programs"] if isinstance(tp, dict)]

    min_loss_shape: Optional[int] = None
    min_loss_cells: Optional[int] = None
    min_loss_depth: Optional[int] = None
    min_loss_cost_bits: Optional[int] = None
    min_shape0_cells: Optional[int] = None
    min_shape0_depth: Optional[int] = None

    ops_hist: Dict[str, int] = {}

    for tp in trace_programs:
        loss = tp.get("loss") if isinstance(tp.get("loss"), dict) else {}
        sh = int(loss.get("shape", 10**9))
        ce = int(loss.get("cells", 10**9))
        depth = int(tp.get("depth", 0))
        cost_bits = int(tp.get("cost_bits", 0))

        steps = tp.get("steps") if isinstance(tp.get("steps"), list) else []
        for s in steps:
            if not isinstance(s, dict):
                continue
            op_id = str(s.get("op_id") or "")
            if not op_id:
                continue
            ops_hist[op_id] = int(ops_hist.get(op_id, 0)) + 1

        if min_loss_shape is None or (sh, ce, cost_bits) < (min_loss_shape, min_loss_cells or 10**9, min_loss_cost_bits or 10**9):
            min_loss_shape = int(sh)
            min_loss_cells = int(ce)
            min_loss_depth = int(depth)
            min_loss_cost_bits = int(cost_bits)

        if sh == 0:
            if min_shape0_cells is None or (ce, cost_bits) < (min_shape0_cells, min_loss_cost_bits or 10**9):
                min_shape0_cells = int(ce)
                min_shape0_depth = int(depth)

    ops_in_trace_top = sorted(((k, int(ops_hist[k])) for k in ops_hist.keys()), key=lambda kv: (-int(kv[1]), str(kv[0])))[:8]

    return TaskFeatV142(
        task_id=str(task_id),
        solver_status=str(solver_status),
        failure_kind=str(failure_kind),
        n_train=int(len(train_pairs)),
        shape_rel=str(shape_rel),
        palette_rel=str(palette_rel),
        delta_density=str(delta_density),
        min_loss_shape=min_loss_shape,
        min_loss_cells=min_loss_cells,
        min_loss_depth=min_loss_depth,
        min_loss_cost_bits=min_loss_cost_bits,
        min_shape0_cells=min_shape0_cells,
        min_shape0_depth=min_shape0_depth,
        ops_in_trace_top=ops_in_trace_top,
    )


def _summarize(feats: List[TaskFeatV142]) -> Dict[str, Any]:
    total = int(len(feats))
    solved = sum(1 for f in feats if f.solver_status == "SOLVED")
    unknown = sum(1 for f in feats if f.solver_status == "UNKNOWN")
    failed = sum(1 for f in feats if f.solver_status not in {"SOLVED", "UNKNOWN"})
    failure_counts: Dict[str, int] = {}
    for f in feats:
        if f.solver_status in {"SOLVED", "UNKNOWN"}:
            continue
        fk = str(f.failure_kind or "")
        if fk:
            failure_counts[fk] = int(failure_counts.get(fk, 0)) + 1
    return {
        "tasks_total": int(total),
        "solved": int(solved),
        "unknown": int(unknown),
        "failed": int(failed),
        "solve_rate": float(solved) / float(total) if total else 0.0,
        "failure_counts": {k: int(failure_counts[k]) for k in sorted(failure_counts.keys())},
    }


def _cluster(feats: List[TaskFeatV142]) -> List[Dict[str, Any]]:
    clusters: Dict[str, Dict[str, Any]] = {}
    for f in feats:
        if f.solver_status in {"SOLVED", "UNKNOWN"}:
            continue
        key = f.cluster_key()
        c = clusters.get(key)
        if c is None:
            c = {"count": 0, "sig": key, "tasks": []}
            clusters[key] = c
        c["count"] = int(c["count"]) + 1
        if len(c["tasks"]) < 24:
            c["tasks"].append(str(f.task_id))
    out = list(clusters.values())
    out.sort(key=lambda r: (-int(r["count"]), str(r["sig"])))
    return out


def _write_text(out_path: Path, text: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_path", required=True)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir))
    out_path = Path(str(args.out_path))
    out_json = Path(str(args.out_json)) if args.out_json else None

    paths = _iter_per_task_paths(run_dir)
    feats = [_task_feats_for_path(p) for p in paths]
    summary = _summarize(feats)
    clusters = _cluster(feats)

    # Extra: near-miss stats for budget/abstraction failures (shape solved but cell diffs remain).
    near_shape0 = [
        f
        for f in feats
        if f.solver_status == "FAIL"
        and f.failure_kind in {"SEARCH_BUDGET_EXCEEDED", "MISSING_ABSTRACTION"}
        and f.min_loss_shape == 0
    ]
    near_shape0_bins: Dict[str, int] = {}
    for f in near_shape0:
        b = _cells_bin(f.min_shape0_cells)
        near_shape0_bins[b] = int(near_shape0_bins.get(b, 0)) + 1

    md = []
    md.append("# ARC_DIAG_V142_FROM_V141_RUN\n")
    md.append(f"- run_dir: `{run_dir}`\n")
    md.append("\n## Summary\n\n")
    md.append(_stable_json(summary) + "\n")
    md.append("\n## Near-miss (SEARCH_BUDGET_EXCEEDED|MISSING_ABSTRACTION, min_loss_shape==0)\n\n")
    md.append(_stable_json({"count": int(len(near_shape0)), "cells_bins": {k: int(near_shape0_bins[k]) for k in sorted(near_shape0_bins.keys())}}) + "\n")
    md.append("\n## Top Failure Clusters\n\n")
    for c in clusters[:20]:
        md.append(f"- count={int(c['count'])} sig={c['sig']} tasks={c['tasks']}\n")

    _write_text(out_path, "".join(md))

    if out_json is not None:
        out_obj = {
            "schema_version": 142,
            "kind": "arc_diag_v142",
            "run_dir": str(run_dir),
            "summary": summary,
            "clusters": clusters,
            "tasks": [f.__dict__ for f in feats],
        }
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    print(_stable_json({"ok": True, "out_path": str(out_path), "out_json": str(out_json) if out_json else ""}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
