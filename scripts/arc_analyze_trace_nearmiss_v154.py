#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_per_task_json(run_dir: Path) -> Iterable[Path]:
    per_task = run_dir / "per_task"
    if not per_task.is_dir():
        raise SystemExit(f"missing_dir:{per_task}")
    yield from sorted(per_task.glob("*.json.json"))


def _flatten_steps(step_rows: List[Dict[str, Any]]) -> List[str]:
    ops: List[str] = []
    for st in step_rows:
        op0 = str(st.get("op_id") or "")
        if not op0:
            continue
        if op0 == "concept_call":
            inner = st.get("args") if isinstance(st.get("args"), dict) else {}
            inner_steps = inner.get("steps") if isinstance(inner.get("steps"), list) else []
            if inner_steps and all(isinstance(x, dict) for x in inner_steps):
                ops.extend(str(x.get("op_id") or "") for x in inner_steps if str(x.get("op_id") or ""))
            else:
                ops.append("concept_call")
            continue
        if op0 == "macro_call":
            ops.append("macro_call")
            continue
        ops.append(op0)
    return ops


def _best_trace_row(trace_programs: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    best: Optional[Tuple[int, int, int, int, str]] = None
    best_row: Optional[Dict[str, Any]] = None
    for row in trace_programs:
        loss = row.get("loss") if isinstance(row.get("loss"), dict) else {}
        key = (
            int(loss.get("shape") or 0),
            int(loss.get("cells") or 0),
            int(row.get("cost_bits") or 0),
            int(row.get("depth") or 0),
            str(row.get("program_sig") or ""),
        )
        if best is None or key < best:
            best = key
            best_row = row
    return best_row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="Path to results/<run>_try1 directory")
    ap.add_argument("--max_cells", type=int, default=40, help="Only include near-misses with loss.cells <= this")
    ap.add_argument("--max_clusters", type=int, default=20, help="How many clusters to print")
    ap.add_argument("--max_tasks_per_cluster", type=int, default=5, help="How many task_ids to show per cluster")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"missing_dir:{run_dir}")

    clusters: DefaultDict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_task_rows: List[Dict[str, Any]] = []

    for p in _iter_per_task_json(run_dir):
        task_id = p.name.replace(".json.json", ".json")
        obj = _load_json(p)
        solver_results = obj.get("solver_results") if isinstance(obj, dict) else None
        if not isinstance(solver_results, list) or not solver_results:
            continue
        res0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
        tr = res0.get("trace") if isinstance(res0.get("trace"), dict) else {}
        trace_programs = tr.get("trace_programs") if isinstance(tr.get("trace_programs"), list) else []
        best_row = _best_trace_row([r for r in trace_programs if isinstance(r, dict)])
        if best_row is None:
            continue
        loss = best_row.get("loss") if isinstance(best_row.get("loss"), dict) else {}
        loss_shape = int(loss.get("shape") or 0)
        loss_cells = int(loss.get("cells") or 0)
        if loss_shape != 0:
            continue
        if loss_cells > int(args.max_cells):
            continue
        step_rows = best_row.get("steps") if isinstance(best_row.get("steps"), list) else []
        flat = _flatten_steps([s for s in step_rows if isinstance(s, dict)])
        key = ">".join(flat) if flat else "<empty>"
        row = {
            "task_id": str(task_id),
            "loss_cells": int(loss_cells),
            "depth": int(best_row.get("depth") or 0),
            "cost_bits": int(best_row.get("cost_bits") or 0),
            "pipeline": key,
        }
        per_task_rows.append(row)
        clusters[key].append(row)

    clusters_sorted = sorted(
        clusters.items(),
        key=lambda kv: (
            -len(kv[1]),
            int(sum(int(r.get("loss_cells") or 0) for r in kv[1]) / max(1, len(kv[1]))),
            str(kv[0]),
        ),
    )

    out_clusters: List[Dict[str, Any]] = []
    for pipeline, rows in clusters_sorted[: int(args.max_clusters)]:
        rows_sorted = sorted(rows, key=lambda r: (int(r["loss_cells"]), int(r["depth"]), int(r["cost_bits"]), str(r["task_id"])))
        out_clusters.append(
            {
                "pipeline": str(pipeline),
                "count": int(len(rows_sorted)),
                "avg_loss_cells": float(sum(int(r["loss_cells"]) for r in rows_sorted) / max(1, len(rows_sorted))),
                "tasks": rows_sorted[: int(args.max_tasks_per_cluster)],
            }
        )

    out = {
        "schema_version": 1,
        "kind": "arc_nearmiss_clusters_v154",
        "run_dir": str(run_dir),
        "max_cells": int(args.max_cells),
        "tasks_included": int(len(per_task_rows)),
        "clusters": out_clusters,
    }
    print(_stable_json(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

