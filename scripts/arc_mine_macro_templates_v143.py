#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple


def _stable_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _iter_per_task_files(per_task_dir: Path) -> Iterable[Path]:
    for p in sorted(per_task_dir.glob("*.json.json")):
        if p.is_file():
            yield p


def _extract_program_op_ids(per_task_obj: Dict[str, object]) -> List[str]:
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return []
    sr0 = solver_results[0]
    if not isinstance(sr0, dict):
        return []
    if str(sr0.get("status") or "") != "SOLVED":
        return []
    steps = sr0.get("program_steps")
    if not isinstance(steps, list):
        return []
    op_ids: List[str] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        op_id = str(st.get("op_id") or "")
        if not op_id:
            continue
        op_ids.append(op_id)
    return op_ids


def _extract_trace_program_op_ids(
    per_task_obj: Dict[str, object],
    *,
    max_programs: int,
    max_loss_shape: int,
    max_loss_cells: int,
) -> List[List[str]]:
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return []
    sr0 = solver_results[0]
    if not isinstance(sr0, dict):
        return []
    trace = sr0.get("trace")
    if not isinstance(trace, dict):
        return []
    tps = trace.get("trace_programs")
    if not isinstance(tps, list) or not tps:
        return []

    out: List[List[str]] = []
    for row in tps[: int(max(0, int(max_programs)))]:
        if not isinstance(row, dict):
            continue
        loss = row.get("loss")
        if not isinstance(loss, dict):
            continue
        ls = int(loss.get("shape") or 0)
        lc = int(loss.get("cells") or 0)
        if ls > int(max_loss_shape) or lc > int(max_loss_cells):
            continue
        steps = row.get("steps")
        if not isinstance(steps, list) or not steps:
            continue
        op_ids: List[str] = []
        for st in steps:
            if not isinstance(st, dict):
                continue
            op_id = str(st.get("op_id") or "")
            if not op_id:
                continue
            if op_id in {"macro_call", "concept_call"}:
                # Avoid recursive macros / nested concepts (templates are sequences of base op_ids).
                op_ids = []
                break
            op_ids.append(op_id)
        if op_ids:
            out.append(op_ids)
    return out


def _all_contiguous_subseqs(op_ids: Sequence[str], *, min_len: int, max_len: int) -> Iterable[Tuple[str, ...]]:
    n = int(len(op_ids))
    if n < min_len:
        return
    for L in range(min_len, min(max_len, n) + 1):
        for i in range(0, n - L + 1):
            yield tuple(op_ids[i : i + L])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=5)
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--max_macros", type=int, default=256)
    ap.add_argument("--include_singletons", action="store_true", help="Include length-1 macros (default: off)")
    ap.add_argument(
        "--from_trace",
        action="store_true",
        help="Mine from per-task trace_programs (top-N programs) instead of only final SOLVED programs.",
    )
    ap.add_argument(
        "--include_examples",
        action="store_true",
        help="Include example task ids per macro (debug-only; default: off to avoid puzzle-specific metadata).",
    )
    ap.add_argument("--trace_max_programs_per_task", type=int, default=12)
    ap.add_argument("--trace_max_loss_shape", type=int, default=0)
    ap.add_argument("--trace_max_loss_cells", type=int, default=120)
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    per_task_dir = run_dir / "per_task"
    if not per_task_dir.is_dir():
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")

    min_len = int(args.min_len)
    max_len = int(args.max_len)
    if bool(args.include_singletons) and min_len > 1:
        min_len = 1
    if min_len < 1 or max_len < min_len:
        raise SystemExit("bad_len_bounds")

    # seq -> set(task_id)
    support: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    solved_tasks = 0
    tasks_seen = 0

    for p in _iter_per_task_files(per_task_dir):
        tasks_seen += 1
        task_id = p.name.split(".json.json")[0]
        obj = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(obj, dict):
            continue
        programs: List[List[str]] = []
        if bool(args.from_trace):
            programs = _extract_trace_program_op_ids(
                obj,
                max_programs=int(args.trace_max_programs_per_task),
                max_loss_shape=int(args.trace_max_loss_shape),
                max_loss_cells=int(args.trace_max_loss_cells),
            )
        else:
            op_ids = _extract_program_op_ids(obj)
            if op_ids:
                programs = [op_ids]

        if not programs:
            continue
        if not bool(args.from_trace):
            solved_tasks += 1

        # Count each subseq at most once per task across all included programs.
        seen_local: Set[Tuple[str, ...]] = set()
        for op_ids in programs:
            for seq in _all_contiguous_subseqs(op_ids, min_len=min_len, max_len=max_len):
                if seq in seen_local:
                    continue
                seen_local.add(seq)
                support[seq].add(task_id)

    rows: List[Dict[str, object]] = []
    for seq, tasks in support.items():
        if len(tasks) < int(args.min_support):
            continue
        op_ids = list(seq)
        macro_id = _sha256_hex(_stable_json({"op_ids": op_ids}))
        row = {
            "kind": "arc_macro_template_v143",
            "schema_version": 143,
            "macro_id": str(macro_id),
            "op_ids": op_ids,
            "support": int(len(tasks)),
        }
        if bool(args.include_examples):
            row["example_task_ids"] = sorted(list(tasks))[:20]
        rows.append(row)

    rows.sort(key=lambda r: (-int(r.get("support") or 0), -len(list(r.get("op_ids") or [])), str(r.get("macro_id") or "")))
    rows = rows[: int(max(0, int(args.max_macros)))]

    out_path = Path(str(args.out)).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(_stable_json(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

    print(
        _stable_json(
            {
                "ok": True,
                "run_dir": str(run_dir),
                "out": str(out_path),
                "tasks_seen": int(tasks_seen),
                "solved_tasks_seen": int(solved_tasks),
                "macros_written": int(len(rows)),
                "min_len": int(min_len),
                "max_len": int(max_len),
                "min_support": int(args.min_support),
                "from_trace": bool(args.from_trace),
                "trace_max_programs_per_task": int(args.trace_max_programs_per_task),
                "trace_max_loss_shape": int(args.trace_max_loss_shape),
                "trace_max_loss_cells": int(args.trace_max_loss_cells),
            }
        )
    )


if __name__ == "__main__":
    main()
