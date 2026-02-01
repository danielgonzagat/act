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
from atos_core.arc_ops_v141 import step_cost_bits_v141


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


def _floor_log2_int(n: int) -> int:
    x = int(n)
    if x <= 0:
        return 0
    return int(x.bit_length() - 1)


def _diff_bucket(loss_cells: int) -> str:
    n = int(loss_cells)
    if n == 0:
        return "ZERO"
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
    return "EQUAL"


def _unique_colors(grid: Sequence[Sequence[int]]) -> List[int]:
    cols: set[int] = set()
    for row in grid:
        for v in row:
            cols.add(int(v))
    return sorted(list(cols))


def _flatten_steps(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Flatten meta steps (macro_call / concept_call) into base steps.
    Returned steps are dicts with {op_id, args}.
    """
    out: List[Dict[str, Any]] = []
    for st in steps:
        if not isinstance(st, dict):
            continue
        op_id = str(st.get("op_id") or "")
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        if not op_id:
            continue
        if op_id in {"macro_call", "concept_call"}:
            inner = args.get("steps")
            if isinstance(inner, list):
                out.extend(_flatten_steps([x for x in inner if isinstance(x, dict)]))
            continue
        out.append({"op_id": str(op_id), "args": dict(args)})
    return out


def _contiguous_subseqs(op_ids: Sequence[str], *, min_len: int, max_len: int) -> Iterable[Tuple[str, ...]]:
    n = int(len(op_ids))
    if n < int(min_len):
        return
    for L in range(int(min_len), min(int(max_len), n) + 1):
        for i in range(0, n - L + 1):
            yield tuple(op_ids[i : i + L])


def _concept_id_for(*, signature: Dict[str, Any], op_ids: Sequence[str]) -> str:
    body = {
        "schema_version": 146,
        "kind": "concept_key_v146",
        "signature": {str(k): signature.get(k) for k in sorted(signature.keys())},
        "op_ids": [str(x) for x in op_ids if str(x)],
    }
    cid = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    return "csg_" + str(cid)[:16]


def _concept_cost_bits(*, op_ids: Sequence[str], support: int) -> int:
    """
    Deterministic MDL proxy for a learned concept template.

    We intentionally make concept_call cheaper than raw primitives to encourage
    concept-as-policy, while still charging for longer sequences.
    """
    L = int(len(list(op_ids)))
    base = 10 + max(0, L - 1)
    bonus = min(4, _floor_log2_int(max(1, int(support))))
    return max(1, int(base) - int(bonus))


def _extract_task_id(per_task_obj: Dict[str, Any]) -> str:
    task = per_task_obj.get("task") if isinstance(per_task_obj.get("task"), dict) else {}
    return str(task.get("task_id") or "")


def _extract_scoring_failure_kind(per_task_obj: Dict[str, Any]) -> str:
    scoring = per_task_obj.get("scoring") if isinstance(per_task_obj.get("scoring"), dict) else {}
    return str(scoring.get("failure_kind") or "")


def _extract_solver_status(per_task_obj: Dict[str, Any]) -> str:
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return ""
    sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
    return str(sr0.get("status") or "")


def _extract_program_steps(per_task_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return []
    sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
    steps = sr0.get("program_steps")
    if not isinstance(steps, list):
        return []
    return [s for s in steps if isinstance(s, dict)]


def _extract_trace_programs(per_task_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return []
    sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
    trace = sr0.get("trace") if isinstance(sr0.get("trace"), dict) else {}
    tps = trace.get("trace_programs")
    if not isinstance(tps, list):
        return []
    return [r for r in tps if isinstance(r, dict)]


def _signature_for_trace(*, loss_shape: int, loss_cells: int, palette_rel: str) -> Dict[str, Any]:
    return {
        "diff_bucket": _diff_bucket(int(loss_cells)),
        "diff_kind": "TRACE_SHAPE_OK" if int(loss_shape) == 0 else "TRACE_NEAR_MISS",
        "palette_relation": str(palette_rel),
        "required_depth": 1,
        "shape_ok": bool(int(loss_shape) == 0),
    }


def _signature_for_no_concept_call() -> Dict[str, Any]:
    return {
        "diff_bucket": "ANY",
        "diff_kind": "NO_CONCEPT_CALL_WRAP",
        "palette_relation": "EQUAL",
        "required_depth": 1,
        "shape_ok": True,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--induction_log", default="", help="Optional WORM jsonl log of per-task induced concept births.")
    ap.add_argument("--min_support", type=int, default=1)
    ap.add_argument("--max_concepts", type=int, default=128)
    ap.add_argument("--only_failure_kind", default="", help="Optional filter for scoring.failure_kind")
    ap.add_argument("--min_len", type=int, default=1)
    ap.add_argument("--max_len", type=int, default=3)
    ap.add_argument("--trace_max_programs_per_task", type=int, default=40)
    ap.add_argument("--trace_max_loss_shape", type=int, default=0)
    ap.add_argument("--trace_max_loss_cells", type=int, default=120)
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
    min_len = max(1, int(args.min_len))
    max_len = max(min_len, int(args.max_len))

    # key -> support count
    support: DefaultDict[Tuple[str, str], int] = defaultdict(int)
    # key -> payload (signature, op_ids)
    payload: Dict[Tuple[str, str], Tuple[Dict[str, Any], List[str]]] = {}
    # optional log rows
    induction_rows: List[Dict[str, Any]] = []

    tasks_seen = 0
    induced_seen = 0

    for p in _iter_per_task_files(run_dir):
        tasks_seen += 1
        obj = _read_json(p)
        if not isinstance(obj, dict):
            continue

        task_id = _extract_task_id(obj)
        fk = _extract_scoring_failure_kind(obj)
        if only_fk and fk != only_fk:
            continue

        # Primary sources:
        # - NO_CONCEPT_CALL: solver produced a correct output program, but validator rejected it.
        # - Failure cones: mine from trace_programs near-misses.
        candidates: List[Tuple[Dict[str, Any], List[str]]] = []

        if fk == "NO_CONCEPT_CALL":
            steps_raw = _extract_program_steps(obj)
            flat = _flatten_steps(steps_raw)
            op_ids = [str(s.get("op_id") or "") for s in flat if str(s.get("op_id") or "")]
            if op_ids:
                candidates.append((_signature_for_no_concept_call(), op_ids))

        trace_programs = _extract_trace_programs(obj)
        if trace_programs:
            # Deterministic: keep the given order (already sorted by solver) and cap.
            for tp in trace_programs[: int(max(0, int(args.trace_max_programs_per_task)))]:
                loss = tp.get("loss") if isinstance(tp.get("loss"), dict) else {}
                ls = int(loss.get("shape") or 0)
                lc = int(loss.get("cells") or 0)
                if ls > int(args.trace_max_loss_shape) or lc <= 0 or lc > int(args.trace_max_loss_cells):
                    continue
                steps_raw = tp.get("steps")
                if not isinstance(steps_raw, list):
                    continue
                flat = _flatten_steps([s for s in steps_raw if isinstance(s, dict)])
                op_ids = [str(s.get("op_id") or "") for s in flat if str(s.get("op_id") or "")]
                if not op_ids:
                    continue
                # Palette relation is optional: use predicted_grid if present.
                pal_rel = "EQUAL"
                want_grid = None
                task = obj.get("task") if isinstance(obj.get("task"), dict) else {}
                test_pairs = task.get("test_pairs")
                if isinstance(test_pairs, list) and test_pairs and isinstance(test_pairs[0], dict):
                    want_grid = test_pairs[0].get("out_grid")
                got_grid = tp.get("predicted_grid")
                if isinstance(want_grid, list) and isinstance(got_grid, list):
                    try:
                        pal_rel = _palette_relation(_unique_colors(want_grid), _unique_colors(got_grid))
                    except Exception:
                        pal_rel = "EQUAL"
                candidates.append((_signature_for_trace(loss_shape=ls, loss_cells=lc, palette_rel=pal_rel), op_ids))

        if not candidates:
            continue

        # Convert candidate programs into concept template keys via contiguous subseqs.
        for sig, op_ids_full in candidates:
            op_ids_full = [str(x) for x in op_ids_full if str(x)]
            for seq in _contiguous_subseqs(op_ids_full, min_len=min_len, max_len=max_len):
                op_ids = list(seq)
                # Reject meta ops and empty.
                op_ids = [x for x in op_ids if x and x not in {"macro_call", "concept_call"}]
                if not op_ids:
                    continue
                key = (_stable_json(sig), _stable_json(op_ids))
                support[key] += 1
                payload[key] = (dict(sig), list(op_ids))

                if induction_log_path is not None and task_id:
                    induced_seen += 1
                    induction_rows.append(
                        {
                            "schema_version": 146,
                            "kind": "arc_concept_induction_v146",
                            "source": "trace_mine_v146",
                            "rank": 0,
                            "task_id": str(task_id),
                            "signature": dict(sig),
                            "op_ids": list(op_ids),
                        }
                    )

    rows: List[Dict[str, Any]] = []
    for key, sup in support.items():
        if int(sup) < int(args.min_support):
            continue
        sig_json, ops_json = key
        sig, op_ids = payload[key]
        cid = _concept_id_for(signature=dict(sig), op_ids=list(op_ids))
        cost_bits = _concept_cost_bits(op_ids=op_ids, support=int(sup))
        # Additional guard: ensure the template is cheaper than calling raw primitives,
        # otherwise it will never be selected as policy.
        inner_cost = 0
        for op in op_ids:
            inner_cost += int(step_cost_bits_v141(op_id=str(op), args={}))
        cost_bits = min(int(cost_bits), max(1, int(inner_cost) - 1))
        rows.append(
            {
                "kind": "arc_concept_template_v146",
                "schema_version": 146,
                "concept_id": str(cid),
                "signature": dict(sig),
                "op_ids": list(op_ids),
                "support": int(sup),
                "cost_bits": int(cost_bits),
            }
        )

    rows.sort(
        key=lambda r: (
            -int(r.get("support") or 0),
            -len(list(r.get("op_ids") or [])),
            int(r.get("cost_bits") or 10),
            str(r.get("concept_id") or ""),
        )
    )
    rows = rows[: int(max(0, int(args.max_concepts)))]

    out_path.write_text("\n".join(_stable_json(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    if induction_log_path is not None:
        # Deterministic: stable sort by task_id then payload.
        induction_rows.sort(key=lambda r: (str(r.get("task_id") or ""), _stable_json(r)))
        induction_log_path.write_text(
            "\n".join(_stable_json(r) for r in induction_rows) + ("\n" if induction_rows else ""), encoding="utf-8"
        )

    print(
        _stable_json(
            {
                "ok": True,
                "run_dir": str(run_dir),
                "out": str(out_path),
                "concepts_written": int(len(rows)),
                "tasks_seen": int(tasks_seen),
                "induction_rows_written": int(len(induction_rows)),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

