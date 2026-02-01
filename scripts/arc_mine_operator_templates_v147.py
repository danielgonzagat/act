#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import heapq
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps
from atos_core.arc_ops_v141 import OP_DEFS_V141
from atos_core.omega_v2 import arc_task_family_id, residual_signature_key


def _stable_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _iter_per_task_files(per_task_dir: Path) -> Iterable[Path]:
    for p in sorted(per_task_dir.glob("*.json.json")):
        if p.is_file():
            yield p


def _extract_failure_kind(per_task_obj: Dict[str, Any]) -> str:
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return ""
    sr0 = solver_results[0]
    if not isinstance(sr0, dict):
        return ""
    status = str(sr0.get("status") or "")
    if status == "SOLVED":
        return ""
    fr = sr0.get("failure_reason")
    if isinstance(fr, dict):
        return str(fr.get("kind") or "")
    return "FAIL"


def _extract_trace_program_op_ids(
    per_task_obj: Dict[str, Any],
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

    def _flatten_steps(steps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Flatten meta steps (macro_call / concept_call) into base steps.

        Operators are sequences of base op_ids; meta-calls are packaging and should not block mining.
        """
        flat: List[Dict[str, Any]] = []
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
                    flat.extend(_flatten_steps([x for x in inner if isinstance(x, dict)]))
                continue
            flat.append({"op_id": str(op_id), "args": dict(args)})
        return flat

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
        flat = _flatten_steps([s for s in steps if isinstance(s, dict)])
        op_ids = [str(s.get("op_id") or "") for s in flat if str(s.get("op_id") or "")]
        if op_ids:
            out.append(op_ids)
    return out


def _all_contiguous_subseqs(op_ids: Sequence[str], *, max_len: int) -> Iterable[Tuple[str, ...]]:
    """
    Generate contiguous subsequences up to max_len.

    NOTE: We intentionally generate from length 1 and apply the final min_len filter after
    deterministic closure (see _close_operator_seq). This enables mining of reusable
    grid->grid closures from single-step traces like bbox_by_color/cc4 by appending the
    minimal typed suffix (e.g., crop_bbox+commit_patch).
    """
    n = int(len(op_ids))
    if n <= 0:
        return
    for L in range(1, min(max_len, n) + 1):
        for i in range(0, n - L + 1):
            yield tuple(op_ids[i : i + L])


def _op_reads_writes(op_id: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    od = OP_DEFS_V141.get(str(op_id) or "")
    reads: Tuple[str, ...] = tuple(getattr(od, "reads", ()) or ()) if od is not None else tuple()
    writes: Tuple[str, ...] = tuple(getattr(od, "writes", ()) or ()) if od is not None else tuple()
    reads_s = tuple(str(x) for x in reads if str(x))
    writes_s = tuple(str(x) for x in writes if str(x))
    return reads_s, writes_s


def _missing_start_types(seq: Sequence[str]) -> Set[str]:
    """
    Compute the minimal set of state keys that must be present at the *start* of `seq`
    for the sequence to be type-feasible (keys persist; we only track presence).
    """
    avail: Set[str] = {"grid", "orig"}
    missing: Set[str] = set()
    for op_id in seq:
        reads, writes = _op_reads_writes(str(op_id))
        if not reads and not writes:
            missing.add("__unknown_op__")
            continue
        for t in reads:
            if t not in avail:
                missing.add(str(t))
        for t in writes:
            avail.add(str(t))
    return set(missing)


def _type_feasible_from_start(seq: Sequence[str]) -> bool:
    avail: Set[str] = {"grid", "orig"}
    for op_id in seq:
        reads, writes = _op_reads_writes(str(op_id))
        if not reads and not writes:
            return False
        if any(str(t) not in avail for t in reads):
            return False
        for t in writes:
            avail.add(str(t))
    return True


def _prefix_to_produce_types(missing_types: Set[str], *, max_len: int) -> Optional[Tuple[str, ...]]:
    """
    Deterministically produce a minimal prefix (non-grid-writing ops only) that makes
    `missing_types` available from the canonical start state {grid, orig}.

    We restrict to ops that do NOT write grid to avoid changing the grid before the mined
    operator itself runs (operators are intended as reusable closures).
    """
    want = {str(t) for t in (missing_types or set()) if str(t) and str(t) not in {"grid", "orig"}}
    if not want:
        return tuple()
    if "__unknown_op__" in want:
        return None
    if int(max_len) <= 0:
        return None

    # Candidate prefix ops: any op that doesn't write grid (they only create typed objects).
    cand_ops: List[Tuple[int, str]] = []
    for op_id, od in OP_DEFS_V141.items():
        writes = tuple(getattr(od, "writes", ()) or ())
        if "grid" in {str(x) for x in writes if str(x)}:
            continue
        cand_ops.append((int(getattr(od, "base_cost_bits", 0) or 0), str(op_id)))
    cand_ops.sort(key=lambda t: (int(t[0]), str(t[1])))

    start = frozenset({"grid", "orig"})

    def _cost_key(seq: Tuple[str, ...]) -> Tuple[int, int, str]:
        cost = 0
        for x in seq:
            od = OP_DEFS_V141.get(str(x) or "")
            cost += int(getattr(od, "base_cost_bits", 0) or 0) if od is not None else 0
        return (int(len(seq)), int(cost), ",".join(seq))

    # Dijkstra over type-sets with lexicographic tie-break. State space is tiny (types<=6).
    pq: List[Tuple[Tuple[int, int, str], frozenset[str], Tuple[str, ...]]] = []
    heapq.heappush(pq, (_cost_key(tuple()), start, tuple()))
    best: Dict[frozenset[str], Tuple[int, int, str]] = {start: _cost_key(tuple())}

    while pq:
        cost0, types0, seq0 = heapq.heappop(pq)
        if best.get(types0) != cost0:
            continue
        if want.issubset(set(types0)):
            return seq0
        if int(len(seq0)) >= int(max_len):
            continue

        for _bc, op_id in cand_ops:
            reads, writes = _op_reads_writes(op_id)
            if not reads and not writes:
                continue
            if any(str(t) not in types0 for t in reads):
                continue
            t1 = frozenset(set(types0) | {str(t) for t in writes})
            s1 = tuple(list(seq0) + [str(op_id)])
            c1 = _cost_key(s1)
            prev = best.get(t1)
            if prev is None or c1 < prev:
                best[t1] = c1
                heapq.heappush(pq, (c1, t1, s1))

    return None


def _close_operator_seq(seq: Tuple[str, ...], *, max_len: int) -> Optional[Tuple[str, ...]]:
    """
    Deterministically close partial operator traces into reusable grid->grid closures.

    Key rule (v141 state machine):
      crop_bbox writes patch, not grid; commit_patch is required to make the crop effective.
    """
    if not seq:
        return None

    last = str(seq[-1] or "")
    if not last:
        return None
    _reads_last, writes_last = _op_reads_writes(last)
    wset = {str(x) for x in writes_last if str(x)}
    closed_suffix: Tuple[str, ...] = tuple()
    if "grid" in wset:
        closed_suffix = seq
    else:
        # Typed closure suffixes (minimal, deterministic) to ensure the final output writes grid.
        # This expands operator discovery beyond a tiny hand-set of "already-closed" traces,
        # and turns common non-grid tails (bbox/obj/objset/patch) into reusable closures.
        suffix: Tuple[str, ...] = tuple()
        if "patch" in wset:
            suffix = ("commit_patch",)
        elif "bbox" in wset:
            suffix = ("crop_bbox", "commit_patch")
        elif "obj" in wset:
            suffix = ("obj_bbox", "crop_bbox", "commit_patch")
        elif "objset" in wset:
            suffix = ("select_obj", "obj_bbox", "crop_bbox", "commit_patch")
        else:
            # Not a grid-producing closure and no deterministic typed suffix available.
            return None

        closed_suffix = tuple(list(seq) + list(suffix))
    if int(len(closed_suffix)) > int(max_len):
        return None

    # Prefix-closure: ensure the sequence is feasible from the canonical start state {grid, orig}.
    missing = _missing_start_types(closed_suffix)
    if "__unknown_op__" in missing:
        return None
    prefix_max = int(max_len) - int(len(closed_suffix))
    prefix = _prefix_to_produce_types(missing, max_len=int(prefix_max))
    if prefix is None:
        return None

    closed = tuple(list(prefix) + list(closed_suffix))
    if int(len(closed)) > int(max_len):
        return None
    if not _type_feasible_from_start(closed):
        return None
    # Sanity: must end in a grid write (commit_patch writes grid; some ops write grid directly).
    last2 = str(closed[-1] or "")
    _reads2, writes2 = _op_reads_writes(last2)
    if "grid" not in {str(x) for x in writes2 if str(x)}:
        return None
    return closed


def _load_origin_clusters_by_task(run_dir: Path) -> Dict[str, str]:
    """
    Deterministic "born-from-failure" provenance for operators.

    We map each task_id to the same cluster key Ω would use:
      - rsig_<hash(signature)> when residual_signature exists
      - else family_id
    """
    events_path = run_dir / "omega_events_v2.jsonl"
    out: Dict[str, str] = {}

    # Preferred source: Ω v2 event stream (contains residual_signature keys).
    if events_path.is_file():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict):
                continue
            tid = str(ev.get("task_id") or "")
            if not tid:
                continue
            if bool(ev.get("episode_success", False)) or str(ev.get("episode_outcome") or "") == "SUCCESS":
                continue
            rs = ev.get("residual_signature") if isinstance(ev.get("residual_signature"), dict) else None
            fam = str(ev.get("task_family_id") or ev.get("family_id") or "")
            if not fam:
                task = ev.get("task") if isinstance(ev.get("task"), dict) else {}
                if isinstance(task, dict) and task:
                    fam = str(arc_task_family_id(task) or "")
            ck = residual_signature_key(rs) if rs else str(fam)
            if not ck:
                continue
            out[tid] = str(ck)

    # Fallback (still deterministic): infer failure-born provenance from per_task failures.
    # This enables deep mining when Ω is disabled in the origin run, while keeping the
    # "born-from-failure" requirement (we only mark tasks that actually failed).
    if not out:
        per_task_dir = run_dir / "per_task"
        if per_task_dir.is_dir():
            for p in _iter_per_task_files(per_task_dir):
                task_id = p.name.split(".json.json")[0]
                try:
                    obj = json.loads(p.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                fk = _extract_failure_kind(obj)
                if not fk:
                    continue
                task = obj.get("task") if isinstance(obj.get("task"), dict) else {}
                fam = str(arc_task_family_id(task) or "") if isinstance(task, dict) and task else ""
                if fam:
                    out[str(task_id)] = str(fam)

    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument(
        "--origin_run_dir",
        default="",
        help=(
            "Optional run dir to source born-from-failure provenance (omega_events_v2.jsonl or per_task failures). "
            "Useful for deep mining where the current run may solve tasks that were failures in the base run."
        ),
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--include_failure_kinds",
        default="MISSING_OPERATOR,SEARCH_BUDGET_EXCEEDED,MISSING_CONCEPT,FAIL",
        help="Comma-separated failure kinds to mine from (default: common structural failures).",
    )
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=5)
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--max_operators", type=int, default=256)
    ap.add_argument("--trace_max_programs_per_task", type=int, default=20)
    ap.add_argument("--trace_max_loss_shape", type=int, default=0)
    ap.add_argument("--trace_max_loss_cells", type=int, default=120)
    ap.add_argument(
        "--include_solved_from_failure",
        action="store_true",
        help=(
            "Also mine from SOLVED tasks if they have born-from-failure provenance in omega_events_v2.jsonl "
            "(useful for deep induction runs that solve previously-failed tasks)."
        ),
    )
    ap.add_argument(
        "--include_examples",
        action="store_true",
        help="Include example task ids per operator (debug-only; default: off to avoid puzzle-specific metadata).",
    )
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir)).resolve()
    origin_run_dir = Path(str(args.origin_run_dir)).resolve() if str(args.origin_run_dir or "").strip() else run_dir
    per_task_dir = run_dir / "per_task"
    if not per_task_dir.is_dir():
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")

    min_len = int(args.min_len)
    max_len = int(args.max_len)
    if min_len < 1 or max_len < min_len:
        raise SystemExit("bad_len_bounds")

    include_kinds = {str(x).strip() for x in str(args.include_failure_kinds or "").split(",") if str(x).strip()}
    if not include_kinds:
        raise SystemExit("empty_include_failure_kinds")

    origin_by_task = _load_origin_clusters_by_task(origin_run_dir)

    # seq -> set(task_id)
    support: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    # seq -> set(cluster_key)
    origin_clusters: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)
    examples: Dict[Tuple[str, ...], Set[str]] = defaultdict(set)

    tasks_seen = 0
    tasks_included = 0
    for p in _iter_per_task_files(per_task_dir):
        tasks_seen += 1
        task_id = p.name.split(".json.json")[0]
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        fk = _extract_failure_kind(obj)
        solved_from_failure = bool(args.include_solved_from_failure) and (not fk) and bool(origin_by_task.get(task_id))
        if fk:
            if fk not in include_kinds:
                continue
        else:
            if not solved_from_failure:
                continue
        tasks_included += 1

        programs = _extract_trace_program_op_ids(
            obj,
            max_programs=int(args.trace_max_programs_per_task),
            max_loss_shape=int(args.trace_max_loss_shape),
            max_loss_cells=int(args.trace_max_loss_cells),
        )
        if not programs:
            continue

        ck = str(origin_by_task.get(task_id) or "")
        if not ck:
            task = obj.get("task") if isinstance(obj.get("task"), dict) else {}
            if isinstance(task, dict):
                ck = str(arc_task_family_id(task) or "")

        seen_local: Set[Tuple[str, ...]] = set()
        for op_ids in programs:
            for seq0 in _all_contiguous_subseqs(op_ids, max_len=max_len):
                seq = _close_operator_seq(seq0, max_len=int(max_len))
                if seq is None:
                    continue
                if int(len(seq)) < int(min_len) or int(len(seq)) > int(max_len):
                    continue
                if seq in seen_local:
                    continue
                seen_local.add(seq)
                support[seq].add(task_id)
                if ck:
                    origin_clusters[seq].add(ck)
                if bool(args.include_examples):
                    examples[seq].add(task_id)

    rows: List[Dict[str, Any]] = []
    for seq, tasks in support.items():
        if len(tasks) < int(args.min_support):
            continue
        op_ids = list(seq)
        operator_id = _sha256_hex(_stable_json({"op_ids": op_ids}))
        row: Dict[str, Any] = {
            "kind": "arc_operator_template_v147",
            "schema_version": 147,
            "operator_id": str(operator_id),
            "op_ids": op_ids,
            "support": int(len(tasks)),
            "emitted_from_failure": 1,
            "origin_clusters": sorted({str(x) for x in (origin_clusters.get(seq) or set()) if str(x)}),
        }
        if bool(args.include_examples):
            row["example_task_ids"] = sorted({str(x) for x in (examples.get(seq) or set()) if str(x)})[:20]
        rows.append(row)

    rows.sort(
        key=lambda r: (-int(r.get("support") or 0), -len(list(r.get("op_ids") or [])), str(r.get("operator_id") or ""))
    )
    rows = rows[: int(max(0, int(args.max_operators)))]

    out_path = Path(str(args.out)).resolve()
    _ensure_absent(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(_stable_json(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")

    print(
        canonical_json_dumps(
            {
                "ok": True,
                "run_dir": str(run_dir),
                "origin_run_dir": str(origin_run_dir),
                "out": str(out_path),
                "tasks_seen": int(tasks_seen),
                "tasks_included": int(tasks_included),
                "operators_written": int(len(rows)),
                "min_len": int(min_len),
                "max_len": int(max_len),
                "min_support": int(args.min_support),
                "include_failure_kinds": sorted(list(include_kinds)),
                "include_solved_from_failure": bool(args.include_solved_from_failure),
                "trace_max_programs_per_task": int(args.trace_max_programs_per_task),
                "trace_max_loss_shape": int(args.trace_max_loss_shape),
                "trace_max_loss_cells": int(args.trace_max_loss_cells),
            }
        )
    )


if __name__ == "__main__":
    main()
