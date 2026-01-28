#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("PYTHONHASHSEED", "0")
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _write_jsonl_x(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_absent(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", encoding="utf-8") as f:
        for row in rows:
            f.write(_stable_json(row) + "\n")


def _iter_per_task_paths(run_dir: Path) -> List[Path]:
    per_task_dir = run_dir / "per_task"
    if not per_task_dir.is_dir():
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")
    return sorted([p for p in per_task_dir.iterdir() if p.is_file() and p.name.endswith(".json")])


def _infer_task_id_from_filename(p: Path) -> str:
    name = p.name
    if name.endswith(".json.json"):
        return name[: -len(".json.json")]
    if name.endswith(".json"):
        return name[: -len(".json")]
    return name


def _steps_to_op_ids(steps: Any) -> Optional[List[str]]:
    if not isinstance(steps, list) or not steps:
        return None
    op_ids: List[str] = []
    for s in steps:
        if not isinstance(s, dict):
            continue
        op_id = str(s.get("op_id") or "")
        if not op_id or op_id == "macro_call":
            continue
        op_ids.append(op_id)
    return op_ids or None


def _extract_candidate_program_op_ids(per_task: Dict[str, Any], *, max_programs: int = 12) -> List[List[str]]:
    """
    Extract multiple candidate programs per task, prioritizing:
      - solved program (train-perfect) when present
      - near-miss structure (loss_shape==0)
      - otherwise best overall by (loss_shape, loss_cells, cost_bits)
    This lets macro mining see recurring subgraphs even when tasks are not solved.
    """
    solver_results = per_task.get("solver_results") if isinstance(per_task.get("solver_results"), list) else []
    if not solver_results or not isinstance(solver_results[0], dict):
        return []
    sr0 = solver_results[0]

    out: List[List[str]] = []

    # 1) Train-perfect program steps (when present).
    op_ids = _steps_to_op_ids(sr0.get("program_steps"))
    if op_ids is not None:
        out.append(op_ids)

    trace = sr0.get("trace") if isinstance(sr0.get("trace"), dict) else {}
    tps = trace.get("trace_programs") if isinstance(trace.get("trace_programs"), list) else []
    tps_dicts = [tp for tp in tps if isinstance(tp, dict)]

    # Sort trace programs deterministically by (loss_shape, loss_cells, cost_bits, depth, program_sig).
    def key(tp: Dict[str, Any]) -> Tuple[int, int, int, int, str]:
        loss = tp.get("loss") if isinstance(tp.get("loss"), dict) else {}
        return (
            int(loss.get("shape", 10**9)),
            int(loss.get("cells", 10**9)),
            int(tp.get("cost_bits") or 0),
            int(tp.get("depth") or 0),
            str(tp.get("program_sig") or ""),
        )

    tps_dicts.sort(key=key)

    # Prefer a handful of shape==0 candidates first (structure stage), then fill with best overall.
    shape0 = [tp for tp in tps_dicts if isinstance(tp.get("loss"), dict) and int(tp["loss"].get("shape", 10**9)) == 0]
    chosen = shape0[: max_programs // 2] + tps_dicts[: max_programs]

    # Dedup by program_sig.
    seen: Set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for tp in chosen:
        ps = str(tp.get("program_sig") or "")
        if ps and ps in seen:
            continue
        if ps:
            seen.add(ps)
        uniq.append(tp)

    for tp in uniq[:max_programs]:
        op_ids2 = _steps_to_op_ids(tp.get("steps"))
        if op_ids2 is not None:
            out.append(op_ids2)

    # Dedup identical op_id sequences (within task).
    seq_seen: Set[Tuple[str, ...]] = set()
    uniq_out: List[List[str]] = []
    for seq in out:
        t = tuple(seq)
        if t in seq_seen:
            continue
        seq_seen.add(t)
        uniq_out.append(seq)
    return uniq_out


def _is_grid_to_grid_op_seq(op_ids: Sequence[str]) -> bool:
    """
    Conservative check: sequence must be executable from the default abstract slots and
    end at the grid stage (no patch/bbox/obj leftovers).

    Uses the solver's abstract slot simulation to avoid encoding task-specific heuristics.
    """
    if not op_ids:
        return False
    try:
        from atos_core.arc_solver_v141 import ProgramStepV141, _abstract_slots_after_steps_v141, _stage_from_avail_v141
    except Exception:
        return False

    steps = tuple(ProgramStepV141(op_id=str(op), args={}) for op in op_ids)
    avail = _abstract_slots_after_steps_v141(steps)
    if bool(avail.get("invalid", False)):
        return False
    if str(_stage_from_avail_v141(avail)) != "grid":
        return False
    # Must not leave a patch/bbox/object slots live.
    if bool(avail.get("patch", False)) or bool(avail.get("bbox", False)) or bool(avail.get("obj", False)) or bool(avail.get("objset", False)):
        return False
    return True


def mine_macros_v143(
    *,
    run_dir: Path,
    min_support: int,
    min_len: int,
    max_len: int,
    top_k: int,
) -> List[Dict[str, Any]]:
    per_task_paths = _iter_per_task_paths(run_dir)

    support: Dict[Tuple[str, ...], Set[str]] = {}
    for p in per_task_paths:
        d = _read_json(p)
        task_id = _infer_task_id_from_filename(p).split(".")[0]
        programs = _extract_candidate_program_op_ids(d, max_programs=12)
        if not programs:
            continue
        for op_ids in programs:
            n = int(len(op_ids))
            for L in range(int(min_len), min(int(max_len), n) + 1):
                for i in range(0, n - L + 1):
                    seq = tuple(op_ids[i : i + L])
                    if not _is_grid_to_grid_op_seq(seq):
                        continue
                    support.setdefault(seq, set()).add(str(task_id))

    rows: List[Dict[str, Any]] = []
    for seq, tids in support.items():
        if len(tids) < int(min_support):
            continue
        score = int(len(tids)) * int(len(seq) - 1)
        body = {"schema_version": 143, "kind": "arc_macro_template_v143", "op_ids": list(seq)}
        from atos_core.act import canonical_json_dumps, sha256_hex

        macro_id = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
        rows.append(
            {
                **body,
                "macro_id": str(macro_id),
                "support_count": int(len(tids)),
                "support_task_ids": [str(x) for x in sorted(list(tids))[:12]],
                "score": int(score),
            }
        )

    rows.sort(key=lambda r: (-int(r.get("score") or 0), -int(r.get("support_count") or 0), _stable_json(r)))
    return rows[: int(top_k)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=5)
    ap.add_argument("--top_k", type=int, default=64)
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir))
    out_jsonl = Path(str(args.out_jsonl))
    rows = mine_macros_v143(
        run_dir=run_dir,
        min_support=int(args.min_support),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        top_k=int(args.top_k),
    )
    _write_jsonl_x(out_jsonl, rows)
    print(_stable_json({"ok": True, "out_jsonl": str(out_jsonl), "n": int(len(rows))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
