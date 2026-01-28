#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


def _iter_per_task_files(run_dir: Path) -> Iterable[Path]:
    per_task_dir = run_dir / "per_task"
    if not per_task_dir.is_dir():
        raise SystemExit(f"missing_per_task_dir:{per_task_dir}")
    for p in sorted(per_task_dir.glob("*.json.json")):
        if p.is_file():
            yield p


def _concept_ids_used_in_task(per_task_obj: Dict[str, Any]) -> Set[str]:
    scoring = per_task_obj.get("scoring") if isinstance(per_task_obj.get("scoring"), dict) else {}
    if str(scoring.get("solver_status") or "") != "SOLVED":
        return set()
    solver_results = per_task_obj.get("solver_results")
    if not isinstance(solver_results, list) or not solver_results:
        return set()
    sr0 = solver_results[0] if isinstance(solver_results[0], dict) else {}
    steps = sr0.get("program_steps")
    if not isinstance(steps, list):
        return set()
    out: Set[str] = set()
    for st in steps:
        if not isinstance(st, dict):
            continue
        if str(st.get("op_id") or "") != "concept_call":
            continue
        args = st.get("args") if isinstance(st.get("args"), dict) else {}
        cid = str(args.get("concept_id") or "")
        if cid:
            out.add(cid)
    return out


def bump_concept_support_v146(
    *,
    concept_bank_in: Path,
    run_dir: Path,
) -> List[Dict[str, Any]]:
    support_add: DefaultDict[str, int] = defaultdict(int)
    for p in _iter_per_task_files(run_dir):
        obj = _read_json(p)
        if not isinstance(obj, dict):
            continue
        for cid in _concept_ids_used_in_task(obj):
            support_add[str(cid)] += 1

    rows: List[Dict[str, Any]] = []
    for row in _read_jsonl(concept_bank_in):
        cid = str(row.get("concept_id") or "")
        if not cid:
            continue
        support = int(row.get("support") or 0)
        support = int(support) + int(support_add.get(cid, 0))
        row2 = dict(row)
        row2["support"] = int(support)
        rows.append(row2)

    rows.sort(key=lambda r: (-int(r.get("support") or 0), str(r.get("concept_id") or ""), _stable_json(r)))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concept_bank_in", required=True, help="Input concept bank jsonl")
    ap.add_argument("--run_dir", required=True, help="ARC scalpel run dir (contains per_task/*.json.json)")
    ap.add_argument("--out", required=True, help="Output bumped concept bank jsonl (WORM)")
    args = ap.parse_args()

    bank_in = Path(str(args.concept_bank_in)).resolve()
    if not bank_in.is_file():
        raise SystemExit(f"missing_concept_bank_in:{bank_in}")

    run_dir = Path(str(args.run_dir)).resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"missing_run_dir:{run_dir}")

    out_path = Path(str(args.out)).resolve()
    _ensure_absent(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = bump_concept_support_v146(concept_bank_in=bank_in, run_dir=run_dir)
    out_path.write_text("\n".join(_stable_json(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    print(_stable_json({"ok": True, "out": str(out_path), "concepts_written": int(len(rows))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

