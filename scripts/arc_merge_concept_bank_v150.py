#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                yield obj


def _get_concept_id(row: Dict[str, Any]) -> str:
    return str(row.get("concept_id") or row.get("csv_id") or row.get("csg_id") or "")


def _intish(v: Any, default: int = 0) -> int:
    if isinstance(v, int):
        return int(v)
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return int(default)


def _row_score(row: Dict[str, Any]) -> Tuple[int, int, int, str]:
    # Higher support first, then lower MDL, then prefer concrete steps, then id.
    support = _intish(row.get("support"), 0)
    cost_bits = _intish(row.get("cost_bits"), 10)
    steps = row.get("steps")
    has_steps = 1 if isinstance(steps, list) and bool(steps) else 0
    cid = _get_concept_id(row)
    return (support, -cost_bits, has_steps, cid)


def _merge_task_ids(a: Any, b: Any) -> Optional[List[str]]:
    ids: List[str] = []
    if isinstance(a, list):
        ids.extend(str(x) for x in a if str(x))
    if isinstance(b, list):
        ids.extend(str(x) for x in b if str(x))
    if not ids:
        return None
    return sorted(set(ids))


def merge_concept_bank_v150(*, inputs: Sequence[Path]) -> List[Dict[str, Any]]:
    allowed_kinds = {"arc_concept_template_v146", "arc_concept_csg_v148", "arc_concept_csg_v149"}
    best_by_id: Dict[str, Dict[str, Any]] = {}

    for p in inputs:
        for row in _read_jsonl(p):
            kind = str(row.get("kind") or "")
            if kind and kind not in allowed_kinds:
                continue
            cid = _get_concept_id(row)
            if not cid:
                continue
            prev = best_by_id.get(cid)
            if prev is None:
                best_by_id[cid] = dict(row)
                continue

            # Merge support_task_ids deterministically when available.
            merged_task_ids = _merge_task_ids(prev.get("support_task_ids"), row.get("support_task_ids"))
            candidate = dict(row)
            if merged_task_ids is not None:
                candidate["support_task_ids"] = merged_task_ids

            # Prefer the row with higher support; on ties, lower cost_bits; on ties, keep concrete steps.
            if _row_score(candidate) > _row_score(prev):
                best_by_id[cid] = candidate
            else:
                if merged_task_ids is not None:
                    prev2 = dict(prev)
                    prev2["support_task_ids"] = merged_task_ids
                    best_by_id[cid] = prev2

    # Stable output ordering; solver will re-order internally anyway.
    rows = list(best_by_id.values())
    rows.sort(
        key=lambda r: (
            str(r.get("kind") or ""),
            -_intish(r.get("support"), 0),
            _intish(r.get("cost_bits"), 10),
            _get_concept_id(r),
            _stable_json(r),
        )
    )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", default=[], help="Input concept bank jsonl (repeat)")
    ap.add_argument(
        "--in_glob",
        action="append",
        default=[],
        help="Glob pattern for input jsonl files (repeat). Expanded deterministically.",
    )
    ap.add_argument("--out", required=True, help="Output concept bank jsonl (WORM: must not exist)")
    args = ap.parse_args()

    inputs: List[Path] = [Path(str(p)).resolve() for p in list(args.inputs or [])]
    globs = [str(g) for g in list(args.in_glob or []) if str(g).strip()]
    for pat in globs:
        matches = sorted(Path().glob(str(pat)))
        if not matches:
            raise SystemExit(f"glob_no_matches:{pat}")
        for m in matches:
            if m.is_file():
                inputs.append(m.resolve())
    if not inputs:
        raise SystemExit("missing_inputs")
    inputs = sorted({str(p): p for p in inputs}.values(), key=lambda p: str(p))
    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"missing_input:{p}")

    out_path = Path(str(args.out)).resolve()
    if out_path.exists():
        raise SystemExit(f"worm_exists:{out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = merge_concept_bank_v150(inputs=inputs)
    out_path.write_text("\n".join(_stable_json(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    print(_stable_json({"ok": True, "out": str(out_path), "rows_written": int(len(rows))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

