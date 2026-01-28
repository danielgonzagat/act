#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


def _norm_op_ids(raw: Any) -> Optional[List[str]]:
    if not isinstance(raw, list) or not raw:
        return None
    op_ids: List[str] = []
    for x in raw:
        s = str(x)
        if s:
            op_ids.append(s)
    return op_ids or None


def _row_support(row: Dict[str, Any]) -> int:
    if isinstance(row.get("support"), int):
        return int(row["support"])
    if isinstance(row.get("support_count"), int):
        return int(row["support_count"])
    if isinstance(row.get("support"), str) and str(row.get("support") or "").isdigit():
        return int(str(row.get("support") or "0"))
    if isinstance(row.get("support_count"), str) and str(row.get("support_count") or "").isdigit():
        return int(str(row.get("support_count") or "0"))
    return 0


def _row_origin_clusters(row: Dict[str, Any]) -> Set[str]:
    oc = row.get("origin_clusters")
    if not isinstance(oc, list):
        return set()
    return {str(x) for x in oc if str(x)}


def merge_operator_templates_v147(
    *,
    inputs: Sequence[Path],
    min_support: int,
    min_len: int,
    max_len: int,
    max_operators: int,
) -> List[Dict[str, Any]]:
    by_seq: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    for p in inputs:
        for row in _read_jsonl(p):
            if not isinstance(row, dict):
                continue
            op_ids = _norm_op_ids(row.get("op_ids"))
            if op_ids is None:
                continue
            if int(len(op_ids)) < int(min_len) or int(len(op_ids)) > int(max_len):
                continue
            seq = tuple(op_ids)
            support = _row_support(row)
            oc = _row_origin_clusters(row)
            prev = by_seq.get(seq)
            if prev is None:
                by_seq[seq] = {"op_ids": list(op_ids), "support": int(support), "origin_clusters": set(oc)}
                continue
            prev_s = int(prev.get("support") or 0)
            prev["support"] = int(max(prev_s, int(support)))
            prev_oc = prev.get("origin_clusters")
            if isinstance(prev_oc, set):
                prev_oc |= set(oc)
            else:
                prev["origin_clusters"] = set(oc)

    out: List[Dict[str, Any]] = []
    for seq, row in by_seq.items():
        support = int(row.get("support") or 0)
        if support < int(min_support):
            continue
        operator_id = _sha256_hex(_stable_json({"op_ids": list(seq)}))
        oc_any = row.get("origin_clusters")
        oc_set: Set[str] = set()
        if isinstance(oc_any, set):
            oc_set = {str(x) for x in oc_any if str(x)}
        out.append(
            {
                "kind": "arc_operator_template_v147",
                "schema_version": 147,
                "operator_id": str(operator_id),
                "op_ids": list(seq),
                "support": int(support),
                "emitted_from_failure": 1,
                "origin_clusters": sorted(list(oc_set)),
            }
        )

    out.sort(
        key=lambda r: (-int(r.get("support") or 0), -len(list(r.get("op_ids") or [])), str(r.get("operator_id") or ""))
    )
    return out[: int(max(0, int(max_operators)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", required=True, help="Input operator templates jsonl (repeat)")
    ap.add_argument("--out", required=True, help="Output operator templates jsonl (WORM: must not exist)")
    ap.add_argument("--min_support", type=int, default=3)
    ap.add_argument("--min_len", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=5)
    ap.add_argument("--max_operators", type=int, default=128)
    args = ap.parse_args()

    inputs = [Path(str(p)).resolve() for p in list(args.inputs or [])]
    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"missing_input:{p}")

    out_path = Path(str(args.out)).resolve()
    if out_path.exists():
        raise SystemExit(f"worm_exists:{out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = merge_operator_templates_v147(
        inputs=inputs,
        min_support=int(args.min_support),
        min_len=int(args.min_len),
        max_len=int(args.max_len),
        max_operators=int(args.max_operators),
    )

    out_text = "\n".join(_stable_json(r) for r in rows) + ("\n" if rows else "")
    out_path.write_text(out_text, encoding="utf-8")
    print(_stable_json({"ok": True, "out": str(out_path), "operators_written": int(len(rows))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

