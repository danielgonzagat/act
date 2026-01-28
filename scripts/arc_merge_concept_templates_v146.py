#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if isinstance(obj, dict):
            yield obj


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


def _norm_signature(sig: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(sig, dict):
        return None
    out: Dict[str, Any] = {}
    for k in sorted(sig.keys()):
        v = sig.get(k)
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[str(k)] = v
        else:
            out[str(k)] = str(v)
    return out


def _norm_op_ids(raw: Any) -> Optional[List[str]]:
    if not isinstance(raw, list) or not raw:
        return None
    op_ids: List[str] = []
    for x in raw:
        s = str(x)
        if s:
            op_ids.append(s)
    return op_ids or None


def merge_concept_templates_v146(
    *,
    inputs: Sequence[Path],
    min_support: int,
    max_concepts: int,
) -> List[Dict[str, Any]]:
    by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for p in inputs:
        for row in _read_jsonl(p):
            sig = _norm_signature(row.get("signature"))
            op_ids = _norm_op_ids(row.get("op_ids"))
            if sig is None or op_ids is None:
                continue
            if len(op_ids) > 3:
                continue
            key_sig = _stable_json(sig)
            key_ops = _stable_json(op_ids)
            key = (key_sig, key_ops)
            support = _row_support(row)
            prev = by_key.get(key)
            if prev is None:
                by_key[key] = {"signature": sig, "op_ids": list(op_ids), "support": int(support), "cost_bits": int(row.get("cost_bits") or 10)}
                continue
            prev["support"] = int(prev.get("support") or 0) + int(support)
            prev["cost_bits"] = min(int(prev.get("cost_bits") or 10), int(row.get("cost_bits") or 10))

    out: List[Dict[str, Any]] = []
    for (_, _), row in by_key.items():
        support = int(row.get("support") or 0)
        if support < int(min_support):
            continue
        signature = row.get("signature") if isinstance(row.get("signature"), dict) else {}
        op_ids = row.get("op_ids") if isinstance(row.get("op_ids"), list) else []
        body = {
            "schema_version": 146,
            "kind": "concept_key_v146",
            "signature": signature,
            "op_ids": list(op_ids),
        }
        cid = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
        out.append(
            {
                "kind": "arc_concept_template_v146",
                "schema_version": 146,
                "concept_id": "csg_" + str(cid)[:16],
                "signature": signature,
                "op_ids": op_ids,
                "support": int(support),
                "cost_bits": int(row.get("cost_bits") or 10),
            }
        )

    out.sort(key=lambda r: (-int(r.get("support") or 0), str(r.get("concept_id") or ""), _stable_json(r)))
    return out[: int(max(0, int(max_concepts)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inputs", action="append", default=[], help="Input concept templates jsonl (repeat)")
    ap.add_argument(
        "--in_glob",
        action="append",
        default=[],
        help="Glob pattern for input jsonl files (repeat). Expanded deterministically.",
    )
    ap.add_argument("--out", required=True, help="Output concept templates jsonl (WORM: must not exist)")
    ap.add_argument("--min_support", type=int, default=2)
    ap.add_argument("--max_concepts", type=int, default=128)
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
    # De-dup deterministically.
    inputs = sorted({str(p): p for p in inputs}.values(), key=lambda p: str(p))
    for p in inputs:
        if not p.is_file():
            raise SystemExit(f"missing_input:{p}")

    out_path = Path(str(args.out)).resolve()
    if out_path.exists():
        raise SystemExit(f"worm_exists:{out_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = merge_concept_templates_v146(inputs=inputs, min_support=int(args.min_support), max_concepts=int(args.max_concepts))
    out_path.write_text("\n".join(_stable_json(r) for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    print(_stable_json({"ok": True, "out": str(out_path), "concepts_written": int(len(rows))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
