#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def _stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("bank_jsonl", help="Concept bank jsonl (arc_concept_bank_v146_*.jsonl)")
    args = ap.parse_args()

    p = Path(args.bank_jsonl).resolve()
    if not p.is_file():
        raise SystemExit(f"missing_file:{p}")

    total = 0
    multiop = 0
    multistage = 0
    max_support = 0
    by_diff_kind: Dict[str, int] = {}

    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        total += 1
        op_ids = obj.get("op_ids") if isinstance(obj.get("op_ids"), list) else []
        op_len = len([str(x) for x in op_ids if str(x)])
        if op_len > 1:
            multiop += 1
        sig = obj.get("signature") if isinstance(obj.get("signature"), dict) else {}
        dk = str(sig.get("diff_kind") or "")
        if dk:
            by_diff_kind[dk] = int(by_diff_kind.get(dk, 0)) + 1
        if dk == "MULTI_STAGE":
            multistage += 1
        try:
            max_support = max(int(max_support), int(obj.get("support") or 0))
        except Exception:
            pass

    out = {
        "bank": str(p),
        "concepts_total": int(total),
        "concepts_multiop": int(multiop),
        "concepts_multistage": int(multistage),
        "concepts_max_support": int(max_support),
        "by_diff_kind": {k: int(by_diff_kind[k]) for k in sorted(by_diff_kind.keys())},
    }
    print(_stable_json(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

