#!/usr/bin/env python3
import collections
import json
import pathlib
import sys


def _pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _norm_status(rec):
    s = _pick(rec, "status", "result", "outcome")
    if isinstance(s, str) and s.strip():
        return s.strip().upper()
    ok = _pick(rec, "ok", "passed", "success")
    if ok is True:
        return "SOLVED"
    if ok is False:
        return "FAIL"
    return "UNKNOWN"


def _norm_reason(rec):
    r = _pick(rec, "failure_kind", "fail_reason", "failure_reason", "reason", "error_type", "error")
    if r is None:
        return ""
    if isinstance(r, str):
        return r.strip() or ""
    if isinstance(r, dict):
        # common shape: {"kind": "...", ...}
        k = r.get("kind")
        if isinstance(k, str) and k.strip():
            return k.strip()
    return str(r)


def main():
    if len(sys.argv) != 2:
        print("usage: diag_reasons.py /path/to/per_task_manifest.jsonl", file=sys.stderr)
        return 2

    p = pathlib.Path(sys.argv[1])
    if not p.exists():
        print(f"ERROR: file not found: {p}", file=sys.stderr)
        return 1

    by_status = collections.Counter()
    by_reason = collections.Counter()
    ex_by_reason = collections.defaultdict(list)

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            status = _norm_status(rec)
            by_status[status] += 1
            if status != "SOLVED":
                reason = _norm_reason(rec) or "UNKNOWN_FAIL_REASON"
                by_reason[reason] += 1
                tid = _pick(rec, "task_id", "id", "name", default="(no_task_id)")
                if len(ex_by_reason[reason]) < 10:
                    ex_by_reason[reason].append(str(tid))

    total = sum(by_status.values())
    print(f"TOTAL: {total}")
    print("\nSTATUS COUNTS:")
    for k, v in by_status.most_common():
        print(f"  {k:>8}: {v}")

    if by_reason:
        print("\nTOP NON-SOLVED REASONS:")
        for r, c in by_reason.most_common(30):
            ex = ", ".join(ex_by_reason[r][:5])
            print(f"  {c:>5}  {r}  | ex: {ex}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

