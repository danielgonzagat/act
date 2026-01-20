#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(2)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def _read_summary(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict) or "summary" not in obj or not isinstance(obj.get("summary"), dict):
        _fail("FAIL: bad summary.json schema")
    return obj


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--out", required=True, help="WORM out dir (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit_tasks", type=int, default=10)
    ap.add_argument("--min_ok", type=int, default=8)
    args = ap.parse_args()

    ensure_absent(args.out)
    os.makedirs(args.out, exist_ok=False)

    # (1) Determinism: run agent_loop_v64 twice with identical args and compare hashes.
    run_a = os.path.join(args.out, "run_a")
    run_b = os.path.join(args.out, "run_b")
    ensure_absent(run_a)
    ensure_absent(run_b)

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "agent_loop_v64.py"),
        "--acts_run",
        str(args.acts_run),
        "--seed",
        str(int(args.seed)),
        "--planner_max_depth",
        "4",
        "--planner_max_expansions",
        "5000",
        "--limit_tasks",
        str(int(args.limit_tasks)),
        "--max_events_per_step",
        "128",
    ]
    subprocess.run(cmd + ["--out", run_a], check=True)
    subprocess.run(cmd + ["--out", run_b], check=True)

    sum_a = os.path.join(run_a, "summary.json")
    sum_b = os.path.join(run_b, "summary.json")
    trace_a = os.path.join(run_a, "traces", "agent_trace_v64.jsonl")
    trace_b = os.path.join(run_b, "traces", "agent_trace_v64.jsonl")
    if not (os.path.exists(sum_a) and os.path.exists(sum_b) and os.path.exists(trace_a) and os.path.exists(trace_b)):
        _fail("FAIL: missing expected artifacts from agent_loop_v64")

    h_sum_a = sha256_file(sum_a)
    h_sum_b = sha256_file(sum_b)
    h_trace_a = sha256_file(trace_a)
    h_trace_b = sha256_file(trace_b)
    if h_sum_a != h_sum_b or h_trace_a != h_trace_b:
        _fail("FAIL: determinism check failed (hash mismatch)")

    js_a = _read_summary(sum_a)
    js_b = _read_summary(sum_b)
    sa = js_a["summary"]
    sb = js_b["summary"]
    if int(sa.get("tasks_total", 0) or 0) != int(args.limit_tasks):
        _fail("FAIL: unexpected tasks_total in run_a")
    if int(sb.get("tasks_total", 0) or 0) != int(args.limit_tasks):
        _fail("FAIL: unexpected tasks_total in run_b")

    ok_a = int(sa.get("tasks_ok", 0) or 0)
    ok_b = int(sb.get("tasks_ok", 0) or 0)
    if ok_a < int(args.min_ok) or ok_b < int(args.min_ok):
        _fail(f"FAIL: planner success too low: ok_a={ok_a} ok_b={ok_b} min_ok={int(args.min_ok)}")
    if int(sa.get("uncertainty_ic_count", 0) or 0) != 0:
        _fail("FAIL: uncertainty_ic_count must be 0")

    out: Dict[str, Any] = {
        "ok": True,
        "seed": int(args.seed),
        "limit_tasks": int(args.limit_tasks),
        "min_ok": int(args.min_ok),
        "determinism": {"summary_sha256": h_sum_a, "trace_sha256": h_trace_a},
        "run_a": {"tasks_ok": ok_a, "pass_rate": float(sa.get("pass_rate", 0.0) or 0.0)},
        "run_b": {"tasks_ok": ok_b, "pass_rate": float(sb.get("pass_rate", 0.0) or 0.0)},
    }

    smoke_path = os.path.join(args.out, "smoke_summary.json")
    ensure_absent(smoke_path)
    with open(smoke_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))

    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

