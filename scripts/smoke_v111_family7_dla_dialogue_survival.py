#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _run_runner(*, tasks: str, out_dir: Path, seed: int) -> None:
    _ensure_absent(out_dir)
    cmd = [
        sys.executable,
        "scripts/run_family7_dla_v111.py",
        "--tasks",
        str(tasks),
        "--out",
        str(out_dir),
        "--seed",
        str(int(seed)),
        "--max_tasks",
        "2",
    ]
    subprocess.check_call(cmd)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    tasks = str(args.tasks)
    seed = int(args.seed)
    out_base = Path(str(args.out_base))

    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")

    _run_runner(tasks=tasks, out_dir=out1, seed=seed)
    _run_runner(tasks=tasks, out_dir=out2, seed=seed)

    s1 = _load_json(out1 / "summary.json")
    s2 = _load_json(out2 / "summary.json")
    e1 = _load_json(out1 / "eval.json")
    e2 = _load_json(out2 / "eval.json")

    determinism_ok = (canonical_json_dumps(s1) == canonical_json_dumps(s2)) and (canonical_json_dumps(e1) == canonical_json_dumps(e2))

    # Check per-task WORM external world ledger is unused (0 bytes) and exists.
    ext_ok = True
    ext_details: List[Dict[str, Any]] = []
    for run_dir in [out1, out2]:
        for task_dir in sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("task_")], key=lambda p: p.name):
            p = task_dir / "external_world_events.jsonl"
            if not p.exists():
                ext_ok = False
                ext_details.append({"task_dir": str(task_dir), "missing": str(p)})
                continue
            if p.stat().st_size != 0:
                ext_ok = False
                ext_details.append({"task_dir": str(task_dir), "size": int(p.stat().st_size)})

    core = {
        "seed": int(seed),
        "try1": {"summary_sha256": _sha256_file(out1 / "summary.json"), "eval_sha256": _sha256_file(out1 / "eval.json"), "tasks_ok": int(s1.get("tasks_ok") or 0)},
        "try2": {"summary_sha256": _sha256_file(out2 / "summary.json"), "eval_sha256": _sha256_file(out2 / "eval.json"), "tasks_ok": int(s2.get("tasks_ok") or 0)},
        "external_world_unused_ok": bool(ext_ok),
    }
    summary_sha = sha256_hex(canonical_json_dumps(core).encode("utf-8"))

    out = {
        "ok": bool(determinism_ok and ext_ok and int(s1.get("tasks_ok") or 0) == int(s1.get("tasks_total") or 0)),
        "determinism_ok": bool(determinism_ok),
        "summary_sha256": str(summary_sha),
        "core": core,
        "try1_dir": str(out1),
        "try2_dir": str(out2),
        "external_world_details": ext_details,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
