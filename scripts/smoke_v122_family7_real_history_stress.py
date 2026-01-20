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
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _ensure_absent(path: Path) -> None:
    if path.exists():
        raise SystemExit(f"worm_exists:{path}")


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_runner(*, tasks: str, out_dir: Path, seed: int) -> None:
    _ensure_absent(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    cmd = [
        sys.executable,
        "scripts/run_family7_dla_v121.py",
        "--tasks",
        str(tasks),
        "--out",
        str(out_dir),
        "--seed",
        str(seed),
        "--max_tasks",
        "9999",
    ]
    p = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("runner_failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}".format(out=p.stdout, err=p.stderr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    args = ap.parse_args()

    seed = int(args.seed)
    tasks_path = str(args.tasks)
    out_base = Path(str(args.out_base))

    tasks: List[Dict[str, Any]] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if len(tasks) < 20:
        raise SystemExit("tasks_total_lt_20")

    out1 = Path(str(out_base) + "_try1")
    out2 = Path(str(out_base) + "_try2")
    _run_runner(tasks=tasks_path, out_dir=out1, seed=seed)
    _run_runner(tasks=tasks_path, out_dir=out2, seed=seed)

    s1 = _load_json(out1 / "summary.json")
    s2 = _load_json(out2 / "summary.json")
    eval_sha1 = str(s1.get("eval_sha256") or "")
    eval_sha2 = str(s2.get("eval_sha256") or "")
    if eval_sha1 != eval_sha2:
        raise SystemExit("determinism_failed:eval_sha")

    ev1 = _load_json(out1 / "eval.json")
    ev2 = _load_json(out2 / "eval.json")
    if canonical_json_dumps(ev1) != canonical_json_dumps(ev2):
        raise SystemExit("determinism_failed:eval_json")
    if int(ev1.get("tasks_ok") or 0) != int(ev1.get("tasks_total") or 0):
        raise SystemExit("tasks_not_all_ok")

    core = {
        "schema_version": 122,
        "seed": int(seed),
        "tasks_sha256": _sha256_file(Path(tasks_path)),
        "try1": {"eval_sha256": eval_sha1, "tasks_ok": int(s1.get("tasks_ok") or 0)},
        "try2": {"eval_sha256": eval_sha2, "tasks_ok": int(s2.get("tasks_ok") or 0)},
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    out = {
        "ok": True,
        "determinism_ok": True,
        "summary_sha256": str(summary_sha256),
        "core": core,
        "try1_dir": str(out1),
        "try2_dir": str(out2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

