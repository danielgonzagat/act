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
from atos_core.external_world_gating_v113 import external_world_access_v113
from atos_core.external_world_ledger_v111 import EXTERNAL_WORLD_ACTION_SEARCH_V111


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
        "scripts/run_family7_dla_v114.py",
        "--tasks",
        str(tasks),
        "--out",
        str(out_dir),
        "--seed",
        str(seed),
        "--max_tasks",
        "9999",
        "--max_rewrites",
        "4",
        "--max_replans_per_turn",
        "3",
    ]
    p = subprocess.run(cmd, env=env, cwd=str(Path(__file__).resolve().parent.parent), capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("runner_failed:\nSTDOUT:\n{out}\nSTDERR:\n{err}".format(out=p.stdout, err=p.stderr))


def _negative_tests(*, world_manifest: str) -> Dict[str, Any]:
    ok1 = False
    reason1 = ""
    try:
        external_world_access_v113(
            allowed=False,
            world_manifest=str(world_manifest),
            action=EXTERNAL_WORLD_ACTION_SEARCH_V111,
            reason_code="validator_failed_fluency_contract",
            args={"query": "x", "limit": 1, "roles": ["user"]},
            seed=0,
            turn_index=0,
            prev_event_sig="",
        )
        ok1 = True
    except ValueError as e:
        reason1 = str(e)

    ok2 = False
    reason2 = ""
    try:
        external_world_access_v113(
            allowed=True,
            world_manifest=str(world_manifest),
            action=EXTERNAL_WORLD_ACTION_SEARCH_V111,
            reason_code="invalid_reason_code_x",
            args={"query": "x", "limit": 1, "roles": ["user"]},
            seed=0,
            turn_index=0,
            prev_event_sig="",
        )
        ok2 = True
    except ValueError as e:
        reason2 = str(e)

    return {
        "access_not_allowed": {"ok": bool(ok1), "reason": str(reason1)},
        "invalid_reason_code": {"ok": bool(ok2), "reason": str(reason2)},
    }


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
    if not tasks:
        raise SystemExit("empty_tasks")
    world_manifest = str(tasks[0].get("world_manifest") or "")

    neg = _negative_tests(world_manifest=world_manifest)
    if neg["access_not_allowed"]["reason"] != "external_world_access_not_allowed":
        raise SystemExit("negative_failed:access_not_allowed")
    if neg["invalid_reason_code"]["reason"] != "invalid_reason_code":
        raise SystemExit("negative_failed:invalid_reason_code")

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

    res1 = ev1.get("results") if isinstance(ev1.get("results"), list) else []
    ext_counts = [int(r.get("external_world_events_total") or 0) for r in res1 if isinstance(r, dict)]
    if sum(1 for c in ext_counts if c == 1) != 1:
        raise SystemExit("external_world_in_cycle_expected_one_call")

    core = {
        "schema_version": 114,
        "seed": int(seed),
        "try1": {"eval_sha256": eval_sha1, "tasks_ok": int(s1.get("tasks_ok") or 0)},
        "try2": {"eval_sha256": eval_sha2, "tasks_ok": int(s2.get("tasks_ok") or 0)},
        "negative_tests": dict(neg),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    out = {
        "ok": True,
        "determinism_ok": True,
        "summary_sha256": str(summary_sha256),
        "core": core,
        "try1_dir": str(out1),
        "try2_dir": str(out2),
        "sha256_eval_json": _sha256_file(out1 / "eval.json"),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
