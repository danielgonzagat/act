#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v123 import run_conversation_v123


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


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        raise SystemExit(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--max_tasks", default=9999, type=int)
    ap.add_argument("--max_plan_attempts", default=8, type=int)
    ap.add_argument("--max_replans_per_turn", default=3, type=int)
    ap.add_argument("--goal_autopilot_total_steps", default=60, type=int)
    args = ap.parse_args()

    tasks_path = Path(str(args.tasks))
    out_dir = Path(str(args.out))
    seed = int(args.seed)

    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    tasks: List[Dict[str, Any]] = []
    with open(tasks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    if not tasks:
        raise SystemExit("tasks_empty")

    tasks_total = min(int(len(tasks)), int(args.max_tasks))
    results: List[Dict[str, Any]] = []

    for i in range(tasks_total):
        task = tasks[i]
        task_id = str(task.get("task_id") or f"task_{i:03d}")
        user_turns = task.get("user_turns") if isinstance(task.get("user_turns"), list) else []
        user_turn_texts = [str(x) for x in user_turns if isinstance(x, str)]
        world_manifest = str(task.get("world_manifest") or "external_world_v122/manifest_v122.json")

        task_dir = out_dir / "task_{i:03d}".format(i=i)
        _ensure_absent(task_dir)
        task_dir.mkdir(parents=True, exist_ok=False)

        attempt_dir = task_dir / "attempt_000"
        _ensure_absent(attempt_dir)

        run_conversation_v123(
            user_turn_texts=list(user_turn_texts),
            out_dir=str(attempt_dir),
            seed=int(seed) + int(i),
            max_plan_attempts=int(args.max_plan_attempts),
            max_replans_per_turn=int(args.max_replans_per_turn),
            goal_autopilot_total_steps=int(args.goal_autopilot_total_steps),
            external_world_manifest_path=str(world_manifest),
        )

        fr = _load_json(attempt_dir / "final_response_v123.json")
        ok = bool(fr.get("ok", False)) if isinstance(fr, dict) else False
        reason = str(fr.get("reason") or "") if isinstance(fr, dict) else "missing_final_response_v123"
        results.append({"task_id": str(task_id), "ok": bool(ok), "reason": str(reason)})

    tasks_ok = sum([1 for r in results if bool(r.get("ok"))])

    eval_obj = {
        "schema_version": 123,
        "kind": "family7_dla_eval_v123",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_ok": int(tasks_ok),
        "tasks": list(results),
        "tasks_sha256": _sha256_file(tasks_path),
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    eval_sha256 = _sha256_file(out_dir / "eval.json")
    summary = {
        "schema_version": 123,
        "kind": "family7_dla_summary_v123",
        "seed": int(seed),
        "tasks_total": int(tasks_total),
        "tasks_ok": int(tasks_ok),
        "eval_sha256": str(eval_sha256),
    }
    summary["summary_sig"] = sha256_hex(canonical_json_dumps(summary).encode("utf-8"))
    _write_once_json(out_dir / "summary.json", summary)

    if int(tasks_ok) != int(tasks_total):
        raise SystemExit("tasks_not_all_ok")


if __name__ == "__main__":
    main()
