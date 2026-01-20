#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v119 import run_conversation_v119


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


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
        _fail(f"worm_exists:{path}")


ACK_LIKE = {
    "ok",
    "okay",
    "certo",
    "beleza",
    "blz",
    "continua",
    "continue",
    "segue",
    "vai",
    "faz",
    "pode",
    "sim",
    "isso",
    "aquilo",
    "that",
    "it",
}


def _canon_ack(s: str) -> str:
    t = str(s or "").strip().lower()
    t = " ".join([x for x in t.split() if x])
    return t


def _ack_ratio(task: Dict[str, Any]) -> Tuple[float, int]:
    turns = task.get("user_turns") if isinstance(task.get("user_turns"), list) else []
    total = int(len(turns))
    if total <= 0:
        return 0.0, 0
    ack = 0
    for t in turns:
        if _canon_ack(str(t)) in ACK_LIKE:
            ack += 1
    ratio = float(ack) / float(total) if total else 0.0
    return float(ratio), int(ack)


def _load_tasks(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        _fail(f"missing_tasks:{path}")
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(dict(obj))
    return out


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _smoke_once(*, out_dir: Path, tasks_path: Path, seed: int, top_n: int) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    out_dir.mkdir(parents=True, exist_ok=False)

    tasks_all = _load_tasks(tasks_path)
    scored: List[Tuple[float, int, str, Dict[str, Any]]] = []
    for t in tasks_all:
        tid = str(t.get("task_id") or "")
        ratio, ack = _ack_ratio(t)
        scored.append((float(ratio), int(ack), str(tid), dict(t)))
    # Hard gate uses the "most ack-like" tasks to stress consecutive_prefix2_repeat deterministically.
    scored.sort(key=lambda x: (-float(x[0]), -int(x[1]), str(x[2])))
    picked = [t for _, _, _, t in scored[: int(top_n)]]
    if not picked:
        _fail("no_tasks_picked")

    tasks_ok = 0
    task_rows: List[Dict[str, Any]] = []
    for i, t in enumerate(picked):
        tid = str(t.get("task_id") or "")
        turns = t.get("user_turns") if isinstance(t.get("user_turns"), list) else []
        task_dir = out_dir / f"task_{i:03d}_{tid[:12]}"
        res = run_conversation_v119(user_turn_texts=[str(x) for x in turns], out_dir=str(task_dir), seed=int(seed))
        fr = res.get("final_response_v119") if isinstance(res, dict) else {}
        ok = bool(fr.get("ok", False)) if isinstance(fr, dict) else False
        reason = str(fr.get("reason") or "") if isinstance(fr, dict) else "missing_final_response_v119"
        if ok:
            tasks_ok += 1
        ratio, ack = _ack_ratio(t)
        task_rows.append(
            {
                "task_id": str(tid),
                "stress_turns": int(t.get("stress_turns", 0) or 0),
                "ack_ratio": float(round(float(ratio), 6)),
                "ack_count": int(ack),
                "ok": bool(ok),
                "reason": str(reason if not ok else "ok"),
            }
        )

    eval_obj = {
        "schema_version": 119,
        "kind": "family7_hard_gate_eval_v119",
        "seed": int(seed),
        "tasks_total": int(len(picked)),
        "tasks_ok": int(tasks_ok),
        "picked_by": "ack_ratio_desc",
        "tasks": list(task_rows),
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    core = {
        "seed": int(seed),
        "tasks_total": int(len(picked)),
        "tasks_ok": int(tasks_ok),
        "eval_sha256": _sha256_file(out_dir / "eval.json"),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    summary = {"schema_version": 119, "kind": "family7_hard_gate_summary_v119", "core": dict(core), "summary_sha256": str(summary_sha256)}
    _write_once_json(out_dir / "summary.json", summary)

    if int(tasks_ok) != int(len(picked)):
        _fail(f"hard_gate_failed:{tasks_ok}/{len(picked)}")
    return {"core": dict(core), "summary_sha256": str(summary_sha256)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--top_n", type=int, default=5)
    args = ap.parse_args()

    tasks_path = Path(str(args.tasks))
    out_base = str(args.out_base)
    seed = int(args.seed)
    top_n = int(args.top_n)

    r1 = _smoke_once(out_dir=Path(out_base + "_try1"), tasks_path=tasks_path, seed=seed, top_n=top_n)
    r2 = _smoke_once(out_dir=Path(out_base + "_try2"), tasks_path=tasks_path, seed=seed, top_n=top_n)

    if canonical_json_dumps(r1["core"]) != canonical_json_dumps(r2["core"]):
        _fail("determinism_core_mismatch")
    if str(r1["summary_sha256"]) != str(r2["summary_sha256"]):
        _fail("determinism_summary_sha256_mismatch")

    print(
        json.dumps(
            {
                "ok": True,
                "determinism_ok": True,
                "summary_sha256": str(r1["summary_sha256"]),
                "out_try1": str(out_base + "_try1"),
                "out_try2": str(out_base + "_try2"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

