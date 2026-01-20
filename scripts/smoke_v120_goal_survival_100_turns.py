#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v120 import run_conversation_v120
from atos_core.goal_persistence_law_v120 import (
    FAIL_REASON_GOAL_DONE_BEFORE_HORIZON_V120,
    verify_goal_persistence_law_v120,
)


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _ensure_absent(path: Path) -> None:
    if path.exists():
        _fail(f"worm_exists:{path}")


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _write_once_json(path: Path, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    if tmp.exists():
        _fail(f"tmp_exists:{tmp}")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(tmp), str(path))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(json.loads(line))
    return out


def _payload_view(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if isinstance(payload, dict):
            out.append(dict(payload))
    return out


def _find_goal_done_system(goal_events_payloads: List[Dict[str, Any]]) -> Tuple[str, int]:
    for ev in goal_events_payloads:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("op") or "") != "GOAL_DONE":
            continue
        cause = ev.get("cause") if isinstance(ev.get("cause"), dict) else {}
        if str(cause.get("cause_type") or "") != "system":
            continue
        gid = str(ev.get("goal_id") or "")
        try:
            ts = int(ev.get("ts_turn_index", -1))
        except Exception:
            ts = -1
        if gid and ts >= 0:
            return str(gid), int(ts)
    return "", -1


def _smoke_once(*, out_dir: Path, seed: int, total_steps: int) -> Dict[str, Any]:
    _ensure_absent(out_dir)
    # Pre-fill required goal slots deterministically so V110 autopilot can advance without over-clarifying.
    user_turns = ["goal: preparar demo; outcome=ok; constraints=none; deadline=none"] + ["ok"] * int(total_steps) + ["end now"]
    run_conversation_v120(
        user_turn_texts=list(user_turns),
        out_dir=str(out_dir),
        seed=int(seed),
        goal_autopilot_total_steps=int(total_steps),
    )

    gps_path = out_dir / "goal_persistence_summary_v120.json"
    gps = _read_json(gps_path)
    if not isinstance(gps, dict) or not bool(gps.get("ok", False)):
        _fail(f"goal_persistence_fail:{(gps.get('reason') if isinstance(gps, dict) else 'missing')}")

    goal_rows = _read_jsonl(out_dir / "goal_events.jsonl")
    goal_payloads = _payload_view(goal_rows)
    done_goal_id, done_ts = _find_goal_done_system(goal_payloads)
    if not done_goal_id or done_ts < 0:
        _fail("missing_system_goal_done_event")

    checked = gps.get("system_goal_done_checked") if isinstance(gps.get("system_goal_done_checked"), list) else []
    checked0 = checked[0] if checked and isinstance(checked[0], dict) else {}
    step = int(checked0.get("step", 0) or 0)
    total = int(checked0.get("total", 0) or 0)
    if int(total) != int(total_steps) or int(step) != int(total_steps):
        _fail("goal_done_proof_mismatch")

    eval_obj = {
        "schema_version": 120,
        "kind": "smoke_v120_goal_survival_100_turns_eval",
        "seed": int(seed),
        "goal_autopilot_total_steps": int(total_steps),
        "goal_done": {"goal_id": str(done_goal_id), "ts_turn_index": int(done_ts), "proof": {"step": int(step), "total": int(total)}},
        "goal_persistence_ok": True,
        "final_ok": bool((_read_json(out_dir / "final_response_v120.json") or {}).get("ok", False)),
    }
    eval_obj["eval_sig"] = sha256_hex(canonical_json_dumps(eval_obj).encode("utf-8"))
    _write_once_json(out_dir / "eval.json", eval_obj)

    core = {
        "seed": int(seed),
        "goal_autopilot_total_steps": int(total_steps),
        "goal_done": {"goal_id": str(done_goal_id), "ts_turn_index": int(done_ts), "proof": {"step": int(step), "total": int(total)}},
        "eval_sha256": _sha256_file(out_dir / "eval.json"),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    summary = {"schema_version": 120, "kind": "smoke_v120_goal_survival_100_turns_summary", "core": dict(core), "summary_sha256": str(summary_sha256)}
    _write_once_json(out_dir / "smoke_summary.json", summary)

    return {"core": dict(core), "summary_sha256": str(summary_sha256)}


def _tamper_progress_proof(*, src_dir: Path, dst_dir: Path, total_steps: int) -> None:
    _ensure_absent(dst_dir)
    shutil.copytree(str(src_dir), str(dst_dir), dirs_exist_ok=False)
    turns_path = dst_dir / "conversation_turns.jsonl"
    rows = _read_jsonl(turns_path)
    payloads = _payload_view(rows)

    # Find the assistant turn that corresponds to the system GOAL_DONE event (ts_turn_index+1),
    # and corrupt its "<step>/<total>" marker (without touching chains elsewhere).
    goal_rows = _read_jsonl(dst_dir / "goal_events.jsonl")
    goal_payloads = _payload_view(goal_rows)
    _gid, done_ts = _find_goal_done_system(goal_payloads)
    if int(done_ts) < 0:
        _fail("tamper_missing_system_goal_done_event")
    target_turn_index = int(done_ts) + 1
    target_assistant = None
    for p in payloads:
        if not isinstance(p, dict) or str(p.get("role") or "") != "assistant":
            continue
        try:
            ti = int(p.get("turn_index", -1))
        except Exception:
            ti = -1
        if int(ti) == int(target_turn_index):
            target_assistant = p
            break
    if not isinstance(target_assistant, dict):
        _fail("tamper_missing_goal_done_assistant_turn")

    txt = str(target_assistant.get("text") or "")
    want = f"{int(total_steps)}/{int(total_steps)}"
    got = f"{int(total_steps - 1)}/{int(total_steps)}"
    if want not in txt:
        _fail("tamper_missing_progress_marker")
    target_assistant["text"] = txt.replace(want, got, 1)
    # Rewrite conversation_turns.jsonl WITHOUT trying to preserve its chain; this is only for verifier negative coverage.
    # We keep it simple: the goal persistence verifier uses this file only as a text source.
    out_lines: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if isinstance(payload, dict) and str(payload.get("turn_id") or "") == str(target_assistant.get("turn_id") or ""):
            r2 = dict(r)
            r2["payload"] = dict(target_assistant)
            out_lines.append(json.dumps(r2, ensure_ascii=False, sort_keys=True))
        else:
            out_lines.append(json.dumps(r, ensure_ascii=False, sort_keys=True))
    turns_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--total_steps", type=int, default=100)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)
    total_steps = int(args.total_steps)

    r1 = _smoke_once(out_dir=Path(out_base + "_try1"), seed=seed, total_steps=total_steps)
    r2 = _smoke_once(out_dir=Path(out_base + "_try2"), seed=seed, total_steps=total_steps)

    if canonical_json_dumps(r1["core"]) != canonical_json_dumps(r2["core"]):
        _fail("determinism_core_mismatch")
    if str(r1["summary_sha256"]) != str(r2["summary_sha256"]):
        _fail("determinism_summary_sha256_mismatch")

    # Negative tamper: corrupt progress proof and require the goal persistence verifier to fail with a stable reason.
    tamper_dir = Path(out_base + "_try1_tamper")
    _tamper_progress_proof(src_dir=Path(out_base + "_try1"), dst_dir=tamper_dir, total_steps=total_steps)
    res = verify_goal_persistence_law_v120(run_dir=str(tamper_dir), expected_autopilot_total_steps=int(total_steps))
    if bool(res.ok):
        _fail("tamper_expected_fail_but_ok")
    if str(res.reason) != str(FAIL_REASON_GOAL_DONE_BEFORE_HORIZON_V120):
        _fail(f"tamper_reason_mismatch:{res.reason}")

    print(
        json.dumps(
            {
                "ok": True,
                "determinism_ok": True,
                "summary_sha256": str(r1["summary_sha256"]),
                "out_try1": str(out_base + "_try1"),
                "out_try2": str(out_base + "_try2"),
                "tamper_reason": str(res.reason),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
