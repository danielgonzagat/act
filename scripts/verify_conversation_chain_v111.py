#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_v100 import no_hybridization_check_v100
from atos_core.conversation_v96 import verify_chained_jsonl_v96
from atos_core.external_world_ledger_v111 import compute_external_world_chain_hash_v111, verify_external_world_event_sig_chain_v111

# Reuse the v110 verifier script loader (scripts/ is on sys.path when invoked directly).
import verify_conversation_chain_v110 as verify_v110


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.exists():
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _infer_tail_k(states: List[Dict[str, Any]]) -> int:
    tail_k = 6
    for st in states[-3:]:
        if not isinstance(st, dict):
            continue
        inv = st.get("invariants") if isinstance(st.get("invariants"), dict) else {}
        try:
            tail_k = int(inv.get("tail_k") or tail_k)
        except Exception:
            pass
    return int(tail_k)


def _verify_task_dir(task_dir: Path, repo_root: str) -> Tuple[bool, str, Dict[str, Any]]:
    required = [
        task_dir / "conversation_turns.jsonl",
        task_dir / "conversation_states.jsonl",
        task_dir / "intent_parses.jsonl",
        task_dir / "dialogue_trials.jsonl",
        task_dir / "action_plans.jsonl",
        task_dir / "memory_events.jsonl",
        task_dir / "belief_events.jsonl",
        task_dir / "evidence_events.jsonl",
        task_dir / "goal_events.jsonl",
        task_dir / "goal_ledger_snapshot.json",
        task_dir / "discourse_events.jsonl",
        task_dir / "fragment_events.jsonl",
        task_dir / "transcript.jsonl",
        task_dir / "fluency_contract_v111.json",
        task_dir / "external_world_events.jsonl",
        task_dir / "external_world_registry_snapshot_v111.json",
    ]
    for p in required:
        if not p.exists():
            return False, "missing_path", {"path": str(p)}

    ok0, reason0, details0 = verify_v110.verify_run_dir_v110(run_dir=str(task_dir))
    if not ok0:
        return False, str(reason0), dict(details0)

    # External world events ledger must be hash-chained (file) and sig-chained (internal).
    ext_path = task_dir / "external_world_events.jsonl"
    ok_file_chain = bool(verify_chained_jsonl_v96(str(ext_path)))
    ext_events = _load_jsonl(ext_path)
    ok_sig_chain, reason_sig_chain, details_sig_chain = verify_external_world_event_sig_chain_v111(ext_events)
    ext_chain_hash = compute_external_world_chain_hash_v111(ext_events)
    if not ok_file_chain:
        return False, "external_world_file_chain_invalid", {}
    if not ok_sig_chain:
        return False, str(reason_sig_chain), dict(details_sig_chain)

    fc = _load_json(task_dir / "fluency_contract_v111.json")
    if not bool(fc.get("ok", False)):
        return False, "fluency_contract_failed", {"reason": str(fc.get("reason") or "")}

    return True, "ok", {"external_world_chain_hash_v111": str(ext_chain_hash), "external_world_events_total": int(len(ext_events))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = Path(str(args.run_dir))
    if not run_dir.is_dir():
        _fail(f"missing_run_dir:{run_dir}")

    repo_root = str(Path(__file__).resolve().parents[1])
    ok_h, reason_h, details_h = no_hybridization_check_v100(repo_root=repo_root)
    if not ok_h:
        _fail(str(reason_h) + ":" + json.dumps(details_h, ensure_ascii=False, sort_keys=True))

    task_dirs = sorted([p for p in run_dir.iterdir() if p.is_dir() and p.name.startswith("task_")], key=lambda p: p.name)
    if not task_dirs:
        _fail("no_task_dirs")

    results: List[Dict[str, Any]] = []
    for td in task_dirs:
        ok, reason, details = _verify_task_dir(td, repo_root=repo_root)
        results.append({"task_dir": str(td), "ok": bool(ok), "reason": str(reason), "details": dict(details)})
        if not ok:
            _fail("task_verify_failed:" + str(reason))

    out = {"ok": True, "reason": "ok", "tasks_total": int(len(task_dirs)), "results": results}
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
