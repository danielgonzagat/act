#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v101 import run_conversation_v101
from atos_core.conversation_v101 import verify_conversation_chain_v101


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists (WORM): {path}")


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _extract_payloads(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        payload = r.get("payload")
        if isinstance(payload, dict):
            out.append(dict(payload))
    return out


def _extract_parse_events(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "turn_id": str(r.get("turn_id") or ""),
                "turn_index": int(r.get("turn_index") or 0),
                "payload": dict(r.get("payload") or {}) if isinstance(r.get("payload"), dict) else {},
            }
        )
    return out


def _extract_trials_for_verify(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        out.append({"objective_kind": str(r.get("objective_kind") or ""), "user_turn_id": str(r.get("user_turn_id") or r.get("turn_id") or "")})
    return out


def _extract_learned_events(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        d = dict(r)
        d.pop("prev_hash", None)
        d.pop("entry_hash", None)
        out.append(d)
    return out


def _strip_chain_fields(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        d = dict(r)
        d.pop("prev_hash", None)
        d.pop("entry_hash", None)
        out.append(d)
    return out


def _load_for_verify(res: Dict[str, Any]) -> Dict[str, Any]:
    paths = res.get("paths") if isinstance(res.get("paths"), dict) else {}
    turns = _extract_payloads(str(paths.get("turns_jsonl") or ""))
    states_path = str(paths.get("states_jsonl") or "")
    states = _extract_payloads(states_path) if states_path and os.path.exists(states_path) else []
    parse_events = _extract_parse_events(str(paths.get("parses_jsonl") or ""))
    trials = _extract_trials_for_verify(str(paths.get("trials_jsonl") or ""))
    learned_events = _extract_learned_events(str(paths.get("learned_rules_jsonl") or ""))
    plans = _strip_chain_fields(str(paths.get("plans_jsonl") or ""))
    memory_events = _strip_chain_fields(str(paths.get("memory_events_jsonl") or ""))
    belief_events = _strip_chain_fields(str(paths.get("belief_events_jsonl") or ""))
    evidence_events = _strip_chain_fields(str(paths.get("evidence_events_jsonl") or ""))
    goal_events = _extract_payloads(str(paths.get("goal_events_jsonl") or ""))
    goal_snapshot = _read_json(str(paths.get("goal_ledger_snapshot_json") or ""))
    discourse_events = _extract_payloads(str(paths.get("discourse_events_jsonl") or ""))
    fragment_events = _extract_payloads(str(paths.get("fragment_events_jsonl") or ""))
    binding_events = _strip_chain_fields(str(paths.get("binding_events_jsonl") or ""))
    binding_snapshot = _read_json(str(paths.get("binding_snapshot_json") or ""))
    return {
        "turns": turns,
        "states": states,
        "parse_events": parse_events,
        "trials": trials,
        "learned_events": learned_events,
        "plans": plans,
        "memory_events": memory_events,
        "belief_events": belief_events,
        "evidence_events": evidence_events,
        "goal_events": goal_events,
        "goal_snapshot": goal_snapshot,
        "discourse_events": discourse_events,
        "fragment_events": fragment_events,
        "binding_events": binding_events,
        "binding_snapshot": binding_snapshot,
    }


def _count_binding_types(binding_events: List[Dict[str, Any]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for ev in binding_events:
        if not isinstance(ev, dict):
            continue
        et = str(ev.get("type") or "")
        out[et] = int(out.get(et, 0)) + 1
    return dict(out)


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    user_turns = [
        "crie um plano de 3 passos para x",
        "agora faz isso mais curto",
        "ok e isso e prioridade alta",
        "isso",
        "end now",
    ]
    res = run_conversation_v101(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    loaded = _load_for_verify(res)
    repo_root = os.path.dirname(os.path.dirname(__file__))

    ok_good, reason_good, details_good = verify_conversation_chain_v101(
        turns=loaded["turns"],
        states=loaded["states"],
        parse_events=loaded["parse_events"],
        trials=loaded["trials"],
        learned_rule_events=loaded["learned_events"],
        action_plans=loaded["plans"],
        memory_events=loaded["memory_events"],
        belief_events=loaded["belief_events"],
        evidence_events=loaded["evidence_events"],
        goal_events=loaded["goal_events"],
        goal_snapshot=loaded["goal_snapshot"],
        discourse_events=loaded["discourse_events"],
        fragment_events=loaded["fragment_events"],
        binding_events=loaded["binding_events"],
        binding_snapshot=loaded["binding_snapshot"],
        tail_k=6,
        repo_root=str(repo_root),
    )
    if not ok_good:
        _fail(f"ERROR: verify_conversation_chain_v101 failed: {reason_good} details={json.dumps(details_good, ensure_ascii=False, sort_keys=True)}")

    binding_types = _count_binding_types(loaded["binding_events"])
    if int(binding_types.get("BIND_RESOLVE", 0)) < 1:
        _fail("ERROR: expected at least 1 BIND_RESOLVE event")
    if int(binding_types.get("BIND_AMBIGUOUS", 0)) < 1:
        _fail("ERROR: expected at least 1 BIND_AMBIGUOUS event")

    # Negative tamper: mutate first binding event_sig and ensure verifier fails with deterministic reason.
    tampered_binding_events = [dict(ev) for ev in loaded["binding_events"]]
    if not tampered_binding_events:
        _fail("ERROR: expected non-empty binding_events for tamper test")
    tampered_binding_events[0]["event_sig"] = "0" * 64
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v101(
        turns=loaded["turns"],
        states=loaded["states"],
        parse_events=loaded["parse_events"],
        trials=loaded["trials"],
        learned_rule_events=loaded["learned_events"],
        action_plans=loaded["plans"],
        memory_events=loaded["memory_events"],
        belief_events=loaded["belief_events"],
        evidence_events=loaded["evidence_events"],
        goal_events=loaded["goal_events"],
        goal_snapshot=loaded["goal_snapshot"],
        discourse_events=loaded["discourse_events"],
        fragment_events=loaded["fragment_events"],
        binding_events=tampered_binding_events,
        binding_snapshot=loaded["binding_snapshot"],
        tail_k=6,
        repo_root=str(repo_root),
    )
    if ok_bad:
        _fail("ERROR: expected verifier to fail on tampered binding event_sig")
    if str(reason_bad) != "binding_event_sig_mismatch":
        _fail(f"ERROR: expected reason=binding_event_sig_mismatch got={reason_bad}")

    # Summarize deterministic core.
    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "transcript_hash": str(res.get("transcript_hash") or ""),
        "state_chain_hash": str(res.get("state_chain_hash") or ""),
        "parse_chain_hash": str(res.get("parse_chain_hash") or ""),
        "learned_chain_hash": str(res.get("learned_chain_hash") or ""),
        "plan_chain_hash": str(res.get("plan_chain_hash") or ""),
        "memory_chain_hash": str(res.get("memory_chain_hash") or ""),
        "belief_chain_hash": str(res.get("belief_chain_hash") or ""),
        "evidence_chain_hash": str(res.get("evidence_chain_hash") or ""),
        "goal_chain_hash": str(res.get("goal_chain_hash") or ""),
        "discourse_chain_hash": str(res.get("discourse_chain_hash") or ""),
        "fragment_chain_hash": str(res.get("fragment_chain_hash") or ""),
        "binding_chain_hash": str(res.get("binding_chain_hash") or ""),
        "ledger_hash": str(res.get("ledger_hash") or ""),
        "summary_sha256": str(res.get("summary_sha256") or ""),
    }

    return {
        "out_dir": str(out_dir),
        "core": dict(core),
        "binding_event_types": dict(binding_types),
        "tamper": {"ok": True, "reason": str(reason_bad)},
        "sha256": {
            "binding_events_jsonl": sha256_file(str(res["paths"]["binding_events_jsonl"])),  # type: ignore[index]
            "binding_snapshot_json": sha256_file(str(res["paths"]["binding_snapshot_json"])),  # type: ignore[index]
            "verify_chain_json": sha256_file(str(res["paths"]["verify_json"])),  # type: ignore[index]
            "manifest_json": sha256_file(str(res["paths"]["manifest_json"])),  # type: ignore[index]
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    out1 = out_base + "_try1"
    out2 = out_base + "_try2"
    ensure_absent(out1)
    ensure_absent(out2)

    r1 = smoke_try(out_dir=out1, seed=seed)
    r2 = smoke_try(out_dir=out2, seed=seed)

    if json.dumps(r1["core"], sort_keys=True) != json.dumps(r2["core"], sort_keys=True):
        _fail("ERROR: determinism mismatch: core differs between try1 and try2")

    out = {
        "ok": True,
        "determinism_ok": True,
        "determinism": {"core": r1["core"], "binding_chain_hash": r1["core"]["binding_chain_hash"]},
        "try1": dict(r1),
        "try2": dict(r2),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

