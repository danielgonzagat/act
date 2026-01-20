#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_v93 import verify_chained_jsonl_v93, verify_conversation_chain_v93


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _extract_payloads(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if isinstance(payload, dict):
            out.append(dict(payload))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = str(args.run_dir)
    turns_path = os.path.join(run_dir, "conversation_turns.jsonl")
    parses_path = os.path.join(run_dir, "intent_parses.jsonl")
    learned_path = os.path.join(run_dir, "learned_intent_rules.jsonl")
    states_path = os.path.join(run_dir, "conversation_states.jsonl")
    trials_path = os.path.join(run_dir, "dialogue_trials.jsonl")

    rows_turns = _read_jsonl(turns_path)
    rows_parses = _read_jsonl(parses_path)
    rows_learned = _read_jsonl(learned_path) if os.path.exists(learned_path) else []
    rows_states = _read_jsonl(states_path) if os.path.exists(states_path) else []
    rows_trials = _read_jsonl(trials_path)

    turns = _extract_payloads(rows_turns)
    states = _extract_payloads(rows_states)

    parse_events: List[Dict[str, Any]] = []
    for r in rows_parses:
        if not isinstance(r, dict):
            continue
        parse_events.append(
            {
                "turn_id": str(r.get("turn_id") or ""),
                "turn_index": int(r.get("turn_index") or 0),
                "payload": dict(r.get("payload") or {}) if isinstance(r.get("payload"), dict) else {},
            }
        )

    trials: List[Dict[str, Any]] = []
    for r in rows_trials:
        if not isinstance(r, dict):
            continue
        trials.append(
            {
                "objective_kind": str(r.get("objective_kind") or ""),
                "user_turn_id": str(r.get("user_turn_id") or r.get("turn_id") or ""),
            }
        )

    learned_events: List[Dict[str, Any]] = []
    for r in rows_learned:
        if not isinstance(r, dict):
            continue
        learned_events.append(dict(r))

    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v93(turns_path)),
        "parses_chain_ok": bool(verify_chained_jsonl_v93(parses_path)),
        "learned_chain_ok": bool(verify_chained_jsonl_v93(learned_path)) if os.path.exists(learned_path) else True,
        "states_chain_ok": bool(verify_chained_jsonl_v93(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v93(trials_path)),
    }

    ok_inv, reason, details = verify_conversation_chain_v93(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        tail_k=6,
    )

    out = {
        "ok": bool(all(chains.values())) and bool(ok_inv),
        "chains": dict(chains),
        "invariants": {"ok": bool(ok_inv), "reason": str(reason), "details": dict(details)},
        "counts": {
            "turns_total": int(len(turns)),
            "states_total": int(len(states)),
            "parses_total": int(len(parse_events)),
            "learned_total": int(len(learned_events)),
            "trials_total": int(len(trials)),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

