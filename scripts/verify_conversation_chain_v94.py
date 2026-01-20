#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_v94 import verify_chained_jsonl_v94, verify_conversation_chain_v94


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


def _extract_parse_events(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload") if isinstance(r.get("payload"), dict) else {}
        events.append(
            {
                "turn_id": str(r.get("turn_id") or ""),
                "turn_index": int(r.get("turn_index") or 0),
                "payload": dict(payload),
            }
        )
    return events


def _extract_trials_for_verify(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "objective_kind": str(r.get("objective_kind") or ""),
                "user_turn_id": str(r.get("user_turn_id") or r.get("turn_id") or ""),
            }
        )
    return out


def _extract_plans(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = dict(r)
        d.pop("prev_hash", None)
        d.pop("entry_hash", None)
        out.append(d)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    run_dir = str(args.run_dir)
    turns_path = os.path.join(run_dir, "conversation_turns.jsonl")
    parses_path = os.path.join(run_dir, "intent_parses.jsonl")
    learned_path = os.path.join(run_dir, "learned_intent_rules.jsonl")
    plans_path = os.path.join(run_dir, "action_plans.jsonl")
    states_path = os.path.join(run_dir, "conversation_states.jsonl")
    trials_path = os.path.join(run_dir, "dialogue_trials.jsonl")
    evals_path = os.path.join(run_dir, "objective_evals.jsonl")
    transcript_path = os.path.join(run_dir, "transcript.jsonl")

    rows_turns = _read_jsonl(turns_path)
    rows_parses = _read_jsonl(parses_path)
    rows_learned = _read_jsonl(learned_path) if os.path.exists(learned_path) else []
    rows_plans = _read_jsonl(plans_path)
    rows_states = _read_jsonl(states_path) if os.path.exists(states_path) else []
    rows_trials = _read_jsonl(trials_path)
    rows_evals = _read_jsonl(evals_path)
    rows_transcript = _read_jsonl(transcript_path)

    turns = _extract_payloads(rows_turns)
    states = _extract_payloads(rows_states)
    parse_events = _extract_parse_events(rows_parses)
    trials = _extract_trials_for_verify(rows_trials)
    learned_events = [dict(r) for r in rows_learned if isinstance(r, dict)]
    action_plans = _extract_plans(rows_plans)

    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v94(turns_path)),
        "parses_chain_ok": bool(verify_chained_jsonl_v94(parses_path)),
        "learned_chain_ok": bool(verify_chained_jsonl_v94(learned_path)) if os.path.exists(learned_path) else True,
        "plans_chain_ok": bool(verify_chained_jsonl_v94(plans_path)),
        "states_chain_ok": bool(verify_chained_jsonl_v94(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v94(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v94(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v94(transcript_path)),
    }

    ok_inv, reason, details = verify_conversation_chain_v94(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        action_plans=action_plans,
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
            "plans_total": int(len(action_plans)),
            "trials_total": int(len(trials)),
            "evals_total": int(len(rows_evals)),
            "transcript_total": int(len(rows_transcript)),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

