#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_v96 import verify_chained_jsonl_v96
from atos_core.conversation_v99 import verify_conversation_chain_v99


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        payload = r.get("payload") if isinstance(r.get("payload"), dict) else {}
        out.append(
            {
                "turn_id": str(r.get("turn_id") or ""),
                "turn_index": int(r.get("turn_index") or 0),
                "payload": dict(payload),
            }
        )
    return out


def _extract_trials_for_verify(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        out.append({"objective_kind": str(r.get("objective_kind") or ""), "user_turn_id": str(r.get("user_turn_id") or r.get("turn_id") or "")})
    return out


def _extract_learned_events(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = dict(r)
        d.pop("prev_hash", None)
        d.pop("entry_hash", None)
        out.append(d)
    return out


def _strip_chain_fields(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
    memory_path = os.path.join(run_dir, "memory_events.jsonl")
    belief_path = os.path.join(run_dir, "belief_events.jsonl")
    evidence_path = os.path.join(run_dir, "evidence_events.jsonl")
    goal_path = os.path.join(run_dir, "goal_events.jsonl")
    goal_snapshot_path = os.path.join(run_dir, "goal_ledger_snapshot.json")
    states_path = os.path.join(run_dir, "conversation_states.jsonl")
    trials_path = os.path.join(run_dir, "dialogue_trials.jsonl")
    evals_path = os.path.join(run_dir, "objective_evals.jsonl")
    transcript_path = os.path.join(run_dir, "transcript.jsonl")
    system_spec_path = os.path.join(run_dir, "system_spec_snapshot.json")
    manifest_path = os.path.join(run_dir, "freeze_manifest_v99.json")

    rows_turns = _read_jsonl(turns_path)
    rows_parses = _read_jsonl(parses_path)
    rows_learned = _read_jsonl(learned_path) if os.path.exists(learned_path) else []
    rows_plans = _read_jsonl(plans_path)
    rows_memory = _read_jsonl(memory_path) if os.path.exists(memory_path) else []
    rows_belief = _read_jsonl(belief_path) if os.path.exists(belief_path) else []
    rows_evidence = _read_jsonl(evidence_path) if os.path.exists(evidence_path) else []
    rows_goals = _read_jsonl(goal_path) if os.path.exists(goal_path) else []
    rows_states = _read_jsonl(states_path) if os.path.exists(states_path) else []
    rows_trials = _read_jsonl(trials_path)
    rows_evals = _read_jsonl(evals_path)
    rows_transcript = _read_jsonl(transcript_path)

    turns = _extract_payloads(rows_turns)
    states = _extract_payloads(rows_states)
    parse_events = _extract_parse_events(rows_parses)
    trials = _extract_trials_for_verify(rows_trials)
    learned_events = _extract_learned_events(rows_learned)
    plans = _strip_chain_fields(rows_plans)
    memory_events = _strip_chain_fields(rows_memory)
    belief_events = _strip_chain_fields(rows_belief)
    evidence_events = _strip_chain_fields(rows_evidence)
    goal_events = _extract_payloads(rows_goals)

    goal_snapshot: Dict[str, Any] = {}
    goal_snapshot_ok = os.path.exists(goal_snapshot_path)
    if goal_snapshot_ok:
        try:
            with open(goal_snapshot_path, "r", encoding="utf-8") as f:
                goal_snapshot = json.load(f)
            if not isinstance(goal_snapshot, dict):
                goal_snapshot = {}
                goal_snapshot_ok = False
        except Exception:
            goal_snapshot = {}
            goal_snapshot_ok = False

    system_spec_ok = os.path.exists(system_spec_path)
    system_spec_sha = sha256_file(system_spec_path) if system_spec_ok else ""
    manifest_ok = os.path.exists(manifest_path)
    manifest_sha = sha256_file(manifest_path) if manifest_ok else ""
    manifest_system_sha = ""
    manifest_evidence_sha = ""
    manifest_goal_sha = ""
    manifest_goal_snapshot_sha = ""
    if manifest_ok:
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                man = json.load(f)
            sha_obj = man.get("sha256") if isinstance(man.get("sha256"), dict) else {}
            manifest_system_sha = str(sha_obj.get("system_spec_snapshot_json") or "")
            manifest_evidence_sha = str(sha_obj.get("evidence_events_jsonl") or "")
            manifest_goal_sha = str(sha_obj.get("goal_events_jsonl") or "")
            manifest_goal_snapshot_sha = str(sha_obj.get("goal_ledger_snapshot_json") or "")
        except Exception:
            manifest_ok = False

    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v96(turns_path)),
        "parses_chain_ok": bool(verify_chained_jsonl_v96(parses_path)),
        "learned_chain_ok": bool(verify_chained_jsonl_v96(learned_path)) if os.path.exists(learned_path) else True,
        "plans_chain_ok": bool(verify_chained_jsonl_v96(plans_path)),
        "memory_chain_ok": bool(verify_chained_jsonl_v96(memory_path)) if os.path.exists(memory_path) else False,
        "belief_chain_ok": bool(verify_chained_jsonl_v96(belief_path)) if os.path.exists(belief_path) else False,
        "evidence_chain_ok": bool(verify_chained_jsonl_v96(evidence_path)) if os.path.exists(evidence_path) else False,
        "goal_chain_ok": bool(verify_chained_jsonl_v96(goal_path)) if os.path.exists(goal_path) else False,
        "states_chain_ok": bool(verify_chained_jsonl_v96(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v96(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v96(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v96(transcript_path)),
        "system_spec_exists": bool(system_spec_ok),
        "goal_snapshot_exists": bool(goal_snapshot_ok),
        "manifest_exists": bool(manifest_ok),
        "manifest_system_spec_sha_match": bool(system_spec_ok and manifest_ok and system_spec_sha and system_spec_sha == manifest_system_sha),
        "manifest_evidence_sha_match": bool(os.path.exists(evidence_path) and manifest_ok and sha256_file(evidence_path) == manifest_evidence_sha),
        "manifest_goal_sha_match": bool(os.path.exists(goal_path) and manifest_ok and sha256_file(goal_path) == manifest_goal_sha),
        "manifest_goal_snapshot_sha_match": bool(goal_snapshot_ok and manifest_ok and sha256_file(goal_snapshot_path) == manifest_goal_snapshot_sha),
    }

    ok_inv, reason, details = verify_conversation_chain_v99(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        action_plans=plans,
        memory_events=memory_events,
        belief_events=belief_events,
        evidence_events=evidence_events,
        goal_events=goal_events,
        goal_snapshot=dict(goal_snapshot),
        tail_k=6,
    )

    out = {
        "ok": bool(all(chains.values())) and bool(ok_inv),
        "chains": dict(chains),
        "invariants": {"ok": bool(ok_inv), "reason": str(reason), "details": dict(details)},
        "files": {
            "system_spec_sha256": str(system_spec_sha),
            "manifest_sha256": str(manifest_sha),
            "manifest_system_spec_sha256": str(manifest_system_sha),
            "manifest_evidence_events_sha256": str(manifest_evidence_sha),
            "manifest_goal_events_sha256": str(manifest_goal_sha),
            "manifest_goal_snapshot_sha256": str(manifest_goal_snapshot_sha),
        },
        "counts": {
            "turns_total": int(len(turns)),
            "states_total": int(len(states)),
            "parses_total": int(len(parse_events)),
            "learned_total": int(len(learned_events)),
            "plans_total": int(len(plans)),
            "memory_events_total": int(len(memory_events)),
            "belief_events_total": int(len(belief_events)),
            "evidence_events_total": int(len(evidence_events)),
            "goal_events_total": int(len(goal_events)),
            "trials_total": int(len(trials)),
            "evals_total": int(len(rows_evals)),
            "transcript_total": int(len(rows_transcript)),
        },
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
