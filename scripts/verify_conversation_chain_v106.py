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
from atos_core.conversation_v106 import verify_conversation_chain_v106


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    return obj if isinstance(obj, dict) else {}


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


def _extract_payloads(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        payload = r.get("payload")
        if isinstance(payload, dict):
            out.append(dict(payload))
    return out


def _extract_parse_events(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        payload = dict(r.get("payload") or {}) if isinstance(r.get("payload"), dict) else {}
        out.append({"turn_id": str(r.get("turn_id") or ""), "turn_index": int(r.get("turn_index") or 0), "payload": dict(payload)})
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
    if not os.path.exists(path):
        return out
    for r in _read_jsonl(path):
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
    repo_root = os.path.dirname(os.path.dirname(__file__))

    turns_path = os.path.join(run_dir, "conversation_turns.jsonl")
    parses_path = os.path.join(run_dir, "intent_parses.jsonl")
    learned_path = os.path.join(run_dir, "learned_intent_rules.jsonl")
    plans_path = os.path.join(run_dir, "action_plans.jsonl")
    plan_events_path = os.path.join(run_dir, "plan_events.jsonl")
    plan_snapshot_path = os.path.join(run_dir, "plan_registry_snapshot_v104.json")
    agency_events_path = os.path.join(run_dir, "agency_events.jsonl")
    agency_snapshot_path = os.path.join(run_dir, "agency_registry_snapshot_v105.json")
    dialogue_events_path = os.path.join(run_dir, "dialogue_events.jsonl")
    dialogue_snapshot_path = os.path.join(run_dir, "dialogue_registry_snapshot_v106.json")
    memory_path = os.path.join(run_dir, "memory_events.jsonl")
    belief_path = os.path.join(run_dir, "belief_events.jsonl")
    evidence_path = os.path.join(run_dir, "evidence_events.jsonl")
    goal_path = os.path.join(run_dir, "goal_events.jsonl")
    goal_snapshot_path = os.path.join(run_dir, "goal_ledger_snapshot.json")
    discourse_path = os.path.join(run_dir, "discourse_events.jsonl")
    fragment_events_path = os.path.join(run_dir, "fragment_events.jsonl")
    fragment_snapshot_path = os.path.join(run_dir, "fragment_library_snapshot.json")
    binding_events_path = os.path.join(run_dir, "binding_events.jsonl")
    binding_snapshot_path = os.path.join(run_dir, "binding_snapshot.json")
    style_path = os.path.join(run_dir, "style_events.jsonl")
    template_snapshot_path = os.path.join(run_dir, "template_library_snapshot_v102.json")
    concept_events_path = os.path.join(run_dir, "concept_events.jsonl")
    concept_snapshot_path = os.path.join(run_dir, "concept_library_snapshot_v103.json")
    states_path = os.path.join(run_dir, "conversation_states.jsonl")
    trials_path = os.path.join(run_dir, "dialogue_trials.jsonl")
    evals_path = os.path.join(run_dir, "objective_evals.jsonl")
    transcript_path = os.path.join(run_dir, "transcript.jsonl")
    system_spec_path = os.path.join(run_dir, "system_spec_snapshot.json")
    verify_path = os.path.join(run_dir, "verify_chain_v106.json")
    manifest_path = os.path.join(run_dir, "freeze_manifest_v106.json")

    system_spec_ok = os.path.exists(system_spec_path)
    system_spec_sha = sha256_file(system_spec_path) if system_spec_ok else ""

    manifest_ok = os.path.exists(manifest_path)
    manifest_sha = sha256_file(manifest_path) if manifest_ok else ""
    manifest_sha_obj: Dict[str, Any] = {}
    if manifest_ok:
        try:
            manifest_sha_obj = _read_json(manifest_path)
        except Exception:
            manifest_sha_obj = {}
    sha_map = manifest_sha_obj.get("sha256") if isinstance(manifest_sha_obj.get("sha256"), dict) else {}

    def _man_sha(key: str) -> str:
        v = sha_map.get(str(key), "")
        return str(v) if isinstance(v, str) else ""

    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v96(turns_path)),
        "parses_chain_ok": bool(verify_chained_jsonl_v96(parses_path)),
        "learned_chain_ok": bool(verify_chained_jsonl_v96(learned_path)) if os.path.exists(learned_path) else True,
        "plans_chain_ok": bool(verify_chained_jsonl_v96(plans_path)),
        "plan_events_chain_ok": bool(verify_chained_jsonl_v96(plan_events_path)),
        "agency_chain_ok": bool(verify_chained_jsonl_v96(agency_events_path)),
        "dialogue_chain_ok": bool(verify_chained_jsonl_v96(dialogue_events_path)),
        "memory_chain_ok": bool(verify_chained_jsonl_v96(memory_path)),
        "belief_chain_ok": bool(verify_chained_jsonl_v96(belief_path)),
        "evidence_chain_ok": bool(verify_chained_jsonl_v96(evidence_path)),
        "goal_chain_ok": bool(verify_chained_jsonl_v96(goal_path)),
        "discourse_chain_ok": bool(verify_chained_jsonl_v96(discourse_path)),
        "fragment_chain_ok": bool(verify_chained_jsonl_v96(fragment_events_path)),
        "binding_chain_ok": bool(verify_chained_jsonl_v96(binding_events_path)),
        "style_chain_ok": bool(verify_chained_jsonl_v96(style_path)),
        "concept_chain_ok": bool(verify_chained_jsonl_v96(concept_events_path)),
        "states_chain_ok": bool(verify_chained_jsonl_v96(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v96(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v96(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v96(transcript_path)),
        "goal_snapshot_exists": bool(os.path.exists(goal_snapshot_path)),
        "binding_snapshot_exists": bool(os.path.exists(binding_snapshot_path)),
        "dialogue_snapshot_exists": bool(os.path.exists(dialogue_snapshot_path)),
        "system_spec_exists": bool(system_spec_ok),
        "fragment_snapshot_exists": bool(os.path.exists(fragment_snapshot_path)),
        "template_snapshot_exists": bool(os.path.exists(template_snapshot_path)),
        "concept_snapshot_exists": bool(os.path.exists(concept_snapshot_path)),
        "plan_snapshot_exists": bool(os.path.exists(plan_snapshot_path)),
        "agency_snapshot_exists": bool(os.path.exists(agency_snapshot_path)),
        "manifest_exists": bool(manifest_ok),
        "verify_json_exists": bool(os.path.exists(verify_path)),
        "manifest_system_spec_sha_match": bool(system_spec_ok and manifest_ok and system_spec_sha and system_spec_sha == _man_sha("system_spec_snapshot_json")),
        "manifest_dialogue_events_sha_match": bool(
            manifest_ok and os.path.exists(dialogue_events_path) and sha256_file(dialogue_events_path) == _man_sha("dialogue_events_v106_jsonl")
        ),
        "manifest_dialogue_snapshot_sha_match": bool(
            manifest_ok and os.path.exists(dialogue_snapshot_path) and sha256_file(dialogue_snapshot_path) == _man_sha("dialogue_registry_snapshot_v106_json")
        ),
        "manifest_verify_sha_match": bool(manifest_ok and os.path.exists(verify_path) and sha256_file(verify_path) == _man_sha("verify_chain_v106_json")),
    }

    turns = _extract_payloads(turns_path)
    states = _extract_payloads(states_path) if os.path.exists(states_path) else []
    parse_events = _extract_parse_events(parses_path)
    trials = _extract_trials_for_verify(trials_path)
    learned_events = _extract_learned_events(learned_path)
    action_plans = _strip_chain_fields(plans_path)
    plan_events = _strip_chain_fields(plan_events_path)
    plan_snapshot = _read_json(plan_snapshot_path) if os.path.exists(plan_snapshot_path) else {}
    agency_events = _strip_chain_fields(agency_events_path)
    agency_snapshot = _read_json(agency_snapshot_path) if os.path.exists(agency_snapshot_path) else {}
    dialogue_events = _strip_chain_fields(dialogue_events_path)
    dialogue_snapshot = _read_json(dialogue_snapshot_path) if os.path.exists(dialogue_snapshot_path) else {}
    memory_events = _strip_chain_fields(memory_path)
    belief_events = _strip_chain_fields(belief_path)
    evidence_events = _strip_chain_fields(evidence_path)
    goal_events = _extract_payloads(goal_path)
    goal_snapshot = _read_json(goal_snapshot_path) if os.path.exists(goal_snapshot_path) else {}
    discourse_events = _extract_payloads(discourse_path)
    fragment_events = _extract_payloads(fragment_events_path)
    binding_events = _strip_chain_fields(binding_events_path)
    binding_snapshot = _read_json(binding_snapshot_path) if os.path.exists(binding_snapshot_path) else {}
    style_events = _extract_payloads(style_path)
    template_snapshot = _read_json(template_snapshot_path) if os.path.exists(template_snapshot_path) else {}
    concept_events = _strip_chain_fields(concept_events_path)
    concept_snapshot = _read_json(concept_snapshot_path) if os.path.exists(concept_snapshot_path) else {}

    ok_chain, reason, details = verify_conversation_chain_v106(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        action_plans=action_plans,
        memory_events=memory_events,
        belief_events=belief_events,
        evidence_events=evidence_events,
        goal_events=goal_events,
        goal_snapshot=dict(goal_snapshot),
        discourse_events=discourse_events,
        fragment_events=fragment_events,
        binding_events=binding_events,
        binding_snapshot=dict(binding_snapshot),
        style_events=style_events,
        template_snapshot=dict(template_snapshot),
        concept_events=concept_events,
        concept_snapshot=dict(concept_snapshot),
        plan_events=plan_events,
        plan_snapshot=dict(plan_snapshot),
        agency_events=agency_events,
        agency_snapshot=dict(agency_snapshot),
        dialogue_events=dialogue_events,
        dialogue_snapshot=dict(dialogue_snapshot),
        tail_k=6,
        repo_root=str(repo_root),
    )

    out = {
        "ok": bool(ok_chain) and bool(all(chains.values())),
        "reason": str(reason),
        "details": dict(details),
        "chains": dict(chains),
        "run_dir": str(run_dir),
        "manifest_sha256": str(manifest_sha),
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

