#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_v96 import verify_chained_jsonl_v96
from atos_core.conversation_v110 import verify_conversation_chain_v110


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


def _strip_outer_chain_fields(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        d = dict(r)
        d.pop("prev_hash", None)
        d.pop("entry_hash", None)
        out.append(d)
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


def _load_for_verify(*, run_dir: str) -> Dict[str, Any]:
    turns_path = os.path.join(run_dir, "conversation_turns.jsonl")
    states_path = os.path.join(run_dir, "conversation_states.jsonl")
    parses_path = os.path.join(run_dir, "intent_parses.jsonl")
    trials_path = os.path.join(run_dir, "dialogue_trials.jsonl")
    learned_path = os.path.join(run_dir, "learned_intent_rules.jsonl")
    plans_path = os.path.join(run_dir, "action_plans.jsonl")
    plan_events_path = os.path.join(run_dir, "plan_events.jsonl")
    plan_snapshot_path = os.path.join(run_dir, "plan_registry_snapshot_v104.json")
    agency_events_path = os.path.join(run_dir, "agency_events.jsonl")
    agency_snapshot_path = os.path.join(run_dir, "agency_registry_snapshot_v105.json")
    dialogue_events_path = os.path.join(run_dir, "dialogue_events.jsonl")
    dialogue_snapshot_path = os.path.join(run_dir, "dialogue_registry_snapshot_v106.json")
    pragmatics_events_path = os.path.join(run_dir, "pragmatics_events.jsonl")
    pragmatics_snapshot_path = os.path.join(run_dir, "pragmatics_registry_snapshot_v107.json")
    flow_events_path = os.path.join(run_dir, "flow_events.jsonl")
    flow_snapshot_path = os.path.join(run_dir, "flow_registry_snapshot_v108.json")
    semantic_events_path = os.path.join(run_dir, "semantic_events.jsonl")
    semantic_snapshot_path = os.path.join(run_dir, "semantic_registry_snapshot_v109.json")
    executive_events_path = os.path.join(run_dir, "executive_events.jsonl")
    executive_snapshot_path = os.path.join(run_dir, "executive_registry_snapshot_v110.json")

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

    turns = _extract_payloads(turns_path)
    states = _extract_payloads(states_path) if os.path.exists(states_path) else []
    parse_events = _extract_parse_events(parses_path)
    trials = _extract_trials_for_verify(trials_path)
    learned_rule_events = _strip_outer_chain_fields(_read_jsonl(learned_path)) if os.path.exists(learned_path) else []
    action_plans = _strip_outer_chain_fields(_read_jsonl(plans_path))
    plan_events = _strip_outer_chain_fields(_read_jsonl(plan_events_path))
    plan_snapshot = _read_json(plan_snapshot_path)
    agency_events = _strip_outer_chain_fields(_read_jsonl(agency_events_path))
    agency_snapshot = _read_json(agency_snapshot_path)
    dialogue_events = _strip_outer_chain_fields(_read_jsonl(dialogue_events_path))
    dialogue_snapshot = _read_json(dialogue_snapshot_path)
    pragmatics_events = _strip_outer_chain_fields(_read_jsonl(pragmatics_events_path))
    pragmatics_snapshot = _read_json(pragmatics_snapshot_path)
    flow_events = _strip_outer_chain_fields(_read_jsonl(flow_events_path))
    flow_snapshot = _read_json(flow_snapshot_path)
    semantic_events = _strip_outer_chain_fields(_read_jsonl(semantic_events_path))
    semantic_snapshot = _read_json(semantic_snapshot_path)
    executive_events = _strip_outer_chain_fields(_read_jsonl(executive_events_path))
    executive_snapshot = _read_json(executive_snapshot_path)

    memory_events = _strip_outer_chain_fields(_read_jsonl(memory_path))
    belief_events = _strip_outer_chain_fields(_read_jsonl(belief_path))
    evidence_events = _strip_outer_chain_fields(_read_jsonl(evidence_path))
    goal_events = _extract_payloads(goal_path)
    goal_snapshot = _read_json(goal_snapshot_path)
    discourse_events = _extract_payloads(discourse_path)
    fragment_events = _extract_payloads(fragment_events_path)
    binding_events = _strip_outer_chain_fields(_read_jsonl(binding_events_path))
    binding_snapshot = _read_json(binding_snapshot_path)
    style_events = _extract_payloads(style_path)
    template_snapshot = _read_json(template_snapshot_path)
    concept_events = _strip_outer_chain_fields(_read_jsonl(concept_events_path))
    concept_snapshot = _read_json(concept_snapshot_path)

    return {
        "turns": turns,
        "states": states,
        "parse_events": parse_events,
        "trials": trials,
        "learned_rule_events": learned_rule_events,
        "action_plans": action_plans,
        "plan_events": plan_events,
        "plan_snapshot": plan_snapshot,
        "agency_events": agency_events,
        "agency_snapshot": agency_snapshot,
        "dialogue_events": dialogue_events,
        "dialogue_snapshot": dialogue_snapshot,
        "pragmatics_events": pragmatics_events,
        "pragmatics_snapshot": pragmatics_snapshot,
        "flow_events": flow_events,
        "flow_snapshot": flow_snapshot,
        "semantic_events": semantic_events,
        "semantic_snapshot": semantic_snapshot,
        "executive_events": executive_events,
        "executive_snapshot": executive_snapshot,
        "memory_events": memory_events,
        "belief_events": belief_events,
        "evidence_events": evidence_events,
        "goal_events": goal_events,
        "goal_snapshot": goal_snapshot,
        "discourse_events": discourse_events,
        "fragment_events": fragment_events,
        "binding_events": binding_events,
        "binding_snapshot": binding_snapshot,
        "style_events": style_events,
        "template_snapshot": template_snapshot,
        "concept_events": concept_events,
        "concept_snapshot": concept_snapshot,
        "paths": {
            "turns_jsonl": turns_path,
            "states_jsonl": states_path,
            "parses_jsonl": parses_path,
            "trials_jsonl": trials_path,
            "learned_rules_jsonl": learned_path,
            "plans_jsonl": plans_path,
            "plan_events_v104_jsonl": plan_events_path,
            "plan_registry_snapshot_v104_json": plan_snapshot_path,
            "agency_events_v105_jsonl": agency_events_path,
            "agency_registry_snapshot_v105_json": agency_snapshot_path,
            "dialogue_events_v106_jsonl": dialogue_events_path,
            "dialogue_registry_snapshot_v106_json": dialogue_snapshot_path,
            "pragmatics_events_v107_jsonl": pragmatics_events_path,
            "pragmatics_registry_snapshot_v107_json": pragmatics_snapshot_path,
            "flow_events_v108_jsonl": flow_events_path,
            "flow_registry_snapshot_v108_json": flow_snapshot_path,
            "semantic_events_v109_jsonl": semantic_events_path,
            "semantic_registry_snapshot_v109_json": semantic_snapshot_path,
            "executive_events_v110_jsonl": executive_events_path,
            "executive_registry_snapshot_v110_json": executive_snapshot_path,
            "memory_events_jsonl": memory_path,
            "belief_events_jsonl": belief_path,
            "evidence_events_jsonl": evidence_path,
            "goal_events_jsonl": goal_path,
            "goal_ledger_snapshot_json": goal_snapshot_path,
            "discourse_events_jsonl": discourse_path,
            "fragment_events_jsonl": fragment_events_path,
            "fragment_library_snapshot_json": fragment_snapshot_path,
            "binding_events_jsonl": binding_events_path,
            "binding_snapshot_json": binding_snapshot_path,
            "style_events_jsonl": style_path,
            "template_library_snapshot_v102_json": template_snapshot_path,
            "concept_events_jsonl": concept_events_path,
            "concept_library_snapshot_v103_json": concept_snapshot_path,
        },
    }


def _verify_manifest_sha(*, manifest_path: str, sha_key: str, file_path: str) -> bool:
    if not os.path.exists(manifest_path) or not os.path.exists(file_path):
        return False
    man = _read_json(manifest_path)
    sha_map = man.get("sha256") if isinstance(man.get("sha256"), dict) else {}
    want = sha_map.get(str(sha_key), "")
    if not isinstance(want, str) or not want:
        return False
    return sha256_file(file_path) == str(want)


def verify_run_dir_v110(*, run_dir: str) -> Tuple[bool, str, Dict[str, Any]]:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data = _load_for_verify(run_dir=str(run_dir))

    # Outer chain checks (append_chained_jsonl_v96).
    paths = data["paths"]
    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v96(paths["turns_jsonl"])),
        "parses_chain_ok": bool(verify_chained_jsonl_v96(paths["parses_jsonl"])),
        "plans_chain_ok": bool(verify_chained_jsonl_v96(paths["plans_jsonl"])),
        "plan_events_chain_ok": bool(verify_chained_jsonl_v96(paths["plan_events_v104_jsonl"])),
        "agency_chain_ok": bool(verify_chained_jsonl_v96(paths["agency_events_v105_jsonl"])),
        "dialogue_chain_ok": bool(verify_chained_jsonl_v96(paths["dialogue_events_v106_jsonl"])),
        "pragmatics_chain_ok": bool(verify_chained_jsonl_v96(paths["pragmatics_events_v107_jsonl"])),
        "flow_chain_ok": bool(verify_chained_jsonl_v96(paths["flow_events_v108_jsonl"])),
        "semantic_chain_ok": bool(verify_chained_jsonl_v96(paths["semantic_events_v109_jsonl"])),
        "executive_chain_ok": bool(verify_chained_jsonl_v96(paths["executive_events_v110_jsonl"])),
        "memory_chain_ok": bool(verify_chained_jsonl_v96(paths["memory_events_jsonl"])),
        "belief_chain_ok": bool(verify_chained_jsonl_v96(paths["belief_events_jsonl"])),
        "evidence_chain_ok": bool(verify_chained_jsonl_v96(paths["evidence_events_jsonl"])),
        "goal_chain_ok": bool(verify_chained_jsonl_v96(paths["goal_events_jsonl"])),
        "discourse_chain_ok": bool(verify_chained_jsonl_v96(paths["discourse_events_jsonl"])),
        "fragment_chain_ok": bool(verify_chained_jsonl_v96(paths["fragment_events_jsonl"])),
        "binding_chain_ok": bool(verify_chained_jsonl_v96(paths["binding_events_jsonl"])),
        "style_chain_ok": bool(verify_chained_jsonl_v96(paths["style_events_jsonl"])),
        "concept_chain_ok": bool(verify_chained_jsonl_v96(paths["concept_events_jsonl"])),
        "plan_snapshot_exists": bool(os.path.exists(paths["plan_registry_snapshot_v104_json"])),
        "agency_snapshot_exists": bool(os.path.exists(paths["agency_registry_snapshot_v105_json"])),
        "dialogue_snapshot_exists": bool(os.path.exists(paths["dialogue_registry_snapshot_v106_json"])),
        "pragmatics_snapshot_exists": bool(os.path.exists(paths["pragmatics_registry_snapshot_v107_json"])),
        "flow_snapshot_exists": bool(os.path.exists(paths["flow_registry_snapshot_v108_json"])),
        "semantic_snapshot_exists": bool(os.path.exists(paths["semantic_registry_snapshot_v109_json"])),
        "executive_snapshot_exists": bool(os.path.exists(paths["executive_registry_snapshot_v110_json"])),
        "goal_snapshot_exists": bool(os.path.exists(paths["goal_ledger_snapshot_json"])),
        "binding_snapshot_exists": bool(os.path.exists(paths["binding_snapshot_json"])),
        "template_snapshot_exists": bool(os.path.exists(paths["template_library_snapshot_v102_json"])),
        "concept_snapshot_exists": bool(os.path.exists(paths["concept_library_snapshot_v103_json"])),
    }

    ok, reason, details = verify_conversation_chain_v110(
        turns=data["turns"],
        states=data["states"],
        parse_events=data["parse_events"],
        trials=data["trials"],
        learned_rule_events=data["learned_rule_events"],
        action_plans=data["action_plans"],
        memory_events=data["memory_events"],
        belief_events=data["belief_events"],
        evidence_events=data["evidence_events"],
        goal_events=data["goal_events"],
        goal_snapshot=dict(data["goal_snapshot"]),
        discourse_events=data["discourse_events"],
        fragment_events=data["fragment_events"],
        binding_events=data["binding_events"],
        binding_snapshot=dict(data["binding_snapshot"]),
        style_events=data["style_events"],
        template_snapshot=dict(data["template_snapshot"]),
        concept_events=data["concept_events"],
        concept_snapshot=dict(data["concept_snapshot"]),
        plan_events=data["plan_events"],
        plan_snapshot=dict(data["plan_snapshot"]),
        agency_events=data["agency_events"],
        agency_snapshot=dict(data["agency_snapshot"]),
        dialogue_events=data["dialogue_events"],
        dialogue_snapshot=dict(data["dialogue_snapshot"]),
        pragmatics_events=data["pragmatics_events"],
        pragmatics_snapshot=dict(data["pragmatics_snapshot"]),
        flow_events=data["flow_events"],
        flow_snapshot=dict(data["flow_snapshot"]),
        semantic_events=data["semantic_events"],
        semantic_snapshot=dict(data["semantic_snapshot"]),
        executive_events=data["executive_events"],
        executive_snapshot=dict(data["executive_snapshot"]),
        tail_k=6,
        repo_root=str(repo_root),
    )

    # Manifest sha checks (optional but useful).
    manifest_path = os.path.join(run_dir, "freeze_manifest_v110.json")
    verify_json_path = os.path.join(run_dir, "verify_chain_v110.json")
    sha_checks = {
        "manifest_exists": bool(os.path.exists(manifest_path)),
        "manifest_verify_chain_sha_match": bool(_verify_manifest_sha(manifest_path=manifest_path, sha_key="verify_chain_v110_json", file_path=verify_json_path)),
        "manifest_executive_events_sha_match": bool(
            _verify_manifest_sha(manifest_path=manifest_path, sha_key="executive_events_v110_jsonl", file_path=paths["executive_events_v110_jsonl"])
        ),
        "manifest_executive_snapshot_sha_match": bool(
            _verify_manifest_sha(
                manifest_path=manifest_path, sha_key="executive_registry_snapshot_v110_json", file_path=paths["executive_registry_snapshot_v110_json"]
            )
        ),
    }

    out = {
        "ok": bool(ok) and bool(all(chains.values())) and bool(all(sha_checks.values() if sha_checks.get("manifest_exists") else [])),
        "reason": str(reason),
        "details": dict(details),
        "chains": dict(chains),
        "sha_checks": dict(sha_checks),
    }
    return bool(out["ok"]), str(out["reason"]), out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    ok, reason, out = verify_run_dir_v110(run_dir=str(args.run_dir))
    print(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2))
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

