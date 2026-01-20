#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.conversation_loop_v110 import run_conversation_v110
from atos_core.conversation_v110 import verify_conversation_chain_v110
from atos_core.conversation_v96 import verify_chained_jsonl_v96


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


def _load_run_dir_for_verify(*, run_dir: str) -> Dict[str, Any]:
    paths = {
        "turns_jsonl": os.path.join(run_dir, "conversation_turns.jsonl"),
        "states_jsonl": os.path.join(run_dir, "conversation_states.jsonl"),
        "parses_jsonl": os.path.join(run_dir, "intent_parses.jsonl"),
        "trials_jsonl": os.path.join(run_dir, "dialogue_trials.jsonl"),
        "learned_rules_jsonl": os.path.join(run_dir, "learned_intent_rules.jsonl"),
        "plans_jsonl": os.path.join(run_dir, "action_plans.jsonl"),
        "plan_events_v104_jsonl": os.path.join(run_dir, "plan_events.jsonl"),
        "plan_registry_snapshot_v104_json": os.path.join(run_dir, "plan_registry_snapshot_v104.json"),
        "agency_events_v105_jsonl": os.path.join(run_dir, "agency_events.jsonl"),
        "agency_registry_snapshot_v105_json": os.path.join(run_dir, "agency_registry_snapshot_v105.json"),
        "dialogue_events_v106_jsonl": os.path.join(run_dir, "dialogue_events.jsonl"),
        "dialogue_registry_snapshot_v106_json": os.path.join(run_dir, "dialogue_registry_snapshot_v106.json"),
        "pragmatics_events_v107_jsonl": os.path.join(run_dir, "pragmatics_events.jsonl"),
        "pragmatics_registry_snapshot_v107_json": os.path.join(run_dir, "pragmatics_registry_snapshot_v107.json"),
        "flow_events_v108_jsonl": os.path.join(run_dir, "flow_events.jsonl"),
        "flow_registry_snapshot_v108_json": os.path.join(run_dir, "flow_registry_snapshot_v108.json"),
        "semantic_events_v109_jsonl": os.path.join(run_dir, "semantic_events.jsonl"),
        "semantic_registry_snapshot_v109_json": os.path.join(run_dir, "semantic_registry_snapshot_v109.json"),
        "executive_events_v110_jsonl": os.path.join(run_dir, "executive_events.jsonl"),
        "executive_registry_snapshot_v110_json": os.path.join(run_dir, "executive_registry_snapshot_v110.json"),
        "memory_events_jsonl": os.path.join(run_dir, "memory_events.jsonl"),
        "belief_events_jsonl": os.path.join(run_dir, "belief_events.jsonl"),
        "evidence_events_jsonl": os.path.join(run_dir, "evidence_events.jsonl"),
        "goal_events_jsonl": os.path.join(run_dir, "goal_events.jsonl"),
        "goal_ledger_snapshot_json": os.path.join(run_dir, "goal_ledger_snapshot.json"),
        "discourse_events_jsonl": os.path.join(run_dir, "discourse_events.jsonl"),
        "fragment_events_jsonl": os.path.join(run_dir, "fragment_events.jsonl"),
        "binding_events_jsonl": os.path.join(run_dir, "binding_events.jsonl"),
        "binding_snapshot_json": os.path.join(run_dir, "binding_snapshot.json"),
        "style_events_jsonl": os.path.join(run_dir, "style_events.jsonl"),
        "template_library_snapshot_v102_json": os.path.join(run_dir, "template_library_snapshot_v102.json"),
        "concept_events_jsonl": os.path.join(run_dir, "concept_events.jsonl"),
        "concept_library_snapshot_v103_json": os.path.join(run_dir, "concept_library_snapshot_v103.json"),
    }
    turns = _extract_payloads(paths["turns_jsonl"])
    states = _extract_payloads(paths["states_jsonl"]) if os.path.exists(paths["states_jsonl"]) else []
    parse_events = _extract_parse_events(paths["parses_jsonl"])
    trials = _extract_trials_for_verify(paths["trials_jsonl"])
    learned_rule_events = _strip_chain_fields(_read_jsonl(paths["learned_rules_jsonl"])) if os.path.exists(paths["learned_rules_jsonl"]) else []
    action_plans = _strip_chain_fields(_read_jsonl(paths["plans_jsonl"]))
    plan_events = _strip_chain_fields(_read_jsonl(paths["plan_events_v104_jsonl"]))
    plan_snapshot = _read_json(paths["plan_registry_snapshot_v104_json"])
    agency_events = _strip_chain_fields(_read_jsonl(paths["agency_events_v105_jsonl"]))
    agency_snapshot = _read_json(paths["agency_registry_snapshot_v105_json"])
    dialogue_events = _strip_chain_fields(_read_jsonl(paths["dialogue_events_v106_jsonl"]))
    dialogue_snapshot = _read_json(paths["dialogue_registry_snapshot_v106_json"])
    pragmatics_events = _strip_chain_fields(_read_jsonl(paths["pragmatics_events_v107_jsonl"]))
    pragmatics_snapshot = _read_json(paths["pragmatics_registry_snapshot_v107_json"])
    flow_events = _strip_chain_fields(_read_jsonl(paths["flow_events_v108_jsonl"]))
    flow_snapshot = _read_json(paths["flow_registry_snapshot_v108_json"])
    semantic_events = _strip_chain_fields(_read_jsonl(paths["semantic_events_v109_jsonl"]))
    semantic_snapshot = _read_json(paths["semantic_registry_snapshot_v109_json"])
    executive_events = _strip_chain_fields(_read_jsonl(paths["executive_events_v110_jsonl"]))
    executive_snapshot = _read_json(paths["executive_registry_snapshot_v110_json"])
    memory_events = _strip_chain_fields(_read_jsonl(paths["memory_events_jsonl"]))
    belief_events = _strip_chain_fields(_read_jsonl(paths["belief_events_jsonl"]))
    evidence_events = _strip_chain_fields(_read_jsonl(paths["evidence_events_jsonl"]))
    goal_events = _extract_payloads(paths["goal_events_jsonl"])
    goal_snapshot = _read_json(paths["goal_ledger_snapshot_json"])
    discourse_events = _extract_payloads(paths["discourse_events_jsonl"])
    fragment_events = _extract_payloads(paths["fragment_events_jsonl"])
    binding_events = _strip_chain_fields(_read_jsonl(paths["binding_events_jsonl"]))
    binding_snapshot = _read_json(paths["binding_snapshot_json"])
    style_events = _extract_payloads(paths["style_events_jsonl"])
    template_snapshot = _read_json(paths["template_library_snapshot_v102_json"])
    concept_events = _strip_chain_fields(_read_jsonl(paths["concept_events_jsonl"]))
    concept_snapshot = _read_json(paths["concept_library_snapshot_v103_json"])
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
        "paths": dict(paths),
    }


def _recompute_outer_chain_v96(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    prev: str = ""
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = dict(r)
        d["prev_hash"] = prev if prev else None
        d.pop("entry_hash", None)
        entry_hash = sha256_hex(canonical_json_dumps(d).encode("utf-8"))
        d["entry_hash"] = str(entry_hash)
        prev = str(entry_hash)
        out.append(d)
    return out


def _tamper_executive_event_sig_keep_outer_chain_ok(*, run_dir: str) -> None:
    path = os.path.join(run_dir, "executive_events.jsonl")
    rows = _read_jsonl(path)
    if not rows:
        _fail("tamper_failed:no_executive_events")
    if not isinstance(rows[0], dict):
        _fail("tamper_failed:first_row_not_dict")
    rows0 = dict(rows[0])
    rows0["event_sig"] = "0" * 64
    rows[0] = rows0
    rows2 = _recompute_outer_chain_v96([dict(r) for r in rows if isinstance(r, dict)])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows2:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)


def _goal_ids_in_order(goal_events: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for ev in goal_events:
        if not isinstance(ev, dict):
            continue
        if str(ev.get("op") or "") == "GOAL_ADD":
            gid = str(ev.get("goal_id") or "")
            if gid:
                ids.append(gid)
    return ids


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # Scenario A: long autopilot (60 turns) without user driving.
    # Scenario B: forced stall/repair, then progress resumes.
    user_turns: List[str] = []
    user_turns.append("goal: outcome=deliver_demo constraints=no_external_deps deadline=none")
    # 60 autopilot acks.
    for _ in range(60):
        user_turns.append("ok")
    # Scenario B (new goal).
    user_turns.append("goal: outcome=second_demo constraints=stay_on_track deadline=none")
    user_turns.extend(["ok", "ok", "ok"])
    user_turns.append("hmm")   # forces topic reset => no-progress (count=1)
    user_turns.append("hmm2")  # forces second no-progress => stall predicted => repair
    user_turns.append("a")     # respond to options (A/B/C) deterministically
    user_turns.append("ok")    # progress resumes (same topic => step increments)
    user_turns.append("end now")

    res = run_conversation_v110(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))
    data = _load_run_dir_for_verify(run_dir=str(out_dir))

    ok_chain, reason, details = verify_conversation_chain_v110(
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
        repo_root=os.path.dirname(os.path.dirname(__file__)),
    )
    if not ok_chain:
        _fail(f"verify_failed:{reason}:{json.dumps(details, ensure_ascii=False)}")

    # Outer chain checks: ensure the executive outer chain is OK (tamper-evident).
    if not verify_chained_jsonl_v96(os.path.join(out_dir, "executive_events.jsonl")):
        _fail("executive_outer_chain_failed")

    goal_ids = _goal_ids_in_order(list(data["goal_events"]))
    if len(goal_ids) < 2:
        _fail(f"expected_2_goals_got:{len(goal_ids)}")
    goal1, goal2 = str(goal_ids[0]), str(goal_ids[1])
    goal1_done = any((isinstance(ev, dict) and str(ev.get("op") or "") == "GOAL_DONE" and str(ev.get("goal_id") or "") == goal1) for ev in data["goal_events"])
    if not goal1_done:
        _fail("goal1_not_done")

    exec_events = list(data["executive_events"])
    g1_exec = [ev for ev in exec_events if isinstance(ev, dict) and str(ev.get("goal_id") or "") == goal1]
    g1_passive = sum(1 for ev in g1_exec if "PASSIVE_WITH_OPEN_GOAL" in set(ev.get("executive_flags_v110") or []))
    g1_stall = sum(1 for ev in g1_exec if "STALL" in set(ev.get("executive_flags_v110") or []))
    g1_blocked = sum(1 for ev in g1_exec if not bool(ev.get("progress_allowed_v110", True)))
    if len(g1_exec) < 60:
        _fail(f"goal1_exec_events_lt_60:{len(g1_exec)}")
    if g1_passive != 0:
        _fail(f"goal1_passive_with_open_goal:{g1_passive}")
    if g1_stall != 0:
        _fail(f"goal1_stall_events_nonzero:{g1_stall}")
    if g1_blocked != 0:
        _fail(f"goal1_progress_blocked_nonzero:{g1_blocked}")

    # Scenario B assertions: stall triggers + repair + progress resumes for goal2.
    g2_exec = [ev for ev in exec_events if isinstance(ev, dict) and str(ev.get("goal_id") or "") == goal2]
    stall_events = [
        ev
        for ev in g2_exec
        if (not bool(ev.get("progress_allowed_v110", True)))
        and str(ev.get("repair_action_v110") or "") == "PLAN_REVISION_OR_ASK"
        and ("STALL" in set(ev.get("executive_flags_v110") or []))
    ]
    if not stall_events:
        _fail("expected_stall_event_missing")
    # Progress resumes: a later event increments step index.
    progress_events = [
        ev
        for ev in g2_exec
        if bool(ev.get("progress_allowed_v110", False))
        and int(ev.get("plan_step_index_after") or 0) > int(ev.get("plan_step_index_before") or 0)
    ]
    if not progress_events:
        _fail("expected_progress_resume_missing")

    # Write eval + smoke_summary (WORM).
    eval_obj = {
        "schema_version": 110,
        "goal1_id": str(goal1),
        "goal1_exec_events_total": int(len(g1_exec)),
        "goal1_done": bool(goal1_done),
        "goal2_id": str(goal2),
        "stall_events_total": int(len(stall_events)),
        "progress_resume_events_total": int(len(progress_events)),
        "hashes": {
            "store_hash": str(res.get("store_hash") or ""),
            "transcript_hash": str(res.get("transcript_hash") or ""),
            "state_chain_hash": str(res.get("state_chain_hash") or ""),
            "parse_chain_hash": str(res.get("parse_chain_hash") or ""),
            "learned_chain_hash": str(res.get("learned_chain_hash") or ""),
            "plan_chain_hash": str(res.get("plan_chain_hash") or ""),
            "plan_events_chain_hash_v104": str(res.get("plan_events_chain_hash_v104") or ""),
            "agency_chain_hash_v105": str(res.get("agency_chain_hash_v105") or ""),
            "dialogue_chain_hash_v106": str(res.get("dialogue_chain_hash_v106") or ""),
            "pragmatics_chain_hash_v107": str(res.get("pragmatics_chain_hash_v107") or ""),
            "flow_chain_hash_v108": str(res.get("flow_chain_hash_v108") or ""),
            "semantic_chain_hash_v109": str(res.get("semantic_chain_hash_v109") or ""),
            "executive_chain_hash_v110": str(res.get("executive_chain_hash_v110") or ""),
            "memory_chain_hash": str(res.get("memory_chain_hash") or ""),
            "belief_chain_hash": str(res.get("belief_chain_hash") or ""),
            "evidence_chain_hash": str(res.get("evidence_chain_hash") or ""),
            "goal_chain_hash": str(res.get("goal_chain_hash") or ""),
            "discourse_chain_hash": str(res.get("discourse_chain_hash") or ""),
            "fragment_chain_hash": str(res.get("fragment_chain_hash") or ""),
            "binding_chain_hash": str(res.get("binding_chain_hash") or ""),
            "style_chain_hash": str(res.get("style_chain_hash") or ""),
            "concept_chain_hash": str(res.get("concept_chain_hash") or ""),
            "ledger_hash": str(res.get("ledger_hash") or ""),
            "summary_sha256": str(res.get("summary_sha256") or ""),
        },
    }
    eval_path = os.path.join(out_dir, "eval.json")
    with open(eval_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(eval_obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    eval_sha256 = sha256_file(eval_path)

    core = {"schema_version": 110, "seed": int(seed), "eval_sha256": str(eval_sha256), "goal1_done": bool(goal1_done), "stall_events_total": int(len(stall_events))}
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    smoke_summary = dict(core, summary_sha256=str(summary_sha256))
    smoke_path = os.path.join(out_dir, "smoke_summary.json")
    with open(smoke_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(smoke_summary, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")

    return {
        "out_dir": str(out_dir),
        "eval_sha256": str(eval_sha256),
        "smoke_summary_sha256": str(summary_sha256),
        "hashes": dict(eval_obj.get("hashes") or {}),
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

    r1 = smoke_try(out_dir=str(out1), seed=int(seed))
    r2 = smoke_try(out_dir=str(out2), seed=int(seed))

    # Determinism checks (try1 == try2).
    det_ok = True
    det_mismatches: List[str] = []
    for k in sorted(r1.get("hashes", {}).keys(), key=str):
        v1 = str(r1["hashes"].get(k) or "")
        v2 = str(r2["hashes"].get(k) or "")
        if v1 != v2:
            det_ok = False
            det_mismatches.append(str(k))
    if str(r1.get("smoke_summary_sha256") or "") != str(r2.get("smoke_summary_sha256") or ""):
        det_ok = False
        det_mismatches.append("smoke_summary_sha256")

    # Negative tamper: executive_event_sig_mismatch.
    tamper_dir = out1 + "_tamper"
    ensure_absent(tamper_dir)
    shutil.copytree(out1, tamper_dir)
    _tamper_executive_event_sig_keep_outer_chain_ok(run_dir=tamper_dir)

    tamper_data = _load_run_dir_for_verify(run_dir=tamper_dir)
    ok_t, reason_t, _details_t = verify_conversation_chain_v110(
        turns=tamper_data["turns"],
        states=tamper_data["states"],
        parse_events=tamper_data["parse_events"],
        trials=tamper_data["trials"],
        learned_rule_events=tamper_data["learned_rule_events"],
        action_plans=tamper_data["action_plans"],
        memory_events=tamper_data["memory_events"],
        belief_events=tamper_data["belief_events"],
        evidence_events=tamper_data["evidence_events"],
        goal_events=tamper_data["goal_events"],
        goal_snapshot=dict(tamper_data["goal_snapshot"]),
        discourse_events=tamper_data["discourse_events"],
        fragment_events=tamper_data["fragment_events"],
        binding_events=tamper_data["binding_events"],
        binding_snapshot=dict(tamper_data["binding_snapshot"]),
        style_events=tamper_data["style_events"],
        template_snapshot=dict(tamper_data["template_snapshot"]),
        concept_events=tamper_data["concept_events"],
        concept_snapshot=dict(tamper_data["concept_snapshot"]),
        plan_events=tamper_data["plan_events"],
        plan_snapshot=dict(tamper_data["plan_snapshot"]),
        agency_events=tamper_data["agency_events"],
        agency_snapshot=dict(tamper_data["agency_snapshot"]),
        dialogue_events=tamper_data["dialogue_events"],
        dialogue_snapshot=dict(tamper_data["dialogue_snapshot"]),
        pragmatics_events=tamper_data["pragmatics_events"],
        pragmatics_snapshot=dict(tamper_data["pragmatics_snapshot"]),
        flow_events=tamper_data["flow_events"],
        flow_snapshot=dict(tamper_data["flow_snapshot"]),
        semantic_events=tamper_data["semantic_events"],
        semantic_snapshot=dict(tamper_data["semantic_snapshot"]),
        executive_events=tamper_data["executive_events"],
        executive_snapshot=dict(tamper_data["executive_snapshot"]),
        tail_k=6,
        repo_root=os.path.dirname(os.path.dirname(__file__)),
    )
    if ok_t or str(reason_t) != "executive_event_sig_mismatch":
        _fail(f"tamper_expected_executive_event_sig_mismatch_got:{str(reason_t)}")

    out = {
        "ok": bool(det_ok),
        "determinism_ok": bool(det_ok),
        "determinism_mismatches": list(det_mismatches),
        "try1": dict(r1),
        "try2": dict(r2),
        "tamper": {"ok": False, "reason": str(reason_t), "run_dir": str(tamper_dir)},
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2))
    if not det_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
