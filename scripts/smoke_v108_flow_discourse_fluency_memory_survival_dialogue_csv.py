#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v108 import run_conversation_v108
from atos_core.conversation_v108 import verify_conversation_chain_v108


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


def _load_for_verify(res: Dict[str, Any]) -> Dict[str, Any]:
    paths = res.get("paths") if isinstance(res.get("paths"), dict) else {}
    turns = _extract_payloads(str(paths.get("turns_jsonl") or ""))
    states_path = str(paths.get("states_jsonl") or "")
    states = _extract_payloads(states_path) if states_path and os.path.exists(states_path) else []
    parse_events = _extract_parse_events(str(paths.get("parses_jsonl") or ""))
    trials = _extract_trials_for_verify(str(paths.get("trials_jsonl") or ""))
    learned_events = _extract_learned_events(str(paths.get("learned_rules_jsonl") or ""))
    action_plans = _strip_chain_fields(str(paths.get("plans_jsonl") or ""))
    plan_events = _strip_chain_fields(str(paths.get("plan_events_v104_jsonl") or ""))
    plan_snapshot = _read_json(str(paths.get("plan_registry_snapshot_v104_json") or ""))
    agency_events = _strip_chain_fields(str(paths.get("agency_events_v105_jsonl") or ""))
    agency_snapshot = _read_json(str(paths.get("agency_registry_snapshot_v105_json") or ""))
    dialogue_events = _strip_chain_fields(str(paths.get("dialogue_events_v106_jsonl") or ""))
    dialogue_snapshot = _read_json(str(paths.get("dialogue_registry_snapshot_v106_json") or ""))
    pragmatics_events = _strip_chain_fields(str(paths.get("pragmatics_events_v107_jsonl") or ""))
    pragmatics_snapshot = _read_json(str(paths.get("pragmatics_registry_snapshot_v107_json") or ""))
    flow_events = _strip_chain_fields(str(paths.get("flow_events_v108_jsonl") or ""))
    flow_snapshot = _read_json(str(paths.get("flow_registry_snapshot_v108_json") or ""))
    memory_events = _strip_chain_fields(str(paths.get("memory_events_jsonl") or ""))
    belief_events = _strip_chain_fields(str(paths.get("belief_events_jsonl") or ""))
    evidence_events = _strip_chain_fields(str(paths.get("evidence_events_jsonl") or ""))
    goal_events = _extract_payloads(str(paths.get("goal_events_jsonl") or ""))
    goal_snapshot = _read_json(str(paths.get("goal_ledger_snapshot_json") or ""))
    discourse_events = _extract_payloads(str(paths.get("discourse_events_jsonl") or ""))
    fragment_events = _extract_payloads(str(paths.get("fragment_events_jsonl") or ""))
    binding_events = _strip_chain_fields(str(paths.get("binding_events_v101_jsonl") or paths.get("binding_events_jsonl") or ""))
    binding_snapshot = _read_json(str(paths.get("binding_snapshot_json") or ""))
    style_events = _extract_payloads(str(paths.get("style_events_jsonl") or ""))
    template_snapshot = _read_json(str(paths.get("template_library_snapshot_v102_json") or ""))
    concept_events = _strip_chain_fields(str(paths.get("concept_events_jsonl") or ""))
    concept_snapshot = _read_json(str(paths.get("concept_library_snapshot_v103_json") or ""))
    return {
        "turns": turns,
        "states": states,
        "parse_events": parse_events,
        "trials": trials,
        "learned_events": learned_events,
        "plans": action_plans,
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
    }


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # Multi-turn conversation that exercises: style preferences, reference ambiguity, repetition, and flow introspection.
    user_turns: List[str] = [
        "oi",
        "responde curto",
        "goal: preparar demo v108",
        "next",
        "faz aquilo",
        "next",
        "por favor defina x como 4",
        "set y to 8",
        "add x and y please",
        "add x and y please",
        "summary please",
        "agora explica bem detalhado",
        "belief: project = IAAA",
        "evidence: project = IAAA_v2",
        "why project",
        "flow",
        "explain_flow 4",
        "trace_flow 6",
        "blorp please",
        "end now",
    ]
    res = run_conversation_v108(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    data = _load_for_verify(res)
    ok_chain, reason, details = verify_conversation_chain_v108(
        turns=data["turns"],
        states=data["states"],
        parse_events=data["parse_events"],
        trials=data["trials"],
        learned_rule_events=data["learned_events"],
        action_plans=data["plans"],
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
        tail_k=6,
        repo_root=os.path.dirname(os.path.dirname(__file__)),
    )
    if not ok_chain:
        _fail(f"verify_fail:{reason}:{json.dumps(details, ensure_ascii=False, sort_keys=True)}")

    # Expect at least one blocked flow turn and at least one repair action.
    blocked = [ev for ev in data["flow_events"] if isinstance(ev, dict) and (not bool(ev.get("progress_allowed_v108", True)))]
    if not blocked:
        _fail("expected at least one flow event with progress_allowed_v108=false")
    repairs = [
        ev
        for ev in data["flow_events"]
        if isinstance(ev, dict) and str(ev.get("repair_action_v108") or "") and str(ev.get("repair_action_v108") or "") != "NONE"
    ]
    if not repairs:
        _fail("expected at least one flow repair_action_v108")

    # Negative tamper: corrupt flow event sig (inner chain).
    tamper_dir = str(out_dir) + "_tamper"
    ensure_absent(tamper_dir)
    shutil.copytree(str(out_dir), tamper_dir)
    p_path = os.path.join(tamper_dir, "flow_events.jsonl")
    rows = _read_jsonl(p_path)
    if not rows:
        _fail("missing flow_events.jsonl rows")
    rows[0]["event_sig"] = "0" * 64
    tmp = p_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    os.replace(tmp, p_path)
    tampered_flow_events = _strip_chain_fields(p_path)
    ok_t, reason_t, _details_t = verify_conversation_chain_v108(
        turns=data["turns"],
        states=data["states"],
        parse_events=data["parse_events"],
        trials=data["trials"],
        learned_rule_events=data["learned_events"],
        action_plans=data["plans"],
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
        flow_events=tampered_flow_events,
        flow_snapshot=dict(data["flow_snapshot"]),
        tail_k=6,
        repo_root=os.path.dirname(os.path.dirname(__file__)),
    )
    if ok_t or str(reason_t) != "flow_event_sig_mismatch":
        _fail(f"expected tamper fail flow_event_sig_mismatch; got ok={ok_t} reason={reason_t}")

    core = {
        "seed": int(seed),
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
        "negative_tamper_reason": "flow_event_sig_mismatch",
    }
    eval_path = os.path.join(str(out_dir), "eval.json")
    with open(eval_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(core, ensure_ascii=False, sort_keys=True, indent=2))
        f.write("\n")
    eval_sha = sha256_file(eval_path)
    core2 = dict(core, sha256_eval_json=str(eval_sha))
    summary_sha = hashlib.sha256(json.dumps(core2, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    smoke_summary = {"core": core2, "summary_sha256": str(summary_sha)}
    summary_path = os.path.join(str(out_dir), "smoke_summary.json")
    with open(summary_path, "x", encoding="utf-8") as f:
        f.write(json.dumps(smoke_summary, ensure_ascii=False, sort_keys=True, indent=2))
        f.write("\n")
    return {"core": core2, "summary_sha256": str(summary_sha), "eval_sha256": str(eval_sha)}


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

    determinism_ok = bool(r1.get("summary_sha256") == r2.get("summary_sha256") and r1.get("eval_sha256") == r2.get("eval_sha256"))
    if not determinism_ok:
        _fail("determinism_fail")

    out = {
        "ok": True,
        "determinism_ok": True,
        "summary_sha256": str(r1.get("summary_sha256") or ""),
        "eval_sha256": str(r1.get("eval_sha256") or ""),
        "try1_dir": str(out1),
        "try2_dir": str(out2),
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
