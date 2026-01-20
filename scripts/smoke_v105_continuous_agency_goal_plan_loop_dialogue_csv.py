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

from atos_core.conversation_loop_v105 import run_conversation_v105
from atos_core.conversation_v105 import verify_conversation_chain_v105


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


def _assistant_text_after_user_turn(*, turns: List[Dict[str, Any]], user_turn_index: int) -> str:
    want_idx = int(user_turn_index + 1)
    for t in turns:
        if not isinstance(t, dict):
            continue
        if int(t.get("turn_index") or -1) == want_idx and str(t.get("role") or "") == "assistant":
            return str(t.get("text") or "")
    return ""


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
    memory_events = _strip_chain_fields(str(paths.get("memory_events_jsonl") or ""))
    belief_events = _strip_chain_fields(str(paths.get("belief_events_jsonl") or ""))
    evidence_events = _strip_chain_fields(str(paths.get("evidence_events_jsonl") or ""))
    goal_events = _extract_payloads(str(paths.get("goal_events_jsonl") or ""))
    goal_snapshot = _read_json(str(paths.get("goal_ledger_snapshot_json") or ""))
    discourse_events = _extract_payloads(str(paths.get("discourse_events_jsonl") or ""))
    fragment_events = _extract_payloads(str(paths.get("fragment_events_jsonl") or ""))
    binding_events = _strip_chain_fields(str(paths.get("binding_events_jsonl") or ""))
    binding_snapshot = _read_json(str(paths.get("binding_snapshot_json") or ""))
    style_events = _extract_payloads(str(paths.get("style_events_jsonl") or ""))
    template_snapshot = _read_json(str(paths.get("template_library_snapshot_v102_json") or ""))
    concept_events = _strip_chain_fields(str(paths.get("concept_events_jsonl") or ""))
    concept_snapshot = _read_json(str(paths.get("concept_library_snapshot_v103_json") or ""))
    agency_events = _strip_chain_fields(str(paths.get("agency_events_v105_jsonl") or ""))
    agency_snapshot = _read_json(str(paths.get("agency_registry_snapshot_v105_json") or ""))
    return {
        "turns": turns,
        "states": states,
        "parse_events": parse_events,
        "trials": trials,
        "learned_events": learned_events,
        "plans": action_plans,
        "plan_events": plan_events,
        "plan_snapshot": plan_snapshot,
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
        "agency_events": agency_events,
        "agency_snapshot": agency_snapshot,
    }


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # Goal -> 1 clarification (multi-slot fill) -> propose options -> choose -> execute -> close.
    user_turns: List[str] = [
        "preparar demo v105",
        "outcome=plano de 3 passos, constraints=sem dependencias, deadline=hoje",
        "A",
        "agency",
        "explain_agency 0",
        "trace_agency 4",
        "end now",
    ]
    res = run_conversation_v105(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))
    paths = res.get("paths") if isinstance(res.get("paths"), dict) else {}

    data = _load_for_verify(res)
    ok_chain, reason, details = verify_conversation_chain_v105(
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
        tail_k=6,
        repo_root=os.path.dirname(os.path.dirname(__file__)),
    )
    if not ok_chain:
        _fail(f"verify_fail:{reason}:{json.dumps(details, sort_keys=True)}")

    turns = data["turns"]
    # Assert the first assistant response is a clarification question (mentions outcome).
    a0 = _assistant_text_after_user_turn(turns=turns, user_turn_index=0 * 2)
    if "outcome" not in a0:
        _fail(f"expected clarification asking for outcome; got={a0!r}")

    # After slot fill, assistant should propose options (OPÇÕES).
    # user_turn_index=4 corresponds to third user message ("A") -> assistant executes; so options should be after slot fill at user_turn_index=2.
    a1 = _assistant_text_after_user_turn(turns=turns, user_turn_index=1 * 2)
    a2 = _assistant_text_after_user_turn(turns=turns, user_turn_index=2 * 2)
    if "OPÇÕES" not in a1:
        _fail(f"expected options prompt; got={a1!r}")

    # After choosing A, assistant should execute and include "PLANO:".
    if "PLANO" not in a2:
        _fail(f"expected execute step with plan; got={a2!r}")

    # Negative tamper: corrupt agency event sig.
    tamper_dir = str(out_dir) + "_tamper"
    ensure_absent(tamper_dir)
    shutil.copytree(str(out_dir), tamper_dir)
    agency_path = os.path.join(tamper_dir, "agency_events.jsonl")
    rows = _read_jsonl(agency_path)
    if not rows:
        _fail("missing agency_events.jsonl rows")
    # Corrupt first row payload.
    rows[0]["event_sig"] = "0" * 64
    tmp = agency_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            f.write("\n")
    os.replace(tmp, agency_path)
    # Expect verifier to fail with agency_event_sig_mismatch.
    data2 = data
    data2_t = _load_for_verify({"paths": {**paths, "agency_events_v105_jsonl": agency_path, "agency_registry_snapshot_v105_json": os.path.join(tamper_dir, "agency_registry_snapshot_v105.json")}})
    ok2, r2, d2 = verify_conversation_chain_v105(
        turns=data2_t["turns"],
        states=data2_t["states"],
        parse_events=data2_t["parse_events"],
        trials=data2_t["trials"],
        learned_rule_events=data2_t["learned_events"],
        action_plans=data2_t["plans"],
        memory_events=data2_t["memory_events"],
        belief_events=data2_t["belief_events"],
        evidence_events=data2_t["evidence_events"],
        goal_events=data2_t["goal_events"],
        goal_snapshot=dict(data2_t["goal_snapshot"]),
        discourse_events=data2_t["discourse_events"],
        fragment_events=data2_t["fragment_events"],
        binding_events=data2_t["binding_events"],
        binding_snapshot=dict(data2_t["binding_snapshot"]),
        style_events=data2_t["style_events"],
        template_snapshot=dict(data2_t["template_snapshot"]),
        concept_events=data2_t["concept_events"],
        concept_snapshot=dict(data2_t["concept_snapshot"]),
        plan_events=data2_t["plan_events"],
        plan_snapshot=dict(data2_t["plan_snapshot"]),
        agency_events=data2_t["agency_events"],
        agency_snapshot=dict(data2_t["agency_snapshot"]),
        tail_k=6,
        repo_root=os.path.dirname(os.path.dirname(__file__)),
    )
    negative = {"ok": (not bool(ok2)) and str(r2) == "agency_event_sig_mismatch", "reason": str(r2), "details": dict(d2)}
    if not negative["ok"]:
        _fail(f"negative_tamper_expected_agency_event_sig_mismatch_got:{negative}")

    summary = _read_json(os.path.join(str(out_dir), "summary.json"))
    summary_sha256_file = sha256_file(os.path.join(str(out_dir), "summary.json"))
    return {
        "ok": True,
        "out_dir": str(out_dir),
        "paths": dict(paths),
        "hashes": {
            "store_hash": str(summary.get("store_hash") or ""),
            "transcript_hash": str(summary.get("transcript_hash") or ""),
            "state_chain_hash": str(summary.get("state_chain_hash") or ""),
            "parse_chain_hash": str(summary.get("parse_chain_hash") or ""),
            "learned_chain_hash": str(summary.get("learned_chain_hash") or ""),
            "plan_chain_hash": str(summary.get("plan_chain_hash") or ""),
            "plan_events_chain_hash_v104": str(summary.get("plan_events_chain_hash_v104") or ""),
            "agency_chain_hash_v105": str(summary.get("agency_chain_hash_v105") or ""),
            "memory_chain_hash": str(summary.get("memory_chain_hash") or ""),
            "belief_chain_hash": str(summary.get("belief_chain_hash") or ""),
            "evidence_chain_hash": str(summary.get("evidence_chain_hash") or ""),
            "goal_chain_hash": str(summary.get("goal_chain_hash") or ""),
            "discourse_chain_hash": str(summary.get("discourse_chain_hash") or ""),
            "fragment_chain_hash": str(summary.get("fragment_chain_hash") or ""),
            "binding_chain_hash": str(summary.get("binding_chain_hash") or ""),
            "style_chain_hash": str(summary.get("style_chain_hash") or ""),
            "concept_chain_hash": str(summary.get("concept_chain_hash") or ""),
            "ledger_hash": str(summary.get("ledger_hash") or ""),
            "summary_sha256": str(summary.get("summary_sha256") or ""),
        },
        "summary_sha256_file": str(summary_sha256_file),
        "negative_tamper": dict(negative),
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

    keys = list(sorted(r1["hashes"].keys()))
    mismatches: Dict[str, Any] = {}
    for k in keys:
        if str(r1["hashes"].get(k)) != str(r2["hashes"].get(k)):
            mismatches[k] = {"try1": str(r1["hashes"].get(k)), "try2": str(r2["hashes"].get(k))}

    ok_det = not bool(mismatches)
    out = {
        "ok": bool(r1.get("ok")) and bool(r2.get("ok")) and ok_det and bool(r1["negative_tamper"]["ok"]) and bool(r2["negative_tamper"]["ok"]),
        "determinism_ok": bool(ok_det),
        "hashes": dict(r1["hashes"]),
        "mismatches": dict(mismatches),
        "negative_tamper": {"ok": True, "reason": "agency_event_sig_mismatch"},
        "runs": {"try1": str(out1), "try2": str(out2)},
        "summary_sha256_file": str(r1.get("summary_sha256_file") or ""),
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
