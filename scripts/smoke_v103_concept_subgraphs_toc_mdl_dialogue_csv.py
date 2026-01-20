#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v103 import run_conversation_v103
from atos_core.conversation_v103 import verify_conversation_chain_v103


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
        "style_events": style_events,
        "template_snapshot": template_snapshot,
        "concept_events": concept_events,
        "concept_snapshot": concept_snapshot,
    }


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # V103 demo: teach concept (pos/neg) -> induce -> list/explain -> ToC pass on different domain -> trace.
    user_turns = [
        'teach_concept: IRONIA += "Nossa, que ótimo... só que não."',
        'teach_concept: IRONIA += "Claro, eu adoro esperar 3 horas na fila, só que não."',
        'teach_concept: IRONIA -= "Que ótimo, gostei muito mesmo."',
        "concepts",
        "explain_concept IRONIA",
        "Esse deploy foi perfeito, sqn.",
        "trace_concepts 10",
        "end now",
    ]
    res = run_conversation_v103(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))
    loaded = _load_for_verify(res)
    repo_root = os.path.dirname(os.path.dirname(__file__))

    ok_good, reason_good, details_good = verify_conversation_chain_v103(
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
        goal_snapshot=dict(loaded["goal_snapshot"]),
        discourse_events=loaded["discourse_events"],
        fragment_events=loaded["fragment_events"],
        binding_events=loaded["binding_events"],
        binding_snapshot=dict(loaded["binding_snapshot"]),
        style_events=loaded["style_events"],
        template_snapshot=dict(loaded["template_snapshot"]),
        concept_events=loaded["concept_events"],
        concept_snapshot=dict(loaded["concept_snapshot"]),
        tail_k=6,
        repo_root=str(repo_root),
    )
    if not ok_good:
        _fail("ERROR: verify_conversation_chain_v103 failed: " + json.dumps({"reason": reason_good, "details": details_good}, ensure_ascii=False, sort_keys=True))

    # Assert ToC pass recorded.
    toc_pass = [ev for ev in loaded["concept_events"] if isinstance(ev, dict) and str(ev.get("type") or "") == "CONCEPT_TOC_PASS"]
    if not toc_pass:
        _fail("ERROR: expected at least 1 CONCEPT_TOC_PASS event")

    # Negative tamper: concept_events[0].event_sig corrupted.
    bad_concept_events = [dict(ev) for ev in loaded["concept_events"]]
    if not bad_concept_events:
        _fail("ERROR: expected concept_events non-empty")
    bad_concept_events[0]["event_sig"] = "0" * 64
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v103(
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
        goal_snapshot=dict(loaded["goal_snapshot"]),
        discourse_events=loaded["discourse_events"],
        fragment_events=loaded["fragment_events"],
        binding_events=loaded["binding_events"],
        binding_snapshot=dict(loaded["binding_snapshot"]),
        style_events=loaded["style_events"],
        template_snapshot=dict(loaded["template_snapshot"]),
        concept_events=bad_concept_events,
        concept_snapshot=dict(loaded["concept_snapshot"]),
        tail_k=6,
        repo_root=str(repo_root),
    )
    if ok_bad:
        _fail("ERROR: expected tampered concept_events to fail verification")
    if str(reason_bad) != "concept_event_sig_mismatch":
        _fail(f"ERROR: expected tamper reason concept_event_sig_mismatch, got: {reason_bad}")

    core = {
        "seed": int(seed),
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
        "style_chain_hash": str(res.get("style_chain_hash") or ""),
        "concept_chain_hash": str(res.get("concept_chain_hash") or ""),
        "ledger_hash": str(res.get("ledger_hash") or ""),
        "summary_sha256": str(res.get("summary_sha256") or ""),
    }
    return {
        "core": dict(core),
        "tamper": {"ok": True, "reason": str(reason_bad)},
        "sha256": {
            "concept_events_jsonl": sha256_file(str(res["paths"]["concept_events_jsonl"])),  # type: ignore[index]
            "concept_snapshot_json": sha256_file(str(res["paths"]["concept_library_snapshot_v103_json"])),  # type: ignore[index]
            "verify_chain_json": sha256_file(str(res["paths"]["verify_json"])),  # type: ignore[index]
            "manifest_json": sha256_file(str(res["paths"]["manifest_json"])),  # type: ignore[index]
        },
        "concept_events_total": int(len(loaded["concept_events"])),
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

    determinism_ok = bool(r1.get("core") == r2.get("core"))
    if not determinism_ok:
        _fail(
            "ERROR: determinism failed (try1 != try2): "
            + json.dumps({"try1": r1.get("core"), "try2": r2.get("core")}, ensure_ascii=False, sort_keys=True)
        )

    out = {"ok": True, "determinism_ok": True, "try1": dict(r1), "try2": dict(r2)}
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

