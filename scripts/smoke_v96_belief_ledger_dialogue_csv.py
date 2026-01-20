#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v96 import run_conversation_v96
from atos_core.conversation_v96 import render_explain_text_v96, verify_conversation_chain_v96
from atos_core.intent_grammar_v94 import INTENT_EXPLAIN_V94


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
    rows = _read_jsonl(path)
    out: List[Dict[str, Any]] = []
    for r in rows:
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


def _find_user_turn_indices(turns: List[Dict[str, Any]], *, text: str) -> List[int]:
    out: List[int] = []
    for t in turns:
        if not isinstance(t, dict):
            continue
        if str(t.get("role") or "") != "user":
            continue
        if str(t.get("text") or "") != str(text):
            continue
        try:
            out.append(int(t.get("turn_index") or 0))
        except Exception:
            continue
    return list(out)


def _get_turn_by_index(turns: List[Dict[str, Any]], idx: int) -> Dict[str, Any]:
    for t in turns:
        if not isinstance(t, dict):
            continue
        try:
            if int(t.get("turn_index") or 0) == int(idx):
                return dict(t)
        except Exception:
            continue
    return {}


def _last_explainable_plan_before(plans: List[Dict[str, Any]], *, user_turn_index: int) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    for p in plans:
        if not isinstance(p, dict):
            continue
        try:
            uidx = int(p.get("user_turn_index", -1))
        except Exception:
            uidx = -1
        if uidx < 0 or uidx >= int(user_turn_index):
            continue
        if str(p.get("intent_id") or "") == INTENT_EXPLAIN_V94:
            continue
        if best is None:
            best = dict(p)
            continue
        try:
            if int(best.get("user_turn_index", -1)) < uidx:
                best = dict(p)
        except Exception:
            best = dict(p)
    return best


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    user_turns = [
        "belief: project = IAAA",
        "beliefs",
        "get project",
        "revise: project = IAAA_v2",
        "beliefs",
        "forget belief project",
        "beliefs",
        "note: meu nome é Daniel",
        "recall",
        "explain",
        "end now",
    ]
    res = run_conversation_v96(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    turns = _extract_payloads(res["paths"]["turns_jsonl"])  # type: ignore[index]
    states = _extract_payloads(res["paths"]["states_jsonl"])  # type: ignore[index]
    parse_events = _extract_parse_events(res["paths"]["parses_jsonl"])  # type: ignore[index]
    trials = _extract_trials_for_verify(res["paths"]["trials_jsonl"])  # type: ignore[index]
    learned_events = _extract_learned_events(res["paths"]["learned_rules_jsonl"])  # type: ignore[index]
    plans = _strip_chain_fields(res["paths"]["plans_jsonl"])  # type: ignore[index]
    memory_events = _strip_chain_fields(res["paths"]["memory_events_jsonl"])  # type: ignore[index]
    belief_events = _strip_chain_fields(res["paths"]["belief_events_jsonl"])  # type: ignore[index]

    ok_good, reason_good, _details_good = verify_conversation_chain_v96(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        action_plans=plans,
        memory_events=memory_events,
        belief_events=belief_events,
        tail_k=6,
    )
    if not ok_good:
        _fail(f"ERROR: verify_conversation_chain_v96 failed: {reason_good}")

    beliefs_idxs = _find_user_turn_indices(turns, text="beliefs")
    if len(beliefs_idxs) != 3:
        _fail(f"ERROR: expected 3 beliefs turns, got {len(beliefs_idxs)}")

    a_beliefs_1 = _get_turn_by_index(turns, beliefs_idxs[0] + 1)
    t1 = str(a_beliefs_1.get("text") or "")
    if "key=project" not in t1 or 'value="IAAA"' not in t1:
        _fail("ERROR: beliefs#1 missing project=IAAA")

    get_idxs = _find_user_turn_indices(turns, text="get project")
    if len(get_idxs) != 1:
        _fail("ERROR: expected exactly 1 get project turn")
    a_get = _get_turn_by_index(turns, get_idxs[0] + 1)
    if str(a_get.get("text") or "") != "project=IAAA":
        _fail("ERROR: get project output mismatch")

    a_beliefs_2 = _get_turn_by_index(turns, beliefs_idxs[1] + 1)
    t2 = str(a_beliefs_2.get("text") or "")
    if 'value="IAAA_v2"' not in t2:
        _fail("ERROR: beliefs#2 missing revised value IAAA_v2")
    if 'value="IAAA"' in t2:
        _fail("ERROR: beliefs#2 unexpectedly contains old value IAAA")

    a_beliefs_3 = _get_turn_by_index(turns, beliefs_idxs[2] + 1)
    if str(a_beliefs_3.get("text") or "") != "BELIEFS: (empty)":
        _fail("ERROR: beliefs#3 expected empty")

    recall_idxs = _find_user_turn_indices(turns, text="recall")
    if len(recall_idxs) != 1:
        _fail("ERROR: expected exactly 1 recall turn")
    a_recall = _get_turn_by_index(turns, recall_idxs[0] + 1)
    rt = str(a_recall.get("text") or "")
    if not rt.startswith("MEMORY:"):
        _fail("ERROR: recall assistant text must start with 'MEMORY:'")
    if "meu nome é Daniel" not in rt:
        _fail("ERROR: recall assistant text missing memory payload")

    explain_idxs = _find_user_turn_indices(turns, text="explain")
    if len(explain_idxs) != 1:
        _fail("ERROR: expected exactly 1 explain turn")
    u_explain = explain_idxs[0]
    a_explain = _get_turn_by_index(turns, u_explain + 1)
    plan0 = _last_explainable_plan_before(plans, user_turn_index=u_explain)
    if plan0 is None:
        _fail("ERROR: missing explainable plan before 'explain'")
    want_explain = render_explain_text_v96(plan0)
    if str(a_explain.get("text") or "") != str(want_explain):
        _fail("ERROR: explain output mismatch")

    # Negative test: corrupt first belief event_sig and ensure verify fails deterministically.
    if not belief_events:
        _fail("ERROR: expected at least 1 belief event")
    bad_belief_events = [dict(ev) for ev in belief_events]
    bad_belief_events[0] = dict(bad_belief_events[0])
    bad_belief_events[0]["event_sig"] = "0" * 64
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v96(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        action_plans=plans,
        memory_events=memory_events,
        belief_events=bad_belief_events,
        tail_k=6,
    )
    if ok_bad:
        _fail("ERROR: expected verify_conversation_chain_v96 to fail on corrupted belief event_sig")
    if str(reason_bad) != "belief_event_sig_mismatch":
        _fail(f"ERROR: expected reason belief_event_sig_mismatch, got {reason_bad}")

    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "transcript_hash": str(res.get("transcript_hash") or ""),
        "state_chain_hash": str(res.get("state_chain_hash") or ""),
        "parse_chain_hash": str(res.get("parse_chain_hash") or ""),
        "learned_chain_hash": str(res.get("learned_chain_hash") or ""),
        "plan_chain_hash": str(res.get("plan_chain_hash") or ""),
        "memory_chain_hash": str(res.get("memory_chain_hash") or ""),
        "belief_chain_hash": str(res.get("belief_chain_hash") or ""),
        "ledger_hash": str(res.get("ledger_hash") or ""),
        "summary_sha256": str(res.get("summary_sha256") or ""),
        "negative_test": {"ok": True, "reason": str(reason_bad)},
        "sha256_summary_json": sha256_file(res["paths"]["summary_json"]),  # type: ignore[index]
        "sha256_manifest_json": sha256_file(res["paths"]["manifest_json"]),  # type: ignore[index]
        "sha256_memory_events_jsonl": sha256_file(res["paths"]["memory_events_jsonl"]),  # type: ignore[index]
        "sha256_belief_events_jsonl": sha256_file(res["paths"]["belief_events_jsonl"]),  # type: ignore[index]
        "sha256_plans_jsonl": sha256_file(res["paths"]["plans_jsonl"]),  # type: ignore[index]
        "sha256_turns_jsonl": sha256_file(res["paths"]["turns_jsonl"]),  # type: ignore[index]
        "sha256_parses_jsonl": sha256_file(res["paths"]["parses_jsonl"]),  # type: ignore[index]
        "sha256_states_jsonl": sha256_file(res["paths"]["states_jsonl"]),  # type: ignore[index]
        "sha256_transcript_jsonl": sha256_file(res["paths"]["transcript_jsonl"]),  # type: ignore[index]
    }
    return dict(res, core=core)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    out1 = f"{out_base}_try1"
    out2 = f"{out_base}_try2"
    ensure_absent(out1)
    ensure_absent(out2)

    r1 = smoke_try(out_dir=out1, seed=seed)
    r2 = smoke_try(out_dir=out2, seed=seed)

    keys = [
        "store_hash",
        "transcript_hash",
        "state_chain_hash",
        "parse_chain_hash",
        "learned_chain_hash",
        "plan_chain_hash",
        "memory_chain_hash",
        "belief_chain_hash",
        "ledger_hash",
        "summary_sha256",
    ]
    for k in keys:
        if str(r1.get(k) or "") != str(r2.get(k) or ""):
            _fail(f"ERROR: determinism mismatch for {k}: try1={r1.get(k)} try2={r2.get(k)}")

    out = {
        "ok": True,
        "seed": int(seed),
        "determinism_ok": True,
        "determinism": {k: str(r1.get(k) or "") for k in keys},
        "negative_test": dict(r1.get("core", {}).get("negative_test", {})),
        "try1": {"out_dir": str(out1), "summary_sha256": str(r1.get("summary_sha256") or "")},
        "try2": {"out_dir": str(out2), "summary_sha256": str(r2.get("summary_sha256") or "")},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

