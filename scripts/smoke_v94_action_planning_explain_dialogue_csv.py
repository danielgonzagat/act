#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v94 import run_conversation_v94
from atos_core.conversation_v94 import render_explain_text_v94, verify_conversation_chain_v94
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
    return [dict(r) for r in _read_jsonl(path) if isinstance(r, dict)]


def _extract_plans(path: str) -> List[Dict[str, Any]]:
    plans: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        d = dict(r)
        d.pop("prev_hash", None)
        d.pop("entry_hash", None)
        plans.append(d)
    return plans


def _find_user_turn_index(turns: List[Dict[str, Any]], *, text: str) -> int:
    for t in turns:
        if not isinstance(t, dict):
            continue
        if str(t.get("role") or "") != "user":
            continue
        if str(t.get("text") or "") == str(text):
            try:
                return int(t.get("turn_index") or 0)
            except Exception:
                return -1
    return -1


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
        "set x to 4",
        "explain",
        "summary",
        "explique:",
        "end now",
    ]
    res = run_conversation_v94(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    turns = _extract_payloads(res["paths"]["turns_jsonl"])  # type: ignore[index]
    states = _extract_payloads(res["paths"]["states_jsonl"])  # type: ignore[index]
    parse_events = _extract_parse_events(res["paths"]["parses_jsonl"])  # type: ignore[index]
    trials = _extract_trials_for_verify(res["paths"]["trials_jsonl"])  # type: ignore[index]
    learned_events = _extract_learned_events(res["paths"]["learned_rules_jsonl"])  # type: ignore[index]
    plans = _extract_plans(res["paths"]["plans_jsonl"])  # type: ignore[index]

    ok_good, reason_good, _details_good = verify_conversation_chain_v94(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        action_plans=plans,
        tail_k=6,
    )
    if not ok_good:
        _fail(f"ERROR: verify_conversation_chain_v94 failed: {reason_good}")

    # Must have one plan per user turn.
    user_turns_total = sum(1 for t in turns if isinstance(t, dict) and str(t.get("role") or "") == "user")
    if int(len(plans)) != int(user_turns_total):
        _fail(f"ERROR: plans_total mismatch: plans={len(plans)} user_turns={user_turns_total}")

    # explain must describe last non-explain plan before this turn.
    u_explain = _find_user_turn_index(turns, text="explain")
    if u_explain < 0:
        _fail("ERROR: missing 'explain' user turn")
    a_explain = _get_turn_by_index(turns, u_explain + 1)
    plan0 = _last_explainable_plan_before(plans, user_turn_index=u_explain)
    if plan0 is None:
        _fail("ERROR: missing explainable plan before 'explain'")
    want_explain = render_explain_text_v94(plan0)
    if str(a_explain.get("text") or "") != str(want_explain):
        _fail("ERROR: explain output mismatch for 'explain' turn")

    u_explique = _find_user_turn_index(turns, text="explique:")
    if u_explique < 0:
        _fail("ERROR: missing 'explique:' user turn")
    a_explique = _get_turn_by_index(turns, u_explique + 1)
    plan1 = _last_explainable_plan_before(plans, user_turn_index=u_explique)
    if plan1 is None:
        _fail("ERROR: missing explainable plan before 'explique:'")
    want_explique = render_explain_text_v94(plan1)
    if str(a_explique.get("text") or "") != str(want_explique):
        _fail("ERROR: explain output mismatch for 'explique:' turn")

    # Negative test: corrupt plan_sig and ensure verify fails deterministically.
    bad_plans = [dict(p) for p in plans]
    bad_plans[0] = dict(bad_plans[0])
    bad_plans[0]["plan_sig"] = "0" * 64
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v94(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_events,
        action_plans=bad_plans,
        tail_k=6,
    )
    if ok_bad:
        _fail("ERROR: expected verify_conversation_chain_v94 to fail on corrupted plan_sig")

    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "transcript_hash": str(res.get("transcript_hash") or ""),
        "state_chain_hash": str(res.get("state_chain_hash") or ""),
        "parse_chain_hash": str(res.get("parse_chain_hash") or ""),
        "learned_chain_hash": str(res.get("learned_chain_hash") or ""),
        "plan_chain_hash": str(res.get("plan_chain_hash") or ""),
        "ledger_hash": str(res.get("ledger_hash") or ""),
        "summary_sha256": str(res.get("summary_sha256") or ""),
        "negative_test": {"ok": True, "reason": str(reason_bad)},
        "sha256_summary_json": sha256_file(res["paths"]["summary_json"]),  # type: ignore[index]
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
