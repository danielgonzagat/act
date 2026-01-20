#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v100 import run_conversation_v100
from atos_core.conversation_v100 import verify_conversation_chain_v100


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
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        d = dict(r)
        d.pop("prev_hash", None)
        d.pop("entry_hash", None)
        out.append(d)
    return out


def _find_user_turn_index(turns: List[Dict[str, Any]], *, text: str) -> Optional[int]:
    for t in turns:
        if not isinstance(t, dict):
            continue
        if str(t.get("role") or "") != "user":
            continue
        if str(t.get("text") or "") != str(text):
            continue
        try:
            return int(t.get("turn_index") or 0)
        except Exception:
            return None
    return None


def _assistant_text_after(turns: List[Dict[str, Any]], *, user_text: str) -> str:
    idx = _find_user_turn_index(turns, text=user_text)
    if idx is None:
        return ""
    for t in turns:
        if not isinstance(t, dict):
            continue
        if str(t.get("role") or "") != "assistant":
            continue
        try:
            if int(t.get("turn_index") or 0) == int(idx + 1):
                return str(t.get("text") or "")
        except Exception:
            continue
    return ""


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
    }


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    user_turns = [
        "belief: alpha = 1",
        "belief: beta = 2",
        "porque isso",
        "set x to 4",
        "set y to 8",
        "add x and y please",
        "explain",
        "discourse",
        "dossier",
        "end now",
    ]
    res = run_conversation_v100(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    # Load run artifacts for verifier + assertions.
    loaded = _load_for_verify(res)
    repo_root = os.path.dirname(os.path.dirname(__file__))

    ok_good, reason_good, _details_good = verify_conversation_chain_v100(
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
        goal_snapshot=loaded["goal_snapshot"],
        discourse_events=loaded["discourse_events"],
        fragment_events=loaded["fragment_events"],
        tail_k=6,
        repo_root=str(repo_root),
    )
    if not ok_good:
        _fail(f"ERROR: verify_conversation_chain_v100 failed: {reason_good}")

    turns = loaded["turns"]
    why_reply = _assistant_text_after(turns, user_text="porque isso")
    if not why_reply.startswith("Qual chave?"):
        _fail("ERROR: expected WHY_REF to ask clarification (Qual chave?)")

    # Must have >= 3 candidates in discourse events and select a labeled variant at least once.
    discourse_events = loaded["discourse_events"]
    if not discourse_events:
        _fail("ERROR: expected discourse_events non-empty")
    has_wrapped = False
    for ev in discourse_events:
        if not isinstance(ev, dict):
            continue
        cands = ev.get("candidates_topk")
        if not isinstance(cands, list) or len(cands) < 3:
            _fail("ERROR: expected >=3 discourse candidates per assistant turn")
        sel_id = str(ev.get("selected_candidate_id") or "")
        sel = None
        for c in cands:
            if isinstance(c, dict) and str(c.get("candidate_id") or "") == sel_id:
                sel = c
                break
        if isinstance(sel, dict) and str(sel.get("text") or "").startswith("Resposta:"):
            frags = sel.get("fragment_ids") if isinstance(sel.get("fragment_ids"), list) else []
            if "frag_v100_prefix_answer_v0" in [str(x) for x in frags]:
                has_wrapped = True
                break
    if not has_wrapped:
        _fail("ERROR: expected at least one wrapped selected candidate using frag_v100_prefix_answer_v0")

    # Fragment promotion must occur (mini-cycle of concepts vivos).
    frag_snap_path = str(res["paths"]["fragment_library_snapshot_json"])  # type: ignore[index]
    snap = _read_json(frag_snap_path)
    frags = snap.get("fragments") if isinstance(snap.get("fragments"), list) else []
    st = None
    for f in frags:
        if isinstance(f, dict) and str(f.get("fragment_id") or "") == "frag_v100_prefix_answer_v0":
            st = dict(f)
            break
    if not isinstance(st, dict):
        _fail("ERROR: fragment snapshot missing frag_v100_prefix_answer_v0")
    if str(st.get("promotion_state") or "") != "promoted":
        _fail("ERROR: expected frag_v100_prefix_answer_v0 promotion_state=promoted")
    if int(st.get("usage_wins") or 0) < 3:
        _fail("ERROR: expected frag_v100_prefix_answer_v0 usage_wins>=3")

    # Negative tamper: corrupt discourse event sig in-memory and expect fail-closed reason.
    tampered_discourse = [dict(x) for x in discourse_events if isinstance(x, dict)]
    if not tampered_discourse:
        _fail("ERROR: no discourse events to tamper")
    tampered_discourse[0]["event_sig"] = "0" * 64
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v100(
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
        goal_snapshot=loaded["goal_snapshot"],
        discourse_events=tampered_discourse,
        fragment_events=loaded["fragment_events"],
        tail_k=6,
        repo_root=str(repo_root),
    )
    if ok_bad or str(reason_bad) != "discourse_event_sig_mismatch":
        _fail(f"ERROR: expected negative tamper failure discourse_event_sig_mismatch, got ok={ok_bad} reason={reason_bad}")

    # Build stable result summary for determinism check.
    result = {
        "out_dir": str(out_dir),
        "store_hash": str(res.get("store_hash") or ""),
        "system_spec_sha256": str(res.get("system_spec_sha256") or ""),
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
        "ledger_hash": str(res.get("ledger_hash") or ""),
        "summary_sha256": str(res.get("summary_sha256") or ""),
        "negative_tamper_reason": str(reason_bad),
        "promoted_fragment_id": "frag_v100_prefix_answer_v0",
    }
    return dict(result)


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

    keys = [
        "store_hash",
        "system_spec_sha256",
        "transcript_hash",
        "state_chain_hash",
        "parse_chain_hash",
        "learned_chain_hash",
        "plan_chain_hash",
        "memory_chain_hash",
        "belief_chain_hash",
        "evidence_chain_hash",
        "goal_chain_hash",
        "discourse_chain_hash",
        "fragment_chain_hash",
        "ledger_hash",
        "summary_sha256",
    ]
    mismatch: List[Dict[str, str]] = []
    for k in keys:
        if str(r1.get(k) or "") != str(r2.get(k) or ""):
            mismatch.append({"key": str(k), "try1": str(r1.get(k) or ""), "try2": str(r2.get(k) or "")})
    if mismatch:
        _fail("ERROR: determinism mismatch\n" + json.dumps(mismatch, ensure_ascii=False, indent=2, sort_keys=True))

    smoke_core = {k: str(r1.get(k) or "") for k in keys}
    smoke_core["negative_tamper_reason"] = str(r1.get("negative_tamper_reason") or "")
    smoke_core["promoted_fragment_id"] = str(r1.get("promoted_fragment_id") or "")
    summary_sha256 = sha256_file(os.path.join(out1, "summary.json"))

    out = {
        "ok": True,
        "determinism_ok": True,
        "out_base": str(out_base),
        "tries": {"try1": str(out1), "try2": str(out2)},
        "hashes": dict(smoke_core),
        "summary_json_try1_sha256": str(summary_sha256),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

