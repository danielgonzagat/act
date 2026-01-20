#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v97 import run_conversation_v97
from atos_core.conversation_v97 import render_explain_text_v97, render_system_text_v97, verify_conversation_chain_v97
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


def _plan_for_user_turn(plans: List[Dict[str, Any]], *, user_turn_index: int) -> Optional[Dict[str, Any]]:
    for p in plans:
        if not isinstance(p, dict):
            continue
        try:
            if int(p.get("user_turn_index", -1)) == int(user_turn_index):
                return dict(p)
        except Exception:
            continue
    return None


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    user_turns = [
        "system",
        "teach: statusz => beliefs",
        "belief: project = IAAA",
        "statusz",
        "explain",
        "end now",
    ]
    res = run_conversation_v97(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    turns = _extract_payloads(res["paths"]["turns_jsonl"])  # type: ignore[index]
    states = _extract_payloads(res["paths"]["states_jsonl"])  # type: ignore[index]
    parse_events = _extract_parse_events(res["paths"]["parses_jsonl"])  # type: ignore[index]
    trials = _extract_trials_for_verify(res["paths"]["trials_jsonl"])  # type: ignore[index]
    learned_events = _extract_learned_events(res["paths"]["learned_rules_jsonl"])  # type: ignore[index]
    plans = _strip_chain_fields(res["paths"]["plans_jsonl"])  # type: ignore[index]
    memory_events = _strip_chain_fields(res["paths"]["memory_events_jsonl"])  # type: ignore[index]
    belief_events = _strip_chain_fields(res["paths"]["belief_events_jsonl"])  # type: ignore[index]

    ok_good, reason_good, _details_good = verify_conversation_chain_v97(
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
        _fail(f"ERROR: verify_conversation_chain_v97 failed: {reason_good}")

    # SYSTEM output must be exactly the deterministic renderer of the stored snapshot.
    sys_idxs = _find_user_turn_indices(turns, text="system")
    if len(sys_idxs) != 1:
        _fail("ERROR: expected exactly 1 system user turn")
    a_sys = _get_turn_by_index(turns, sys_idxs[0] + 1)
    sys_spec_path = res["paths"]["system_spec_snapshot_json"]  # type: ignore[index]
    sys_spec = _read_json(str(sys_spec_path))
    want_sys = render_system_text_v97(sys_spec)
    if str(a_sys.get("text") or "") != str(want_sys):
        _fail("ERROR: system output mismatch vs render_system_text_v97(system_spec_snapshot.json)")

    # TEACH must create exactly 1 learned rule event, and it must map to BELIEFS.
    if len(learned_events) != 1:
        _fail(f"ERROR: expected exactly 1 learned rule event, got {len(learned_events)}")
    lr_ev = dict(learned_events[0])
    lr = lr_ev.get("learned_rule")
    if not isinstance(lr, dict):
        _fail("ERROR: learned_rule missing in learned event")
    learned_rule_id = str(lr.get("rule_id") or "")
    if not learned_rule_id:
        _fail("ERROR: learned rule_id empty")
    if str(lr.get("intent_id") or "") != "INTENT_BELIEF_LIST":
        _fail("ERROR: learned rule intent_id expected INTENT_BELIEF_LIST")

    # statusz must be parsed by the learned rule and list beliefs.
    statusz_idxs = _find_user_turn_indices(turns, text="statusz")
    if len(statusz_idxs) != 1:
        _fail("ERROR: expected exactly 1 statusz user turn")
    u_statusz = statusz_idxs[0]
    u_statusz_turn = _get_turn_by_index(turns, u_statusz)
    urefs = u_statusz_turn.get("refs") if isinstance(u_statusz_turn.get("refs"), dict) else {}
    used_rule_id = str(urefs.get("matched_rule_id") or "")
    if used_rule_id != learned_rule_id:
        _fail(f"ERROR: statusz matched_rule_id mismatch: want {learned_rule_id} got {used_rule_id}")
    p_statusz = _plan_for_user_turn(plans, user_turn_index=u_statusz)
    if p_statusz is None:
        _fail("ERROR: missing action plan for statusz user turn")
    prov = p_statusz.get("provenance") if isinstance(p_statusz.get("provenance"), dict) else {}
    prov_lrids = prov.get("learned_rule_ids") if isinstance(prov.get("learned_rule_ids"), list) else []
    prov_lrids = [str(x) for x in prov_lrids if isinstance(x, str) and x]
    if prov_lrids != [learned_rule_id]:
        _fail(f"ERROR: plan.provenance.learned_rule_ids mismatch: want [{learned_rule_id}] got {prov_lrids}")

    a_statusz = _get_turn_by_index(turns, u_statusz + 1)
    st = str(a_statusz.get("text") or "")
    if not st.startswith("BELIEFS:"):
        _fail("ERROR: statusz assistant response must start with 'BELIEFS:'")
    if "key=project" not in st or 'value="IAAA"' not in st:
        _fail("ERROR: statusz beliefs output missing project=IAAA")

    # EXPLAIN must match deterministic renderer of last non-explain plan (statusz).
    explain_idxs = _find_user_turn_indices(turns, text="explain")
    if len(explain_idxs) != 1:
        _fail("ERROR: expected exactly 1 explain user turn")
    u_explain = explain_idxs[0]
    a_explain = _get_turn_by_index(turns, u_explain + 1)
    plan0 = _last_explainable_plan_before(plans, user_turn_index=u_explain)
    if plan0 is None:
        _fail("ERROR: missing explainable plan before 'explain'")
    want_explain = render_explain_text_v97(plan0)
    if str(a_explain.get("text") or "") != str(want_explain):
        _fail("ERROR: explain output mismatch")

    # Negative test: corrupt learned_rule.rule_sig and ensure verify fails deterministically.
    if not learned_events:
        _fail("ERROR: expected at least 1 learned event")
    bad_learned_events = [dict(ev) for ev in learned_events]
    bad_learned_events[0] = dict(bad_learned_events[0])
    bad_lr = dict(bad_learned_events[0].get("learned_rule") or {}) if isinstance(bad_learned_events[0].get("learned_rule"), dict) else {}
    bad_lr["rule_sig"] = "0" * 64
    bad_learned_events[0]["learned_rule"] = dict(bad_lr)
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v97(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=bad_learned_events,
        action_plans=plans,
        memory_events=memory_events,
        belief_events=belief_events,
        tail_k=6,
    )
    if ok_bad:
        _fail("ERROR: expected verify_conversation_chain_v97 to fail on corrupted learned rule_sig")
    if str(reason_bad) != "rule_sig_mismatch":
        _fail(f"ERROR: expected reason rule_sig_mismatch, got {reason_bad}")

    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "system_spec_sha256": str(res.get("system_spec_sha256") or ""),
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
        "sha256_system_spec_snapshot_json": sha256_file(res["paths"]["system_spec_snapshot_json"]),  # type: ignore[index]
        "sha256_learned_rules_jsonl": sha256_file(res["paths"]["learned_rules_jsonl"]),  # type: ignore[index]
        "sha256_plans_jsonl": sha256_file(res["paths"]["plans_jsonl"]),  # type: ignore[index]
        "sha256_turns_jsonl": sha256_file(res["paths"]["turns_jsonl"]),  # type: ignore[index]
        "sha256_parses_jsonl": sha256_file(res["paths"]["parses_jsonl"]),  # type: ignore[index]
        "sha256_states_jsonl": sha256_file(res["paths"]["states_jsonl"]),  # type: ignore[index]
        "sha256_transcript_jsonl": sha256_file(res["paths"]["transcript_jsonl"]),  # type: ignore[index]
        "sha256_belief_events_jsonl": sha256_file(res["paths"]["belief_events_jsonl"]),  # type: ignore[index]
        "sha256_memory_events_jsonl": sha256_file(res["paths"]["memory_events_jsonl"]),  # type: ignore[index]
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
        "system_spec_sha256",
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

