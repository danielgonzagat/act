#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v93 import run_conversation_v93
from atos_core.conversation_v93 import verify_conversation_chain_v93


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
    events: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        events.append(
            {
                "turn_id": str(r.get("turn_id") or ""),
                "turn_index": int(r.get("turn_index") or 0),
                "payload": dict(r.get("payload") or {}) if isinstance(r.get("payload"), dict) else {},
            }
        )
    return events


def _extract_trials_for_verify(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in _read_jsonl(path):
        if not isinstance(r, dict):
            continue
        out.append(
            {
                "objective_kind": str(r.get("objective_kind") or ""),
                "user_turn_id": str(r.get("user_turn_id") or r.get("turn_id") or ""),
            }
        )
    return out


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


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    user_turns = [
        "show all vars",
        "teach: show all vars => summary",
        "show all vars",
        "teach: foo => set x to 4",
        "teach: summary => summary",
        "set x to 4",
        "show all vars",
        "ensine: bye => end now",
        "bye",
    ]
    res = run_conversation_v93(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    turns = _extract_payloads(res["paths"]["turns_jsonl"])  # type: ignore[index]
    states = _extract_payloads(res["paths"]["states_jsonl"])  # type: ignore[index]
    parse_events = _extract_parse_events(res["paths"]["parses_jsonl"])  # type: ignore[index]
    trials = _extract_trials_for_verify(res["paths"]["trials_jsonl"])  # type: ignore[index]
    learned_rows = _read_jsonl(res["paths"]["learned_rules_jsonl"]) if os.path.exists(res["paths"]["learned_rules_jsonl"]) else []  # type: ignore[index]

    # Pre-teach: "show all vars" must be fail-closed (COMM_CORRECT).
    u0 = _find_user_turn_index(turns, text="show all vars")
    if u0 < 0:
        _fail("ERROR: missing first 'show all vars' user turn")
    a0 = _get_turn_by_index(turns, u0 + 1)
    if str(a0.get("text") or "") != "Comando inválido: show all vars":
        _fail("ERROR: expected COMM_CORRECT for pre-teach 'show all vars'")

    # Teach accepted must create learned event and state update.
    teach_text = "teach: show all vars => summary"
    u1 = _find_user_turn_index(turns, text=teach_text)
    if u1 < 0:
        _fail("ERROR: missing TEACH user turn")
    ut1 = _get_turn_by_index(turns, u1)
    ut1_id = str(ut1.get("turn_id") or "")
    pe1 = None
    for pe in parse_events:
        if str(pe.get("turn_index") or -1) == str(u1):
            pe1 = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
            break
    if pe1 is None:
        _fail("ERROR: missing parse payload for TEACH")
    if str(pe1.get("intent_id") or "") != "INTENT_TEACH":
        _fail("ERROR: expected intent_id=INTENT_TEACH for TEACH parse payload")
    if not bool(pe1.get("teach_ok", False)):
        _fail("ERROR: expected teach_ok=true for TEACH accepted")
    learned_rule_id = str(pe1.get("learned_rule_id") or "")
    if not learned_rule_id:
        _fail("ERROR: missing learned_rule_id in TEACH accepted payload")
    if not any(isinstance(r, dict) and str(r.get("teacher_turn_id") or "") == ut1_id for r in learned_rows):
        _fail("ERROR: missing learned rule event for TEACH accepted (teacher_turn_id)")
    state_utids = {str(s.get("last_user_turn_id") or "") for s in states if isinstance(s, dict)}
    if ut1_id not in state_utids:
        _fail("ERROR: TEACH accepted should create state update")
    found_state = False
    for s in states:
        if not isinstance(s, dict):
            continue
        if str(s.get("last_user_turn_id") or "") != ut1_id:
            continue
        bindings = s.get("bindings") if isinstance(s.get("bindings"), dict) else {}
        lra = bindings.get("learned_rules_active")
        lra_list = [str(x) for x in lra if isinstance(lra, list) and isinstance(x, str) and x]
        if learned_rule_id not in set(lra_list):
            _fail("ERROR: state.bindings.learned_rules_active missing learned_rule_id")
        found_state = True
        break
    if not found_state:
        _fail("ERROR: missing state row for TEACH accepted user turn")

    # Post-teach: "show all vars" becomes SUMMARY.
    # This is the second occurrence after teach (turn index u1+2).
    u2 = -1
    seen = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        if str(t.get("role") or "") != "user":
            continue
        if str(t.get("text") or "") != "show all vars":
            continue
        seen += 1
        if seen == 2:
            u2 = int(t.get("turn_index") or -1)
            break
    if u2 < 0:
        _fail("ERROR: missing post-teach 'show all vars' occurrence")
    a2 = _get_turn_by_index(turns, u2 + 1)
    if str(a2.get("text") or "") != "Resumo:":
        _fail("ERROR: expected SUMMARY output after teach (empty vars)")
    # Parse must match learned rule id.
    pe2 = None
    for pe in parse_events:
        if int(pe.get("turn_index") or -1) == int(u2):
            pe2 = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
            break
    if pe2 is None or str(pe2.get("matched_rule_id") or "") != learned_rule_id:
        _fail("ERROR: expected matched_rule_id==learned_rule_id for post-teach parse")

    # Invalid TEACH (rhs has slots) must be rejected and must NOT update state.
    bad_teach = "teach: foo => set x to 4"
    ubad = _find_user_turn_index(turns, text=bad_teach)
    if ubad < 0:
        _fail("ERROR: missing invalid TEACH user turn")
    utbad = _get_turn_by_index(turns, ubad)
    utbad_id = str(utbad.get("turn_id") or "")
    if utbad_id and utbad_id in state_utids:
        _fail("ERROR: invalid TEACH must NOT create state update")

    # Ambiguous TEACH must be rejected with COMM_CONFIRM and NOT update state.
    amb_teach = "teach: summary => summary"
    uamb = _find_user_turn_index(turns, text=amb_teach)
    if uamb < 0:
        _fail("ERROR: missing ambiguous TEACH user turn")
    aamb = _get_turn_by_index(turns, uamb + 1)
    if not str(aamb.get("text") or "").startswith("Confirme"):
        _fail("ERROR: expected COMM_CONFIRM output for ambiguous TEACH")
    utamb = _get_turn_by_index(turns, uamb)
    utamb_id = str(utamb.get("turn_id") or "")
    if utamb_id and utamb_id in state_utids:
        _fail("ERROR: ambiguous TEACH must NOT create state update")

    # Normal SET + SUMMARY still works with learned rule.
    uset = _find_user_turn_index(turns, text="set x to 4")
    if uset < 0:
        _fail("ERROR: missing set x to 4")
    # third occurrence of show all vars includes x=4
    u3 = -1
    seen = 0
    for t in turns:
        if not isinstance(t, dict):
            continue
        if str(t.get("role") or "") != "user":
            continue
        if str(t.get("text") or "") != "show all vars":
            continue
        seen += 1
        if seen == 3:
            u3 = int(t.get("turn_index") or -1)
            break
    if u3 < 0:
        _fail("ERROR: missing third 'show all vars' occurrence")
    a3 = _get_turn_by_index(turns, u3 + 1)
    if str(a3.get("text") or "") != "Resumo: x=4":
        _fail("ERROR: expected SUMMARY output including x=4")

    # Teach END alias and ensure it ends.
    ubye = _find_user_turn_index(turns, text="bye")
    if ubye < 0:
        _fail("ERROR: missing bye user turn")
    aby = _get_turn_by_index(turns, ubye + 1)
    if str(aby.get("text") or "") != "Encerrado.":
        _fail("ERROR: expected conversation to end with 'Encerrado.'")

    # Negative test: corrupt learned rule_sig and ensure verifier fails deterministically.
    bad_learned = [dict(x) for x in learned_rows]
    if not bad_learned:
        _fail("ERROR: expected at least 1 learned rule for negative test")
    bad_learned[0] = dict(bad_learned[0])
    lr = bad_learned[0].get("learned_rule")
    lr = dict(lr) if isinstance(lr, dict) else {}
    lr["rule_sig"] = "0" * 64
    bad_learned[0]["learned_rule"] = lr
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v93(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=bad_learned,
        tail_k=6,
    )
    if ok_bad:
        _fail("ERROR: expected verify_conversation_chain_v93 to fail on corrupted learned rule_sig")

    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "transcript_hash": str(res.get("transcript_hash") or ""),
        "state_chain_hash": str(res.get("state_chain_hash") or ""),
        "parse_chain_hash": str(res.get("parse_chain_hash") or ""),
        "learned_chain_hash": str(res.get("learned_chain_hash") or ""),
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

    keys = ["store_hash", "transcript_hash", "state_chain_hash", "parse_chain_hash", "learned_chain_hash", "ledger_hash", "summary_sha256"]
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
