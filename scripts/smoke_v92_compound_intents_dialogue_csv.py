#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v92 import run_conversation_v92
from atos_core.conversation_v92 import verify_conversation_chain_v92


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
        "por favor defina x como 4",
        "set y to 8 please",
        "get x please",
        "summary please",
        "set x to four; set y to 8; add x and y please",
        "blorp please",
        "set x please",
        "set x to 5",
        "add x and y",
        "set x to 4; set z",
        "resumo por favor",
        "end now",
    ]
    res = run_conversation_v92(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))

    turns = _extract_payloads(res["paths"]["turns_jsonl"])  # type: ignore[index]
    states = _extract_payloads(res["paths"]["states_jsonl"])  # type: ignore[index]
    parse_events = _extract_parse_events(res["paths"]["parses_jsonl"])  # type: ignore[index]
    trials = _extract_trials_for_verify(res["paths"]["trials_jsonl"])  # type: ignore[index]

    # Courtesy parsing assertions (prefix/suffix ignorable tokens).
    if not parse_events:
        _fail("ERROR: missing parse_events")
    first_parse = parse_events[0].get("payload") if isinstance(parse_events[0].get("payload"), dict) else {}
    if not bool(first_parse.get("parse_ok", False)):
        _fail("ERROR: expected first turn parse_ok=true (por favor defina x como 4)")
    if str(first_parse.get("intent_id") or "") != "INTENT_SET":
        _fail("ERROR: expected first turn intent_id=INTENT_SET")
    pref = first_parse.get("ignored_prefix_tokens")
    pref_list = pref if isinstance(pref, list) else []
    if [str(x) for x in pref_list] != ["por", "favor"]:
        _fail(f"ERROR: expected ignored_prefix_tokens ['por','favor'], got={pref_list}")

    # Compound success assertion (all segments executed; aggregated output).
    compound_text = "set x to four; set y to 8; add x and y please"
    uidx = _find_user_turn_index(turns, text=compound_text)
    if uidx < 0:
        _fail("ERROR: missing compound user turn")
    asst = _get_turn_by_index(turns, uidx + 1)
    if str(asst.get("role") or "") != "assistant":
        _fail("ERROR: expected assistant turn after compound user turn")
    want_compound_out = "OK: x=4\nOK: y=8\nSUM=12"
    if str(asst.get("text") or "") != want_compound_out:
        _fail(f"ERROR: compound output mismatch: want={want_compound_out!r} got={str(asst.get('text') or '')!r}")

    # Fail-closed unknown with courtesy token: must be COMM_CORRECT.
    bad_text = "blorp please"
    uidx2 = _find_user_turn_index(turns, text=bad_text)
    if uidx2 < 0:
        _fail("ERROR: missing blorp user turn")
    asst2 = _get_turn_by_index(turns, uidx2 + 1)
    if not str(asst2.get("text") or "").startswith("Comando inválido: blorp please"):
        _fail("ERROR: expected COMM_CORRECT response for 'blorp please'")

    # Compound fail-closed (all-or-nothing) on missing slots: no execution, ask clarify.
    bad_comp = "set x to 4; set z"
    uidx3 = _find_user_turn_index(turns, text=bad_comp)
    if uidx3 < 0:
        _fail("ERROR: missing compound-missing user turn")
    asst3 = _get_turn_by_index(turns, uidx3 + 1)
    if str(asst3.get("text") or "") != "Qual é o valor de z?":
        _fail("ERROR: expected ask-clarify for missing value of z in compound")

    # Courtesy token must NOT be consumed as SET value: "set x please" behaves as incomplete SET.
    set_x_please = "set x please"
    uidx4 = _find_user_turn_index(turns, text=set_x_please)
    if uidx4 < 0:
        _fail("ERROR: missing 'set x please' user turn")
    asst4 = _get_turn_by_index(turns, uidx4 + 1)
    if str(asst4.get("text") or "") != "Qual é o valor de x?":
        _fail("ERROR: expected ask-clarify for missing value of x in 'set x please'")
    # Clarification must NOT create a state update.
    uturn = _get_turn_by_index(turns, uidx4)
    utid = str(uturn.get("turn_id") or "")
    state_utids = {str(s.get("last_user_turn_id") or "") for s in states if isinstance(s, dict)}
    if utid and utid in state_utids:
        _fail("ERROR: state update created for clarification turn (set x please)")

    # Negative test: corrupt a user-turn parse_sig and ensure verify fails (fail-closed).
    if not turns:
        _fail("ERROR: missing turns payloads for negative test")
    bad_turns = [dict(t) for t in turns]
    bad_turns[0] = dict(bad_turns[0])
    refs = bad_turns[0].get("refs")
    refs = dict(refs) if isinstance(refs, dict) else {}
    refs["parse_sig"] = "0" * 64
    bad_turns[0]["refs"] = refs
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v92(
        turns=bad_turns, states=states, parse_events=parse_events, trials=trials, tail_k=6
    )
    if ok_bad:
        _fail("ERROR: expected verify_conversation_chain_v92 to fail on corrupted turn.refs.parse_sig")

    core = {
        "store_hash": str(res.get("store_hash") or ""),
        "transcript_hash": str(res.get("transcript_hash") or ""),
        "state_chain_hash": str(res.get("state_chain_hash") or ""),
        "parse_chain_hash": str(res.get("parse_chain_hash") or ""),
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

    keys = ["store_hash", "transcript_hash", "state_chain_hash", "parse_chain_hash", "ledger_hash", "summary_sha256"]
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
