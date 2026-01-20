#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.conversation_loop_v104 import run_conversation_v104
from atos_core.conversation_v104 import verify_conversation_chain_v104


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
    # Turns are payloads with turn_index monotonic; assistant is at user_turn_index+1.
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
    plans = _strip_chain_fields(str(paths.get("plans_jsonl") or ""))
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
    return {
        "turns": turns,
        "states": states,
        "parse_events": parse_events,
        "trials": trials,
        "learned_events": learned_events,
        "plans": plans,
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
    }


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # Teach pragmatic concepts -> use as planner features -> show explain_plan/trace_plans/trace_concepts -> multi-turn plan persistence.
    user_turns: List[str] = []

    # Teach concepts (2 pos + 1 neg each).
    user_turns += [
        # NOTE: avoid number-words like "um"/"uma" here because V92 tokenization maps them to digits ("1"),
        # and the V103 MDL rule inducer may pick overly-broad features like t:1.
        'teach_concept: USER_WANTS_PLAN += "quero plano passo a passo"',
        'teach_concept: USER_WANTS_PLAN += "preciso de plano em passos"',
        'teach_concept: USER_WANTS_PLAN -= "ok obrigado"',
        # Make USER_WANTS_OPTIONS discriminate by "opcoes" (neg does not contain it).
        'teach_concept: USER_WANTS_OPTIONS += "quero opcoes"',
        'teach_concept: USER_WANTS_OPTIONS += "mostre opcoes"',
        'teach_concept: USER_WANTS_OPTIONS -= "nao entendi"',
        # Confusion concept based on "confuso" token.
        'teach_concept: USER_CONFUSED += "estou confuso"',
        'teach_concept: USER_CONFUSED += "muito confuso"',
        'teach_concept: USER_CONFUSED -= "entendi perfeitamente"',
    ]

    # Unknown input that triggers compare_options override (concept bonus).
    options_turn_pos = len(user_turns)
    user_turns.append("quais sao as opcoes por favor")
    options_turn_index = int(options_turn_pos * 2)

    # A transfer-like input for USER_WANTS_PLAN (different surface); it should still match and produce ToC pass if distance is high.
    transfer_turn_pos = len(user_turns)
    user_turns.append("preciso de um plano agora")
    transfer_turn_index = int(transfer_turn_pos * 2)

    # Plan create command (strict grammar) repeated to prove active plan persistence across turns (same topic_sig).
    plan_cmd = "crie um plano de 3 passos para demo"
    plan1_pos = len(user_turns)
    user_turns.append(plan_cmd)
    plan1_turn_index = int(plan1_pos * 2)
    plan2_pos = len(user_turns)
    user_turns.append(plan_cmd)
    plan2_turn_index = int(plan2_pos * 2)

    # Introspection: plans + explain/trace on the options turn + trace_concepts for options turn.
    user_turns.append("plans")
    user_turns.append(f"explain_plan {options_turn_index}")
    user_turns.append(f"trace_plans {options_turn_index}")
    user_turns.append(f"trace_concepts {options_turn_index}")

    # Also trace_concepts for the transfer input to prove ToC path can fire in a different domain.
    user_turns.append(f"trace_concepts {transfer_turn_index}")

    # End.
    user_turns.append("end now")

    res = run_conversation_v104(user_turn_texts=list(user_turns), out_dir=str(out_dir), seed=int(seed))
    loaded = _load_for_verify(res)
    repo_root = os.path.dirname(os.path.dirname(__file__))

    ok_good, reason_good, details_good = verify_conversation_chain_v104(
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
        plan_events=loaded["plan_events"],
        plan_snapshot=dict(loaded["plan_snapshot"]),
        tail_k=6,
        repo_root=str(repo_root),
    )
    if not ok_good:
        _fail(
            "ERROR: verify_conversation_chain_v104 failed: "
            + json.dumps({"reason": reason_good, "details": details_good}, ensure_ascii=False, sort_keys=True)
        )

    # Assert options override produced an options list.
    opt_text = _assistant_text_after_user_turn(turns=loaded["turns"], user_turn_index=int(options_turn_index))
    if "OPÇÕES:" not in opt_text:
        _fail("ERROR: expected options response to include 'OPÇÕES:'")

    # Assert plans output shows active plan and step_index incremented to 1 after the repeated plan command.
    # plans command is after plan2; its user_turn_index is (plan2_pos+1)*2.
    plans_cmd_turn_index = int((plan2_pos + 1) * 2)
    plans_text = _assistant_text_after_user_turn(turns=loaded["turns"], user_turn_index=int(plans_cmd_turn_index))
    if "PLANS:" not in plans_text or "step_index=1" not in plans_text or "kind=propose_steps" not in plans_text:
        _fail("ERROR: expected plans output to include kind=propose_steps and step_index=1")

    # Assert explain_plan output for options turn includes concept hits and chosen plan kind compare_options.
    explain_cmd_turn_index = int((plan2_pos + 2) * 2)
    explain_text = _assistant_text_after_user_turn(turns=loaded["turns"], user_turn_index=int(explain_cmd_turn_index))
    if "EXPLAIN_PLAN:" not in explain_text:
        _fail("ERROR: expected explain_plan output")
    if "chosen=compare_options" not in explain_text:
        _fail("ERROR: expected chosen plan_kind compare_options in explain_plan output")
    if "USER_WANTS_OPTIONS" not in explain_text:
        _fail("ERROR: expected concept hit USER_WANTS_OPTIONS in explain_plan output")

    # Negative tamper: plan_events[0].event_sig corrupted.
    bad_plan_events = [dict(ev) for ev in loaded["plan_events"]]
    if not bad_plan_events:
        _fail("ERROR: expected plan_events non-empty")
    bad_plan_events[0]["event_sig"] = "0" * 64
    ok_bad, reason_bad, _details_bad = verify_conversation_chain_v104(
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
        plan_events=bad_plan_events,
        plan_snapshot=dict(loaded["plan_snapshot"]),
        tail_k=6,
        repo_root=str(repo_root),
    )
    if ok_bad or str(reason_bad) != "plan_event_sig_mismatch":
        _fail("ERROR: expected plan tamper to fail with reason plan_event_sig_mismatch, got: " + json.dumps({"ok": ok_bad, "reason": reason_bad}))

    return {
        "res": res,
        "details_good": details_good,
        "negative_tamper_reason": str(reason_bad),
        "turn_indices": {
            "options_turn_index": int(options_turn_index),
            "transfer_turn_index": int(transfer_turn_index),
            "plan1_turn_index": int(plan1_turn_index),
            "plan2_turn_index": int(plan2_turn_index),
        },
    }


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

    t1 = smoke_try(out_dir=out1, seed=seed)
    t2 = smoke_try(out_dir=out2, seed=seed)

    r1 = t1["res"]
    r2 = t2["res"]

    keys = [
        "store_hash",
        "transcript_hash",
        "state_chain_hash",
        "parse_chain_hash",
        "learned_chain_hash",
        "plan_chain_hash",
        "plan_events_chain_hash_v104",
        "memory_chain_hash",
        "belief_chain_hash",
        "evidence_chain_hash",
        "goal_chain_hash",
        "discourse_chain_hash",
        "fragment_chain_hash",
        "binding_chain_hash",
        "style_chain_hash",
        "concept_chain_hash",
        "ledger_hash",
        "summary_sha256",
    ]
    mismatches: Dict[str, Any] = {}
    for k in keys:
        if str(r1.get(k) or "") != str(r2.get(k) or ""):
            mismatches[k] = {"try1": str(r1.get(k) or ""), "try2": str(r2.get(k) or "")}

    determinism_ok = not bool(mismatches)

    # Stable smoke summary hash (canonical JSON).
    core = {k: str(r1.get(k) or "") for k in keys}
    core["negative_tamper_reason"] = str(t1.get("negative_tamper_reason") or "")
    core["turn_indices"] = dict(t1.get("turn_indices") or {})
    summary_sha256 = sha256_file(str(r1.get("paths", {}).get("summary_json") or os.path.join(out1, "summary.json")))

    final = {
        "ok": bool(determinism_ok),
        "determinism_ok": bool(determinism_ok),
        "mismatches": dict(mismatches),
        "hashes": dict(core),
        "summary_sha256_file": str(summary_sha256),
        "runs": {"try1": out1, "try2": out2},
        "negative_tamper": {"ok": True, "reason": str(t1.get("negative_tamper_reason") or "")},
    }
    print(json.dumps(final, ensure_ascii=False, sort_keys=True, indent=2))
    if not determinism_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
