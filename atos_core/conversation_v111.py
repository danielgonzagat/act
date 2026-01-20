from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Sequence, Tuple

from .conversation_v100 import no_hybridization_check_v100
from .conversation_v110 import verify_conversation_chain_v110
from .conversation_v96 import verify_chained_jsonl_v96
from .external_world_ledger_v111 import compute_external_world_chain_hash_v111, verify_external_world_event_sig_chain_v111


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def verify_conversation_chain_v111(*, run_dir: str, repo_root: str) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Wrapper over V110 verifier plus V111 checks:
      - external_world_events.jsonl exists and is hash-chained + sig-chained.
      - fluency_contract_v111.json exists and ok==true.
      - no-hybridization guardrail.
    """
    ok_h, reason_h, details_h = no_hybridization_check_v100(repo_root=str(repo_root))
    if not ok_h:
        return False, str(reason_h), dict(details_h)

    rd = str(run_dir)
    required = [
        os.path.join(rd, "conversation_turns.jsonl"),
        os.path.join(rd, "conversation_states.jsonl"),
        os.path.join(rd, "intent_parses.jsonl"),
        os.path.join(rd, "dialogue_trials.jsonl"),
        os.path.join(rd, "learned_intent_rules.jsonl"),
        os.path.join(rd, "action_plans.jsonl"),
        os.path.join(rd, "memory_events.jsonl"),
        os.path.join(rd, "belief_events.jsonl"),
        os.path.join(rd, "evidence_events.jsonl"),
        os.path.join(rd, "goal_events.jsonl"),
        os.path.join(rd, "goal_ledger_snapshot.json"),
        os.path.join(rd, "discourse_events.jsonl"),
        os.path.join(rd, "fragment_events.jsonl"),
    ]
    for p in required:
        if not os.path.exists(p):
            return False, "missing_path", {"path": str(p)}

    turns = _load_jsonl(os.path.join(rd, "conversation_turns.jsonl"))
    states = _load_jsonl(os.path.join(rd, "conversation_states.jsonl"))
    parse_events = _load_jsonl(os.path.join(rd, "intent_parses.jsonl"))
    trials = _load_jsonl(os.path.join(rd, "dialogue_trials.jsonl"))
    learned_rule_events = _load_jsonl(os.path.join(rd, "learned_intent_rules.jsonl"))
    action_plans = _load_jsonl(os.path.join(rd, "action_plans.jsonl"))
    memory_events = _load_jsonl(os.path.join(rd, "memory_events.jsonl"))
    belief_events = _load_jsonl(os.path.join(rd, "belief_events.jsonl"))
    evidence_events = _load_jsonl(os.path.join(rd, "evidence_events.jsonl"))
    goal_events = _load_jsonl(os.path.join(rd, "goal_events.jsonl"))
    goal_snapshot = json.loads(open(os.path.join(rd, "goal_ledger_snapshot.json"), "r", encoding="utf-8").read())
    discourse_events = _load_jsonl(os.path.join(rd, "discourse_events.jsonl"))
    fragment_events = _load_jsonl(os.path.join(rd, "fragment_events.jsonl"))

    # Infer tail_k.
    tail_k = 6
    for st in states[-3:]:
        if not isinstance(st, dict):
            continue
        inv = st.get("invariants") if isinstance(st.get("invariants"), dict) else {}
        try:
            tail_k = int(inv.get("tail_k") or tail_k)
        except Exception:
            pass

    ok0, reason0, details0 = verify_conversation_chain_v110(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_rule_events,
        action_plans=action_plans,
        memory_events=memory_events,
        belief_events=belief_events,
        evidence_events=evidence_events,
        goal_events=goal_events,
        goal_snapshot=dict(goal_snapshot),
        discourse_events=discourse_events,
        fragment_events=fragment_events,
        tail_k=int(tail_k),
        repo_root=str(repo_root),
    )
    if not ok0:
        return False, str(reason0), dict(details0)

    # External world ledger (may be empty, but must exist and be verifiable).
    ext_path = os.path.join(rd, "external_world_events.jsonl")
    if not os.path.exists(ext_path):
        return False, "missing_external_world_ledger", {"path": str(ext_path)}
    if not bool(verify_chained_jsonl_v96(str(ext_path))):
        return False, "external_world_file_chain_invalid", {}
    ext_events = _load_jsonl(str(ext_path))
    ok_sig, reason_sig, details_sig = verify_external_world_event_sig_chain_v111(ext_events)
    if not ok_sig:
        return False, str(reason_sig), dict(details_sig)
    ext_chain_hash = compute_external_world_chain_hash_v111(ext_events)

    # Fluency contract file.
    fc_path = os.path.join(rd, "fluency_contract_v111.json")
    if not os.path.exists(fc_path):
        return False, "missing_fluency_contract", {"path": str(fc_path)}
    fc = json.loads(open(fc_path, "r", encoding="utf-8").read())
    if not bool(fc.get("ok", False)):
        return False, "fluency_contract_failed", {"reason": str(fc.get("reason") or "")}

    return True, "ok", {"external_world_chain_hash_v111": str(ext_chain_hash), "external_world_events_total": int(len(ext_events))}

