from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_v96 import verify_conversation_chain_v96


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _round6(x: Any) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return 0.0


def _canon_str_list(items: Any) -> List[str]:
    out: List[str] = []
    if not isinstance(items, list):
        return out
    for x in items:
        if isinstance(x, str) and x:
            out.append(str(x))
    return sorted(set(out))


def _canon_candidates_topk_v97(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        intent_id = str(it.get("intent_id") or "")
        if not intent_id:
            continue
        out.append(
            {
                "intent_id": intent_id,
                "score": _round6(it.get("score")),
                "reason": str(it.get("reason") or ""),
            }
        )
    # Deterministic order: score desc, id asc, reason asc.
    out.sort(key=lambda d: (-float(d.get("score", 0.0)), str(d.get("intent_id") or ""), str(d.get("reason") or "")))
    return list(out)


def _format_topk_line_v97(items: Sequence[Dict[str, Any]], k: int) -> str:
    parts: List[str] = []
    for i, it in enumerate(list(items)[: int(k)]):
        if not isinstance(it, dict):
            continue
        iid = str(it.get("intent_id") or "")
        sc = _round6(it.get("score"))
        parts.append(f"{i+1}){iid} score={sc:.6f}")
    return "TOPK: " + "; ".join(parts) if parts else "TOPK:"


def _format_attempted_line_v97(items: Any) -> str:
    attempted = items if isinstance(items, list) else []
    parts: List[str] = []
    for a in attempted[:5]:
        if not isinstance(a, dict):
            continue
        parts.append(f"{str(a.get('act_id') or '')}(ok={str(bool(a.get('ok', False))).lower()})")
    return "ATTEMPTED: " + "; ".join(parts) if parts else "ATTEMPTED:"


def render_explain_text_v97(plan: Dict[str, Any]) -> str:
    """
    Deterministic, audit-only explain text. No rationale beyond plan fields.
    """
    objective_kind = str(plan.get("objective_kind") or "")
    chosen_action_id = str(plan.get("chosen_action_id") or "")
    ok = bool(plan.get("chosen_ok", False))
    line1 = f"EXPLAIN: objective={objective_kind} chosen={chosen_action_id} ok={str(ok).lower()}"

    topk = _canon_candidates_topk_v97(plan.get("candidates_topk"))
    line2 = _format_topk_line_v97(topk, 5)

    line3 = _format_attempted_line_v97(plan.get("attempted_actions"))

    prov = plan.get("provenance") if isinstance(plan.get("provenance"), dict) else {}
    learned_ids = _canon_str_list(prov.get("learned_rule_ids"))
    line4 = f"PROVENANCE: learned_rule_ids={json.dumps(learned_ids, ensure_ascii=False)}"

    mem_read = _canon_str_list(plan.get("memory_read_ids"))
    mem_write = _canon_str_list(plan.get("memory_write_event_ids"))
    bel_read_keys = _canon_str_list(plan.get("belief_read_keys"))
    bel_read_ids = _canon_str_list(plan.get("belief_read_ids"))
    bel_write = _canon_str_list(plan.get("belief_write_event_ids"))
    line5 = (
        "EFFECTS: "
        + f"memory_read_ids={json.dumps(mem_read, ensure_ascii=False)} "
        + f"memory_write_event_ids={json.dumps(mem_write, ensure_ascii=False)} "
        + f"belief_read_keys={json.dumps(bel_read_keys, ensure_ascii=False)} "
        + f"belief_read_ids={json.dumps(bel_read_ids, ensure_ascii=False)} "
        + f"belief_write_event_ids={json.dumps(bel_write, ensure_ascii=False)}"
    )
    return "\n".join([line1, line2, line3, line4, line5])


@dataclass(frozen=True)
class SystemSpecV97:
    spec: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        body = dict(self.spec) if isinstance(self.spec, dict) else {}
        sem = dict(body)
        sem.setdefault("schema_version", 97)
        sem.setdefault("kind", "system_spec_v97")
        sig = _stable_hash_obj(sem)
        return dict(sem, spec_sig=str(sig))


def make_system_spec_v97() -> Dict[str, Any]:
    """
    Deterministic system spec snapshot for audit/repro (no runtime-dependent values).
    """
    spec = {
        "schema_version": 97,
        "kind": "system_spec_v97",
        "engine_id": "EngineV80",
        "core_modules": [
            "atos_core/act.py",
            "atos_core/store.py",
            "atos_core/engine_v80.py",
            "atos_core/intent_grammar_v92.py",
            "atos_core/intent_grammar_v93.py",
            "atos_core/intent_grammar_v94.py",
            "atos_core/intent_grammar_v96.py",
            "atos_core/intent_grammar_v97.py",
            "atos_core/conversation_loop_v97.py",
            "atos_core/conversation_v97.py",
        ],
        "data_flow": [
            "user_text -> raw_intercepts (TEACH/EXPLAIN/SYSTEM/NOTE/RECALL/BELIEF/...)",
            "else -> parse_intent_v92 (rules + learned_rules_active)",
            "-> objective_kind (COMM_*)",
            "-> rank action candidates via supports(G) stats",
            "-> execute concept_csv via EngineV80",
            "-> evaluate objective_csv_v88 (exact equality contract)",
            "-> append WORM logs (turns/parses/plans/states/trials/evals/ledgers)",
        ],
        "invariants": {
            "determinism": "seed=0; stable ordering; canonical_json_dumps; no wall-clock",
            "worm": "write-once files; jsonl logs are append-only hash-chained",
            "fail_closed": "unknown/ambiguous -> COMM_CORRECT/COMM_CONFIRM/COMM_ASK_CLARIFY; never invent",
        },
        "verification": {
            "verify_chains": "scripts/verify_conversation_chain_v97.py --run_dir <RUN_DIR>",
            "verify_freeze": "scripts/verify_freeze.py --freeze LEDGER_ATOLANG_V0_2_30_BASELINE_V97_SELF_EXPLAIN_LIVE_LEARNING_fix1.json | tee results/verify_freeze_v97_fix1_try1.json",
        },
        "repro": {
            "py_compile": "PYTHONPYCACHEPREFIX=$PWD/.pycache python3 -m py_compile atos_core/*.py scripts/*.py",
            "smoke": "PYTHONPYCACHEPREFIX=$PWD/.pycache python3 scripts/smoke_v97_self_explain_live_learning_dialogue_csv.py --out_base results/run_smoke_v97_self_explain_live_learning_fix1 --seed 0 | tee results/smoke_v97_self_explain_live_learning_fix1_try1.json",
        },
        "limits": [
            "No LLM / embeddings / gradient training",
            "No non-deterministic time sources",
            "No execution when parse is ambiguous/missing required slots",
        ],
        "formats": {
            "logs_jsonl": [
                "conversation_turns.jsonl",
                "intent_parses.jsonl",
                "learned_intent_rules.jsonl",
                "action_plans.jsonl",
                "memory_events.jsonl",
                "belief_events.jsonl",
                "conversation_states.jsonl",
                "dialogue_trials.jsonl",
                "objective_evals.jsonl",
                "transcript.jsonl",
            ],
            "snapshots_json": ["intent_grammar_snapshot.json", "system_spec_snapshot.json", "verify_chain_v97.json", "freeze_manifest_v97.json", "summary.json"],
        },
    }
    return SystemSpecV97(spec).to_dict()


def render_system_text_v97(spec: Dict[str, Any]) -> str:
    """
    Deterministic, audit-oriented system self-description. No dynamic state.
    """
    engine_id = str(spec.get("engine_id") or "")
    inv = spec.get("invariants") if isinstance(spec.get("invariants"), dict) else {}
    det = str(inv.get("determinism") or "")
    worm = str(inv.get("worm") or "")
    failc = str(inv.get("fail_closed") or "")
    repro = spec.get("repro") if isinstance(spec.get("repro"), dict) else {}
    cmd_compile = str(repro.get("py_compile") or "")
    cmd_smoke = str(repro.get("smoke") or "")
    ver = spec.get("verification") if isinstance(spec.get("verification"), dict) else {}
    cmd_verify = str(ver.get("verify_chains") or "")
    cmd_freeze = str(ver.get("verify_freeze") or "")

    lines = [
        f"SYSTEM: engine_id={engine_id}",
        f"INVARIANTS: determinism={det}; worm={worm}; fail_closed={failc}",
        "REPRO:",
        f"1) {cmd_compile}",
        f"2) {cmd_smoke}",
        "VERIFY:",
        f"1) {cmd_verify}",
        f"2) {cmd_freeze}",
        "LIMITS: no_llm; no_random_time; no_execute_on_ambiguous_parse",
    ]
    return "\n".join([str(x) for x in lines])


def verify_conversation_chain_v97(
    *,
    turns: Sequence[Dict[str, Any]],
    states: Sequence[Dict[str, Any]],
    parse_events: Sequence[Dict[str, Any]],
    trials: Sequence[Dict[str, Any]],
    learned_rule_events: Sequence[Dict[str, Any]],
    action_plans: Sequence[Dict[str, Any]],
    memory_events: Sequence[Dict[str, Any]],
    belief_events: Sequence[Dict[str, Any]],
    tail_k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    V97 invariants:
      - All V96 invariants must hold.
      - If a user turn matches a learned rule, plan.provenance.learned_rule_ids must include it.
    """
    ok0, reason0, details0 = verify_conversation_chain_v96(
        turns=turns,
        states=states,
        parse_events=parse_events,
        trials=trials,
        learned_rule_events=learned_rule_events,
        action_plans=action_plans,
        memory_events=memory_events,
        belief_events=belief_events,
        tail_k=tail_k,
    )
    if not ok0:
        return False, str(reason0), dict(details0)

    learned_ids: List[str] = []
    for ev in learned_rule_events:
        if not isinstance(ev, dict):
            continue
        lr = ev.get("learned_rule")
        if isinstance(lr, dict):
            rid = str(lr.get("rule_id") or "")
            if rid:
                learned_ids.append(rid)
    learned_set = set(learned_ids)

    parses_by_turn_id: Dict[str, Dict[str, Any]] = {}
    for pe in parse_events:
        if not isinstance(pe, dict):
            continue
        tid = str(pe.get("turn_id") or "")
        payload = pe.get("payload") if isinstance(pe.get("payload"), dict) else {}
        if tid:
            parses_by_turn_id[tid] = dict(payload)

    plans_by_user_turn_id: Dict[str, Dict[str, Any]] = {}
    for p in action_plans:
        if not isinstance(p, dict):
            continue
        utid = str(p.get("user_turn_id") or "")
        if utid:
            plans_by_user_turn_id[utid] = dict(p)

    for utid, payload in parses_by_turn_id.items():
        mrid = str(payload.get("matched_rule_id") or "")
        if not mrid or mrid not in learned_set:
            continue
        plan = plans_by_user_turn_id.get(utid)
        if not isinstance(plan, dict):
            return False, "missing_plan_for_learned_rule_turn", {"turn_id": utid, "learned_rule_id": mrid}
        prov = plan.get("provenance") if isinstance(plan.get("provenance"), dict) else None
        if not isinstance(prov, dict):
            return False, "missing_plan_provenance", {"turn_id": utid}
        ids_in_plan = prov.get("learned_rule_ids")
        ids_list = _canon_str_list(ids_in_plan)
        if mrid not in set(ids_list):
            return False, "missing_learned_rule_id_in_provenance", {"turn_id": utid, "learned_rule_id": mrid, "got": list(ids_list)}

    return True, "ok", dict(details0)

