from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .conversation_v96 import _belief_active_after_step_v96, verify_conversation_chain_v96
from .intent_grammar_v98 import INTENT_EVIDENCE_ADD_V98, INTENT_EVIDENCE_LIST_V98, INTENT_WHY_V98


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


def sha256_file_v98(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def evidence_item_sig_v98(item_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(item_sem_sig).encode("utf-8"))


def evidence_item_id_v98(evidence_sig: str) -> str:
    return f"evidence_v98_{str(evidence_sig)}"


@dataclass(frozen=True)
class EvidenceItemV98:
    conversation_id: str
    evidence_kind: str  # "OBSERVE"
    evidence_key: str
    evidence_value: str
    source_turn_id: str
    created_step: int

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "schema_version": 98,
            "kind": "evidence_item_v98",
            "conversation_id": str(self.conversation_id),
            "evidence_kind": str(self.evidence_kind),
            "evidence_key": str(self.evidence_key).strip(),
            "evidence_value": str(self.evidence_value).strip(),
            "source_turn_id": str(self.source_turn_id),
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
        }
        sig = evidence_item_sig_v98(sem)
        eid = evidence_item_id_v98(sig)
        return dict(sem, evidence_sig=str(sig), evidence_id=str(eid))


def evidence_event_sig_v98(event_sem_sig: Dict[str, Any]) -> str:
    return sha256_hex(canonical_json_dumps(event_sem_sig).encode("utf-8"))


def evidence_event_id_v98(event_sig: str) -> str:
    return f"evidence_event_v98_{str(event_sig)}"


@dataclass(frozen=True)
class EvidenceEventV98:
    conversation_id: str
    evidence_item: Dict[str, Any]
    created_step: int
    source_turn_id: str

    def to_dict(self) -> Dict[str, Any]:
        sem = {
            "schema_version": 98,
            "kind": "evidence_event_v98",
            "conversation_id": str(self.conversation_id),
            "event_kind": "OBSERVE",
            "created_step": int(self.created_step),
            "created_at": deterministic_iso(step=int(self.created_step)),
            "source_turn_id": str(self.source_turn_id or ""),
            "evidence_item": dict(self.evidence_item) if isinstance(self.evidence_item, dict) else None,
        }
        sig = evidence_event_sig_v98(sem)
        eid = evidence_event_id_v98(sig)
        return dict(sem, event_sig=str(sig), event_id=str(eid))


def compute_evidence_chain_hash_v98(evidence_events: Sequence[Dict[str, Any]]) -> str:
    ids: List[str] = []
    for ev in evidence_events:
        if not isinstance(ev, dict):
            continue
        item = ev.get("evidence_item")
        if isinstance(item, dict):
            ids.append(str(item.get("evidence_id") or ""))
    return sha256_hex(canonical_json_dumps(ids).encode("utf-8"))


def _verify_evidence_item_sig_v98(item: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    d = dict(item)
    got_sig = str(d.pop("evidence_sig", "") or "")
    got_id = str(d.pop("evidence_id", "") or "")
    if not got_sig:
        return False, "missing_evidence_sig", {}
    want_sig = sha256_hex(canonical_json_dumps(d).encode("utf-8"))
    if want_sig != got_sig:
        return False, "evidence_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = evidence_item_id_v98(got_sig)
    if want_id != got_id:
        return False, "evidence_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    return True, "ok", {}


def _verify_evidence_event_sig_v98(ev: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
    d = dict(ev)
    got_sig = str(d.pop("event_sig", "") or "")
    got_id = str(d.pop("event_id", "") or "")
    if not got_sig:
        return False, "missing_evidence_event_sig", {}
    want_sig = sha256_hex(canonical_json_dumps(d).encode("utf-8"))
    if want_sig != got_sig:
        return False, "evidence_event_sig_mismatch", {"want": str(want_sig), "got": str(got_sig)}
    want_id = evidence_event_id_v98(got_sig)
    if want_id != got_id:
        return False, "evidence_event_id_mismatch", {"want": str(want_id), "got": str(got_id)}
    item = ev.get("evidence_item")
    if not isinstance(item, dict):
        return False, "evidence_event_missing_item", {"event_id": got_id}
    ok_item, rreason, rdetails = _verify_evidence_item_sig_v98(dict(item))
    if not ok_item:
        dd = dict(rdetails)
        dd["event_id"] = got_id
        return False, str(rreason), dd
    return True, "ok", {}


def render_evidence_added_ack_text_v98(*, evidence_id: str, key: str) -> str:
    return f"EVIDENCE ADDED: {str(evidence_id)} key={str(key)}"


def render_evidences_text_v98(evidence_events: Sequence[Dict[str, Any]]) -> str:
    items: List[Dict[str, Any]] = []
    for ev in evidence_events:
        if not isinstance(ev, dict):
            continue
        it = ev.get("evidence_item")
        if isinstance(it, dict):
            items.append(dict(it))
    items.sort(key=lambda d: (int(d.get("created_step", 0) or 0), str(d.get("evidence_id") or "")))
    if not items:
        return "EVIDENCES: (empty)"
    lines: List[str] = ["EVIDENCES:"]
    for i, it in enumerate(items, start=1):
        key = str(it.get("evidence_key") or "")
        eid = str(it.get("evidence_id") or "")
        kind = str(it.get("evidence_kind") or "")
        val = str(it.get("evidence_value") or "")
        lines.append(f'{i}) key={key} id={eid} kind={kind} value={json.dumps(val, ensure_ascii=False)}')
    return "\n".join(lines)


def render_why_belief_text_v98(
    *,
    key: str,
    belief_events: Sequence[Dict[str, Any]],
    evidence_events: Sequence[Dict[str, Any]],
    step: int,
) -> str:
    """
    Deterministic explanation for belief(key) based on belief/evidence ledgers up to created_step<=step.
    """
    bkey = str(key or "").strip()
    active_beliefs, all_beliefs = _belief_active_after_step_v96(belief_events=belief_events, step=int(step))
    cur = active_beliefs.get(bkey) if isinstance(active_beliefs.get(bkey), dict) else None
    if not isinstance(cur, dict):
        return f"WHY: key={bkey} (no_active_belief)"

    cur_id = str(cur.get("belief_id") or "")
    cur_val = str(cur.get("belief_value") or "")

    # Find last evidence-driven ADD for this key.
    last_add: Optional[Dict[str, Any]] = None
    for ev in reversed(list(belief_events)):
        if not isinstance(ev, dict):
            continue
        try:
            cstep = int(ev.get("created_step", -1))
        except Exception:
            cstep = -1
        if cstep < 0 or cstep > int(step):
            continue
        if str(ev.get("event_kind") or "") != "ADD":
            continue
        bi = ev.get("belief_item")
        if not isinstance(bi, dict):
            continue
        if str(bi.get("belief_key") or "").strip() != bkey:
            continue
        if str(ev.get("cause_evidence_id") or ""):
            last_add = dict(ev)
            break

    eid = str(last_add.get("cause_evidence_id") or "") if isinstance(last_add, dict) else ""
    old_id = ""
    if eid:
        for ev in reversed(list(belief_events)):
            if not isinstance(ev, dict):
                continue
            try:
                cstep = int(ev.get("created_step", -1))
            except Exception:
                cstep = -1
            if cstep < 0 or cstep > int(step):
                continue
            if str(ev.get("event_kind") or "") != "RETRACT":
                continue
            if str(ev.get("cause_evidence_id") or "") != eid:
                continue
            old_id = str(ev.get("target_belief_id") or "")
            break

    old_val = ""
    if old_id and old_id in all_beliefs:
        old_val = str(all_beliefs[old_id].get("belief_value") or "")

    evidence_by_id: Dict[str, Dict[str, Any]] = {}
    for ev in evidence_events:
        if not isinstance(ev, dict):
            continue
        it = ev.get("evidence_item")
        if not isinstance(it, dict):
            continue
        evid = str(it.get("evidence_id") or "")
        if evid:
            evidence_by_id[evid] = dict(it)
    eitem = evidence_by_id.get(eid) if eid else None
    evalue = str(eitem.get("evidence_value") or "") if isinstance(eitem, dict) else ""

    lines: List[str] = [f"WHY: key={bkey}"]
    if old_id:
        lines.append(f'WAS: id={old_id} value={json.dumps(old_val, ensure_ascii=False)}')
    else:
        lines.append("WAS: (none)")
    lines.append(f'NOW: id={cur_id} value={json.dumps(cur_val, ensure_ascii=False)}')
    if eid:
        lines.append(f'CAUSE: evidence_id={eid} value={json.dumps(evalue, ensure_ascii=False)}')
    else:
        lines.append("CAUSE: (no_evidence)")
    return "\n".join(lines)


def render_versions_text_v98(*, repo_root: str) -> str:
    root = str(repo_root)
    paths: List[str] = []
    for name in os.listdir(root):
        if not isinstance(name, str):
            continue
        if name.startswith("LEDGER_") and name.endswith(".json"):
            paths.append(name)
        if name.startswith("STATUS_PACK_") and name.endswith(".md"):
            paths.append(name)
    patches_dir = os.path.join(root, "patches")
    if os.path.isdir(patches_dir):
        for name in os.listdir(patches_dir):
            if not isinstance(name, str):
                continue
            if name.startswith("v") and name.endswith(".diff"):
                paths.append(os.path.join("patches", name))
    paths = sorted(set(paths), key=str)
    lines: List[str] = ["VERSIONS:"]
    for i, p in enumerate(paths, start=1):
        abs_p = os.path.join(root, p) if not os.path.isabs(p) else p
        if not os.path.exists(abs_p):
            continue
        lines.append(f"{i}) {p} sha256={sha256_file_v98(abs_p)}")
    return "\n".join(lines)


def render_dossier_text_v98() -> str:
    """
    Deterministic short dossier: how to verify/replay. No marketing.
    """
    lines = [
        "DOSSIER:",
        "1) Record-keeping: WORM write-once files + append-only hash-chained JSONL logs.",
        "2) Replay: run verifier scripts on run_dir (e.g., scripts/verify_conversation_chain_v98.py --run_dir <RUN_DIR>).",
        "3) Freeze: verify ledgers via scripts/verify_freeze.py --freeze <LEDGER_JSON> (warnings must be []).",
        "4) Determinism: seed=0; canonical_json_dumps; explicit sorting; no wall-clock time.",
        "5) Fail-closed: ambiguous/invalid commands do not mutate state; they produce deterministic correction/clarification.",
    ]
    return "\n".join(lines)


def _canon_candidates_topk_v98(items: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not isinstance(items, list):
        return out
    for it in items:
        if not isinstance(it, dict):
            continue
        act_id = str(it.get("act_id") or it.get("intent_id") or "")
        if not act_id:
            continue
        out.append(
            {
                "act_id": act_id,
                "expected_success": _round6(it.get("expected_success")),
                "expected_cost": _round6(it.get("expected_cost")),
                "score": _round6(it.get("score")),
                "reason": str(it.get("reason") or ""),
            }
        )
    out.sort(key=lambda d: (-float(d.get("score", 0.0)), str(d.get("act_id") or "")))
    return list(out)


def _format_topk_line_v98(items: Sequence[Dict[str, Any]], k: int) -> str:
    parts: List[str] = []
    for i, it in enumerate(list(items)[: int(k)]):
        if not isinstance(it, dict):
            continue
        aid = str(it.get("act_id") or "")
        s = _round6(it.get("expected_success"))
        c = _round6(it.get("expected_cost"))
        sc = _round6(it.get("score"))
        parts.append(f"{i+1}){aid} s={s:.6f} c={c:.6f} score={sc:.6f}")
    return "TOPK: " + "; ".join(parts) if parts else "TOPK:"


def _format_attempted_line_v98(items: Any) -> str:
    attempted = items if isinstance(items, list) else []
    parts: List[str] = []
    for a in attempted[:5]:
        if not isinstance(a, dict):
            continue
        parts.append(f"{str(a.get('act_id') or '')}(ok={str(bool(a.get('ok', False))).lower()})")
    return "ATTEMPTED: " + "; ".join(parts) if parts else "ATTEMPTED:"


def render_explain_text_v98(plan: Dict[str, Any]) -> str:
    """
    Deterministic, audit-only explain text. No rationale beyond plan fields.
    """
    objective_kind = str(plan.get("objective_kind") or "")
    chosen_action_id = str(plan.get("chosen_action_id") or "")
    ok = bool(plan.get("chosen_ok", False))
    line1 = f"EXPLAIN: objective={objective_kind} chosen={chosen_action_id} ok={str(ok).lower()}"

    topk = _canon_candidates_topk_v98(plan.get("candidates_topk"))
    line2 = _format_topk_line_v98(topk, 5)

    line3 = _format_attempted_line_v98(plan.get("attempted_actions"))

    prov = plan.get("provenance") if isinstance(plan.get("provenance"), dict) else {}
    learned_ids = _canon_str_list(prov.get("learned_rule_ids"))
    cause_eids = _canon_str_list(prov.get("cause_evidence_ids"))
    line4 = (
        "PROVENANCE: "
        + f"learned_rule_ids={json.dumps(learned_ids, ensure_ascii=False)} "
        + f"cause_evidence_ids={json.dumps(cause_eids, ensure_ascii=False)}"
    )

    mem_read = _canon_str_list(plan.get("memory_read_ids"))
    mem_write = _canon_str_list(plan.get("memory_write_event_ids"))
    bel_read_keys = _canon_str_list(plan.get("belief_read_keys"))
    bel_read_ids = _canon_str_list(plan.get("belief_read_ids"))
    bel_write = _canon_str_list(plan.get("belief_write_event_ids"))
    ev_read = _canon_str_list(plan.get("evidence_read_ids"))
    ev_write = _canon_str_list(plan.get("evidence_write_ids"))
    line5 = (
        "EFFECTS: "
        + f"memory_read_ids={json.dumps(mem_read, ensure_ascii=False)} "
        + f"memory_write_event_ids={json.dumps(mem_write, ensure_ascii=False)} "
        + f"belief_read_keys={json.dumps(bel_read_keys, ensure_ascii=False)} "
        + f"belief_read_ids={json.dumps(bel_read_ids, ensure_ascii=False)} "
        + f"belief_write_event_ids={json.dumps(bel_write, ensure_ascii=False)} "
        + f"evidence_read_ids={json.dumps(ev_read, ensure_ascii=False)} "
        + f"evidence_write_ids={json.dumps(ev_write, ensure_ascii=False)}"
    )
    return "\n".join([line1, line2, line3, line4, line5])


def verify_conversation_chain_v98(
    *,
    turns: Sequence[Dict[str, Any]],
    states: Sequence[Dict[str, Any]],
    parse_events: Sequence[Dict[str, Any]],
    trials: Sequence[Dict[str, Any]],
    learned_rule_events: Sequence[Dict[str, Any]],
    action_plans: Sequence[Dict[str, Any]],
    memory_events: Sequence[Dict[str, Any]],
    belief_events: Sequence[Dict[str, Any]],
    evidence_events: Sequence[Dict[str, Any]],
    tail_k: int,
) -> Tuple[bool, str, Dict[str, Any]]:
    ok0, reason0, details0 = verify_conversation_chain_v97(
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

    # Verify evidence ledger sigs and build mappings.
    evidence_item_by_id: Dict[str, Dict[str, Any]] = {}
    evidence_events_by_turn: Dict[str, List[Dict[str, Any]]] = {}
    prev_step = -1
    for ev in evidence_events:
        if not isinstance(ev, dict):
            return False, "evidence_event_not_dict", {}
        ok_ev, ereason, edetails = _verify_evidence_event_sig_v98(dict(ev))
        if not ok_ev:
            return False, str(ereason), dict(edetails)
        try:
            cstep = int(ev.get("created_step", -1))
        except Exception:
            cstep = -1
        if cstep < 0:
            return False, "evidence_event_bad_created_step", {"event_id": str(ev.get("event_id") or "")}
        if cstep < prev_step:
            return False, "evidence_events_not_monotonic", {"event_id": str(ev.get("event_id") or "")}
        prev_step = int(cstep)
        it = ev.get("evidence_item")
        if isinstance(it, dict):
            evid = str(it.get("evidence_id") or "")
            if evid:
                evidence_item_by_id[evid] = dict(it)
        stid = str(ev.get("source_turn_id") or "")
        if stid:
            evidence_events_by_turn.setdefault(stid, []).append(dict(ev))

    # Build belief items for RETRACT validation.
    belief_items_by_id: Dict[str, Dict[str, Any]] = {}
    belief_events_by_id: Dict[str, Dict[str, Any]] = {}
    for bev in belief_events:
        if not isinstance(bev, dict):
            continue
        eid = str(bev.get("event_id") or "")
        if eid:
            belief_events_by_id[eid] = dict(bev)
        if str(bev.get("event_kind") or "") != "ADD":
            continue
        bi = bev.get("belief_item")
        if isinstance(bi, dict):
            bid = str(bi.get("belief_id") or "")
            if bid:
                belief_items_by_id[bid] = dict(bi)

    # Belief events referencing evidence must be consistent.
    for bev in belief_events:
        if not isinstance(bev, dict):
            continue
        ceid = str(bev.get("cause_evidence_id") or "")
        if not ceid:
            continue
        eitem = evidence_item_by_id.get(ceid)
        if not isinstance(eitem, dict):
            return False, "evidence_missing", {"cause_evidence_id": ceid}
        want_k = str(eitem.get("evidence_key") or "").strip()
        want_v = str(eitem.get("evidence_value") or "").strip()
        got_k = str(bev.get("cause_evidence_key") or "").strip()
        got_v = str(bev.get("cause_evidence_value") or "").strip()
        if got_k != want_k:
            return False, "belief_evidence_key_mismatch", {"cause_evidence_id": ceid, "want": want_k, "got": got_k}
        if got_v != want_v:
            return False, "belief_evidence_value_mismatch", {"cause_evidence_id": ceid, "want": want_v, "got": got_v}
        ek = str(bev.get("event_kind") or "")
        if ek == "ADD":
            bi = bev.get("belief_item")
            if not isinstance(bi, dict):
                return False, "belief_add_missing_item", {"cause_evidence_id": ceid}
            if str(bi.get("belief_key") or "").strip() != want_k:
                return False, "belief_evidence_key_mismatch", {"cause_evidence_id": ceid, "want": want_k, "got": str(bi.get("belief_key") or "")}
            if str(bi.get("belief_value") or "").strip() != want_v:
                return False, "belief_evidence_value_mismatch", {"cause_evidence_id": ceid, "want": want_v, "got": str(bi.get("belief_value") or "")}
        elif ek == "RETRACT":
            tid = str(bev.get("target_belief_id") or "")
            if not tid:
                return False, "belief_retract_missing_target", {"cause_evidence_id": ceid}
            bi2 = belief_items_by_id.get(tid)
            if not isinstance(bi2, dict):
                return False, "belief_retract_unknown_target", {"cause_evidence_id": ceid, "target_belief_id": tid}
            if str(bi2.get("belief_key") or "").strip() != want_k:
                return False, "belief_evidence_key_mismatch", {"cause_evidence_id": ceid, "want": want_k, "got": str(bi2.get("belief_key") or "")}

    # Build helpers for turn->parse and plan lookup.
    by_index: Dict[int, Dict[str, Any]] = {}
    max_idx = -1
    for t in turns:
        if not isinstance(t, dict):
            continue
        try:
            idx = int(t.get("turn_index", -1))
        except Exception:
            idx = -1
        if idx >= 0:
            by_index[idx] = dict(t)
            if idx > max_idx:
                max_idx = int(idx)

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

    # Cross-check rendered outputs for 'evidences' and 'why <key>'.
    for i in range(0, max_idx + 1, 2):
        ut = by_index.get(i)
        at = by_index.get(i + 1)
        if not isinstance(ut, dict) or not isinstance(at, dict):
            continue
        utid = str(ut.get("turn_id") or "")
        payload = parses_by_turn_id.get(utid)
        if not isinstance(payload, dict):
            continue
        intent_id = str(payload.get("intent_id") or "")
        plan = plans_by_user_turn_id.get(utid)
        if not isinstance(plan, dict):
            continue
        try:
            cutoff = int(plan.get("created_step", -1))
        except Exception:
            cutoff = -1
        if cutoff < 0:
            continue

        if intent_id == INTENT_EVIDENCE_LIST_V98:
            evs_at = [dict(ev) for ev in evidence_events if isinstance(ev, dict) and int(ev.get("created_step", -1) or -1) <= int(cutoff)]
            want = render_evidences_text_v98(evs_at)
            if str(at.get("text") or "") != str(want):
                return False, "evidences_text_mismatch", {"turn_id": utid}
        if intent_id == INTENT_WHY_V98:
            key = str(payload.get("key") or "").strip()
            want = render_why_belief_text_v98(key=key, belief_events=belief_events, evidence_events=evidence_events, step=int(cutoff))
            if str(at.get("text") or "") != str(want):
                return False, "why_text_mismatch", {"turn_id": utid, "key": key}

    # Action plans: canonical evidence lists + references exist.
    for p in action_plans:
        if not isinstance(p, dict):
            continue
        utid = str(p.get("user_turn_id") or "")
        ev_read = _canon_str_list(p.get("evidence_read_ids"))
        ev_write = _canon_str_list(p.get("evidence_write_ids"))
        if _canon_str_list(p.get("evidence_read_ids")) != ev_read or _canon_str_list(p.get("evidence_write_ids")) != ev_write:
            return False, "plan_evidence_lists_not_canonical", {"user_turn_id": utid}
        for evid in list(ev_read) + list(ev_write):
            if evid and evid not in evidence_item_by_id:
                return False, "evidence_missing", {"evidence_id": evid, "user_turn_id": utid}

        prov = p.get("provenance") if isinstance(p.get("provenance"), dict) else {}
        cause_ids = _canon_str_list(prov.get("cause_evidence_ids"))
        bel_write = _canon_str_list(p.get("belief_write_event_ids"))
        want_cause_set: List[str] = []
        for wid in bel_write:
            bev = belief_events_by_id.get(str(wid))
            if not isinstance(bev, dict):
                return False, "plan_belief_event_missing", {"user_turn_id": utid, "belief_event_id": str(wid)}
            ceid = str(bev.get("cause_evidence_id") or "")
            if ceid:
                want_cause_set.append(str(ceid))
        want_cause = sorted(set([c for c in want_cause_set if isinstance(c, str) and c]))
        if want_cause != list(cause_ids):
            return False, "plan_cause_evidence_ids_mismatch", {"user_turn_id": utid, "want": want_cause, "got": list(cause_ids)}
        for cid in cause_ids:
            if cid and cid not in evidence_item_by_id:
                return False, "evidence_missing", {"cause_evidence_id": cid, "user_turn_id": utid}

    d = dict(details0)
    d["evidence_events_total"] = int(len(evidence_events))
    return True, "ok", d
