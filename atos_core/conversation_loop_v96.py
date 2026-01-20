from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .conversation_actions_v90 import action_concepts_for_dsl_v90
from .conversation_objectives_v90 import COMM_OBJECTIVES_V90, comm_objective_ids_v90, make_comm_objective_eq_text_v90
from .conversation_v96 import (
    ActionPlanV96,
    BeliefEventV96,
    BeliefItemV96,
    ConversationStateV96,
    MemoryEventV96,
    MemoryItemV96,
    TurnV96,
    append_chained_jsonl_v96,
    compute_belief_chain_hash_v96,
    compute_learned_chain_hash_v96,
    compute_memory_chain_hash_v96,
    compute_parse_chain_hash_v96,
    compute_plan_chain_hash_v96,
    compute_state_chain_hash_v96,
    compute_transcript_hash_v96,
    normalize_text_v96,
    render_belief_added_ack_text_v96,
    render_belief_retracted_ack_text_v96,
    render_belief_revised_ack_text_v96,
    render_beliefs_text_v96,
    render_forget_ack_text_v96,
    render_note_ack_text_v96,
    render_recall_text_v96,
    render_explain_text_v96,
    text_sig_v96,
    verify_chained_jsonl_v96,
    verify_conversation_chain_v96,
)
from .engine_v80 import EngineV80
from .goal_supports_v89 import SupportClaimV89, fold_support_stats_v89, list_supporting_concepts_for_goal_v89, make_goal_support_evidence_event_v89
from .intent_grammar_v92 import (
    INTENT_ADD_V92,
    INTENT_COMPOUND_V92,
    INTENT_END_V92,
    INTENT_GET_V92,
    INTENT_SET_V92,
    INTENT_SUMMARY_V92,
    INTENT_UNKNOWN_V92,
    default_intent_rule_acts_v92,
    default_intent_rules_v92,
    grammar_hash_v92,
    intent_grammar_snapshot_v92,
    parse_intent_v92,
)
from .intent_grammar_v93 import (
    INTENT_TEACH_V93,
    IntentRuleV93,
    canonize_lhs_for_learned_rule_v93,
    is_teach_command_v93,
    make_learned_intent_rule_v93,
    parse_teach_command_v93,
)
from .intent_grammar_v94 import INTENT_EXPLAIN_V94, is_explain_command_v94, parse_explain_command_v94
from .intent_grammar_v96 import (
    INTENT_BELIEF_ADD_V96,
    INTENT_BELIEF_FORGET_V96,
    INTENT_BELIEF_LIST_V96,
    INTENT_BELIEF_REVISE_V96,
    INTENT_FORGET_V96,
    INTENT_NOTE_V96,
    INTENT_RECALL_V96,
    is_belief_add_command_v96,
    is_belief_revise_command_v96,
    is_beliefs_list_command_v96,
    is_forget_command_v96,
    is_note_command_v96,
    is_recall_command_v96,
    parse_belief_add_command_v96,
    parse_belief_revise_command_v96,
    parse_beliefs_list_command_v96,
    parse_forget_command_v96,
    parse_note_command_v96,
    parse_recall_command_v96,
)
from .objective_v88 import execute_objective_csv_v88
from .store import ActStore


def _fail(msg: str) -> None:
    raise ValueError(msg)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"path_exists:{path}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _round6(x: Any) -> float:
    try:
        return float(round(float(x), 6))
    except Exception:
        return 0.0


def _rank_action_candidates_v96(
    *,
    candidates: Sequence[Tuple[str, SupportClaimV89]],
    events: Sequence[Dict[str, Any]],
    goal_id: str,
) -> List[Tuple[str, SupportClaimV89, float, float]]:
    scored: List[Tuple[str, SupportClaimV89, float, float]] = []
    for act_id, claim in candidates:
        stats = fold_support_stats_v89(events=events, goal_id=str(goal_id), concept_key=str(act_id), claim=claim)
        scored.append((str(act_id), claim, float(stats.expected_success), float(stats.expected_cost)))
    scored.sort(key=lambda t: (-float(t[2]), float(t[3]), str(t[0])))
    return scored


def _is_int_literal(s: str) -> bool:
    ss = str(s or "")
    return bool(ss) and ss.isdigit()


def _parse_int_or_var(*, vars_map: Dict[str, Any], tok: str, last_answer: Any) -> Tuple[Optional[int], str]:
    t = str(tok or "")
    if _is_int_literal(t):
        return int(t), "ok"
    if t == "last_answer":
        try:
            return int(last_answer), "ok"
        except Exception:
            return None, "missing_last_answer"
    if t in vars_map:
        try:
            return int(vars_map.get(t)), "ok"
        except Exception:
            return None, "bad_var_type"
    return None, "missing_key"


def _summarize_bindings_v96(*, vars_map: Dict[str, Any], last_answer: Any) -> str:
    parts: List[str] = []
    for k in sorted(vars_map.keys(), key=str):
        parts.append(f"{k}={vars_map.get(k)}")
    if last_answer != "":
        parts.append(f"last_answer={last_answer}")
    inner = "; ".join(parts)
    return f"Resumo: {inner}".rstrip()


def _simulate_compound_execution_v96(
    *, parse: Dict[str, Any], vars_map: Dict[str, Any], last_answer: Any, beliefs_by_key: Dict[str, Dict[str, Any]]
) -> Tuple[bool, str, Dict[str, Any]]:
    segs = parse.get("segments") if isinstance(parse.get("segments"), list) else []
    cur_vars = dict(vars_map)
    cur_last = last_answer
    lines: List[str] = []
    stop_after = False

    for seg in segs:
        if not isinstance(seg, dict):
            return False, "", {"reason": "segment_not_dict"}
        sp = seg.get("segment_parse") if isinstance(seg.get("segment_parse"), dict) else {}
        intent_id = str(sp.get("intent_id") or "")
        slots = sp.get("slots") if isinstance(sp.get("slots"), dict) else {}

        if intent_id == INTENT_SET_V92:
            k = str(slots.get("k") or "")
            v = str(slots.get("v") or "")
            if not k or not v:
                return False, "", {"reason": "segment_missing_slot", "missing_slot": "k/v"}
            cur_vars[k] = int(v) if _is_int_literal(v) else v
            lines.append(f"OK: {k}={v}")
            continue

        if intent_id == INTENT_GET_V92:
            k = str(slots.get("k") or "")
            if not k:
                return False, "", {"reason": "segment_missing_key", "missing_key": k}
            if k in cur_vars:
                lines.append(f"{k}={cur_vars.get(k)}")
                continue
            if k in beliefs_by_key:
                bv = beliefs_by_key.get(k) if isinstance(beliefs_by_key.get(k), dict) else {}
                lines.append(f"{k}={bv.get('belief_value')}")
                continue
            return False, "", {"reason": "segment_missing_key", "missing_key": k}

        if intent_id == INTENT_ADD_V92:
            a = str(slots.get("a") or "")
            b = str(slots.get("b") or "")
            va, ra = _parse_int_or_var(vars_map=cur_vars, tok=a, last_answer=cur_last)
            if va is None:
                return False, "", {"reason": "segment_missing_key", "missing_key": a, "detail": ra}
            vb, rb = _parse_int_or_var(vars_map=cur_vars, tok=b, last_answer=cur_last)
            if vb is None:
                return False, "", {"reason": "segment_missing_key", "missing_key": b, "detail": rb}
            s = int(va) + int(vb)
            cur_last = int(s)
            lines.append(f"SUM={s}")
            continue

        if intent_id == INTENT_SUMMARY_V92:
            lines.append(_summarize_bindings_v96(vars_map=cur_vars, last_answer=cur_last))
            continue

        if intent_id == INTENT_END_V92:
            lines.append("Encerrado.")
            stop_after = True
            break

        return False, "", {"reason": "segment_intent_unsupported", "intent_id": intent_id}

    agg = "\n".join(lines)
    return True, str(agg), {"vars_map": dict(cur_vars), "last_answer": cur_last, "stop_after": bool(stop_after)}


def _choose_comm_objective_v96(
    *, parse: Dict[str, Any], vars_map: Dict[str, Any], last_answer: Any, beliefs_by_key: Dict[str, Dict[str, Any]]
) -> Tuple[str, Dict[str, Any]]:
    intent_id = str(parse.get("intent_id") or "")
    slots = parse.get("slots") if isinstance(parse.get("slots"), dict) else {}
    missing = parse.get("missing_slots") if isinstance(parse.get("missing_slots"), list) else []
    if str(intent_id) == INTENT_UNKNOWN_V92:
        return "COMM_CORRECT", {"reason": "unknown_intent"}
    if missing:
        missing_slot = str(missing[0])
        return "COMM_ASK_CLARIFY", {"missing_slot": missing_slot}
    if intent_id == INTENT_SUMMARY_V92:
        return "COMM_SUMMARIZE", {}
    if intent_id == INTENT_END_V92:
        return "COMM_END", {}
    if intent_id == INTENT_GET_V92:
        k = str(slots.get("k") or "")
        if not k:
            return "COMM_ASK_CLARIFY", {"missing_slot": "k"}
        if k in vars_map:
            return "COMM_RESPOND", {}
        if k in beliefs_by_key:
            return "COMM_RESPOND", {"belief_key": k}
        return "COMM_ASK_CLARIFY", {"missing_key": k}
    if intent_id == INTENT_SET_V92:
        return "COMM_RESPOND", {}
    if intent_id == INTENT_ADD_V92:
        a = str(slots.get("a") or "")
        b = str(slots.get("b") or "")
        va, _ra = _parse_int_or_var(vars_map=vars_map, tok=a, last_answer=last_answer)
        if va is None:
            return "COMM_ASK_CLARIFY", {"missing_key": a}
        vb, _rb = _parse_int_or_var(vars_map=vars_map, tok=b, last_answer=last_answer)
        if vb is None:
            return "COMM_ASK_CLARIFY", {"missing_key": b}
        return "COMM_RESPOND", {}
    return "COMM_CORRECT", {"reason": f"unsupported_intent:{intent_id}"}


def _build_expected_and_action_inputs_v96(
    *,
    objective_kind: str,
    parse: Dict[str, Any],
    vars_map: Dict[str, Any],
    last_answer: Any,
    beliefs_by_key: Dict[str, Dict[str, Any]],
    ctx: Dict[str, Any],
    user_text: str,
) -> Tuple[str, Dict[str, Any], str]:
    intent_id = str(parse.get("intent_id") or "")
    slots = parse.get("slots") if isinstance(parse.get("slots"), dict) else {}

    if intent_id == INTENT_TEACH_V93:
        if bool(parse.get("teach_ok", False)):
            text = str(ctx.get("teach_ack_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        if str(parse.get("reason") or "") == "ambiguous":
            amb = ctx.get("ambiguous")
            amb_list = amb if isinstance(amb, list) else []
            opts: List[str] = []
            for x in amb_list:
                if not isinstance(x, dict):
                    continue
                rid = str(x.get("rule_id") or "")
                iid = str(x.get("intent_id") or "")
                if rid and iid:
                    opts.append(f"{rid}:{iid}")
            opts = sorted(set(opts))
            text = "Confirme: " + "; ".join(opts) if opts else "Confirme."
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_EXPLAIN_V94:
        if bool(parse.get("parse_ok", False)) and str(parse.get("explained_plan_id") or ""):
            text = str(ctx.get("explain_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_NOTE_V96:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("note_ack_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_RECALL_V96:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("recall_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_FORGET_V96:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("forget_ack_text") or ""):
            text = str(ctx.get("forget_ack_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id in {INTENT_BELIEF_ADD_V96, INTENT_BELIEF_REVISE_V96, INTENT_BELIEF_LIST_V96, INTENT_BELIEF_FORGET_V96}:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("belief_text") or ctx.get("belief_ack_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if bool(ctx.get("compound", False)) and objective_kind == "COMM_RESPOND":
        text = str(ctx.get("compound_text") or "")
        return text, {"text": text}, "concept_v90_emit_text_v0"

    if objective_kind == "COMM_END":
        return "Encerrado.", {}, "concept_v90_end_conversation_v0"
    if objective_kind == "COMM_ADMIT_UNKNOWN":
        return "Não sei.", {}, "concept_v90_admit_unknown_v0"
    if objective_kind == "COMM_CORRECT":
        msg = normalize_text_v96(str(user_text))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"
    if objective_kind == "COMM_CONFIRM":
        amb = ctx.get("ambiguous")
        amb_list = amb if isinstance(amb, list) else []
        opts2: List[str] = []
        for x in amb_list:
            if not isinstance(x, dict):
                continue
            rid = str(x.get("rule_id") or "")
            iid = str(x.get("intent_id") or "")
            if rid and iid:
                opts2.append(f"{rid}:{iid}")
        opts2 = sorted(set(opts2))
        text = "Confirme: " + "; ".join(opts2) if opts2 else "Confirme."
        return text, {"text": text}, "concept_v90_emit_text_v0"
    if objective_kind == "COMM_SUMMARIZE":
        summ = _summarize_bindings_v96(vars_map=vars_map, last_answer=last_answer)
        return summ, {"text": summ}, "concept_v90_emit_text_v0"
    if objective_kind == "COMM_ASK_CLARIFY":
        missing_key = str(ctx.get("missing_key") or "")
        missing_slot = str(ctx.get("missing_slot") or "")
        if missing_key:
            return f"Qual é o valor de {missing_key}?", {"k": missing_key}, "concept_v90_ask_clarify_v0"
        if missing_slot == "v":
            k = str(slots.get("k") or "")
            if k:
                return f"Qual é o valor de {k}?", {"k": k}, "concept_v90_ask_clarify_v0"
        q = f"Faltando: {missing_slot}" if missing_slot else "Faltando: dados"
        return q, {"text": q}, "concept_v90_emit_text_v0"

    if intent_id == INTENT_SET_V92:
        k = str(slots.get("k") or "")
        v = str(slots.get("v") or "")
        return f"OK: {k}={v}", {"k": k, "v": v}, "concept_v90_confirm_set_v0"
    if intent_id == INTENT_GET_V92:
        k = str(slots.get("k") or "")
        if k in vars_map:
            text = f"{k}={vars_map.get(k)}"
            return text, {"text": text}, "concept_v90_emit_text_v0"
        if k in beliefs_by_key:
            bv = beliefs_by_key.get(k) if isinstance(beliefs_by_key.get(k), dict) else {}
            text = f"{k}={bv.get('belief_value')}"
            return text, {"text": text}, "concept_v90_emit_text_v0"
        return "Não sei.", {}, "concept_v90_admit_unknown_v0"
    if intent_id == INTENT_ADD_V92:
        a = str(slots.get("a") or "")
        b = str(slots.get("b") or "")
        va, _ = _parse_int_or_var(vars_map=vars_map, tok=a, last_answer=last_answer)
        vb, _ = _parse_int_or_var(vars_map=vars_map, tok=b, last_answer=last_answer)
        if va is None or vb is None:
            return "Não sei.", {}, "concept_v90_admit_unknown_v0"
        s = int(int(va) + int(vb))
        text = f"SUM={s}"
        return text, {"sum": str(s)}, "concept_v90_emit_sum_v0"

    return "Não sei.", {}, "concept_v90_admit_unknown_v0"


def _parse_user_text_compound_v96(*, user_text: str, rules: Sequence[Any]) -> Dict[str, Any]:
    raw = str(user_text or "")
    if (";" in raw) or ("\n" in raw):
        parts: List[str] = []
        for seg in raw.replace("\n", ";").split(";"):
            seg2 = seg.strip()
            if seg2:
                parts.append(seg2)
        if len(parts) <= 1:
            return parse_intent_v92(user_text=str(user_text), rules=list(rules))
        segments: List[Dict[str, Any]] = []
        seg_fail: Optional[Dict[str, Any]] = None
        for i, seg_text in enumerate(parts):
            sp = parse_intent_v92(user_text=str(seg_text), rules=list(rules))
            seg_entry = {"segment_index": int(i), "segment_text": str(seg_text), "segment_parse": dict(sp)}
            segments.append(seg_entry)
            if seg_fail is None:
                if not bool(sp.get("parse_ok", False)) or (isinstance(sp.get("missing_slots"), list) and sp.get("missing_slots")):
                    seg_fail = dict(sp)
        sem = {
            "schema_version": 96,
            "intent_id": INTENT_COMPOUND_V92,
            "compound": True,
            "policy": "all_or_nothing",
            "segments": list(segments),
            "matched_rule_id": "COMPOUND",
        }
        parse_ok = seg_fail is None
        reason = "ok"
        if seg_fail is not None:
            reason = str(seg_fail.get("reason") or "segment_fail")
        sem["parse_ok"] = bool(parse_ok)
        sem["reason"] = str(reason)
        sig = _stable_hash_obj(sem)
        return dict(sem, parse_sig=str(sig))
    return parse_intent_v92(user_text=str(user_text), rules=list(rules))


def run_conversation_v96(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
) -> Dict[str, Any]:
    ensure_absent(str(out_dir))
    os.makedirs(str(out_dir), exist_ok=False)

    store_path = os.path.join(str(out_dir), "store.jsonl")
    grammar_snapshot_path = os.path.join(str(out_dir), "intent_grammar_snapshot.json")
    turns_path = os.path.join(str(out_dir), "conversation_turns.jsonl")
    parses_path = os.path.join(str(out_dir), "intent_parses.jsonl")
    learned_path = os.path.join(str(out_dir), "learned_intent_rules.jsonl")
    plans_path = os.path.join(str(out_dir), "action_plans.jsonl")
    memory_path = os.path.join(str(out_dir), "memory_events.jsonl")
    belief_path = os.path.join(str(out_dir), "belief_events.jsonl")
    states_path = os.path.join(str(out_dir), "conversation_states.jsonl")
    trials_path = os.path.join(str(out_dir), "dialogue_trials.jsonl")
    evals_path = os.path.join(str(out_dir), "objective_evals.jsonl")
    transcript_path = os.path.join(str(out_dir), "transcript.jsonl")
    verify_path = os.path.join(str(out_dir), "verify_chain_v96.json")
    manifest_path = os.path.join(str(out_dir), "freeze_manifest_v96.json")
    summary_path = os.path.join(str(out_dir), "summary.json")

    store = ActStore()

    obj_ids = comm_objective_ids_v90()
    for okind, oid in sorted(obj_ids.items(), key=lambda kv: str(kv[0])):
        store.add(make_comm_objective_eq_text_v90(objective_id=str(oid), objective_kind=str(okind), created_step=0))

    goal_ids = {k: str(k) for k in COMM_OBJECTIVES_V90}
    for act in action_concepts_for_dsl_v90(goal_ids=goal_ids):
        store.add(act)

    rules = default_intent_rules_v92()
    for act in default_intent_rule_acts_v92(created_step=0):
        store.add(act)

    if os.path.exists(store_path):
        _fail(f"store_path_exists:{store_path}")
    store.save_jsonl(store_path)
    store_hash = store.content_hash()

    if os.path.exists(grammar_snapshot_path):
        _fail(f"grammar_snapshot_exists:{grammar_snapshot_path}")
    grammar_snapshot = intent_grammar_snapshot_v92(rules)
    tmpg = grammar_snapshot_path + ".tmp"
    with open(tmpg, "w", encoding="utf-8") as f:
        f.write(json.dumps(grammar_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpg, grammar_snapshot_path)

    # Ensure ledgers exist even if empty (write-once; WORM).
    with open(memory_path, "x", encoding="utf-8") as _f:
        pass
    with open(belief_path, "x", encoding="utf-8") as _f:
        pass

    engine = EngineV80(store, seed=int(seed))

    conv_body = {"turns": list(user_turn_texts), "grammar_hash": str(grammar_hash_v92(rules))}
    conversation_id = f"conv_v96_{sha256_hex(canonical_json_dumps(conv_body).encode('utf-8'))}"

    vars_map: Dict[str, Any] = {}
    last_answer: Any = ""

    turns: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []
    transcript: List[Dict[str, Any]] = []
    parse_events: List[Dict[str, Any]] = []
    trials: List[Dict[str, Any]] = []
    learned_rule_events: List[Dict[str, Any]] = []
    action_plans: List[Dict[str, Any]] = []
    memory_events: List[Dict[str, Any]] = []
    belief_events: List[Dict[str, Any]] = []

    prev_turns_hash: Optional[str] = None
    prev_parses_hash: Optional[str] = None
    prev_learned_hash: Optional[str] = None
    prev_plans_hash: Optional[str] = None
    prev_memory_hash: Optional[str] = None
    prev_belief_hash: Optional[str] = None
    prev_states_hash: Optional[str] = None
    prev_trials_hash: Optional[str] = None
    prev_evals_hash: Optional[str] = None
    prev_transcript_hash: Optional[str] = None

    support_events: List[Dict[str, Any]] = []
    learned_rules_active: Dict[str, IntentRuleV93] = {}

    memory_items_by_id: Dict[str, Dict[str, Any]] = {}
    memory_active_ids: Dict[str, bool] = {}

    belief_items_by_id: Dict[str, Dict[str, Any]] = {}
    belief_active_by_key: Dict[str, str] = {}

    state_index = 0
    turn_index = 0
    step = 0

    def _objective_act_id(okind: str) -> str:
        return str(obj_ids.get(okind) or "")

    def _active_memory_items_sorted() -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for mid in sorted(memory_active_ids.keys(), key=str):
            if not bool(memory_active_ids.get(mid, False)):
                continue
            mi = memory_items_by_id.get(mid)
            if isinstance(mi, dict):
                items.append(dict(mi))
        items.sort(key=lambda it: (int(it.get("created_step", 0)), str(it.get("memory_id") or "")))
        return items

    def _last_active_memory_id() -> str:
        items = _active_memory_items_sorted()
        if not items:
            return ""
        items.sort(key=lambda it: (int(it.get("created_step", 0)), str(it.get("memory_id") or "")))
        return str(items[-1].get("memory_id") or "")

    def _active_beliefs_by_key() -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for k in sorted(belief_active_by_key.keys(), key=str):
            bid = str(belief_active_by_key.get(k) or "")
            bi = belief_items_by_id.get(bid)
            if isinstance(bi, dict) and bid:
                out[str(k)] = dict(bi)
        return dict(out)

    def _last_explainable_plan() -> Optional[Dict[str, Any]]:
        for p in reversed(action_plans):
            if not isinstance(p, dict):
                continue
            if str(p.get("intent_id") or "") == INTENT_EXPLAIN_V94:
                continue
            return dict(p)
        return None

    def _execute_action(act_id: str, *, goal_kind: str, inputs: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any], float]:
        concept_act = store.get_concept_act(str(act_id))
        if concept_act is None:
            return False, "", {"ok": False, "reason": "action_not_found"}, 0.0
        iface = concept_act.evidence.get("interface") if isinstance(concept_act.evidence, dict) else {}
        in_schema = iface.get("input_schema") if isinstance(iface, dict) else {}
        in_schema = in_schema if isinstance(in_schema, dict) else {}
        inps: Dict[str, Any] = {}
        for k in sorted(in_schema.keys(), key=str):
            ks = str(k)
            val = inputs.get(ks)
            if isinstance(in_schema.get(k), str) and str(in_schema.get(k)) == "str":
                inps[ks] = "" if val is None else str(val)
            else:
                inps[ks] = val
        exec_res = engine.execute_concept_csv(
            concept_act_id=str(act_id),
            inputs=dict(inps),
            goal_kind=str(goal_kind),
            expected=None,
            step=int(step),
            max_depth=6,
            max_events=256,
            validate_output=False,
        )
        meta = exec_res.get("meta") if isinstance(exec_res.get("meta"), dict) else {}
        if not bool(meta.get("ok", False)):
            return False, "", dict(meta), 0.0
        out_text = str(meta.get("output_text") or exec_res.get("output") or "")
        trace = exec_res.get("trace") if isinstance(exec_res.get("trace"), dict) else {}
        calls = trace.get("concept_calls") if isinstance(trace.get("concept_calls"), list) else []
        cost_used = 0.0
        for c in calls:
            if not isinstance(c, dict):
                continue
            cost_used += float(c.get("cost") or 0.0)
        return True, str(out_text), dict(meta), float(cost_used)

    for user_text in list(user_turn_texts):
        active_learned_rules = [learned_rules_active[k] for k in sorted(learned_rules_active.keys(), key=str)]
        rules_for_parse: List[Any] = list(rules) + list(active_learned_rules)

        parse: Dict[str, Any] = {}
        ctx: Dict[str, Any] = {}

        # Intercepts: TEACH -> EXPLAIN -> BELIEFS -> BELIEF/REVISE -> FORGET (memory/belief) -> NOTE/RECALL -> parser V92/compound.
        if is_teach_command_v93(str(user_text)):
            te = parse_teach_command_v93(str(user_text))
            teach_ok = False
            teach_reason = str(te.get("reason") or "not_recognized")
            lhs_raw = str(te.get("lhs_raw") or "")
            rhs_raw = str(te.get("rhs_raw") or "")
            rhs_parse_sig = ""
            rhs_intent_id = ""
            rhs_rule_id = ""
            rhs_reason = ""
            lhs_info: Dict[str, Any] = {}
            ambiguous_rule_ids: List[str] = []
            ambiguous_intents: List[Dict[str, Any]] = []
            pending_learned_rule: Optional[IntentRuleV93] = None

            if bool(te.get("recognized", False)) and bool(te.get("ok", False)):
                lhs_info = canonize_lhs_for_learned_rule_v93(lhs_raw)
                stripped = lhs_info.get("lhs_tokens_canon_stripped")
                stripped_list = [str(x) for x in stripped if isinstance(stripped, list) and isinstance(x, str) and x]
                if not stripped_list:
                    teach_ok = False
                    teach_reason = "lhs_empty_after_normalization"
                else:
                    rhs_parse = parse_intent_v92(user_text=str(rhs_raw), rules=list(rules))
                    rhs_parse_sig = str(rhs_parse.get("parse_sig") or "")
                    rhs_intent_id = str(rhs_parse.get("intent_id") or "")
                    rhs_rule_id = str(rhs_parse.get("matched_rule_id") or "")
                    rhs_reason = str(rhs_parse.get("reason") or "")
                    if not bool(rhs_parse.get("parse_ok", False)):
                        teach_reason = f"rhs_parse_fail:{rhs_reason or 'no_match'}"
                    elif rhs_parse.get("missing_slots"):
                        teach_reason = "rhs_missing_slots"
                    elif rhs_intent_id not in {INTENT_SUMMARY_V92, INTENT_END_V92}:
                        teach_reason = "rhs_intent_disallowed"
                    else:
                        learned_rule = make_learned_intent_rule_v93(intent_id=str(rhs_intent_id), lhs_tokens_canon=stripped_list)
                        conflicts = []
                        for r in rules:
                            try:
                                rid = str(getattr(r, "rule_id", ""))
                                pat = getattr(r, "pattern", [])
                            except Exception:
                                rid = ""
                                pat = []
                            if not isinstance(pat, list):
                                continue
                            if [str(x) for x in pat] == stripped_list:
                                conflicts.append({"rule_id": str(rid), "intent_id": str(getattr(r, "intent_id", ""))})
                        conflicts = sorted(conflicts, key=lambda d: str(d.get("rule_id") or ""))
                        if conflicts:
                            teach_ok = False
                            teach_reason = "ambiguous"
                            ambiguous_rule_ids = [str(x.get("rule_id") or "") for x in conflicts]
                            ambiguous_intents = list(conflicts)
                        else:
                            teach_ok = True
                            teach_reason = "ok"
                            pending_learned_rule = learned_rule

            sem = {
                "schema_version": 96,
                "intent_id": INTENT_TEACH_V93,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(teach_ok),
                "reason": str(teach_reason),
                "lhs_raw": str(lhs_raw),
                "rhs_raw": str(rhs_raw),
                "rhs_parse_sig": str(rhs_parse_sig),
                "rhs_intent_id": str(rhs_intent_id),
                "rhs_rule_id": str(rhs_rule_id),
                "rhs_reason": str(rhs_reason),
                "teach_ok": bool(teach_ok),
                "learned_rule_id": str(pending_learned_rule.rule_id) if pending_learned_rule else "",
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if teach_ok and pending_learned_rule is not None:
                learned_row = {
                    "kind": "learned_intent_rule_v96",
                    "time": deterministic_iso(step=int(step)),
                    "step": int(step),
                    "teacher_turn_id": "",  # filled after user turn exists
                    "lhs_raw": str(lhs_raw),
                    "rhs_raw": str(rhs_raw),
                    "rhs_parse_sig": str(rhs_parse_sig),
                    "rhs_intent_id": str(rhs_intent_id),
                    "rhs_rule_id": str(rhs_rule_id),
                    "learned_rule": dict(pending_learned_rule.to_dict()),
                    "provenance": {
                        "lhs_raw": str(lhs_raw),
                        "rhs_raw": str(rhs_raw),
                    },
                }
                ctx["pending_learned_rule"] = pending_learned_rule
                ctx["pending_learned_row"] = learned_row
            if teach_reason == "ambiguous":
                ctx["ambiguous"] = list(ambiguous_intents)

        elif is_explain_command_v94(str(user_text)):
            ex = parse_explain_command_v94(str(user_text))
            ok = bool(ex.get("recognized", False)) and bool(ex.get("ok", False))
            reason = str(ex.get("reason") or "not_recognized")
            explained_plan_id = ""
            explain_text = ""
            if ok:
                lastp = _last_explainable_plan()
                if lastp is None:
                    ok = False
                    reason = "no_prior_plan"
                else:
                    explained_plan_id = str(lastp.get("plan_id") or "")
                    explain_text = render_explain_text_v96(lastp)
            sem = {
                "schema_version": 96,
                "intent_id": INTENT_EXPLAIN_V94,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "explained_plan_id": str(explained_plan_id),
                "prefix": str(ex.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            ctx["explain_text"] = str(explain_text)
            ctx["msg"] = str(reason or "")

        elif is_beliefs_list_command_v96(str(user_text)):
            be = parse_beliefs_list_command_v96(str(user_text))
            ok = bool(be.get("recognized", False)) and bool(be.get("ok", False))
            reason = str(be.get("reason") or "not_recognized")
            sem = {
                "schema_version": 96,
                "intent_id": INTENT_BELIEF_LIST_V96,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(be.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_belief_add_command_v96(str(user_text)):
            be = parse_belief_add_command_v96(str(user_text))
            ok = bool(be.get("recognized", False)) and bool(be.get("ok", False))
            reason = str(be.get("reason") or "not_recognized")
            key = str(be.get("belief_key") or "")
            val = str(be.get("belief_value") or "")
            if ok and key in belief_active_by_key:
                ok = False
                reason = "key_exists_use_revise"
            sem = {
                "schema_version": 96,
                "intent_id": INTENT_BELIEF_ADD_V96,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(be.get("prefix") or ""),
                "belief_key": str(key),
                "belief_value": str(val),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_belief_revise_command_v96(str(user_text)):
            be = parse_belief_revise_command_v96(str(user_text))
            ok = bool(be.get("recognized", False)) and bool(be.get("ok", False))
            reason = str(be.get("reason") or "not_recognized")
            key = str(be.get("belief_key") or "")
            val = str(be.get("belief_value") or "")
            if ok and key not in belief_active_by_key:
                ok = False
                reason = "missing_key_use_belief"
            sem = {
                "schema_version": 96,
                "intent_id": INTENT_BELIEF_REVISE_V96,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(be.get("prefix") or ""),
                "belief_key": str(key),
                "belief_value": str(val),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_forget_command_v96(str(user_text)):
            fe = parse_forget_command_v96(str(user_text))
            ok = bool(fe.get("recognized", False)) and bool(fe.get("ok", False))
            reason = str(fe.get("reason") or "not_recognized")
            target_kind = str(fe.get("target_kind") or "")
            belief_key = str(fe.get("belief_key") or "")
            if ok and target_kind == "belief":
                if belief_key not in belief_active_by_key:
                    ok = False
                    reason = "missing_key"
            sem = {
                "schema_version": 96,
                "intent_id": INTENT_BELIEF_FORGET_V96 if target_kind == "belief" else INTENT_FORGET_V96,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(fe.get("prefix") or ""),
                "target_kind": str(target_kind),
                "target": str(fe.get("target") or ""),
                "belief_key": str(belief_key),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_note_command_v96(str(user_text)):
            ne = parse_note_command_v96(str(user_text))
            ok = bool(ne.get("recognized", False)) and bool(ne.get("ok", False))
            reason = str(ne.get("reason") or "not_recognized")
            sem = {
                "schema_version": 96,
                "intent_id": INTENT_NOTE_V96,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(ne.get("prefix") or ""),
                "memory_text_raw": str(ne.get("memory_text_raw") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_recall_command_v96(str(user_text)):
            re = parse_recall_command_v96(str(user_text))
            ok = bool(re.get("recognized", False)) and bool(re.get("ok", False))
            reason = str(re.get("reason") or "not_recognized")
            sem = {
                "schema_version": 96,
                "intent_id": INTENT_RECALL_V96,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(re.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        else:
            parse = _parse_user_text_compound_v96(user_text=str(user_text), rules=list(rules_for_parse))

        # Create user turn.
        ut = TurnV96(
            conversation_id=str(conversation_id),
            turn_index=int(turn_index),
            role="user",
            text=str(user_text),
            created_step=int(step),
            offset_us=0,
            parse_sig=str(parse.get("parse_sig") or ""),
            intent_id=str(parse.get("intent_id") or ""),
            matched_rule_id=str(parse.get("matched_rule_id") or ""),
        ).to_dict()
        turn_index += 1
        step += 1
        turns.append(dict(ut))
        prev_turns_hash = append_chained_jsonl_v96(
            turns_path,
            {"time": deterministic_iso(step=int(ut["created_step"])), "step": int(ut["created_step"]), "event": "TURN", "payload": dict(ut)},
            prev_hash=prev_turns_hash,
        )
        transcript.append({"role": "user", "text": str(ut.get("text") or ""), "turn_id": str(ut.get("turn_id") or "")})
        prev_transcript_hash = append_chained_jsonl_v96(
            transcript_path,
            {"time": deterministic_iso(step=int(ut["created_step"])), "step": int(ut["created_step"]), "event": "UTTERANCE", "payload": dict(transcript[-1])},
            prev_hash=prev_transcript_hash,
        )

        # Log parse (WORM, hash-chained).
        parse_event = {
            "kind": "intent_parse_v96",
            "time": deterministic_iso(step=int(step)),
            "step": int(step),
            "turn_id": str(ut.get("turn_id") or ""),
            "turn_index": int(ut.get("turn_index") or 0),
            "payload": dict(parse),
        }
        prev_parses_hash = append_chained_jsonl_v96(parses_path, dict(parse_event), prev_hash=prev_parses_hash)
        parse_events.append({"turn_id": str(parse_event["turn_id"]), "turn_index": int(parse_event["turn_index"]), "payload": dict(parse)})

        # Apply learning (TEACH) after user turn exists, before executing actions.
        if str(parse.get("intent_id") or "") == "INTENT_TEACH" and bool(parse.get("teach_ok", False)):
            lr = ctx.get("pending_learned_rule")
            row = ctx.get("pending_learned_row")
            if isinstance(lr, IntentRuleV93) and isinstance(row, dict):
                learned_row = dict(row)
                learned_row["teacher_turn_id"] = str(ut.get("turn_id") or "")
                learned_row["provenance"] = dict(learned_row.get("provenance") or {})
                learned_row["provenance"]["teacher_turn_id"] = str(ut.get("turn_id") or "")
                prev_learned_hash = append_chained_jsonl_v96(learned_path, dict(learned_row), prev_hash=prev_learned_hash)
                learned_rule_events.append(dict(learned_row))
                learned_rules_active[str(lr.rule_id)] = lr

        # Memory/belief ops produce read/write refs for plan.
        memory_read_ids: List[str] = []
        memory_write_event_ids: List[str] = []
        belief_read_keys: List[str] = []
        belief_read_ids: List[str] = []
        belief_write_event_ids: List[str] = []

        objective_kind = ""
        ctx2: Dict[str, Any] = {}

        beliefs_active = _active_beliefs_by_key()

        if str(parse.get("intent_id") or "") == INTENT_NOTE_V96:
            if bool(parse.get("parse_ok", False)):
                txt = str(parse.get("memory_text_raw") or "")
                mi = MemoryItemV96(conversation_id=str(conversation_id), memory_text=str(txt), source_turn_id=str(ut.get("turn_id") or ""), created_step=int(step)).to_dict()
                mid = str(mi.get("memory_id") or "")
                ev = MemoryEventV96(
                    conversation_id=str(conversation_id),
                    event_kind="ADD",
                    created_step=int(step),
                    source_turn_id=str(ut.get("turn_id") or ""),
                    memory_item=dict(mi),
                ).to_dict()
                prev_memory_hash = append_chained_jsonl_v96(memory_path, dict(ev), prev_hash=prev_memory_hash)
                memory_events.append(dict(ev))
                if mid:
                    memory_items_by_id[mid] = dict(mi)
                    memory_active_ids[mid] = True
                memory_write_event_ids.append(str(ev.get("event_id") or ""))
                objective_kind = "COMM_RESPOND"
                ctx2 = {"note_ack_text": render_note_ack_text_v96(mid)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"note_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_RECALL_V96:
            if bool(parse.get("parse_ok", False)):
                items = _active_memory_items_sorted()
                memory_read_ids = [str(it.get("memory_id") or "") for it in items if isinstance(it, dict) and str(it.get("memory_id") or "")]
                objective_kind = "COMM_RESPOND"
                ctx2 = {"recall_text": render_recall_text_v96(items)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"recall_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_FORGET_V96:
            if bool(parse.get("parse_ok", False)):
                target_mid = _last_active_memory_id()
                if not target_mid:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "no_active_memory"}
                else:
                    ev = MemoryEventV96(
                        conversation_id=str(conversation_id),
                        event_kind="RETRACT",
                        created_step=int(step),
                        source_turn_id=str(ut.get("turn_id") or ""),
                        target_memory_id=str(target_mid),
                        retract_reason="user_forget_last",
                    ).to_dict()
                    prev_memory_hash = append_chained_jsonl_v96(memory_path, dict(ev), prev_hash=prev_memory_hash)
                    memory_events.append(dict(ev))
                    memory_active_ids.pop(str(target_mid), None)
                    memory_read_ids = [str(target_mid)]
                    memory_write_event_ids.append(str(ev.get("event_id") or ""))
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"forget_ack_text": render_forget_ack_text_v96(target_mid)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"forget_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_BELIEF_ADD_V96:
            if bool(parse.get("parse_ok", False)):
                key = str(parse.get("belief_key") or "").strip()
                val = str(parse.get("belief_value") or "").strip()
                bi = BeliefItemV96(conversation_id=str(conversation_id), belief_key=str(key), belief_value=str(val), source_turn_id=str(ut.get("turn_id") or ""), created_step=int(step)).to_dict()
                bid = str(bi.get("belief_id") or "")
                ev = BeliefEventV96(
                    conversation_id=str(conversation_id),
                    event_kind="ADD",
                    created_step=int(step),
                    source_turn_id=str(ut.get("turn_id") or ""),
                    belief_item=dict(bi),
                ).to_dict()
                prev_belief_hash = append_chained_jsonl_v96(belief_path, dict(ev), prev_hash=prev_belief_hash)
                belief_events.append(dict(ev))
                if bid and key:
                    belief_items_by_id[bid] = dict(bi)
                    belief_active_by_key[key] = bid
                belief_write_event_ids.append(str(ev.get("event_id") or ""))
                objective_kind = "COMM_RESPOND"
                ctx2 = {"belief_ack_text": render_belief_added_ack_text_v96(belief_id=bid, key=key)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"belief_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_BELIEF_REVISE_V96:
            if bool(parse.get("parse_ok", False)):
                key = str(parse.get("belief_key") or "").strip()
                val = str(parse.get("belief_value") or "").strip()
                old_id = str(belief_active_by_key.get(key) or "")
                if not old_id:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "missing_key_use_belief"}
                else:
                    belief_read_keys = [str(key)]
                    belief_read_ids = [str(old_id)]
                    evr = BeliefEventV96(
                        conversation_id=str(conversation_id),
                        event_kind="RETRACT",
                        created_step=int(step),
                        source_turn_id=str(ut.get("turn_id") or ""),
                        target_belief_id=str(old_id),
                        retract_reason="user_revise",
                    ).to_dict()
                    prev_belief_hash = append_chained_jsonl_v96(belief_path, dict(evr), prev_hash=prev_belief_hash)
                    belief_events.append(dict(evr))
                    bi = BeliefItemV96(conversation_id=str(conversation_id), belief_key=str(key), belief_value=str(val), source_turn_id=str(ut.get("turn_id") or ""), created_step=int(step)).to_dict()
                    bid = str(bi.get("belief_id") or "")
                    eva = BeliefEventV96(
                        conversation_id=str(conversation_id),
                        event_kind="ADD",
                        created_step=int(step),
                        source_turn_id=str(ut.get("turn_id") or ""),
                        belief_item=dict(bi),
                    ).to_dict()
                    prev_belief_hash = append_chained_jsonl_v96(belief_path, dict(eva), prev_hash=prev_belief_hash)
                    belief_events.append(dict(eva))
                    if bid and key:
                        belief_items_by_id[bid] = dict(bi)
                        belief_active_by_key[key] = bid
                    belief_write_event_ids.extend([str(evr.get("event_id") or ""), str(eva.get("event_id") or "")])
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"belief_ack_text": render_belief_revised_ack_text_v96(key=key, old_id=old_id, new_id=bid)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"revise_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_BELIEF_LIST_V96:
            if bool(parse.get("parse_ok", False)):
                beliefs_active = _active_beliefs_by_key()
                belief_read_keys = sorted(beliefs_active.keys(), key=str)
                belief_read_ids = [str(beliefs_active[k].get("belief_id") or "") for k in belief_read_keys if isinstance(beliefs_active.get(k), dict)]
                objective_kind = "COMM_RESPOND"
                ctx2 = {"belief_text": render_beliefs_text_v96(beliefs_active)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"beliefs_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_BELIEF_FORGET_V96:
            if bool(parse.get("parse_ok", False)):
                key = str(parse.get("belief_key") or "").strip()
                old_id = str(belief_active_by_key.get(key) or "")
                if not old_id:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "missing_key"}
                else:
                    belief_read_keys = [str(key)]
                    belief_read_ids = [str(old_id)]
                    ev = BeliefEventV96(
                        conversation_id=str(conversation_id),
                        event_kind="RETRACT",
                        created_step=int(step),
                        source_turn_id=str(ut.get("turn_id") or ""),
                        target_belief_id=str(old_id),
                        retract_reason="user_forget_belief",
                    ).to_dict()
                    prev_belief_hash = append_chained_jsonl_v96(belief_path, dict(ev), prev_hash=prev_belief_hash)
                    belief_events.append(dict(ev))
                    belief_active_by_key.pop(str(key), None)
                    belief_write_event_ids.append(str(ev.get("event_id") or ""))
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"belief_ack_text": render_belief_retracted_ack_text_v96(belief_id=old_id, key=key)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"forget_belief_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_TEACH_V93:
            teach_ok = bool(parse.get("teach_ok", False))
            teach_reason = str(parse.get("reason") or "")
            if teach_ok:
                objective_kind = "COMM_RESPOND"
                lrid = str(parse.get("learned_rule_id") or "")
                rhs_intent_id = str(parse.get("rhs_intent_id") or "")
                lhs_raw = str(parse.get("lhs_raw") or "")
                ack = f'Aprendido: "{lhs_raw}" → {rhs_intent_id} (rule_id={lrid})'
                ctx2 = {"teach_ack_text": str(ack)}
            elif teach_reason == "ambiguous":
                objective_kind = "COMM_CONFIRM"
                ctx2 = {"ambiguous": list(ctx.get("ambiguous") if isinstance(ctx.get("ambiguous"), list) else [])}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"teach_reject:{teach_reason}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_V94:
            if bool(parse.get("parse_ok", False)) and str(parse.get("explained_plan_id") or ""):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"explain_text": str(ctx.get("explain_text") or "")}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_reject:{r or 'parse_fail'}"}

        else:
            objective_kind, ctx2 = _choose_comm_objective_v96(
                parse=dict(parse), vars_map=dict(vars_map), last_answer=last_answer, beliefs_by_key=dict(beliefs_active)
            )

        # Compound execution (same as V92/V95).
        compound_stop_after = False
        if bool(parse.get("compound", False)) and bool(parse.get("parse_ok", False)):
            ok_comp, agg_text, info = _simulate_compound_execution_v96(
                parse=dict(parse), vars_map=dict(vars_map), last_answer=last_answer, beliefs_by_key=dict(beliefs_active)
            )
            if ok_comp:
                vars_map = dict(info.get("vars_map") or {})
                last_answer = info.get("last_answer")
                compound_stop_after = bool(info.get("stop_after", False))
                objective_kind = "COMM_RESPOND"
                ctx2 = {"compound": True, "compound_text": str(agg_text)}
            else:
                r = str(info.get("reason") or "")
                if r == "segment_missing_key":
                    objective_kind = "COMM_ASK_CLARIFY"
                    ctx2 = {"missing_key": str(info.get("missing_key") or "")}
                else:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"reason": f"compound_exec_fail:{r}"}

        # Apply DSL state mutation for non-compound, parse_ok and complete.
        slots = parse.get("slots") if isinstance(parse.get("slots"), dict) else {}
        if (not bool(parse.get("compound", False))) and bool(parse.get("parse_ok", False)) and not (parse.get("missing_slots") or []):
            if str(parse.get("intent_id") or "") == INTENT_SET_V92:
                k = str(slots.get("k") or "")
                v = str(slots.get("v") or "")
                if k and v:
                    vars_map[k] = int(v) if _is_int_literal(v) else v
            elif str(parse.get("intent_id") or "") == INTENT_ADD_V92:
                a = str(slots.get("a") or "")
                b = str(slots.get("b") or "")
                va, _ra = _parse_int_or_var(vars_map=vars_map, tok=a, last_answer=last_answer)
                vb, _rb = _parse_int_or_var(vars_map=vars_map, tok=b, last_answer=last_answer)
                if va is not None and vb is not None:
                    last_answer = int(int(va) + int(vb))

        # GET reads belief if var missing.
        if str(parse.get("intent_id") or "") == INTENT_GET_V92:
            k = str(slots.get("k") or "")
            if k and (k not in vars_map) and (k in beliefs_active):
                it = beliefs_active.get(k) if isinstance(beliefs_active.get(k), dict) else {}
                belief_read_keys = [str(k)]
                belief_read_ids = [str(it.get("belief_id") or "")]

        expected_text, action_inputs, hint_action_id = _build_expected_and_action_inputs_v96(
            objective_kind=str(objective_kind),
            parse=dict(parse),
            vars_map=dict(vars_map),
            last_answer=last_answer,
            beliefs_by_key=dict(beliefs_active),
            ctx=dict(ctx2),
            user_text=str(user_text),
        )

        chosen_action_id = ""
        chosen_objective_id = ""
        chosen_eval_id = ""
        chosen_ok = False
        chosen_cost = 0.0
        assistant_text = ""
        plan_ranked_candidates: List[Dict[str, Any]] = []
        plan_attempted_actions: List[Dict[str, Any]] = []
        plan_objective_kind = ""
        plan_objective_id = ""

        if str(objective_kind) == "COMM_END":
            assistant_text = "Encerrado."
            chosen_action_id = "concept_v90_end_conversation_v0"
            chosen_objective_id = _objective_act_id("COMM_END")
            chosen_eval_id = _stable_hash_obj({"turn_id": str(ut.get("turn_id") or ""), "kind": "end"})
            chosen_ok = True
            chosen_cost = 0.0
            plan_ranked_candidates = [{"act_id": chosen_action_id, "expected_success": 1.0, "expected_cost": 0.0}]
            plan_attempted_actions = [{"act_id": chosen_action_id, "eval_id": chosen_eval_id, "ok": True}]
            plan_objective_kind = "COMM_END"
            plan_objective_id = chosen_objective_id
        else:
            goal_id = str(objective_kind)
            goal_kind = str(objective_kind)
            candidates0 = list_supporting_concepts_for_goal_v89(store=store, goal_id=str(goal_id))
            ranked = _rank_action_candidates_v96(candidates=list(candidates0), events=list(support_events), goal_id=str(goal_id))

            hint = str(hint_action_id or "")
            ranked_ids = [str(act_id) for act_id, _claim, _es, _ec in ranked]
            if hint and hint in set(ranked_ids):
                ranked.sort(key=lambda t: (0 if str(t[0]) == hint else 1, -float(t[2]), float(t[3]), str(t[0])))

            plan_ranked_candidates = [{"act_id": str(a), "expected_success": _round6(es), "expected_cost": _round6(ec)} for a, _c, es, ec in ranked]
            plan_objective_kind = str(goal_kind)
            plan_objective_id = _objective_act_id(str(goal_kind))

            # Attempt in ranked order until objective passes.
            def _try_actions() -> bool:
                nonlocal chosen_action_id, chosen_objective_id, chosen_eval_id, chosen_ok, chosen_cost, assistant_text
                for act_id, _claim, _es, _ec in ranked:
                    ok_exec, out_text, _meta, cost_used = _execute_action(str(act_id), goal_kind=str(goal_kind), inputs=dict(action_inputs))
                    eval_id = _stable_hash_obj(
                        {
                            "conversation_id": str(conversation_id),
                            "turn_id": str(ut.get("turn_id") or ""),
                            "step": int(step),
                            "objective_kind": str(goal_kind),
                            "objective_id": str(plan_objective_id),
                            "act_id": str(act_id),
                            "expected_text_sig": text_sig_v96(str(expected_text)),
                            "output_text_sig": text_sig_v96(str(out_text)),
                        }
                    )
                    objective_inputs = {
                        "__output": str(out_text),
                        "expected": str(expected_text),
                        "__goal": str(goal_kind),
                        "__step": int(step),
                    }
                    verdict_ok = False
                    verdict: Dict[str, Any] = {}
                    if not ok_exec:
                        verdict_ok = False
                        verdict = {
                            "ok": False,
                            "score": 0,
                            "reason": f"action_exec_failed:{str(_meta.get('reason') or '')}",
                            "details": {"meta": dict(_meta) if isinstance(_meta, dict) else {}},
                        }
                    else:
                        verdict_obj = execute_objective_csv_v88(
                            store=store,
                            seed=int(seed),
                            objective_act_id=str(plan_objective_id),
                            inputs=dict(objective_inputs),
                            step=int(step),
                            goal_kind=str(goal_kind),
                        )
                        verdict = verdict_obj.to_dict()
                        verdict_ok = bool(verdict_obj.ok)
                    expected_sig = text_sig_v96(str(expected_text))
                    output_sig = text_sig_v96(str(out_text))
                    eval_row = {
                        "kind": "objective_eval_v96",
                        "time": deterministic_iso(step=int(step)),
                        "step": int(step),
                        "eval_id": str(eval_id),
                        "conversation_id": str(conversation_id),
                        "turn_id": str(ut.get("turn_id") or ""),
                        "objective_kind": str(goal_kind),
                        "objective_id": str(plan_objective_id),
                        "action_concept_id": str(act_id),
                        "expected_text": str(expected_text),
                        "output_text": str(out_text),
                        "expected_text_sig": str(expected_sig),
                        "output_text_sig": str(output_sig),
                        "verdict": dict(verdict),
                    }
                    nonlocal prev_evals_hash
                    prev_evals_hash = append_chained_jsonl_v96(evals_path, dict(eval_row), prev_hash=prev_evals_hash)

                    trial_id = _stable_hash_obj(
                        {
                            "conversation_id": str(conversation_id),
                            "turn_id": str(ut.get("turn_id") or ""),
                            "step": int(step),
                            "objective_kind": str(goal_kind),
                            "objective_id": str(plan_objective_id),
                            "act_id": str(act_id),
                            "eval_id": str(eval_id),
                        }
                    )
                    trial_row = {
                        "kind": "dialogue_trial_v96",
                        "time": deterministic_iso(step=int(step)),
                        "step": int(step),
                        "trial_id": str(trial_id),
                        "conversation_id": str(conversation_id),
                        "turn_id": str(ut.get("turn_id") or ""),
                        "user_turn_id": str(ut.get("turn_id") or ""),
                        "objective_kind": str(goal_kind),
                        "objective_id": str(plan_objective_id),
                        "action_concept_id": str(act_id),
                        "eval_id": str(eval_id),
                        "expected_text": str(expected_text),
                        "expected_text_sig": str(expected_sig),
                        "assistant_text": str(out_text),
                        "assistant_text_sig": str(output_sig),
                        "ok": bool(verdict_ok),
                        "cost_used": float(cost_used),
                    }
                    nonlocal prev_trials_hash
                    prev_trials_hash = append_chained_jsonl_v96(trials_path, dict(trial_row), prev_hash=prev_trials_hash)

                    support_ev = make_goal_support_evidence_event_v89(
                        step=int(step),
                        goal_id=str(goal_kind),
                        concept_key=str(act_id),
                        attempt_id=str(trial_id),
                        ok=bool(verdict_ok),
                        cost_used=float(cost_used),
                        note=str(verdict.get("reason") or ""),
                    )
                    support_events.append(dict(support_ev))

                    plan_attempted_actions.append({"act_id": str(act_id), "eval_id": str(eval_id), "ok": bool(verdict_ok)})

                    if verdict_ok and ok_exec:
                        assistant_text = str(out_text)
                        chosen_action_id = str(act_id)
                        chosen_objective_id = str(plan_objective_id)
                        chosen_eval_id = str(eval_id)
                        chosen_ok = True
                        chosen_cost = float(cost_used)
                        return True
                return False

            ok_any = _try_actions()
            if not ok_any:
                objective_kind = "COMM_ADMIT_UNKNOWN"
                expected_text, action_inputs, hint_action_id = _build_expected_and_action_inputs_v96(
                    objective_kind=str(objective_kind),
                    parse=dict(parse),
                    vars_map=dict(vars_map),
                    last_answer=last_answer,
                    beliefs_by_key=dict(_active_beliefs_by_key()),
                    ctx=dict(ctx2),
                    user_text=str(user_text),
                )
                # minimal fallback attempt: admit_unknown
                ranked2 = [("concept_v90_admit_unknown_v0", SupportClaimV89(goal_id="COMM_ADMIT_UNKNOWN", prior_success=1.0, prior_strength=1, prior_cost=1.0, note=""))]
                plan_ranked_candidates = [{"act_id": "concept_v90_admit_unknown_v0", "expected_success": 1.0, "expected_cost": 1.0}]
                plan_attempted_actions = []
                ok_exec, out_text, _meta, cost_used = _execute_action("concept_v90_admit_unknown_v0", goal_kind="COMM_ADMIT_UNKNOWN", inputs=dict(action_inputs))
                eval_id = _stable_hash_obj({"conversation_id": str(conversation_id), "turn_id": str(ut.get("turn_id") or ""), "step": int(step), "kind": "fallback"})
                plan_attempted_actions.append({"act_id": "concept_v90_admit_unknown_v0", "eval_id": str(eval_id), "ok": bool(ok_exec)})
                assistant_text = str(out_text) if ok_exec else "Não sei."
                chosen_action_id = "concept_v90_admit_unknown_v0"
                chosen_objective_id = _objective_act_id("COMM_ADMIT_UNKNOWN")
                chosen_eval_id = str(eval_id)
                chosen_ok = bool(ok_exec)
                chosen_cost = float(cost_used)
                plan_objective_kind = "COMM_ADMIT_UNKNOWN"
                plan_objective_id = chosen_objective_id

        # Create assistant turn.
        at = TurnV96(
            conversation_id=str(conversation_id),
            turn_index=int(turn_index),
            role="assistant",
            text=str(assistant_text),
            created_step=int(step),
            offset_us=0,
            objective_id=str(chosen_objective_id),
            objective_kind=str(plan_objective_kind or objective_kind),
            action_concept_id=str(chosen_action_id),
            eval_id=str(chosen_eval_id),
        ).to_dict()
        turn_index += 1
        step += 1
        turns.append(dict(at))
        prev_turns_hash = append_chained_jsonl_v96(
            turns_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "TURN", "payload": dict(at)},
            prev_hash=prev_turns_hash,
        )
        transcript.append({"role": "assistant", "text": str(at.get("text") or ""), "turn_id": str(at.get("turn_id") or "")})
        prev_transcript_hash = append_chained_jsonl_v96(
            transcript_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "UTTERANCE", "payload": dict(transcript[-1])},
            prev_hash=prev_transcript_hash,
        )

        # Plan record (1 per user turn).
        notes = "max expected_success, tie-break min expected_cost, tie-break act_id"
        plan_obj = ActionPlanV96(
            conversation_id=str(conversation_id),
            user_turn_id=str(ut.get("turn_id") or ""),
            user_turn_index=int(ut.get("turn_index") or 0),
            intent_id=str(parse.get("intent_id") or ""),
            parse_sig=str(parse.get("parse_sig") or ""),
            objective_kind=str(plan_objective_kind or objective_kind),
            objective_id=str(plan_objective_id or chosen_objective_id),
            ranked_candidates=list(plan_ranked_candidates),
            attempted_actions=list(plan_attempted_actions),
            chosen_action_id=str(chosen_action_id),
            chosen_eval_id=str(chosen_eval_id),
            chosen_ok=bool(chosen_ok),
            notes=str(notes),
            created_step=int(step),
            memory_read_ids=list(memory_read_ids),
            memory_write_event_ids=list(memory_write_event_ids),
            belief_read_keys=list(belief_read_keys),
            belief_read_ids=list(belief_read_ids),
            belief_write_event_ids=list(belief_write_event_ids),
        ).to_dict()
        prev_plans_hash = append_chained_jsonl_v96(plans_path, dict(plan_obj), prev_hash=prev_plans_hash)
        action_plans.append(dict(plan_obj))

        trials.append(
            {
                "objective_kind": str(plan_objective_kind or objective_kind),
                "user_turn_id": str(ut.get("turn_id") or ""),
                "assistant_turn_id": str(at.get("turn_id") or ""),
                "ok": bool(chosen_ok),
                "cost_used": float(chosen_cost),
            }
        )

        teach_rejected = str(parse.get("intent_id") or "") == "INTENT_TEACH" and not bool(parse.get("teach_ok", False))
        if str(plan_objective_kind or objective_kind) not in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"} and not bool(teach_rejected):
            end_idx = int(turn_index) - 1
            start_idx = max(0, end_idx - (6 - 1))
            tail_turn_ids = [str(turns[i]["turn_id"]) for i in range(start_idx, end_idx + 1)]
            lra = sorted(learned_rules_active.keys())
            mem_active = sorted([mid for mid, okv in memory_active_ids.items() if bool(okv)], key=str)
            beliefs_active_keys = sorted([str(k) for k in belief_active_by_key.keys() if str(k)], key=str)
            st = ConversationStateV96(
                conversation_id=str(conversation_id),
                state_index=int(state_index),
                prev_state_id=str(states[-1]["state_id"] if states else ""),
                active_goals=[],
                bindings={
                    "vars": {str(k): vars_map.get(k) for k in sorted(vars_map.keys(), key=str)},
                    "last_answer": last_answer,
                    "learned_rules_active": list(lra),
                    "learned_rule_count": int(len(lra)),
                    "memory_active": list(mem_active),
                    "memory_active_count": int(len(mem_active)),
                    "belief_active_keys": list(beliefs_active_keys),
                    "belief_active_count": int(len(beliefs_active_keys)),
                },
                tail_turn_ids=list(tail_turn_ids),
                last_user_turn_id=str(ut.get("turn_id") or ""),
                last_assistant_turn_id=str(at.get("turn_id") or ""),
                created_step=int(step),
                last_step=int(step),
            ).to_dict()
            state_index += 1
            step += 1
            states.append(dict(st))
            prev_states_hash = append_chained_jsonl_v96(
                states_path,
                {"time": deterministic_iso(step=int(st["created_step"])), "step": int(st["created_step"]), "event": "STATE", "payload": dict(st)},
                prev_hash=prev_states_hash,
            )

        if bool(compound_stop_after) or str(parse.get("intent_id") or "") == INTENT_END_V92:
            break

    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v96(turns_path)),
        "parses_chain_ok": bool(verify_chained_jsonl_v96(parses_path)),
        "learned_chain_ok": bool(verify_chained_jsonl_v96(learned_path)) if os.path.exists(learned_path) else True,
        "plans_chain_ok": bool(verify_chained_jsonl_v96(plans_path)),
        "memory_chain_ok": bool(verify_chained_jsonl_v96(memory_path)),
        "belief_chain_ok": bool(verify_chained_jsonl_v96(belief_path)),
        "states_chain_ok": bool(verify_chained_jsonl_v96(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v96(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v96(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v96(transcript_path)),
    }
    ok_chain, chain_reason, chain_details = verify_conversation_chain_v96(
        turns=list(turns),
        states=list(states),
        parse_events=list(parse_events),
        trials=list(trials),
        learned_rule_events=list(learned_rule_events),
        action_plans=list(action_plans),
        memory_events=list(memory_events),
        belief_events=list(belief_events),
        tail_k=6,
    )

    transcript_hash = compute_transcript_hash_v96(turns)
    state_chain_hash = compute_state_chain_hash_v96(states)
    parse_chain_hash = compute_parse_chain_hash_v96(parse_events)
    learned_chain_hash = compute_learned_chain_hash_v96(learned_rule_events)
    plan_chain_hash = compute_plan_chain_hash_v96(action_plans)
    memory_chain_hash = compute_memory_chain_hash_v96(memory_events)
    belief_chain_hash = compute_belief_chain_hash_v96(belief_events)

    verify_obj = {
        "ok": bool(all(chains.values())) and bool(ok_chain),
        "chains": dict(chains),
        "chain_invariants": {"ok": bool(ok_chain), "reason": str(chain_reason), "details": dict(chain_details)},
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
    }
    tmpv = verify_path + ".tmp"
    with open(tmpv, "w", encoding="utf-8") as f:
        f.write(json.dumps(verify_obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpv, verify_path)

    manifest_core = {
        "schema_version": 5,
        "conversation_id": str(conversation_id),
        "seed": int(seed),
        "store_hash": str(store_hash),
        "grammar_hash": str(grammar_snapshot.get("grammar_hash") or ""),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "verify_ok": bool(verify_obj.get("ok", False)),
        "sha256": {
            "store_jsonl": str(sha256_file(store_path)),
            "intent_grammar_snapshot_json": str(sha256_file(grammar_snapshot_path)),
            "conversation_turns_jsonl": str(sha256_file(turns_path)),
            "intent_parses_jsonl": str(sha256_file(parses_path)),
            "learned_intent_rules_jsonl": str(sha256_file(learned_path)) if os.path.exists(learned_path) else "",
            "action_plans_jsonl": str(sha256_file(plans_path)),
            "memory_events_jsonl": str(sha256_file(memory_path)),
            "belief_events_jsonl": str(sha256_file(belief_path)),
            "conversation_states_jsonl": str(sha256_file(states_path)) if os.path.exists(states_path) else "",
            "dialogue_trials_jsonl": str(sha256_file(trials_path)),
            "objective_evals_jsonl": str(sha256_file(evals_path)),
            "transcript_jsonl": str(sha256_file(transcript_path)),
            "verify_chain_v96_json": str(sha256_file(verify_path)),
        },
    }
    tmpm = manifest_path + ".tmp"
    with open(tmpm, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_core, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpm, manifest_path)
    ledger_hash = sha256_file(manifest_path)

    user_turns_total = (len(turns) + 1) // 2 if turns else 0
    parses_ok = sum(
        1
        for p in parse_events
        if isinstance(p, dict) and isinstance(p.get("payload"), dict) and bool(p["payload"].get("parse_ok", False))
    )
    clarifications = sum(1 for tr in trials if isinstance(tr, dict) and str(tr.get("objective_kind") or "") == "COMM_ASK_CLARIFY")
    unknowns = sum(
        1
        for p in parse_events
        if isinstance(p, dict) and isinstance(p.get("payload"), dict) and str(p["payload"].get("intent_id") or "") == INTENT_UNKNOWN_V92
    )

    beliefs_add_total = sum(
        1
        for p in parse_events
        if isinstance(p, dict) and isinstance(p.get("payload"), dict) and str(p["payload"].get("intent_id") or "") == INTENT_BELIEF_ADD_V96
    )
    beliefs_revise_total = sum(
        1
        for p in parse_events
        if isinstance(p, dict) and isinstance(p.get("payload"), dict) and str(p["payload"].get("intent_id") or "") == INTENT_BELIEF_REVISE_V96
    )
    beliefs_list_total = sum(
        1
        for p in parse_events
        if isinstance(p, dict) and isinstance(p.get("payload"), dict) and str(p["payload"].get("intent_id") or "") == INTENT_BELIEF_LIST_V96
    )
    beliefs_forget_total = sum(
        1
        for p in parse_events
        if isinstance(p, dict) and isinstance(p.get("payload"), dict) and str(p["payload"].get("intent_id") or "") == INTENT_BELIEF_FORGET_V96
    )

    core = {
        "schema_version": 5,
        "seed": int(seed),
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "ledger_hash": str(ledger_hash),
        "turns_total": int(len(turns)),
        "user_turns_total": int(user_turns_total),
        "states_total": int(len(states)),
        "plans_total": int(len(action_plans)),
        "memory_events_total": int(len(memory_events)),
        "belief_events_total": int(len(belief_events)),
        "parses_total": int(len(parse_events)),
        "parses_ok": int(parses_ok),
        "clarifications": int(clarifications),
        "unknowns": int(unknowns),
        "belief_add_total": int(beliefs_add_total),
        "belief_revise_total": int(beliefs_revise_total),
        "belief_list_total": int(beliefs_list_total),
        "belief_forget_total": int(beliefs_forget_total),
        "verify_ok": bool(verify_obj.get("ok", False)),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    summary = dict(core, summary_sha256=str(summary_sha256))
    tmps = summary_path + ".tmp"
    with open(tmps, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmps, summary_path)

    return {
        "schema_version": 5,
        "out_dir": str(out_dir),
        "conversation_id": str(conversation_id),
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "ledger_hash": str(ledger_hash),
        "summary_sha256": str(summary_sha256),
        "paths": {
            "store_jsonl": str(store_path),
            "grammar_snapshot_json": str(grammar_snapshot_path),
            "turns_jsonl": str(turns_path),
            "parses_jsonl": str(parses_path),
            "learned_rules_jsonl": str(learned_path),
            "plans_jsonl": str(plans_path),
            "memory_events_jsonl": str(memory_path),
            "belief_events_jsonl": str(belief_path),
            "states_jsonl": str(states_path),
            "trials_jsonl": str(trials_path),
            "evals_jsonl": str(evals_path),
            "transcript_jsonl": str(transcript_path),
            "verify_json": str(verify_path),
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
        },
    }
