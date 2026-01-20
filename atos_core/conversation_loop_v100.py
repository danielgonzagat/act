from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .conversation_actions_v90 import action_concepts_for_dsl_v90
from .conversation_objectives_v90 import COMM_OBJECTIVES_V90, comm_objective_ids_v90, make_comm_objective_eq_text_v90
from .conversation_v96 import (
    BeliefEventV96,
    BeliefItemV96,
    ConversationStateV96,
    MemoryEventV96,
    MemoryItemV96,
    TurnV96,
    append_chained_jsonl_v96,
    action_plan_id_v96,
    belief_event_id_v96,
    belief_event_sig_v96,
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
    text_sig_v96,
    verify_chained_jsonl_v96,
)
from .conversation_v97 import make_system_spec_v97, render_system_text_v97
from .conversation_v98 import (
    EvidenceEventV98,
    EvidenceItemV98,
    compute_evidence_chain_hash_v98,
    render_evidence_added_ack_text_v98,
    render_evidences_text_v98,
    render_versions_text_v98,
    render_why_belief_text_v98,
)
from .conversation_v99 import compute_goal_chain_hash_v99, render_auto_text_v99, render_goal_added_ack_text_v99, render_goal_done_ack_text_v99, render_goals_text_v99, render_next_text_v99
from .conversation_v100 import render_explain_text_v100, verify_conversation_chain_v100
from .discourse_ledger_v100 import DiscourseEventV100, compute_discourse_chain_hash_v100, generate_text_candidates_v100, render_discourse_text_v100, render_dossier_text_v100, sha256_text_v100
from .engine_v80 import EngineV80
from .fragment_library_v100 import (
    FragmentEventV100,
    base_fragments_v100,
    compute_fragment_chain_hash_v100,
    fold_fragment_library_v100,
    fragment_library_snapshot_v100,
    fragment_should_promote_v100,
)
from .goal_ledger_v99 import GoalEventV99, fold_goal_ledger_v99, goal_id_v99, goal_ledger_snapshot_v99
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
from .intent_grammar_v98 import (
    INTENT_DOSSIER_V98,
    INTENT_EVIDENCE_ADD_V98,
    INTENT_EVIDENCE_LIST_V98,
    INTENT_SYSTEM_V98,
    INTENT_VERSIONS_V98,
    INTENT_WHY_V98,
    is_dossier_command_v98,
    is_evidence_add_command_v98,
    is_evidences_list_command_v98,
    is_system_command_v98,
    is_versions_command_v98,
    is_why_command_v98,
    parse_dossier_command_v98,
    parse_evidence_add_command_v98,
    parse_evidences_list_command_v98,
    parse_system_command_v98,
    parse_versions_command_v98,
    parse_why_command_v98,
)
from .intent_grammar_v99 import (
    INTENT_GOAL_ADD_V99,
    INTENT_GOAL_AUTO_V99,
    INTENT_GOAL_DONE_V99,
    INTENT_GOAL_LIST_V99,
    INTENT_GOAL_NEXT_V99,
    is_auto_command_v99,
    is_done_command_v99,
    is_goal_add_command_v99,
    is_goals_list_command_v99,
    is_next_command_v99,
    parse_auto_command_v99,
    parse_done_command_v99,
    parse_goal_add_command_v99,
    parse_goals_list_command_v99,
    parse_next_command_v99,
)
from .intent_grammar_v100 import (
    INTENT_DISCOURSE_V100,
    INTENT_WHY_REF_V100,
    is_discourse_command_v100,
    is_why_ref_command_v100,
    parse_discourse_command_v100,
    parse_why_ref_command_v100,
)
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


def _deepcopy_json_obj(obj: Any) -> Any:
    return json.loads(canonical_json_dumps(obj))


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

    if intent_id == INTENT_SYSTEM_V98:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("system_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_DISCOURSE_V100:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("discourse_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_WHY_REF_V100:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("why_ref_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_EVIDENCE_ADD_V98:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("evidence_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_EVIDENCE_LIST_V98:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("evidences_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_WHY_V98:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("why_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_VERSIONS_V98:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("versions_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_DOSSIER_V98:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("dossier_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_GOAL_ADD_V99:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("goal_ack_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_GOAL_LIST_V99:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("goals_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_GOAL_DONE_V99:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("goal_done_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_GOAL_NEXT_V99:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("next_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_GOAL_AUTO_V99:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("auto_text") or "")
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


def run_conversation_v100(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
) -> Dict[str, Any]:
    ensure_absent(str(out_dir))
    os.makedirs(str(out_dir), exist_ok=False)
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    store_path = os.path.join(str(out_dir), "store.jsonl")
    grammar_snapshot_path = os.path.join(str(out_dir), "intent_grammar_snapshot.json")
    system_spec_path = os.path.join(str(out_dir), "system_spec_snapshot.json")
    turns_path = os.path.join(str(out_dir), "conversation_turns.jsonl")
    parses_path = os.path.join(str(out_dir), "intent_parses.jsonl")
    learned_path = os.path.join(str(out_dir), "learned_intent_rules.jsonl")
    plans_path = os.path.join(str(out_dir), "action_plans.jsonl")
    memory_path = os.path.join(str(out_dir), "memory_events.jsonl")
    belief_path = os.path.join(str(out_dir), "belief_events.jsonl")
    evidence_path = os.path.join(str(out_dir), "evidence_events.jsonl")
    goal_path = os.path.join(str(out_dir), "goal_events.jsonl")
    goal_snapshot_path = os.path.join(str(out_dir), "goal_ledger_snapshot.json")
    discourse_path = os.path.join(str(out_dir), "discourse_events.jsonl")
    fragment_events_path = os.path.join(str(out_dir), "fragment_events.jsonl")
    fragment_snapshot_path = os.path.join(str(out_dir), "fragment_library_snapshot.json")
    states_path = os.path.join(str(out_dir), "conversation_states.jsonl")
    trials_path = os.path.join(str(out_dir), "dialogue_trials.jsonl")
    evals_path = os.path.join(str(out_dir), "objective_evals.jsonl")
    transcript_path = os.path.join(str(out_dir), "transcript.jsonl")
    verify_path = os.path.join(str(out_dir), "verify_chain_v100.json")
    manifest_path = os.path.join(str(out_dir), "freeze_manifest_v100.json")
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

    # System spec snapshot (WORM; deterministic, no runtime-dependent fields).
    if os.path.exists(system_spec_path):
        _fail(f"system_spec_snapshot_exists:{system_spec_path}")
    system_spec = make_system_spec_v97()
    tmpss = system_spec_path + ".tmp"
    with open(tmpss, "w", encoding="utf-8") as f:
        f.write(json.dumps(system_spec, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpss, system_spec_path)

    # Ensure ledgers exist even if empty (write-once; WORM).
    with open(memory_path, "x", encoding="utf-8") as _f:
        pass
    with open(belief_path, "x", encoding="utf-8") as _f:
        pass
    with open(evidence_path, "x", encoding="utf-8") as _f:
        pass
    with open(goal_path, "x", encoding="utf-8") as _f:
        pass
    with open(discourse_path, "x", encoding="utf-8") as _f:
        pass
    with open(fragment_events_path, "x", encoding="utf-8") as _f:
        pass

    engine = EngineV80(store, seed=int(seed))

    conv_body = {"turns": list(user_turn_texts), "grammar_hash": str(grammar_hash_v92(rules))}
    conversation_id = f"conv_v100_{sha256_hex(canonical_json_dumps(conv_body).encode('utf-8'))}"

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
    evidence_events: List[Dict[str, Any]] = []
    goal_events: List[Dict[str, Any]] = []
    discourse_events: List[Dict[str, Any]] = []
    fragment_events: List[Dict[str, Any]] = []

    prev_turns_hash: Optional[str] = None
    prev_parses_hash: Optional[str] = None
    prev_learned_hash: Optional[str] = None
    prev_plans_hash: Optional[str] = None
    prev_memory_hash: Optional[str] = None
    prev_belief_hash: Optional[str] = None
    prev_evidence_hash: Optional[str] = None
    prev_goal_hash: Optional[str] = None
    prev_discourse_hash: Optional[str] = None
    prev_fragment_hash: Optional[str] = None
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

    prev_goal_event_sig = ""
    prev_discourse_event_sig = ""
    prev_fragment_event_sig = ""

    # Discourse state (persisted via discourse_events; also echoed in conversation state bindings).
    discourse_state: Dict[str, Any] = {
        "schema_version": 100,
        "kind": "discourse_state_v100",
        "active_topics": [],
        "open_questions": [],
        "commitments": [],
        "style_prefs": {},
        "ref_map": {},
    }

    base_fragments = base_fragments_v100()

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
                    # Allow a small set of raw-intercept commands as RHS targets (deterministic, fail-closed).
                    # This keeps TEACH consistent with the conversation loop's raw intercept layer (e.g., BELIEFS).
                    if not bool(rhs_parse.get("parse_ok", False)) and is_beliefs_list_command_v96(str(rhs_raw)):
                        be2 = parse_beliefs_list_command_v96(str(rhs_raw))
                        ok2 = bool(be2.get("recognized", False)) and bool(be2.get("ok", False))
                        reason2 = str(be2.get("reason") or "not_recognized")
                        sem2 = {
                            "schema_version": 96,
                            "intent_id": INTENT_BELIEF_LIST_V96,
                            "matched_rule_id": "",
                            "compound": False,
                            "parse_ok": bool(ok2),
                            "reason": str(reason2),
                            "prefix": str(be2.get("prefix") or ""),
                        }
                        sig2 = _stable_hash_obj(sem2)
                        rhs_parse = dict(sem2, parse_sig=str(sig2))
                    rhs_parse_sig = str(rhs_parse.get("parse_sig") or "")
                    rhs_intent_id = str(rhs_parse.get("intent_id") or "")
                    rhs_rule_id = str(rhs_parse.get("matched_rule_id") or "")
                    rhs_reason = str(rhs_parse.get("reason") or "")
                    if not bool(rhs_parse.get("parse_ok", False)):
                        teach_reason = f"rhs_parse_fail:{rhs_reason or 'no_match'}"
                    elif rhs_parse.get("missing_slots"):
                        teach_reason = "rhs_missing_slots"
                    elif rhs_intent_id not in {INTENT_SUMMARY_V92, INTENT_END_V92, INTENT_BELIEF_LIST_V96}:
                        teach_reason = "rhs_intent_disallowed"
                    else:
                        learned_rule = make_learned_intent_rule_v93(
                            intent_id=str(rhs_intent_id),
                            lhs_tokens_canon_stripped=list(stripped_list),
                        )
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
                    explain_text = render_explain_text_v100(lastp)
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

        elif is_system_command_v98(str(user_text)):
            se = parse_system_command_v98(str(user_text))
            ok = bool(se.get("recognized", False)) and bool(se.get("ok", False))
            reason = str(se.get("reason") or "not_recognized")
            sem = {
                "schema_version": 98,
                "intent_id": INTENT_SYSTEM_V98,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(se.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_discourse_command_v100(str(user_text)):
            de = parse_discourse_command_v100(str(user_text))
            ok = bool(de.get("recognized", False)) and bool(de.get("ok", False))
            reason = str(de.get("reason") or "not_recognized")
            sem = {
                "schema_version": 100,
                "intent_id": INTENT_DISCOURSE_V100,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(de.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_evidence_add_command_v98(str(user_text)):
            ee = parse_evidence_add_command_v98(str(user_text))
            ok = bool(ee.get("recognized", False)) and bool(ee.get("ok", False))
            reason = str(ee.get("reason") or "not_recognized")
            sem = {
                "schema_version": 98,
                "intent_id": INTENT_EVIDENCE_ADD_V98,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(ee.get("prefix") or ""),
                "evidence_kind": str(ee.get("evidence_kind") or ""),
                "evidence_key": str(ee.get("evidence_key") or ""),
                "evidence_value": str(ee.get("evidence_value") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_evidences_list_command_v98(str(user_text)):
            el = parse_evidences_list_command_v98(str(user_text))
            ok = bool(el.get("recognized", False)) and bool(el.get("ok", False))
            reason = str(el.get("reason") or "not_recognized")
            sem = {
                "schema_version": 98,
                "intent_id": INTENT_EVIDENCE_LIST_V98,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(el.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_why_ref_command_v100(str(user_text)):
            wr = parse_why_ref_command_v100(str(user_text))
            ok = bool(wr.get("recognized", False)) and bool(wr.get("ok", False))
            reason = str(wr.get("reason") or "not_recognized")
            sem = {
                "schema_version": 100,
                "intent_id": INTENT_WHY_REF_V100,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(wr.get("prefix") or ""),
                "ref_token": str(wr.get("ref_token") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_why_command_v98(str(user_text)):
            wy = parse_why_command_v98(str(user_text))
            ok = bool(wy.get("recognized", False)) and bool(wy.get("ok", False))
            reason = str(wy.get("reason") or "not_recognized")
            sem = {
                "schema_version": 98,
                "intent_id": INTENT_WHY_V98,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(wy.get("prefix") or ""),
                "key": str(wy.get("key") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_versions_command_v98(str(user_text)):
            ve = parse_versions_command_v98(str(user_text))
            ok = bool(ve.get("recognized", False)) and bool(ve.get("ok", False))
            reason = str(ve.get("reason") or "not_recognized")
            sem = {
                "schema_version": 98,
                "intent_id": INTENT_VERSIONS_V98,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(ve.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_dossier_command_v98(str(user_text)):
            de = parse_dossier_command_v98(str(user_text))
            ok = bool(de.get("recognized", False)) and bool(de.get("ok", False))
            reason = str(de.get("reason") or "not_recognized")
            sem = {
                "schema_version": 98,
                "intent_id": INTENT_DOSSIER_V98,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(de.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_goal_add_command_v99(str(user_text)):
            ge = parse_goal_add_command_v99(str(user_text))
            ok = bool(ge.get("recognized", False)) and bool(ge.get("ok", False))
            reason = str(ge.get("reason") or "not_recognized")
            sem = {
                "schema_version": 99,
                "intent_id": INTENT_GOAL_ADD_V99,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(ge.get("prefix") or ""),
                "goal_text": str(ge.get("goal_text") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_goals_list_command_v99(str(user_text)):
            gl = parse_goals_list_command_v99(str(user_text))
            ok = bool(gl.get("recognized", False)) and bool(gl.get("ok", False))
            reason = str(gl.get("reason") or "not_recognized")
            sem = {
                "schema_version": 99,
                "intent_id": INTENT_GOAL_LIST_V99,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(gl.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_done_command_v99(str(user_text)):
            dn = parse_done_command_v99(str(user_text))
            ok = bool(dn.get("recognized", False)) and bool(dn.get("ok", False))
            reason = str(dn.get("reason") or "not_recognized")
            sem = {
                "schema_version": 99,
                "intent_id": INTENT_GOAL_DONE_V99,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(dn.get("prefix") or ""),
                "goal_id": str(dn.get("goal_id") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_next_command_v99(str(user_text)):
            nx = parse_next_command_v99(str(user_text))
            ok = bool(nx.get("recognized", False)) and bool(nx.get("ok", False))
            reason = str(nx.get("reason") or "not_recognized")
            sem = {
                "schema_version": 99,
                "intent_id": INTENT_GOAL_NEXT_V99,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(nx.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_auto_command_v99(str(user_text)):
            au = parse_auto_command_v99(str(user_text))
            ok = bool(au.get("recognized", False)) and bool(au.get("ok", False))
            reason = str(au.get("reason") or "not_recognized")
            sem = {
                "schema_version": 99,
                "intent_id": INTENT_GOAL_AUTO_V99,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(au.get("prefix") or ""),
                "n": int(au.get("n") or 0) if bool(ok) else 0,
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

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
        evidence_read_ids: List[str] = []
        evidence_write_ids: List[str] = []
        cause_evidence_ids: List[str] = []
        goal_read_ids: List[str] = []
        goal_write_ids: List[str] = []
        goal_tick_actions: List[Dict[str, Any]] = []

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

        elif str(parse.get("intent_id") or "") == INTENT_SYSTEM_V98:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"system_text": render_system_text_v97(system_spec)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"system_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_DISCOURSE_V100:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"discourse_text": render_discourse_text_v100(dict(discourse_state))}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"discourse_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_WHY_REF_V100:
            if bool(parse.get("parse_ok", False)):
                keys = sorted([str(k) for k in belief_active_by_key.keys() if isinstance(k, str) and k])
                if len(keys) >= 2:
                    objective_kind = "COMM_CONFIRM"
                    txt = "Qual chave? " + "; ".join(keys[:5])
                    ctx2 = {"why_ref_text": str(txt), "open_question_keys": list(keys[:5])}
                elif len(keys) == 1:
                    objective_kind = "COMM_RESPOND"
                    txt2 = render_why_belief_text_v98(key=str(keys[0]), belief_events=belief_events, evidence_events=evidence_events, step=int(step))
                    ctx2 = {"why_ref_text": str(txt2)}
                else:
                    objective_kind = "COMM_ADMIT_UNKNOWN"
                    ctx2 = {}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"why_ref_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_EVIDENCE_ADD_V98:
            if bool(parse.get("parse_ok", False)):
                ekey = str(parse.get("evidence_key") or "").strip()
                eval0 = str(parse.get("evidence_value") or "").strip()
                # 1) Always append evidence to evidence ledger.
                ei = EvidenceItemV98(
                    conversation_id=str(conversation_id),
                    evidence_kind="OBSERVE",
                    evidence_key=str(ekey),
                    evidence_value=str(eval0),
                    source_turn_id=str(ut.get("turn_id") or ""),
                    created_step=int(step),
                ).to_dict()
                evidence_id = str(ei.get("evidence_id") or "")
                ee = EvidenceEventV98(
                    conversation_id=str(conversation_id),
                    evidence_item=dict(ei),
                    created_step=int(step),
                    source_turn_id=str(ut.get("turn_id") or ""),
                ).to_dict()
                prev_evidence_hash = append_chained_jsonl_v96(evidence_path, dict(ee), prev_hash=prev_evidence_hash)
                evidence_events.append(dict(ee))
                if evidence_id:
                    evidence_write_ids.append(str(evidence_id))

                # 2) Deterministic belief revision driven by evidence.
                belief_changed = False
                ack_lines: List[str] = [render_evidence_added_ack_text_v98(evidence_id=str(evidence_id), key=str(ekey))]
                if ekey in belief_active_by_key:
                    old_bid = str(belief_active_by_key.get(ekey) or "")
                    old_bi = belief_items_by_id.get(old_bid) if old_bid else None
                    old_val = str(old_bi.get("belief_value") or "") if isinstance(old_bi, dict) else ""
                    if old_val != eval0:
                        # RETRACT(old) then ADD(new), both bound to cause evidence.
                        retract_sem = {
                            "schema_version": 96,
                            "kind": "belief_event_v96",
                            "conversation_id": str(conversation_id),
                            "event_kind": "RETRACT",
                            "created_step": int(step),
                            "created_at": deterministic_iso(step=int(step)),
                            "source_turn_id": str(ut.get("turn_id") or ""),
                            "belief_item": None,
                            "target_belief_id": str(old_bid),
                            "retract_reason": "evidence_observe",
                            "cause_evidence_id": str(evidence_id),
                            "cause_evidence_key": str(ekey),
                            "cause_evidence_value": str(eval0),
                        }
                        rsig = belief_event_sig_v96(dict(retract_sem))
                        rid = belief_event_id_v96(rsig)
                        retract_ev = dict(retract_sem, event_sig=str(rsig), event_id=str(rid))
                        prev_belief_hash = append_chained_jsonl_v96(belief_path, dict(retract_ev), prev_hash=prev_belief_hash)
                        belief_events.append(dict(retract_ev))
                        belief_write_event_ids.append(str(retract_ev.get("event_id") or ""))
                        belief_active_by_key.pop(str(ekey), None)

                        bi_new = BeliefItemV96(
                            conversation_id=str(conversation_id),
                            belief_key=str(ekey),
                            belief_value=str(eval0),
                            source_turn_id=str(ut.get("turn_id") or ""),
                            created_step=int(step),
                        ).to_dict()
                        new_bid = str(bi_new.get("belief_id") or "")
                        if new_bid:
                            belief_items_by_id[new_bid] = dict(bi_new)
                            belief_active_by_key[str(ekey)] = str(new_bid)
                        add_sem = {
                            "schema_version": 96,
                            "kind": "belief_event_v96",
                            "conversation_id": str(conversation_id),
                            "event_kind": "ADD",
                            "created_step": int(step),
                            "created_at": deterministic_iso(step=int(step)),
                            "source_turn_id": str(ut.get("turn_id") or ""),
                            "belief_item": dict(bi_new),
                            "target_belief_id": "",
                            "retract_reason": "",
                            "cause_evidence_id": str(evidence_id),
                            "cause_evidence_key": str(ekey),
                            "cause_evidence_value": str(eval0),
                        }
                        asig = belief_event_sig_v96(dict(add_sem))
                        aid = belief_event_id_v96(asig)
                        add_ev = dict(add_sem, event_sig=str(asig), event_id=str(aid))
                        prev_belief_hash = append_chained_jsonl_v96(belief_path, dict(add_ev), prev_hash=prev_belief_hash)
                        belief_events.append(dict(add_ev))
                        belief_write_event_ids.append(str(add_ev.get("event_id") or ""))
                        belief_changed = True
                        cause_evidence_ids = [str(evidence_id)] if evidence_id else []
                        ack_lines.append(render_belief_revised_ack_text_v96(key=str(ekey), old_id=str(old_bid), new_id=str(new_bid)))
                    else:
                        # Evidence matches current belief: no belief update.
                        belief_changed = False
                else:
                    bi_new = BeliefItemV96(
                        conversation_id=str(conversation_id),
                        belief_key=str(ekey),
                        belief_value=str(eval0),
                        source_turn_id=str(ut.get("turn_id") or ""),
                        created_step=int(step),
                    ).to_dict()
                    new_bid = str(bi_new.get("belief_id") or "")
                    if new_bid:
                        belief_items_by_id[new_bid] = dict(bi_new)
                        belief_active_by_key[str(ekey)] = str(new_bid)
                    add_sem = {
                        "schema_version": 96,
                        "kind": "belief_event_v96",
                        "conversation_id": str(conversation_id),
                        "event_kind": "ADD",
                        "created_step": int(step),
                        "created_at": deterministic_iso(step=int(step)),
                        "source_turn_id": str(ut.get("turn_id") or ""),
                        "belief_item": dict(bi_new),
                        "target_belief_id": "",
                        "retract_reason": "",
                        "cause_evidence_id": str(evidence_id),
                        "cause_evidence_key": str(ekey),
                        "cause_evidence_value": str(eval0),
                    }
                    asig = belief_event_sig_v96(dict(add_sem))
                    aid = belief_event_id_v96(asig)
                    add_ev = dict(add_sem, event_sig=str(asig), event_id=str(aid))
                    prev_belief_hash = append_chained_jsonl_v96(belief_path, dict(add_ev), prev_hash=prev_belief_hash)
                    belief_events.append(dict(add_ev))
                    belief_write_event_ids.append(str(add_ev.get("event_id") or ""))
                    belief_changed = True
                    cause_evidence_ids = [str(evidence_id)] if evidence_id else []
                    ack_lines.append(render_belief_added_ack_text_v96(belief_id=str(new_bid), key=str(ekey)))

                objective_kind = "COMM_RESPOND"
                ctx2 = {"evidence_text": "\n".join([ln for ln in ack_lines if isinstance(ln, str) and ln])}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"evidence_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_EVIDENCE_LIST_V98:
            if bool(parse.get("parse_ok", False)):
                evs_at = [dict(ev) for ev in evidence_events if isinstance(ev, dict) and int(ev.get("created_step", -1) or -1) <= int(step)]
                evidence_read_ids = sorted(
                    {
                        str(it.get("evidence_id") or "")
                        for ev in evs_at
                        for it in [ev.get("evidence_item")]
                        if isinstance(it, dict) and str(it.get("evidence_id") or "")
                    }
                )
                objective_kind = "COMM_RESPOND"
                ctx2 = {"evidences_text": render_evidences_text_v98(evs_at)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"evidences_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_WHY_V98:
            if bool(parse.get("parse_ok", False)):
                wkey = str(parse.get("key") or "").strip()
                beliefs_active2 = _active_beliefs_by_key()
                if wkey and (wkey in beliefs_active2):
                    bit = beliefs_active2.get(wkey) if isinstance(beliefs_active2.get(wkey), dict) else {}
                    belief_read_keys = [str(wkey)]
                    belief_read_ids = [str(bit.get("belief_id") or "")] if str(bit.get("belief_id") or "") else []
                    # Best-effort: include the last cause_evidence_id for this key (if any).
                    ceid = ""
                    for bev in reversed(list(belief_events)):
                        if not isinstance(bev, dict):
                            continue
                        try:
                            cstep = int(bev.get("created_step", -1))
                        except Exception:
                            cstep = -1
                        if cstep < 0 or cstep > int(step):
                            continue
                        if str(bev.get("event_kind") or "") != "ADD":
                            continue
                        bi = bev.get("belief_item")
                        if not isinstance(bi, dict):
                            continue
                        if str(bi.get("belief_key") or "").strip() != str(wkey):
                            continue
                        ceid = str(bev.get("cause_evidence_id") or "")
                        if ceid:
                            break
                    if ceid:
                        evidence_read_ids = [str(ceid)]
                objective_kind = "COMM_RESPOND"
                ctx2 = {
                    "why_text": render_why_belief_text_v98(
                        key=str(wkey), belief_events=list(belief_events), evidence_events=list(evidence_events), step=int(step)
                    )
                }
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"why_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_VERSIONS_V98:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"versions_text": render_versions_text_v98(repo_root=str(repo_root))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"versions_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_DOSSIER_V98:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"dossier_text": render_dossier_text_v100()}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"dossier_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_GOAL_ADD_V99:
            if bool(parse.get("parse_ok", False)):
                gtext = str(parse.get("goal_text") or "").strip()
                if not gtext:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "empty_goal_text"}
                else:
                    gid = goal_id_v99(conversation_id=str(conversation_id), ts_turn_index=int(ut.get("turn_index") or 0), text=str(gtext), parent_goal_id="")
                    ev = GoalEventV99(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        op="GOAL_ADD",
                        goal_id=str(gid),
                        parent_goal_id="",
                        priority=100,
                        status="active",
                        text=str(gtext),
                        cause_type="user_intent",
                        cause_id=str(ut.get("turn_id") or ""),
                        created_step=int(step),
                        prev_event_sig=str(prev_goal_event_sig),
                    ).to_dict()
                    prev_goal_hash = append_chained_jsonl_v96(
                        goal_path,
                        {"time": deterministic_iso(step=int(step)), "step": int(step), "event": "GOAL_EVENT", "payload": dict(ev)},
                        prev_hash=prev_goal_hash,
                    )
                    goal_events.append(dict(ev))
                    prev_goal_event_sig = str(ev.get("event_sig") or "")
                    goal_write_ids = [str(gid)]
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"goal_ack_text": render_goal_added_ack_text_v99(str(gid))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"goal_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_GOAL_LIST_V99:
            if bool(parse.get("parse_ok", False)):
                snap = goal_ledger_snapshot_v99(list(goal_events))
                goals_active = snap.get("goals_active") if isinstance(snap.get("goals_active"), list) else []
                goal_read_ids = [str(it.get("goal_id") or "") for it in goals_active if isinstance(it, dict) and str(it.get("goal_id") or "")]
                objective_kind = "COMM_RESPOND"
                ctx2 = {"goals_text": render_goals_text_v99(goals_active)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"goals_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_GOAL_DONE_V99:
            if bool(parse.get("parse_ok", False)):
                target_gid = str(parse.get("goal_id") or "")
                active_by_id, _gdetails = fold_goal_ledger_v99(list(goal_events))
                cur = active_by_id.get(target_gid) if isinstance(active_by_id.get(target_gid), dict) else {}
                if not target_gid or not cur:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "missing_goal"}
                else:
                    ev = GoalEventV99(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        op="GOAL_DONE",
                        goal_id=str(target_gid),
                        parent_goal_id=str(cur.get("parent_goal_id") or ""),
                        priority=int(cur.get("priority") or 0),
                        status="done",
                        text=str(cur.get("text") or ""),
                        cause_type="user_intent",
                        cause_id=str(ut.get("turn_id") or ""),
                        created_step=int(step),
                        prev_event_sig=str(prev_goal_event_sig),
                    ).to_dict()
                    prev_goal_hash = append_chained_jsonl_v96(
                        goal_path,
                        {"time": deterministic_iso(step=int(step)), "step": int(step), "event": "GOAL_EVENT", "payload": dict(ev)},
                        prev_hash=prev_goal_hash,
                    )
                    goal_events.append(dict(ev))
                    prev_goal_event_sig = str(ev.get("event_sig") or "")
                    goal_read_ids = [str(target_gid)]
                    goal_write_ids = [str(target_gid)]
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"goal_done_text": render_goal_done_ack_text_v99(str(target_gid))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"done_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_GOAL_NEXT_V99:
            if bool(parse.get("parse_ok", False)):
                active_by_id2, _gdetails2 = fold_goal_ledger_v99(list(goal_events))
                goals_active2 = [dict(active_by_id2[k]) for k in sorted(active_by_id2.keys(), key=str) if isinstance(active_by_id2.get(k), dict)]
                goals_active2.sort(key=lambda d: (-int(d.get("priority") or 0), int(d.get("created_ts_turn_index") or 0), str(d.get("goal_id") or "")))
                if not goals_active2:
                    ta = {"tick_index": 0, "selected_goal_id": "", "effect": "no_active_goals", "subgoal_id": ""}
                    goal_tick_actions = [dict(ta)]
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"next_text": render_next_text_v99(dict(ta))}
                else:
                    sel = dict(goals_active2[0])
                    sel_gid = str(sel.get("goal_id") or "")
                    sub_text = f"subgoal_tick0_for:{sel_gid}"
                    sub_gid = goal_id_v99(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        text=str(sub_text),
                        parent_goal_id=str(sel_gid),
                    )
                    ev = GoalEventV99(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        op="GOAL_ADD",
                        goal_id=str(sub_gid),
                        parent_goal_id=str(sel_gid),
                        priority=max(0, int(sel.get("priority") or 0) - 1),
                        status="active",
                        text=str(sub_text),
                        cause_type="user_intent",
                        cause_id=str(ut.get("turn_id") or ""),
                        created_step=int(step),
                        prev_event_sig=str(prev_goal_event_sig),
                    ).to_dict()
                    prev_goal_hash = append_chained_jsonl_v96(
                        goal_path,
                        {"time": deterministic_iso(step=int(step)), "step": int(step), "event": "GOAL_EVENT", "payload": dict(ev)},
                        prev_hash=prev_goal_hash,
                    )
                    goal_events.append(dict(ev))
                    prev_goal_event_sig = str(ev.get("event_sig") or "")
                    goal_read_ids = [str(sel_gid)]
                    goal_write_ids = [str(sub_gid)]
                    ta2 = {"tick_index": 0, "selected_goal_id": str(sel_gid), "effect": "created_subgoal", "subgoal_id": str(sub_gid)}
                    goal_tick_actions = [dict(ta2)]
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"next_text": render_next_text_v99(dict(ta2))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"next_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_GOAL_AUTO_V99:
            if bool(parse.get("parse_ok", False)):
                n = int(parse.get("n") or 0)
                if n <= 0:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "auto_bad_n"}
                else:
                    tick_actions: List[Dict[str, Any]] = []
                    for ti in range(int(n)):
                        active_by_id3, _gdetails3 = fold_goal_ledger_v99(list(goal_events))
                        goals_active3 = [dict(active_by_id3[k]) for k in sorted(active_by_id3.keys(), key=str) if isinstance(active_by_id3.get(k), dict)]
                        goals_active3.sort(
                            key=lambda d: (-int(d.get("priority") or 0), int(d.get("created_ts_turn_index") or 0), str(d.get("goal_id") or ""))
                        )
                        if not goals_active3:
                            tick_actions.append({"tick_index": int(ti), "selected_goal_id": "", "effect": "no_active_goals", "subgoal_id": ""})
                            continue
                        sel3 = dict(goals_active3[0])
                        sel_gid3 = str(sel3.get("goal_id") or "")
                        sub_text3 = f"subgoal_tick{int(ti)}_for:{sel_gid3}"
                        sub_gid3 = goal_id_v99(
                            conversation_id=str(conversation_id),
                            ts_turn_index=int(ut.get("turn_index") or 0),
                            text=str(sub_text3),
                            parent_goal_id=str(sel_gid3),
                        )
                        ev3 = GoalEventV99(
                            conversation_id=str(conversation_id),
                            ts_turn_index=int(ut.get("turn_index") or 0),
                            op="GOAL_ADD",
                            goal_id=str(sub_gid3),
                            parent_goal_id=str(sel_gid3),
                            priority=max(0, int(sel3.get("priority") or 0) - 1),
                            status="active",
                            text=str(sub_text3),
                            cause_type="user_intent",
                            cause_id=str(ut.get("turn_id") or ""),
                            created_step=int(step),
                            prev_event_sig=str(prev_goal_event_sig),
                        ).to_dict()
                        prev_goal_hash = append_chained_jsonl_v96(
                            goal_path,
                            {"time": deterministic_iso(step=int(step)), "step": int(step), "event": "GOAL_EVENT", "payload": dict(ev3)},
                            prev_hash=prev_goal_hash,
                        )
                        goal_events.append(dict(ev3))
                        prev_goal_event_sig = str(ev3.get("event_sig") or "")
                        goal_read_ids.append(str(sel_gid3))
                        goal_write_ids.append(str(sub_gid3))
                        tick_actions.append(
                            {"tick_index": int(ti), "selected_goal_id": str(sel_gid3), "effect": "created_subgoal", "subgoal_id": str(sub_gid3)}
                        )
                    goal_read_ids = sorted(set([str(x) for x in goal_read_ids if isinstance(x, str) and x]))
                    goal_write_ids = sorted(set([str(x) for x in goal_write_ids if isinstance(x, str) and x]))
                    goal_tick_actions = [dict(x) for x in tick_actions]
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"auto_text": render_auto_text_v99(goal_tick_actions, int(n))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"auto_reject:{str(parse.get('reason') or '')}"}

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

        # V100: discourse/fluency candidates (>=3) selected deterministically before execution when possible.
        intent_id0 = str(parse.get("intent_id") or "")
        response_kind = "locked"
        if str(objective_kind) in {"COMM_RESPOND", "COMM_SUMMARIZE"}:
            response_kind = "respond"
        elif str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
            response_kind = "clarify"
        allow_wrappers = bool(
            response_kind == "respond"
            and bool(parse.get("parse_ok", False))
            and not (parse.get("missing_slots") or [])
            and (intent_id0 in {INTENT_SET_V92, INTENT_ADD_V92, INTENT_SUMMARY_V92})
        )
        discourse_base_text = str(expected_text)
        discourse_candidates_topk = generate_text_candidates_v100(
            base_text=str(discourse_base_text),
            response_kind=str(response_kind),
            allow_wrappers=bool(allow_wrappers),
        )
        discourse_selected = dict(discourse_candidates_topk[0]) if discourse_candidates_topk else {}
        if allow_wrappers and str(discourse_selected.get("text") or ""):
            # Drive execution through emit_text so the selected candidate becomes the actual assistant output.
            expected_text = str(discourse_selected.get("text") or "")
            action_inputs = {"text": str(expected_text)}
            hint_action_id = "concept_v90_emit_text_v0"

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
                # Recompute discourse candidates for the new objective/base text (fail-closed).
                discourse_base_text = str(expected_text)
                discourse_candidates_topk = generate_text_candidates_v100(
                    base_text=str(discourse_base_text),
                    response_kind="locked",
                    allow_wrappers=False,
                )
                discourse_selected = dict(discourse_candidates_topk[0]) if discourse_candidates_topk else {}
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

        # V100: discourse ledger (variants + fluency metrics) + fragment lifecycle events.
        discourse_state_before = _deepcopy_json_obj(discourse_state)

        # Update discourse_state deterministically based on the current turn's intent/effects.
        topic_map = {
            INTENT_SET_V92: "dsl",
            INTENT_GET_V92: "dsl",
            INTENT_ADD_V92: "dsl",
            INTENT_SUMMARY_V92: "dsl",
            INTENT_TEACH_V93: "learning",
            INTENT_EXPLAIN_V94: "explain",
            INTENT_SYSTEM_V98: "system",
            INTENT_VERSIONS_V98: "system",
            INTENT_DOSSIER_V98: "system",
            INTENT_EVIDENCE_ADD_V98: "evidence",
            INTENT_EVIDENCE_LIST_V98: "evidence",
            INTENT_WHY_V98: "belief",
            INTENT_GOAL_ADD_V99: "goals",
            INTENT_GOAL_LIST_V99: "goals",
            INTENT_GOAL_DONE_V99: "goals",
            INTENT_GOAL_NEXT_V99: "goals",
            INTENT_GOAL_AUTO_V99: "goals",
            INTENT_DISCOURSE_V100: "discourse",
            INTENT_WHY_REF_V100: "discourse",
        }
        topic = str(topic_map.get(str(parse.get("intent_id") or ""), "other"))
        topics = discourse_state.get("active_topics")
        topics_list = topics if isinstance(topics, list) else []
        new_topics = [topic] + [str(x) for x in topics_list if isinstance(x, str) and x and str(x) != topic]
        discourse_state["active_topics"] = list(new_topics[:5])

        # Open questions (only when we asked to clarify).
        if str(parse.get("intent_id") or "") == INTENT_WHY_REF_V100 and str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
            keys = ctx2.get("open_question_keys") if isinstance(ctx2.get("open_question_keys"), list) else []
            keys2 = [str(x) for x in keys if isinstance(x, str) and x]
            oq = {
                "question_id": _stable_hash_obj({"kind": "why_ref", "turn_id": str(ut.get("turn_id") or ""), "keys": list(keys2)}),
                "kind": "why_ref",
                "keys": list(keys2),
                "created_turn_id": str(ut.get("turn_id") or ""),
            }
            oqs = discourse_state.get("open_questions")
            oqs_list = oqs if isinstance(oqs, list) else []
            oqs_list.append(dict(oq))
            # keep last 5, deterministically ordered by question_id
            oqs_list2 = sorted([dict(x) for x in oqs_list if isinstance(x, dict)], key=lambda d: str(d.get("question_id") or ""))
            discourse_state["open_questions"] = list(oqs_list2[-5:])

        # Commitments: if evidence caused a belief revision/add, record minimal commitment link.
        if evidence_write_ids and belief_write_event_ids:
            commitments = discourse_state.get("commitments")
            c_list = commitments if isinstance(commitments, list) else []
            c_list.append(
                {
                    "kind": "belief_revision",
                    "cause_evidence_ids": sorted(set([str(x) for x in evidence_write_ids if isinstance(x, str) and x])),
                    "belief_write_event_ids": sorted(set([str(x) for x in belief_write_event_ids if isinstance(x, str) and x])),
                    "turn_id": str(at.get("turn_id") or ""),
                }
            )
            discourse_state["commitments"] = list(c_list[-10:])

        # Ref map: track available belief keys for "isso/that" references.
        bkeys = sorted([str(k) for k in belief_active_by_key.keys() if isinstance(k, str) and k])
        discourse_state["ref_map"] = {"isso": list(bkeys), "that": list(bkeys)}

        discourse_state_after = _deepcopy_json_obj(discourse_state)

        # Discourse event (inner sig-chain + outer jsonl chain).
        sel_candidate_id = str(discourse_selected.get("candidate_id") or "")
        cause_ids = {
            "cause_goal_ids": sorted(set(list(goal_read_ids) + list(goal_write_ids))),
            "cause_belief_ids": sorted(set(list(belief_read_ids))),
            "cause_evidence_ids": sorted(set(list(evidence_read_ids) + list(evidence_write_ids))),
        }
        de = DiscourseEventV100(
            conversation_id=str(conversation_id),
            turn_id=str(at.get("turn_id") or ""),
            turn_index=int(at.get("turn_index") or 0),
            discourse_state_before=dict(discourse_state_before),
            discourse_state_after=dict(discourse_state_after),
            candidates_topk=[dict(x) for x in discourse_candidates_topk if isinstance(x, dict)][:8],
            selected_candidate_id=str(sel_candidate_id),
            cause_ids=dict(cause_ids),
            created_step=int(at.get("created_step") or 0),
            prev_event_sig=str(prev_discourse_event_sig),
        ).to_dict()
        prev_discourse_hash = append_chained_jsonl_v96(
            discourse_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "DISCOURSE_EVENT", "payload": dict(de)},
            prev_hash=prev_discourse_hash,
        )
        discourse_events.append(dict(de))
        prev_discourse_event_sig = str(de.get("event_sig") or "")

        # Fragment USE events for selected candidate, plus deterministic promotion events.
        sel_frags = discourse_selected.get("fragment_ids") if isinstance(discourse_selected.get("fragment_ids"), list) else []
        sel_frag_ids = [str(x) for x in sel_frags if isinstance(x, str) and x]
        sel_score = float(discourse_selected.get("fluency_score") or 0.0)
        for fid in sorted(set(sel_frag_ids), key=str):
            fe = FragmentEventV100(
                conversation_id=str(conversation_id),
                event_kind="USE",
                fragment_id=str(fid),
                turn_id=str(at.get("turn_id") or ""),
                candidate_id=str(sel_candidate_id),
                fluency_score=float(sel_score),
                created_step=int(at.get("created_step") or 0),
                prev_event_sig=str(prev_fragment_event_sig),
            ).to_dict()
            prev_fragment_hash = append_chained_jsonl_v96(
                fragment_events_path,
                {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "FRAGMENT_EVENT", "payload": dict(fe)},
                prev_hash=prev_fragment_hash,
            )
            fragment_events.append(dict(fe))
            prev_fragment_event_sig = str(fe.get("event_sig") or "")

        # Promotion check (after USE).
        if sel_frag_ids:
            st_by_id = fold_fragment_library_v100(base_fragments=list(base_fragments), fragment_events=list(fragment_events))
            for fid in sorted(set(sel_frag_ids), key=str):
                st = st_by_id.get(str(fid))
                if not isinstance(st, dict):
                    continue
                if fragment_should_promote_v100(dict(st)):
                    fe2 = FragmentEventV100(
                        conversation_id=str(conversation_id),
                        event_kind="PROMOTE",
                        fragment_id=str(fid),
                        turn_id=str(at.get("turn_id") or ""),
                        candidate_id=str(sel_candidate_id),
                        fluency_score=float(sel_score),
                        created_step=int(at.get("created_step") or 0),
                        prev_event_sig=str(prev_fragment_event_sig),
                    ).to_dict()
                    prev_fragment_hash = append_chained_jsonl_v96(
                        fragment_events_path,
                        {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "FRAGMENT_EVENT", "payload": dict(fe2)},
                        prev_hash=prev_fragment_hash,
                    )
                    fragment_events.append(dict(fe2))
                    prev_fragment_event_sig = str(fe2.get("event_sig") or "")

        # Plan record (1 per user turn).
        notes = "max expected_success, tie-break min expected_cost, tie-break act_id"
        # Provenance: include learned rule ids used by this parse (if any).
        prov_learned: List[str] = []
        mrid = str(parse.get("matched_rule_id") or "")
        if mrid and mrid in set(learned_rules_active.keys()):
            prov_learned = [mrid]
        # TEACH writes a learned rule id (audit) on success.
        prov_learned_written: List[str] = []
        if str(parse.get("intent_id") or "") == INTENT_TEACH_V93 and bool(parse.get("teach_ok", False)):
            lrid = str(parse.get("learned_rule_id") or "")
            if lrid:
                prov_learned_written = [lrid]

        ranked_topk = []
        for rc in list(plan_ranked_candidates)[:8]:
            if not isinstance(rc, dict):
                continue
            act_id = str(rc.get("act_id") or "")
            es = _round6(rc.get("expected_success"))
            ec = _round6(rc.get("expected_cost"))
            score = float(es) / (float(ec) + 1e-9)
            ranked_topk.append(
                {
                    "act_id": str(act_id),
                    "expected_success": float(es),
                    "expected_cost": float(ec),
                    "score": _round6(score),
                    "reason": "score=expected_success/(expected_cost+eps)",
                }
            )
        ranked_topk.sort(key=lambda d: (-float(d.get("score", 0.0)), str(d.get("act_id") or "")))

        sel_frag_ids2 = discourse_selected.get("fragment_ids") if isinstance(discourse_selected.get("fragment_ids"), list) else []
        sel_frag_ids_canon = sorted(set([str(x) for x in sel_frag_ids2 if isinstance(x, str) and x]))
        discourse_plan = {
            "discourse_event_id": str(de.get("event_id") or "") if isinstance(locals().get("de"), dict) else "",
            "selected_candidate_id": str(discourse_selected.get("candidate_id") or ""),
            "selected_text_sha256": str(discourse_selected.get("text_sha256") or ""),
            "selected_fluency_score": _round6(discourse_selected.get("fluency_score")),
            "selected_fragment_ids": list(sel_frag_ids_canon),
            "candidates_topk": [
                {
                    "candidate_id": str(c.get("candidate_id") or ""),
                    "variant_id": str(c.get("variant_id") or ""),
                    "text_sha256": str(c.get("text_sha256") or ""),
                    "fluency_score": _round6(c.get("fluency_score")),
                    "fluency_metrics": dict(c.get("fluency_metrics") or {}) if isinstance(c.get("fluency_metrics"), dict) else {},
                    "fragment_ids": [str(x) for x in (c.get("fragment_ids") if isinstance(c.get("fragment_ids"), list) else []) if isinstance(x, str) and x],
                }
                for c in list(discourse_candidates_topk)[:8]
                if isinstance(c, dict)
            ],
        }

        plan_sem = {
            "schema_version": 100,
            "kind": "action_plan_v100",
            "conversation_id": str(conversation_id),
            "user_turn_id": str(ut.get("turn_id") or ""),
            "user_turn_index": int(ut.get("turn_index") or 0),
            "intent_id": str(parse.get("intent_id") or ""),
            "parse_sig": str(parse.get("parse_sig") or ""),
            "objective_kind": str(plan_objective_kind or objective_kind),
            "objective_id": str(plan_objective_id or chosen_objective_id),
            "ranked_candidates": list(plan_ranked_candidates),
            "attempted_actions": list(plan_attempted_actions),
            "chosen_action_id": str(chosen_action_id),
            "chosen_eval_id": str(chosen_eval_id),
            "chosen_ok": bool(chosen_ok),
            "notes": str(notes),
            "created_step": int(step),
            "discourse": dict(discourse_plan),
            "memory_read_ids": sorted(set([str(x) for x in list(memory_read_ids) if isinstance(x, str) and x])),
            "memory_write_event_ids": sorted(set([str(x) for x in list(memory_write_event_ids) if isinstance(x, str) and x])),
            "belief_read_keys": sorted(set([str(x) for x in list(belief_read_keys) if isinstance(x, str) and x])),
            "belief_read_ids": sorted(set([str(x) for x in list(belief_read_ids) if isinstance(x, str) and x])),
            "belief_write_event_ids": sorted(set([str(x) for x in list(belief_write_event_ids) if isinstance(x, str) and x])),
            "evidence_read_ids": sorted(set([str(x) for x in list(evidence_read_ids) if isinstance(x, str) and x])),
            "evidence_write_ids": sorted(set([str(x) for x in list(evidence_write_ids) if isinstance(x, str) and x])),
            "goal_read_ids": sorted(set([str(x) for x in list(goal_read_ids) if isinstance(x, str) and x])),
            "goal_write_ids": sorted(set([str(x) for x in list(goal_write_ids) if isinstance(x, str) and x])),
            "goal_tick_actions": [dict(x) for x in list(goal_tick_actions) if isinstance(x, dict)],
            "candidates_topk": list(ranked_topk),
            "selected_intent_id": str(chosen_action_id),
            "provenance": {
                "learned_rule_ids": list(sorted(set(prov_learned))),
                "learned_rule_ids_written": list(sorted(set(prov_learned_written))),
                "fragment_ids_used": list(sel_frag_ids_canon),
                "cause_evidence_ids": sorted(set([str(x) for x in list(cause_evidence_ids) if isinstance(x, str) and x])),
                "belief_read_keys": sorted(set([str(x) for x in list(belief_read_keys) if isinstance(x, str) and x])),
                "belief_read_ids": sorted(set([str(x) for x in list(belief_read_ids) if isinstance(x, str) and x])),
                "belief_write_event_ids": sorted(set([str(x) for x in list(belief_write_event_ids) if isinstance(x, str) and x])),
                "memory_read_ids": sorted(set([str(x) for x in list(memory_read_ids) if isinstance(x, str) and x])),
                "memory_write_event_ids": sorted(set([str(x) for x in list(memory_write_event_ids) if isinstance(x, str) and x])),
                "evidence_read_ids": sorted(set([str(x) for x in list(evidence_read_ids) if isinstance(x, str) and x])),
                "evidence_write_ids": sorted(set([str(x) for x in list(evidence_write_ids) if isinstance(x, str) and x])),
                "goal_ids": sorted(
                    set(
                        [str(x) for x in list(goal_read_ids) if isinstance(x, str) and x]
                        + [str(x) for x in list(goal_write_ids) if isinstance(x, str) and x]
                    )
                ),
            },
        }
        plan_sig = _stable_hash_obj(plan_sem)
        plan_obj = dict(plan_sem, plan_sig=str(plan_sig), plan_id=str(action_plan_id_v96(plan_sig)))
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
            goals_active_by_id, _gdet = fold_goal_ledger_v99(list(goal_events))
            goals_active_ids = sorted([str(k) for k in goals_active_by_id.keys() if str(k)], key=str)
            frag_states = fold_fragment_library_v100(base_fragments=list(base_fragments), fragment_events=list(fragment_events))
            frag_promoted = sorted([fid for fid, st in frag_states.items() if isinstance(st, dict) and str(st.get("promotion_state") or "") == "promoted"], key=str)
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
                    "goal_active_ids": list(goals_active_ids),
                    "goal_active_count": int(len(goals_active_ids)),
                    "discourse_state": dict(discourse_state),
                    "fragment_promoted_ids": list(frag_promoted),
                    "fragment_promoted_count": int(len(frag_promoted)),
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

    # Goal ledger snapshot (derived; write-once WORM).
    if os.path.exists(goal_snapshot_path):
        _fail(f"goal_snapshot_exists:{goal_snapshot_path}")
    goal_snapshot = goal_ledger_snapshot_v99(list(goal_events))
    tmpgs = goal_snapshot_path + ".tmp"
    with open(tmpgs, "w", encoding="utf-8") as f:
        f.write(json.dumps(goal_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpgs, goal_snapshot_path)

    # Fragment library snapshot (derived; write-once WORM).
    if os.path.exists(fragment_snapshot_path):
        _fail(f"fragment_snapshot_exists:{fragment_snapshot_path}")
    frag_snapshot = fragment_library_snapshot_v100(base_fragments=list(base_fragments), fragment_events=list(fragment_events))
    tmpfs = fragment_snapshot_path + ".tmp"
    with open(tmpfs, "w", encoding="utf-8") as f:
        f.write(json.dumps(frag_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpfs, fragment_snapshot_path)

    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v96(turns_path)),
        "parses_chain_ok": bool(verify_chained_jsonl_v96(parses_path)),
        "learned_chain_ok": bool(verify_chained_jsonl_v96(learned_path)) if os.path.exists(learned_path) else True,
        "plans_chain_ok": bool(verify_chained_jsonl_v96(plans_path)),
        "memory_chain_ok": bool(verify_chained_jsonl_v96(memory_path)),
        "belief_chain_ok": bool(verify_chained_jsonl_v96(belief_path)),
        "evidence_chain_ok": bool(verify_chained_jsonl_v96(evidence_path)),
        "goal_chain_ok": bool(verify_chained_jsonl_v96(goal_path)),
        "discourse_chain_ok": bool(verify_chained_jsonl_v96(discourse_path)),
        "fragment_chain_ok": bool(verify_chained_jsonl_v96(fragment_events_path)),
        "states_chain_ok": bool(verify_chained_jsonl_v96(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v96(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v96(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v96(transcript_path)),
    }
    ok_chain, chain_reason, chain_details = verify_conversation_chain_v100(
        turns=list(turns),
        states=list(states),
        parse_events=list(parse_events),
        trials=list(trials),
        learned_rule_events=list(learned_rule_events),
        action_plans=list(action_plans),
        memory_events=list(memory_events),
        belief_events=list(belief_events),
        evidence_events=list(evidence_events),
        goal_events=list(goal_events),
        goal_snapshot=dict(goal_snapshot),
        discourse_events=list(discourse_events),
        fragment_events=list(fragment_events),
        tail_k=6,
        repo_root=str(repo_root),
    )

    transcript_hash = compute_transcript_hash_v96(turns)
    state_chain_hash = compute_state_chain_hash_v96(states)
    parse_chain_hash = compute_parse_chain_hash_v96(parse_events)
    learned_chain_hash = compute_learned_chain_hash_v96(learned_rule_events)
    plan_chain_hash = compute_plan_chain_hash_v96(action_plans)
    memory_chain_hash = compute_memory_chain_hash_v96(memory_events)
    belief_chain_hash = compute_belief_chain_hash_v96(belief_events)
    evidence_chain_hash = compute_evidence_chain_hash_v98(evidence_events)
    goal_chain_hash = compute_goal_chain_hash_v99(list(goal_events))
    discourse_chain_hash = compute_discourse_chain_hash_v100(list(discourse_events))
    fragment_chain_hash = compute_fragment_chain_hash_v100(list(fragment_events))
    system_spec_sha256 = sha256_file(system_spec_path)

    verify_obj = {
        "ok": bool(all(chains.values())) and bool(ok_chain),
        "chains": dict(chains),
        "chain_invariants": {"ok": bool(ok_chain), "reason": str(chain_reason), "details": dict(chain_details)},
        "store_hash": str(store_hash),
        "system_spec_sha256": str(system_spec_sha256),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
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
        "system_spec_sha256": str(system_spec_sha256),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
        "verify_ok": bool(verify_obj.get("ok", False)),
        "sha256": {
            "store_jsonl": str(sha256_file(store_path)),
            "intent_grammar_snapshot_json": str(sha256_file(grammar_snapshot_path)),
            "system_spec_snapshot_json": str(system_spec_sha256),
            "conversation_turns_jsonl": str(sha256_file(turns_path)),
            "intent_parses_jsonl": str(sha256_file(parses_path)),
            "learned_intent_rules_jsonl": str(sha256_file(learned_path)) if os.path.exists(learned_path) else "",
            "action_plans_jsonl": str(sha256_file(plans_path)),
            "memory_events_jsonl": str(sha256_file(memory_path)),
            "belief_events_jsonl": str(sha256_file(belief_path)),
            "evidence_events_jsonl": str(sha256_file(evidence_path)),
            "goal_events_jsonl": str(sha256_file(goal_path)),
            "goal_ledger_snapshot_json": str(sha256_file(goal_snapshot_path)),
            "discourse_events_jsonl": str(sha256_file(discourse_path)),
            "fragment_events_jsonl": str(sha256_file(fragment_events_path)),
            "fragment_library_snapshot_json": str(sha256_file(fragment_snapshot_path)),
            "conversation_states_jsonl": str(sha256_file(states_path)) if os.path.exists(states_path) else "",
            "dialogue_trials_jsonl": str(sha256_file(trials_path)),
            "objective_evals_jsonl": str(sha256_file(evals_path)),
            "transcript_jsonl": str(sha256_file(transcript_path)),
            "verify_chain_v100_json": str(sha256_file(verify_path)),
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
        "system_spec_sha256": str(system_spec_sha256),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
        "ledger_hash": str(ledger_hash),
        "turns_total": int(len(turns)),
        "user_turns_total": int(user_turns_total),
        "states_total": int(len(states)),
        "plans_total": int(len(action_plans)),
        "memory_events_total": int(len(memory_events)),
        "belief_events_total": int(len(belief_events)),
        "evidence_events_total": int(len(evidence_events)),
        "goal_events_total": int(len(goal_events)),
        "discourse_events_total": int(len(discourse_events)),
        "fragment_events_total": int(len(fragment_events)),
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
        "system_spec_sha256": str(system_spec_sha256),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "learned_chain_hash": str(learned_chain_hash),
        "plan_chain_hash": str(plan_chain_hash),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
        "ledger_hash": str(ledger_hash),
        "summary_sha256": str(summary_sha256),
        "paths": {
            "store_jsonl": str(store_path),
            "grammar_snapshot_json": str(grammar_snapshot_path),
            "system_spec_snapshot_json": str(system_spec_path),
            "turns_jsonl": str(turns_path),
            "parses_jsonl": str(parses_path),
            "learned_rules_jsonl": str(learned_path),
            "plans_jsonl": str(plans_path),
            "memory_events_jsonl": str(memory_path),
            "belief_events_jsonl": str(belief_path),
            "evidence_events_jsonl": str(evidence_path),
            "goal_events_jsonl": str(goal_path),
            "goal_ledger_snapshot_json": str(goal_snapshot_path),
            "discourse_events_jsonl": str(discourse_path),
            "fragment_events_jsonl": str(fragment_events_path),
            "fragment_library_snapshot_json": str(fragment_snapshot_path),
            "states_jsonl": str(states_path),
            "trials_jsonl": str(trials_path),
            "evals_jsonl": str(evals_path),
            "transcript_jsonl": str(transcript_path),
            "verify_json": str(verify_path),
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
        },
    }
