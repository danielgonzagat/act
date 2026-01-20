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
from .conversation_v100 import render_explain_text_v100
from .conversation_v101 import (
    render_bindings_text_v101,
    render_explain_binding_text_v101,
    render_trace_ref_text_v101,
)
from .conversation_v103 import (
    render_concepts_text_v103,
    render_explain_concept_text_v103,
    render_trace_concepts_text_v103,
)
from .conversation_v108 import verify_conversation_chain_v108
from .plan_engine_v104 import (
    PLAN_KIND_ASK_CLARIFY_V104,
    PLAN_KIND_COMPARE_OPTIONS_V104,
    build_plan_candidates_v104,
)
from .plan_ledger_v104 import PlanEventV104, compute_plan_chain_hash_v104
from .plan_registry_v104 import (
    fold_plan_ledger_v104,
    lookup_plan_event_v104,
    plan_registry_snapshot_v104,
    render_explain_plan_text_v104,
    render_plans_text_v104,
    render_trace_plans_text_v104,
)
from .agency_ledger_v105 import (
    AGENCY_KIND_ASK_CLARIFY_V105,
    AGENCY_KIND_CLOSE_GOAL_V105,
    AGENCY_KIND_EXECUTE_STEP_V105,
    AGENCY_KIND_IDLE_V105,
    AGENCY_KIND_PROPOSE_OPTIONS_V105,
    AgencyEventV105,
    agency_registry_snapshot_v105,
    compute_agency_chain_hash_v105,
    lookup_agency_event_v105,
    render_agency_text_v105,
    render_explain_agency_text_v105,
    render_trace_agency_text_v105,
)
from .dialogue_engine_v106 import COHERENCE_THRESH_V106, compute_metrics_v106, decide_repair_action_v106, repair_text_v106, select_candidate_v106
from .dialogue_ledger_v106 import (
    DialogueEventV106,
    compute_dialogue_chain_hash_v106,
    dialogue_registry_snapshot_v106,
    render_dialogue_text_v106,
    render_explain_dialogue_text_v106,
    render_trace_dialogue_text_v106,
)
from .pragmatics_engine_v107 import (
    PRAGMATICS_THRESH_V107,
    classify_user_intent_act_v107,
    compute_pragmatics_metrics_v107,
    decide_regime_next_v107,
    decide_repair_action_v107,
    infer_assistant_intent_act_v107,
    is_pragmatics_progress_blocked_v107,
)
from .pragmatics_ledger_v107 import (
    PragmaticsEventV107,
    compute_pragmatics_chain_hash_v107,
    pragmatics_registry_snapshot_v107,
    render_explain_pragmatics_text_v107,
    render_pragmatics_text_v107,
    render_trace_pragmatics_text_v107,
)
from .flow_engine_v108 import compute_flow_metrics_v108, select_candidate_dsc_v108
from .flow_ledger_v108 import (
    FlowEventV108,
    compute_flow_chain_hash_v108,
    flow_registry_snapshot_v108,
    render_explain_flow_text_v108,
    render_flow_text_v108,
    render_trace_flow_text_v108,
)
from .intent_grammar_v108 import (
    INTENT_EXPLAIN_FLOW_V108,
    INTENT_FLOW_V108,
    INTENT_TRACE_FLOW_V108,
    is_explain_flow_command_v108,
    is_flow_command_v108,
    is_trace_flow_command_v108,
    parse_explain_flow_command_v108,
    parse_flow_command_v108,
    parse_trace_flow_command_v108,
)
from .concept_engine_v103 import domain_distance_v103, example_sig_v103, induce_rule_v103, match_csg_v103, normalize_example_text_v103
from .concept_ledger_v103 import ConceptEventV103, compute_concept_chain_hash_v103
from .concept_model_v103 import make_csg_rule_v103
from .concept_registry_v103 import concept_library_snapshot_v103, fold_concept_ledger_v103, lookup_concept_id_by_name_v103
from .binding_ledger_v101 import (
    binding_chain_hash_v101,
    binding_snapshot_v101,
    choose_prune_candidate_v101,
    make_binding_ambiguous_event_v101,
    make_binding_create_event_v101,
    make_binding_miss_event_v101,
    make_binding_prune_event_v101,
    make_binding_resolve_event_v101,
)
from .bindings_v101 import infer_kind_hint_v101, resolve_reference_v101
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
from .intent_grammar_v103 import (
    INTENT_CONCEPTS_V103,
    INTENT_EXPLAIN_CONCEPT_V103,
    INTENT_TEACH_CONCEPT_V103,
    INTENT_TRACE_CONCEPTS_V103,
    is_concepts_command_v103,
    is_explain_concept_command_v103,
    is_teach_concept_command_v103,
    is_trace_concepts_command_v103,
    parse_concepts_command_v103,
    parse_explain_concept_command_v103,
    parse_teach_concept_command_v103,
    parse_trace_concepts_command_v103,
)
from .intent_grammar_v104 import (
    INTENT_EXPLAIN_PLAN_V104,
    INTENT_PLANS_V104,
    INTENT_TRACE_PLANS_V104,
    is_explain_plan_command_v104,
    is_plans_command_v104,
    is_trace_plans_command_v104,
    parse_explain_plan_command_v104,
    parse_plans_command_v104,
    parse_trace_plans_command_v104,
)
from .intent_grammar_v105 import (
    INTENT_AGENCY_V105,
    INTENT_EXPLAIN_AGENCY_V105,
    INTENT_TRACE_AGENCY_V105,
    is_agency_command_v105,
    is_explain_agency_command_v105,
    is_trace_agency_command_v105,
    parse_agency_command_v105,
    parse_explain_agency_command_v105,
    parse_trace_agency_command_v105,
)
from .intent_grammar_v106 import (
    INTENT_DIALOGUE_V106,
    INTENT_EXPLAIN_DIALOGUE_V106,
    INTENT_TRACE_DIALOGUE_V106,
    is_dialogue_command_v106,
    is_explain_dialogue_command_v106,
    is_trace_dialogue_command_v106,
    parse_dialogue_command_v106,
    parse_explain_dialogue_command_v106,
    parse_trace_dialogue_command_v106,
)
from .intent_grammar_v107 import (
    INTENT_EXPLAIN_PRAGMATICS_V107,
    INTENT_PRAGMATICS_V107,
    INTENT_TRACE_PRAGMATICS_V107,
    is_explain_pragmatics_command_v107,
    is_pragmatics_command_v107,
    is_trace_pragmatics_command_v107,
    parse_explain_pragmatics_command_v107,
    parse_pragmatics_command_v107,
    parse_trace_pragmatics_command_v107,
)
from .intent_grammar_v100 import (
    INTENT_DISCOURSE_V100,
    INTENT_WHY_REF_V100,
    is_discourse_command_v100,
    is_why_ref_command_v100,
    parse_discourse_command_v100,
    parse_why_ref_command_v100,
)
from .intent_grammar_v101 import (
    INTENT_BINDINGS_LIST_V101,
    INTENT_EXPLAIN_BINDING_V101,
    INTENT_GOAL_PRIORITY_HIGH_REF_V101,
    INTENT_PLAN_CREATE_V101,
    INTENT_PLAN_SHORTEN_REF_V101,
    INTENT_TRACE_REF_V101,
    is_bindings_command_v101,
    is_explain_binding_command_v101,
    is_plan_create_3_command_v101,
    is_priority_high_ref_command_v101,
    is_shorten_ref_command_v101,
    is_trace_ref_command_v101,
    parse_bindings_command_v101,
    parse_explain_binding_command_v101,
    parse_plan_create_3_command_v101,
    parse_priority_high_ref_command_v101,
    parse_shorten_ref_command_v101,
    parse_trace_ref_command_v101,
)
from .intent_grammar_v102 import (
    INTENT_EXPLAIN_STYLE_V102,
    INTENT_STYLE_PROFILE_V102,
    INTENT_TEMPLATES_V102,
    INTENT_TRACE_STYLE_V102,
    is_explain_style_command_v102,
    is_style_profile_command_v102,
    is_templates_command_v102,
    is_trace_style_command_v102,
    parse_explain_style_command_v102,
    parse_style_profile_command_v102,
    parse_templates_command_v102,
    parse_trace_style_command_v102,
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
from .discourse_templates_v102 import base_templates_v102, render_templates_list_text_v102
from .style_ledger_v102 import StyleEventV102, compute_style_chain_hash_v102, fold_template_stats_v102, template_library_snapshot_v102
from .style_profile_v102 import StyleProfileV102, coerce_style_profile_v102, default_style_profile_v102, derive_style_profile_update_v102, render_style_profile_text_v102
from .style_selector_v102 import build_and_select_style_candidates_v102


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

    # V104: planner override text (used for safe option/clarify responses on INTENT_UNKNOWN).
    if isinstance(ctx, dict) and str(ctx.get("planned_text") or ""):
        text = str(ctx.get("planned_text") or "")
        return text, {"text": text}, "concept_v90_emit_text_v0"

    if intent_id == INTENT_PLANS_V104:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("plans_text") or ""):
            text = str(ctx.get("plans_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_EXPLAIN_PLAN_V104:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("explain_plan_text") or ""):
            text = str(ctx.get("explain_plan_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_TRACE_PLANS_V104:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("trace_plans_text") or ""):
            text = str(ctx.get("trace_plans_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

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

    if intent_id == INTENT_STYLE_PROFILE_V102:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("style_profile_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_TEMPLATES_V102:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("templates_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_EXPLAIN_STYLE_V102:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("explain_style_text") or ""):
            text = str(ctx.get("explain_style_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_TRACE_STYLE_V102:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("trace_style_text") or ""):
            text = str(ctx.get("trace_style_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_TEACH_CONCEPT_V103:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("teach_concept_ack_text") or ""):
            text = str(ctx.get("teach_concept_ack_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_CONCEPTS_V103:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("concepts_text") or ""):
            text = str(ctx.get("concepts_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_EXPLAIN_CONCEPT_V103:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("explain_concept_text") or ""):
            text = str(ctx.get("explain_concept_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_TRACE_CONCEPTS_V103:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("trace_concepts_text") or ""):
            text = str(ctx.get("trace_concepts_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or str(parse.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_BINDINGS_LIST_V101:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("bindings_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_EXPLAIN_BINDING_V101:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("binding_explain_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_TRACE_REF_V101:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("trace_ref_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id in {INTENT_PLAN_CREATE_V101, INTENT_PLAN_SHORTEN_REF_V101}:
        if bool(parse.get("parse_ok", False)):
            text = str(ctx.get("plan_text") or "")
            return text, {"text": text}, "concept_v90_emit_text_v0"
        msg = str(ctx.get("msg") or "")
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"

    if intent_id == INTENT_GOAL_PRIORITY_HIGH_REF_V101:
        if bool(parse.get("parse_ok", False)) and str(ctx.get("msg") or ""):
            text = str(ctx.get("msg") or "")
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

    # V102: explicit style ack (meta-instruction processed without mutating vars).
    if str(ctx.get("style_ack_text") or ""):
        text = str(ctx.get("style_ack_text") or "")
        return text, {"text": text}, "concept_v90_emit_text_v0"

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
        ctext = str(ctx.get("confirm_text") or ctx.get("msg") or "")
        if ctext:
            return str(ctext), {"text": str(ctext)}, "concept_v90_emit_text_v0"
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
        missing_ref = str(ctx.get("missing_ref") or "")
        if missing_ref:
            qref = f"A que você se refere com {missing_ref}?"
            return qref, {"text": qref}, "concept_v90_emit_text_v0"
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


def run_conversation_v108(
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
    binding_events_path = os.path.join(str(out_dir), "binding_events.jsonl")
    binding_snapshot_path = os.path.join(str(out_dir), "binding_snapshot.json")
    states_path = os.path.join(str(out_dir), "conversation_states.jsonl")
    trials_path = os.path.join(str(out_dir), "dialogue_trials.jsonl")
    evals_path = os.path.join(str(out_dir), "objective_evals.jsonl")
    transcript_path = os.path.join(str(out_dir), "transcript.jsonl")
    style_path = os.path.join(str(out_dir), "style_events.jsonl")
    template_snapshot_path = os.path.join(str(out_dir), "template_library_snapshot_v102.json")
    concept_events_path = os.path.join(str(out_dir), "concept_events.jsonl")
    concept_snapshot_path = os.path.join(str(out_dir), "concept_library_snapshot_v103.json")
    plan_events_path = os.path.join(str(out_dir), "plan_events.jsonl")
    plan_snapshot_path = os.path.join(str(out_dir), "plan_registry_snapshot_v104.json")
    agency_events_path = os.path.join(str(out_dir), "agency_events.jsonl")
    agency_snapshot_path = os.path.join(str(out_dir), "agency_registry_snapshot_v105.json")
    dialogue_events_path = os.path.join(str(out_dir), "dialogue_events.jsonl")
    dialogue_snapshot_path = os.path.join(str(out_dir), "dialogue_registry_snapshot_v106.json")
    pragmatics_events_path = os.path.join(str(out_dir), "pragmatics_events.jsonl")
    pragmatics_snapshot_path = os.path.join(str(out_dir), "pragmatics_registry_snapshot_v107.json")
    flow_events_path = os.path.join(str(out_dir), "flow_events.jsonl")
    flow_snapshot_path = os.path.join(str(out_dir), "flow_registry_snapshot_v108.json")
    verify_path = os.path.join(str(out_dir), "verify_chain_v108.json")
    manifest_path = os.path.join(str(out_dir), "freeze_manifest_v108.json")
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
    with open(binding_events_path, "x", encoding="utf-8") as _f:
        pass
    with open(style_path, "x", encoding="utf-8") as _f:
        pass
    with open(concept_events_path, "x", encoding="utf-8") as _f:
        pass
    with open(plan_events_path, "x", encoding="utf-8") as _f:
        pass
    with open(agency_events_path, "x", encoding="utf-8") as _f:
        pass
    with open(dialogue_events_path, "x", encoding="utf-8") as _f:
        pass
    with open(pragmatics_events_path, "x", encoding="utf-8") as _f:
        pass
    with open(flow_events_path, "x", encoding="utf-8") as _f:
        pass

    engine = EngineV80(store, seed=int(seed))

    conv_body = {"turns": list(user_turn_texts), "grammar_hash": str(grammar_hash_v92(rules))}
    conversation_id = f"conv_v108_{sha256_hex(canonical_json_dumps(conv_body).encode('utf-8'))}"

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
    style_events: List[Dict[str, Any]] = []
    concept_events: List[Dict[str, Any]] = []
    plan_events: List[Dict[str, Any]] = []
    agency_events: List[Dict[str, Any]] = []
    dialogue_events: List[Dict[str, Any]] = []
    pragmatics_events: List[Dict[str, Any]] = []
    flow_events: List[Dict[str, Any]] = []

    templates_v102 = base_templates_v102()
    style_profile = default_style_profile_v102()
    recent_style_template_ids: List[str] = []

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
    prev_agency_hash: Optional[str] = None
    prev_dialogue_hash: Optional[str] = None
    prev_pragmatics_hash: Optional[str] = None
    prev_flow_hash: Optional[str] = None

    support_events: List[Dict[str, Any]] = []
    learned_rules_active: Dict[str, IntentRuleV93] = {}

    memory_items_by_id: Dict[str, Dict[str, Any]] = {}
    memory_active_ids: Dict[str, bool] = {}

    belief_items_by_id: Dict[str, Dict[str, Any]] = {}
    belief_active_by_key: Dict[str, str] = {}

    prev_goal_event_sig = ""
    prev_discourse_event_sig = ""
    prev_fragment_event_sig = ""
    prev_style_event_sig = ""
    prev_concept_event_sig = ""
    prev_plan_event_sig = ""
    prev_agency_event_sig = ""
    prev_dialogue_event_sig = ""
    prev_pragmatics_event_sig = ""
    prev_flow_event_sig = ""
    prev_binding_event_sig = ""

    # V108 flow memory (explicit, deterministic; updated by fold over flow events).
    flow_memory_v108: Dict[str, Any] = {
        "topic_stack": [],
        "pending_questions": [],
        "commitments": [],
        "episodic_memory": [],
        "recent_phrases_hashes": [],
        "recent_discourse_acts": [],
        "verbosity_mode_v108": "",
        "tone_mode_v108": "",
    }
    prev_binding_hash: Optional[str] = None
    prev_style_hash: Optional[str] = None
    prev_concept_hash: Optional[str] = None
    prev_plan_hash: Optional[str] = None
    binding_events: List[Dict[str, Any]] = []
    binding_active_by_id: Dict[str, Dict[str, Any]] = {}
    binding_ts_logical = 0

    active_plan_state_v104: Dict[str, Any] = {}

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

    # V107 pragmatics state (MED/MDE minimal; derived by replay of pragmatics_events).
    pragmatics_regime_v107 = "GREETING"
    pragmatics_pending_questions_active: List[str] = []
    pragmatics_commitments_active: List[str] = []
    pragmatics_topic_stack_v107: List[str] = []
    pragmatics_event_index_v107 = 0

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

    def _active_bindings_list() -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for bid in sorted(binding_active_by_id.keys(), key=str):
            b = binding_active_by_id.get(bid)
            if isinstance(b, dict):
                items.append(dict(b))
        items.sort(key=lambda b: (-int(b.get("last_used_turn_index") or 0), str(b.get("binding_kind") or ""), str(b.get("binding_id") or "")))
        return items

    def _append_binding_event(ev: Dict[str, Any]) -> None:
        nonlocal prev_binding_hash, prev_binding_event_sig, binding_ts_logical
        ev2 = dict(ev)
        # ts_logical is part of the signed body; must be set before event_sig is computed.
        if int(ev2.get("ts_logical", -1)) != int(binding_ts_logical):
            _fail(f"binding_ts_logical_mismatch:want={binding_ts_logical} got={ev2.get('ts_logical')}")
        # Append to WORM chain on disk.
        prev_binding_hash = append_chained_jsonl_v96(binding_events_path, dict(ev2), prev_hash=prev_binding_hash)
        # Keep in-memory list without outer chain fields.
        binding_events.append({k: v for k, v in dict(ev2).items() if k not in {"prev_hash", "entry_hash"}})
        prev_binding_event_sig = str(ev2.get("event_sig") or prev_binding_event_sig)
        binding_ts_logical += 1
        # Apply fold incrementally to binding_active_by_id.
        et = str(ev2.get("type") or "")
        bid = str(ev2.get("binding_id") or "")
        if et == "BIND_CREATE" and bid:
            bval = ev2.get("binding_value") if isinstance(ev2.get("binding_value"), dict) else {}
            binding_active_by_id[bid] = {
                "binding_id": str(bid),
                "binding_kind": str(ev2.get("binding_kind") or ""),
                "value": dict(bval),
                "value_hash": str(ev2.get("binding_value_hash") or ""),
                "value_preview": str(ev2.get("binding_preview") or ""),
                "created_turn_index": int((ev2.get("evidence") or {}).get("created_turn_index") or 0),
                "last_used_turn_index": int((ev2.get("evidence") or {}).get("last_used_turn_index") or (ev2.get("evidence") or {}).get("created_turn_index") or 0),
                "use_count": int((ev2.get("evidence") or {}).get("use_count") or 0),
                "provenance": dict(ev2.get("provenance") or {}) if isinstance(ev2.get("provenance"), dict) else {},
            }
        elif et == "BIND_RESOLVE" and bid and bid in binding_active_by_id:
            ev_e = ev2.get("evidence") if isinstance(ev2.get("evidence"), dict) else {}
            if "last_used_turn_index" in ev_e:
                binding_active_by_id[bid]["last_used_turn_index"] = int(ev_e.get("last_used_turn_index") or binding_active_by_id[bid].get("last_used_turn_index") or 0)
            if "use_count" in ev_e:
                binding_active_by_id[bid]["use_count"] = int(ev_e.get("use_count") or binding_active_by_id[bid].get("use_count") or 0)
        elif et == "BIND_PRUNE" and bid:
            binding_active_by_id.pop(bid, None)

    def _append_concept_event(ev: Dict[str, Any]) -> None:
        nonlocal prev_concept_hash, prev_concept_event_sig
        ev2 = dict(ev)
        # Inner chain (prev_event_sig/event_sig) is verified by V103 verifier.
        if str(ev2.get("prev_event_sig") or "") != str(prev_concept_event_sig):
            _fail(
                "concept_prev_event_sig_mismatch:want={w} got={g}".format(
                    w=str(prev_concept_event_sig), g=str(ev2.get("prev_event_sig") or "")
                )
            )
        prev_concept_hash = append_chained_jsonl_v96(concept_events_path, dict(ev2), prev_hash=prev_concept_hash)
        concept_events.append({k: v for k, v in dict(ev2).items() if k not in {"prev_hash", "entry_hash"}})
        prev_concept_event_sig = str(ev2.get("event_sig") or prev_concept_event_sig)

    def _append_plan_event(ev: Dict[str, Any]) -> None:
        nonlocal prev_plan_hash, prev_plan_event_sig
        ev2 = dict(ev)
        # Inner chain (prev_event_sig/event_sig) is verified by V104 verifier.
        if str(ev2.get("prev_event_sig") or "") != str(prev_plan_event_sig):
            _fail(
                "plan_prev_event_sig_mismatch:want={w} got={g}".format(
                    w=str(prev_plan_event_sig), g=str(ev2.get("prev_event_sig") or "")
                )
            )
        prev_plan_hash = append_chained_jsonl_v96(plan_events_path, dict(ev2), prev_hash=prev_plan_hash)
        plan_events.append({k: v for k, v in dict(ev2).items() if k not in {"prev_hash", "entry_hash"}})
        prev_plan_event_sig = str(ev2.get("event_sig") or prev_plan_event_sig)

    def _append_agency_event(ev: Dict[str, Any]) -> None:
        nonlocal prev_agency_hash, prev_agency_event_sig
        ev2 = dict(ev)
        # Inner chain (prev_event_sig/event_sig) is verified by V105 verifier.
        if str(ev2.get("prev_event_sig") or "") != str(prev_agency_event_sig):
            _fail(
                "agency_prev_event_sig_mismatch:want={w} got={g}".format(
                    w=str(prev_agency_event_sig), g=str(ev2.get("prev_event_sig") or "")
                )
            )
        prev_agency_hash = append_chained_jsonl_v96(agency_events_path, dict(ev2), prev_hash=prev_agency_hash)
        agency_events.append({k: v for k, v in dict(ev2).items() if k not in {"prev_hash", "entry_hash"}})
        prev_agency_event_sig = str(ev2.get("event_sig") or prev_agency_event_sig)

    def _append_dialogue_event(ev: Dict[str, Any]) -> None:
        nonlocal prev_dialogue_hash, prev_dialogue_event_sig
        ev2 = dict(ev)
        # Inner chain (prev_event_sig/event_sig) is verified by V106 verifier.
        if str(ev2.get("prev_event_sig") or "") != str(prev_dialogue_event_sig):
            _fail(
                "dialogue_prev_event_sig_mismatch:want={w} got={g}".format(
                    w=str(prev_dialogue_event_sig), g=str(ev2.get("prev_event_sig") or "")
                )
            )
        prev_dialogue_hash = append_chained_jsonl_v96(dialogue_events_path, dict(ev2), prev_hash=prev_dialogue_hash)
        dialogue_events.append({k: v for k, v in dict(ev2).items() if k not in {"prev_hash", "entry_hash"}})
        prev_dialogue_event_sig = str(ev2.get("event_sig") or prev_dialogue_event_sig)

    def _append_pragmatics_event(ev: Dict[str, Any]) -> None:
        nonlocal prev_pragmatics_hash, prev_pragmatics_event_sig
        ev2 = dict(ev)
        # Inner chain (prev_event_sig/event_sig) is verified by V107 verifier.
        if str(ev2.get("prev_event_sig") or "") != str(prev_pragmatics_event_sig):
            _fail(
                "pragmatics_prev_event_sig_mismatch:want={w} got={g}".format(
                    w=str(prev_pragmatics_event_sig), g=str(ev2.get("prev_event_sig") or "")
                )
            )
        prev_pragmatics_hash = append_chained_jsonl_v96(pragmatics_events_path, dict(ev2), prev_hash=prev_pragmatics_hash)
        pragmatics_events.append({k: v for k, v in dict(ev2).items() if k not in {"prev_hash", "entry_hash"}})
        prev_pragmatics_event_sig = str(ev2.get("event_sig") or prev_pragmatics_event_sig)

    def _append_flow_event(ev: Dict[str, Any]) -> None:
        nonlocal prev_flow_hash, prev_flow_event_sig
        ev2 = dict(ev)
        # Inner chain (prev_event_sig/event_sig) is verified by V108 verifier.
        if str(ev2.get("prev_event_sig") or "") != str(prev_flow_event_sig):
            _fail(
                "flow_prev_event_sig_mismatch:want={w} got={g}".format(
                    w=str(prev_flow_event_sig), g=str(ev2.get("prev_event_sig") or "")
                )
            )
        prev_flow_hash = append_chained_jsonl_v96(flow_events_path, dict(ev2), prev_hash=prev_flow_hash)
        flow_events.append({k: v for k, v in dict(ev2).items() if k not in {"prev_hash", "entry_hash"}})
        prev_flow_event_sig = str(ev2.get("event_sig") or prev_flow_event_sig)

    def _ensure_binding_create(
        *,
        turn_id: str,
        created_turn_index: int,
        binding_kind: str,
        value: Dict[str, Any],
        value_preview: str,
        provenance: Dict[str, Any],
    ) -> str:
        # Deterministic binding_id derived from kind + value_hash.
        ev_create = make_binding_create_event_v101(
            conversation_id=str(conversation_id),
            ts_logical=int(binding_ts_logical),
            turn_id=str(turn_id),
            binding_kind=str(binding_kind),
            value=dict(value),
            value_preview=str(value_preview),
            provenance=dict(provenance),
            created_turn_index=int(created_turn_index),
            created_step=int(step),
            prev_event_sig=str(prev_binding_event_sig),
        )
        bid = str(ev_create.get("binding_id") or "")
        if bid and bid in binding_active_by_id:
            return bid
        _append_binding_event(ev_create)
        # Deterministic prune to enforce bounded growth.
        max_total = 12
        if len(binding_active_by_id) > int(max_total):
            prune = choose_prune_candidate_v101(list(binding_active_by_id.values()))
            if isinstance(prune, dict):
                pid = str(prune.get("binding_id") or "")
                if pid and pid in binding_active_by_id and pid != bid:
                    ev_prune = make_binding_prune_event_v101(
                        conversation_id=str(conversation_id),
                        ts_logical=int(binding_ts_logical),
                        turn_id=str(turn_id),
                        binding_id=str(pid),
                        binding_kind=str(prune.get("binding_kind") or ""),
                        binding_value_hash=str(prune.get("value_hash") or ""),
                        reason="max_bindings_total",
                        created_step=int(step),
                        prev_event_sig=str(prev_binding_event_sig),
                    )
                    _append_binding_event(ev_prune)
        return str(bid)

    def _binding_get(binding_id: str) -> Optional[Dict[str, Any]]:
        b = binding_active_by_id.get(str(binding_id) or "")
        return dict(b) if isinstance(b, dict) else None

    def _binding_use(*, turn_id: str, user_turn_index: int, resolution: Dict[str, Any], chosen_binding_id: str) -> None:
        b = _binding_get(str(chosen_binding_id))
        if not isinstance(b, dict):
            return
        use_count = int(b.get("use_count") or 0) + 1
        bkind = str(b.get("binding_kind") or "")
        vhash = str(b.get("value_hash") or "")
        ev_res = make_binding_resolve_event_v101(
            conversation_id=str(conversation_id),
            ts_logical=int(binding_ts_logical),
            turn_id=str(turn_id),
            pronoun=str(resolution.get("pronoun") or ""),
            resolution=dict(resolution),
            chosen_binding_id=str(chosen_binding_id),
            binding_kind=str(bkind),
            binding_value_hash=str(vhash),
            created_step=int(step),
            prev_event_sig=str(prev_binding_event_sig),
            use_count=int(use_count),
            last_used_turn_index=int(user_turn_index),
        )
        _append_binding_event(ev_res)

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

        # Intercepts: agency/explain_agency/trace_agency -> plans/explain_plan/trace_plans -> TEACH_CONCEPT -> TEACH -> EXPLAIN -> BELIEFS -> BELIEF/REVISE -> FORGET (memory/belief) -> NOTE/RECALL -> parser V92/compound.
        if is_agency_command_v105(str(user_text)):
            pr = parse_agency_command_v105(str(user_text))
            ok = bool(pr.get("recognized", False)) and bool(pr.get("ok", False))
            sem = {
                "schema_version": 105,
                "intent_id": INTENT_AGENCY_V105,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(pr.get("reason") or "not_recognized"),
                "query": "",
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_explain_agency_command_v105(str(user_text)):
            pr = parse_explain_agency_command_v105(str(user_text))
            ok = bool(pr.get("recognized", False)) and bool(pr.get("ok", False))
            sem = {
                "schema_version": 105,
                "intent_id": INTENT_EXPLAIN_AGENCY_V105,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(pr.get("reason") or "not_recognized"),
                "query": str(pr.get("query") or "") if ok else "",
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_trace_agency_command_v105(str(user_text)):
            pr = parse_trace_agency_command_v105(str(user_text))
            ok = bool(pr.get("recognized", False)) and bool(pr.get("ok", False))
            sem = {
                "schema_version": 105,
                "intent_id": INTENT_TRACE_AGENCY_V105,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(pr.get("reason") or "not_recognized"),
                "query": str(pr.get("query") or "") if ok else "",
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_plans_command_v104(str(user_text)):
            pr = parse_plans_command_v104(str(user_text))
            ok = bool(pr.get("recognized", False)) and bool(pr.get("ok", False))
            sem = {
                "schema_version": 104,
                "intent_id": INTENT_PLANS_V104,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(pr.get("reason") or "not_recognized"),
                "query": "",
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_explain_plan_command_v104(str(user_text)):
            pr = parse_explain_plan_command_v104(str(user_text))
            ok = bool(pr.get("recognized", False)) and bool(pr.get("ok", False))
            sem = {
                "schema_version": 104,
                "intent_id": INTENT_EXPLAIN_PLAN_V104,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(pr.get("reason") or "not_recognized"),
                "query": str(pr.get("query") or "") if ok else "",
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_trace_plans_command_v104(str(user_text)):
            pr = parse_trace_plans_command_v104(str(user_text))
            ok = bool(pr.get("recognized", False)) and bool(pr.get("ok", False))
            sem = {
                "schema_version": 104,
                "intent_id": INTENT_TRACE_PLANS_V104,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(pr.get("reason") or "not_recognized"),
                "query": str(pr.get("query") or "") if ok else "",
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_teach_concept_command_v103(str(user_text)):
            tc = parse_teach_concept_command_v103(str(user_text))
            ok = bool(tc.get("recognized", False)) and bool(tc.get("ok", False))
            reason = str(tc.get("reason") or "not_recognized")
            sem = {
                "schema_version": 103,
                "intent_id": INTENT_TEACH_CONCEPT_V103,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "name": str(tc.get("name") or ""),
                "polarity": str(tc.get("polarity") or ""),
                "text": str(tc.get("text") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                ctx["teach_concept"] = {"name": str(tc.get("name") or ""), "polarity": str(tc.get("polarity") or ""), "text": str(tc.get("text") or "")}

        elif is_teach_command_v93(str(user_text)):
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

        elif is_style_profile_command_v102(str(user_text)):
            sp = parse_style_profile_command_v102(str(user_text))
            ok = bool(sp.get("recognized", False)) and bool(sp.get("ok", False))
            reason = str(sp.get("reason") or "not_recognized")
            sem = {
                "schema_version": 102,
                "intent_id": INTENT_STYLE_PROFILE_V102,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(sp.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                ctx["style_profile_text"] = render_style_profile_text_v102(style_profile)

        elif is_templates_command_v102(str(user_text)):
            tp = parse_templates_command_v102(str(user_text))
            ok = bool(tp.get("recognized", False)) and bool(tp.get("ok", False))
            reason = str(tp.get("reason") or "not_recognized")
            sem = {
                "schema_version": 102,
                "intent_id": INTENT_TEMPLATES_V102,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(tp.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                stats_now = fold_template_stats_v102(templates=list(templates_v102), style_events=list(style_events))
                ctx["templates_text"] = render_templates_list_text_v102(templates=list(templates_v102), template_stats=dict(stats_now))

        elif is_explain_style_command_v102(str(user_text)):
            es = parse_explain_style_command_v102(str(user_text))
            ok = bool(es.get("recognized", False)) and bool(es.get("ok", False))
            reason = str(es.get("reason") or "not_recognized")
            turn_id = str(es.get("turn_id") or "")
            sem = {
                "schema_version": 102,
                "intent_id": INTENT_EXPLAIN_STYLE_V102,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "turn_id": str(turn_id),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                target = None
                for sev in style_events:
                    if isinstance(sev, dict) and str(sev.get("turn_id") or "") == str(turn_id):
                        target = dict(sev)
                        break
                if target is None:
                    parse["parse_ok"] = False
                    parse["reason"] = "unknown_turn_id"
                else:
                    ctx["explain_style_text"] = explain_style_text_v102(style_event=dict(target))

        elif is_trace_style_command_v102(str(user_text)):
            ts = parse_trace_style_command_v102(str(user_text))
            ok = bool(ts.get("recognized", False)) and bool(ts.get("ok", False))
            reason = str(ts.get("reason") or "not_recognized")
            turn_id = str(ts.get("turn_id") or "")
            sem = {
                "schema_version": 102,
                "intent_id": INTENT_TRACE_STYLE_V102,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "turn_id": str(turn_id),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                eid = ""
                for sev in style_events:
                    if isinstance(sev, dict) and str(sev.get("turn_id") or "") == str(turn_id):
                        eid = str(sev.get("event_id") or "")
                        break
                ctx["trace_style_text"] = f"TRACE_STYLE: turn_id={turn_id} style_event_id={eid}"

        elif is_concepts_command_v103(str(user_text)):
            cc = parse_concepts_command_v103(str(user_text))
            ok = bool(cc.get("recognized", False)) and bool(cc.get("ok", False))
            reason = str(cc.get("reason") or "not_recognized")
            sem = {
                "schema_version": 103,
                "intent_id": INTENT_CONCEPTS_V103,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                reg_now, _det = fold_concept_ledger_v103(list(concept_events))
                ctx["concepts_text"] = render_concepts_text_v103(registry=dict(reg_now))

        elif is_explain_concept_command_v103(str(user_text)):
            ec = parse_explain_concept_command_v103(str(user_text))
            ok = bool(ec.get("recognized", False)) and bool(ec.get("ok", False))
            reason = str(ec.get("reason") or "not_recognized")
            q = str(ec.get("query") or "")
            sem = {
                "schema_version": 103,
                "intent_id": INTENT_EXPLAIN_CONCEPT_V103,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "query": str(q),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                reg_now, _det = fold_concept_ledger_v103(list(concept_events))
                ctx["explain_concept_text"] = render_explain_concept_text_v103(registry=dict(reg_now), query=str(q))

        elif is_trace_concepts_command_v103(str(user_text)):
            tc = parse_trace_concepts_command_v103(str(user_text))
            ok = bool(tc.get("recognized", False)) and bool(tc.get("ok", False))
            reason = str(tc.get("reason") or "not_recognized")
            tref = str(tc.get("turn_ref") or "")
            sem = {
                "schema_version": 103,
                "intent_id": INTENT_TRACE_CONCEPTS_V103,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "turn_ref": str(tref),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))
            if ok:
                # resolve turn_ref to turn_id (turn_id literal or numeric turn_index)
                target_tid = ""
                if str(tref).isdigit():
                    try:
                        idx = int(tref)
                    except Exception:
                        idx = -1
                    if idx >= 0 and idx < len(turns) and isinstance(turns[idx], dict):
                        target_tid = str(turns[idx].get("turn_id") or "")
                    else:
                        for t in turns:
                            if isinstance(t, dict) and int(t.get("turn_index", -1)) == idx:
                                target_tid = str(t.get("turn_id") or "")
                                break
                else:
                    target_tid = str(tref)
                reg_now, _det = fold_concept_ledger_v103(list(concept_events))
                ctx["trace_concepts_text"] = render_trace_concepts_text_v103(
                    registry_prefix=dict(reg_now), concept_events=list(concept_events), turns=list(turns), target_turn_id=str(target_tid)
                )

        elif is_bindings_command_v101(str(user_text)):
            be = parse_bindings_command_v101(str(user_text))
            ok = bool(be.get("recognized", False)) and bool(be.get("ok", False))
            reason = str(be.get("reason") or "not_recognized")
            sem = {
                "schema_version": 101,
                "intent_id": INTENT_BINDINGS_LIST_V101,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(be.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_explain_binding_command_v101(str(user_text)):
            eb = parse_explain_binding_command_v101(str(user_text))
            ok = bool(eb.get("recognized", False)) and bool(eb.get("ok", False))
            reason = str(eb.get("reason") or "not_recognized")
            sem = {
                "schema_version": 101,
                "intent_id": INTENT_EXPLAIN_BINDING_V101,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(eb.get("prefix") or ""),
                "binding_id": str(eb.get("binding_id") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_trace_ref_command_v101(str(user_text)):
            tr = parse_trace_ref_command_v101(str(user_text))
            ok = bool(tr.get("recognized", False)) and bool(tr.get("ok", False))
            reason = str(tr.get("reason") or "not_recognized")
            sem = {
                "schema_version": 101,
                "intent_id": INTENT_TRACE_REF_V101,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(tr.get("prefix") or ""),
                "turn_id": str(tr.get("turn_id") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_plan_create_3_command_v101(str(user_text)):
            pc = parse_plan_create_3_command_v101(str(user_text))
            ok = bool(pc.get("recognized", False)) and bool(pc.get("ok", False))
            reason = str(pc.get("reason") or "not_recognized")
            sem = {
                "schema_version": 101,
                "intent_id": INTENT_PLAN_CREATE_V101,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "target": str(pc.get("target") or ""),
                "steps_total": int(pc.get("steps_total") or 3),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_shorten_ref_command_v101(str(user_text)):
            sc = parse_shorten_ref_command_v101(str(user_text))
            ok = bool(sc.get("recognized", False)) and bool(sc.get("ok", False))
            reason = str(sc.get("reason") or "not_recognized")
            sem = {
                "schema_version": 101,
                "intent_id": INTENT_PLAN_SHORTEN_REF_V101,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "pronoun": str(sc.get("pronoun") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_priority_high_ref_command_v101(str(user_text)):
            pr = parse_priority_high_ref_command_v101(str(user_text))
            ok = bool(pr.get("recognized", False)) and bool(pr.get("ok", False))
            reason = str(pr.get("reason") or "not_recognized")
            sem = {
                "schema_version": 101,
                "intent_id": INTENT_GOAL_PRIORITY_HIGH_REF_V101,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "pronoun": str(pr.get("pronoun") or ""),
                "priority": str(pr.get("priority") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_explain_command_v94(str(user_text)) and (not is_explain_dialogue_command_v106(str(user_text))) and (not is_explain_pragmatics_command_v107(str(user_text))) and (not is_explain_flow_command_v108(str(user_text))):
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

        elif is_dialogue_command_v106(str(user_text)):
            dg = parse_dialogue_command_v106(str(user_text))
            ok = bool(dg.get("recognized", False)) and bool(dg.get("ok", False))
            reason = str(dg.get("reason") or "not_recognized")
            sem = {
                "schema_version": 106,
                "intent_id": INTENT_DIALOGUE_V106,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(dg.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_explain_dialogue_command_v106(str(user_text)):
            edg = parse_explain_dialogue_command_v106(str(user_text))
            ok = bool(edg.get("recognized", False)) and bool(edg.get("ok", False))
            reason = str(edg.get("reason") or "not_recognized")
            sem = {
                "schema_version": 106,
                "intent_id": INTENT_EXPLAIN_DIALOGUE_V106,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(edg.get("prefix") or ""),
                "query": str(edg.get("query") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_trace_dialogue_command_v106(str(user_text)):
            tdg = parse_trace_dialogue_command_v106(str(user_text))
            ok = bool(tdg.get("recognized", False)) and bool(tdg.get("ok", False))
            reason = str(tdg.get("reason") or "not_recognized")
            sem = {
                "schema_version": 106,
                "intent_id": INTENT_TRACE_DIALOGUE_V106,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(tdg.get("prefix") or ""),
                "query": str(tdg.get("query") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_pragmatics_command_v107(str(user_text)):
            pg = parse_pragmatics_command_v107(str(user_text))
            ok = bool(pg.get("recognized", False)) and bool(pg.get("ok", False))
            reason = str(pg.get("reason") or "not_recognized")
            sem = {
                "schema_version": 107,
                "intent_id": INTENT_PRAGMATICS_V107,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(pg.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_explain_pragmatics_command_v107(str(user_text)):
            epg = parse_explain_pragmatics_command_v107(str(user_text))
            ok = bool(epg.get("recognized", False)) and bool(epg.get("ok", False))
            reason = str(epg.get("reason") or "not_recognized")
            sem = {
                "schema_version": 107,
                "intent_id": INTENT_EXPLAIN_PRAGMATICS_V107,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(epg.get("prefix") or ""),
                "query": str(epg.get("query") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_trace_pragmatics_command_v107(str(user_text)):
            tpg = parse_trace_pragmatics_command_v107(str(user_text))
            ok = bool(tpg.get("recognized", False)) and bool(tpg.get("ok", False))
            reason = str(tpg.get("reason") or "not_recognized")
            sem = {
                "schema_version": 107,
                "intent_id": INTENT_TRACE_PRAGMATICS_V107,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(tpg.get("prefix") or ""),
                "query": str(tpg.get("query") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_flow_command_v108(str(user_text)):
            fl = parse_flow_command_v108(str(user_text))
            ok = bool(fl.get("recognized", False)) and bool(fl.get("ok", False))
            reason = str(fl.get("reason") or "not_recognized")
            sem = {
                "schema_version": 108,
                "intent_id": INTENT_FLOW_V108,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(fl.get("prefix") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_explain_flow_command_v108(str(user_text)):
            efl = parse_explain_flow_command_v108(str(user_text))
            ok = bool(efl.get("recognized", False)) and bool(efl.get("ok", False))
            reason = str(efl.get("reason") or "not_recognized")
            sem = {
                "schema_version": 108,
                "intent_id": INTENT_EXPLAIN_FLOW_V108,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(efl.get("prefix") or ""),
                "query": str(efl.get("query") or ""),
            }
            sig = _stable_hash_obj(sem)
            parse = dict(sem, parse_sig=str(sig))

        elif is_trace_flow_command_v108(str(user_text)):
            tfl = parse_trace_flow_command_v108(str(user_text))
            ok = bool(tfl.get("recognized", False)) and bool(tfl.get("ok", False))
            reason = str(tfl.get("reason") or "not_recognized")
            sem = {
                "schema_version": 108,
                "intent_id": INTENT_TRACE_FLOW_V108,
                "matched_rule_id": "",
                "compound": False,
                "parse_ok": bool(ok),
                "reason": str(reason),
                "prefix": str(tfl.get("prefix") or ""),
                "query": str(tfl.get("query") or ""),
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

        # V102: style profile update from explicit user signals (deterministic, no ML).
        style_profile_before = style_profile
        style_profile, style_profile_updates = derive_style_profile_update_v102(
            prev=style_profile,
            user_text=str(user_text),
            user_turn_id=str(ut.get("turn_id") or ""),
        )

        # V107: classify user intent act (deterministic; used for pragmatics + survival gating).
        user_intent_act_v107 = classify_user_intent_act_v107(user_text=str(user_text), parse_intent_id=str(parse.get("intent_id") or ""))

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
        concept_hit_names_turn: List[str] = []
        agency_event_appended = False

        objective_kind = ""
        ctx2: Dict[str, Any] = {}

        beliefs_active = _active_beliefs_by_key()

        # V103: concept matching side-effects (does not affect primary response).
        # Important: skip concept meta-commands to avoid verifier mismatch (they render from prefix state).
        intent_now = str(parse.get("intent_id") or "")
        if intent_now not in {
            INTENT_TEACH_CONCEPT_V103,
            INTENT_CONCEPTS_V103,
            INTENT_EXPLAIN_CONCEPT_V103,
            INTENT_TRACE_CONCEPTS_V103,
        }:
            reg_now, _det = fold_concept_ledger_v103(list(concept_events))
            c_by_id = reg_now.get("concepts_by_id") if isinstance(reg_now.get("concepts_by_id"), dict) else {}
            csv_by_id = reg_now.get("csv_by_id") if isinstance(reg_now.get("csv_by_id"), dict) else {}
            for cid in sorted(c_by_id.keys(), key=str):
                st = csv_by_id.get(cid) if isinstance(csv_by_id.get(cid), dict) else {}
                if str(st.get("status") or "ALIVE") != "ALIVE":
                    continue
                csg = c_by_id.get(cid) if isinstance(c_by_id.get(cid), dict) else {}
                m = match_csg_v103(csg=dict(csg), text=str(user_text))
                seed_tokens = st.get("seed_tokens") if isinstance(st.get("seed_tokens"), list) else []
                dd = domain_distance_v103(seed_tokens=list(seed_tokens), tokens=list(m.get("tokens") or []))
                matched = bool(m.get("matched", False))
                evidence = list(m.get("evidence") or [])
                if matched:
                    cname = str(csg.get("name") or "")
                    concept_hit_names_turn.append(cname if cname else str(cid))
                    evm = ConceptEventV103(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        turn_id=str(ut.get("turn_id") or ""),
                        type="CONCEPT_MATCH",
                        payload={
                            "concept_id": str(cid),
                            "evidence": list(evidence),
                        },
                        created_step=int(step),
                        prev_event_sig=str(prev_concept_event_sig),
                    ).to_dict()
                    _append_concept_event(evm)
                    # ToC pressure: if domain_distance is high, record PASS for this match.
                    if float(dd) >= 0.6:
                        evt = ConceptEventV103(
                            conversation_id=str(conversation_id),
                            ts_turn_index=int(ut.get("turn_index") or 0),
                            turn_id=str(ut.get("turn_id") or ""),
                            type="CONCEPT_TOC_PASS",
                            payload={"concept_id": str(cid), "domain_distance": float(round(float(dd), 6))},
                            created_step=int(step),
                            prev_event_sig=str(prev_concept_event_sig),
                        ).to_dict()
                        _append_concept_event(evt)
                else:
                    # Fail-closed ToC signal: only record FAIL when the input shares at least 1 rule feature,
                    # and is far from the seed domain (high domain_distance).
                    if float(dd) >= 0.6 and evidence:
                        evtf = ConceptEventV103(
                            conversation_id=str(conversation_id),
                            ts_turn_index=int(ut.get("turn_index") or 0),
                            turn_id=str(ut.get("turn_id") or ""),
                            type="CONCEPT_TOC_FAIL",
                            payload={"concept_id": str(cid), "domain_distance": float(round(float(dd), 6)), "evidence": list(evidence)},
                            created_step=int(step),
                            prev_event_sig=str(prev_concept_event_sig),
                        ).to_dict()
                        _append_concept_event(evtf)
                        # Pressure: prune only when ToC fails accumulate and usage is non-trivial.
                        reg2, _d2 = fold_concept_ledger_v103(list(concept_events))
                        st2_all = reg2.get("csv_by_id") if isinstance(reg2.get("csv_by_id"), dict) else {}
                        st2 = st2_all.get(cid) if isinstance(st2_all.get(cid), dict) else {}
                        toc_fail = int(st2.get("toc_fail") or 0)
                        usage = int(st2.get("usage_count") or 0)
                        status = str(st2.get("status") or "ALIVE")
                        if status == "ALIVE" and usage >= 5 and toc_fail >= 2:
                            new_status = "DEAD" if toc_fail >= 3 else "DEPRECATED"
                            evp = ConceptEventV103(
                                conversation_id=str(conversation_id),
                                ts_turn_index=int(ut.get("turn_index") or 0),
                                turn_id=str(ut.get("turn_id") or ""),
                                type="CONCEPT_PRUNE",
                                payload={"concept_id": str(cid), "status": str(new_status), "reason": "toc_fail_threshold"},
                                created_step=int(step),
                                prev_event_sig=str(prev_concept_event_sig),
                            ).to_dict()
                            _append_concept_event(evp)

        if str(parse.get("intent_id") or "") == INTENT_TEACH_CONCEPT_V103:
            if bool(parse.get("parse_ok", False)):
                name = str(parse.get("name") or "").strip()
                polarity = str(parse.get("polarity") or "").strip()
                raw_txt = str(parse.get("text") or "")
                ex_norm = normalize_example_text_v103(str(raw_txt))
                ex_sig = example_sig_v103(str(ex_norm))

                reg_before, _d = fold_concept_ledger_v103(list(concept_events))
                fb_all = reg_before.get("feedback_by_name") if isinstance(reg_before.get("feedback_by_name"), dict) else {}
                fb = fb_all.get(name) if isinstance(fb_all.get(name), dict) else {}
                pos_sigs = set([str(x) for x in (fb.get("pos_sigs") or []) if str(x)])
                neg_sigs = set([str(x) for x in (fb.get("neg_sigs") or []) if str(x)])
                if (polarity == "+" and ex_sig in neg_sigs) or (polarity == "-" and ex_sig in pos_sigs):
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "teach_concept_conflict_polarity"}
                    evrej = ConceptEventV103(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        turn_id=str(ut.get("turn_id") or ""),
                        type="CONCEPT_REJECT",
                        payload={
                            "name": str(name),
                            "reason": "example_polarity_conflict",
                            "example_sig": str(ex_sig),
                        },
                        created_step=int(step),
                        prev_event_sig=str(prev_concept_event_sig),
                    ).to_dict()
                    _append_concept_event(evrej)
                else:
                    # Record feedback event.
                    cid0 = lookup_concept_id_by_name_v103(registry=dict(reg_before), name_or_id=str(name))
                    evfb = ConceptEventV103(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        turn_id=str(ut.get("turn_id") or ""),
                        type="CONCEPT_FEEDBACK",
                        payload={
                            "name": str(name),
                            "concept_id": str(cid0),
                            "polarity": str(polarity),
                            "example_sig": str(ex_sig),
                            "example_text_norm": str(ex_norm),
                        },
                        created_step=int(step),
                        prev_event_sig=str(prev_concept_event_sig),
                    ).to_dict()
                    _append_concept_event(evfb)

                    induced_cid = ""
                    induced_delta = 0
                    reg_after, _d2 = fold_concept_ledger_v103(list(concept_events))
                    cid_after = lookup_concept_id_by_name_v103(registry=dict(reg_after), name_or_id=str(name))
                    fb2 = (reg_after.get("feedback_by_name") or {}).get(name) if isinstance(reg_after.get("feedback_by_name"), dict) else {}
                    pos_sigs2 = [str(x) for x in (fb2.get("pos_sigs") or []) if str(x)]
                    neg_sigs2 = [str(x) for x in (fb2.get("neg_sigs") or []) if str(x)]
                    # Auto-induce once, when no concept exists yet and evidence threshold is met.
                    if (not cid_after) and len(pos_sigs2) >= 2 and len(neg_sigs2) >= 1:
                        pos_text_map = fb2.get("pos_text") if isinstance(fb2.get("pos_text"), dict) else {}
                        neg_text_map = fb2.get("neg_text") if isinstance(fb2.get("neg_text"), dict) else {}
                        pos_examples = [str(pos_text_map.get(sig) or "") for sig in pos_sigs2 if str(pos_text_map.get(sig) or "")]
                        neg_examples = [str(neg_text_map.get(sig) or "") for sig in neg_sigs2 if str(neg_text_map.get(sig) or "")]
                        rule, dbg = induce_rule_v103(pos_examples=list(pos_examples), neg_examples=list(neg_examples), max_k=2)
                        if rule is None:
                            evrej2 = ConceptEventV103(
                                conversation_id=str(conversation_id),
                                ts_turn_index=int(ut.get("turn_index") or 0),
                                turn_id=str(ut.get("turn_id") or ""),
                                type="CONCEPT_REJECT",
                                payload={"name": str(name), "reason": str(dbg.get("reason") or "induce_failed")},
                                created_step=int(step),
                                prev_event_sig=str(prev_concept_event_sig),
                            ).to_dict()
                            _append_concept_event(evrej2)
                        else:
                            csg = make_csg_rule_v103(name=str(name), features=list(rule.features))
                            induced_cid = str(csg.get("concept_id") or "")
                            evc = ConceptEventV103(
                                conversation_id=str(conversation_id),
                                ts_turn_index=int(ut.get("turn_index") or 0),
                                turn_id=str(ut.get("turn_id") or ""),
                                type="CONCEPT_CREATE",
                                payload={"csg": dict(csg)},
                                created_step=int(step),
                                prev_event_sig=str(prev_concept_event_sig),
                            ).to_dict()
                            _append_concept_event(evc)
                            induced_delta = int(rule.mdl_delta_bits)
                            evi = ConceptEventV103(
                                conversation_id=str(conversation_id),
                                ts_turn_index=int(ut.get("turn_index") or 0),
                                turn_id=str(ut.get("turn_id") or ""),
                                type="CONCEPT_INDUCE",
                                payload=dict(
                                    {
                                        "concept_id": str(induced_cid),
                                        "mdl_baseline_bits": int(rule.mdl_baseline_bits),
                                        "mdl_model_bits": int(rule.mdl_model_bits),
                                        "mdl_data_bits": int(rule.mdl_data_bits),
                                        "mdl_delta_bits": int(rule.mdl_delta_bits),
                                        "seed_tokens": list(rule.seed_tokens),
                                    }
                                ),
                                created_step=int(step),
                                prev_event_sig=str(prev_concept_event_sig),
                            ).to_dict()
                            _append_concept_event(evi)

                    objective_kind = "COMM_RESPOND"
                    ctx2 = {
                        "teach_concept_ack_text": "TEACH_CONCEPT OK: name={n} polarity={p} example_sig={s} induced_concept_id={cid} mdl_delta_bits={dd}".format(
                            n=str(name),
                            p=str(polarity),
                            s=str(ex_sig),
                            cid=str(induced_cid or cid0),
                            dd=int(induced_delta),
                        )
                    }
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"teach_concept_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_CONCEPTS_V103:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"concepts_text": str(ctx.get("concepts_text") or "")}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"concepts_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_CONCEPT_V103:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"explain_concept_text": str(ctx.get("explain_concept_text") or "")}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_concept_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_CONCEPTS_V103:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"trace_concepts_text": str(ctx.get("trace_concepts_text") or "")}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_concepts_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_AGENCY_V105:
            if bool(parse.get("parse_ok", False)):
                # Render from prefix (exclude this turn's agency event to avoid recursion).
                cur_idx = int(ut.get("turn_index") or 0)
                prefix = []
                for ev0 in list(agency_events):
                    if not isinstance(ev0, dict):
                        continue
                    try:
                        ti = int(ev0.get("ts_turn_index") or 0)
                    except Exception:
                        ti = 0
                    if ti >= cur_idx:
                        break
                    prefix.append(dict(ev0))
                snap = agency_registry_snapshot_v105(list(prefix))
                objective_kind = "COMM_RESPOND"
                ctx2 = {"planned_text": render_agency_text_v105(dict(snap))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"agency_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_AGENCY_V105:
            if bool(parse.get("parse_ok", False)):
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                ev0 = lookup_agency_event_v105(agency_events=list(agency_events), user_turn_index=int(qidx))
                objective_kind = "COMM_RESPOND"
                ctx2 = {"planned_text": render_explain_agency_text_v105(dict(ev0) if isinstance(ev0, dict) else {})}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_agency_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_AGENCY_V105:
            if bool(parse.get("parse_ok", False)):
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                objective_kind = "COMM_RESPOND"
                ctx2 = {"planned_text": render_trace_agency_text_v105(agency_events=list(agency_events), until_turn_index=int(qidx))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_agency_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_PLANS_V104:
            if bool(parse.get("parse_ok", False)):
                regp = fold_plan_ledger_v104(plan_events=list(plan_events))
                objective_kind = "COMM_RESPOND"
                ctx2 = {"plans_text": render_plans_text_v104(registry=dict(regp))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"plans_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_PLAN_V104:
            if bool(parse.get("parse_ok", False)):
                q = str(parse.get("query") or "")
                evp = lookup_plan_event_v104(plan_events=list(plan_events), query=str(q))
                if not isinstance(evp, dict):
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "plan_not_found"}
                else:
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"explain_plan_text": render_explain_plan_text_v104(plan_event=dict(evp))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_plan_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_PLANS_V104:
            if bool(parse.get("parse_ok", False)):
                q = str(parse.get("query") or "")
                evp = lookup_plan_event_v104(plan_events=list(plan_events), query=str(q))
                if not isinstance(evp, dict):
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "plan_not_found"}
                else:
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"trace_plans_text": render_trace_plans_text_v104(plan_event=dict(evp))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_plans_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_NOTE_V96:
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
                    _ensure_binding_create(
                        turn_id=str(ut.get("turn_id") or ""),
                        created_turn_index=int(ut.get("turn_index") or 0),
                        binding_kind="memory",
                        value={"memory_id": str(mid), "memory_text": str(txt)},
                        value_preview=f"memory:{str(txt)[:32]}",
                        provenance={"ledger": "memory_ledger_v96", "source_event_id": str(ev.get("event_id") or "")},
                    )
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
                    _ensure_binding_create(
                        turn_id=str(ut.get("turn_id") or ""),
                        created_turn_index=int(ut.get("turn_index") or 0),
                        binding_kind="belief",
                        value={"belief_key": str(key)},
                        value_preview=f"belief:{str(key)}",
                        provenance={"ledger": "belief_ledger_v96", "source_event_id": str(ev.get("event_id") or "")},
                    )
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
                        _ensure_binding_create(
                            turn_id=str(ut.get("turn_id") or ""),
                            created_turn_index=int(ut.get("turn_index") or 0),
                            binding_kind="belief",
                            value={"belief_key": str(key)},
                            value_preview=f"belief:{str(key)}",
                            provenance={"ledger": "belief_ledger_v96", "source_event_id": str(eva.get("event_id") or "")},
                        )
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

        elif str(parse.get("intent_id") or "") == INTENT_STYLE_PROFILE_V102:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"style_profile_text": str(ctx.get("style_profile_text") or render_style_profile_text_v102(style_profile))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"style_profile_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_TEMPLATES_V102:
            if bool(parse.get("parse_ok", False)):
                stats_now = fold_template_stats_v102(templates=list(templates_v102), style_events=list(style_events))
                objective_kind = "COMM_RESPOND"
                ctx2 = {"templates_text": render_templates_list_text_v102(templates=list(templates_v102), template_stats=dict(stats_now))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"templates_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_STYLE_V102:
            if bool(parse.get("parse_ok", False)) and str(ctx.get("explain_style_text") or ""):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"explain_style_text": str(ctx.get("explain_style_text") or "")}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_style_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_STYLE_V102:
            if bool(parse.get("parse_ok", False)) and str(ctx.get("trace_style_text") or ""):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"trace_style_text": str(ctx.get("trace_style_text") or "")}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_style_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_BINDINGS_LIST_V101:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                ctx2 = {"bindings_text": render_bindings_text_v101(_active_bindings_list())}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"bindings_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_BINDING_V101:
            if bool(parse.get("parse_ok", False)):
                bid = str(parse.get("binding_id") or "")
                b = _binding_get(str(bid))
                if not isinstance(b, dict):
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "binding_not_found"}
                else:
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"binding_explain_text": render_explain_binding_text_v101(b)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_binding_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_REF_V101:
            if bool(parse.get("parse_ok", False)):
                tid = str(parse.get("turn_id") or "")
                objective_kind = "COMM_RESPOND"
                ctx2 = {"trace_ref_text": render_trace_ref_text_v101(turn_id=str(tid), binding_events=list(binding_events))}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_ref_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_PLAN_CREATE_V101:
            if bool(parse.get("parse_ok", False)):
                target = str(parse.get("target") or "").strip()
                if not target:
                    objective_kind = "COMM_CORRECT"
                    ctx2 = {"msg": "missing_target"}
                else:
                    # Create a goal ledger entry (deterministic, internal) and bind it.
                    gid = goal_id_v99(conversation_id=str(conversation_id), ts_turn_index=int(ut.get("turn_index") or 0), text=str(target), parent_goal_id="")
                    ge = GoalEventV99(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        op="GOAL_ADD",
                        goal_id=str(gid),
                        parent_goal_id="",
                        priority=100,
                        status="active",
                        text=str(target),
                        cause_type="user_intent",
                        cause_id=str(parse.get("parse_sig") or ""),
                        created_step=int(step),
                        prev_event_sig=str(prev_goal_event_sig),
                    ).to_dict()
                    prev_goal_event_sig = str(ge.get("event_sig") or prev_goal_event_sig)
                    prev_goal_hash = append_chained_jsonl_v96(
                        goal_path,
                        {"time": deterministic_iso(step=int(ge["created_step"])), "step": int(ge["created_step"]), "event": "GOAL", "payload": dict(ge)},
                        prev_hash=prev_goal_hash,
                    )
                    goal_events.append(dict(ge))
                    goal_write_ids.append(str(gid))
                    # Bind the goal pointer.
                    _ensure_binding_create(
                        turn_id=str(ut.get("turn_id") or ""),
                        created_turn_index=int(ut.get("turn_index") or 0),
                        binding_kind="goal",
                        value={"goal_id": str(gid), "goal_text": str(target)},
                        value_preview=f"goal:{str(target)}",
                        provenance={"ledger": "goal_ledger_v99", "source_event_id": str(ge.get("event_id") or "")},
                    )
                    # Create a plan object (3 steps) and bind it.
                    steps = [
                        f"1) Definir objetivo: {target}",
                        f"2) Executar ação principal para: {target}",
                        f"3) Verificar resultado de: {target}",
                    ]
                    plan_obj = {"schema_version": 101, "kind": "plan_v101", "goal_id": str(gid), "goal_text": str(target), "steps": list(steps)}
                    pbid = _ensure_binding_create(
                        turn_id=str(ut.get("turn_id") or ""),
                        created_turn_index=int(ut.get("turn_index") or 0),
                        binding_kind="plan",
                        value=dict(plan_obj),
                        value_preview=f"plan(3):{str(target)}",
                        provenance={"ledger": "plan_v101", "source_event_id": str(_stable_hash_obj(plan_obj))},
                    )
                    objective_kind = "COMM_RESPOND"
                    ctx2 = {"plan_text": "PLAN(3):\n" + "\n".join(steps), "plan_binding_id": str(pbid), "goal_id": str(gid)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"plan_create_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_PLAN_SHORTEN_REF_V101:
            if bool(parse.get("parse_ok", False)):
                pron = str(parse.get("pronoun") or "")
                goals_active_by_id, _gdet = fold_goal_ledger_v99(list(goal_events))
                goals_active_ids = sorted([str(k) for k in goals_active_by_id.keys() if str(k)], key=str)
                res = resolve_reference_v101(
                    pronoun=str(pron),
                    kind_hint="plan",
                    bindings_active=_active_bindings_list(),
                    current_user_turn_index=int(ut.get("turn_index") or 0),
                    goals_active_ids=list(goals_active_ids),
                )
                status = str(res.get("status") or "")
                if status == "RESOLVED":
                    chosen_bid = str(res.get("chosen_binding_id") or "")
                    b = _binding_get(chosen_bid)
                    if not isinstance(b, dict):
                        objective_kind = "COMM_CORRECT"
                        ctx2 = {"msg": "binding_not_found"}
                    else:
                        _binding_use(turn_id=str(ut.get("turn_id") or ""), user_turn_index=int(ut.get("turn_index") or 0), resolution=dict(res), chosen_binding_id=str(chosen_bid))
                        pval = b.get("value") if isinstance(b.get("value"), dict) else {}
                        gtxt = str(pval.get("goal_text") or "")
                        gid = str(pval.get("goal_id") or "")
                        steps2 = [f"1) Executar: {gtxt}", f"2) Verificar: {gtxt}"]
                        plan2 = {"schema_version": 101, "kind": "plan_v101", "goal_id": str(gid), "goal_text": str(gtxt), "steps": list(steps2)}
                        _ensure_binding_create(
                            turn_id=str(ut.get("turn_id") or ""),
                            created_turn_index=int(ut.get("turn_index") or 0),
                            binding_kind="plan",
                            value=dict(plan2),
                            value_preview=f"plan(2):{str(gtxt)}",
                            provenance={"ledger": "plan_v101", "source_event_id": str(_stable_hash_obj(plan2))},
                        )
                        objective_kind = "COMM_RESPOND"
                        ctx2 = {"plan_text": "PLAN(2):\n" + "\n".join(steps2)}
                elif status == "AMBIGUOUS":
                    ev_amb = make_binding_ambiguous_event_v101(
                        conversation_id=str(conversation_id),
                        ts_logical=int(binding_ts_logical),
                        turn_id=str(ut.get("turn_id") or ""),
                        pronoun=str(pron),
                        resolution=dict(res),
                        created_step=int(step),
                        prev_event_sig=str(prev_binding_event_sig),
                    )
                    _append_binding_event(ev_amb)
                    objective_kind = "COMM_CONFIRM"
                    cands = res.get("candidates") if isinstance(res.get("candidates"), list) else []
                    opts = [f"{i+1}) {str(c.get('value_preview') or '')}" for i, c in enumerate(cands[:3]) if isinstance(c, dict)]
                    ctx2 = {"msg": "Referência ambígua. Qual você quer dizer? " + "; ".join(opts)}
                else:
                    ev_miss = make_binding_miss_event_v101(
                        conversation_id=str(conversation_id),
                        ts_logical=int(binding_ts_logical),
                        turn_id=str(ut.get("turn_id") or ""),
                        pronoun=str(pron),
                        resolution=dict(res),
                        created_step=int(step),
                        prev_event_sig=str(prev_binding_event_sig),
                    )
                    _append_binding_event(ev_miss)
                    objective_kind = "COMM_ASK_CLARIFY"
                    ctx2 = {"missing_ref": str(pron)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"shorten_reject:{str(parse.get('reason') or '')}"}

        elif str(parse.get("intent_id") or "") == INTENT_GOAL_PRIORITY_HIGH_REF_V101:
            if bool(parse.get("parse_ok", False)):
                pron = str(parse.get("pronoun") or "")
                goals_active_by_id, _gdet = fold_goal_ledger_v99(list(goal_events))
                goals_active_ids = sorted([str(k) for k in goals_active_by_id.keys() if str(k)], key=str)
                res = resolve_reference_v101(
                    pronoun=str(pron),
                    kind_hint="goal",
                    bindings_active=_active_bindings_list(),
                    current_user_turn_index=int(ut.get("turn_index") or 0),
                    goals_active_ids=list(goals_active_ids),
                )
                status = str(res.get("status") or "")
                if status == "RESOLVED":
                    chosen_bid = str(res.get("chosen_binding_id") or "")
                    b = _binding_get(chosen_bid)
                    if not isinstance(b, dict):
                        objective_kind = "COMM_CORRECT"
                        ctx2 = {"msg": "binding_not_found"}
                    else:
                        _binding_use(turn_id=str(ut.get("turn_id") or ""), user_turn_index=int(ut.get("turn_index") or 0), resolution=dict(res), chosen_binding_id=str(chosen_bid))
                        gval = b.get("value") if isinstance(b.get("value"), dict) else {}
                        gid = str(gval.get("goal_id") or "")
                        gtxt = str(gval.get("goal_text") or "")
                        if not gid:
                            objective_kind = "COMM_CORRECT"
                            ctx2 = {"msg": "goal_id_missing_in_binding"}
                        else:
                            ge2 = GoalEventV99(
                                conversation_id=str(conversation_id),
                                ts_turn_index=int(ut.get("turn_index") or 0),
                                op="GOAL_UPDATE",
                                goal_id=str(gid),
                                parent_goal_id="",
                                priority=200,
                                status="active",
                                text=str(gtxt),
                                cause_type="user_intent",
                                cause_id=str(parse.get("parse_sig") or ""),
                                created_step=int(step),
                                prev_event_sig=str(prev_goal_event_sig),
                            ).to_dict()
                            prev_goal_event_sig = str(ge2.get("event_sig") or prev_goal_event_sig)
                            prev_goal_hash = append_chained_jsonl_v96(
                                goal_path,
                                {"time": deterministic_iso(step=int(ge2["created_step"])), "step": int(ge2["created_step"]), "event": "GOAL", "payload": dict(ge2)},
                                prev_hash=prev_goal_hash,
                            )
                            goal_events.append(dict(ge2))
                            goal_write_ids.append(str(gid))
                            objective_kind = "COMM_RESPOND"
                            ctx2 = {"msg": f"OK: goal_priority_high goal_id={gid}"}
                elif status == "AMBIGUOUS":
                    ev_amb = make_binding_ambiguous_event_v101(
                        conversation_id=str(conversation_id),
                        ts_logical=int(binding_ts_logical),
                        turn_id=str(ut.get("turn_id") or ""),
                        pronoun=str(pron),
                        resolution=dict(res),
                        created_step=int(step),
                        prev_event_sig=str(prev_binding_event_sig),
                    )
                    _append_binding_event(ev_amb)
                    objective_kind = "COMM_CONFIRM"
                    cands = res.get("candidates") if isinstance(res.get("candidates"), list) else []
                    opts = [f"{i+1}) {str(c.get('value_preview') or '')}" for i, c in enumerate(cands[:3]) if isinstance(c, dict)]
                    ctx2 = {"msg": "Referência ambígua. Qual você quer dizer? " + "; ".join(opts)}
                else:
                    ev_miss = make_binding_miss_event_v101(
                        conversation_id=str(conversation_id),
                        ts_logical=int(binding_ts_logical),
                        turn_id=str(ut.get("turn_id") or ""),
                        pronoun=str(pron),
                        resolution=dict(res),
                        created_step=int(step),
                        prev_event_sig=str(prev_binding_event_sig),
                    )
                    _append_binding_event(ev_miss)
                    objective_kind = "COMM_ASK_CLARIFY"
                    ctx2 = {"missing_ref": str(pron)}
            else:
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"priority_reject:{str(parse.get('reason') or '')}"}

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

        elif str(parse.get("intent_id") or "") == INTENT_DIALOGUE_V106:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                txt = render_dialogue_text_v106(dialogue_events=list(dialogue_events), until_user_turn_index_exclusive=int(ut.get("turn_index") or 0))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"dialogue_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_DIALOGUE_V106:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                txt = render_explain_dialogue_text_v106(dialogue_events=list(dialogue_events), user_turn_index=int(qidx))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_dialogue_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_DIALOGUE_V106:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                txt = render_trace_dialogue_text_v106(dialogue_events=list(dialogue_events), until_user_turn_index=int(qidx))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_dialogue_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_PRAGMATICS_V107:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                txt = render_pragmatics_text_v107(pragmatics_events=list(pragmatics_events), until_user_turn_index_exclusive=int(ut.get("turn_index") or 0))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"pragmatics_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_PRAGMATICS_V107:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                txt = render_explain_pragmatics_text_v107(pragmatics_events=list(pragmatics_events), user_turn_index=int(qidx))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_pragmatics_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_PRAGMATICS_V107:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                txt = render_trace_pragmatics_text_v107(pragmatics_events=list(pragmatics_events), until_user_turn_index=int(qidx))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_pragmatics_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_FLOW_V108:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                txt = render_flow_text_v108(flow_events=list(flow_events), until_user_turn_index_exclusive=int(ut.get("turn_index") or 0))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"flow_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_EXPLAIN_FLOW_V108:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                txt = render_explain_flow_text_v108(flow_events=list(flow_events), user_turn_index=int(qidx))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"explain_flow_reject:{r or 'parse_fail'}"}

        elif str(parse.get("intent_id") or "") == INTENT_TRACE_FLOW_V108:
            if bool(parse.get("parse_ok", False)):
                objective_kind = "COMM_RESPOND"
                q = str(parse.get("query") or "")
                try:
                    qidx = int(q)
                except Exception:
                    qidx = -1
                txt = render_trace_flow_text_v108(flow_events=list(flow_events), until_user_turn_index=int(qidx))
                ctx2 = {"planned_text": str(txt)}
            else:
                r = str(parse.get("reason") or "")
                objective_kind = "COMM_CORRECT"
                ctx2 = {"msg": f"trace_flow_reject:{r or 'parse_fail'}"}

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

        elif str(parse.get("intent_id") or "") == INTENT_UNKNOWN_V92:
            # V102: style-only user messages (meta preference updates) are treated as valid,
            # deterministic actions when they would otherwise be "no_match". This persists the
            # style_profile in conversation state without executing DSL state mutations.
            if (not bool(parse.get("parse_ok", False))) and isinstance(style_profile_updates, list) and style_profile_updates:
                reasons = [str(u.get("reason") or "") for u in style_profile_updates if isinstance(u, dict)]
                reasons = sorted(set([r for r in reasons if r]))
                reasons_set = set(reasons)
                objective_kind = "COMM_RESPOND"
                if "user_requested_detailed_steps" in reasons_set:
                    ctx2 = {"style_ack_text": "OK: estilo atualizado. Vou responder com mais detalhes, passo a passo."}
                elif "user_requested_short" in reasons_set:
                    ctx2 = {"style_ack_text": "OK: vou responder curto daqui pra frente."}
                elif "user_requested_bullets" in reasons_set:
                    ctx2 = {"style_ack_text": "OK: vou responder em tópicos daqui pra frente."}
                elif "user_requested_example" in reasons_set:
                    ctx2 = {"style_ack_text": "OK: vou incluir um exemplo quando fizer sentido."}
                elif "user_confusion_signal" in reasons_set:
                    ctx2 = {"style_ack_text": "Entendi. Qual parte ficou confusa?"}
                else:
                    ctx2 = {"style_ack_text": "OK: estilo atualizado."}
            else:
                # Reference resolution attempt for bare deictics/pronouns (fail-closed).
                toks = [t for t in normalize_text_v96(str(user_text)).lower().split() if t]
                pron = ""
                for p in ["isso", "aquilo", "isto", "this", "that", "it"]:
                    if p in toks:
                        pron = str(p)
                        break
                if pron:
                    goals_active_by_id, _gdet = fold_goal_ledger_v99(list(goal_events))
                    goals_active_ids = sorted([str(k) for k in goals_active_by_id.keys() if str(k)], key=str)
                    kind_hint = infer_kind_hint_v101(tokens=list(toks))
                    res = resolve_reference_v101(
                        pronoun=str(pron),
                        kind_hint=str(kind_hint),
                        bindings_active=_active_bindings_list(),
                        current_user_turn_index=int(ut.get("turn_index") or 0),
                        goals_active_ids=list(goals_active_ids),
                    )
                    status = str(res.get("status") or "")
                    if status == "MISS":
                        ev_miss = make_binding_miss_event_v101(
                            conversation_id=str(conversation_id),
                            ts_logical=int(binding_ts_logical),
                            turn_id=str(ut.get("turn_id") or ""),
                            pronoun=str(pron),
                            resolution=dict(res),
                            created_step=int(step),
                            prev_event_sig=str(prev_binding_event_sig),
                        )
                        _append_binding_event(ev_miss)
                        objective_kind = "COMM_ASK_CLARIFY"
                        ctx2 = {"missing_ref": str(pron), "binding_status": "MISS"}
                    elif status == "RESOLVED":
                        chosen_bid = str(res.get("chosen_binding_id") or "")
                        if chosen_bid:
                            _binding_use(
                                turn_id=str(ut.get("turn_id") or ""),
                                user_turn_index=int(ut.get("turn_index") or 0),
                                resolution=dict(res),
                                chosen_binding_id=str(chosen_bid),
                            )
                        objective_kind = "COMM_CONFIRM"
                        ctx2 = {"confirm_text": f"Você quis dizer: {chosen_bid}?" if chosen_bid else "Confirme.", "binding_status": "RESOLVED"}
                    else:
                        ev_amb = make_binding_ambiguous_event_v101(
                            conversation_id=str(conversation_id),
                            ts_logical=int(binding_ts_logical),
                            turn_id=str(ut.get("turn_id") or ""),
                            pronoun=str(pron),
                            resolution=dict(res),
                            created_step=int(step),
                            prev_event_sig=str(prev_binding_event_sig),
                        )
                        _append_binding_event(ev_amb)
                        objective_kind = "COMM_CONFIRM"
                        cands = res.get("candidates") if isinstance(res.get("candidates"), list) else []
                        opts = [f"{i+1}) {str(c.get('value_preview') or '')}" for i, c in enumerate(cands[:3]) if isinstance(c, dict)]
                        ctx2 = {"confirm_text": "Referência ambígua. Qual você quer dizer? " + "; ".join(opts), "binding_status": "AMBIGUOUS"}
                else:
                    objective_kind, ctx2 = _choose_comm_objective_v96(
                        parse=dict(parse), vars_map=dict(vars_map), last_answer=last_answer, beliefs_by_key=dict(beliefs_active)
                    )

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

        # V107: pragmatics survival gating (S3): if the dialogue requires repair, do not progress internal ledgers.
        progress_allowed_v107 = True
        # Promise extraction attempts must not produce external commitments; force a safe repair response.
        if str((user_intent_act_v107.get("kind") if isinstance(user_intent_act_v107, dict) else "") or "") == "PROMISE_ATTEMPT":
            objective_kind = "COMM_CONFIRM"
            ctx2 = {"planned_text": "Eu não posso assumir compromissos externos. Posso ajudar com opções verificáveis. Você prefere A ou B?"}
        # Unknown commands must fail-closed: ask to clarify rather than progressing goals/plans/agency.
        if str(parse.get("intent_id") or "") == INTENT_UNKNOWN_V92 and str(objective_kind) == "COMM_CORRECT":
            objective_kind = "COMM_CONFIRM"
            ctx2 = {"planned_text": "Não entendi. Você pode esclarecer o que você quer dizer?"}
        if str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
            progress_allowed_v107 = False

        # V104: planning options A/B/C (plan candidates) + plan ledger (WORM).
        intent_id_for_plan = str(parse.get("intent_id") or "")
        if bool(progress_allowed_v107) and intent_id_for_plan not in {INTENT_PLANS_V104, INTENT_EXPLAIN_PLAN_V104, INTENT_TRACE_PLANS_V104}:
            style_prof_dict = style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict()
            plan_build = build_plan_candidates_v104(
                conversation_id=str(conversation_id),
                user_turn_id=str(ut.get("turn_id") or ""),
                user_turn_index=int(ut.get("turn_index") or 0),
                intent_id=str(intent_id_for_plan),
                parse_sig=str(parse.get("parse_sig") or ""),
                parse=dict(parse),
                user_text=str(user_text),
                concept_hit_names=list(concept_hit_names_turn),
                style_profile_dict=dict(style_prof_dict),
                active_plan_state_before=dict(active_plan_state_v104),
                top_k=3,
            )
            plan_candidates_topk = plan_build.get("candidates_topk") if isinstance(plan_build.get("candidates_topk"), list) else []
            plan_selected = plan_build.get("selected") if isinstance(plan_build.get("selected"), dict) else {}
            active_after = plan_build.get("active_plan_state_after") if isinstance(plan_build.get("active_plan_state_after"), dict) else {}
            selected_kind = str(plan_selected.get("plan_kind") or "")
            selected_cid = str(plan_selected.get("candidate_id") or "")

            # Safe override for unknown commands: offer options or ask clarifying question (no hallucination).
            if intent_id_for_plan == INTENT_UNKNOWN_V92 and str(objective_kind) == "COMM_CORRECT":
                if selected_kind == PLAN_KIND_COMPARE_OPTIONS_V104:
                    objective_kind = "COMM_RESPOND"
                    # Deterministic options from the plan candidates (A/B/C).
                    labels = ["A", "B", "C"]
                    human = {
                        "direct_answer": "Responder direto (se verificável)",
                        "ask_clarify": "Pedir clarificação (uma pergunta específica)",
                        "propose_steps": "Propor um plano em passos",
                        "compare_options": "Comparar opções (A/B/C)",
                        "summarize_and_confirm": "Resumir e confirmar entendimento",
                        "refuse_safe": "Recusar com segurança (não sei)",
                    }
                    lines: List[str] = []
                    for i, c in enumerate(plan_candidates_topk[:3]):
                        if not isinstance(c, dict):
                            continue
                        pk = str(c.get("plan_kind") or "")
                        lines.append(f"{labels[i]}) {human.get(pk, pk)}")
                    text = "OPÇÕES:\n" + "\n".join(lines) + "\nEscolha: A/B/C"
                    ctx2 = dict(ctx2)
                    ctx2["planned_text"] = str(text)
                elif selected_kind == PLAN_KIND_ASK_CLARIFY_V104:
                    objective_kind = "COMM_CONFIRM"
                    ctx2 = dict(ctx2)
                    ctx2["planned_text"] = "Não entendi. Você pode reformular usando um comando suportado (set/get/add/summary/end) ou descrever o objetivo?"

            # V105: continuous agency decision (goal slots -> ask -> options -> execute -> close).
            # Triggered on UNKNOWN input only to preserve deterministic outputs for legacy commands (e.g., goal:).
            if intent_id_for_plan == INTENT_UNKNOWN_V92 and not ((not bool(parse.get("parse_ok", False))) and isinstance(style_profile_updates, list) and style_profile_updates):
                active_by_id, _gdet = fold_goal_ledger_v99(list(goal_events))
                active_list = [dict(active_by_id[k]) for k in sorted(active_by_id.keys(), key=str)]
                active_list.sort(key=lambda d: (-int(d.get("priority") or 0), int(d.get("created_ts_turn_index") or 0), str(d.get("goal_id") or "")))
                active_goal = dict(active_list[0]) if active_list else {}
                active_gid = str(active_goal.get("goal_id") or "")
                # If no active goal exists, interpret the unknown user text as a new goal text (fail-closed: ask clarifying slots).
                if not active_gid:
                    gtext0 = str(user_text).strip()
                    if gtext0 and gtext0.upper() not in {"A", "B", "C"}:
                        gid0 = goal_id_v99(conversation_id=str(conversation_id), ts_turn_index=int(ut.get("turn_index") or 0), text=str(gtext0), parent_goal_id="")
                        ev0 = GoalEventV99(
                            conversation_id=str(conversation_id),
                            ts_turn_index=int(ut.get("turn_index") or 0),
                            op="GOAL_ADD",
                            goal_id=str(gid0),
                            parent_goal_id="",
                            priority=100,
                            status="active",
                            text=str(gtext0),
                            cause_type="user_intent",
                            cause_id=str(ut.get("turn_id") or ""),
                            created_step=int(step),
                            prev_event_sig=str(prev_goal_event_sig),
                        ).to_dict()
                        prev_goal_hash = append_chained_jsonl_v96(
                            goal_path,
                            {"time": deterministic_iso(step=int(step)), "step": int(step), "event": "GOAL_EVENT", "payload": dict(ev0)},
                            prev_hash=prev_goal_hash,
                        )
                        goal_events.append(dict(ev0))
                        prev_goal_event_sig = str(ev0.get("event_sig") or "")
                        goal_write_ids.append(str(gid0))
                        active_gid = str(gid0)
                        active_goal = {"goal_id": str(gid0), "parent_goal_id": "", "priority": 100, "text": str(gtext0)}

                if active_gid:
                    # Derive current goal slots/pending from agency ledger prefix.
                    areg = agency_registry_snapshot_v105(list(agency_events))
                    by_goal = areg.get("by_goal") if isinstance(areg.get("by_goal"), dict) else {}
                    stg = by_goal.get(active_gid) if isinstance(by_goal.get(active_gid), dict) else {}
                    cur_slots = stg.get("slots") if isinstance(stg.get("slots"), dict) else {}
                    pending = stg.get("pending") if isinstance(stg.get("pending"), dict) else {}
                    pending_kind = str(pending.get("kind") or "")
                    pending_slot = str(pending.get("slot") or "")
                    pending_options = pending.get("options") if isinstance(pending.get("options"), list) else []

                    required_slots = ["outcome", "constraints", "deadline"]
                    slots2 = {str(k): cur_slots.get(k) for k in sorted(cur_slots.keys(), key=str)}

                    # If we previously asked for a slot and user is on an UNKNOWN intent, treat as slot fill.
                    # Allow multi-slot fill in one message (deterministic, fail-closed): parse `k=v` / `k: v`
                    # for known slot keys in semicolon/newline-separated segments.
                    slot_updates: Dict[str, Any] = {}
                    if pending_kind == "await_slot" and pending_slot and intent_id_for_plan == INTENT_UNKNOWN_V92:
                        raw_text = str(user_text).strip()
                        if raw_text:
                            # slot aliases (accent-fold already applied by upstream parser; keep lower ascii here)
                            slot_aliases = {
                                "outcome": ["outcome", "resultado"],
                                "constraints": ["constraints", "restricoes", "restricoes", "restricao"],
                                "deadline": ["deadline", "prazo"],
                            }
                            parsed_any = False
                            parts = [p.strip() for p in raw_text.replace("\n", ";").replace(",", ";").split(";") if p.strip()]
                            for part in parts:
                                pl = part.lower().strip()
                                for slot_k in list(required_slots):
                                    for alias in slot_aliases.get(slot_k, []):
                                        a = str(alias)
                                        if not a:
                                            continue
                                        if pl.startswith(a):
                                            rest = part[len(a) :].strip()
                                            if rest.startswith("=") or rest.startswith(":"):
                                                vv = rest[1:].strip()
                                                if vv:
                                                    slot_updates[str(slot_k)] = str(vv)
                                                    slots2[str(slot_k)] = str(vv)
                                                    parsed_any = True
                                            break
                            if not parsed_any:
                                # Fallback: treat entire text as the requested slot value.
                                slot_updates[str(pending_slot)] = str(raw_text)
                                slots2[str(pending_slot)] = str(raw_text)
                            pending_kind = ""
                            pending_slot = ""
                            pending_options = []

                    # Recompute missing slots.
                    missing_slots: List[str] = []
                    for s in list(required_slots):
                        v = slots2.get(str(s))
                        if v is None or (isinstance(v, str) and not str(v).strip()):
                            missing_slots.append(str(s))

                    # If we previously proposed options, parse a choice (A/B/C) deterministically.
                    choice_label = ""
                    if pending_kind == "await_choice" and intent_id_for_plan == INTENT_UNKNOWN_V92:
                        raw = str(user_text).strip().upper()
                        if raw in {"A", "B", "C"}:
                            choice_label = str(raw)
                        elif raw.startswith("OPCAO ") and raw.split()[-1] in {"A", "B", "C"}:
                            choice_label = str(raw.split()[-1])
                        elif raw.startswith("OPTION ") and raw.split()[-1] in {"A", "B", "C"}:
                            choice_label = str(raw.split()[-1])

                    # Build agency candidates (TOP-3) deterministically (integers only).
                    candidates: List[Dict[str, Any]] = []
                    if missing_slots:
                        candidates.append(
                            {
                                "kind": AGENCY_KIND_ASK_CLARIFY_V105,
                                "score_int": 1000,
                                "reason_codes": ["missing_slots"],
                                "required_slots": [str(missing_slots[0])],
                                "goal_id": str(active_gid),
                            }
                        )
                        candidates.append({"kind": AGENCY_KIND_IDLE_V105, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(active_gid)})
                    elif pending_kind == "await_choice" and not choice_label:
                        candidates.append({"kind": AGENCY_KIND_PROPOSE_OPTIONS_V105, "score_int": 900, "reason_codes": ["awaiting_choice"], "required_slots": [], "goal_id": str(active_gid)})
                        candidates.append({"kind": AGENCY_KIND_IDLE_V105, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(active_gid)})
                    elif pending_kind == "await_choice" and choice_label:
                        candidates.append(
                            {
                                "kind": AGENCY_KIND_EXECUTE_STEP_V105,
                                "score_int": 900,
                                "reason_codes": ["choice_selected"],
                                "required_slots": [],
                                "goal_id": str(active_gid),
                                "choice_label": str(choice_label),
                            }
                        )
                        candidates.append({"kind": AGENCY_KIND_IDLE_V105, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(active_gid)})
                    else:
                        candidates.append({"kind": AGENCY_KIND_PROPOSE_OPTIONS_V105, "score_int": 800, "reason_codes": ["slots_complete"], "required_slots": [], "goal_id": str(active_gid)})
                        candidates.append({"kind": AGENCY_KIND_IDLE_V105, "score_int": 0, "reason_codes": ["fallback"], "required_slots": [], "goal_id": str(active_gid)})

                    # Deterministic ordering + selection.
                    candidates.sort(key=lambda d: (-int(d.get("score_int") or 0), str(d.get("kind") or ""), str(d.get("goal_id") or "")))
                    candidates_topk = list(candidates[:3])
                    chosen = dict(candidates_topk[0]) if candidates_topk else {"kind": AGENCY_KIND_IDLE_V105, "score_int": 0, "reason_codes": ["empty"]}
                    chosen_kind = str(chosen.get("kind") or AGENCY_KIND_IDLE_V105)
                    chosen_reason_codes = [str(x) for x in (chosen.get("reason_codes") if isinstance(chosen.get("reason_codes"), list) else []) if str(x)]
                    chosen_cid = _stable_hash_obj({"kind": str(chosen_kind), "goal_id": str(active_gid), "turn_index": int(ut.get("turn_index") or 0)})

                    # Apply chosen action to objective_kind/ctx2 deterministically (fail-closed).
                    if chosen_kind == AGENCY_KIND_ASK_CLARIFY_V105:
                        slot_name = str((chosen.get("required_slots") or [""])[0] if isinstance(chosen.get("required_slots"), list) else "")
                        objective_kind = "COMM_CONFIRM"
                        q = f"Para avançar no objetivo, informe {slot_name}."
                        ctx2 = dict(ctx2)
                        ctx2["planned_text"] = str(q)
                        pending_kind = "await_slot"
                        pending_slot = str(slot_name)
                        pending_options = []
                    elif chosen_kind == AGENCY_KIND_PROPOSE_OPTIONS_V105:
                        # Ask the user to choose among plan_engine_v104 candidates A/B/C (auditável).
                        objective_kind = "COMM_CONFIRM"
                        labels = ["A", "B", "C"]
                        human = {
                            "direct_answer": "Responder direto",
                            "ask_clarify": "Pedir clarificação",
                            "propose_steps": "Propor passos",
                            "compare_options": "Comparar opções",
                            "summarize_and_confirm": "Resumir e confirmar",
                            "refuse_safe": "Recusar com segurança",
                        }
                        lines: List[str] = []
                        opts_for_pending: List[Dict[str, Any]] = []
                        for i, c in enumerate(plan_candidates_topk[:3]):
                            if not isinstance(c, dict):
                                continue
                            pk = str(c.get("plan_kind") or "")
                            lbl = str(labels[i])
                            lines.append(f"{lbl}) {human.get(pk, pk)}")
                            opts_for_pending.append({"label": lbl, "plan_kind": pk, "candidate_id": str(c.get('candidate_id') or '')})
                        text = "OPÇÕES:\n" + "\n".join(lines) + "\nEscolha: A/B/C"
                        ctx2 = dict(ctx2)
                        ctx2["planned_text"] = str(text)
                        pending_kind = "await_choice"
                        pending_slot = ""
                        pending_options = list(opts_for_pending)
                    elif chosen_kind == AGENCY_KIND_EXECUTE_STEP_V105:
                        # Execute a deterministic "next step": for now, emit a 3-step plan and close the goal.
                        pk = ""
                        for o in pending_options:
                            if isinstance(o, dict) and str(o.get("label") or "") == str(choice_label):
                                pk = str(o.get("plan_kind") or "")
                                break
                        if not pk:
                            pk = "propose_steps"
                        objective_kind = "COMM_RESPOND"
                        goal_text = str(active_goal.get("text") or "")
                        outcome = str(slots2.get("outcome") or "")
                        constraints = str(slots2.get("constraints") or "")
                        deadline = str(slots2.get("deadline") or "")
                        steps_lines = [
                            f"PLANO: {goal_text}".strip(),
                            f"1) Resultado: {outcome}".strip(),
                            f"2) Restrições: {constraints}".strip(),
                            f"3) Prazo: {deadline}".strip(),
                        ]
                        ctx2 = dict(ctx2)
                        ctx2["planned_text"] = "\n".join([x for x in steps_lines if x])
                        # Close the goal deterministically (GOAL_DONE) with a system cause.
                        ev_done = GoalEventV99(
                            conversation_id=str(conversation_id),
                            ts_turn_index=int(ut.get("turn_index") or 0),
                            op="GOAL_DONE",
                            goal_id=str(active_gid),
                            parent_goal_id=str(active_goal.get("parent_goal_id") or ""),
                            priority=int(active_goal.get("priority") or 0),
                            status="done",
                            text=str(active_goal.get("text") or ""),
                            cause_type="system",
                            cause_id=str(ut.get("turn_id") or ""),
                            created_step=int(step),
                            prev_event_sig=str(prev_goal_event_sig),
                        ).to_dict()
                        prev_goal_hash = append_chained_jsonl_v96(
                            goal_path,
                            {"time": deterministic_iso(step=int(step)), "step": int(step), "event": "GOAL_EVENT", "payload": dict(ev_done)},
                            prev_hash=prev_goal_hash,
                        )
                        goal_events.append(dict(ev_done))
                        prev_goal_event_sig = str(ev_done.get("event_sig") or "")
                        goal_write_ids.append(str(active_gid))
                        pending_kind = ""
                        pending_slot = ""
                        pending_options = []

                    # Append agency event (WORM, inner chain) for this user turn.
                    slots_state = {
                        "slots": dict(slots2),
                        "missing_slots": list(missing_slots),
                        "pending": {"kind": str(pending_kind), "slot": str(pending_slot), "options": list(pending_options)},
                    }
                    ae = AgencyEventV105(
                        conversation_id=str(conversation_id),
                        ts_turn_index=int(ut.get("turn_index") or 0),
                        user_turn_id=str(ut.get("turn_id") or ""),
                        user_turn_index=int(ut.get("turn_index") or 0),
                        goal_id=str(active_gid),
                        plan_turn_index_ref=int(ut.get("turn_index") or 0),
                        slots_state=dict(slots_state),
                        candidates_topk=[dict(c) for c in candidates_topk if isinstance(c, dict)],
                        chosen_kind=str(chosen_kind),
                        chosen_reason_codes=list(chosen_reason_codes),
                        chosen_candidate_id=str(chosen_cid),
                        created_step=int(step),
                        prev_event_sig=str(prev_agency_event_sig),
                    ).to_dict()
                    _append_agency_event(ae)
                    agency_event_appended = True

            # objective_kind may have been overridden by agency logic inside this block; keep gating consistent.
            if str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
                progress_allowed_v107 = False

            # Append plan event (WORM) after objective kind is finalized, but only if progress is allowed.
            if bool(progress_allowed_v107):
                pe = PlanEventV104(
                    conversation_id=str(conversation_id),
                    ts_turn_index=int(ut.get("turn_index") or 0),
                    user_turn_id=str(ut.get("turn_id") or ""),
                    user_turn_index=int(ut.get("turn_index") or 0),
                    intent_id=str(intent_id_for_plan),
                    parse_sig=str(parse.get("parse_sig") or ""),
                    objective_kind=str(objective_kind),
                    objective_id=str(_objective_act_id(str(objective_kind))),
                    active_plan_state_before=dict(active_plan_state_v104),
                    active_plan_state_after=dict(active_after),
                    candidates_topk=[dict(c) for c in plan_candidates_topk if isinstance(c, dict)],
                    selected_candidate_id=str(selected_cid),
                    selected_plan_kind=str(selected_kind),
                    notes="plan_engine_v104: argmax score_total; tie-break plan_kind asc; candidate_id asc",
                    concept_hit_names=[str(x) for x in concept_hit_names_turn if isinstance(x, str) and x],
                    created_step=int(step),
                    prev_event_sig=str(prev_plan_event_sig),
                ).to_dict()
                _append_plan_event(pe)
                active_plan_state_v104 = dict(active_after)

        # Ensure every user turn has an agency event (WORM), even if it's a no-op/IDLE.
        if not bool(agency_event_appended):
            active_by_id, _gdet2 = fold_goal_ledger_v99(list(goal_events))
            active_list2 = [dict(active_by_id[k]) for k in sorted(active_by_id.keys(), key=str)]
            active_list2.sort(key=lambda d: (-int(d.get("priority") or 0), int(d.get("created_ts_turn_index") or 0), str(d.get("goal_id") or "")))
            active_goal2 = dict(active_list2[0]) if active_list2 else {}
            active_gid2 = str(active_goal2.get("goal_id") or "")
            areg2 = agency_registry_snapshot_v105(list(agency_events))
            by_goal2 = areg2.get("by_goal") if isinstance(areg2.get("by_goal"), dict) else {}
            stg2 = by_goal2.get(active_gid2) if active_gid2 and isinstance(by_goal2.get(active_gid2), dict) else {}
            slots_state2 = {
                "slots": dict(stg2.get("slots") or {}) if isinstance(stg2.get("slots"), dict) else {},
                "missing_slots": list(stg2.get("missing_slots") or []) if isinstance(stg2.get("missing_slots"), list) else [],
                "pending": dict(stg2.get("pending") or {}) if isinstance(stg2.get("pending"), dict) else {},
            }
            cand_idle = {"kind": AGENCY_KIND_IDLE_V105, "score_int": 0, "reason_codes": ["no_agency"], "required_slots": [], "goal_id": str(active_gid2)}
            ae2 = AgencyEventV105(
                conversation_id=str(conversation_id),
                ts_turn_index=int(ut.get("turn_index") or 0),
                user_turn_id=str(ut.get("turn_id") or ""),
                user_turn_index=int(ut.get("turn_index") or 0),
                goal_id=str(active_gid2),
                plan_turn_index_ref=-1,
                slots_state=dict(slots_state2),
                candidates_topk=[dict(cand_idle)],
                chosen_kind=AGENCY_KIND_IDLE_V105,
                chosen_reason_codes=["no_agency"],
                chosen_candidate_id=_stable_hash_obj({"kind": AGENCY_KIND_IDLE_V105, "turn_index": int(ut.get("turn_index") or 0)}),
                created_step=int(step),
                prev_event_sig=str(prev_agency_event_sig),
            ).to_dict()
            _append_agency_event(ae2)
            agency_event_appended = True

        expected_text, action_inputs, hint_action_id = _build_expected_and_action_inputs_v96(
            objective_kind=str(objective_kind),
            parse=dict(parse),
            vars_map=dict(vars_map),
            last_answer=last_answer,
            beliefs_by_key=dict(beliefs_active),
            ctx=dict(ctx2),
            user_text=str(user_text),
        )

        # V106: per-turn dialogue coherence metadata for dialogue ledger.
        dialogue_candidates_topk_v106: List[Dict[str, Any]] = []
        dialogue_selected_metrics_v106: Dict[str, Any] = {"coherence_score": 100, "components": {}, "flags": {}}
        dialogue_repair_action_v106: str = ""
        dialogue_chosen_cand_id_v106: str = ""

        # V107: per-turn pragmatics metadata (computed deterministically from user intent + candidate text).
        pragmatics_candidates_topk_v107: List[Dict[str, Any]] = []
        pragmatics_selected_metrics_v107: Dict[str, Any] = {"pragmatics_score": 100, "components": {}, "flags": {}, "repetition_metrics": {}}
        pragmatics_repair_action_kind_v107: str = ""

        # Response kind for discourse + style.
        intent_id0 = str(parse.get("intent_id") or "")
        response_kind = "locked"
        if str(objective_kind) in {"COMM_RESPOND", "COMM_SUMMARIZE"}:
            response_kind = "respond"
        elif str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
            response_kind = "clarify"

        # V102: style candidates (templates + critics) and deterministic selection.
        binding_status = str(ctx2.get("binding_status") or "")
        style_build = build_and_select_style_candidates_v102(
            templates=list(templates_v102),
            core_text=str(expected_text),
            response_kind=str(response_kind),
            style_profile=style_profile,
            intent_id=str(intent_id0),
            slots=dict(slots),
            binding_status=str(binding_status),
            recent_assistant_texts=[str(t.get("text") or "") for t in reversed(turns) if isinstance(t, dict) and str(t.get("role") or "") == "assistant"][:3],
            recent_template_ids=list(recent_style_template_ids)[-5:],
            seed=int(seed),
            selection_salt=str(ut.get("turn_id") or ""),
            top_k=8,
        )
        style_candidates_topk = style_build.get("candidates_topk") if isinstance(style_build.get("candidates_topk"), list) else []
        style_selected = style_build.get("selected") if isinstance(style_build.get("selected"), dict) else {}
        style_selection = style_build.get("selection") if isinstance(style_build.get("selection"), dict) else {}
        # If this intent is a deterministic renderer/introspection command, do NOT allow style overrides:
        # the assistant text must match the stable renderer output for verifiers.
        style_lock_intents = {
            INTENT_TEACH_V93,
            INTENT_EXPLAIN_V94,
            INTENT_NOTE_V96,
            INTENT_RECALL_V96,
            INTENT_FORGET_V96,
            INTENT_BELIEF_ADD_V96,
            INTENT_BELIEF_REVISE_V96,
            INTENT_BELIEF_LIST_V96,
            INTENT_BELIEF_FORGET_V96,
            INTENT_SYSTEM_V98,
            INTENT_VERSIONS_V98,
            INTENT_DOSSIER_V98,
            INTENT_EVIDENCE_ADD_V98,
            INTENT_EVIDENCE_LIST_V98,
            INTENT_WHY_V98,
            INTENT_GOAL_ADD_V99,
            INTENT_GOAL_LIST_V99,
            INTENT_GOAL_DONE_V99,
            INTENT_GOAL_NEXT_V99,
            INTENT_GOAL_AUTO_V99,
            INTENT_DISCOURSE_V100,
            INTENT_WHY_REF_V100,
            INTENT_BINDINGS_LIST_V101,
            INTENT_EXPLAIN_BINDING_V101,
            INTENT_TRACE_REF_V101,
            INTENT_PLAN_CREATE_V101,
            INTENT_PLAN_SHORTEN_REF_V101,
            INTENT_GOAL_PRIORITY_HIGH_REF_V101,
            INTENT_STYLE_PROFILE_V102,
            INTENT_TEMPLATES_V102,
            INTENT_EXPLAIN_STYLE_V102,
            INTENT_TRACE_STYLE_V102,
            INTENT_TEACH_CONCEPT_V103,
            INTENT_CONCEPTS_V103,
            INTENT_EXPLAIN_CONCEPT_V103,
            INTENT_TRACE_CONCEPTS_V103,
            INTENT_PLANS_V104,
            INTENT_EXPLAIN_PLAN_V104,
            INTENT_TRACE_PLANS_V104,
            INTENT_AGENCY_V105,
            INTENT_EXPLAIN_AGENCY_V105,
            INTENT_TRACE_AGENCY_V105,
            INTENT_DIALOGUE_V106,
            INTENT_EXPLAIN_DIALOGUE_V106,
            INTENT_TRACE_DIALOGUE_V106,
            INTENT_PRAGMATICS_V107,
            INTENT_EXPLAIN_PRAGMATICS_V107,
            INTENT_TRACE_PRAGMATICS_V107,
            INTENT_FLOW_V108,
            INTENT_EXPLAIN_FLOW_V108,
            INTENT_TRACE_FLOW_V108,
        }
        style_allow_override = str(objective_kind) in {"COMM_RESPOND", "COMM_SUMMARIZE", "COMM_CONFIRM"} and str(intent_id0) not in set(
            [str(x) for x in style_lock_intents]
        )

        # V106/V107/V108: evaluate coherence+pragmatics+flow for top candidates and select
        # deterministically using DSC (V108).
        style_selected_v106: Dict[str, Any] = dict(style_selected)
        metrics_by_cid_v106: Dict[str, Dict[str, Any]] = {}
        prag_by_cid_v107: Dict[str, Dict[str, Any]] = {}
        flow_by_cid_v108: Dict[str, Dict[str, Any]] = {}
        dialogue_candidates_topk_v106: List[Dict[str, Any]] = []
        pragmatics_candidates_topk_v107: List[Dict[str, Any]] = []
        flow_candidates_topk_v108: List[Dict[str, Any]] = []
        evaluated_candidates_v108: List[Dict[str, Any]] = []
        recent_assistant_texts_v106 = [str(t.get("text") or "") for t in reversed(turns) if isinstance(t, dict) and str(t.get("role") or "") == "assistant"][:3]
        recent_topics_v106 = discourse_state.get("active_topics") if isinstance(discourse_state.get("active_topics"), list) else []
        topic_now_v106 = "dsl" if str(intent_id0) in {INTENT_SET_V92, INTENT_GET_V92, INTENT_ADD_V92, INTENT_SUMMARY_V92} else "other"
        style_prof_dict_v107 = style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict()
        regime_next_guess_v107 = decide_regime_next_v107(
            regime_prev=str(pragmatics_regime_v107),
            user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
            pending_questions_active=list(pragmatics_pending_questions_active),
            repair_action_kind="",
        )
        recent_phrase_hashes_v108 = flow_memory_v108.get("recent_phrases_hashes") if isinstance(flow_memory_v108.get("recent_phrases_hashes"), list) else []
        recent_discourse_acts_v108 = flow_memory_v108.get("recent_discourse_acts") if isinstance(flow_memory_v108.get("recent_discourse_acts"), list) else []

        for c in list(style_candidates_topk)[:3]:
            if not isinstance(c, dict):
                continue
            cid = str(c.get("candidate_id") or "")
            ctext = str(c.get("text") or "")
            m = compute_metrics_v106(
                candidate_text=str(ctext),
                user_text=str(user_text),
                objective_kind=str(objective_kind),
                binding_status=str(binding_status),
                recent_assistant_texts=list(recent_assistant_texts_v106),
                recent_topics=[str(x) for x in list(recent_topics_v106) if isinstance(x, str) and x],
                current_topic=str(topic_now_v106),
            )
            metrics_by_cid_v106[str(cid)] = dict(m)

            # V107 pragmatics score for the candidate (deterministic; no ML).
            assistant_act_c = infer_assistant_intent_act_v107(objective_kind=str(objective_kind), planned_text=str(ctext), repair_action="")
            pm = compute_pragmatics_metrics_v107(
                user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
                assistant_intent_act=dict(assistant_act_c) if isinstance(assistant_act_c, dict) else {},
                candidate_text=str(ctext),
                style_profile=dict(style_prof_dict_v107),
                pending_questions_active=list(pragmatics_pending_questions_active),
                recent_assistant_texts=list(recent_assistant_texts_v106),
                regime_prev=str(pragmatics_regime_v107),
                regime_next=str(regime_next_guess_v107),
            )
            prag_by_cid_v107[str(cid)] = dict(pm)

            fm = compute_flow_metrics_v108(
                candidate_text=str(ctext),
                user_text=str(user_text),
                objective_kind=str(objective_kind),
                binding_status=str(binding_status),
                user_intent_act_v107=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
                pending_questions_active=list(pragmatics_pending_questions_active),
                # Avoid spurious ABRUPT_TOPIC_SHIFT on regime transitions: V108 flow focuses on
                # repair gating (references/pending) and repetition, not regime labels.
                regime_prev=str(pragmatics_regime_v107),
                regime_next=str(pragmatics_regime_v107),
                style_profile_dict=dict(style_prof_dict_v107),
                recent_assistant_texts=list(recent_assistant_texts_v106),
                recent_phrase_hashes=[str(x) for x in list(recent_phrase_hashes_v108) if isinstance(x, str) and x],
                recent_discourse_acts=[str(x) for x in list(recent_discourse_acts_v108) if isinstance(x, str) and x],
                current_user_turn_index=int(ut.get("turn_index") or 0),
            )
            flow_by_cid_v108[str(cid)] = dict(fm)

            c_score = min(int(m.get("coherence_score") or 0), int(pm.get("pragmatics_score") or 0))
            flags_digest = _stable_hash_obj({"v106": str(m.get("flags_digest") or ""), "v107": str(pm.get("flags_digest") or "")})
            dialogue_candidates_topk_v106.append(
                {
                    "cand_id": str(cid),
                    "text_sha256": str(c.get("text_sha256") or ""),
                    "len_chars": int(m.get("len_chars") or 0),
                    "len_tokens": int(m.get("len_tokens") or 0),
                    "score": int(c_score),
                    "flags_digest": str(flags_digest),
                }
            )
            pragmatics_candidates_topk_v107.append(
                {
                    "candidate_hash": sha256_hex(str(ctext).encode("utf-8")),
                    "coherence_score_v106": int(m.get("coherence_score") or 0),
                    "pragmatics_score_v107": int(pm.get("pragmatics_score") or 0),
                    "flags_v106": dict(m.get("flags") or {}) if isinstance(m.get("flags"), dict) else {},
                    "flags_v107": dict(pm.get("flags") or {}) if isinstance(pm.get("flags"), dict) else {},
                    "length_chars": int(pm.get("len_chars") or 0),
                    "repetition_metrics": dict(pm.get("repetition_metrics") or {}) if isinstance(pm.get("repetition_metrics"), dict) else {},
                }
            )

            flow_candidates_topk_v108.append(
                {
                    "candidate_hash": sha256_hex(str(ctext).encode("utf-8")),
                    "coherence_score_v106": int(m.get("coherence_score") or 0),
                    "pragmatics_score_v107": int(pm.get("pragmatics_score") or 0),
                    "flow_score_v108": int(fm.get("flow_score_v108") or 0),
                    "flow_flags_v108": dict(fm.get("flags") or {}) if isinstance(fm.get("flags"), dict) else {},
                    "repetition_score": int(fm.get("repetition_score") or 0),
                    "chosen": False,
                }
            )

            score_base = min(int(m.get("coherence_score") or 0), int(pm.get("pragmatics_score") or 0), int(fm.get("flow_score_v108") or 0))
            evaluated_candidates_v108.append(
                {
                    "candidate_id": str(cid),
                    "candidate_hash": sha256_hex(str(ctext).encode("utf-8")),
                    "text_sha256": str(c.get("text_sha256") or ""),
                    "len_tokens": int(m.get("len_tokens") or 0),
                    "len_chars": int(m.get("len_chars") or 0),
                    "score_base": int(score_base),
                    "coherence_score_v106": int(m.get("coherence_score") or 0),
                    "pragmatics_score_v107": int(pm.get("pragmatics_score") or 0),
                    "flow_score_v108": int(fm.get("flow_score_v108") or 0),
                    "flow_flags_v108": dict(fm.get("flags") or {}) if isinstance(fm.get("flags"), dict) else {},
                    "repetition_score": int(fm.get("repetition_score") or 0),
                }
            )

        chosen_eval_v108 = select_candidate_dsc_v108(
            evaluated_candidates=list(evaluated_candidates_v108),
            seed=int(seed),
            prev_flow_event_sig=str(prev_flow_event_sig),
            selection_salt=str(ut.get("turn_id") or ""),
        )
        chosen_cand_id_v106 = str(chosen_eval_v108.get("candidate_id") or chosen_eval_v108.get("cand_id") or str(style_selected.get("candidate_id") or ""))
        cand_by_id_v106: Dict[str, Dict[str, Any]] = {}
        for c in style_candidates_topk:
            if isinstance(c, dict) and str(c.get("candidate_id") or ""):
                cand_by_id_v106[str(c.get("candidate_id") or "")] = dict(c)
        if str(chosen_cand_id_v106) and str(chosen_cand_id_v106) in cand_by_id_v106:
            style_selected_v106 = dict(cand_by_id_v106.get(str(chosen_cand_id_v106)) or {})
        dialogue_chosen_cand_id_v106 = str(chosen_cand_id_v106)
        dialogue_selected_metrics_v106 = dict(metrics_by_cid_v106.get(str(chosen_cand_id_v106)) or {})
        pragmatics_selected_metrics_v107 = dict(prag_by_cid_v107.get(str(chosen_cand_id_v106)) or {})
        flow_selected_metrics_v108 = dict(flow_by_cid_v108.get(str(chosen_cand_id_v106)) or {})
        for cc in flow_candidates_topk_v108:
            if isinstance(cc, dict) and str(cc.get("candidate_hash") or "") == sha256_hex(str(style_selected_v106.get("text") or "").encode("utf-8")):
                cc["chosen"] = True
        if not dialogue_selected_metrics_v106:
            dialogue_selected_metrics_v106 = compute_metrics_v106(
                candidate_text=str(style_selected_v106.get("text") or expected_text),
                user_text=str(user_text),
                objective_kind=str(objective_kind),
                binding_status=str(binding_status),
                recent_assistant_texts=list(recent_assistant_texts_v106),
                recent_topics=[str(x) for x in list(recent_topics_v106) if isinstance(x, str) and x],
                current_topic=str(topic_now_v106),
            )

        # Survival rule (2 consecutive low turns): if needed, force a repair path.
        last_score_v106 = 100
        if dialogue_events:
            try:
                last_score_v106 = int(dialogue_events[-1].get("coherence_score") or 0)
            except Exception:
                last_score_v106 = 0
        active_by_id_d, _gdet_d = fold_goal_ledger_v99(list(goal_events))
        active_list_d = [dict(active_by_id_d[k]) for k in sorted(active_by_id_d.keys(), key=str) if isinstance(active_by_id_d.get(k), dict)]
        active_list_d.sort(key=lambda d: (-int(d.get("priority") or 0), int(d.get("created_ts_turn_index") or 0), str(d.get("goal_id") or "")))
        active_goal_text_v106 = str(active_list_d[0].get("text") or "") if active_list_d else ""
        dialogue_repair_action_v106 = decide_repair_action_v106(
            prev_coherence_scores=[int(last_score_v106)],
            selected_score=int(dialogue_selected_metrics_v106.get("coherence_score") or 0),
            binding_status=str(binding_status),
            has_active_goal=bool(active_list_d),
            objective_kind=str(objective_kind),
        )
        force_repair_v106 = (
            int(last_score_v106) < int(COHERENCE_THRESH_V106)
            and int(dialogue_selected_metrics_v106.get("coherence_score") or 0) < int(COHERENCE_THRESH_V106)
            and str(objective_kind) not in {"COMM_ASK_CLARIFY", "COMM_CONFIRM", "COMM_SUMMARIZE", "COMM_END"}
            and str(intent_id0) not in set([str(x) for x in style_lock_intents])
            and bool(dialogue_repair_action_v106)
        )
        if bool(force_repair_v106):
            objective_kind = "COMM_CONFIRM"
            expected_text = repair_text_v106(
                repair_action=str(dialogue_repair_action_v106),
                user_text=str(user_text),
                active_goal_text=str(active_goal_text_v106),
            )
            action_inputs = {"text": str(expected_text)}
            hint_action_id = "concept_v90_emit_text_v0"
            response_kind = "clarify"
            style_build = build_and_select_style_candidates_v102(
                templates=list(templates_v102),
                core_text=str(expected_text),
                response_kind=str(response_kind),
                style_profile=style_profile,
                intent_id=str(intent_id0),
                slots=dict(slots),
                binding_status=str(binding_status),
                recent_assistant_texts=list(recent_assistant_texts_v106),
                recent_template_ids=list(recent_style_template_ids)[-5:],
                seed=int(seed),
                selection_salt=str(ut.get("turn_id") or ""),
                top_k=8,
            )
            style_candidates_topk = style_build.get("candidates_topk") if isinstance(style_build.get("candidates_topk"), list) else []
            style_selected = style_build.get("selected") if isinstance(style_build.get("selected"), dict) else {}
            style_selection = style_build.get("selection") if isinstance(style_build.get("selection"), dict) else {}
            # Recompute combined coherence+pragmatics+flow selection under repair (DSC).
            metrics_by_cid_v106 = {}
            metrics_by_cid_v107 = {}
            flow_by_cid_v108 = {}
            dialogue_candidates_topk_v106 = []
            pragmatics_candidates_topk_v107 = []
            flow_candidates_topk_v108 = []
            evaluated_candidates_v108 = []
            regime_next_repair_v107 = decide_regime_next_v107(
                regime_prev=str(pragmatics_regime_v107),
                user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
                pending_questions_active=list(pragmatics_pending_questions_active),
                repair_action_kind=str(dialogue_repair_action_v106 or ""),
            )
            for c in list(style_candidates_topk)[:3]:
                if not isinstance(c, dict):
                    continue
                cid = str(c.get("candidate_id") or "")
                m = compute_metrics_v106(
                    candidate_text=str(c.get("text") or ""),
                    user_text=str(user_text),
                    objective_kind=str(objective_kind),
                    binding_status=str(binding_status),
                    recent_assistant_texts=list(recent_assistant_texts_v106),
                    recent_topics=[str(x) for x in list(recent_topics_v106) if isinstance(x, str) and x],
                    current_topic=str(topic_now_v106),
                )
                ctext = str(c.get("text") or "")
                assistant_act = infer_assistant_intent_act_v107(
                    objective_kind=str(objective_kind),
                    planned_text=str(ctext),
                    repair_action=str(dialogue_repair_action_v106 or ""),
                )
                pm = compute_pragmatics_metrics_v107(
                    user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
                    assistant_intent_act=dict(assistant_act),
                    candidate_text=str(ctext),
                    style_profile=style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict(),
                    pending_questions_active=list(pragmatics_pending_questions_active),
                    recent_assistant_texts=list(recent_assistant_texts_v106),
                    regime_prev=str(pragmatics_regime_v107),
                    regime_next=str(regime_next_repair_v107),
                )
                fm = compute_flow_metrics_v108(
                    candidate_text=str(ctext),
                    user_text=str(user_text),
                    objective_kind=str(objective_kind),
                    binding_status=str(binding_status),
                    user_intent_act_v107=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
                    pending_questions_active=list(pragmatics_pending_questions_active),
                    regime_prev=str(pragmatics_regime_v107),
                    regime_next=str(pragmatics_regime_v107),
                    style_profile_dict=style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict(),
                    recent_assistant_texts=list(recent_assistant_texts_v106),
                    recent_phrase_hashes=[str(x) for x in list(recent_phrase_hashes_v108) if isinstance(x, str) and x],
                    recent_discourse_acts=[str(x) for x in list(recent_discourse_acts_v108) if isinstance(x, str) and x],
                    current_user_turn_index=int(ut.get("turn_index") or 0),
                )
                metrics_by_cid_v106[str(cid)] = dict(m)
                metrics_by_cid_v107[str(cid)] = dict(pm)
                flow_by_cid_v108[str(cid)] = dict(fm)
                c_score = min(int(m.get("coherence_score") or 0), int(pm.get("pragmatics_score") or 0))
                flags_digest = _stable_hash_obj({"v106": str(m.get("flags_digest") or ""), "v107": str(pm.get("flags_digest") or "")})
                dialogue_candidates_topk_v106.append(
                    {
                        "cand_id": str(cid),
                        "text_sha256": str(c.get("text_sha256") or ""),
                        "len_chars": int(m.get("len_chars") or 0),
                        "len_tokens": int(m.get("len_tokens") or 0),
                        "score": int(c_score),
                        "flags_digest": str(flags_digest),
                    }
                )
                pragmatics_candidates_topk_v107.append(
                    {
                        "candidate_hash": sha256_hex(str(ctext).encode("utf-8")),
                        "coherence_score_v106": int(m.get("coherence_score") or 0),
                        "pragmatics_score_v107": int(pm.get("pragmatics_score") or 0),
                        "flags_v106": dict(m.get("flags") or {}) if isinstance(m.get("flags"), dict) else {},
                        "flags_v107": dict(pm.get("flags") or {}) if isinstance(pm.get("flags"), dict) else {},
                        "length_chars": int(pm.get("len_chars") or 0),
                        "repetition_metrics": dict(pm.get("repetition_metrics") or {}) if isinstance(pm.get("repetition_metrics"), dict) else {},
                    }
                )
                flow_candidates_topk_v108.append(
                    {
                        "candidate_hash": sha256_hex(str(ctext).encode("utf-8")),
                        "coherence_score_v106": int(m.get("coherence_score") or 0),
                        "pragmatics_score_v107": int(pm.get("pragmatics_score") or 0),
                        "flow_score_v108": int(fm.get("flow_score_v108") or 0),
                        "flow_flags_v108": dict(fm.get("flags") or {}) if isinstance(fm.get("flags"), dict) else {},
                        "repetition_score": int(fm.get("repetition_score") or 0),
                        "chosen": False,
                    }
                )
                score_base = min(int(m.get("coherence_score") or 0), int(pm.get("pragmatics_score") or 0), int(fm.get("flow_score_v108") or 0))
                evaluated_candidates_v108.append(
                    {
                        "candidate_id": str(cid),
                        "candidate_hash": sha256_hex(str(ctext).encode("utf-8")),
                        "text_sha256": str(c.get("text_sha256") or ""),
                        "len_tokens": int(m.get("len_tokens") or 0),
                        "len_chars": int(m.get("len_chars") or 0),
                        "score_base": int(score_base),
                    }
                )
            chosen_eval_v108 = select_candidate_dsc_v108(
                evaluated_candidates=list(evaluated_candidates_v108),
                seed=int(seed),
                prev_flow_event_sig=str(prev_flow_event_sig),
                selection_salt=str(ut.get("turn_id") or ""),
            )
            chosen_cand_id_v106 = str(chosen_eval_v108.get("candidate_id") or chosen_eval_v108.get("cand_id") or str(style_selected.get("candidate_id") or ""))
            cand_by_id_v106 = {str(c.get("candidate_id") or ""): dict(c) for c in style_candidates_topk if isinstance(c, dict) and str(c.get("candidate_id") or "")}
            style_selected_v106 = dict(cand_by_id_v106.get(str(chosen_cand_id_v106)) or dict(style_selected))
            dialogue_chosen_cand_id_v106 = str(chosen_cand_id_v106)
            dialogue_selected_metrics_v106 = dict(metrics_by_cid_v106.get(str(chosen_cand_id_v106)) or {})
            pragmatics_selected_metrics_v107 = dict(metrics_by_cid_v107.get(str(chosen_cand_id_v106)) or {})
            flow_selected_metrics_v108 = dict(flow_by_cid_v108.get(str(chosen_cand_id_v106)) or {})
            for cc in flow_candidates_topk_v108:
                if isinstance(cc, dict) and str(cc.get("candidate_hash") or "") == sha256_hex(str(style_selected_v106.get("text") or "").encode("utf-8")):
                    cc["chosen"] = True
            style_allow_override = str(objective_kind) in {"COMM_RESPOND", "COMM_SUMMARIZE", "COMM_CONFIRM"} and str(intent_id0) not in set(
                [str(x) for x in style_lock_intents]
            )

        if style_allow_override and str(style_selected_v106.get("text") or ""):
            if str(style_selected_v106.get("candidate_id") or "") and str(style_selected_v106.get("candidate_id") or "") != str(style_selected.get("candidate_id") or ""):
                style_selection = {"method": "locked", "margin": 0.0, "soft_index": 0}
            style_selected = dict(style_selected_v106)
            expected_text = str(style_selected.get("text") or "")
            action_inputs = {"text": str(expected_text)}
            hint_action_id = "concept_v90_emit_text_v0"
        else:
            # Ensure selected matches base output deterministically.
            style_selection = {"method": "locked", "margin": 0.0, "soft_index": 0}
            matched = None
            for c in style_candidates_topk:
                if isinstance(c, dict) and str(c.get("text") or "") == str(expected_text):
                    matched = dict(c)
                    break
            style_selected = dict(matched) if isinstance(matched, dict) else {"candidate_id": "", "template_id": "tmpl_v102_direct_medium_v0", "text": str(expected_text), "text_sha256": "", "critics": {"ok": True}}

        # Finalize selected metrics for the actually emitted text.
        dialogue_chosen_cand_id_v106 = str(style_selected.get("candidate_id") or dialogue_chosen_cand_id_v106)
        dialogue_selected_metrics_v106 = compute_metrics_v106(
            candidate_text=str(expected_text),
            user_text=str(user_text),
            objective_kind=str(objective_kind),
            binding_status=str(binding_status),
            recent_assistant_texts=list(recent_assistant_texts_v106),
            recent_topics=[str(x) for x in list(recent_topics_v106) if isinstance(x, str) and x],
            current_topic=str(topic_now_v106),
        )

        # V107: finalize pragmatics metrics for the selected assistant text (used for pragmatics ledger + survival checks).
        style_prof_dict_v107 = style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict()
        repair_kind0_v107 = str(dialogue_repair_action_v106 or "")
        regime_next0_v107 = decide_regime_next_v107(
            regime_prev=str(pragmatics_regime_v107),
            user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
            pending_questions_active=list(pragmatics_pending_questions_active),
            repair_action_kind="",
        )
        assistant_act0_v107 = infer_assistant_intent_act_v107(
            objective_kind=str(objective_kind),
            planned_text=str(expected_text),
            repair_action=str(repair_kind0_v107),
        )
        pm0_v107 = compute_pragmatics_metrics_v107(
            user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
            assistant_intent_act=dict(assistant_act0_v107),
            candidate_text=str(expected_text),
            style_profile=dict(style_prof_dict_v107),
            pending_questions_active=list(pragmatics_pending_questions_active),
            recent_assistant_texts=list(recent_assistant_texts_v106),
            regime_prev=str(pragmatics_regime_v107),
            regime_next=str(regime_next0_v107),
        )
        pragmatics_repair_action_kind_v107 = decide_repair_action_v107(
            pragmatics_metrics=dict(pm0_v107),
            user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
        )
        repair_kind_v107 = str(dialogue_repair_action_v106 or pragmatics_repair_action_kind_v107 or "")
        regime_next_v107 = decide_regime_next_v107(
            regime_prev=str(pragmatics_regime_v107),
            user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
            pending_questions_active=list(pragmatics_pending_questions_active),
            repair_action_kind=str(repair_kind_v107),
        )
        assistant_act_v107 = infer_assistant_intent_act_v107(
            objective_kind=str(objective_kind),
            planned_text=str(expected_text),
            repair_action=str(repair_kind_v107),
        )
        pragmatics_selected_metrics_v107 = compute_pragmatics_metrics_v107(
            user_intent_act=dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {},
            assistant_intent_act=dict(assistant_act_v107),
            candidate_text=str(expected_text),
            style_profile=dict(style_prof_dict_v107),
            pending_questions_active=list(pragmatics_pending_questions_active),
            recent_assistant_texts=list(recent_assistant_texts_v106),
            regime_prev=str(pragmatics_regime_v107),
            regime_next=str(regime_next_v107),
        )
        pragmatics_selected_metrics_v107_for_event = dict(pragmatics_selected_metrics_v107)
        pragmatics_regime_next_v107_for_event = str(regime_next_v107)
        pragmatics_assistant_intent_act_v107_for_event = dict(assistant_act_v107)

        sel_tid = str(style_selected.get("template_id") or "")
        if sel_tid:
            recent_style_template_ids.append(str(sel_tid))
            recent_style_template_ids = list(recent_style_template_ids[-10:])

        # V100: discourse candidates (>=3) selected deterministically from the final expected text.
        # Disable wrappers so the style decision remains the single source of truth for wording/format.
        discourse_base_text = str(expected_text)
        discourse_candidates_topk = generate_text_candidates_v100(
            base_text=str(discourse_base_text),
            response_kind=str(response_kind),
            allow_wrappers=False,
        )
        discourse_selected = dict(discourse_candidates_topk[0]) if discourse_candidates_topk else {}

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
            # V107: if a deterministic hint action exists but isn't declared in supports(G),
            # include it explicitly as a safe fallback (still audited in plan_attempts).
            if hint and hint not in set(ranked_ids) and store.get(str(hint)) is not None:
                ranked = [
                    (
                        str(hint),
                        SupportClaimV89(goal_id=str(goal_id), prior_success=1.0, prior_strength=1, prior_cost=0.0, note="hint_action_v107"),
                        1.0,
                        0.0,
                    )
                ] + list(ranked)
                ranked_ids = [str(hint)] + list(ranked_ids)
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

        # V102: style ledger (templates + critics + selection; inner sig-chain + outer jsonl chain).
        style_profile_before_dict = style_profile_before.to_dict() if isinstance(style_profile_before, StyleProfileV102) else default_style_profile_v102().to_dict()
        style_profile_after_dict = style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict()
        sel_ok = bool(((style_selected.get("critics") or {}).get("ok")) if isinstance(style_selected.get("critics"), dict) else False)
        style_cause_ids = {
            "cause_goal_ids": sorted(set(list(goal_read_ids) + list(goal_write_ids))),
            "cause_belief_ids": sorted(set(list(belief_read_ids))),
            "cause_evidence_ids": sorted(set(list(evidence_read_ids) + list(evidence_write_ids))),
        }
        se = StyleEventV102(
            conversation_id=str(conversation_id),
            turn_id=str(at.get("turn_id") or ""),
            turn_index=int(at.get("turn_index") or 0),
            event_kind="STYLE_CHOSEN",
            style_profile_before=dict(style_profile_before_dict),
            style_profile_after=dict(style_profile_after_dict),
            candidates_topk=[dict(x) for x in style_candidates_topk if isinstance(x, dict)],
            selected_candidate_id=str(style_selected.get("candidate_id") or ""),
            selected_template_id=str(style_selected.get("template_id") or ""),
            selected_ok=bool(sel_ok),
            selection=dict(style_selection) if isinstance(style_selection, dict) else {},
            cause_ids=dict(style_cause_ids),
            created_step=int(at.get("created_step") or 0),
            prev_event_sig=str(prev_style_event_sig),
        ).to_dict()
        prev_style_hash = append_chained_jsonl_v96(
            style_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "STYLE_EVENT", "payload": dict(se)},
            prev_hash=prev_style_hash,
        )
        style_events.append(dict(se))
        prev_style_event_sig = str(se.get("event_sig") or "")

        # V106: dialogue coherence survival ledger (per-turn, WORM + inner sig chain).
        de = DialogueEventV106(
            conversation_id=str(conversation_id),
            user_turn_id=str(ut.get("turn_id") or ""),
            user_turn_index=int(ut.get("turn_index") or 0),
            assistant_turn_id=str(at.get("turn_id") or ""),
            assistant_turn_index=int(at.get("turn_index") or 0),
            coherence_score=int(dialogue_selected_metrics_v106.get("coherence_score") or 0),
            components=dict(dialogue_selected_metrics_v106.get("components") or {})
            if isinstance(dialogue_selected_metrics_v106.get("components"), dict)
            else {},
            flags=dict(dialogue_selected_metrics_v106.get("flags") or {})
            if isinstance(dialogue_selected_metrics_v106.get("flags"), dict)
            else {},
            candidates_topk=[dict(x) for x in dialogue_candidates_topk_v106 if isinstance(x, dict)],
            chosen_cand_id=str(dialogue_chosen_cand_id_v106 or style_selected.get("candidate_id") or ""),
            repair_action=str(dialogue_repair_action_v106 or ""),
            created_step=int(at.get("created_step") or 0),
            prev_event_sig=str(prev_dialogue_event_sig),
        ).to_dict()
        _append_dialogue_event(de)

        # V107: pragmatics + dialogue-state survival ledger (per user turn, WORM + inner sig chain).
        user_intent_act_for_ev = dict(user_intent_act_v107) if isinstance(user_intent_act_v107, dict) else {}
        assistant_intent_act_for_ev = dict(pragmatics_assistant_intent_act_v107_for_event) if isinstance(pragmatics_assistant_intent_act_v107_for_event, dict) else {}
        repair_kind_for_ev = str(dialogue_repair_action_v106 or pragmatics_repair_action_kind_v107 or "")

        # Pending questions delta (minimal, deterministic): open on clarify/confirm; close on respond/end/summarize.
        pq_opened: List[str] = []
        pq_closed: List[str] = []
        if str(objective_kind) in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
            qid = f"pq_v107_{int(ut.get('turn_index') or 0)}"
            if str(qid) and str(qid) not in set([str(x) for x in pragmatics_pending_questions_active if str(x)]):
                pragmatics_pending_questions_active.append(str(qid))
                pq_opened.append(str(qid))
        else:
            if pragmatics_pending_questions_active:
                pq_closed = [str(x) for x in pragmatics_pending_questions_active if str(x)]
                pragmatics_pending_questions_active = []

        # Topic stack delta: push new regime on regime change (minimal).
        topic_push: List[str] = []
        topic_pop: List[str] = []
        regime_prev_for_ev = str(pragmatics_regime_v107)
        regime_next_for_ev = str(pragmatics_regime_next_v107_for_event or regime_prev_for_ev)
        if regime_next_for_ev and regime_next_for_ev != regime_prev_for_ev:
            pragmatics_topic_stack_v107.append(str(regime_next_for_ev))
            topic_push.append(str(regime_next_for_ev))

        # Commitments (minimal): record unsafe commitment attempt as "violated".
        violated: List[str] = []
        if str((user_intent_act_for_ev.get("kind") or "")) == "PROMISE_ATTEMPT":
            violated = [f"unsafe_commitment_attempt_{int(ut.get('turn_index') or 0)}"]

        # Selected pragmatics metrics should align to the emitted text (assistant_text).
        selected_pm = compute_pragmatics_metrics_v107(
            user_intent_act=dict(user_intent_act_for_ev),
            assistant_intent_act=dict(assistant_intent_act_for_ev),
            candidate_text=str(assistant_text),
            style_profile=style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict(),
            pending_questions_active=list(pragmatics_pending_questions_active),
            recent_assistant_texts=list(recent_assistant_texts_v106),
            regime_prev=str(regime_prev_for_ev),
            regime_next=str(regime_next_for_ev),
        )
        pragmatics_selected_metrics_v107_for_event = dict(selected_pm)

        progress_blocked_v107 = bool(not progress_allowed_v107)
        blocked_ledgers_v107 = ["goal", "plan", "agency"] if progress_blocked_v107 else []
        repair_action_obj_v107: Dict[str, Any] = {}
        if str(repair_kind_for_ev):
            flags = selected_pm.get("flags") if isinstance(selected_pm.get("flags"), dict) else {}
            crit = [str(k) for k in sorted(flags.keys(), key=str) if bool(flags.get(k, False)) and str(k)]
            repair_action_obj_v107 = {"kind": str(repair_kind_for_ev), "reason_flags": list(crit)}

        pe_v107 = PragmaticsEventV107(
            conversation_id=str(conversation_id),
            event_index=int(pragmatics_event_index_v107),
            user_turn_id=str(ut.get("turn_id") or ""),
            user_turn_index=int(ut.get("turn_index") or 0),
            assistant_turn_id=str(at.get("turn_id") or ""),
            assistant_turn_index=int(at.get("turn_index") or 0),
            user_text_hash=str(text_sig_v96(str(ut.get("text") or ""))),
            user_intent_act=dict(user_intent_act_for_ev),
            assistant_intent_act=dict(assistant_intent_act_for_ev),
            dialogue_regime={"prev": str(regime_prev_for_ev), "next": str(regime_next_for_ev)},
            dialogue_state_delta={"updates": {}, "adds": [], "removes": []},
            pending_questions={"opened": list(pq_opened), "closed": list(pq_closed)},
            commitments={"opened": [], "closed": [], "violated": list(violated)},
            topic_stack={"push": list(topic_push), "pop": list(topic_pop), "top_after": str(pragmatics_topic_stack_v107[-1] if pragmatics_topic_stack_v107 else "")},
            candidates=[dict(x) for x in pragmatics_candidates_topk_v107 if isinstance(x, dict)],
            chosen_candidate_hash=sha256_hex(str(assistant_text).encode("utf-8")),
            pragmatics_score=int(selected_pm.get("pragmatics_score") or 0),
            flags_v107=dict(selected_pm.get("flags") or {}) if isinstance(selected_pm.get("flags"), dict) else {},
            repair_action=dict(repair_action_obj_v107),
            progress_blocked=bool(progress_blocked_v107),
            blocked_ledgers=list(blocked_ledgers_v107),
            created_step=int(at.get("created_step") or 0),
            prev_event_sig=str(prev_pragmatics_event_sig),
        ).to_dict()
        _append_pragmatics_event(pe_v107)
        pragmatics_regime_v107 = str(regime_next_for_ev)
        pragmatics_event_index_v107 += 1

        # V108: FLOW ledger (fluency/flow/memory survival). This is an assistant-side event per user turn.
        flow_sel = compute_flow_metrics_v108(
            candidate_text=str(assistant_text),
            user_text=str(user_text),
            objective_kind=str(objective_kind),
            binding_status=str(binding_status),
            user_intent_act_v107=dict(user_intent_act_for_ev),
            pending_questions_active=list(pragmatics_pending_questions_active),
            regime_prev=str(pragmatics_regime_v107),
            regime_next=str(pragmatics_regime_v107),
            style_profile_dict=style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict(),
            recent_assistant_texts=list(recent_assistant_texts_v106),
            recent_phrase_hashes=[str(x) for x in (flow_memory_v108.get("recent_phrases_hashes") if isinstance(flow_memory_v108.get("recent_phrases_hashes"), list) else []) if isinstance(x, str) and x],
            recent_discourse_acts=[str(x) for x in (flow_memory_v108.get("recent_discourse_acts") if isinstance(flow_memory_v108.get("recent_discourse_acts"), list) else []) if isinstance(x, str) and x],
            current_user_turn_index=int(ut.get("turn_index") or 0),
        )
        chosen_hash_v108 = sha256_hex(str(assistant_text).encode("utf-8"))
        flow_cands_for_ev: List[Dict[str, Any]] = []
        for cc in list(flow_candidates_topk_v108)[:3]:
            if not isinstance(cc, dict):
                continue
            row = dict(cc)
            row["chosen"] = str(row.get("candidate_hash") or "") == str(chosen_hash_v108)
            flow_cands_for_ev.append(row)

        # FLOW memory mirror: bridge pragmatics (topic/pending/commitments) into FLOW memory.
        flow_memory_v108["topic_stack"] = list(pragmatics_topic_stack_v107)
        flow_memory_v108["pending_questions"] = list(pragmatics_pending_questions_active)
        flow_memory_v108["commitments"] = list(pragmatics_commitments_active)
        flow_memory_v108["verbosity_mode_v108"] = str(flow_sel.get("verbosity_mode_v108") or "")
        flow_memory_v108["tone_mode_v108"] = str(flow_sel.get("tone_mode_v108") or "")

        # Update hierarchical episodic memory (minimal, deterministic).
        top_topic = str(pragmatics_topic_stack_v107[-1] if pragmatics_topic_stack_v107 else pragmatics_regime_v107)
        episodes = flow_memory_v108.get("episodic_memory") if isinstance(flow_memory_v108.get("episodic_memory"), list) else []
        episodes2 = [dict(e) for e in episodes if isinstance(e, dict)]
        if (not episodes2) or str(episodes2[0].get("topic_label") or "") != str(top_topic):
            ep_id = _stable_hash_obj({"v": 108, "topic": str(top_topic), "turn_index": int(ut.get("turn_index") or 0)})
            episodes2.insert(
                0,
                {
                    "episode_id": str(ep_id),
                    "topic_label": str(top_topic),
                    "facts": [],
                    "commitments_refs": [],
                    "open_threads": [str(x) for x in pragmatics_pending_questions_active if isinstance(x, str) and x],
                    "salience": 1.0,
                    "last_touched_turn": int(ut.get("turn_index") or 0),
                },
            )
        else:
            episodes2[0]["last_touched_turn"] = int(ut.get("turn_index") or 0)
            episodes2[0]["open_threads"] = [str(x) for x in pragmatics_pending_questions_active if isinstance(x, str) and x]
        # Deterministic compaction: keep at most 8 episodes.
        if len(episodes2) > 8:
            flow_memory_v108["memory_compaction_performed_v108"] = True
            episodes2 = episodes2[:8]
        else:
            flow_memory_v108["memory_compaction_performed_v108"] = False
        flow_memory_v108["episodic_memory"] = list(episodes2)

        # Update recency buffers.
        rh = [str(chosen_hash_v108)] + [str(x) for x in (flow_memory_v108.get("recent_phrases_hashes") if isinstance(flow_memory_v108.get("recent_phrases_hashes"), list) else []) if isinstance(x, str) and x and str(x) != str(chosen_hash_v108)]
        flow_memory_v108["recent_phrases_hashes"] = list(rh[:8])
        ra = [str(x) for x in (flow_sel.get("discourse_plan_v108") if isinstance(flow_sel.get("discourse_plan_v108"), list) else []) if isinstance(x, str) and x]
        prev_ra = [str(x) for x in (flow_memory_v108.get("recent_discourse_acts") if isinstance(flow_memory_v108.get("recent_discourse_acts"), list) else []) if isinstance(x, str) and x]
        flow_memory_v108["recent_discourse_acts"] = list((ra + prev_ra)[:12])

        # Progress gating for flow: inherit prior gating and apply flow metrics.
        progress_allowed_v108 = bool(progress_allowed_v107) and bool(flow_sel.get("progress_allowed_v108", True))
        repair_action_v108 = str(flow_sel.get("repair_action_v108") or "")
        if (not progress_allowed_v108) and (not repair_action_v108):
            # Ensure a repair action is always recorded when progress is blocked.
            repair_action_v108 = "clarify_reference" if str(binding_status) in {"AMBIGUOUS", "MISS"} else "clarify_preference"

        fe_v108 = FlowEventV108(
            conversation_id=str(conversation_id),
            user_turn_id=str(ut.get("turn_id") or ""),
            user_turn_index=int(ut.get("turn_index") or 0),
            assistant_turn_id=str(at.get("turn_id") or ""),
            assistant_turn_index=int(at.get("turn_index") or 0),
            user_text_hash=sha256_hex(str(user_text).encode("utf-8")),
            assistant_text_hash=str(chosen_hash_v108),
            coherence_score_v106=int(dialogue_selected_metrics_v106.get("coherence_score") or 0),
            coherence_flags_v106=dict(dialogue_selected_metrics_v106.get("flags") or {}) if isinstance(dialogue_selected_metrics_v106.get("flags"), dict) else {},
            pragmatics_score_v107=int(selected_pm.get("pragmatics_score") or 0),
            pragmatics_flags_v107=dict(selected_pm.get("flags") or {}) if isinstance(selected_pm.get("flags"), dict) else {},
            intent_act_v107=dict(user_intent_act_for_ev),
            discourse_plan_v108=[str(x) for x in (flow_sel.get("discourse_plan_v108") if isinstance(flow_sel.get("discourse_plan_v108"), list) else []) if isinstance(x, str) and x],
            verbosity_mode_v108=str(flow_sel.get("verbosity_mode_v108") or ""),
            tone_mode_v108=str(flow_sel.get("tone_mode_v108") or ""),
            flow_memory_v108=dict(flow_memory_v108),
            candidates=[dict(x) for x in flow_cands_for_ev if isinstance(x, dict)],
            selection_method_v108="DSC",
            chosen_candidate_hash=str(chosen_hash_v108),
            flow_score_v108=int(flow_sel.get("flow_score_v108") or 0),
            flow_flags_v108=dict(flow_sel.get("flags") or {}) if isinstance(flow_sel.get("flags"), dict) else {},
            progress_allowed_v108=bool(progress_allowed_v108),
            repair_action_v108=str(repair_action_v108),
            survival_rule_hits_v108=[str(x) for x in (flow_sel.get("survival_rule_hits_v108") if isinstance(flow_sel.get("survival_rule_hits_v108"), list) else []) if isinstance(x, str) and x],
            created_step=int(at.get("created_step") or 0),
            prev_event_sig=str(prev_flow_event_sig),
        ).to_dict()
        _append_flow_event(fe_v108)

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
        style_plan = {
            "selected_candidate_id": str(style_selected.get("candidate_id") or ""),
            "selected_template_id": str(style_selected.get("template_id") or ""),
            "selected_text_sha256": str(style_selected.get("text_sha256") or ""),
            "selected_fluency_score": _round6(style_selected.get("fluency_score")),
            "selection": dict(style_selection) if isinstance(style_selection, dict) else {},
            "candidates_topk": [
                {
                    "candidate_id": str(c.get("candidate_id") or ""),
                    "template_id": str(c.get("template_id") or ""),
                    "text_sha256": str(c.get("text_sha256") or ""),
                    "fluency_score": _round6(c.get("fluency_score")),
                    "ok": bool(((c.get("critics") or {}).get("ok")) if isinstance(c.get("critics"), dict) else False),
                }
                for c in list(style_candidates_topk)[:8]
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
            "style": dict(style_plan),
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
                "style_template_id": str(style_selected.get("template_id") or ""),
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
                    "style_profile": style_profile.to_dict() if isinstance(style_profile, StyleProfileV102) else default_style_profile_v102().to_dict(),
                    "discourse_state": dict(discourse_state),
                    "fragment_promoted_ids": list(frag_promoted),
                    "fragment_promoted_count": int(len(frag_promoted)),
                    "active_plan_state_v104": dict(active_plan_state_v104) if isinstance(active_plan_state_v104, dict) else {},
                    "agency_registry_v105": dict((agency_registry_snapshot_v105(list(agency_events))).get("by_goal") or {}),
                    "agency_chain_hash_v105": str(compute_agency_chain_hash_v105(list(agency_events))),
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

    # Binding snapshot (derived; write-once WORM).
    if os.path.exists(binding_snapshot_path):
        _fail(f"binding_snapshot_exists:{binding_snapshot_path}")
    binding_snapshot = binding_snapshot_v101(list(binding_events))
    tmpbs = binding_snapshot_path + ".tmp"
    with open(tmpbs, "w", encoding="utf-8") as f:
        f.write(json.dumps(binding_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpbs, binding_snapshot_path)

    # Template library snapshot (derived from style events; write-once WORM).
    if os.path.exists(template_snapshot_path):
        _fail(f"template_snapshot_exists:{template_snapshot_path}")
    template_snapshot = template_library_snapshot_v102(templates=list(templates_v102), style_events=list(style_events))
    tmpts = template_snapshot_path + ".tmp"
    with open(tmpts, "w", encoding="utf-8") as f:
        f.write(json.dumps(template_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpts, template_snapshot_path)

    # Concept library snapshot (derived from concept events; write-once WORM).
    if os.path.exists(concept_snapshot_path):
        _fail(f"concept_snapshot_exists:{concept_snapshot_path}")
    concept_snapshot = concept_library_snapshot_v103(concept_events=list(concept_events))
    tmpcs = concept_snapshot_path + ".tmp"
    with open(tmpcs, "w", encoding="utf-8") as f:
        f.write(json.dumps(concept_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpcs, concept_snapshot_path)

    # Plan registry snapshot (derived; write-once WORM).
    if os.path.exists(plan_snapshot_path):
        _fail(f"plan_snapshot_exists:{plan_snapshot_path}")
    plan_snapshot = plan_registry_snapshot_v104(plan_events=list(plan_events))
    tmpps = plan_snapshot_path + ".tmp"
    with open(tmpps, "w", encoding="utf-8") as f:
        f.write(json.dumps(plan_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpps, plan_snapshot_path)

    # Agency registry snapshot (derived; write-once WORM).
    if os.path.exists(agency_snapshot_path):
        _fail(f"agency_snapshot_exists:{agency_snapshot_path}")
    agency_snapshot = agency_registry_snapshot_v105(list(agency_events))
    tmpas = agency_snapshot_path + ".tmp"
    with open(tmpas, "w", encoding="utf-8") as f:
        f.write(json.dumps(agency_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpas, agency_snapshot_path)

    # Dialogue registry snapshot (derived; write-once WORM).
    if os.path.exists(dialogue_snapshot_path):
        _fail(f"dialogue_snapshot_exists:{dialogue_snapshot_path}")
    dialogue_snapshot = dialogue_registry_snapshot_v106(list(dialogue_events))
    tmpds = dialogue_snapshot_path + ".tmp"
    with open(tmpds, "w", encoding="utf-8") as f:
        f.write(json.dumps(dialogue_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpds, dialogue_snapshot_path)

    # Pragmatics registry snapshot (derived; write-once WORM).
    if os.path.exists(pragmatics_snapshot_path):
        _fail(f"pragmatics_snapshot_exists:{pragmatics_snapshot_path}")
    pragmatics_snapshot = pragmatics_registry_snapshot_v107(list(pragmatics_events))
    tmpps2 = pragmatics_snapshot_path + ".tmp"
    with open(tmpps2, "w", encoding="utf-8") as f:
        f.write(json.dumps(pragmatics_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpps2, pragmatics_snapshot_path)

    # Flow registry snapshot (derived; write-once WORM).
    if os.path.exists(flow_snapshot_path):
        _fail(f"flow_snapshot_exists:{flow_snapshot_path}")
    flow_snapshot = flow_registry_snapshot_v108(list(flow_events), flow_memory_last=dict(flow_memory_v108))
    tmpfs = flow_snapshot_path + ".tmp"
    with open(tmpfs, "w", encoding="utf-8") as f:
        f.write(json.dumps(flow_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpfs, flow_snapshot_path)

    binding_chain_hash = binding_chain_hash_v101(list(binding_events))

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
        "binding_chain_ok": bool(verify_chained_jsonl_v96(binding_events_path)),
        "style_chain_ok": bool(verify_chained_jsonl_v96(style_path)),
        "concept_chain_ok": bool(verify_chained_jsonl_v96(concept_events_path)),
        "plan_events_chain_ok": bool(verify_chained_jsonl_v96(plan_events_path)),
        "agency_chain_ok": bool(verify_chained_jsonl_v96(agency_events_path)),
        "dialogue_chain_ok": bool(verify_chained_jsonl_v96(dialogue_events_path)),
        "pragmatics_chain_ok": bool(verify_chained_jsonl_v96(pragmatics_events_path)),
        "flow_chain_ok": bool(verify_chained_jsonl_v96(flow_events_path)),
        "states_chain_ok": bool(verify_chained_jsonl_v96(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v96(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v96(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v96(transcript_path)),
        "template_snapshot_exists": bool(os.path.exists(template_snapshot_path)),
        "concept_snapshot_exists": bool(os.path.exists(concept_snapshot_path)),
        "plan_snapshot_exists": bool(os.path.exists(plan_snapshot_path)),
        "agency_snapshot_exists": bool(os.path.exists(agency_snapshot_path)),
        "dialogue_snapshot_exists": bool(os.path.exists(dialogue_snapshot_path)),
        "pragmatics_snapshot_exists": bool(os.path.exists(pragmatics_snapshot_path)),
        "flow_snapshot_exists": bool(os.path.exists(flow_snapshot_path)),
    }
    ok_chain, chain_reason, chain_details = verify_conversation_chain_v108(
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
        binding_events=list(binding_events),
        binding_snapshot=dict(binding_snapshot),
        style_events=list(style_events),
        template_snapshot=dict(template_snapshot),
        concept_events=list(concept_events),
        concept_snapshot=dict(concept_snapshot),
        plan_events=list(plan_events),
        plan_snapshot=dict(plan_snapshot),
        agency_events=list(agency_events),
        agency_snapshot=dict(agency_snapshot),
        dialogue_events=list(dialogue_events),
        dialogue_snapshot=dict(dialogue_snapshot),
        pragmatics_events=list(pragmatics_events),
        pragmatics_snapshot=dict(pragmatics_snapshot),
        flow_events=list(flow_events),
        flow_snapshot=dict(flow_snapshot),
        tail_k=6,
        repo_root=str(repo_root),
    )

    transcript_hash = compute_transcript_hash_v96(turns)
    state_chain_hash = compute_state_chain_hash_v96(states)
    parse_chain_hash = compute_parse_chain_hash_v96(parse_events)
    learned_chain_hash = compute_learned_chain_hash_v96(learned_rule_events)
    plan_chain_hash = compute_plan_chain_hash_v96(action_plans)
    plan_events_chain_hash_v104 = compute_plan_chain_hash_v104(list(plan_events))
    agency_chain_hash_v105 = compute_agency_chain_hash_v105(list(agency_events))
    memory_chain_hash = compute_memory_chain_hash_v96(memory_events)
    belief_chain_hash = compute_belief_chain_hash_v96(belief_events)
    evidence_chain_hash = compute_evidence_chain_hash_v98(evidence_events)
    goal_chain_hash = compute_goal_chain_hash_v99(list(goal_events))
    discourse_chain_hash = compute_discourse_chain_hash_v100(list(discourse_events))
    fragment_chain_hash = compute_fragment_chain_hash_v100(list(fragment_events))
    style_chain_hash = compute_style_chain_hash_v102(list(style_events))
    concept_chain_hash = compute_concept_chain_hash_v103(list(concept_events))
    dialogue_chain_hash_v106 = compute_dialogue_chain_hash_v106(list(dialogue_events))
    pragmatics_chain_hash_v107 = compute_pragmatics_chain_hash_v107(list(pragmatics_events))
    flow_chain_hash_v108 = compute_flow_chain_hash_v108(list(flow_events))
    system_spec_sha256 = sha256_file(system_spec_path)
    binding_chain_hash = str(chain_details.get("binding_chain_hash") or binding_chain_hash)
    concept_chain_hash = str(chain_details.get("concept_chain_hash") or concept_chain_hash)
    concept_snapshot_sig = str(chain_details.get("concept_snapshot_sig") or concept_snapshot.get("snapshot_sig") or "")
    plan_events_chain_hash_v104 = str(chain_details.get("plan_chain_hash_v104") or plan_events_chain_hash_v104)
    agency_chain_hash_v105 = str(chain_details.get("agency_chain_hash_v105") or agency_chain_hash_v105)
    dialogue_chain_hash_v106 = str(chain_details.get("dialogue_chain_hash_v106") or dialogue_chain_hash_v106)
    pragmatics_chain_hash_v107 = str(chain_details.get("pragmatics_chain_hash_v107") or pragmatics_chain_hash_v107)
    flow_chain_hash_v108 = str(chain_details.get("flow_chain_hash_v108") or flow_chain_hash_v108)
    plan_snapshot_sig_v104 = str(chain_details.get("plan_snapshot_sig_v104") or plan_snapshot.get("snapshot_sig") or "")

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
        "plan_events_chain_hash_v104": str(plan_events_chain_hash_v104),
        "agency_chain_hash_v105": str(agency_chain_hash_v105),
        "dialogue_chain_hash_v106": str(dialogue_chain_hash_v106),
        "pragmatics_chain_hash_v107": str(pragmatics_chain_hash_v107),
        "flow_chain_hash_v108": str(flow_chain_hash_v108),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
        "binding_chain_hash": str(binding_chain_hash),
        "style_chain_hash": str(style_chain_hash),
        "concept_chain_hash": str(concept_chain_hash),
        "concept_snapshot_sig": str(concept_snapshot_sig),
        "plan_snapshot_sig_v104": str(plan_snapshot_sig_v104),
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
        "plan_events_chain_hash_v104": str(plan_events_chain_hash_v104),
        "agency_chain_hash_v105": str(agency_chain_hash_v105),
        "dialogue_chain_hash_v106": str(dialogue_chain_hash_v106),
        "pragmatics_chain_hash_v107": str(pragmatics_chain_hash_v107),
        "flow_chain_hash_v108": str(flow_chain_hash_v108),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
        "binding_chain_hash": str(binding_chain_hash),
        "style_chain_hash": str(style_chain_hash),
        "concept_chain_hash": str(concept_chain_hash),
        "concept_snapshot_sig": str(concept_snapshot_sig),
        "plan_snapshot_sig_v104": str(plan_snapshot_sig_v104),
        "verify_ok": bool(verify_obj.get("ok", False)),
        "sha256": {
            "store_jsonl": str(sha256_file(store_path)),
            "intent_grammar_snapshot_json": str(sha256_file(grammar_snapshot_path)),
            "system_spec_snapshot_json": str(system_spec_sha256),
            "conversation_turns_jsonl": str(sha256_file(turns_path)),
            "intent_parses_jsonl": str(sha256_file(parses_path)),
            "learned_intent_rules_jsonl": str(sha256_file(learned_path)) if os.path.exists(learned_path) else "",
            "action_plans_jsonl": str(sha256_file(plans_path)),
            "plan_events_v104_jsonl": str(sha256_file(plan_events_path)),
            "plan_registry_snapshot_v104_json": str(sha256_file(plan_snapshot_path)),
            "agency_events_v105_jsonl": str(sha256_file(agency_events_path)),
            "agency_registry_snapshot_v105_json": str(sha256_file(agency_snapshot_path)),
            "dialogue_events_v106_jsonl": str(sha256_file(dialogue_events_path)),
            "dialogue_registry_snapshot_v106_json": str(sha256_file(dialogue_snapshot_path)),
            "pragmatics_events_v107_jsonl": str(sha256_file(pragmatics_events_path)),
            "pragmatics_registry_snapshot_v107_json": str(sha256_file(pragmatics_snapshot_path)),
            "flow_events_v108_jsonl": str(sha256_file(flow_events_path)),
            "flow_registry_snapshot_v108_json": str(sha256_file(flow_snapshot_path)),
            "memory_events_jsonl": str(sha256_file(memory_path)),
            "belief_events_jsonl": str(sha256_file(belief_path)),
            "evidence_events_jsonl": str(sha256_file(evidence_path)),
            "goal_events_jsonl": str(sha256_file(goal_path)),
            "goal_ledger_snapshot_json": str(sha256_file(goal_snapshot_path)),
            "discourse_events_jsonl": str(sha256_file(discourse_path)),
            "fragment_events_jsonl": str(sha256_file(fragment_events_path)),
            "fragment_library_snapshot_json": str(sha256_file(fragment_snapshot_path)),
            "binding_events_jsonl": str(sha256_file(binding_events_path)),
            "binding_snapshot_json": str(sha256_file(binding_snapshot_path)),
            "style_events_jsonl": str(sha256_file(style_path)),
            "template_library_snapshot_v102_json": str(sha256_file(template_snapshot_path)),
            "concept_events_jsonl": str(sha256_file(concept_events_path)),
            "concept_library_snapshot_v103_json": str(sha256_file(concept_snapshot_path)),
            "conversation_states_jsonl": str(sha256_file(states_path)) if os.path.exists(states_path) else "",
            "dialogue_trials_jsonl": str(sha256_file(trials_path)),
            "objective_evals_jsonl": str(sha256_file(evals_path)),
            "transcript_jsonl": str(sha256_file(transcript_path)),
            "verify_chain_v108_json": str(sha256_file(verify_path)),
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
        "plan_events_chain_hash_v104": str(plan_events_chain_hash_v104),
        "agency_chain_hash_v105": str(agency_chain_hash_v105),
        "dialogue_chain_hash_v106": str(dialogue_chain_hash_v106),
        "pragmatics_chain_hash_v107": str(pragmatics_chain_hash_v107),
        "flow_chain_hash_v108": str(flow_chain_hash_v108),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
        "binding_chain_hash": str(binding_chain_hash),
        "style_chain_hash": str(style_chain_hash),
        "concept_chain_hash": str(concept_chain_hash),
        "ledger_hash": str(ledger_hash),
        "turns_total": int(len(turns)),
        "user_turns_total": int(user_turns_total),
        "states_total": int(len(states)),
        "plans_total": int(len(action_plans)),
        "plan_events_total_v104": int(len(plan_events)),
        "agency_events_total_v105": int(len(agency_events)),
        "dialogue_events_total_v106": int(len(dialogue_events)),
        "pragmatics_events_total_v107": int(len(pragmatics_events)),
        "memory_events_total": int(len(memory_events)),
        "belief_events_total": int(len(belief_events)),
        "evidence_events_total": int(len(evidence_events)),
        "goal_events_total": int(len(goal_events)),
        "discourse_events_total": int(len(discourse_events)),
        "fragment_events_total": int(len(fragment_events)),
        "binding_events_total": int(len(binding_events)),
        "style_events_total": int(len(style_events)),
        "concept_events_total": int(len(concept_events)),
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
        "plan_events_chain_hash_v104": str(plan_events_chain_hash_v104),
        "agency_chain_hash_v105": str(agency_chain_hash_v105),
        "dialogue_chain_hash_v106": str(dialogue_chain_hash_v106),
        "pragmatics_chain_hash_v107": str(pragmatics_chain_hash_v107),
        "flow_chain_hash_v108": str(flow_chain_hash_v108),
        "memory_chain_hash": str(memory_chain_hash),
        "belief_chain_hash": str(belief_chain_hash),
        "evidence_chain_hash": str(evidence_chain_hash),
        "goal_chain_hash": str(goal_chain_hash),
        "discourse_chain_hash": str(discourse_chain_hash),
        "fragment_chain_hash": str(fragment_chain_hash),
        "binding_chain_hash": str(binding_chain_hash),
        "style_chain_hash": str(style_chain_hash),
        "concept_chain_hash": str(concept_chain_hash),
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
            "plan_events_v104_jsonl": str(plan_events_path),
            "plan_registry_snapshot_v104_json": str(plan_snapshot_path),
            "agency_events_v105_jsonl": str(agency_events_path),
            "agency_registry_snapshot_v105_json": str(agency_snapshot_path),
            "dialogue_events_v106_jsonl": str(dialogue_events_path),
            "dialogue_registry_snapshot_v106_json": str(dialogue_snapshot_path),
            "pragmatics_events_v107_jsonl": str(pragmatics_events_path),
            "pragmatics_registry_snapshot_v107_json": str(pragmatics_snapshot_path),
            "flow_events_v108_jsonl": str(flow_events_path),
            "flow_registry_snapshot_v108_json": str(flow_snapshot_path),
            "memory_events_jsonl": str(memory_path),
            "belief_events_jsonl": str(belief_path),
            "evidence_events_jsonl": str(evidence_path),
            "goal_events_jsonl": str(goal_path),
            "goal_ledger_snapshot_json": str(goal_snapshot_path),
            "discourse_events_jsonl": str(discourse_path),
            "fragment_events_jsonl": str(fragment_events_path),
            "fragment_library_snapshot_json": str(fragment_snapshot_path),
            "binding_events_jsonl": str(binding_events_path),
            "binding_snapshot_json": str(binding_snapshot_path),
            "style_events_jsonl": str(style_path),
            "template_library_snapshot_v102_json": str(template_snapshot_path),
            "concept_events_jsonl": str(concept_events_path),
            "concept_library_snapshot_v103_json": str(concept_snapshot_path),
            "states_jsonl": str(states_path),
            "trials_jsonl": str(trials_path),
            "evals_jsonl": str(evals_path),
            "transcript_jsonl": str(transcript_path),
            "verify_json": str(verify_path),
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
        },
    }
