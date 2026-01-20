from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .conversation_actions_v90 import action_concepts_for_dsl_v90
from .conversation_objectives_v90 import COMM_OBJECTIVES_V90, comm_objective_ids_v90, make_comm_objective_eq_text_v90
from .conversation_v91 import (
    ConversationStateV91,
    TurnV91,
    append_chained_jsonl_v91,
    compute_parse_chain_hash_v91,
    compute_state_chain_hash_v91,
    compute_transcript_hash_v91,
    normalize_text_v91,
    text_sig_v91,
    verify_chained_jsonl_v91,
    verify_conversation_chain_v91,
)
from .engine_v80 import EngineV80
from .goal_supports_v89 import SupportClaimV89, fold_support_stats_v89, list_supporting_concepts_for_goal_v89, make_goal_support_evidence_event_v89
from .intent_grammar_v91 import (
    INTENT_ADD_V91,
    INTENT_END_V91,
    INTENT_GET_V91,
    INTENT_SET_V91,
    INTENT_SUMMARY_V91,
    INTENT_UNKNOWN_V91,
    default_intent_rule_acts_v91,
    default_intent_rules_v91,
    grammar_hash_v91,
    intent_grammar_snapshot_v91,
    parse_intent_v91,
    tokenize_user_text_v91,
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


def _rank_action_candidates_v91(
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


def _summarize_bindings_v91(*, vars_map: Dict[str, Any], last_answer: Any) -> str:
    parts: List[str] = []
    for k in sorted(vars_map.keys(), key=str):
        parts.append(f"{str(k)}={str(vars_map.get(k))}")
    if last_answer not in ("", None):
        parts.append(f"last_answer={str(last_answer)}")
    return "Resumo: " + "; ".join(parts)


def _choose_comm_objective_v91(
    *,
    parse: Dict[str, Any],
    vars_map: Dict[str, Any],
    last_answer: Any,
) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministic communicative objective selection.
    Returns (objective_kind, ctx).
    """
    if not bool(parse.get("parse_ok", False)):
        reason = str(parse.get("reason") or "")
        if reason == "ambiguous":
            return "COMM_CONFIRM", {"ambiguous": list(parse.get("ambiguous_intents") or [])}
        # Missing slots -> ask (fail-closed).
        missing = parse.get("missing_slots")
        missing_list = missing if isinstance(missing, list) else []
        missing_list2 = [str(x) for x in missing_list if isinstance(x, str) and x]
        missing_list2.sort()
        if missing_list2:
            return "COMM_ASK_CLARIFY", {"missing_slot": missing_list2[0], "parse_reason": reason}
        # No-match / invalid input -> correct (do not ask for nonexistent slots).
        return "COMM_CORRECT", {"reason": f"parse_fail:{reason or 'no_match'}"}

    intent_id = str(parse.get("intent_id") or "")
    slots = parse.get("slots") if isinstance(parse.get("slots"), dict) else {}
    if intent_id == INTENT_SUMMARY_V91:
        return "COMM_SUMMARIZE", {}
    if intent_id == INTENT_END_V91:
        return "COMM_END", {}
    if intent_id == INTENT_GET_V91:
        k = str(slots.get("k") or "")
        if not k or k not in vars_map:
            return "COMM_ASK_CLARIFY", {"missing_key": str(k or "")}
        return "COMM_RESPOND", {}
    if intent_id == INTENT_SET_V91:
        return "COMM_RESPOND", {}
    if intent_id == INTENT_ADD_V91:
        a = str(slots.get("a") or "")
        b = str(slots.get("b") or "")
        va, ra = _parse_int_or_var(vars_map=vars_map, tok=a, last_answer=last_answer)
        if va is None:
            return "COMM_ASK_CLARIFY", {"missing_key": str(a), "reason": str(ra)}
        vb, rb = _parse_int_or_var(vars_map=vars_map, tok=b, last_answer=last_answer)
        if vb is None:
            return "COMM_ASK_CLARIFY", {"missing_key": str(b), "reason": str(rb)}
        return "COMM_RESPOND", {}
    if intent_id == INTENT_UNKNOWN_V91:
        return "COMM_CORRECT", {"reason": "intent_unknown"}
    return "COMM_ADMIT_UNKNOWN", {"reason": "no_rule"}


def _build_expected_and_action_inputs_v91(
    *,
    objective_kind: str,
    parse: Dict[str, Any],
    vars_map: Dict[str, Any],
    last_answer: Any,
    ctx: Dict[str, Any],
    user_text: str,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Returns (expected_text, action_inputs, hint_action_id).
    """
    intent_id = str(parse.get("intent_id") or "")
    slots = parse.get("slots") if isinstance(parse.get("slots"), dict) else {}

    if objective_kind == "COMM_END":
        return "Encerrado.", {}, "concept_v90_end_conversation_v0"
    if objective_kind == "COMM_ADMIT_UNKNOWN":
        return "Não sei.", {}, "concept_v90_admit_unknown_v0"
    if objective_kind == "COMM_CORRECT":
        msg = normalize_text_v91(str(user_text))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"
    if objective_kind == "COMM_CONFIRM":
        amb = ctx.get("ambiguous")
        amb_list = amb if isinstance(amb, list) else []
        opts = []
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
    if objective_kind == "COMM_SUMMARIZE":
        summ = _summarize_bindings_v91(vars_map=vars_map, last_answer=last_answer)
        return summ, {"text": summ}, "concept_v90_emit_text_v0"
    if objective_kind == "COMM_ASK_CLARIFY":
        missing_key = str(ctx.get("missing_key") or "")
        missing_slot = str(ctx.get("missing_slot") or "")
        # Prefer asking for a concrete key when present.
        if missing_key:
            return f"Qual é o valor de {missing_key}?", {"k": missing_key}, "concept_v90_ask_clarify_v0"
        # Missing value for SET: ask value of the known key.
        if missing_slot == "v":
            k = str(slots.get("k") or "")
            if k:
                return f"Qual é o valor de {k}?", {"k": k}, "concept_v90_ask_clarify_v0"
        # Generic slot clarification.
        q = f"Faltando: {missing_slot}" if missing_slot else "Faltando: dados"
        return q, {"text": q}, "concept_v90_emit_text_v0"

    # COMM_RESPOND
    if intent_id == INTENT_SET_V91:
        k = str(slots.get("k") or "")
        v = str(slots.get("v") or "")
        return f"OK: {k}={v}", {"k": k, "v": v}, "concept_v90_confirm_set_v0"
    if intent_id == INTENT_GET_V91:
        k = str(slots.get("k") or "")
        v = vars_map.get(k)
        text = f"{k}={v}"
        return text, {"text": text}, "concept_v90_emit_text_v0"
    if intent_id == INTENT_ADD_V91:
        a = str(slots.get("a") or "")
        b = str(slots.get("b") or "")
        va, _ = _parse_int_or_var(vars_map=vars_map, tok=a, last_answer=last_answer)
        vb, _ = _parse_int_or_var(vars_map=vars_map, tok=b, last_answer=last_answer)
        s = int(va or 0) + int(vb or 0)
        return f"SUM={s}", {"sum": str(s)}, "concept_v90_emit_sum_v0"
    if intent_id == INTENT_SUMMARY_V91:
        summ = _summarize_bindings_v91(vars_map=vars_map, last_answer=last_answer)
        return summ, {"text": summ}, "concept_v90_emit_text_v0"
    return "Não sei.", {}, "concept_v90_admit_unknown_v0"


def run_conversation_v91(
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
    states_path = os.path.join(str(out_dir), "conversation_states.jsonl")
    trials_path = os.path.join(str(out_dir), "dialogue_trials.jsonl")
    evals_path = os.path.join(str(out_dir), "objective_evals.jsonl")
    transcript_path = os.path.join(str(out_dir), "transcript.jsonl")
    verify_path = os.path.join(str(out_dir), "verify_chain_v91.json")
    manifest_path = os.path.join(str(out_dir), "freeze_manifest_v91.json")
    summary_path = os.path.join(str(out_dir), "summary.json")

    store = ActStore()

    # Communicative objectives (Objective CSV, V88 kind).
    obj_ids = comm_objective_ids_v90()
    for okind, oid in sorted(obj_ids.items(), key=lambda kv: str(kv[0])):
        store.add(make_comm_objective_eq_text_v90(objective_id=str(oid), objective_kind=str(okind), created_step=0))

    # Language actions (concept_csv).
    goal_ids = {k: str(k) for k in COMM_OBJECTIVES_V90}
    for act in action_concepts_for_dsl_v90(goal_ids=goal_ids):
        store.add(act)

    # Intent grammar as explicit acts (non-executable).
    rules = default_intent_rules_v91()
    for act in default_intent_rule_acts_v91(created_step=0):
        store.add(act)

    if os.path.exists(store_path):
        _fail(f"store_path_exists:{store_path}")
    store.save_jsonl(store_path)
    store_hash = store.content_hash()

    # Snapshot grammar for audit (WORM write-once).
    if os.path.exists(grammar_snapshot_path):
        _fail(f"grammar_snapshot_exists:{grammar_snapshot_path}")
    grammar_snapshot = intent_grammar_snapshot_v91(rules)
    tmpg = grammar_snapshot_path + ".tmp"
    with open(tmpg, "w", encoding="utf-8") as f:
        f.write(json.dumps(grammar_snapshot, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpg, grammar_snapshot_path)

    engine = EngineV80(store, seed=int(seed))

    # Deterministic conversation id binds user turns + grammar hash.
    conv_body = {"turns": list(user_turn_texts), "grammar_hash": str(grammar_hash_v91(rules))}
    conversation_id = f"conv_v91_{sha256_hex(canonical_json_dumps(conv_body).encode('utf-8'))}"

    # Runtime explicit bindings.
    vars_map: Dict[str, Any] = {}
    last_answer: Any = ""

    turns: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []
    transcript: List[Dict[str, Any]] = []
    parse_events: List[Dict[str, Any]] = []
    trials: List[Dict[str, Any]] = []

    prev_turns_hash: Optional[str] = None
    prev_parses_hash: Optional[str] = None
    prev_states_hash: Optional[str] = None
    prev_trials_hash: Optional[str] = None
    prev_evals_hash: Optional[str] = None
    prev_transcript_hash: Optional[str] = None

    support_events: List[Dict[str, Any]] = []

    prev_state_id = ""
    state_index = 0
    turn_index = 0
    step = 0

    def _objective_act_id(okind: str) -> str:
        return str(obj_ids.get(okind) or "")

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
            try:
                cost_used += float(c.get("cost", 0.0) or 0.0)
            except Exception:
                pass
        return True, str(out_text), dict(meta), float(cost_used)

    for user_text in list(user_turn_texts):
        # Parse user intent deterministically.
        parse = parse_intent_v91(user_text=str(user_text), rules=list(rules))
        tokens = parse.get("tokens")
        tokens = tokens if isinstance(tokens, list) else tokenize_user_text_v91(str(user_text))

        # Create user turn.
        ut = TurnV91(
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
        prev_turns_hash = append_chained_jsonl_v91(
            turns_path,
            {"time": deterministic_iso(step=int(ut["created_step"])), "step": int(ut["created_step"]), "event": "TURN", "payload": dict(ut)},
            prev_hash=prev_turns_hash,
        )
        transcript.append({"role": "user", "text": str(ut.get("text") or ""), "turn_id": str(ut.get("turn_id") or "")})
        prev_transcript_hash = append_chained_jsonl_v91(
            transcript_path,
            {"time": deterministic_iso(step=int(ut["created_step"])), "step": int(ut["created_step"]), "event": "UTTERANCE", "payload": dict(transcript[-1])},
            prev_hash=prev_transcript_hash,
        )

        # Log parse (WORM, hash-chained).
        parse_event = {
            "kind": "intent_parse_v91",
            "time": deterministic_iso(step=int(step)),
            "step": int(step),
            "turn_id": str(ut.get("turn_id") or ""),
            "turn_index": int(ut.get("turn_index") or 0),
            "payload": dict(parse),
        }
        prev_parses_hash = append_chained_jsonl_v91(parses_path, dict(parse_event), prev_hash=prev_parses_hash)
        parse_events.append({"turn_id": str(parse_event["turn_id"]), "turn_index": int(parse_event["turn_index"]), "payload": dict(parse)})

        # Determine objective kind.
        objective_kind, ctx = _choose_comm_objective_v91(parse=parse, vars_map=dict(vars_map), last_answer=last_answer)

        # Apply state updates only when safe.
        slots = parse.get("slots") if isinstance(parse.get("slots"), dict) else {}
        if bool(parse.get("parse_ok", False)) and not (parse.get("missing_slots") or []):
            if str(parse.get("intent_id") or "") == INTENT_SET_V91:
                k = str(slots.get("k") or "")
                v = str(slots.get("v") or "")
                if k and v:
                    vars_map[k] = int(v) if _is_int_literal(v) else v
            if str(parse.get("intent_id") or "") == INTENT_ADD_V91 and objective_kind == "COMM_RESPOND":
                a = str(slots.get("a") or "")
                b = str(slots.get("b") or "")
                va, _ = _parse_int_or_var(vars_map=vars_map, tok=a, last_answer=last_answer)
                vb, _ = _parse_int_or_var(vars_map=vars_map, tok=b, last_answer=last_answer)
                if va is not None and vb is not None:
                    last_answer = int(va) + int(vb)

        # Build expected output and action inputs.
        expected_text, action_inputs, hint_action_id = _build_expected_and_action_inputs_v91(
            objective_kind=str(objective_kind),
            parse=dict(parse),
            vars_map=dict(vars_map),
            last_answer=last_answer,
            ctx=dict(ctx),
            user_text=str(user_text),
        )
        expected_sig = text_sig_v91(expected_text)

        # Pick supporting actions.
        candidates = list_supporting_concepts_for_goal_v89(store=store, goal_id=str(objective_kind))
        ranked = _rank_action_candidates_v91(candidates=candidates, events=support_events, goal_id=str(objective_kind))
        ordered_action_ids: List[str] = []
        if hint_action_id:
            ordered_action_ids.append(str(hint_action_id))
        for aid, _cl, _es, _ec in ranked:
            if str(aid) not in set(ordered_action_ids):
                ordered_action_ids.append(str(aid))

        # Fallback objective cascade (deterministic).
        fallback_objectives = ["COMM_ASK_CLARIFY", "COMM_ADMIT_UNKNOWN", "COMM_END"]
        tried_objectives: List[str] = []

        assistant_text = ""
        chosen_action_id = ""
        chosen_objective_id = _objective_act_id(str(objective_kind))
        chosen_eval_id = ""
        chosen_ok = False
        chosen_cost = 0.0

        def _try_objective(okind: str, *, ctx2: Dict[str, Any]) -> bool:
            nonlocal assistant_text, chosen_action_id, chosen_objective_id, chosen_eval_id, chosen_ok, chosen_cost, objective_kind, expected_text, expected_sig, action_inputs, hint_action_id
            objective_kind = str(okind)
            chosen_objective_id = _objective_act_id(str(okind))
            expected_text, action_inputs, hint_action_id = _build_expected_and_action_inputs_v91(
                objective_kind=str(okind),
                parse=dict(parse),
                vars_map=dict(vars_map),
                last_answer=last_answer,
                ctx=dict(ctx2),
                user_text=str(user_text),
            )
            expected_sig = text_sig_v91(expected_text)

            cand2 = list_supporting_concepts_for_goal_v89(store=store, goal_id=str(okind))
            ranked2 = _rank_action_candidates_v91(candidates=cand2, events=support_events, goal_id=str(okind))
            ordered2: List[str] = []
            if hint_action_id:
                ordered2.append(str(hint_action_id))
            for aid, _cl, _es, _ec in ranked2:
                if str(aid) not in set(ordered2):
                    ordered2.append(str(aid))

            for act_id in ordered2:
                ok_exec, out_text, meta, cost_used = _execute_action(act_id=str(act_id), goal_kind=str(okind), inputs=dict(action_inputs))
                output_sig = text_sig_v91(out_text)
                eval_id = _stable_hash_obj(
                    {
                        "conversation_id": str(conversation_id),
                        "turn_id": str(ut.get("turn_id") or ""),
                        "objective_kind": str(okind),
                        "objective_id": str(chosen_objective_id),
                        "act_id": str(act_id),
                        "expected_sig": str(expected_sig),
                        "output_sig": str(output_sig),
                    }
                )

                if not ok_exec:
                    verdict_ok = False
                    verdict = {"ok": False, "score": 0, "reason": f"action_exec_failed:{str(meta.get('reason') or '')}", "details": {"meta": dict(meta)}}
                else:
                    verdict_obj = execute_objective_csv_v88(
                        store=store,
                        seed=int(seed),
                        objective_act_id=str(chosen_objective_id),
                        inputs={"__output": str(out_text), "expected": str(expected_text)},
                        step=int(step),
                        goal_kind=str(okind),
                    )
                    verdict = verdict_obj.to_dict()
                    verdict_ok = bool(verdict_obj.ok)

                # Record eval + trial (WORM hash-chained).
                eval_row = {
                    "kind": "objective_eval_v91",
                    "time": deterministic_iso(step=int(step)),
                    "step": int(step),
                    "eval_id": str(eval_id),
                    "objective_kind": str(okind),
                    "objective_id": str(chosen_objective_id),
                    "expected_text_sig": str(expected_sig),
                    "output_text_sig": str(output_sig),
                    "verdict": dict(verdict),
                }
                nonlocal prev_evals_hash
                prev_evals_hash = append_chained_jsonl_v91(evals_path, dict(eval_row), prev_hash=prev_evals_hash)

                trial_id = _stable_hash_obj(
                    {
                        "conversation_id": str(conversation_id),
                        "turn_id": str(ut.get("turn_id") or ""),
                        "step": int(step),
                        "objective_kind": str(okind),
                        "objective_id": str(chosen_objective_id),
                        "act_id": str(act_id),
                        "eval_id": str(eval_id),
                    }
                )
                trial_row = {
                    "kind": "dialogue_trial_v91",
                    "time": deterministic_iso(step=int(step)),
                    "step": int(step),
                    "trial_id": str(trial_id),
                    "conversation_id": str(conversation_id),
                    "turn_id": str(ut.get("turn_id") or ""),
                    "user_turn_id": str(ut.get("turn_id") or ""),
                    "objective_kind": str(okind),
                    "objective_id": str(chosen_objective_id),
                    "action_concept_id": str(act_id),
                    "expected_text": str(expected_text),
                    "expected_text_sig": str(expected_sig),
                    "assistant_text": str(out_text),
                    "assistant_text_sig": str(output_sig),
                    "ok": bool(verdict_ok),
                    "cost_used": float(cost_used),
                }
                nonlocal prev_trials_hash
                prev_trials_hash = append_chained_jsonl_v91(trials_path, dict(trial_row), prev_hash=prev_trials_hash)

                support_ev = make_goal_support_evidence_event_v89(
                    step=int(step),
                    goal_id=str(okind),
                    concept_key=str(act_id),
                    attempt_id=str(trial_id),
                    ok=bool(verdict_ok),
                    cost_used=float(cost_used),
                    note=str(verdict.get("reason") or ""),
                )
                support_events.append(dict(support_ev))

                if verdict_ok and ok_exec:
                    assistant_text = str(out_text)
                    chosen_action_id = str(act_id)
                    chosen_eval_id = str(eval_id)
                    chosen_ok = True
                    chosen_cost = float(cost_used)
                    return True
            return False

        ok_done = _try_objective(str(objective_kind), ctx2=dict(ctx))
        tried_objectives.append(str(objective_kind))
        if not ok_done:
            for fb in fallback_objectives:
                if fb in set(tried_objectives):
                    continue
                ok_done = _try_objective(str(fb), ctx2={})
                tried_objectives.append(str(fb))
                if ok_done:
                    break
        if not ok_done:
            _try_objective("COMM_END", ctx2={})

        # Create assistant turn.
        at = TurnV91(
            conversation_id=str(conversation_id),
            turn_index=int(turn_index),
            role="assistant",
            text=str(assistant_text),
            created_step=int(step),
            offset_us=1,
            objective_id=str(chosen_objective_id),
            objective_kind=str(objective_kind),
            action_concept_id=str(chosen_action_id),
            eval_id=str(chosen_eval_id),
        ).to_dict()
        turn_index += 1
        step += 1
        turns.append(dict(at))
        prev_turns_hash = append_chained_jsonl_v91(
            turns_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "TURN", "payload": dict(at)},
            prev_hash=prev_turns_hash,
        )
        transcript.append({"role": "assistant", "text": str(at.get("text") or ""), "turn_id": str(at.get("turn_id") or "")})
        prev_transcript_hash = append_chained_jsonl_v91(
            transcript_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "UTTERANCE", "payload": dict(transcript[-1])},
            prev_hash=prev_transcript_hash,
        )

        # Save trial meta for verifier (objective kind + turn ids).
        trials.append(
            {
                "objective_kind": str(objective_kind),
                "user_turn_id": str(ut.get("turn_id") or ""),
                "assistant_turn_id": str(at.get("turn_id") or ""),
                "ok": bool(chosen_ok),
                "cost_used": float(chosen_cost),
            }
        )

        # Create new state only when not a clarification/confirmation.
        if str(objective_kind) not in {"COMM_ASK_CLARIFY", "COMM_CONFIRM"}:
            end_idx = int(turn_index) - 1
            start_idx = max(0, end_idx - (6 - 1))
            tail_turn_ids = [str(turns[i]["turn_id"]) for i in range(start_idx, end_idx + 1)]
            st = ConversationStateV91(
                conversation_id=str(conversation_id),
                state_index=int(state_index),
                prev_state_id=str(prev_state_id),
                active_goals=[],
                bindings={
                    "vars": {str(k): vars_map.get(k) for k in sorted(vars_map.keys(), key=str)},
                    "last_answer": last_answer,
                    "last_intent": str(parse.get("intent_id") or ""),
                    "last_rule_id": str(parse.get("matched_rule_id") or ""),
                    "last_user_text": normalize_text_v91(str(user_text)),
                    "last_assistant_text": str(assistant_text),
                },
                tail_turn_ids=list(tail_turn_ids),
                last_user_turn_id=str(ut.get("turn_id") or ""),
                last_assistant_turn_id=str(at.get("turn_id") or ""),
                created_step=int(step),
                last_step=int(step),
            ).to_dict()
            state_index += 1
            step += 1
            prev_state_id = str(st.get("state_id") or "")
            states.append(dict(st))
            prev_states_hash = append_chained_jsonl_v91(
                states_path,
                {"time": deterministic_iso(step=int(st["created_step"])), "step": int(st["created_step"]), "event": "STATE", "payload": dict(st)},
                prev_hash=prev_states_hash,
            )

        # End condition.
        if str(parse.get("intent_id") or "") == INTENT_END_V91:
            break

    # Verify hash-chains and invariants.
    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v91(turns_path)),
        "parses_chain_ok": bool(verify_chained_jsonl_v91(parses_path)),
        "states_chain_ok": bool(verify_chained_jsonl_v91(states_path)) if os.path.exists(states_path) else True,
        "trials_chain_ok": bool(verify_chained_jsonl_v91(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v91(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v91(transcript_path)),
    }
    ok_chain, chain_reason, chain_details = verify_conversation_chain_v91(
        turns=list(turns),
        states=list(states),
        parse_events=list(parse_events),
        trials=list(trials),
        tail_k=6,
    )

    transcript_hash = compute_transcript_hash_v91(turns)
    state_chain_hash = compute_state_chain_hash_v91(states)
    parse_chain_hash = compute_parse_chain_hash_v91(parse_events)

    verify_obj = {
        "ok": bool(all(chains.values())) and bool(ok_chain),
        "chains": dict(chains),
        "chain_invariants": {"ok": bool(ok_chain), "reason": str(chain_reason), "details": dict(chain_details)},
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
    }
    tmpv = verify_path + ".tmp"
    with open(tmpv, "w", encoding="utf-8") as f:
        f.write(json.dumps(verify_obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpv, verify_path)

    # Freeze manifest per try (used for determinism checks).
    manifest_core = {
        "schema_version": 1,
        "conversation_id": str(conversation_id),
        "seed": int(seed),
        "store_hash": str(store_hash),
        "grammar_hash": str(grammar_snapshot.get("grammar_hash") or ""),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "verify_ok": bool(verify_obj.get("ok", False)),
        "sha256": {
            "store_jsonl": str(sha256_file(store_path)),
            "intent_grammar_snapshot_json": str(sha256_file(grammar_snapshot_path)),
            "conversation_turns_jsonl": str(sha256_file(turns_path)),
            "intent_parses_jsonl": str(sha256_file(parses_path)),
            "conversation_states_jsonl": str(sha256_file(states_path)) if os.path.exists(states_path) else "",
            "dialogue_trials_jsonl": str(sha256_file(trials_path)),
            "objective_evals_jsonl": str(sha256_file(evals_path)),
            "transcript_jsonl": str(sha256_file(transcript_path)),
            "verify_chain_v91_json": str(sha256_file(verify_path)),
        },
    }
    tmpm = manifest_path + ".tmp"
    with open(tmpm, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_core, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmpm, manifest_path)
    ledger_hash = sha256_file(manifest_path)

    # Summary (deterministic, no paths).
    user_turns_total = (len(turns) + 1) // 2 if turns else 0
    parses_ok = sum(1 for p in parse_events if isinstance(p, dict) and isinstance(p.get("payload"), dict) and bool(p["payload"].get("parse_ok", False)))
    clarifications = sum(1 for tr in trials if isinstance(tr, dict) and str(tr.get("objective_kind") or "") == "COMM_ASK_CLARIFY")
    unknowns = sum(1 for p in parse_events if isinstance(p, dict) and isinstance(p.get("payload"), dict) and str(p["payload"].get("intent_id") or "") == INTENT_UNKNOWN_V91)
    core = {
        "schema_version": 1,
        "seed": int(seed),
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "ledger_hash": str(ledger_hash),
        "turns_total": int(len(turns)),
        "user_turns_total": int(user_turns_total),
        "states_total": int(len(states)),
        "parses_total": int(len(parse_events)),
        "parses_ok": int(parses_ok),
        "clarifications": int(clarifications),
        "unknowns": int(unknowns),
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
        "schema_version": 1,
        "out_dir": str(out_dir),
        "conversation_id": str(conversation_id),
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "parse_chain_hash": str(parse_chain_hash),
        "ledger_hash": str(ledger_hash),
        "summary_sha256": str(summary_sha256),
        "paths": {
            "store_jsonl": str(store_path),
            "intent_grammar_snapshot_json": str(grammar_snapshot_path),
            "turns_jsonl": str(turns_path),
            "parses_jsonl": str(parses_path),
            "states_jsonl": str(states_path),
            "trials_jsonl": str(trials_path),
            "evals_jsonl": str(evals_path),
            "transcript_jsonl": str(transcript_path),
            "verify_json": str(verify_path),
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
        },
    }
