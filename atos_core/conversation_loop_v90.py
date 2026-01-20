from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .conversation_actions_v90 import action_concepts_for_dsl_v90
from .conversation_objectives_v90 import COMM_OBJECTIVES_V90, comm_objective_ids_v90, make_comm_objective_eq_text_v90
from .conversation_v90 import (
    ConversationStateV90,
    TurnV90,
    append_chained_jsonl_v90,
    compute_state_chain_hash_v90,
    compute_transcript_hash_v90,
    normalize_text_v90,
    text_sig_v90,
    verify_chained_jsonl_v90,
    verify_conversation_chain_v90,
)
from .goal_supports_v89 import (
    SupportClaimV89,
    fold_support_stats_v89,
    list_supporting_concepts_for_goal_v89,
    make_goal_support_evidence_event_v89,
)
from .objective_v88 import execute_objective_csv_v88
from .engine_v80 import EngineV80
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


def _sha256_canon(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _stable_hash_obj(obj: Any) -> str:
    return _sha256_canon(obj)


TAIL_K_V90 = 6


def _parse_user_command_v90(text: str) -> Dict[str, Any]:
    """
    Deterministic micro-DSL parser:
      SET <k> <v>
      GET <k>
      ADD <a> <b>
      SUMMARY
      END
    """
    t = normalize_text_v90(text)
    toks = [x for x in t.split(" ") if x]
    if not toks:
        return {"ok": False, "reason": "empty", "op": ""}
    op = str(toks[0]).upper()
    if op == "SET" and len(toks) >= 3:
        return {"ok": True, "op": "SET", "k": str(toks[1]), "v": str(toks[2])}
    if op == "GET" and len(toks) >= 2:
        return {"ok": True, "op": "GET", "k": str(toks[1])}
    if op == "ADD" and len(toks) >= 3:
        return {"ok": True, "op": "ADD", "a": str(toks[1]), "b": str(toks[2])}
    if op == "SUMMARY" and len(toks) == 1:
        return {"ok": True, "op": "SUMMARY"}
    if op == "END" and len(toks) == 1:
        return {"ok": True, "op": "END"}
    return {"ok": False, "reason": "bad_syntax", "op": str(op)}


def _is_int_literal(s: str) -> bool:
    ss = str(s or "")
    return bool(ss) and ss.isdigit()


def _get_int_from_binding(*, vars_map: Dict[str, Any], key: str, last_answer: Any) -> Tuple[Optional[int], str]:
    k = str(key or "")
    if _is_int_literal(k):
        return int(k), "ok"
    if k == "last_answer":
        if isinstance(last_answer, bool) or last_answer is None:
            return None, "missing_last_answer"
        try:
            return int(last_answer), "ok"
        except Exception:
            return None, "bad_last_answer"
    if k in vars_map:
        v = vars_map.get(k)
        if isinstance(v, bool) or v is None:
            return None, "missing_var"
        try:
            return int(v), "ok"
        except Exception:
            return None, "bad_var_type"
    return None, "missing_key"


def _summarize_bindings_v90(*, vars_map: Dict[str, Any], last_answer: Any) -> str:
    parts: List[str] = []
    for k in sorted(vars_map.keys(), key=str):
        v = vars_map.get(k)
        parts.append(f"{str(k)}={str(v)}")
    if last_answer is not None and last_answer != "":
        parts.append(f"last_answer={str(last_answer)}")
    return "Resumo: " + "; ".join(parts)


def _choose_objective_kind_v90(*, parsed: Dict[str, Any], vars_map: Dict[str, Any], last_answer: Any) -> Tuple[str, Dict[str, Any]]:
    """
    Deterministic objective selection (Rule 0..6).
    Returns (objective_kind, ctx).
    """
    if not bool(parsed.get("ok", False)):
        return "COMM_CORRECT", {"reason": str(parsed.get("reason") or "parse_fail"), "missing_key": ""}

    op = str(parsed.get("op") or "")
    if op == "SUMMARY":
        return "COMM_SUMMARIZE", {"missing_key": ""}
    if op == "END":
        return "COMM_END", {"missing_key": ""}
    if op == "GET":
        k = str(parsed.get("k") or "")
        if k not in vars_map:
            return "COMM_ASK_CLARIFY", {"missing_key": str(k)}
        return "COMM_RESPOND", {"missing_key": ""}
    if op == "SET":
        return "COMM_RESPOND", {"missing_key": ""}
    if op == "ADD":
        a = str(parsed.get("a") or "")
        b = str(parsed.get("b") or "")
        va, ra = _get_int_from_binding(vars_map=vars_map, key=a, last_answer=last_answer)
        if va is None:
            return "COMM_ASK_CLARIFY", {"missing_key": str(a), "reason": str(ra)}
        vb, rb = _get_int_from_binding(vars_map=vars_map, key=b, last_answer=last_answer)
        if vb is None:
            return "COMM_ASK_CLARIFY", {"missing_key": str(b), "reason": str(rb)}
        return "COMM_RESPOND", {"missing_key": ""}
    return "COMM_ADMIT_UNKNOWN", {"missing_key": "", "reason": "no_rule"}


def _build_expected_and_action_inputs_v90(
    *,
    objective_kind: str,
    parsed: Dict[str, Any],
    vars_map: Dict[str, Any],
    last_answer: Any,
    missing_key: str,
) -> Tuple[str, Dict[str, Any], str]:
    """
    Returns (expected_text, action_inputs, preferred_action_id_hint).
    """
    ok = bool(parsed.get("ok", False))
    op = str(parsed.get("op") or "")
    if objective_kind == "COMM_CORRECT":
        msg = normalize_text_v90(str(parsed.get("op") or "") + ":" + str(parsed.get("reason") or ""))
        return f"Comando inválido: {msg}", {"msg": msg}, "concept_v90_correct_user_v0"
    if objective_kind == "COMM_ADMIT_UNKNOWN":
        return "Não sei.", {}, "concept_v90_admit_unknown_v0"
    if objective_kind == "COMM_END":
        return "Encerrado.", {}, "concept_v90_end_conversation_v0"
    if objective_kind == "COMM_ASK_CLARIFY":
        k = str(missing_key or "")
        return f"Qual é o valor de {k}?", {"k": k}, "concept_v90_ask_clarify_v0"
    if objective_kind == "COMM_SUMMARIZE":
        summ = _summarize_bindings_v90(vars_map=vars_map, last_answer=last_answer)
        return summ, {"text": summ}, "concept_v90_emit_text_v0"
    if objective_kind == "COMM_CONFIRM":
        t = normalize_text_v90(str(parsed.get("raw") or ""))
        return f"Entendi: {t}", {"text": f"Entendi: {t}"}, "concept_v90_emit_text_v0"

    # COMM_RESPOND
    if ok and op == "SET":
        k = str(parsed.get("k") or "")
        v = str(parsed.get("v") or "")
        return f"OK: {k}={v}", {"k": k, "v": v}, "concept_v90_confirm_set_v0"
    if ok and op == "GET":
        k = str(parsed.get("k") or "")
        v = vars_map.get(k)
        return f"{k}={v}", {"text": f"{k}={v}"}, "concept_v90_emit_text_v0"
    if ok and op == "ADD":
        a = str(parsed.get("a") or "")
        b = str(parsed.get("b") or "")
        va, _ = _get_int_from_binding(vars_map=vars_map, key=a, last_answer=last_answer)
        vb, _ = _get_int_from_binding(vars_map=vars_map, key=b, last_answer=last_answer)
        s = int(va or 0) + int(vb or 0)
        return f"SUM={s}", {"sum": str(s)}, "concept_v90_emit_sum_v0"
    return "Não sei.", {}, "concept_v90_admit_unknown_v0"


def _rank_action_candidates_v90(
    *,
    candidates: Sequence[Tuple[str, SupportClaimV89]],
    events: Sequence[Dict[str, Any]],
    goal_id: str,
) -> List[Tuple[str, SupportClaimV89, float, float]]:
    scored: List[Tuple[str, SupportClaimV89, float, float]] = []
    for act_id, claim in candidates:
        stats = fold_support_stats_v89(events=events, goal_id=str(goal_id), concept_key=str(act_id), claim=claim)
        scored.append((str(act_id), claim, float(stats.expected_success), float(stats.expected_cost)))
    # Deterministic ordering: expected_success desc, expected_cost asc, act_id asc.
    scored.sort(key=lambda t: (-float(t[2]), float(t[3]), str(t[0])))
    return scored


def run_conversation_v90(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
) -> Dict[str, Any]:
    ensure_absent(str(out_dir))
    os.makedirs(str(out_dir), exist_ok=False)

    # Artifacts (WORM write-once; jsonl append-only hash-chained).
    store_path = os.path.join(str(out_dir), "store.jsonl")
    turns_path = os.path.join(str(out_dir), "conversation_turns.jsonl")
    states_path = os.path.join(str(out_dir), "conversation_states.jsonl")
    trials_path = os.path.join(str(out_dir), "dialogue_trials.jsonl")
    evals_path = os.path.join(str(out_dir), "objective_evals.jsonl")
    transcript_path = os.path.join(str(out_dir), "transcript.jsonl")
    summary_path = os.path.join(str(out_dir), "summary.json")
    verify_path = os.path.join(str(out_dir), "verify_chain_v90.json")
    manifest_path = os.path.join(str(out_dir), "freeze_manifest_v90.json")

    store = ActStore()

    # Objective acts (communicative).
    obj_ids = comm_objective_ids_v90()
    for okind, oid in sorted(obj_ids.items(), key=lambda kv: str(kv[0])):
        store.add(make_comm_objective_eq_text_v90(objective_id=str(oid), objective_kind=str(okind), created_step=0))

    # Action concept acts.
    goal_ids = {k: str(k) for k in COMM_OBJECTIVES_V90}
    for act in action_concepts_for_dsl_v90(goal_ids=goal_ids):
        store.add(act)

    # WORM store snapshot.
    if os.path.exists(store_path):
        _fail(f"store_path_exists:{store_path}")
    store.save_jsonl(store_path)
    store_hash = store.content_hash()

    engine = EngineV80(store, seed=int(seed))

    # Conversation identity (deterministic).
    conversation_id = f"conv_v90_{sha256_hex(canonical_json_dumps(list(user_turn_texts)).encode('utf-8'))}"

    # Runtime state (explicit bindings).
    vars_map: Dict[str, Any] = {}
    last_answer: Any = ""

    turns: List[Dict[str, Any]] = []
    states: List[Dict[str, Any]] = []
    transcript: List[Dict[str, Any]] = []

    prev_turns_hash: Optional[str] = None
    prev_states_hash: Optional[str] = None
    prev_trials_hash: Optional[str] = None
    prev_evals_hash: Optional[str] = None
    prev_transcript_hash: Optional[str] = None

    support_events: List[Dict[str, Any]] = []

    prev_state_id = ""
    turn_index = 0
    state_index = 0
    step = 0

    for user_text in list(user_turn_texts):
        # (1) observe_turn(user)
        ut = TurnV90(
            conversation_id=str(conversation_id),
            turn_index=int(turn_index),
            role="user",
            text=str(user_text),
            created_step=int(step),
            offset_us=0,
        ).to_dict()
        turn_index += 1
        step += 1
        turns.append(dict(ut))
        prev_turns_hash = append_chained_jsonl_v90(
            turns_path,
            {"time": deterministic_iso(step=int(ut["created_step"])), "step": int(ut["created_step"]), "event": "TURN", "payload": dict(ut)},
            prev_hash=prev_turns_hash,
        )
        transcript.append({"role": "user", "text": str(ut.get("text") or ""), "turn_id": str(ut.get("turn_id") or "")})
        prev_transcript_hash = append_chained_jsonl_v90(
            transcript_path,
            {"time": deterministic_iso(step=int(ut["created_step"])), "step": int(ut["created_step"]), "event": "UTTERANCE", "payload": dict(transcript[-1])},
            prev_hash=prev_transcript_hash,
        )

        # (2) update bindings deterministically.
        parsed = _parse_user_command_v90(str(user_text))
        missing_key = ""
        if bool(parsed.get("ok", False)) and str(parsed.get("op") or "") == "SET":
            k = str(parsed.get("k") or "")
            v = str(parsed.get("v") or "")
            vars_map[str(k)] = v if not _is_int_literal(v) else int(v)
        # For ADD/GET missing, we only set missing_key later.

        # (3) choose objective
        objective_kind, ctx = _choose_objective_kind_v90(parsed=parsed, vars_map=dict(vars_map), last_answer=last_answer)
        if objective_kind == "COMM_ASK_CLARIFY":
            missing_key = str(ctx.get("missing_key") or "")

        # (4) choose action concept supporting objective
        goal_id = str(objective_kind)
        candidates = list_supporting_concepts_for_goal_v89(store=store, goal_id=str(goal_id))
        ranked = _rank_action_candidates_v90(candidates=candidates, events=support_events, goal_id=str(goal_id))

        # (5) build expected text and action inputs
        expected_text, action_inputs, hint_action_id = _build_expected_and_action_inputs_v90(
            objective_kind=str(objective_kind),
            parsed=dict(parsed, raw=str(user_text)),
            vars_map=dict(vars_map),
            last_answer=last_answer,
            missing_key=str(missing_key),
        )
        expected_sig = text_sig_v90(expected_text)

        # Fallback cascade if no candidates: ask_clarify -> admit_unknown -> end
        fallback_order = ["COMM_ASK_CLARIFY", "COMM_ADMIT_UNKNOWN", "COMM_END"]
        tried_objectives: List[str] = []

        def _objective_act_id(okind: str) -> str:
            return str(obj_ids.get(okind) or "")

        def _execute_action(act_id: str, inputs: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any], float]:
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
                goal_kind=str(objective_kind),
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

        assistant_text = ""
        chosen_action_id = ""
        chosen_objective_id = _objective_act_id(str(objective_kind))
        eval_row: Dict[str, Any] = {}
        trial_row: Dict[str, Any] = {}

        def _try_objective_and_actions(okind: str) -> bool:
            nonlocal assistant_text, chosen_action_id, chosen_objective_id, eval_row, trial_row, expected_text, expected_sig, action_inputs, hint_action_id, objective_kind
            objective_kind = str(okind)
            chosen_objective_id = _objective_act_id(str(okind))
            # Rebuild expected/action inputs for fallback objective kinds.
            expected_text, action_inputs, hint_action_id = _build_expected_and_action_inputs_v90(
                objective_kind=str(okind),
                parsed=dict(parsed, raw=str(user_text)),
                vars_map=dict(vars_map),
                last_answer=last_answer,
                missing_key=str(missing_key),
            )
            expected_sig = text_sig_v90(expected_text)

            cand2 = list_supporting_concepts_for_goal_v89(store=store, goal_id=str(okind))
            ranked2 = _rank_action_candidates_v90(candidates=cand2, events=support_events, goal_id=str(okind))
            # Hint: prefer matching action id first if present.
            ordered_action_ids: List[str] = []
            if hint_action_id:
                ordered_action_ids.append(str(hint_action_id))
            for aid, _cl, _es, _ec in ranked2:
                if str(aid) not in set(ordered_action_ids):
                    ordered_action_ids.append(str(aid))

            for act_id in ordered_action_ids:
                ok_exec, out_text, meta, cost_used = _execute_action(str(act_id), dict(action_inputs))
                action_sig = text_sig_v90(out_text)
                eval_id = _stable_hash_obj(
                    {
                        "objective_id": str(chosen_objective_id),
                        "objective_kind": str(okind),
                        "act_id": str(act_id),
                        "expected_sig": str(expected_sig),
                        "output_sig": str(action_sig),
                    }
                )
                if not ok_exec:
                    verdict_ok = False
                    verdict = {"ok": False, "score": 0, "reason": f"action_exec_failed:{str(meta.get('reason') or '')}", "details": {"meta": dict(meta)}}
                else:
                    # (6) evaluate assistant_text with objective CSV
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

                # (7) record eval + trial (WORM)
                eval_row = {
                    "kind": "objective_eval_v90",
                    "time": deterministic_iso(step=int(step)),
                    "step": int(step),
                    "eval_id": str(eval_id),
                    "objective_kind": str(okind),
                    "objective_id": str(chosen_objective_id),
                    "expected_text_sig": str(expected_sig),
                    "output_text_sig": str(action_sig),
                    "verdict": dict(verdict),
                }
                nonlocal prev_evals_hash
                prev_evals_hash = append_chained_jsonl_v90(evals_path, dict(eval_row), prev_hash=prev_evals_hash)

                trial_id = _stable_hash_obj(
                    {
                        "conversation_id": str(conversation_id),
                        "step": int(step),
                        "objective_kind": str(okind),
                        "objective_id": str(chosen_objective_id),
                        "act_id": str(act_id),
                        "eval_id": str(eval_id),
                    }
                )
                trial_row = {
                    "kind": "dialogue_trial_v90",
                    "time": deterministic_iso(step=int(step)),
                    "step": int(step),
                    "trial_id": str(trial_id),
                    "conversation_id": str(conversation_id),
                    "objective_kind": str(okind),
                    "objective_id": str(chosen_objective_id),
                    "action_concept_id": str(act_id),
                    "expected_text": str(expected_text),
                    "expected_text_sig": str(expected_sig),
                    "assistant_text": str(out_text),
                    "assistant_text_sig": str(action_sig),
                    "ok": bool(verdict_ok),
                    "cost_used": float(cost_used),
                }
                nonlocal prev_trials_hash
                prev_trials_hash = append_chained_jsonl_v90(trials_path, dict(trial_row), prev_hash=prev_trials_hash)

                # Also append a supports(G) evidence event for scoring (V89-compatible schema).
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
                    return True
            return False

        ok_done = _try_objective_and_actions(str(objective_kind))
        tried_objectives.append(str(objective_kind))
        if not ok_done:
            for fb in fallback_order:
                if fb in set(tried_objectives):
                    continue
                ok_done = _try_objective_and_actions(str(fb))
                tried_objectives.append(str(fb))
                if ok_done:
                    break

        if not ok_done:
            # Last resort: end.
            _try_objective_and_actions("COMM_END")

        # Update last_answer deterministically from ADD success.
        if bool(parsed.get("ok", False)) and str(parsed.get("op") or "") == "ADD" and assistant_text.startswith("SUM="):
            try:
                last_answer = int(assistant_text.split("=", 1)[1])
            except Exception:
                pass

        # (9) create assistant turn
        at = TurnV90(
            conversation_id=str(conversation_id),
            turn_index=int(turn_index),
            role="assistant",
            text=str(assistant_text),
            created_step=int(step),
            offset_us=1,
            objective_id=str(chosen_objective_id),
            action_concept_id=str(chosen_action_id),
            eval_id=str(eval_row.get("eval_id") or ""),
        ).to_dict()
        turn_index += 1
        step += 1
        turns.append(dict(at))
        prev_turns_hash = append_chained_jsonl_v90(
            turns_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "TURN", "payload": dict(at)},
            prev_hash=prev_turns_hash,
        )
        transcript.append({"role": "assistant", "text": str(at.get("text") or ""), "turn_id": str(at.get("turn_id") or "")})
        prev_transcript_hash = append_chained_jsonl_v90(
            transcript_path,
            {"time": deterministic_iso(step=int(at["created_step"])), "step": int(at["created_step"]), "event": "UTTERANCE", "payload": dict(transcript[-1])},
            prev_hash=prev_transcript_hash,
        )

        # (10) create new conversation state
        end_idx = int(turn_index) - 1
        start_idx = max(0, end_idx - (TAIL_K_V90 - 1))
        tail_turn_ids = [str(turns[i]["turn_id"]) for i in range(start_idx, end_idx + 1)]
        st = ConversationStateV90(
            conversation_id=str(conversation_id),
            state_index=int(state_index),
            prev_state_id=str(prev_state_id),
            active_goals=[],
            bindings={
                "vars": {str(k): vars_map.get(k) for k in sorted(vars_map.keys(), key=str)},
                "last_answer": last_answer,
                "last_command": str(parsed.get("op") or ""),
                "missing_key": str(missing_key),
                "last_user_text": normalize_text_v90(str(user_text)),
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
        prev_states_hash = append_chained_jsonl_v90(
            states_path,
            {"time": deterministic_iso(step=int(st["created_step"])), "step": int(st["created_step"]), "event": "STATE", "payload": dict(st)},
            prev_hash=prev_states_hash,
        )

        if normalize_text_v90(str(user_text)).upper() == "END":
            break

    # Verify chains and invariants.
    chains = {
        "turns_chain_ok": bool(verify_chained_jsonl_v90(turns_path)),
        "states_chain_ok": bool(verify_chained_jsonl_v90(states_path)),
        "trials_chain_ok": bool(verify_chained_jsonl_v90(trials_path)),
        "evals_chain_ok": bool(verify_chained_jsonl_v90(evals_path)),
        "transcript_chain_ok": bool(verify_chained_jsonl_v90(transcript_path)),
    }
    ok_chain, chain_reason, chain_details = verify_conversation_chain_v90(
        turns=list(turns), states=list(states), tail_k=int(TAIL_K_V90)
    )

    transcript_hash = compute_transcript_hash_v90(turns)
    state_chain_hash = compute_state_chain_hash_v90(states)

    verify_obj = {
        "ok": bool(all(chains.values())) and bool(ok_chain),
        "chains": dict(chains),
        "chain_invariants": {"ok": bool(ok_chain), "reason": str(chain_reason), "details": dict(chain_details)},
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
    }
    tmp = verify_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(verify_obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, verify_path)

    # Freeze manifest per try (used for determinism checks).
    manifest_core = {
        "schema_version": 1,
        "conversation_id": str(conversation_id),
        "seed": int(seed),
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "verify_ok": bool(verify_obj.get("ok", False)),
        "sha256": {
            "store_jsonl": str(sha256_file(store_path)),
            "conversation_turns_jsonl": str(sha256_file(turns_path)),
            "conversation_states_jsonl": str(sha256_file(states_path)),
            "dialogue_trials_jsonl": str(sha256_file(trials_path)),
            "objective_evals_jsonl": str(sha256_file(evals_path)),
            "transcript_jsonl": str(sha256_file(transcript_path)),
            "verify_chain_v90_json": str(sha256_file(verify_path)),
        },
    }
    tmp2 = manifest_path + ".tmp"
    with open(tmp2, "w", encoding="utf-8") as f:
        f.write(json.dumps(manifest_core, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp2, manifest_path)
    ledger_hash = sha256_file(manifest_path)

    # Summary (deterministic, no paths).
    core = {
        "schema_version": 1,
        "seed": int(seed),
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "ledger_hash": str(ledger_hash),
        "turns_total": int(len(turns)),
        "states_total": int(len(states)),
        "verify_ok": bool(verify_obj.get("ok", False)),
    }
    summary_sha256 = sha256_hex(canonical_json_dumps(core).encode("utf-8"))
    summary = dict(core, summary_sha256=str(summary_sha256))
    tmp3 = summary_path + ".tmp"
    with open(tmp3, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp3, summary_path)

    return {
        "schema_version": 1,
        "out_dir": str(out_dir),
        "conversation_id": str(conversation_id),
        "store_hash": str(store_hash),
        "transcript_hash": str(transcript_hash),
        "state_chain_hash": str(state_chain_hash),
        "ledger_hash": str(ledger_hash),
        "summary_sha256": str(summary_sha256),
        "paths": {
            "store_jsonl": str(store_path),
            "turns_jsonl": str(turns_path),
            "states_jsonl": str(states_path),
            "trials_jsonl": str(trials_path),
            "evals_jsonl": str(evals_path),
            "transcript_jsonl": str(transcript_path),
            "verify_json": str(verify_path),
            "manifest_json": str(manifest_path),
            "summary_json": str(summary_path),
        },
    }
