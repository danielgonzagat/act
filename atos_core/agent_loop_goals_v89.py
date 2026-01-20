from __future__ import annotations

import hashlib
import json
import os
import datetime as _dt
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex
from .engine_v80 import EngineV80
from .goal_act_v75 import list_goal_acts_v75
from .goal_objective_v88 import evaluate_goal_success_v88
from .goal_policy_v89 import (
    ANTI_REPEAT_COOLDOWN_STEPS_V89,
    COOLDOWN_STEPS_V89,
    CandidateScoreV89,
    apply_anti_repeat_v89,
    candidate_expected_score_v89,
    edge_key_v89,
    goal_priority_v89,
    rank_candidates_v89,
    should_abandon_goal_v89,
    should_cooldown_edge_v89,
)
from .goal_supports_v89 import (
    SupportClaimV89,
    SupportStatsV89,
    append_chained_jsonl_v89,
    fold_support_stats_v89,
    list_supporting_concepts_for_goal_v89,
    make_goal_support_evidence_event_v89,
    verify_chained_jsonl_v89,
)


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


def goal_created_step_v89(goal_act: Act) -> int:
    """
    Deterministic created_step derived from deterministic_iso used in goal act created_at.
    """
    try:
        ts = str(getattr(goal_act, "created_at", "") or "")
        if not ts:
            return 0
        dt = _dt.datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=_dt.timezone.utc)
        epoch = _dt.datetime(1970, 1, 1, tzinfo=_dt.timezone.utc)
        return int((dt - epoch).total_seconds())
    except Exception:
        return 0


def _goal_kind(goal_act: Act) -> str:
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    g = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    return str(g.get("goal_kind") or "")


def _goal_status(goal_act: Act) -> str:
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    g = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    st = g.get("state") if isinstance(g.get("state"), dict) else {}
    return str(st.get("status") or "active")


def _goal_body(goal_act: Act) -> Dict[str, Any]:
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    g = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    bindings = g.get("bindings") if isinstance(g.get("bindings"), dict) else {}
    bindings2 = {str(k): bindings.get(k) for k in sorted(bindings.keys(), key=str)}
    return {
        "goal_id": str(getattr(goal_act, "id", "") or ""),
        "goal_kind": str(g.get("goal_kind") or ""),
        "bindings": bindings2,
        "output_key": str(g.get("output_key") or ""),
        "expected": g.get("expected"),
        "validator_id": str(g.get("validator_id") or ""),
        "objective_act_id": str(g.get("objective_act_id") or ""),
    }


def _make_event(kind: str, *, step: int, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "kind": str(kind),
        "time": deterministic_iso(step=int(step)),
        "step": int(step),
        "payload": dict(payload),
    }


def _cooldown_event(*, step: int, goal_id: str, concept_key: str, until_step: int, reason: str) -> Dict[str, Any]:
    return _make_event(
        "goal_support_cooldown_v89",
        step=int(step),
        payload={
            "goal_id": str(goal_id),
            "concept_key": str(concept_key),
            "until_step": int(until_step),
            "reason": str(reason),
        },
    )


def _abandoned_event(*, step: int, goal_id: str, reason: str, attempts: int, cost_total: float) -> Dict[str, Any]:
    return _make_event(
        "goal_abandoned_v89",
        step=int(step),
        payload={
            "goal_id": str(goal_id),
            "reason": str(reason),
            "attempts": int(attempts),
            "cost_total": float(cost_total),
        },
    )


def _satisfied_event(*, step: int, goal_id: str, score: int, reason: str) -> Dict[str, Any]:
    return _make_event(
        "goal_satisfied_v89",
        step=int(step),
        payload={"goal_id": str(goal_id), "score": int(score), "reason": str(reason or "")},
    )


def _no_candidates_event(*, step: int, goal_id: str) -> Dict[str, Any]:
    return _make_event("goal_no_support_candidates_v89", step=int(step), payload={"goal_id": str(goal_id)})


def _selected_event(
    *,
    step: int,
    goal_id: str,
    concept_key: str,
    score: float,
    expected_success: float,
    expected_cost: float,
) -> Dict[str, Any]:
    return _make_event(
        "goal_support_selected_v89",
        step=int(step),
        payload={
            "goal_id": str(goal_id),
            "concept_key": str(concept_key),
            "score": float(score),
            "expected_success": float(expected_success),
            "expected_cost": float(expected_cost),
        },
    )


def _concept_inputs_from_goal(
    *,
    concept_act: Act,
    goal_act: Act,
    last_output: Any,
    step: int,
) -> Dict[str, Any]:
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    g = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    bindings = g.get("bindings") if isinstance(g.get("bindings"), dict) else {}
    bindings2 = {str(k): bindings.get(k) for k in sorted(bindings.keys(), key=str)}
    expected = g.get("expected")

    # Reserved.
    reserved: Dict[str, Any] = {
        "__output": "" if last_output is None else (last_output if isinstance(last_output, (str, dict, list)) else str(last_output)),
        "__goal": dict(_goal_body(goal_act)),
        "__step": int(step),
        "expected": expected,
    }

    iface = (concept_act.evidence or {}).get("interface") if isinstance(concept_act.evidence, dict) else {}
    in_schema = iface.get("input_schema") if isinstance(iface, dict) else {}
    in_schema = in_schema if isinstance(in_schema, dict) else {}

    inps: Dict[str, Any] = {}
    for k in sorted(in_schema.keys(), key=str):
        ks = str(k)
        if ks in bindings2:
            inps[ks] = bindings2.get(ks)
        elif ks in reserved:
            inps[ks] = reserved.get(ks)
        else:
            inps[ks] = None
    return inps


def run_goals_v89(
    *,
    store,
    seed: int,
    out_dir: str,
    max_ticks: int = 50,
) -> Dict[str, Any]:
    """
    V89: means-ends loop over goals using supports(G) and cost-guided deterministic selection.
    This loop is opt-in and does not mutate goals in the store; all state is WORM via events + fold.
    """
    ensure_absent(str(out_dir))
    os.makedirs(str(out_dir), exist_ok=False)

    events_path = os.path.join(str(out_dir), "goals_v89_events.jsonl")
    summary_path = os.path.join(str(out_dir), "summary.json")

    prev_hash: Optional[str] = None
    events: List[Dict[str, Any]] = []

    engine = EngineV80(store, seed=int(seed))

    goal_states: Dict[str, Dict[str, Any]] = {}
    goals = list_goal_acts_v75(store)
    for ga in goals:
        gid = str(getattr(ga, "id", "") or "")
        if not gid:
            continue
        init_status = _goal_status(ga)
        if init_status not in ("active", "satisfied", "failed", "abandoned"):
            init_status = "active"
        goal_states[gid] = {
            "status": str(init_status),
            "attempts": 0,
            "cost_total": 0.0,
            "last_output": "",
            "last_verdict": None,
            "created_step": int(goal_created_step_v89(ga)),
            "parent_goal_id": "",
        }

    cooldown_until: Dict[str, int] = {}
    prev_edge = ""
    prev_repeats = 0
    last_attempt_ok = False

    def _append_event(row: Dict[str, Any]) -> None:
        nonlocal prev_hash
        prev_hash = append_chained_jsonl_v89(events_path, row, prev_hash=prev_hash)
        events.append(dict(row))

    # Initial snapshot event.
    _append_event(_make_event("v89_init", step=0, payload={"goals_total": int(len(goal_states))}))

    for step in range(int(max_ticks)):
        # Evaluate active goals with current last_output (objective CSV if present).
        active_goal_ids: List[str] = []
        for ga in goals:
            gid = str(getattr(ga, "id", "") or "")
            if gid not in goal_states:
                continue
            st = goal_states[gid]
            if str(st.get("status") or "") != "active":
                continue
            verdict, _dbg = evaluate_goal_success_v88(
                store=store,
                seed=int(seed),
                goal_act=ga,
                goal_output=st.get("last_output"),
                step=int(step),
                goal_kind=_goal_kind(ga),
            )
            st["last_verdict"] = verdict.to_dict()
            if bool(verdict.ok):
                st["status"] = "satisfied"
                _append_event(
                    _satisfied_event(step=int(step), goal_id=str(gid), score=int(verdict.score), reason=str(verdict.reason))
                )
            else:
                active_goal_ids.append(str(gid))

        if not active_goal_ids:
            break

        # Choose goal with max priority.
        scored_goals: List[Tuple[float, str]] = []
        for ga in goals:
            gid = str(getattr(ga, "id", "") or "")
            if gid not in set(active_goal_ids):
                continue
            st = goal_states.get(gid, {})
            pr = goal_priority_v89(goal_act=ga, created_step=int(st.get("created_step", 0) or 0), now_step=int(step))
            scored_goals.append((float(pr), str(gid)))
        scored_goals.sort(key=lambda x: (-float(x[0]), str(x[1])))
        target_goal_id = scored_goals[0][1]
        target_goal = next((g for g in goals if str(getattr(g, "id", "") or "") == target_goal_id), None)
        if target_goal is None:
            break

        # Abandonment check before selecting support.
        gs = goal_states.get(target_goal_id, {})
        abandon, ab_reason = should_abandon_goal_v89(
            attempts=int(gs.get("attempts", 0) or 0), cost_total=float(gs.get("cost_total", 0.0) or 0.0)
        )
        if abandon:
            gs["status"] = "abandoned"
            _append_event(
                _abandoned_event(
                    step=int(step),
                    goal_id=str(target_goal_id),
                    reason=str(ab_reason),
                    attempts=int(gs.get("attempts", 0) or 0),
                    cost_total=float(gs.get("cost_total", 0.0) or 0.0),
                )
            )
            continue

        # Choose supporting concept by score.
        candidates = list_supporting_concepts_for_goal_v89(store=store, goal_id=str(target_goal_id))
        if not candidates:
            _append_event(_no_candidates_event(step=int(step), goal_id=str(target_goal_id)))
            continue

        goal_pr = scored_goals[0][0]
        scored_candidates: List[CandidateScoreV89] = []
        for concept_key, claim in candidates:
            ek = edge_key_v89(str(target_goal_id), str(concept_key))
            until = int(cooldown_until.get(str(ek), -1))
            if until >= 0 and int(step) < int(until):
                continue
            stats = fold_support_stats_v89(events=events, goal_id=str(target_goal_id), concept_key=str(concept_key), claim=claim)
            score = candidate_expected_score_v89(goal_priority=float(goal_pr), stats=stats)
            scored_candidates.append(
                CandidateScoreV89(concept_key=str(concept_key), score=float(score), claim=claim, stats=stats)
            )

        ranked = rank_candidates_v89(scored_candidates)
        if not ranked:
            _append_event(_no_candidates_event(step=int(step), goal_id=str(target_goal_id)))
            continue

        chosen = ranked[0]
        _append_event(
            _selected_event(
                step=int(step),
                goal_id=str(target_goal_id),
                concept_key=str(chosen.concept_key),
                score=float(chosen.score),
                expected_success=float(chosen.stats.expected_success),
                expected_cost=float(chosen.stats.expected_cost),
            )
        )

        # Execute exactly one CSG/concept per tick.
        concept_act = store.get_concept_act(str(chosen.concept_key))
        if concept_act is None:
            # Record as failed evidence (concept missing).
            attempt_id = _stable_hash_obj({"step": int(step), "goal_id": str(target_goal_id), "concept_key": str(chosen.concept_key)})
            ev_row = make_goal_support_evidence_event_v89(
                step=int(step),
                goal_id=str(target_goal_id),
                concept_key=str(chosen.concept_key),
                attempt_id=str(attempt_id),
                ok=False,
                cost_used=0.0,
                note="concept_not_found",
            )
            _append_event(ev_row)
            continue

        inps = _concept_inputs_from_goal(
            concept_act=concept_act,
            goal_act=target_goal,
            last_output=gs.get("last_output"),
            step=int(step),
        )
        exec_res = engine.execute_concept_csv(
            concept_act_id=str(chosen.concept_key),
            inputs=dict(inps),
            goal_kind=str(_goal_kind(target_goal)),
            expected=None,
            step=int(step),
            max_depth=8,
            max_events=256,
            validate_output=False,
        )
        meta = exec_res.get("meta") if isinstance(exec_res.get("meta"), dict) else {}
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

        output_text = str(meta.get("output_text") or exec_res.get("output") or "")
        gs["attempts"] = int(gs.get("attempts", 0) or 0) + 1
        gs["cost_total"] = float(gs.get("cost_total", 0.0) or 0.0) + float(cost_used)
        gs["last_output"] = str(output_text)

        verdict, _dbg2 = evaluate_goal_success_v88(
            store=store,
            seed=int(seed),
            goal_act=target_goal,
            goal_output=output_text,
            step=int(step),
            goal_kind=_goal_kind(target_goal),
        )
        gs["last_verdict"] = verdict.to_dict()
        ok_now = bool(verdict.ok)

        attempt_id = _stable_hash_obj({"step": int(step), "goal_id": str(target_goal_id), "concept_key": str(chosen.concept_key)})
        ev_row = make_goal_support_evidence_event_v89(
            step=int(step),
            goal_id=str(target_goal_id),
            concept_key=str(chosen.concept_key),
            attempt_id=str(attempt_id),
            ok=bool(ok_now),
            cost_used=float(cost_used),
            note=str(meta.get("reason") or ""),
        )
        _append_event(ev_row)

        # Anti-repeat: if same edge keeps failing, apply short cooldown.
        ek = edge_key_v89(str(target_goal_id), str(chosen.concept_key))
        prev_repeats2, trigger_cd = apply_anti_repeat_v89(
            prev_edge_key=str(prev_edge),
            prev_repeats=int(prev_repeats),
            cur_edge_key=str(ek),
            last_attempt_ok=bool(ok_now),
        )
        prev_edge = str(ek)
        prev_repeats = int(prev_repeats2)
        last_attempt_ok = bool(ok_now)
        if trigger_cd:
            until = int(step) + int(ANTI_REPEAT_COOLDOWN_STEPS_V89)
            cooldown_until[str(ek)] = int(max(int(cooldown_until.get(str(ek), -1)), int(until)))
            _append_event(
                _cooldown_event(
                    step=int(step),
                    goal_id=str(target_goal_id),
                    concept_key=str(chosen.concept_key),
                    until_step=int(cooldown_until[str(ek)]),
                    reason="anti_repeat",
                )
            )

        # Prune bad edges deterministically (cooldown).
        stats_after = fold_support_stats_v89(
            events=events, goal_id=str(target_goal_id), concept_key=str(chosen.concept_key), claim=chosen.claim
        )
        if should_cooldown_edge_v89(stats=stats_after):
            until = int(step) + int(COOLDOWN_STEPS_V89)
            cooldown_until[str(ek)] = int(max(int(cooldown_until.get(str(ek), -1)), int(until)))
            _append_event(
                _cooldown_event(
                    step=int(step),
                    goal_id=str(target_goal_id),
                    concept_key=str(chosen.concept_key),
                    until_step=int(cooldown_until[str(ek)]),
                    reason="low_expected_success",
                )
            )

        if ok_now:
            gs["status"] = "satisfied"
            _append_event(
                _satisfied_event(step=int(step), goal_id=str(target_goal_id), score=int(verdict.score), reason=str(verdict.reason))
            )
        else:
            abandon2, ab_reason2 = should_abandon_goal_v89(
                attempts=int(gs.get("attempts", 0) or 0), cost_total=float(gs.get("cost_total", 0.0) or 0.0)
            )
            if abandon2:
                gs["status"] = "abandoned"
                _append_event(
                    _abandoned_event(
                        step=int(step),
                        goal_id=str(target_goal_id),
                        reason=str(ab_reason2),
                        attempts=int(gs.get("attempts", 0) or 0),
                        cost_total=float(gs.get("cost_total", 0.0) or 0.0),
                    )
                )

    chains_ok = bool(verify_chained_jsonl_v89(events_path))
    summary = {
        "schema_version": 1,
        "seed": int(seed),
        "max_ticks": int(max_ticks),
        "chains_ok": bool(chains_ok),
        "goals_total": int(len(goal_states)),
        "goals_satisfied": int(len([1 for g in goal_states.values() if str(g.get("status") or "") == "satisfied"])),
        "goals_abandoned": int(len([1 for g in goal_states.values() if str(g.get("status") or "") == "abandoned"])),
        "goal_states": {str(k): dict(v) for k, v in sorted(goal_states.items(), key=lambda kv: str(kv[0]))},
        "sha256": {
            "events_jsonl": str(sha256_file(events_path)),
        },
    }
    tmp = summary_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, summary_path)
    summary["sha256"]["summary_json"] = sha256_file(summary_path)
    return summary
