from __future__ import annotations

import copy
from typing import Any, Dict, List

from .act import Act, canonical_json_dumps, deterministic_iso, sha256_hex


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


def _goal_body(
    *,
    goal_kind: str,
    bindings: Dict[str, Any],
    output_key: str,
    expected: Any,
    validator_id: str,
) -> Dict[str, Any]:
    b = bindings if isinstance(bindings, dict) else {}
    return {
        "goal_kind": str(goal_kind or ""),
        "bindings": {str(k): b.get(k) for k in sorted(b.keys(), key=str)},
        "output_key": str(output_key or ""),
        "expected": expected,
        "validator_id": str(validator_id or ""),
    }


def goal_sig_v75_from_fields(
    *,
    goal_kind: str,
    bindings: Dict[str, Any],
    output_key: str,
    expected: Any,
    validator_id: str,
) -> str:
    body = _goal_body(
        goal_kind=str(goal_kind),
        bindings=dict(bindings) if isinstance(bindings, dict) else {},
        output_key=str(output_key),
        expected=expected,
        validator_id=str(validator_id),
    )
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def goal_sig_v75(act: Act) -> str:
    if act is None or str(getattr(act, "kind", "")) != "goal_v75":
        return ""
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    goal = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    body = _goal_body(
        goal_kind=str(goal.get("goal_kind") or ""),
        bindings=goal.get("bindings") if isinstance(goal.get("bindings"), dict) else {},
        output_key=str(goal.get("output_key") or ""),
        expected=goal.get("expected"),
        validator_id=str(goal.get("validator_id") or ""),
    )
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def make_goal_act_v75(
    *,
    goal_kind: str,
    bindings: Dict[str, Any],
    output_key: str,
    expected: Any,
    validator_id: str,
    created_step: int,
) -> Act:
    goal_kind_s = str(goal_kind or "")
    output_key_s = str(output_key or "")
    validator_id_s = str(validator_id or "text_exact")
    b = bindings if isinstance(bindings, dict) else {}
    b2 = {str(k): b.get(k) for k in sorted(b.keys(), key=str)}

    gsig = goal_sig_v75_from_fields(
        goal_kind=goal_kind_s,
        bindings=b2,
        output_key=output_key_s,
        expected=expected,
        validator_id=validator_id_s,
    )
    goal_id = f"goal_v75_{gsig}"

    ev_goal: Dict[str, Any] = {
        "schema_version": 1,
        "goal_kind": str(goal_kind_s),
        "bindings": dict(b2),
        "output_key": str(output_key_s),
        "expected": expected,
        "validator_id": str(validator_id_s),
        "state": {
            "status": "active",
            "progress": 0.0,
            "last_update_step": int(created_step),
        },
        "outcome": {
            "ok": False,
            "got": "",
            "expected": expected,
            "plan_sig": "",
            "graph_sig": "",
            "steps_total": 0,
            "reason": "",
        },
        "runs": [],
    }

    return Act(
        id=str(goal_id),
        version=1,
        created_at=deterministic_iso(step=int(created_step)),
        kind="goal_v75",
        match={},
        program=[],
        evidence={"goal": ev_goal},
        cost={},
        deps=[],
        active=True,
    )


def goal_v75_is_satisfied(act: Act) -> bool:
    if act is None or str(getattr(act, "kind", "")) != "goal_v75":
        return False
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    goal = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    state = goal.get("state") if isinstance(goal.get("state"), dict) else {}
    return str(state.get("status") or "") == "satisfied"


def goal_v75_update_from_run(
    *,
    act: Act,
    run_res: Dict[str, Any],
    trace_sig: str,
    run_dir_sha256: str,
    step: int,
) -> Act:
    if act is None or str(getattr(act, "kind", "")) != "goal_v75":
        raise ValueError("not_goal_v75_act")

    # Work on a deep snapshot to prevent aliasing.
    base = Act.from_dict(act.to_dict())
    ev = base.evidence if isinstance(base.evidence, dict) else {}
    goal = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}

    plan = run_res.get("plan") if isinstance(run_res.get("plan"), dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    raw_steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

    final = run_res.get("final") if isinstance(run_res.get("final"), dict) else {}
    final = final if isinstance(final, dict) else {}
    validator = final.get("validator") if isinstance(final.get("validator"), dict) else {}

    graph = run_res.get("graph") if isinstance(run_res.get("graph"), dict) else {}
    graph = graph if isinstance(graph, dict) else {}

    ok = bool(run_res.get("ok", False))
    got = str(final.get("got") or "")
    expected = goal.get("expected")

    # Fail-closed satisfaction: validator must pass AND got must match expected text representation.
    validator_passed = bool(validator.get("passed", ok))
    satisfied = bool(ok and validator_passed and got == str(expected))

    reason = str(run_res.get("reason") or "")
    if not reason:
        reason = str(validator.get("reason") or "")

    state = goal.get("state") if isinstance(goal.get("state"), dict) else {}
    outcome = goal.get("outcome") if isinstance(goal.get("outcome"), dict) else {}
    runs = goal.get("runs") if isinstance(goal.get("runs"), list) else []
    runs2 = [r for r in runs if isinstance(r, dict)]

    state2 = dict(state)
    state2["status"] = "satisfied" if satisfied else "active"
    state2["progress"] = 1.0 if satisfied else 0.0
    state2["last_update_step"] = int(step)

    outcome2 = dict(outcome)
    outcome2["ok"] = bool(ok)
    outcome2["got"] = str(got)
    outcome2["expected"] = expected
    outcome2["plan_sig"] = str(plan.get("plan_sig") or "")
    outcome2["graph_sig"] = str(graph.get("graph_sig") or "")
    outcome2["steps_total"] = int(len(raw_steps))
    outcome2["reason"] = str(reason)

    runs2.append({"trace_sig": str(trace_sig or ""), "run_dir_sha256": str(run_dir_sha256 or ""), "step": int(step)})

    goal2 = dict(goal)
    goal2["state"] = _safe_deepcopy(state2)
    goal2["outcome"] = _safe_deepcopy(outcome2)
    goal2["runs"] = _safe_deepcopy(runs2)
    ev["goal"] = _safe_deepcopy(goal2)
    base.evidence = ev
    base.version = int(getattr(act, "version", 1) or 1) + 1
    return base


def list_goal_acts_v75(store) -> List[Act]:
    acts: List[Act] = []
    try:
        vals = store.active()
    except Exception:
        vals = []
    for a in vals:
        if a is None:
            continue
        if str(getattr(a, "kind", "")) == "goal_v75":
            acts.append(a)
    acts.sort(key=lambda a: str(getattr(a, "id", "")))
    return acts
