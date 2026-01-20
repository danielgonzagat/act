from __future__ import annotations

import copy
from typing import Any, Dict, Optional, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .objective_v88 import ObjectiveVerdictV88, execute_objective_csv_v88, normalize_objective_verdict_v88
from .validators import run_validator


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


def _goal_summary_v88(goal_act: Act) -> Dict[str, Any]:
    """
    Deterministic summary of a goal act (no nondeterministic fields).
    """
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


def goal_sig_body_v88(goal_act: Act) -> str:
    """
    Stable hash of the goal "body" (excluding mutable state/outcome).
    """
    body = _goal_summary_v88(goal_act)
    # Remove outcome/state fields if present in summary in the future.
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _legacy_goal_verdict_v88(*, goal_act: Act, goal_output: Any) -> ObjectiveVerdictV88:
    """
    Backwards-compatible (pre-V88) goal success:
      - got == expected (string compare) AND validator passes (when validator_id non-empty)
    """
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    g = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}
    expected = g.get("expected")
    validator_id = str(g.get("validator_id") or "")

    got_text = "" if goal_output is None else str(goal_output)
    exp_text = "" if expected is None else str(expected)

    if validator_id:
        v = run_validator(validator_id, got_text, exp_text)
        ok = bool(v.passed) and got_text == exp_text
        reason = "" if ok else str(v.reason or "legacy_mismatch")
        details = {"validator_id": validator_id, "validator_reason": str(v.reason), "got": got_text, "expected": exp_text}
        return ObjectiveVerdictV88(ok=ok, score=1 if ok else 0, reason=reason, details=details)

    ok2 = got_text == exp_text
    return ObjectiveVerdictV88(
        ok=bool(ok2),
        score=1 if ok2 else 0,
        reason="" if ok2 else "legacy_mismatch",
        details={"got": got_text, "expected": exp_text},
    )


def evaluate_goal_success_v88(
    *,
    store,
    seed: int,
    goal_act: Act,
    goal_output: Any,
    step: int,
    goal_kind: Optional[str] = None,
) -> Tuple[ObjectiveVerdictV88, Dict[str, Any]]:
    """
    V88 goal validation:
      - if objective_act_id present: run objective_csv_v88 (same CSV runtime as concept_csv)
      - else: use legacy expected+validator behavior

    Returns (verdict, debug).
    """
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    g = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}

    gk = str(goal_kind or g.get("goal_kind") or "")
    objective_act_id = str(g.get("objective_act_id") or "")

    if objective_act_id:
        bindings = g.get("bindings") if isinstance(g.get("bindings"), dict) else {}
        inputs: Dict[str, Any] = {str(k): bindings.get(k) for k in sorted(bindings.keys(), key=str)}
        # Reserved keys (avoid collisions with bindings by using "__" prefix).
        inputs["__output"] = "" if goal_output is None else (goal_output if isinstance(goal_output, (str, dict, list)) else str(goal_output))
        inputs["__goal"] = _safe_deepcopy(_goal_summary_v88(goal_act))
        inputs["__step"] = int(step)
        # Common convenience passthrough.
        inputs["expected"] = g.get("expected")

        verdict = execute_objective_csv_v88(
            store=store,
            seed=int(seed),
            objective_act_id=str(objective_act_id),
            inputs=inputs,
            step=int(step),
            goal_kind=str(gk),
            max_depth=8,
            max_events=256,
        )
        dbg = {
            "used_objective": True,
            "objective_act_id": str(objective_act_id),
            "goal_kind": str(gk),
            "goal_sig": goal_sig_body_v88(goal_act),
        }
        return verdict, dbg

    verdict2 = _legacy_goal_verdict_v88(goal_act=goal_act, goal_output=goal_output)
    dbg2 = {"used_objective": False, "goal_kind": str(gk), "goal_sig": goal_sig_body_v88(goal_act)}
    return normalize_objective_verdict_v88(verdict2.to_dict()), dbg2

