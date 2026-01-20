from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, sha256_hex
from .conversation_loop_v121 import FAIL_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V121, run_conversation_v121
from .world_pressure_validators_v123 import (
    WORLD_PRESSURE_SCHEMA_VERSION_V123,
    consult_external_world_v123,
    fail_signature_v123,
    validate_exhaustion_with_world_v123,
    validate_historical_regression_v123,
    validate_iac_v123,
    validate_reuse_required_v123,
)


def _ensure_absent(path: str) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")


def _write_once_json(path: str, obj: Any) -> None:
    _ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                out.append(dict(obj))
    return list(out)


def _iac_snapshot_from_run_dir(run_dir: str) -> Dict[str, bool]:
    rd = str(run_dir)
    goal_ok = os.path.exists(os.path.join(rd, "goal_plan_eval_summary_v115.json"))
    plan_ok = os.path.exists(os.path.join(rd, "action_plans.jsonl")) or os.path.exists(os.path.join(rd, "plan_events.jsonl"))
    eval_ok = os.path.exists(os.path.join(rd, "objective_evals.jsonl"))
    consequence_ok = bool(eval_ok) and bool(plan_ok)
    return {
        "goal_ok": bool(goal_ok),
        "plan_ok": bool(plan_ok),
        "eval_ok": bool(eval_ok),
        "consequence_ok": bool(consequence_ok),
    }


def _pick_repeated_fail_reason_v123(*, replan_trace: Dict[str, Any]) -> Tuple[str, int]:
    attempts = replan_trace.get("attempts") if isinstance(replan_trace.get("attempts"), list) else []
    counts: Dict[str, int] = {}
    for a in attempts:
        if not isinstance(a, dict):
            continue
        ok = bool(a.get("ok_final_v116", False))
        if ok:
            continue
        reason = str(a.get("reason_final_v116") or "")
        if not reason:
            continue
        counts[reason] = int(counts.get(reason, 0)) + 1
    for reason, cnt in sorted(counts.items(), key=lambda kv: (-int(kv[1]), str(kv[0]))):
        if int(cnt) >= 2:
            return str(reason), int(cnt)
    return "", 0


def run_conversation_v123(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
    max_plan_attempts: int = 8,
    max_replans_per_turn: int = 3,
    goal_autopilot_total_steps: int = 60,
    external_world_manifest_path: str = "external_world_v122/manifest_v122.json",
) -> Dict[str, Any]:
    """
    V123: "world pressure validators as law" wrapper over V121.

    Minimal hard gates:
      - IAC presence (goal/plan/eval/consequence) as a deterministic contract.
      - Exhaustion consult: if exhausted, consult external world under progress_blocked.
        If any hits exist, exhaustion is treated as not-proved (fail reason overridden).
    """
    od = str(out_dir)
    _ensure_absent(od)

    base = run_conversation_v121(
        user_turn_texts=list(user_turn_texts),
        out_dir=str(od),
        seed=int(seed),
        max_plan_attempts=int(max_plan_attempts),
        max_replans_per_turn=int(max_replans_per_turn),
        goal_autopilot_total_steps=int(goal_autopilot_total_steps),
    )

    fr121_path = os.path.join(od, "final_response_v121.json")
    fr121 = _read_json(fr121_path) if os.path.exists(fr121_path) else {}
    base_ok = bool(fr121.get("ok", False)) if isinstance(fr121, dict) else False
    base_reason = str(fr121.get("reason") or "") if isinstance(fr121, dict) else "missing_final_response_v121"

    iac = _iac_snapshot_from_run_dir(str(od))
    ok_iac, reason_iac = validate_iac_v123(
        goal_ok=bool(iac.get("goal_ok", False)),
        plan_ok=bool(iac.get("plan_ok", False)),
        eval_ok=bool(iac.get("eval_ok", False)),
        consequence_ok=bool(iac.get("consequence_ok", False)),
    )

    consulted: Optional[Dict[str, Any]] = None
    anti_regression: Optional[Dict[str, Any]] = None
    reuse: Optional[Dict[str, Any]] = None
    final_ok = bool(base_ok) and bool(ok_iac)
    final_reason = "ok" if bool(final_ok) else (str(reason_iac) if not bool(ok_iac) else str(base_reason or "fail"))

    progress_blocked = not bool(final_ok)

    # Anti-regression (minimal): if a failure reason repeats across attempts AND we're already blocked,
    # consult the world (forced) and require an explicit causal diff if the world has evidence.
    replan_trace_path = os.path.join(od, "replan_trace_v121.json")
    replan_trace = _read_json(replan_trace_path) if os.path.exists(replan_trace_path) else {}
    rep_reason, rep_count = _pick_repeated_fail_reason_v123(replan_trace=dict(replan_trace) if isinstance(replan_trace, dict) else {})
    if bool(progress_blocked) and rep_reason:
        sig = fail_signature_v123(validator_name="anti_regression_v123", reason_code=str(rep_reason), context={"count": int(rep_count)})
        try:
            rep_consult = consult_external_world_v123(
                manifest_path=str(external_world_manifest_path),
                query=str(rep_reason),
                seed=int(seed),
                turn_index=int(len(list(user_turn_texts))),
                prev_event_sig="",
                out_dir=str(od),
                allowed=True,
                reason_code="progress_blocked",
                limit=3,
                source_filter="engineering_doc",
                artifact_prefix="external_world_antiregression",
            )
            ok_ar, reason_ar = validate_historical_regression_v123(
                repeated=True,
                world_hits_total=int(rep_consult.hits_total),
                # V123 default: no causal diff object is produced by the runtime yet.
                causal_diff_present=False,
            )
            anti_regression = {
                "repeated_reason": str(rep_reason),
                "repeated_count": int(rep_count),
                "fail_signature": str(sig),
                "world_hits_total": int(rep_consult.hits_total),
                "evidence_ids": list(rep_consult.evidence_ids),
                "ok": bool(ok_ar),
                "reason": str(reason_ar),
            }
            if not bool(ok_ar):
                final_ok = False
                final_reason = str(reason_ar)
        except Exception:
            final_ok = False
            final_reason = "historical_world_ignored_v123"

    # If exhausted/budget failure, consult world once (hard pressure).
    is_exhaustion = str(base_reason) in {str(FAIL_REASON_PLAN_SEARCH_BUDGET_EXHAUSTED_V121), "goal_exhausted_v121"} or "exhaust" in str(base_reason)
    if bool(progress_blocked) and bool(is_exhaustion):
        try:
            consult = consult_external_world_v123(
                manifest_path=str(external_world_manifest_path),
                query=str(base_reason),
                seed=int(seed),
                turn_index=int(len(list(user_turn_texts))),
                prev_event_sig="",
                out_dir=str(od),
                allowed=True,
                reason_code="progress_blocked",
                limit=3,
                source_filter="engineering_doc",
                artifact_prefix="external_world_exhaustion",
            )
            consulted = {
                "query": str(consult.query),
                "hits_total": int(consult.hits_total),
                "evidence_ids": list(consult.evidence_ids),
                "external_world_chain_hash_v122": str(consult.external_world_chain_hash_v122),
            }
            ok_exh, reason_exh = validate_exhaustion_with_world_v123(exhausted=True, world_hits_total=int(consult.hits_total))
            if not bool(ok_exh):
                final_ok = False
                final_reason = str(reason_exh)
        except Exception:
            # Fail-closed: world consult required but failed.
            final_ok = False
            final_reason = "historical_world_ignored_v123"

    # Reuse-required (minimal): when blocked (non-exhaustion), consult the world for a reuse hint.
    # If any hits exist, fail-closed unless the runtime records an explicit reuse attempt.
    if bool(progress_blocked) and not bool(is_exhaustion) and not rep_reason:
        try:
            consult = consult_external_world_v123(
                manifest_path=str(external_world_manifest_path),
                query=str(base_reason),
                seed=int(seed),
                turn_index=int(len(list(user_turn_texts))),
                prev_event_sig="",
                out_dir=str(od),
                allowed=True,
                reason_code="progress_blocked",
                limit=3,
                source_filter="engineering_doc",
                artifact_prefix="external_world_reuse",
            )
            ok_reuse, reason_reuse = validate_reuse_required_v123(world_hits_total=int(consult.hits_total), reuse_attempted=False)
            reuse = {
                "query": str(consult.query),
                "hits_total": int(consult.hits_total),
                "evidence_ids": list(consult.evidence_ids),
                "ok": bool(ok_reuse),
                "reason": str(reason_reuse),
            }
            if not bool(ok_reuse):
                final_ok = False
                final_reason = str(reason_reuse)
        except Exception:
            final_ok = False
            final_reason = "historical_world_ignored_v123"

    world_pressure_summary = {
        "schema_version": int(WORLD_PRESSURE_SCHEMA_VERSION_V123),
        "kind": "world_pressure_summary_v123",
        "iac": dict(iac),
        "iac_ok": bool(ok_iac),
        "iac_reason": str(reason_iac),
        "anti_regression": dict(anti_regression) if isinstance(anti_regression, dict) else None,
        "reuse": dict(reuse) if isinstance(reuse, dict) else None,
        "consulted": dict(consulted) if isinstance(consulted, dict) else None,
        "final_ok": bool(final_ok),
        "final_reason": str(final_reason),
        "upstream": {"final_response_v121": dict(fr121) if isinstance(fr121, dict) else {}},
    }
    world_pressure_summary["summary_sig"] = sha256_hex(canonical_json_dumps(world_pressure_summary).encode("utf-8"))
    _write_once_json(os.path.join(od, "world_pressure_summary_v123.json"), dict(world_pressure_summary))

    final_obj = {
        "schema_version": int(WORLD_PRESSURE_SCHEMA_VERSION_V123),
        "kind": "final_response_v123",
        "ok": bool(final_ok),
        "reason": str(final_reason if not bool(final_ok) else "ok"),
        "upstream": {"final_response_v121": dict(fr121) if isinstance(fr121, dict) else {}},
    }
    final_obj["final_sig"] = sha256_hex(canonical_json_dumps(final_obj).encode("utf-8"))
    _write_once_json(os.path.join(od, "final_response_v123.json"), dict(final_obj))

    out = dict(base) if isinstance(base, dict) else {}
    out.update({"final_ok_v123": bool(final_ok), "final_reason_v123": str(final_reason), "iac": dict(iac)})
    return out
