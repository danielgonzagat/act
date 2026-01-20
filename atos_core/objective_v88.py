from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from .concepts import PRIMITIVE_OPS, PrimitiveOpSpec
from .engine_v80 import EngineV80
from .store import ActStore


OBJECTIVE_KIND_V88 = "objective_csv_v88"


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _safe_deepcopy(obj: Any) -> Any:
    try:
        return copy.deepcopy(obj)
    except Exception:
        if isinstance(obj, dict):
            return dict(obj)
        if isinstance(obj, list):
            return list(obj)
        return obj


@dataclass(frozen=True)
class ObjectiveVerdictV88:
    ok: bool
    score: int
    reason: str
    details: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "score": int(self.score),
            "reason": str(self.reason),
            "details": dict(self.details) if isinstance(self.details, dict) else {},
        }


def normalize_objective_verdict_v88(raw: Any) -> ObjectiveVerdictV88:
    """
    Canonical verdict normalization:
      - bool -> ok=bool, score=1/0, reason="", details={}
      - dict -> extract keys (ok/score/reason/details) with deterministic defaults
      - other -> fail-closed with deterministic reason
    """
    if isinstance(raw, bool):
        return ObjectiveVerdictV88(ok=bool(raw), score=1 if bool(raw) else 0, reason="", details={})

    if isinstance(raw, dict):
        ok = bool(raw.get("ok", False))
        score_raw = raw.get("score", 1 if ok else 0)
        try:
            score = int(score_raw)
        except Exception:
            score = 1 if ok else 0
        reason = str(raw.get("reason") or "")
        details_raw = raw.get("details")
        details = details_raw if isinstance(details_raw, dict) else {}
        # Ensure JSON-ish determinism: keys sorted by insertion at dumps time (canonical_json_dumps uses sort_keys).
        return ObjectiveVerdictV88(ok=ok, score=score, reason=reason, details=dict(details))

    tname = str(type(raw).__name__)
    return ObjectiveVerdictV88(
        ok=False,
        score=0,
        reason=f"invalid_objective_output_type:{tname}",
        details={"got_type": tname},
    )


def ensure_objective_primitives_registered_v88() -> None:
    """
    Deterministic, idempotent extension of PRIMITIVE_OPS with V88 primitives.
    Keeps V88 opt-in (only active when imported/called).
    """
    if "objective_eq_text_v88" in PRIMITIVE_OPS:
        return

    def _objective_eq_text_v88(a: Any, b: Any) -> Dict[str, Any]:
        sa = "" if a is None else str(a)
        sb = "" if b is None else str(b)
        ok = sa == sb
        return {
            "ok": bool(ok),
            "score": 1 if ok else 0,
            "reason": "" if ok else "mismatch",
            "details": {"got": sa, "expected": sb},
        }

    PRIMITIVE_OPS["objective_eq_text_v88"] = (
        PrimitiveOpSpec("objective_eq_text_v88", 2, ("str", "str"), "dict"),
        _objective_eq_text_v88,
    )


def make_objective_eq_text_act_v88(*, created_step: int = 0, match: Optional[Dict[str, Any]] = None) -> Act:
    """
    Example objective: textual equality on reserved input "__output" and field "expected".
    Returns a dict verdict (ObjectiveVerdictV88-shaped).
    """
    ensure_objective_primitives_registered_v88()

    prog = [
        Instruction("CSV_GET_INPUT", {"name": "__output", "out": "__output"}),
        Instruction("CSV_GET_INPUT", {"name": "expected", "out": "expected"}),
        Instruction("CSV_PRIMITIVE", {"fn": "objective_eq_text_v88", "in": ["__output", "expected"], "out": "verdict"}),
        Instruction("CSV_RETURN", {"var": "verdict"}),
    ]
    iface = {
        "input_schema": {"__output": "str", "expected": "str"},
        "output_schema": {"verdict": "dict"},
        "validator_id": "",
    }
    body = {
        "schema_version": 1,
        "kind": OBJECTIVE_KIND_V88,
        "match": dict(match) if isinstance(match, dict) else {},
        "interface": dict(iface),
        "program": [ins.to_dict() for ins in prog],
        "objective": {"schema_version": 1, "output_type": "ObjectiveVerdictV88"},
    }
    obj_id = f"objective_csv_v88_{_stable_hash_obj(body)}"
    return Act(
        id=str(obj_id),
        version=1,
        created_at=deterministic_iso(step=int(created_step)),
        kind=OBJECTIVE_KIND_V88,  # type: ignore[assignment]
        match=dict(match) if isinstance(match, dict) else {},
        program=list(prog),
        evidence={
            "interface": dict(iface),
            "objective_v88": {"schema_version": 1, "output_type": "ObjectiveVerdictV88"},
        },
        cost={},
        deps=[],
        active=True,
    )


class _OverlayStoreForObjective:
    """
    Minimal overlay so EngineV80 sees objective act as a concept_csv at execution time,
    without mutating the base store.
    """

    def __init__(self, base: ActStore, *, objective_concept_act: Act):
        self._base = base
        self._override = {str(objective_concept_act.id): objective_concept_act}

    def get(self, act_id: str) -> Optional[Act]:
        aid = str(act_id or "")
        if aid in self._override:
            return self._override[aid]
        return self._base.get(aid)

    def get_concept_act(self, act_id: str) -> Optional[Act]:
        act = self.get(str(act_id))
        if act is None or (not bool(getattr(act, "active", True))):
            return None
        if str(getattr(act, "kind", "")) != "concept_csv":
            return None
        return act


def _objective_as_concept_csv_act_v88(objective_act: Act) -> Act:
    """
    Convert objective_csv_v88 ACT -> concept_csv ACT for EngineV80 execution.
    Does not change id/program/match; only kind and interface normalization.
    """
    ev = objective_act.evidence if isinstance(objective_act.evidence, dict) else {}
    iface = ev.get("interface")
    iface = dict(iface) if isinstance(iface, dict) else {}
    if "input_schema" not in iface or "output_schema" not in iface or "validator_id" not in iface:
        raise ValueError("objective_missing_interface_fields")

    act2 = Act.from_dict(objective_act.to_dict())
    act2.kind = "concept_csv"  # type: ignore[assignment]
    # Ensure interface exists for validate_before_execute.
    ev2 = act2.evidence if isinstance(act2.evidence, dict) else {}
    ev2["interface"] = dict(iface)
    act2.evidence = ev2
    return act2


def execute_objective_csv_v88(
    *,
    store: ActStore,
    seed: int,
    objective_act_id: str,
    inputs: Dict[str, Any],
    step: int,
    goal_kind: str = "",
    max_depth: int = 8,
    max_events: int = 256,
) -> ObjectiveVerdictV88:
    """
    Execute objective_csv_v88 via the same CSV runtime (EngineV80) used for concept_csv.
    """
    ensure_objective_primitives_registered_v88()

    obj_id = str(objective_act_id or "")
    if not obj_id:
        return ObjectiveVerdictV88(ok=False, score=0, reason="missing_objective_act_id", details={})

    act = store.get(obj_id)
    if act is None:
        return ObjectiveVerdictV88(ok=False, score=0, reason="objective_not_found", details={"objective_act_id": obj_id})
    if str(getattr(act, "kind", "")) != OBJECTIVE_KIND_V88:
        return ObjectiveVerdictV88(
            ok=False,
            score=0,
            reason="wrong_objective_kind",
            details={"objective_act_id": obj_id, "kind": str(getattr(act, "kind", ""))},
        )

    try:
        objective_as_concept = _objective_as_concept_csv_act_v88(act)
    except Exception as e:
        return ObjectiveVerdictV88(
            ok=False,
            score=0,
            reason="objective_invalid_interface",
            details={"objective_act_id": obj_id, "error": str(e)},
        )

    overlay = _OverlayStoreForObjective(store, objective_concept_act=objective_as_concept)
    engine = EngineV80(overlay, seed=int(seed))

    inps = dict(inputs) if isinstance(inputs, dict) else {}
    exec_res = engine.execute_concept_csv(
        concept_act_id=str(obj_id),
        inputs=_safe_deepcopy(inps),
        goal_kind=str(goal_kind or ""),
        expected=None,
        step=int(step),
        max_depth=int(max_depth),
        max_events=int(max_events),
        validate_output=False,
    )
    meta = exec_res.get("meta") if isinstance(exec_res.get("meta"), dict) else {}
    if not bool(meta.get("ok", False)):
        return ObjectiveVerdictV88(
            ok=False,
            score=0,
            reason=f"objective_exec_failed:{str(meta.get('reason') or '')}",
            details={"objective_act_id": obj_id, "meta": _safe_deepcopy(meta)},
        )

    raw_out = exec_res.get("output")
    verdict = normalize_objective_verdict_v88(raw_out)
    # Attach deterministic execution evidence without changing semantics.
    trace = exec_res.get("trace") if isinstance(exec_res.get("trace"), dict) else {}
    trace_sig = _stable_hash_obj(trace)
    details = dict(verdict.details) if isinstance(verdict.details, dict) else {}
    if "_objective_exec" not in details:
        details["_objective_exec"] = {
            "objective_act_id": obj_id,
            "goal_kind": str(goal_kind or ""),
            "trace_sig": str(trace_sig),
            "output_sig": str(meta.get("output_sig") or ""),
        }
    return ObjectiveVerdictV88(ok=bool(verdict.ok), score=int(verdict.score), reason=str(verdict.reason), details=details)

