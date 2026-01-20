from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from .act import canonical_json_dumps, sha256_hex
from .goal_spec_v72 import GoalSpecV72


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
class PlanStepTraceV73:
    idx: int
    concept_id: str
    bind_map: Dict[str, str]
    produces: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "idx", int(self.idx or 0))
        object.__setattr__(self, "concept_id", str(self.concept_id or ""))
        object.__setattr__(self, "produces", str(self.produces or ""))
        bm = self.bind_map if isinstance(self.bind_map, dict) else {}
        try:
            bm2 = copy.deepcopy(bm)
        except Exception:
            bm2 = dict(bm)
        if not isinstance(bm2, dict):
            bm2 = {}
        # Stable stringification + ordering on serialization only.
        object.__setattr__(self, "bind_map", {str(k): str(bm2.get(k) or "") for k in bm2.keys()})

    def to_canonical_dict(self) -> Dict[str, Any]:
        return {
            "idx": int(self.idx),
            "concept_id": str(self.concept_id),
            "bind_map": {str(k): str(self.bind_map.get(k) or "") for k in sorted(self.bind_map.keys(), key=str)},
            "produces": str(self.produces),
        }


@dataclass(frozen=True)
class TraceV73:
    context_id: str
    goal_sig: str
    goal_id: str
    goal_kind: str
    bindings: Dict[str, Any]
    output_key: str
    expected: Any
    validator_id: str
    steps: List[PlanStepTraceV73]
    outcome: Dict[str, Any]
    cost_units: Dict[str, Any]

    schema_version: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "schema_version", int(self.schema_version or 1))
        object.__setattr__(self, "context_id", str(self.context_id or ""))
        object.__setattr__(self, "goal_sig", str(self.goal_sig or ""))
        object.__setattr__(self, "goal_id", str(self.goal_id or ""))
        object.__setattr__(self, "goal_kind", str(self.goal_kind or ""))
        object.__setattr__(self, "output_key", str(self.output_key or ""))
        object.__setattr__(self, "validator_id", str(self.validator_id or ""))

        b = self.bindings if isinstance(self.bindings, dict) else {}
        object.__setattr__(self, "bindings", _safe_deepcopy(b) if isinstance(b, dict) else {})

        steps = self.steps if isinstance(self.steps, list) else []
        steps2 = [s for s in steps if isinstance(s, PlanStepTraceV73)]
        object.__setattr__(self, "steps", list(steps2))

        outcome = self.outcome if isinstance(self.outcome, dict) else {}
        object.__setattr__(self, "outcome", _safe_deepcopy(outcome) if isinstance(outcome, dict) else {})

        cu = self.cost_units if isinstance(self.cost_units, dict) else {}
        object.__setattr__(self, "cost_units", _safe_deepcopy(cu) if isinstance(cu, dict) else {})

    def acts_path(self) -> List[str]:
        return [str(s.concept_id) for s in self.steps if str(s.concept_id)]

    def to_canonical_dict(self, *, include_sig: bool) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "schema_version": int(self.schema_version),
            "context_id": str(self.context_id),
            "goal_sig": str(self.goal_sig),
            "goal_id": str(self.goal_id),
            "goal_kind": str(self.goal_kind),
            "bindings": {str(k): self.bindings.get(k) for k in sorted(self.bindings.keys(), key=str)},
            "output_key": str(self.output_key),
            "expected": self.expected,
            "validator_id": str(self.validator_id),
            "steps": [s.to_canonical_dict() for s in self.steps],
            "acts_path": [str(x) for x in self.acts_path()],
            "outcome": dict(self.outcome),
            "cost_units": dict(self.cost_units),
        }
        if include_sig:
            body["trace_sig"] = _stable_hash_obj(body)
        return body

    def trace_sig(self) -> str:
        return _stable_hash_obj(self.to_canonical_dict(include_sig=False))


def context_id_from_bindings(bindings: Dict[str, Any]) -> str:
    b = bindings if isinstance(bindings, dict) else {}
    sig = _stable_hash_obj({str(k): b.get(k) for k in sorted(b.keys(), key=str)})
    return f"ctx_v73_{sig}"


def trace_from_agent_loop_v72(*, goal_spec: GoalSpecV72, result: Dict[str, Any]) -> TraceV73:
    plan = result.get("plan") if isinstance(result, dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    raw_steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []

    steps: List[PlanStepTraceV73] = []
    for s in raw_steps:
        if not isinstance(s, dict):
            continue
        idx = int(s.get("idx", 0) or 0)
        concept_id = str(s.get("concept_id") or "")
        bind_map = s.get("bind_map") if isinstance(s.get("bind_map"), dict) else {}
        produces = str(s.get("produces") or "")
        steps.append(
            PlanStepTraceV73(
                idx=int(idx),
                concept_id=str(concept_id),
                bind_map={str(k): str(bind_map.get(k) or "") for k in bind_map.keys()},
                produces=str(produces),
            )
        )

    final = result.get("final") if isinstance(result.get("final"), dict) else {}
    final = final if isinstance(final, dict) else {}
    ok = bool(result.get("ok", False))
    got = str(final.get("got") or "")
    expected = goal_spec.expected

    outcome = {"ok": bool(ok), "got": str(got), "expected": expected}
    cost_units = {"steps_total": int(len(steps))}

    bindings = goal_spec.bindings if isinstance(goal_spec.bindings, dict) else {}
    ctx_id = context_id_from_bindings(bindings)
    goal_sig = str(goal_spec.goal_sig())
    goal_id = str(goal_spec.goal_id())

    return TraceV73(
        context_id=str(ctx_id),
        goal_sig=str(goal_sig),
        goal_id=str(goal_id),
        goal_kind=str(goal_spec.goal_kind),
        bindings=_safe_deepcopy(bindings) if isinstance(bindings, dict) else {},
        output_key=str(goal_spec.output_key),
        expected=expected,
        validator_id=str(goal_spec.validator_id),
        steps=list(steps),
        outcome=dict(outcome),
        cost_units=dict(cost_units),
    )

