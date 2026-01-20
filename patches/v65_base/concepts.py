from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .validators import ValidatorResult, run_validator


def sha256_text(s: str) -> str:
    return sha256_hex(str(s).encode("utf-8"))


def stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(row))
        f.write("\n")


def _read_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    if not os.path.exists(path):
        return iter(())
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def append_chained_jsonl(path: str, row: Dict[str, Any], *, prev_hash: Optional[str]) -> str:
    body = dict(row)
    body["prev_hash"] = prev_hash
    entry_hash = sha256_hex(canonical_json_dumps(body).encode("utf-8"))
    body["entry_hash"] = entry_hash
    _append_jsonl(path, body)
    return entry_hash


def verify_chained_jsonl(path: str) -> bool:
    prev = None
    for row in _read_jsonl(path):
        row = dict(row)
        entry_hash = row.pop("entry_hash", None)
        if row.get("prev_hash") != prev:
            return False
        expected = sha256_hex(canonical_json_dumps(row).encode("utf-8"))
        if expected != entry_hash:
            return False
        prev = entry_hash
    return True


def _scan_digits(text: str) -> str:
    m = re.search(r"[0-9]+", str(text))
    if not m:
        return "0"
    return m.group(0)


def _strip_one_leading_zero(digits: str) -> str:
    s = str(digits).strip()
    if len(s) >= 2 and s.startswith("0"):
        return s[1:]
    return s


def _digits_to_int(digits: str) -> int:
    s = str(digits).strip()
    if not s:
        return 0
    return int(s)


def _int_to_digits(n: Any) -> str:
    if isinstance(n, bool):
        return "0"
    return str(int(n))


def _add_int(a: Any, b: Any) -> int:
    if isinstance(a, bool) or isinstance(b, bool):
        return 0
    return int(a) + int(b)


def _make_dict_ab(a: Any, b: Any) -> Dict[str, Any]:
    if isinstance(a, bool) or isinstance(b, bool):
        return {"a": 0, "b": 0}
    return {"a": int(a), "b": int(b)}


def _json_canonical(obj: Any) -> str:
    return canonical_json_dumps(obj)


@dataclass(frozen=True)
class PrimitiveOpSpec:
    op_id: str
    arity: int
    input_types: Tuple[str, ...]
    output_type: str


PRIMITIVE_OPS: Dict[str, Tuple[PrimitiveOpSpec, Any]] = {
    "scan_digits": (PrimitiveOpSpec("scan_digits", 1, ("str",), "str"), _scan_digits),
    "strip_one_leading_zero": (
        PrimitiveOpSpec("strip_one_leading_zero", 1, ("str",), "str"),
        _strip_one_leading_zero,
    ),
    "digits_to_int": (PrimitiveOpSpec("digits_to_int", 1, ("str",), "int"), _digits_to_int),
    "int_to_digits": (PrimitiveOpSpec("int_to_digits", 1, ("int",), "str"), _int_to_digits),
    "add_int": (PrimitiveOpSpec("add_int", 2, ("int", "int"), "int"), _add_int),
    "make_dict_ab": (PrimitiveOpSpec("make_dict_ab", 2, ("int", "int"), "dict"), _make_dict_ab),
    "json_canonical": (PrimitiveOpSpec("json_canonical", 1, ("dict",), "str"), _json_canonical),
}


def execute_unary_pipeline(ops: List[str], x: Any) -> Any:
    v = x
    for op_id in ops:
        spec_fn = PRIMITIVE_OPS.get(op_id)
        if spec_fn is None:
            raise KeyError(f"unknown_op:{op_id}")
        _, fn = spec_fn
        v = fn(v)
    return v


def execute_binary_op(op_id: str, a: Any, b: Any) -> Any:
    spec_fn = PRIMITIVE_OPS.get(op_id)
    if spec_fn is None:
        raise KeyError(f"unknown_op:{op_id}")
    spec, fn = spec_fn
    if int(spec.arity) != 2:
        raise ValueError(f"not_binary_op:{op_id}")
    return fn(a, b)


def infer_unary_io_types(ops: List[str]) -> Optional[Tuple[str, str]]:
    if not ops:
        return None
    first = PRIMITIVE_OPS.get(ops[0])
    if first is None:
        return None
    in_t = first[0].input_types[0]
    cur = first[0].output_type
    for op_id in ops[1:]:
        spec_fn = PRIMITIVE_OPS.get(op_id)
        if spec_fn is None:
            return None
        spec = spec_fn[0]
        if int(spec.arity) != 1:
            return None
        if spec.input_types[0] != cur:
            return None
        cur = spec.output_type
    return in_t, cur


@dataclass(frozen=True)
class ConceptInterface:
    input_schema: Dict[str, str]
    output_schema: Dict[str, str]
    validator_id: str
    preconditions: Dict[str, Any] = field(default_factory=dict)
    postconditions: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_schema": dict(self.input_schema),
            "output_schema": dict(self.output_schema),
            "validator_id": str(self.validator_id),
            "preconditions": dict(self.preconditions),
            "postconditions": dict(self.postconditions),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ConceptInterface":
        return ConceptInterface(
            input_schema=dict(d.get("input_schema", {})),
            output_schema=dict(d.get("output_schema", {})),
            validator_id=str(d.get("validator_id", "")),
            preconditions=dict(d.get("preconditions", {})),
            postconditions=dict(d.get("postconditions", {})),
        )

    def type_signature(self) -> str:
        return stable_hash_obj(
            {"in": self.input_schema, "out": self.output_schema, "validator_id": self.validator_id}
        )


@dataclass(frozen=True)
class ConceptPolicies:
    ema_alpha: float = 0.2
    prune_min_calls: int = 8
    prune_min_lifetime_steps: int = 8
    prune_fail_streak: int = 3
    prune_u_threshold: float = 0.6
    prune_s_threshold: float = 0.35

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ema_alpha": float(self.ema_alpha),
            "prune_min_calls": int(self.prune_min_calls),
            "prune_min_lifetime_steps": int(self.prune_min_lifetime_steps),
            "prune_fail_streak": int(self.prune_fail_streak),
            "prune_u_threshold": float(self.prune_u_threshold),
            "prune_s_threshold": float(self.prune_s_threshold),
        }


@dataclass
class Concept:
    id: str
    created_step: int
    subgraph_ref: Dict[str, Any]
    interface: ConceptInterface
    policies: ConceptPolicies

    alive: bool = True
    calls_total: int = 0
    pass_total: int = 0
    fail_total: int = 0
    fail_streak: int = 0
    last_seen_step: int = 0
    contexts_seen: Dict[str, int] = field(default_factory=dict)

    u_ema: float = 0.5
    pass2_ema: float = 0.25
    k_ema: float = 0.0
    s_t: float = 0.0

    def contexts_distinct(self) -> int:
        return len(self.contexts_seen)

    def lifetime_steps(self, *, step: int) -> int:
        return int(step) - int(self.created_step)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "created_step": int(self.created_step),
            "alive": bool(self.alive),
            "subgraph_ref": self.subgraph_ref,
            "interface": self.interface.to_dict(),
            "policies": self.policies.to_dict(),
            "calls_total": int(self.calls_total),
            "pass_total": int(self.pass_total),
            "fail_total": int(self.fail_total),
            "fail_streak": int(self.fail_streak),
            "last_seen_step": int(self.last_seen_step),
            "contexts_distinct": int(self.contexts_distinct()),
            "u_ema": float(self.u_ema),
            "k_ema": float(self.k_ema),
            "pass2_ema": float(self.pass2_ema),
            "s_t": float(self.s_t),
        }


def concept_id_for(subgraph_ref: Dict[str, Any], interface: ConceptInterface) -> str:
    return stable_hash_obj({"subgraph_ref": subgraph_ref, "interface": interface.to_dict()})


def execute_concept_subgraph(subgraph_ref: Dict[str, Any], inputs: Dict[str, Any]) -> Any:
    kind = str(subgraph_ref.get("kind", ""))
    if kind == "engine_turn_subgraph_v0":
        ids = subgraph_ref.get("executed_predictor_act_ids") or []
        if not isinstance(ids, list):
            return []
        return [str(x) for x in ids if isinstance(x, str)]
    if kind == "unary_pipeline_v0":
        ops = list(subgraph_ref.get("ops", []))
        if len(inputs) != 1:
            raise ValueError("unary_pipeline_requires_1_input")
        x = next(iter(inputs.values()))
        return execute_unary_pipeline(ops, x)
    if kind == "binary_op_v0":
        op_id = str(subgraph_ref.get("op", ""))
        if len(inputs) != 2:
            raise ValueError("binary_op_requires_2_inputs")
        keys = sorted(inputs.keys())
        return execute_binary_op(op_id, inputs[keys[0]], inputs[keys[1]])
    raise ValueError(f"unknown_subgraph_kind:{kind}")


def estimate_subgraph_cost_units(subgraph_ref: Dict[str, Any], output: Any) -> float:
    kind = str(subgraph_ref.get("kind", ""))
    if kind == "engine_turn_subgraph_v0":
        overhead = float(subgraph_ref.get("scan_overhead_units", 0) or 0.0)
        if isinstance(output, list):
            return float(overhead + float(len(output)))
        return 0.0
    # Default: treat concept invocation as a single unit (compressed call).
    return 1.0


@dataclass
class ConceptRegistry:
    run_dir: str
    concepts_path: str = field(init=False)
    evidence_path: str = field(init=False)
    telemetry_path: str = field(init=False)

    _concepts: Dict[str, Concept] = field(default_factory=dict, init=False)
    _concepts_prev_hash: Optional[str] = field(default=None, init=False)
    _evidence_prev_hash: Optional[str] = field(default=None, init=False)
    _telemetry_prev_hash: Optional[str] = field(default=None, init=False)

    def __post_init__(self) -> None:
        os.makedirs(self.run_dir, exist_ok=False)
        self.concepts_path = os.path.join(self.run_dir, "concepts.jsonl")
        self.evidence_path = os.path.join(self.run_dir, "concept_evidence.jsonl")
        self.telemetry_path = os.path.join(self.run_dir, "concept_telemetry.jsonl")

    def concepts(self) -> Iterable[Concept]:
        return self._concepts.values()

    def get(self, concept_id: str) -> Optional[Concept]:
        return self._concepts.get(str(concept_id))

    def alive_concepts(self) -> List[Concept]:
        return [c for c in self._concepts.values() if bool(c.alive)]

    def define(
        self,
        *,
        step: int,
        subgraph_ref: Dict[str, Any],
        interface: ConceptInterface,
        policies: Optional[ConceptPolicies] = None,
        birth_reason: str,
        birth_prior: Optional[Dict[str, Any]] = None,
    ) -> Concept:
        cid = concept_id_for(subgraph_ref, interface)
        if cid in self._concepts:
            return self._concepts[cid]

        pol = policies or ConceptPolicies()
        c = Concept(
            id=cid,
            created_step=int(step),
            subgraph_ref=subgraph_ref,
            interface=interface,
            policies=pol,
            last_seen_step=int(step),
        )
        if birth_prior:
            if "u_ema" in birth_prior:
                c.u_ema = float(birth_prior["u_ema"])
            if "pass2_ema" in birth_prior:
                c.pass2_ema = float(birth_prior["pass2_ema"])
            if "k_ema" in birth_prior:
                c.k_ema = float(birth_prior["k_ema"])
        self._concepts[cid] = c

        self._concepts_prev_hash = append_chained_jsonl(
            self.concepts_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "DEFINE",
                "concept": c.to_dict(),
                "birth_reason": str(birth_reason),
                "birth_prior": dict(birth_prior or {}),
            },
            prev_hash=self._concepts_prev_hash,
        )
        return c

    def _append_state(self, *, step: int, concept: Concept, reason: str) -> None:
        self._concepts_prev_hash = append_chained_jsonl(
            self.concepts_path,
            {
                "time": deterministic_iso(step=int(step)),
                "step": int(step),
                "event": "STATE",
                "concept": concept.to_dict(),
                "reason": str(reason),
            },
            prev_hash=self._concepts_prev_hash,
        )

    def _append_evidence(self, *, step: int, row: Dict[str, Any]) -> None:
        self._evidence_prev_hash = append_chained_jsonl(
            self.evidence_path,
            {"time": deterministic_iso(step=int(step)), "step": int(step), **row},
            prev_hash=self._evidence_prev_hash,
        )

    def _append_telemetry(self, *, step: int, row: Dict[str, Any]) -> None:
        self._telemetry_prev_hash = append_chained_jsonl(
            self.telemetry_path,
            {"time": deterministic_iso(step=int(step)), "step": int(step), **row},
            prev_hash=self._telemetry_prev_hash,
        )

    def _update_scores(
        self,
        *,
        concept: Concept,
        step: int,
        passed: bool,
        cost_used: float,
        context_signature: str,
    ) -> None:
        alpha = float(concept.policies.ema_alpha)
        x_u = 1.0 if bool(passed) else 0.0
        x_k = float(cost_used)
        concept.u_ema = alpha * x_u + (1.0 - alpha) * float(concept.u_ema)
        concept.pass2_ema = alpha * (x_u * x_u) + (1.0 - alpha) * float(concept.pass2_ema)
        concept.k_ema = alpha * x_k + (1.0 - alpha) * float(concept.k_ema)

        concept.calls_total += 1
        if bool(passed):
            concept.pass_total += 1
            concept.fail_streak = 0
        else:
            concept.fail_total += 1
            concept.fail_streak += 1

        concept.last_seen_step = int(step)
        concept.contexts_seen[str(context_signature)] = concept.contexts_seen.get(str(context_signature), 0) + 1

        var_p = max(0.0, float(concept.pass2_ema) - float(concept.u_ema) * float(concept.u_ema))
        var_pen = 1.0 - min(1.0, var_p / 0.25)
        reuse_fac = min(1.0, math.log1p(concept.calls_total) / math.log1p(20.0))
        ctx_fac = min(1.0, math.log1p(concept.contexts_distinct()) / math.log1p(10.0))
        cost_fac = 1.0 / (1.0 + float(concept.k_ema))
        concept.s_t = float(concept.u_ema) * var_pen * reuse_fac * ctx_fac * cost_fac

        self._append_state(step=int(step), concept=concept, reason="score_update")

    def _prune_check(self, *, concept: Concept, step: int) -> bool:
        if not bool(concept.alive):
            return False
        if concept.calls_total < int(concept.policies.prune_min_calls):
            return False
        if concept.lifetime_steps(step=int(step)) < int(concept.policies.prune_min_lifetime_steps):
            return False
        if concept.fail_streak < int(concept.policies.prune_fail_streak):
            return False
        if float(concept.u_ema) >= float(concept.policies.prune_u_threshold):
            return False
        if float(concept.s_t) >= float(concept.policies.prune_s_threshold):
            return False

        concept.alive = False
        self._append_state(step=int(step), concept=concept, reason="pruned")
        self._append_evidence(
            step=int(step),
            row={
                "event": "PRUNE",
                "concept_id": str(concept.id),
                "reason": "thresholds",
                "u_ema": float(concept.u_ema),
                "k_ema": float(concept.k_ema),
                "s_t": float(concept.s_t),
                "calls_total": int(concept.calls_total),
                "fail_streak": int(concept.fail_streak),
            },
        )
        return True

    def call(
        self,
        *,
        step: int,
        concept_id: str,
        inputs: Dict[str, Any],
        expected: Any,
        context_signature: str,
        call_depth: int,
        baseline_cost: float,
        contract_active: bool = False,
    ) -> Tuple[Any, ValidatorResult, float]:
        c = self._concepts[str(concept_id)]

        if bool(contract_active):
            out = execute_concept_subgraph(c.subgraph_ref, inputs)
            vr = run_validator(c.interface.validator_id, out, expected)
            cost_used = float(baseline_cost)
            self._append_telemetry(
                step=int(step),
                row={
                    "event": "CALL_BYPASS_CONTRACT",
                    "concept_id": str(c.id),
                    "context_signature": str(context_signature),
                    "call_depth": int(call_depth),
                    "inputs": inputs,
                    "expected": expected,
                    "output": out,
                    "output_signature": stable_hash_obj(out),
                    "validator_id": str(c.interface.validator_id),
                    "validator_passed": bool(vr.passed),
                    "validator_reason": str(vr.reason),
                    "delta_utility": 0.0,
                    "delta_cost": 0.0,
                    "cost_used": float(cost_used),
                    "baseline_cost": float(baseline_cost),
                },
            )
            return out, vr, cost_used

        out = execute_concept_subgraph(c.subgraph_ref, inputs)
        vr = run_validator(c.interface.validator_id, out, expected)
        cost_used = float(estimate_subgraph_cost_units(c.subgraph_ref, out))

        self._append_evidence(
            step=int(step),
            row={
                "event": "CALL",
                "concept_id": str(c.id),
                "context_signature": str(context_signature),
                "call_depth": int(call_depth),
                "inputs": inputs,
                "expected": expected,
                "output": out,
                "output_signature": stable_hash_obj(out),
                "validator_id": str(c.interface.validator_id),
                "validator_passed": bool(vr.passed),
                "validator_reason": str(vr.reason),
                "cost_used": float(cost_used),
                "baseline_cost": float(baseline_cost),
            },
        )
        self._append_telemetry(
            step=int(step),
            row={
                "event": "CALL",
                "concept_id": str(c.id),
                "concept_type": str(c.interface.type_signature()),
                "context_signature": str(context_signature),
                "call_depth": int(call_depth),
                "inputs": inputs,
                "expected": expected,
                "output": out,
                "output_signature": stable_hash_obj(out),
                "validator_id": str(c.interface.validator_id),
                "validator_passed": bool(vr.passed),
                "validator_reason": str(vr.reason),
                "delta_utility": 1.0 if bool(vr.passed) else -1.0,
                "delta_cost": float(baseline_cost) - float(cost_used),
                "cost_used": float(cost_used),
                "baseline_cost": float(baseline_cost),
            },
        )

        self._update_scores(
            concept=c, step=int(step), passed=bool(vr.passed), cost_used=float(cost_used), context_signature=str(context_signature)
        )
        self._prune_check(concept=c, step=int(step))
        return out, vr, cost_used

    def log_primitives(
        self,
        *,
        step: int,
        subgraph_ref: Dict[str, Any],
        interface: ConceptInterface,
        inputs: Dict[str, Any],
        expected: Any,
        output: Any,
        validator_result: ValidatorResult,
        cost_used: float,
        baseline_cost: float,
        context_signature: str,
        call_depth: int,
        note: str,
    ) -> None:
        self._append_telemetry(
            step=int(step),
            row={
                "event": "PRIMITIVES",
                "context_signature": str(context_signature),
                "call_depth": int(call_depth),
                "subgraph_ref": subgraph_ref,
                "concept_type": str(interface.type_signature()),
                "inputs": inputs,
                "expected": expected,
                "output": output,
                "output_signature": stable_hash_obj(output),
                "validator_id": str(interface.validator_id),
                "validator_passed": bool(validator_result.passed),
                "validator_reason": str(validator_result.reason),
                "delta_utility": 1.0 if bool(validator_result.passed) else -1.0,
                "delta_cost": float(baseline_cost) - float(cost_used),
                "cost_used": float(cost_used),
                "baseline_cost": float(baseline_cost),
                "note": str(note),
            },
        )

    def verify_chains(self) -> Dict[str, bool]:
        return {
            "concepts_jsonl_chain_ok": bool(verify_chained_jsonl(self.concepts_path)),
            "evidence_jsonl_chain_ok": bool(verify_chained_jsonl(self.evidence_path)),
            "telemetry_jsonl_chain_ok": bool(verify_chained_jsonl(self.telemetry_path)),
        }
