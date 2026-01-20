from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .act import estimate_act_cost_bits
from .engine import Engine
from .validators import run_validator


def _hash_obj(obj: Any) -> str:
    try:
        return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))
    except Exception:
        return sha256_hex(str(obj).encode("utf-8"))


def _value_to_text(v: Any) -> str:
    if isinstance(v, (dict, list, tuple)):
        return canonical_json_dumps(v)
    if v is None:
        return ""
    return str(v)


def _output_type_from_act(act: Act) -> str:
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
    out_schema = dict(out_schema)
    if len(out_schema) != 1:
        return ""
    return str(next(iter(out_schema.values())) or "")


def _type_name(v: Any) -> str:
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, int):
        return "int"
    if isinstance(v, str):
        return "str"
    if isinstance(v, dict):
        return "dict"
    if isinstance(v, list):
        return "list"
    return str(type(v).__name__)


@dataclass(frozen=True)
class PlanStep:
    concept_id: str
    bind: Dict[str, str]  # concept input name -> env var name
    out_var: str

    def to_dict(self) -> Dict[str, Any]:
        return {"concept_id": str(self.concept_id), "bind": dict(self.bind), "out_var": str(self.out_var)}


@dataclass(frozen=True)
class PlanResult:
    ok: bool
    reason: str
    plan: List[PlanStep]
    best_cost: int
    expanded: int
    pruned: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "reason": str(self.reason),
            "plan": [p.to_dict() for p in self.plan],
            "best_cost": int(self.best_cost),
            "expanded": int(self.expanded),
            "pruned": int(self.pruned),
        }


def search_plan(
    *,
    engine: Engine,
    concept_acts: Sequence[Act],
    available_inputs: Dict[str, Any],
    target_output_type: str,
    validator_id: str,
    expected: Any,
    expected_output_text: str,
    max_depth: int = 4,
    max_expansions: int = 5000,
) -> PlanResult:
    """
    Deterministic planner via bounded best-first search (program synthesis over concept_csv acts).

    - State = environment of typed values.
    - Action = call a concept_csv with bound inputs (by type) to produce one new value.
    - Goal test = produced output passes validator and matches expected_output_text.

    Notes:
    - Uses Engine.execute_concept_csv(... validate_output=False) for intermediate steps.
    - Final goal is checked via validator + exact output_text match.
    """
    max_depth = max(0, int(max_depth))
    max_expansions = max(1, int(max_expansions))

    # Normalize concept set (stable order).
    cands: List[Act] = [a for a in concept_acts if str(getattr(a, "kind", "")) == "concept_csv" and bool(getattr(a, "active", True))]
    cands.sort(key=lambda a: str(a.id))

    # Initial env.
    env0: Dict[str, Dict[str, Any]] = {}
    for k in sorted(list(available_inputs.keys())):
        v = available_inputs.get(k)
        env0[str(k)] = {"type": _type_name(v), "value": v, "text": _value_to_text(v), "sig": _hash_obj(v)}

    def _env_sig(env: Dict[str, Dict[str, Any]]) -> str:
        items = [(k, env[k]["type"], env[k]["sig"]) for k in sorted(env.keys())]
        return _hash_obj(items)

    def _goal_test(env: Dict[str, Dict[str, Any]]) -> Optional[str]:
        want_type = str(target_output_type or "")
        for name in sorted(env.keys()):
            rec = env[name]
            if want_type and str(rec.get("type") or "") != want_type:
                continue
            out_text = str(rec.get("text") or "")
            if out_text != str(expected_output_text or ""):
                continue
            vres = run_validator(str(validator_id or ""), out_text, expected) if validator_id else None
            if vres is not None and not bool(vres.passed):
                continue
            return name
        return None

    # Priority queue: (cost_bits, depth, plan_sig, env_sig, env, plan)
    q: List[Tuple[int, int, str, str, Dict[str, Dict[str, Any]], List[PlanStep]]] = []
    start_sig = _env_sig(env0)
    heapq.heappush(q, (0, 0, "", start_sig, env0, []))
    seen_cost: Dict[Tuple[int, str], int] = {(0, start_sig): 0}

    expanded = 0
    pruned = 0

    if _goal_test(env0) is not None:
        return PlanResult(True, "already_satisfied", [], 0, 0, 0)

    while q and expanded < max_expansions:
        cost, depth, plan_sig, env_sig, env, plan = heapq.heappop(q)
        expanded += 1

        if depth >= max_depth:
            continue

        # Expand by applying any concept to any compatible bindings.
        for act in cands:
            ev = act.evidence if isinstance(act.evidence, dict) else {}
            iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
            iface = iface if isinstance(iface, dict) else {}
            in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
            in_schema = dict(in_schema)
            out_type = _output_type_from_act(act)
            if not out_type:
                continue

            # Find binding candidates by type match (keys are semantic roles, vars can differ).
            keys = sorted(list(in_schema.keys()))
            # Small, deterministic backtracking for bindings.
            bindings: List[Dict[str, str]] = [{}]
            for k in keys:
                want_t = str(in_schema.get(k) or "")
                next_bindings: List[Dict[str, str]] = []
                for b in bindings:
                    for var_name in sorted(env.keys()):
                        if str(env[var_name]["type"]) != want_t:
                            continue
                        b2 = dict(b)
                        b2[str(k)] = str(var_name)
                        next_bindings.append(b2)
                bindings = next_bindings
                if not bindings:
                    break
            if not bindings:
                continue

            for bind in bindings:
                # Execute concept to obtain value (deterministic).
                inps = {k: env[var]["value"] for k, var in bind.items()}
                r = engine.execute_concept_csv(
                    concept_act_id=str(act.id),
                    inputs=dict(inps),
                    expected=None,
                    step=0,
                    validate_output=False,
                )
                meta = r.get("meta") if isinstance(r, dict) else {}
                meta = meta if isinstance(meta, dict) else {}
                if not bool(meta.get("ok", False)):
                    pruned += 1
                    continue
                out_val = r.get("output")
                out_text = str(meta.get("output_text") or _value_to_text(out_val))
                out_sig = _hash_obj(out_val)

                out_var = f"v{depth}_{len(plan)}_{str(act.id)[:8]}"
                if out_var in env:
                    pruned += 1
                    continue
                env2 = dict(env)
                env2[out_var] = {"type": str(out_type), "value": out_val, "text": out_text, "sig": out_sig}
                env2_sig = _env_sig(env2)
                plan2 = list(plan) + [PlanStep(concept_id=str(act.id), bind=dict(bind), out_var=str(out_var))]
                plan2_sig = _hash_obj([p.to_dict() for p in plan2])
                step_cost_bits = int(estimate_act_cost_bits(act))
                cost2 = int(cost) + int(step_cost_bits)

                # Goal check.
                if _goal_test(env2) is not None:
                    return PlanResult(True, "ok", plan2, int(cost2), expanded, pruned)

                key = (depth + 1, env2_sig)
                prev = seen_cost.get(key)
                if prev is not None and int(prev) <= int(cost2):
                    pruned += 1
                    continue
                seen_cost[key] = int(cost2)
                heapq.heappush(q, (int(cost2), depth + 1, plan2_sig, env2_sig, env2, plan2))

    return PlanResult(False, "no_plan", [], -1, expanded, pruned)
