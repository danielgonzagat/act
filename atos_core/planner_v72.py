from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .goal_spec_v72 import GoalSpecV72


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _sorted_keys(d: Dict[str, Any]) -> List[str]:
    return [str(k) for k in sorted(d.keys(), key=str)]


def _state_sig(vars_avail: Set[str]) -> str:
    return _stable_hash_obj({"vars": sorted(set(str(v) for v in vars_avail if str(v)))})


@dataclass(frozen=True)
class PlanStepV72:
    step_id: str
    idx: int
    concept_id: str
    bind_map: Dict[str, str]
    produces: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_id": str(self.step_id),
            "idx": int(self.idx),
            "concept_id": str(self.concept_id),
            "bind_map": {str(k): str(self.bind_map.get(k) or "") for k in sorted(self.bind_map.keys(), key=str)},
            "produces": str(self.produces),
        }


@dataclass(frozen=True)
class PlanV72:
    steps: List[PlanStepV72]
    plan_sig: str

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": 1, "steps": [s.to_dict() for s in self.steps], "plan_sig": str(self.plan_sig)}


@dataclass(frozen=True)
class OperatorTemplateV72:
    concept_id: str
    input_keys: List[str]
    output_key: str
    validator_id: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": str(self.concept_id),
            "input_keys": list(self.input_keys),
            "output_key": str(self.output_key),
            "validator_id": str(self.validator_id),
        }


def _operator_templates_from_store(store) -> List[OperatorTemplateV72]:
    ops: List[OperatorTemplateV72] = []
    concept_acts: Iterable[Act]
    try:
        concept_acts = store.concept_acts()
    except Exception:
        concept_acts = []
    for act in concept_acts:
        if act is None or str(getattr(act, "kind", "")) != "concept_csv" or (not bool(getattr(act, "active", True))):
            continue
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
        out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
        validator_id = str(iface.get("validator_id") or "")
        in_keys = _sorted_keys(in_schema)
        out_keys = _sorted_keys(out_schema)
        for out_k in out_keys:
            ops.append(
                OperatorTemplateV72(
                    concept_id=str(act.id),
                    input_keys=list(in_keys),
                    output_key=str(out_k),
                    validator_id=str(validator_id),
                )
            )
    ops.sort(key=lambda o: (str(o.concept_id), str(o.output_key)))
    return ops


def _enumerate_bind_maps(
    *,
    input_keys: Sequence[str],
    vars_avail: Sequence[str],
    max_maps: int = 64,
) -> List[Dict[str, str]]:
    """
    Deterministic bind_map enumeration:
      - Prefer identity bindings when possible.
      - Otherwise allow mapping each input slot to an available var.
      - Enforce distinct var bindings when input_keys are distinct.
    """
    avail = [str(v) for v in vars_avail if str(v)]
    avail = sorted(set(avail))
    if not input_keys:
        return [{}]
    keys = [str(k) for k in input_keys]

    # Candidate vars per slot: identity-first then others.
    cand_per: List[List[str]] = []
    for k in keys:
        cands = []
        if k in avail:
            cands.append(k)
        cands.extend([v for v in avail if v != k])
        # Stable unique.
        seen: Set[str] = set()
        out: List[str] = []
        for v in cands:
            if v not in seen:
                seen.add(v)
                out.append(v)
        cand_per.append(out)

    out_maps: List[Dict[str, str]] = []

    def _rec(i: int, cur: Dict[str, str], used: Set[str]) -> None:
        if len(out_maps) >= int(max_maps):
            return
        if i >= len(keys):
            out_maps.append(dict(cur))
            return
        slot = keys[i]
        for v in cand_per[i]:
            # Enforce distinctness across different input slots.
            if slot not in cur and v in used:
                continue
            cur[slot] = v
            used2 = set(used)
            used2.add(v)
            _rec(i + 1, cur, used2)
            cur.pop(slot, None)
            if len(out_maps) >= int(max_maps):
                return

    _rec(0, {}, set())

    # Deterministic order by canonical json.
    out_maps.sort(key=lambda m: canonical_json_dumps({str(k): str(m.get(k) or "") for k in sorted(m.keys(), key=str)}))
    return out_maps


def _apply_operator(
    *,
    op: OperatorTemplateV72,
    vars_avail: Set[str],
) -> Optional[Tuple[Set[str], Dict[str, str]]]:
    if str(op.output_key) in vars_avail:
        return None
    # Slot-causal binding (minimal, deterministic):
    # each required input slot key must already exist in vars_avail, and bind_map is identity.
    if not vars_avail:
        return None
    req = [str(k) for k in (op.input_keys or []) if str(k)]
    if any(k not in vars_avail for k in req):
        return None
    bm0 = {str(k): str(k) for k in req}
    nxt = set(vars_avail)
    nxt.add(str(op.output_key))
    return nxt, bm0


@dataclass
class PlannerV72:
    max_depth: int = 6
    max_expansions: int = 256

    def plan(self, *, goal_spec: GoalSpecV72, store) -> Tuple[Optional[PlanV72], Dict[str, Any]]:
        """
        Deterministic small search over interface-defined operators.
        Goal: produce goal_spec.output_key from initial bindings.
        """
        ops = _operator_templates_from_store(store)
        init_vars = set(str(k) for k in (goal_spec.bindings or {}).keys() if str(k))
        target = str(goal_spec.output_key or "")
        debug: Dict[str, Any] = {
            "init_vars": sorted(init_vars),
            "target": str(target),
            "operators_total": int(len(ops)),
            "expanded": 0,
            "found": False,
        }
        if not target:
            return None, {**debug, "reason": "missing_target"}
        if target in init_vars:
            # Already satisfied: empty plan.
            plan_body = {"schema_version": 1, "steps": []}
            psig = _stable_hash_obj(plan_body)
            return PlanV72(steps=[], plan_sig=psig), {**debug, "found": True, "reason": "already_satisfied"}

        # Frontier entries: (cost, depth, state_sig, vars, steps_so_far_as_templates+bindmaps)
        frontier: List[Tuple[int, int, str, Set[str], List[Tuple[OperatorTemplateV72, Dict[str, str]]]]] = []
        frontier.append((0, 0, _state_sig(init_vars), set(init_vars), []))
        seen: Set[str] = set()

        def _push(ent: Tuple[int, int, str, Set[str], List[Tuple[OperatorTemplateV72, Dict[str, str]]]]) -> None:
            frontier.append(ent)
            frontier.sort(key=lambda x: (int(x[0]), int(x[1]), str(x[2])))

        while frontier and debug["expanded"] < int(self.max_expansions):
            cost, depth, sig, vars_avail, path = frontier.pop(0)
            if sig in seen:
                continue
            seen.add(sig)
            debug["expanded"] = int(debug["expanded"]) + 1

            if target in vars_avail:
                # Build deterministic PlanV72.
                steps: List[PlanStepV72] = []
                for idx, (op, bm) in enumerate(path):
                    step_body = {
                        "idx": int(idx),
                        "concept_id": str(op.concept_id),
                        "bind_map": {str(k): str(bm.get(k) or "") for k in sorted(bm.keys(), key=str)},
                        "produces": str(op.output_key),
                    }
                    step_id = _stable_hash_obj(step_body)
                    steps.append(
                        PlanStepV72(
                            step_id=str(step_id),
                            idx=int(idx),
                            concept_id=str(op.concept_id),
                            bind_map=dict(step_body["bind_map"]),
                            produces=str(op.output_key),
                        )
                    )
                plan_body = {"schema_version": 1, "steps": [s.to_dict() for s in steps]}
                psig = _stable_hash_obj(plan_body)
                debug["found"] = True
                debug["reason"] = "ok"
                debug["plan_sig"] = str(psig)
                return PlanV72(steps=steps, plan_sig=str(psig)), debug

            if int(depth) >= int(self.max_depth):
                continue

            for op in ops:
                applied = _apply_operator(op=op, vars_avail=set(vars_avail))
                if applied is None:
                    continue
                nxt_vars, bm = applied
                nxt_sig = _state_sig(nxt_vars)
                nxt_path = list(path) + [(op, dict(bm))]
                _push((int(cost) + 1, int(depth) + 1, nxt_sig, nxt_vars, nxt_path))

        return None, {**debug, "reason": "not_found_or_budget_exhausted"}
