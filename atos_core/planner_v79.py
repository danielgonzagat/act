from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .act import Act, canonical_json_dumps, sha256_hex
from .goal_spec_v72 import GoalSpecV72
from .match_v79 import is_act_allowed_for_goal_kind


def _stable_hash_obj(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _sorted_keys(d: Dict[str, Any]) -> List[str]:
    return [str(k) for k in sorted(d.keys(), key=str)]


def _state_sig(vars_types: Dict[str, str]) -> str:
    return _stable_hash_obj(
        {
            "vars": [
                (str(k), str(vars_types.get(str(k)) or ""))
                for k in sorted((str(v) for v in vars_types.keys() if str(v)), key=str)
            ]
        }
    )


@dataclass(frozen=True)
class PlanStepV79:
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
class PlanV79:
    steps: List[PlanStepV79]
    plan_sig: str

    def to_dict(self) -> Dict[str, Any]:
        return {"schema_version": 1, "steps": [s.to_dict() for s in self.steps], "plan_sig": str(self.plan_sig)}


@dataclass(frozen=True)
class OperatorTemplateV79:
    concept_id: str
    input_keys: List[str]
    output_key: str
    validator_id: str
    input_types: Dict[str, str]
    output_type: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "concept_id": str(self.concept_id),
            "input_keys": list(self.input_keys),
            "output_key": str(self.output_key),
            "validator_id": str(self.validator_id),
            "input_types": {str(k): str(self.input_types.get(k) or "") for k in sorted(self.input_types.keys(), key=str)},
            "output_type": str(self.output_type),
        }


def _operator_templates_from_store(*, store, goal_kind: str) -> List[OperatorTemplateV79]:
    ops: List[OperatorTemplateV79] = []
    concept_acts: Iterable[Act]
    try:
        concept_acts = store.concept_acts()
    except Exception:
        concept_acts = []
    for act in concept_acts:
        if act is None or str(getattr(act, "kind", "")) != "concept_csv" or (not bool(getattr(act, "active", True))):
            continue
        if not is_act_allowed_for_goal_kind(act=act, goal_kind=str(goal_kind or "")):
            continue
        ev = act.evidence if isinstance(act.evidence, dict) else {}
        iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
        out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
        validator_id = str(iface.get("validator_id") or "")
        in_keys = _sorted_keys(in_schema)
        out_keys = _sorted_keys(out_schema)
        in_types = {str(k): str(in_schema.get(str(k)) or "") for k in in_keys if str(k)}
        for out_k in out_keys:
            ops.append(
                OperatorTemplateV79(
                    concept_id=str(act.id),
                    input_keys=list(in_keys),
                    output_key=str(out_k),
                    validator_id=str(validator_id),
                    input_types=dict(in_types),
                    output_type=str(out_schema.get(str(out_k)) or ""),
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

    out_maps.sort(key=lambda m: canonical_json_dumps({str(k): str(m.get(k) or "") for k in sorted(m.keys(), key=str)}))
    return out_maps


def _apply_operator(
    *,
    op: OperatorTemplateV79,
    vars_types: Dict[str, str],
    target: str = "",
    target_type: str = "",
    allow_target_aliasing: bool = True,
    max_bind_maps: int = 32,
    max_output_aliases: int = 2,
) -> List[Tuple[Dict[str, str], Dict[str, str], str]]:
    """
    Apply an operator template to the current typed state, returning possible next states.

    This planner is intentionally lightweight and deterministic:
      - It does not execute operators; it uses only interface schemas (types).
      - It supports non-identity bindings via bind_map enumeration.
      - It supports output aliasing to avoid the "single output key" deadlock.
    """

    def _fresh_var_name(base: str) -> str:
        b = str(base or "v")
        if b not in vars_types:
            return b
        for i in range(2, 512):
            cand = f"{b}_{i}"
            if cand not in vars_types:
                return cand
        # Deterministic fallback; extremely unlikely under max_depth limits.
        return f"{b}_{_stable_hash_obj(vars_types)[:8]}"

    req_slots = [str(k) for k in (op.input_keys or []) if str(k)]

    # Enumerate candidate bind_maps (typed, deterministic).
    bind_maps: List[Dict[str, str]] = []
    if not req_slots:
        bind_maps = [{}]
    else:
        if not vars_types:
            return []
        # Candidate vars per slot: identity-first, type-filtered.
        cand_per: List[List[str]] = []
        for slot in req_slots:
            want_t = str(op.input_types.get(slot) or "str")
            matches = [str(v) for v, t in vars_types.items() if str(v) and str(t or "str") == want_t]
            matches = sorted(set(matches))
            if not matches:
                return []
            if slot in matches:
                matches = [slot] + [v for v in matches if v != slot]
            cand_per.append(matches)

        out_maps: List[Dict[str, str]] = []

        def _rec(i: int, cur: Dict[str, str], used: Set[str]) -> None:
            if len(out_maps) >= int(max_bind_maps):
                return
            if i >= len(req_slots):
                out_maps.append(dict(cur))
                return
            slot = req_slots[i]
            for v in cand_per[i]:
                # Enforce distinctness across different input slots (stable, reduces branching).
                if slot not in cur and v in used:
                    continue
                cur[slot] = v
                used2 = set(used)
                used2.add(v)
                _rec(i + 1, cur, used2)
                cur.pop(slot, None)
                if len(out_maps) >= int(max_bind_maps):
                    return

        _rec(0, {}, set())
        out_maps.sort(
            key=lambda m: canonical_json_dumps({str(k): str(m.get(k) or "") for k in sorted(m.keys(), key=str)})
        )
        bind_maps = out_maps

    # Output aliasing: allow producing into a fresh var name, and (optionally) into the goal target.
    base_out = str(op.output_key or "value")
    out_vars: List[str] = []
    out_vars.append(_fresh_var_name(base_out))
    if bool(allow_target_aliasing) and str(target) and str(target) not in vars_types and str(target) not in out_vars:
        # Only allow aliasing directly into the goal target when the output type matches the goal.
        if (not str(target_type)) or str(op.output_type or "str") == str(target_type):
            out_vars.insert(0, str(target))
    out_vars = [v for v in out_vars if v and v not in vars_types]
    out_vars = out_vars[: max(1, int(max_output_aliases))]

    out: List[Tuple[Dict[str, str], Dict[str, str], str]] = []
    for bm in bind_maps:
        # Sanity/type check.
        ok = True
        for slot, vname in bm.items():
            want = str(op.input_types.get(str(slot)) or "str")
            got = str(vars_types.get(str(vname)) or "str")
            if want != got:
                ok = False
                break
        if not ok:
            continue
        for produces in out_vars:
            nxt = dict(vars_types)
            nxt[str(produces)] = str(op.output_type or "str")
            out.append((nxt, dict(bm), str(produces)))

    out.sort(
        key=lambda t: (
            _state_sig(t[0]),
            canonical_json_dumps({str(k): str(t[1].get(k) or "") for k in sorted(t[1].keys(), key=str)}),
            str(t[2]),
        )
    )
    return out


@dataclass
class PlannerV79:
    max_depth: int = 6
    max_expansions: int = 256
    # Safety: concept mining can create very wide interface schemas (dozens of inputs).
    # Enumerating bind maps for such operators is combinatorial and can stall training.
    # Keep planning operators "small" by default; large-arity concepts are still usable as
    # direct executors, but not as search operators.
    max_operator_input_keys: int = 12
    # When enabled, allow aliasing an operator output directly into the goal target variable.
    # This helps in general planning (avoids output-key deadlocks), but for trace-mining we
    # sometimes disable it to prevent spurious one-step plans that satisfy types but fail
    # semantic validators.
    allow_target_aliasing: bool = True

    def plan(self, *, goal_spec: GoalSpecV72, store) -> Tuple[Optional[PlanV79], Dict[str, Any]]:
        """
        Deterministic small search over interface-defined operators.
        Match-aware routing: filters concept acts by explicit act.match.goal_kinds metadata.
        """
        ops = _operator_templates_from_store(store=store, goal_kind=str(goal_spec.goal_kind))
        try:
            max_keys = int(getattr(self, "max_operator_input_keys", 0) or 0)
        except Exception:
            max_keys = 0
        if int(max_keys) > 0:
            ops = [o for o in ops if int(len(o.input_keys)) <= int(max_keys)]
        def _infer_type(v: Any) -> str:
            if isinstance(v, bool):
                return "int"
            if isinstance(v, int):
                return "int"
            if isinstance(v, dict):
                return "dict"
            return "str"

        bindings0 = dict(goal_spec.bindings or {}) if isinstance(goal_spec.bindings, dict) else {}
        init_types: Dict[str, str] = {str(k): _infer_type(v) for k, v in bindings0.items() if str(k)}
        target = str(goal_spec.output_key or "")
        # Best-effort output type expectation: plan/state validators always require a string output.
        target_type = ""
        try:
            if str(goal_spec.validator_id or "") in {"plan_validator", "state_validator"}:
                target_type = "str"
        except Exception:
            target_type = ""

        # Prefer operators whose validator_id matches the goal validator_id (best-effort, audit-first).
        goal_validator = str(getattr(goal_spec, "validator_id", "") or "")
        if goal_validator:
            ops.sort(
                key=lambda o: (
                    0 if str(o.validator_id or "") == goal_validator else 1,
                    str(o.concept_id),
                    str(o.output_key),
                )
            )
        debug: Dict[str, Any] = {
            "init_vars": sorted(init_types.keys(), key=str),
            "target": str(target),
            "goal_kind": str(goal_spec.goal_kind),
            "operators_total": int(len(ops)),
            "operators_max_input_keys": int(max_keys),
            "expanded": 0,
            "found": False,
        }
        if not target:
            return None, {**debug, "reason": "missing_target"}
        if target in init_types and (not target_type or str(init_types.get(target) or "") == str(target_type)):
            plan_body = {"schema_version": 1, "steps": []}
            psig = _stable_hash_obj(plan_body)
            return PlanV79(steps=[], plan_sig=psig), {**debug, "found": True, "reason": "already_satisfied"}

        frontier: List[
            Tuple[int, int, str, Dict[str, str], List[Tuple[OperatorTemplateV79, Dict[str, str], str]]]
        ] = []
        frontier.append((0, 0, _state_sig(dict(init_types)), dict(init_types), []))
        seen: Set[str] = set()

        def _push(
            ent: Tuple[int, int, str, Dict[str, str], List[Tuple[OperatorTemplateV79, Dict[str, str], str]]]
        ) -> None:
            frontier.append(ent)
            frontier.sort(key=lambda x: (int(x[0]), int(x[1]), str(x[2])))

        while frontier and debug["expanded"] < int(self.max_expansions):
            cost, depth, sig, vars_types, path = frontier.pop(0)
            if sig in seen:
                continue
            seen.add(sig)
            debug["expanded"] = int(debug["expanded"]) + 1

            if target in vars_types and (not target_type or str(vars_types.get(target) or "") == str(target_type)):
                steps: List[PlanStepV79] = []
                for idx, (op, bm, produces) in enumerate(path):
                    step_body = {
                        "idx": int(idx),
                        "concept_id": str(op.concept_id),
                        "bind_map": {str(k): str(bm.get(k) or "") for k in sorted(bm.keys(), key=str)},
                        "produces": str(produces),
                    }
                    step_id = _stable_hash_obj(step_body)
                    steps.append(
                        PlanStepV79(
                            step_id=str(step_id),
                            idx=int(idx),
                            concept_id=str(op.concept_id),
                            bind_map=dict(step_body["bind_map"]),
                            produces=str(produces),
                        )
                    )
                plan_body = {"schema_version": 1, "steps": [s.to_dict() for s in steps]}
                psig = _stable_hash_obj(plan_body)
                debug["found"] = True
                debug["reason"] = "ok"
                debug["plan_sig"] = str(psig)
                return PlanV79(steps=steps, plan_sig=str(psig)), debug

            if int(depth) >= int(self.max_depth):
                continue

            for op in ops:
                applied = _apply_operator(
                    op=op,
                    vars_types=dict(vars_types),
                    target=str(target),
                    target_type=str(target_type),
                    allow_target_aliasing=bool(getattr(self, "allow_target_aliasing", True)),
                )
                for nxt_types, bm, produces in applied:
                    nxt_sig = _state_sig(dict(nxt_types))
                    nxt_path = list(path) + [(op, dict(bm), str(produces))]
                    _push((int(cost) + 1, int(depth) + 1, nxt_sig, nxt_types, nxt_path))

        return None, {**debug, "reason": "search_exhausted"}
