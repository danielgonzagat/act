from __future__ import annotations

import copy
import os
from typing import Any, Dict, List, Sequence

from .act import Act, canonical_json_dumps, sha256_hex
from .ato_v71 import ATOv71
from .engine_v80 import EngineV80
from .mind_graph_v71 import MindGraphV71
from .planner_v79 import PlanV79, PlannerV79
from .render_v71 import render_projection
from .store import ActStore
from .validators import run_validator


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


def _concept_ato_from_act(*, act: Act, step: int, concept_state: str = "ACTIVE") -> ATOv71:
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    in_schema = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
    out_schema = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
    validator_id = str(iface.get("validator_id") or "")
    interface_sig = _stable_hash_obj({"in": in_schema, "out": out_schema, "validator_id": validator_id})
    program_sha256 = _stable_hash_obj([ins.to_dict() for ins in (act.program or [])])
    slots = {"inputs": sorted(str(k) for k in in_schema.keys()), "outputs": sorted(str(k) for k in out_schema.keys())}
    invariants = {
        "input_schema": {str(k): str(v) for k, v in sorted(in_schema.items(), key=lambda kv: str(kv[0]))},
        "output_schema": {str(k): str(v) for k, v in sorted(out_schema.items(), key=lambda kv: str(kv[0]))},
        "validator_id": str(validator_id),
    }
    subgraph = {
        "interface_sig": str(interface_sig),
        "program_sha256": str(program_sha256),
        "program_len": int(len(act.program or [])),
        "concept_state": str(concept_state),
    }
    return ATOv71(
        ato_id=str(act.id),
        ato_type="CONCEPT",
        subgraph=subgraph,
        slots=slots,
        bindings={},
        cost=float(len(act.program or [])),
        evidence_refs=[{"kind": "concept_act", "act_id": str(act.id)}],
        invariants=invariants,
        created_step=int(step),
        last_step=int(step),
    )


def _ingest_concept_calls_as_edges(
    *,
    mg: MindGraphV71,
    store: ActStore,
    step_base: int,
    concept_calls: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    calls = [c for c in concept_calls if isinstance(c, dict)]
    created_nodes = 0
    created_edges = 0
    stack: Dict[int, str] = {}

    for idx, ce in enumerate(calls):
        cid = str(ce.get("concept_id") or "")
        if not cid:
            continue
        if cid not in [n.get("ato_id") for n in (mg.snapshot_graph_state().get("nodes") or []) if isinstance(n, dict)]:
            act = store.get_concept_act(cid)
            if act is not None:
                mg.add_node(step=int(step_base) + int(idx), ato=_concept_ato_from_act(act=act, step=int(step_base)), reason="from_concept_call")
                created_nodes += 1

        depth = int(ce.get("call_depth", 0) or 0)
        if depth > 0 and (depth - 1) in stack:
            caller = str(stack.get(depth - 1) or "")
            callee = cid
            if caller and callee and caller != callee:
                ev_ref = {
                    "kind": "concept_call",
                    "idx": int(idx),
                    "call_depth": int(depth),
                    "bindings_sig": str(ce.get("bindings_sig") or ""),
                    "return_sig": str(ce.get("return_sig") or ""),
                    "concept_sig": str(ce.get("concept_sig") or ""),
                    "interface_sig": str(ce.get("interface_sig") or ""),
                    "program_sha256": str(ce.get("program_sha256") or ""),
                    "blocked": bool(ce.get("blocked", False)),
                    "blocked_reason": str(ce.get("blocked_reason") or ""),
                    "goal_kind": str(ce.get("goal_kind") or ""),
                    "bindings": ce.get("bindings") if isinstance(ce.get("bindings"), dict) else {},
                    "return": ce.get("return") if isinstance(ce.get("return"), dict) else {},
                }
                mg.add_edge(
                    step=int(step_base) + int(idx),
                    src_ato_id=str(caller),
                    dst_ato_id=str(callee),
                    edge_type="CALLS",
                    evidence_refs=[_safe_deepcopy(ev_ref)],
                    reason="from_engine_concept_calls",
                )
                created_edges += 1

        stack[depth] = cid
        for k in list(stack.keys()):
            if k > depth:
                stack.pop(k, None)

    return {"nodes_added": int(created_nodes), "edges_added": int(created_edges)}


def run_goal_spec_v80(
    *,
    goal_spec,
    store: ActStore,
    seed: int,
    out_dir: str,
    max_depth: int = 6,
    max_expansions: int = 256,
    max_events: int = 512,
) -> Dict[str, Any]:
    """
    V80 agent loop:
      GOAL_SPEC -> MATCH-AWARE PLANNER -> MATCH-ENFORCED EXECUTOR -> MIND_GRAPH (WORM) -> RENDER/RESPONSE
    """
    planner = PlannerV79(max_depth=int(max_depth), max_expansions=int(max_expansions))
    plan, planner_debug = planner.plan(goal_spec=goal_spec, store=store)
    if plan is None:
        return {"ok": False, "reason": "plan_not_found", "planner": dict(planner_debug)}

    mg_dir = f"{str(out_dir).rstrip(os.sep)}/mind_graph"
    mg = MindGraphV71(mg_dir)

    step_ctr = 0
    goal_id = goal_spec.goal_id()
    goal_sig = goal_spec.goal_sig()
    plan_id = f"plan_v80_{str(plan.plan_sig)}"

    goal_node = ATOv71(
        ato_id=str(goal_id),
        ato_type="GOAL",
        subgraph={"goal_kind": str(goal_spec.goal_kind), "goal_sig": str(goal_sig), "validator_id": str(goal_spec.validator_id)},
        slots={"output_key": str(goal_spec.output_key)},
        bindings=_safe_deepcopy(goal_spec.bindings),
        cost=0.0,
        evidence_refs=[{"kind": "goal_spec_v72", "goal_sig": str(goal_sig)}],
        invariants={"expected": goal_spec.expected, "validator_id": str(goal_spec.validator_id)},
        created_step=int(step_ctr),
        last_step=int(step_ctr),
    )
    mg.add_node(step=int(step_ctr), ato=goal_node, reason="goal_spec")
    step_ctr += 1

    plan_node = ATOv71(
        ato_id=str(plan_id),
        ato_type="PLAN",
        subgraph={"plan_sig": str(plan.plan_sig), "planner": "planner_v79"},
        slots={"steps_total": int(len(plan.steps))},
        bindings={"output_key": str(goal_spec.output_key)},
        cost=float(len(plan.steps)),
        evidence_refs=[{"kind": "planner_debug", "state": _safe_deepcopy(planner_debug)}],
        invariants={"plan": plan.to_dict()},
        created_step=int(step_ctr),
        last_step=int(step_ctr),
    )
    mg.add_node(step=int(step_ctr), ato=plan_node, reason="plan")
    mg.add_edge(
        step=int(step_ctr),
        src_ato_id=str(goal_id),
        dst_ato_id=str(plan_id),
        edge_type="DEPENDS_ON",
        evidence_refs=[{"kind": "goal_depends_on_plan", "goal_sig": str(goal_sig), "plan_sig": str(plan.plan_sig)}],
        reason="goal_plan",
    )
    step_ctr += 1

    concept_acts = store.concept_acts()
    for act in sorted(concept_acts, key=lambda a: str(a.id)):
        mg.add_node(step=int(step_ctr), ato=_concept_ato_from_act(act=act, step=int(step_ctr)), reason="concept_catalog")
        step_ctr += 1

    vars_state: Dict[str, Any] = _safe_deepcopy(goal_spec.bindings)
    engine = EngineV80(store, seed=int(seed))

    all_concept_calls: List[Dict[str, Any]] = []
    state_ids: List[str] = []

    for s in plan.steps:
        op_id = f"op_v80_{str(s.step_id)}"
        op_node = ATOv71(
            ato_id=str(op_id),
            ato_type="OPERATOR",
            subgraph={"kind": "CALL_CONCEPT", "concept_id": str(s.concept_id), "produces": str(s.produces), "idx": int(s.idx)},
            slots={"bind_map": _safe_deepcopy(s.bind_map)},
            bindings={},
            cost=1.0,
            evidence_refs=[{"kind": "plan_step", "step": s.to_dict()}],
            invariants={"requires": list(sorted(s.bind_map.keys()))},
            created_step=int(step_ctr),
            last_step=int(step_ctr),
        )
        mg.add_node(step=int(step_ctr), ato=op_node, reason="operator_step")
        mg.add_edge(
            step=int(step_ctr),
            src_ato_id=str(plan_id),
            dst_ato_id=str(op_id),
            edge_type="DEPENDS_ON",
            evidence_refs=[{"kind": "plan_depends_on_operator", "idx": int(s.idx), "step_id": str(s.step_id)}],
            reason="plan_ops",
        )
        mg.add_edge(
            step=int(step_ctr),
            src_ato_id=str(op_id),
            dst_ato_id=str(s.concept_id),
            edge_type="CALLS",
            evidence_refs=[{"kind": "operator_calls_concept", "concept_id": str(s.concept_id)}],
            reason="op_calls_concept",
        )
        step_ctr += 1

        concept_inputs: Dict[str, Any] = {}
        for k, vname in sorted(s.bind_map.items(), key=lambda kv: str(kv[0])):
            concept_inputs[str(k)] = vars_state.get(str(vname))

        exec_res = engine.execute_concept_csv(
            concept_act_id=str(s.concept_id),
            inputs=dict(concept_inputs),
            goal_kind=str(goal_spec.goal_kind),
            expected=None,
            step=int(step_ctr),
            max_depth=8,
            max_events=int(max_events),
            validate_output=False,
        )
        meta = exec_res.get("meta") if isinstance(exec_res, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        ok_exec = bool(meta.get("ok", False))
        out_text = str(meta.get("output_text") or exec_res.get("output") or "")
        out_sig = str(meta.get("output_sig") or "")

        vars_state[str(s.produces)] = out_text

        state_snapshot = _safe_deepcopy(vars_state)
        state_sig = _stable_hash_obj({"vars": state_snapshot, "idx": int(s.idx)})
        state_id = f"state_v80_{state_sig}"
        state_ids.append(state_id)
        st_node = ATOv71(
            ato_id=str(state_id),
            ato_type="STATE",
            subgraph={"idx": int(s.idx), "after_op": str(op_id)},
            slots={"keys": sorted(str(k) for k in state_snapshot.keys())},
            bindings=state_snapshot,
            cost=0.0,
            evidence_refs=[{"kind": "operator_result", "ok": bool(ok_exec), "output_sig": str(out_sig)}],
            invariants={},
            created_step=int(step_ctr),
            last_step=int(step_ctr),
        )
        mg.add_node(step=int(step_ctr), ato=st_node, reason="state_after_step")
        mg.add_edge(
            step=int(step_ctr),
            src_ato_id=str(op_id),
            dst_ato_id=str(state_id),
            edge_type="CAUSES",
            evidence_refs=[
                {
                    "kind": "op_causes_state",
                    "idx": int(s.idx),
                    "produces": str(s.produces),
                    "output_text": str(out_text),
                    "output_sig": str(out_sig),
                    "bind_map": _safe_deepcopy(s.bind_map),
                }
            ],
            reason="op_state",
        )

        tr = exec_res.get("trace") if isinstance(exec_res.get("trace"), dict) else {}
        concept_calls = tr.get("concept_calls") if isinstance(tr.get("concept_calls"), list) else []
        all_concept_calls.extend([_safe_deepcopy(c) for c in concept_calls if isinstance(c, dict)])
        _ingest_concept_calls_as_edges(
            mg=mg,
            store=store,
            step_base=int(step_ctr) + 1,
            concept_calls=concept_calls,
        )
        step_ctr += 2

    final_val = vars_state.get(str(goal_spec.output_key))
    vres = run_validator(str(goal_spec.validator_id), str(final_val or ""), goal_spec.expected)
    eval_ok = bool(vres.passed)
    eval_id = f"eval_v80_{goal_sig}"
    eval_node = ATOv71(
        ato_id=str(eval_id),
        ato_type="EVAL",
        subgraph={"goal_sig": str(goal_sig), "plan_sig": str(plan.plan_sig)},
        slots={"output_key": str(goal_spec.output_key)},
        bindings={"ok": bool(eval_ok), "reason": str(vres.reason), "got": str(final_val or ""), "expected": goal_spec.expected},
        cost=0.0,
        evidence_refs=[{"kind": "validator", "validator_id": str(goal_spec.validator_id), "reason": str(vres.reason)}],
        invariants={},
        created_step=int(step_ctr),
        last_step=int(step_ctr),
    )
    mg.add_node(step=int(step_ctr), ato=eval_node, reason="eval_final")
    if state_ids:
        mg.add_edge(
            step=int(step_ctr),
            src_ato_id=str(eval_id),
            dst_ato_id=str(state_ids[-1]),
            edge_type="DERIVED_FROM",
            evidence_refs=[{"kind": "eval_from_state", "state_id": str(state_ids[-1])}],
            reason="eval_state",
        )
    step_ctr += 1

    snap_before_irrelevant = mg.snapshot_graph_state()
    graph_sig_before_irrelevant = mg.graph_sig()
    roots = [str(goal_id), str(plan_id), str(eval_id)]
    r_v1 = render_projection(
        graph_snapshot=snap_before_irrelevant,
        root_ids=roots,
        max_depth=16,
        bindings=_safe_deepcopy(goal_spec.bindings),
        goals=[{"goal_id": str(goal_id), "goal_sig": str(goal_sig)}],
        plan_state={"plan_id": str(plan_id), "plan_sig": str(plan.plan_sig)},
        style="v1",
    )
    r_v2 = render_projection(
        graph_snapshot=snap_before_irrelevant,
        root_ids=roots,
        max_depth=16,
        bindings=_safe_deepcopy(goal_spec.bindings),
        goals=[{"goal_id": str(goal_id), "goal_sig": str(goal_sig)}],
        plan_state={"plan_id": str(plan_id), "plan_sig": str(plan.plan_sig)},
        style="v2",
    )

    irrelevant = ATOv71(
        ato_id="obs_v80_irrelevant_00",
        ato_type="OBS",
        subgraph={"kind": "irrelevant"},
        slots={},
        bindings={"note": "ignored"},
        cost=0.0,
        evidence_refs=[{"kind": "irrelevant"}],
        invariants={},
        created_step=int(step_ctr),
        last_step=int(step_ctr),
    )
    mg.add_node(step=int(step_ctr), ato=irrelevant, reason="irrelevant_unreachable")
    step_ctr += 1

    snap_after_irrelevant = mg.snapshot_graph_state()
    graph_sig_after_irrelevant = mg.graph_sig()
    r_v1_after = render_projection(
        graph_snapshot=snap_after_irrelevant,
        root_ids=roots,
        max_depth=16,
        bindings=_safe_deepcopy(goal_spec.bindings),
        goals=[{"goal_id": str(goal_id), "goal_sig": str(goal_sig)}],
        plan_state={"plan_id": str(plan_id), "plan_sig": str(plan.plan_sig)},
        style="v1",
    )

    return {
        "ok": bool(eval_ok),
        "goal_id": str(goal_id),
        "goal_sig": str(goal_sig),
        "plan": plan.to_dict(),
        "planner": dict(planner_debug),
        "final": {
            "output_key": str(goal_spec.output_key),
            "got": str(final_val or ""),
            "expected": goal_spec.expected,
            "validator": {"passed": bool(vres.passed), "reason": str(vres.reason)},
        },
        "graph": {
            "graph_sig_before_irrelevant": str(graph_sig_before_irrelevant),
            "graph_sig": str(graph_sig_after_irrelevant),
            "chains": mg.verify_chains(),
        },
        "render": {
            "render_sig_v1": str(r_v1.get("render_sig") or ""),
            "render_sig_v2": str(r_v2.get("render_sig") or ""),
            "render_sig_v1_after_irrelevant": str(r_v1_after.get("render_sig") or ""),
            "text_v1": str(r_v1.get("text") or ""),
            "text_v2": str(r_v2.get("text") or ""),
        },
        "trace": {"concept_calls": list(all_concept_calls)},
        "snapshot": snap_after_irrelevant,
    }

