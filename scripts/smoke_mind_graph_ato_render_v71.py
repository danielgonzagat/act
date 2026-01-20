#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.ato_v71 import ATOv71
from atos_core.concept_registry_v70 import ConceptRegistryV70
from atos_core.engine import Engine, EngineConfig
from atos_core.mind_graph_v71 import MindGraphV71, ato_from_concept_registry_v70
from atos_core.render_v71 import render_projection
from atos_core.store import ActStore


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(str(s).encode("utf-8")).hexdigest()


def sha256_canon(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def write_json(path: str, obj: Any) -> str:
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def make_concept_act(
    *,
    act_id: str,
    input_schema: Dict[str, str],
    output_schema: Dict[str, str],
    validator_id: str,
    program: List[Instruction],
) -> Act:
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match={},
        program=list(program),
        evidence={
            "interface": {
                "input_schema": dict(input_schema),
                "output_schema": dict(output_schema),
                "validator_id": str(validator_id),
            }
        },
        cost={},
        deps=[],
        active=True,
    )


def _interface_sig_from_act(act: Act) -> str:
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    body = {
        "in": iface.get("input_schema", {}),
        "out": iface.get("output_schema", {}),
        "validator_id": str(iface.get("validator_id") or ""),
    }
    return sha256_hex(canonical_json_dumps(body).encode("utf-8"))


def _program_sha256_from_act(act: Act) -> str:
    prog = [ins.to_dict() for ins in (act.program or [])]
    return sha256_hex(canonical_json_dumps(prog).encode("utf-8"))


def ato_from_concept_act(*, act: Act, step: int, concept_state: str) -> ATOv71:
    iface_sig = _interface_sig_from_act(act)
    prog_sha = _program_sha256_from_act(act)
    ev = act.evidence if isinstance(act.evidence, dict) else {}
    iface = ev.get("interface") if isinstance(ev.get("interface"), dict) else {}
    iface = iface if isinstance(iface, dict) else {}
    inp = iface.get("input_schema") if isinstance(iface.get("input_schema"), dict) else {}
    out = iface.get("output_schema") if isinstance(iface.get("output_schema"), dict) else {}
    slots = {"inputs": sorted(str(k) for k in inp.keys()), "outputs": sorted(str(k) for k in out.keys())}
    invariants = {
        "input_schema": {str(k): str(v) for k, v in sorted(inp.items(), key=lambda kv: str(kv[0]))},
        "output_schema": {str(k): str(v) for k, v in sorted(out.items(), key=lambda kv: str(kv[0]))},
        "validator_id": str(iface.get("validator_id") or ""),
    }
    subgraph = {
        "interface_sig": str(iface_sig),
        "program_sha256": str(prog_sha),
        "program_len": int(len(act.program or [])),
        "concept_state": str(concept_state),
    }
    return ATOv71(
        ato_id=str(act.id),
        ato_type="CONCEPT",
        subgraph=dict(subgraph),
        slots=dict(slots),
        bindings={},
        cost=float(len(act.program or [])),
        evidence_refs=[{"kind": "concept_act", "act_id": str(act.id)}],
        invariants=dict(invariants),
        created_step=int(step),
        last_step=int(step),
    )


def _assert_ato_sig_deterministic() -> Dict[str, Any]:
    a = ATOv71(
        ato_id="test_ato",
        ato_type="STATE",
        subgraph={"k": "v"},
        slots={"s": ["x"]},
        bindings={"x": 1},
        cost=1.0,
        evidence_refs=[{"kind": "unit"}],
        invariants={"v": 1},
        created_step=0,
        last_step=0,
    )
    d1 = a.to_dict(include_sig=True)
    d2 = a.to_dict(include_sig=True)
    if str(d1.get("ato_sig") or "") != str(d2.get("ato_sig") or ""):
        _fail("ERROR: ato_sig_not_deterministic")
    try:
        _ = ATOv71(ato_id="bad", ato_type="NOT_A_TYPE")  # type: ignore[arg-type]
        _fail("ERROR: expected closed ato_type rejection")
    except Exception:
        pass
    return {"ok": True, "ato_sig": str(d1.get("ato_sig") or "")}


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    # (1) ATO types closed + deterministic signature.
    ato_sig_test = _assert_ato_sig_deterministic()

    # Build store (V70 smoke source): normalize + add_xy + clone.
    store = ActStore()
    out_schema = {"out": "str"}

    normalize_id = "concept_v70_normalize_x_v0"
    normalize = make_concept_act(
        act_id=normalize_id,
        input_schema={"x": "str"},
        output_schema=out_schema,
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["x"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "out"}),
            Instruction("CSV_RETURN", {"var": "out"}),
        ],
    )
    store.add(normalize)

    good_id = "concept_v70_add_xy_v0"
    good = make_concept_act(
        act_id=good_id,
        input_schema={"x": "str", "y": "str"},
        output_schema=out_schema,
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
            Instruction("CSV_GET_INPUT", {"name": "y", "out": "y"}),
            Instruction("CSV_CALL", {"concept_id": normalize_id, "bind": {"x": "x"}, "out": "nx"}),
            Instruction("CSV_CALL", {"concept_id": normalize_id, "bind": {"x": "y"}, "out": "ny"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["nx"], "out": "ix"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["ny"], "out": "iy"}),
            Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["ix", "iy"], "out": "sum"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["sum"], "out": "out"}),
            Instruction("CSV_RETURN", {"var": "out"}),
        ],
    )
    store.add(good)

    clone_id = "concept_v70_add_xy_clone_v0"
    clone = make_concept_act(
        act_id=clone_id,
        input_schema={"x": "str", "y": "str"},
        output_schema=out_schema,
        validator_id="text_exact",
        program=list(good.program),
    )
    store.add(clone)

    # V70 registry (objects) -> V71 nodes.
    reg_dir = os.path.join(out_dir, "concept_registry_v70")
    reg = ConceptRegistryV70(reg_dir, toc_fail_threshold=2, similarity_threshold=0.95)
    vectors_A = [
        {"inputs": {"x": "007", "y": "0"}, "expected": "7", "expected_output_text": "7"},
        {"inputs": {"x": "0004", "y": "0"}, "expected": "4", "expected_output_text": "4"},
        {"inputs": {"x": "42", "y": "0"}, "expected": "42", "expected_output_text": "42"},
    ]
    vectors_B = [
        {"inputs": {"x": "4", "y": "8"}, "expected": "12", "expected_output_text": "12"},
        {"inputs": {"x": "09", "y": "1"}, "expected": "10", "expected_output_text": "10"},
        {"inputs": {"x": "100", "y": "23"}, "expected": "123", "expected_output_text": "123"},
    ]

    r_good = reg.attempt_promote_with_toc(
        step=1,
        candidate=good,
        store=store,
        vectors_A=vectors_A,
        vectors_B=vectors_B,
        domain_A="A_normalize",
        domain_B="B_add",
        existing_for_duplicates=[],
    )
    r_clone = reg.attempt_promote_with_toc(
        step=2,
        candidate=clone,
        store=store,
        vectors_A=vectors_A,
        vectors_B=vectors_B,
        domain_A="A_normalize",
        domain_B="B_add",
        existing_for_duplicates=[good],
    )

    concept_good = r_good.get("concept") if isinstance(r_good.get("concept"), dict) else {}
    concept_clone = r_clone.get("concept") if isinstance(r_clone.get("concept"), dict) else {}
    if str(concept_good.get("concept_state") or "") != "ACTIVE":
        _fail("ERROR: expected good concept ACTIVE in registry")
    if str(concept_clone.get("concept_state") or "") == "ACTIVE":
        _fail("ERROR: expected clone NOT ACTIVE in registry")

    # Mind graph store (WORM).
    mg_dir = os.path.join(out_dir, "mind_graph")
    mg = MindGraphV71(mg_dir)

    # Add concept nodes from registry (V70 -> V71 adapter).
    mg.add_node(step=10, ato=ato_from_concept_registry_v70(concept_good), reason="from_v70_registry")
    mg.add_node(step=11, ato=ato_from_concept_registry_v70(concept_clone), reason="from_v70_registry")

    # Ensure callee concept node exists (normalize) for call edges.
    mg.add_node(step=12, ato=ato_from_concept_act(act=normalize, step=12, concept_state="ACTIVE"), reason="from_concept_act")

    # Create GOAL + PLAN (minimal).
    goal_bindings = {"x": "0004", "y": "0008"}
    goal_id = "goal_v71_add_xy_00"
    goal = ATOv71(
        ato_id=goal_id,
        ato_type="GOAL",
        subgraph={"kind": "add_xy_goal_v0"},
        slots={"inputs": ["x", "y"], "outputs": ["out"]},
        bindings=dict(goal_bindings),
        cost=0.0,
        evidence_refs=[{"kind": "smoke", "name": "v71"}],
        invariants={"output_validator_id": "text_exact"},
        created_step=13,
        last_step=13,
    )
    plan_id = "plan_v71_use_add_xy_00"
    plan = ATOv71(
        ato_id=plan_id,
        ato_type="PLAN",
        subgraph={"kind": "call_concept_v0", "concept_id": good_id},
        slots={"steps": ["call:add_xy"]},
        bindings={"concept_id": good_id},
        cost=0.0,
        evidence_refs=[{"kind": "smoke", "name": "v71"}],
        invariants={"requires": ["concept_active"]},
        created_step=14,
        last_step=14,
    )
    mg.add_node(step=13, ato=goal, reason="smoke_goal")
    mg.add_node(step=14, ato=plan, reason="smoke_plan")

    mg.add_edge(
        step=15,
        src_ato_id=goal_id,
        dst_ato_id=plan_id,
        edge_type="DEPENDS_ON",
        evidence_refs=[{"kind": "goal_depends_on_plan"}],
        reason="goal_plan",
    )
    mg.add_edge(
        step=16,
        src_ato_id=plan_id,
        dst_ato_id=good_id,
        edge_type="DEPENDS_ON",
        evidence_refs=[{"kind": "plan_depends_on_concept", "concept_id": good_id}],
        reason="plan_concept",
    )

    # (3) Execute concept and ingest callstack edges.
    engine = Engine(store, seed=int(seed), config=EngineConfig(enable_contracts=False))
    inputs = {"x": "0004", "y": "0008"}
    out = engine.execute_concept_csv(concept_act_id=good_id, inputs=dict(inputs), expected="12", step=20)
    tr = out.get("trace") if isinstance(out, dict) else None
    tr = tr if isinstance(tr, dict) else {}
    calls = tr.get("concept_calls")
    if not isinstance(calls, list) or not calls:
        _fail("ERROR: missing trace.concept_calls")

    # Add any missing concept nodes discovered via runtime calls.
    for ce in calls:
        if not isinstance(ce, dict):
            continue
        cid = str(ce.get("concept_id") or "")
        if not cid:
            continue
        if cid in {str(n.get("ato_id") or "") for n in mg.snapshot_graph_state().get("nodes", []) if isinstance(n, dict)}:
            continue
        act = store.get_concept_act(cid)
        if act is None:
            continue
        mg.add_node(step=21, ato=ato_from_concept_act(act=act, step=21, concept_state="ACTIVE"), reason="from_runtime_call")

    stack: Dict[int, str] = {}
    # Keep a handle to an evidence payload for aliasing test.
    first_call_with_bindings: Optional[Dict[str, Any]] = None
    first_edge_sig = ""
    first_edge_bindings_x = ""

    for idx, ce in enumerate(calls):
        if not isinstance(ce, dict):
            continue
        depth = int(ce.get("call_depth", 0) or 0)
        cid = str(ce.get("concept_id") or "")
        if not cid:
            continue
        if depth > 0 and (depth - 1) in stack:
            parent = str(stack.get(depth - 1) or "")
            if parent and parent != cid:
                ev_ref = {
                    "kind": "concept_call_event",
                    "idx": int(idx),
                    "call_depth": int(depth),
                    "bindings": ce.get("bindings") if isinstance(ce.get("bindings"), dict) else {},
                    "bindings_sig": str(ce.get("bindings_sig") or ""),
                    "return": ce.get("return") if isinstance(ce.get("return"), dict) else {},
                    "return_sig": str(ce.get("return_sig") or ""),
                }
                edge = mg.add_edge(
                    step=30 + int(idx),
                    src_ato_id=parent,
                    dst_ato_id=cid,
                    edge_type="CALLS",
                    evidence_refs=[ev_ref],
                    reason="from_trace_concept_calls",
                )
                if first_call_with_bindings is None:
                    first_call_with_bindings = ce
                    first_edge_sig = str(edge.get("edge_sig") or "")
                    first_edge_bindings_x = str((ev_ref.get("bindings") or {}).get("x") or "")
        stack[depth] = cid
        for k in list(stack.keys()):
            if k > depth:
                stack.pop(k, None)

    # (7) Anti-aliasing: mutate source bindings after ingestion and assert graph snapshot unchanged.
    if first_call_with_bindings is not None and isinstance(first_call_with_bindings.get("bindings"), dict):
        first_call_with_bindings["bindings"]["x"] = "MUTATED"

    # (4) Projection depends only on active subgraph.
    roots = [goal_id, plan_id]
    base_snapshot = mg.snapshot_graph_state()
    base_graph_sig = mg.graph_sig()
    r_v1 = render_projection(
        graph_snapshot=base_snapshot,
        root_ids=roots,
        max_depth=8,
        bindings=dict(goal_bindings),
        goals=[{"goal_id": goal_id}],
        plan_state={"plan_id": plan_id, "concept_id": good_id},
        style="v1",
    )
    r_v2 = render_projection(
        graph_snapshot=base_snapshot,
        root_ids=roots,
        max_depth=8,
        bindings=dict(goal_bindings),
        goals=[{"goal_id": goal_id}],
        plan_state={"plan_id": plan_id, "concept_id": good_id},
        style="v2",
    )
    render_sig_v1 = str(r_v1.get("render_sig") or "")
    render_sig_v2 = str(r_v2.get("render_sig") or "")
    if not render_sig_v1 or not render_sig_v2:
        _fail("ERROR: missing render_sig")
    if render_sig_v1 == render_sig_v2:
        _fail("ERROR: expected different render_sig for different styles")

    # Add an irrelevant node (unreachable from roots) and prove render v1 unchanged.
    irrelevant = ATOv71(
        ato_id="obs_v71_irrelevant_00",
        ato_type="OBS",
        subgraph={"kind": "irrelevant"},
        slots={},
        bindings={"note": "ignored"},
        cost=0.0,
        evidence_refs=[{"kind": "irrelevant"}],
        invariants={},
        created_step=40,
        last_step=40,
    )
    mg.add_node(step=40, ato=irrelevant, reason="irrelevant_node_unreachable")
    full_snapshot = mg.snapshot_graph_state()
    full_graph_sig = mg.graph_sig()
    r_v1_after = render_projection(
        graph_snapshot=full_snapshot,
        root_ids=roots,
        max_depth=8,
        bindings=dict(goal_bindings),
        goals=[{"goal_id": goal_id}],
        plan_state={"plan_id": plan_id, "concept_id": good_id},
        style="v1",
    )
    render_sig_v1_after = str(r_v1_after.get("render_sig") or "")
    if render_sig_v1_after != render_sig_v1:
        _fail("ERROR: render_sig_v1_changed_after_adding_unreachable_node")

    # WORM chain verification.
    chains = mg.verify_chains()
    if not (bool(chains.get("mind_nodes_chain_ok")) and bool(chains.get("mind_edges_chain_ok"))):
        _fail(f"ERROR: mind_graph_chain_verify_failed: {chains}")

    # Validate anti-aliasing on stored edge evidence refs (by edge_sig lookup).
    anti_aliasing_ok = True
    if first_edge_sig:
        stored_edges = full_snapshot.get("edges") if isinstance(full_snapshot.get("edges"), list) else []
        got_x = None
        for e in stored_edges:
            if not isinstance(e, dict):
                continue
            if str(e.get("edge_sig") or "") != first_edge_sig:
                continue
            refs = e.get("evidence_refs") if isinstance(e.get("evidence_refs"), list) else []
            for r in refs:
                if isinstance(r, dict) and isinstance(r.get("bindings"), dict):
                    got_x = str(r["bindings"].get("x") or "")
        if got_x is None or got_x != first_edge_bindings_x:
            anti_aliasing_ok = False

    # Validate goal node bindings snapshot anti-aliasing.
    goal_bindings["x"] = "MUTATED"
    stored_goal_x = ""
    for n in full_snapshot.get("nodes", []) if isinstance(full_snapshot.get("nodes"), list) else []:
        if isinstance(n, dict) and str(n.get("ato_id") or "") == goal_id:
            b = n.get("bindings") if isinstance(n.get("bindings"), dict) else {}
            stored_goal_x = str(b.get("x") or "")
    if stored_goal_x != "0004":
        anti_aliasing_ok = False

    return {
        "seed": int(seed),
        "ato_sig_test": dict(ato_sig_test),
        "graph": {
            "base_graph_sig": str(base_graph_sig),
            "full_graph_sig": str(full_graph_sig),
            "nodes_total": int(len(full_snapshot.get("nodes", [])) if isinstance(full_snapshot.get("nodes"), list) else 0),
            "edges_total": int(len(full_snapshot.get("edges", [])) if isinstance(full_snapshot.get("edges"), list) else 0),
        },
        "render": {
            "render_sig_v1": str(render_sig_v1),
            "render_sig_v2": str(render_sig_v2),
            "render_sig_v1_after_irrelevant": str(render_sig_v1_after),
        },
        "checks": {
            "render_invariance_unreachable_node": bool(render_sig_v1_after == render_sig_v1),
            "render_substitutable_styles": bool(render_sig_v1 != render_sig_v2),
            "anti_aliasing_ok": bool(anti_aliasing_ok),
            "chains": dict(chains),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_mind_graph_ato_render_v71")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    sigs: List[Tuple[str, str]] = []
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)
        os.makedirs(out_dir, exist_ok=False)

        ev = smoke_try(out_dir=out_dir, seed=seed)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        core = {
            "seed": seed,
            "graph_sig": str(ev.get("graph", {}).get("full_graph_sig") if isinstance(ev.get("graph"), dict) else ""),
            "render_sig_v1": str(ev.get("render", {}).get("render_sig_v1") if isinstance(ev.get("render"), dict) else ""),
            "render_sig_v2": str(ev.get("render", {}).get("render_sig_v2") if isinstance(ev.get("render"), dict) else ""),
            "sha256_eval_json": str(eval_sha),
        }
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": str(summary_sha)}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)

        sigs.append((str(core["graph_sig"]), str(core["render_sig_v1"])))
        summary_shas.append(str(summary_sha))

        results["tries"][f"try{t}"] = {
            "out_dir": out_dir,
            "eval_json": {"path": eval_path, "sha256": eval_sha},
            "smoke_summary_json": {"path": smoke_path, "sha256": smoke_sha},
            "summary_sha256": summary_sha,
        }

    determinism_ok = bool(len(sigs) == 2 and sigs[0] == sigs[1] and len(summary_shas) == 2 and summary_shas[0] == summary_shas[1])
    if not determinism_ok:
        _fail(f"ERROR: determinism mismatch: sigs={sigs} summary_shas={summary_shas}")
    results["determinism"] = {"ok": True, "summary_sha256": summary_shas[0], "graph_sig": sigs[0][0], "render_sig_v1": sigs[0][1]}
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

