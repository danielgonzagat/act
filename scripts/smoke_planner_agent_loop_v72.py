#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.agent_loop_v72 import run_goal_spec_v72
from atos_core.goal_spec_v72 import GoalSpecV72
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


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    store = ActStore()

    # Micro-world concepts forcing multi-step planning.
    normalize_x_id = "concept_v72_normalize_x_v0"
    normalize_y_id = "concept_v72_normalize_y_v0"
    add_nx_ny_id = "concept_v72_add_nx_ny_v0"

    normalize_x = make_concept_act(
        act_id=normalize_x_id,
        input_schema={"x": "str"},
        output_schema={"nx": "str"},
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["x"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "nx"}),
            Instruction("CSV_RETURN", {"var": "nx"}),
        ],
    )
    store.add(normalize_x)

    normalize_y = make_concept_act(
        act_id=normalize_y_id,
        input_schema={"y": "str"},
        output_schema={"ny": "str"},
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "y", "out": "y"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["y"], "out": "d"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["d"], "out": "i"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["i"], "out": "ny"}),
            Instruction("CSV_RETURN", {"var": "ny"}),
        ],
    )
    store.add(normalize_y)

    add_nx_ny = make_concept_act(
        act_id=add_nx_ny_id,
        input_schema={"nx": "str", "ny": "str"},
        output_schema={"sum": "str"},
        validator_id="text_exact",
        program=[
            Instruction("CSV_GET_INPUT", {"name": "nx", "out": "nx"}),
            Instruction("CSV_GET_INPUT", {"name": "ny", "out": "ny"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["nx"], "out": "dx"}),
            Instruction("CSV_PRIMITIVE", {"fn": "scan_digits", "in": ["ny"], "out": "dy"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dx"], "out": "ix"}),
            Instruction("CSV_PRIMITIVE", {"fn": "digits_to_int", "in": ["dy"], "out": "iy"}),
            Instruction("CSV_PRIMITIVE", {"fn": "add_int", "in": ["ix", "iy"], "out": "sum_i"}),
            Instruction("CSV_PRIMITIVE", {"fn": "int_to_digits", "in": ["sum_i"], "out": "sum"}),
            Instruction("CSV_RETURN", {"var": "sum"}),
        ],
    )
    store.add(add_nx_ny)

    goal_spec = GoalSpecV72(
        goal_kind="v72_sum_norm",
        bindings={"x": "0004", "y": "0008"},
        output_key="sum",
        expected="12",
        validator_id="text_exact",
        created_step=0,
    )

    res = run_goal_spec_v72(
        goal_spec=goal_spec,
        store=store,
        seed=int(seed),
        out_dir=str(out_dir),
        max_depth=6,
        max_expansions=256,
        max_events=256,
    )

    if not bool(res.get("ok", False)):
        _fail(f"ERROR: agent_loop_v72 not ok: {res.get('final')}")

    plan = res.get("plan") if isinstance(res.get("plan"), dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    plan_sig = str(plan.get("plan_sig") or "")
    if not plan_sig:
        _fail("ERROR: missing plan_sig")
    if int(len(steps)) < 2:
        _fail(f"ERROR: expected multi-step plan >=2, got={len(steps)}")

    final = res.get("final") if isinstance(res.get("final"), dict) else {}
    got = str(final.get("got") or "")
    expected = str(final.get("expected") or "")
    v = final.get("validator") if isinstance(final.get("validator"), dict) else {}
    if got != expected or not bool(v.get("passed", False)):
        _fail(f"ERROR: final_mismatch: got={got} expected={expected} validator={v}")

    chains = (res.get("graph") or {}).get("chains") if isinstance(res.get("graph"), dict) else {}
    if not (bool(chains.get("mind_nodes_chain_ok")) and bool(chains.get("mind_edges_chain_ok"))):
        _fail(f"ERROR: mind_graph_chain_failed: {chains}")

    render = res.get("render") if isinstance(res.get("render"), dict) else {}
    render_sig_v1 = str(render.get("render_sig_v1") or "")
    render_sig_v2 = str(render.get("render_sig_v2") or "")
    render_sig_v1_after = str(render.get("render_sig_v1_after_irrelevant") or "")
    if not render_sig_v1 or not render_sig_v2:
        _fail("ERROR: missing render_sig")
    if render_sig_v1 == render_sig_v2:
        _fail("ERROR: expected render_sig_v1 != render_sig_v2")
    if render_sig_v1_after != render_sig_v1:
        _fail("ERROR: render_sig_v1 changed after adding unreachable node")

    # Anti-aliasing: mutating goal_spec.bindings and returned trace must not change the stored snapshot.
    snap = res.get("snapshot") if isinstance(res.get("snapshot"), dict) else {}
    snap_sig_before = sha256_canon(snap)
    try:
        goal_spec.bindings["x"] = "MUTATED"
    except Exception:
        pass
    tr = res.get("trace") if isinstance(res.get("trace"), dict) else {}
    calls = tr.get("concept_calls") if isinstance(tr.get("concept_calls"), list) else []
    if calls and isinstance(calls[0], dict) and isinstance(calls[0].get("bindings"), dict):
        calls[0]["bindings"]["x"] = "MUTATED"
    snap_sig_after = sha256_canon(snap)
    if snap_sig_after != snap_sig_before:
        _fail("ERROR: anti_aliasing_failed: snapshot changed after external mutations")

    nodes_total = int(len(snap.get("nodes", [])) if isinstance(snap.get("nodes"), list) else 0)
    edges_total = int(len(snap.get("edges", [])) if isinstance(snap.get("edges"), list) else 0)

    return {
        "seed": int(seed),
        "plan": {"plan_sig": str(plan_sig), "steps_total": int(len(steps))},
        "final": {"got": str(got), "expected": str(expected), "validator": dict(v)},
        "graph": {
            "graph_sig": str((res.get("graph") or {}).get("graph_sig") if isinstance(res.get("graph"), dict) else ""),
            "chains": dict(chains) if isinstance(chains, dict) else {},
            "nodes_total": int(nodes_total),
            "edges_total": int(edges_total),
        },
        "render": {
            "render_sig_v1": str(render_sig_v1),
            "render_sig_v2": str(render_sig_v2),
            "render_sig_v1_after_irrelevant": str(render_sig_v1_after),
        },
        "checks": {
            "plan_multi_step": True,
            "output_ok": True,
            "chains_ok": True,
            "anti_aliasing_ok": True,
            "render_substitutable_styles": bool(render_sig_v1 != render_sig_v2),
            "render_invariance_unreachable": bool(render_sig_v1_after == render_sig_v1),
        },
        "snapshot_sig": {"before": str(snap_sig_before), "after": str(snap_sig_after)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_planner_agent_loop_v72")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    sigs: List[Tuple[str, str, str]] = []
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)
        os.makedirs(out_dir, exist_ok=False)

        ev = smoke_try(out_dir=out_dir, seed=seed)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        core = {
            "seed": int(seed),
            "plan_sig": str(ev.get("plan", {}).get("plan_sig") if isinstance(ev.get("plan"), dict) else ""),
            "steps_total": int(ev.get("plan", {}).get("steps_total") if isinstance(ev.get("plan"), dict) else 0),
            "graph_sig": str(ev.get("graph", {}).get("graph_sig") if isinstance(ev.get("graph"), dict) else ""),
            "render_sig_v1": str(ev.get("render", {}).get("render_sig_v1") if isinstance(ev.get("render"), dict) else ""),
            "render_sig_v2": str(ev.get("render", {}).get("render_sig_v2") if isinstance(ev.get("render"), dict) else ""),
            "final_got": str(ev.get("final", {}).get("got") if isinstance(ev.get("final"), dict) else ""),
            "sha256_eval_json": str(eval_sha),
        }
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": str(summary_sha)}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)

        sigs.append((str(core["plan_sig"]), str(core["graph_sig"]), str(core["render_sig_v1"])))
        summary_shas.append(str(summary_sha))

        results["tries"][f"try{t}"] = {
            "out_dir": out_dir,
            "eval_json": {"path": eval_path, "sha256": eval_sha},
            "smoke_summary_json": {"path": smoke_path, "sha256": smoke_sha},
            "summary_sha256": summary_sha,
        }

    determinism_ok = bool(
        len(sigs) == 2
        and sigs[0] == sigs[1]
        and len(summary_shas) == 2
        and summary_shas[0] == summary_shas[1]
    )
    if not determinism_ok:
        _fail(f"ERROR: determinism mismatch: sigs={sigs} summary_shas={summary_shas}")
    results["determinism"] = {
        "ok": True,
        "summary_sha256": summary_shas[0],
        "plan_sig": sigs[0][0],
        "graph_sig": sigs[0][1],
        "render_sig_v1": sigs[0][2],
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

