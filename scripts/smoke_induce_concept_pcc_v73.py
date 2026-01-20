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
from atos_core.induce_concept_v73 import induce_concept_v73
from atos_core.pcc_v73 import verify_pcc_v73
from atos_core.store import ActStore
from atos_core.trace_v73 import trace_from_agent_loop_v72


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


def _eval_from_run(res: Dict[str, Any]) -> Dict[str, Any]:
    plan = res.get("plan") if isinstance(res.get("plan"), dict) else {}
    plan = plan if isinstance(plan, dict) else {}
    steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
    final = res.get("final") if isinstance(res.get("final"), dict) else {}
    final = final if isinstance(final, dict) else {}
    graph = res.get("graph") if isinstance(res.get("graph"), dict) else {}
    graph = graph if isinstance(graph, dict) else {}
    chains = graph.get("chains") if isinstance(graph.get("chains"), dict) else {}
    return {
        "ok": bool(res.get("ok", False)),
        "plan_sig": str(plan.get("plan_sig") or ""),
        "steps_total": int(len(steps)),
        "got": str(final.get("got") or ""),
        "expected": final.get("expected"),
        "graph_sig": str(graph.get("graph_sig") or ""),
        "chains": dict(chains) if isinstance(chains, dict) else {},
    }


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    store = ActStore()

    # Base micro-world (same semantics as V72 smoke; concept path is mined).
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

    store_hash_base = store.content_hash()

    # Generate >=2 traces from agent_loop_v72 (different contexts).
    goal1 = GoalSpecV72(
        goal_kind="v73_sum_norm",
        bindings={"x": "0004", "y": "0008"},
        output_key="sum",
        expected="12",
        validator_id="text_exact",
        created_step=0,
    )
    goal2 = GoalSpecV72(
        goal_kind="v73_sum_norm",
        bindings={"x": "0010", "y": "0003"},
        output_key="sum",
        expected="13",
        validator_id="text_exact",
        created_step=0,
    )

    g1_before_dir = os.path.join(out_dir, "goal1_before")
    g2_before_dir = os.path.join(out_dir, "goal2_before")
    ensure_absent(g1_before_dir)
    ensure_absent(g2_before_dir)
    os.makedirs(g1_before_dir, exist_ok=False)
    os.makedirs(g2_before_dir, exist_ok=False)

    res1_before = run_goal_spec_v72(goal_spec=goal1, store=store, seed=int(seed), out_dir=g1_before_dir)
    res2_before = run_goal_spec_v72(goal_spec=goal2, store=store, seed=int(seed), out_dir=g2_before_dir)
    if not bool(res1_before.get("ok", False)):
        _fail("ERROR: goal1_before not ok")
    if not bool(res2_before.get("ok", False)):
        _fail("ERROR: goal2_before not ok")

    eval1_before = _eval_from_run(res1_before)
    eval2_before = _eval_from_run(res2_before)

    steps_before = int(eval1_before.get("steps_total", 0) or 0)
    if steps_before < 2:
        _fail(f"ERROR: expected >=2 steps before induction, got={steps_before}")

    traces = [
        trace_from_agent_loop_v72(goal_spec=goal1, result=res1_before),
        trace_from_agent_loop_v72(goal_spec=goal2, result=res2_before),
    ]
    traces_sorted = sorted([t.to_canonical_dict(include_sig=True) for t in traces], key=lambda d: str(d.get("trace_sig") or ""))
    traces_path = os.path.join(out_dir, "traces_v73.json")
    traces_sha = write_json(traces_path, {"schema_version": 1, "traces": list(traces_sorted)})

    # Induce a composed concept + PCC certificate, verify, then promote.
    candidate_act, certificate, induce_debug = induce_concept_v73(traces, store, max_k=6, min_support=2, seed=int(seed))
    cand_path = os.path.join(out_dir, "candidate_act.json")
    cand_sha = write_json(cand_path, candidate_act.to_dict())
    cert_path = os.path.join(out_dir, "certificate_v1.json")
    cert_sha = write_json(cert_path, certificate)

    ok_pcc, reason_pcc, details_pcc = verify_pcc_v73(
        candidate_act=candidate_act, certificate=certificate, store_base=store, seed=int(seed)
    )
    if not bool(ok_pcc):
        _fail(f"ERROR: PCC verify failed: {reason_pcc}: {details_pcc}")

    # Promote (learn = edit store/graph state).
    store.add(candidate_act)
    store_hash_after = store.content_hash()

    # Re-run agent loop on goal1 and prove plan compression (strictly fewer steps).
    g1_after_dir = os.path.join(out_dir, "goal1_after")
    ensure_absent(g1_after_dir)
    os.makedirs(g1_after_dir, exist_ok=False)
    res1_after = run_goal_spec_v72(goal_spec=goal1, store=store, seed=int(seed), out_dir=g1_after_dir)
    if not bool(res1_after.get("ok", False)):
        _fail("ERROR: goal1_after not ok")
    eval1_after = _eval_from_run(res1_after)

    steps_after = int(eval1_after.get("steps_total", 0) or 0)
    if int(steps_after) >= int(steps_before):
        _fail(f"ERROR: expected plan compression, got steps_before={steps_before} steps_after={steps_after}")
    if str(eval1_after.get("got") or "") != str(goal1.expected):
        _fail("ERROR: goal1_after output mismatch")

    # Persist eval snapshots.
    before_path = os.path.join(out_dir, "eval_before.json")
    before_sha = write_json(
        before_path,
        {
            "schema_version": 1,
            "goal1": dict(eval1_before),
            "goal2": dict(eval2_before),
        },
    )
    after_path = os.path.join(out_dir, "eval_after.json")
    after_sha = write_json(after_path, {"schema_version": 1, "goal1": dict(eval1_after)})

    cert_sig = str(certificate.get("certificate_sig") or "")
    cand_iface_sig = str(
        ((candidate_act.evidence or {}).get("pcc_v73") or {}).get("certificate_sig") if isinstance(candidate_act.evidence, dict) else ""
    )
    if cand_iface_sig and cand_iface_sig != cert_sig:
        _fail("ERROR: candidate_evidence_certificate_sig_mismatch")

    return {
        "seed": int(seed),
        "store": {"hash_base": str(store_hash_base), "hash_after": str(store_hash_after)},
        "traces": {
            "traces_total": 2,
            "trace_sigs": [str(t.get("trace_sig") or "") for t in traces_sorted if isinstance(t, dict)],
        },
        "induced": {
            "act_id": str(candidate_act.id),
            "interface_sig": str(certificate.get("candidate", {}).get("interface_sig") if isinstance(certificate.get("candidate"), dict) else ""),
            "program_sha256": str(certificate.get("candidate", {}).get("program_sha256") if isinstance(certificate.get("candidate"), dict) else ""),
            "certificate_sig": str(cert_sig),
        },
        "pcc_verify": {"ok": True, "reason": str(reason_pcc), "details": dict(details_pcc)},
        "before": {"goal1": dict(eval1_before), "goal2": dict(eval2_before)},
        "after": {"goal1": dict(eval1_after)},
        "delta": {
            "steps_before": int(steps_before),
            "steps_after": int(steps_after),
            "delta_steps": int(steps_before - steps_after),
        },
        "delta_mdl_steps": int(steps_before - steps_after),
        "artifacts": {
            # Keep eval.json deterministic across tries: store only hashes (no paths containing tryN).
            "traces_v73_json": {"sha256": str(traces_sha)},
            "candidate_act_json": {"sha256": str(cand_sha)},
            "certificate_v1_json": {"sha256": str(cert_sha)},
            "eval_before_json": {"sha256": str(before_sha)},
            "eval_after_json": {"sha256": str(after_sha)},
        },
        "debug": {"induce": dict(induce_debug)},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", default="results/run_smoke_induce_concept_pcc_v73")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    results: Dict[str, Any] = {"seed": seed, "tries": {}}
    sigs: List[Tuple[str, str, int, int]] = []
    summary_shas: List[str] = []

    for t in (1, 2):
        out_dir = f"{out_base}_try{t}"
        ensure_absent(out_dir)
        os.makedirs(out_dir, exist_ok=False)

        ev = smoke_try(out_dir=out_dir, seed=seed)
        eval_path = os.path.join(out_dir, "eval.json")
        eval_sha = write_json(eval_path, ev)

        induced = ev.get("induced") if isinstance(ev.get("induced"), dict) else {}
        delta = ev.get("delta") if isinstance(ev.get("delta"), dict) else {}
        before = ev.get("before") if isinstance(ev.get("before"), dict) else {}
        after = ev.get("after") if isinstance(ev.get("after"), dict) else {}
        before_g1 = before.get("goal1") if isinstance(before.get("goal1"), dict) else {}
        after_g1 = after.get("goal1") if isinstance(after.get("goal1"), dict) else {}

        core = {
            "seed": int(seed),
            "induced_act_id": str(induced.get("act_id") or ""),
            "certificate_sig": str(induced.get("certificate_sig") or ""),
            "plan_sig_before": str(before_g1.get("plan_sig") or ""),
            "plan_sig_after": str(after_g1.get("plan_sig") or ""),
            "steps_before": int(delta.get("steps_before", 0) or 0),
            "steps_after": int(delta.get("steps_after", 0) or 0),
            "delta_steps": int(delta.get("delta_steps", 0) or 0),
            "sha256_eval_json": str(eval_sha),
        }
        summary_sha = sha256_text(canonical_json_dumps(core))
        smoke = {"summary": core, "determinism": {"summary_sha256": str(summary_sha)}}
        smoke_path = os.path.join(out_dir, "smoke_summary.json")
        smoke_sha = write_json(smoke_path, smoke)

        sigs.append(
            (
                str(core["certificate_sig"]),
                str(core["plan_sig_before"]),
                int(core["steps_before"]),
                int(core["steps_after"]),
            )
        )
        summary_shas.append(str(summary_sha))

        results["tries"][f"try{t}"] = {
            "out_dir": out_dir,
            "eval_json": {"path": eval_path, "sha256": eval_sha},
            "smoke_summary_json": {"path": smoke_path, "sha256": smoke_sha},
            "summary_sha256": summary_sha,
        }

    determinism_ok = bool(
        len(sigs) == 2 and sigs[0] == sigs[1] and len(summary_shas) == 2 and summary_shas[0] == summary_shas[1]
    )
    if not determinism_ok:
        _fail(f"ERROR: determinism mismatch: sigs={sigs} summary_shas={summary_shas}")
    results["determinism"] = {
        "ok": True,
        "summary_sha256": summary_shas[0],
        "certificate_sig": sigs[0][0],
        "plan_sig_before": sigs[0][1],
        "steps_before": sigs[0][2],
        "steps_after": sigs[0][3],
    }
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
