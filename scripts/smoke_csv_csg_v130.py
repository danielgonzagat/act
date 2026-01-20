#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.csg_v130 import (
    CsvCsgLogsV130,
    build_csg_concept_def_v130,
    canonicalize_csg_v130,
    csg_expand_v130,
    csg_hash_v130,
    csg_to_concept_program_v130,
)
from atos_core.engine_v80 import EngineV80
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
    match: Optional[Dict[str, Any]] = None,
    evidence_extra: Optional[Dict[str, Any]] = None,
) -> Act:
    ev: Dict[str, Any] = {
        "interface": {
            "input_schema": dict(input_schema),
            "output_schema": dict(output_schema),
            "validator_id": str(validator_id),
        }
    }
    if isinstance(evidence_extra, dict) and evidence_extra:
        ev.update(dict(evidence_extra))
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match=dict(match) if isinstance(match, dict) else {},
        program=list(program),
        evidence=ev,
        cost={},
        deps=[],
        active=True,
    )


def _execute_inline_replay(
    *,
    engine: EngineV80,
    steps: List[Dict[str, Any]],
    inputs: Dict[str, Any],
    goal_kind: str,
) -> Dict[str, Any]:
    env: Dict[str, Any] = dict(inputs)
    replay_calls: List[Dict[str, Any]] = []
    for st in steps:
        act_id = str(st.get("act_id") or "")
        bind = st.get("bind") if isinstance(st.get("bind"), dict) else {}
        produces = str(st.get("produces") or "")
        sub_inputs: Dict[str, Any] = {}
        for slot, var in bind.items():
            sub_inputs[str(slot)] = env.get(str(var))
        res = engine.execute_concept_csv(
            concept_act_id=str(act_id),
            inputs=sub_inputs,
            goal_kind=str(goal_kind),
            expected=None,
            step=0,
            max_depth=6,
            max_events=256,
            validate_output=False,
        )
        replay_calls.append(
            {
                "act_id": str(act_id),
                "inputs_sig": sha256_canon(sub_inputs),
                "meta_ok": bool((res.get("meta") or {}).get("ok", False)),
                "meta_reason": str((res.get("meta") or {}).get("reason", "")),
                "output_sig": str((res.get("meta") or {}).get("output_sig", "")),
            }
        )
        meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
        if not bool(meta.get("ok", False)):
            return {"ok": False, "reason": "inline_step_failed", "failing_act_id": str(act_id), "meta": dict(meta)}
        env[produces] = res.get("output")
    last_out = steps[-1].get("produces") if steps else ""
    return {
        "ok": True,
        "output": env.get(str(last_out)),
        "replay_calls": list(replay_calls),
    }


def _assert_match_disallowed(*, exec_res: Dict[str, Any], want_concept_id: str, want_goal_kind: str) -> None:
    meta = exec_res.get("meta") if isinstance(exec_res.get("meta"), dict) else {}
    if bool(meta.get("ok", True)):
        _fail(f"ERROR: expected ok=false, got meta.ok=true: {meta}")
    if str(meta.get("reason") or "") != "match_disallowed":
        _fail(f"ERROR: expected reason=match_disallowed, got={meta.get('reason')}")
    if str(meta.get("concept_id") or "") != str(want_concept_id):
        _fail(f"ERROR: expected concept_id={want_concept_id}, got={meta.get('concept_id')}")
    if str(meta.get("goal_kind") or "") != str(want_goal_kind):
        _fail(f"ERROR: expected goal_kind={want_goal_kind}, got={meta.get('goal_kind')}")
    tr = exec_res.get("trace") if isinstance(exec_res.get("trace"), dict) else {}
    calls = tr.get("concept_calls") if isinstance(tr.get("concept_calls"), list) else []
    if not calls:
        _fail("ERROR: expected trace.concept_calls non-empty for blocked call")
    call0 = calls[0] if isinstance(calls[0], dict) else {}
    if not bool(call0.get("blocked", False)):
        _fail(f"ERROR: expected blocked=true in concept_calls[0], got={call0}")
    if str(call0.get("blocked_reason") or "") != "match_disallowed":
        _fail(f"ERROR: expected blocked_reason=match_disallowed, got={call0.get('blocked_reason')}")


def smoke_try(*, out_dir: str, seed: int) -> Dict[str, Any]:
    logs = CsvCsgLogsV130.init(str(out_dir))
    store = ActStore()

    # Base micro-world concepts.
    normalize_x_id = "concept_v130_normalize_x_v0"
    normalize_y_id = "concept_v130_normalize_y_v0"
    add_nx_ny_id = "concept_v130_add_nx_ny_v0"

    store.add(
        make_concept_act(
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
    )
    store.add(
        make_concept_act(
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
    )
    store.add(
        make_concept_act(
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
    )

    allowed_goal_kind = "v130_sum"
    disallowed_goal_kind = "v130_forbidden"

    # Explicit CSG (Concept SubGraph) definition for a composed concept.
    csg = {
        "schema_version": 1,
        "nodes": [
            {"act_id": normalize_x_id, "bind": {"x": "x"}, "produces": "nx", "role": "entry"},
            {"act_id": normalize_y_id, "bind": {"y": "y"}, "produces": "ny", "role": "entry"},
            {"act_id": add_nx_ny_id, "bind": {"nx": "nx", "ny": "ny"}, "produces": "sum", "role": "exit"},
        ],
        "interface": {
            "inputs": ["x", "y"],
            "outputs": ["sum"],
            "validator_ids": ["text_exact"],
            "match": {"goal_kinds": [allowed_goal_kind]},
        },
    }
    csg_canon = canonicalize_csg_v130(csg)
    csg_hash = csg_hash_v130(csg_canon)
    concept_def = build_csg_concept_def_v130(csg=csg_canon, store=store)
    concept_id = str(concept_def.get("concept_id") or "")
    if not concept_id:
        _fail("ERROR: missing concept_id from build_csg_concept_def_v130")

    # Materialize executable concept_csv ACT with embedded CSG.
    prog = csg_to_concept_program_v130(csg_canon)
    concept_act = make_concept_act(
        act_id=concept_id,
        input_schema={"x": "str", "y": "str"},
        output_schema={"sum": "str"},
        validator_id="text_exact",
        program=prog,
        match={"goal_kinds": [allowed_goal_kind]},
        evidence_extra={"csg_v130": {"schema_version": 1, "csg_hash": csg_hash, "csg": csg_canon}},
    )
    store.add(concept_act)

    step = 0
    logs.append_concept_def(step=step, concept=concept_def)
    step += 1

    engine = EngineV80(store, seed=int(seed))

    inputs = {"x": "0004", "y": "0008"}

    # Allowed execution (must match inline replay).
    allowed_res = engine.execute_concept_csv(
        concept_act_id=concept_id,
        inputs=dict(inputs),
        goal_kind=allowed_goal_kind,
        expected="12",
        step=0,
        max_depth=6,
        max_events=256,
        validate_output=True,
    )
    allowed_meta = allowed_res.get("meta") if isinstance(allowed_res.get("meta"), dict) else {}
    if not bool(allowed_meta.get("ok", False)):
        _fail(f"ERROR: allowed execution failed: {allowed_meta}")
    if str(allowed_res.get("output") or "") != "12":
        _fail(f"ERROR: allowed output mismatch: got={allowed_res.get('output')}")
    trace = allowed_res.get("trace") if isinstance(allowed_res.get("trace"), dict) else {}
    trace_sig = sha256_canon(trace)

    logs.append_telemetry(
        step=step,
        telemetry={
            "schema_version": 1,
            "event": "CALL",
            "concept_id": concept_id,
            "csg_hash": csg_hash,
            "goal_kind": allowed_goal_kind,
            "ok": True,
            "reason": str(allowed_meta.get("reason") or ""),
            "output_sig": str(allowed_meta.get("output_sig") or ""),
            "trace_sig": str(trace_sig),
            "call_site": "smoke_v130:allowed",
        },
    )
    step += 1

    # Inline replay from CSG expansion (audit).
    steps = csg_expand_v130(csg_canon, store)
    replay_res = _execute_inline_replay(engine=engine, steps=steps, inputs=dict(inputs), goal_kind=allowed_goal_kind)
    if not bool(replay_res.get("ok", False)):
        _fail(f"ERROR: inline replay failed: {replay_res}")
    if str(replay_res.get("output") or "") != str(allowed_res.get("output") or ""):
        _fail(f"ERROR: replay output mismatch: replay={replay_res.get('output')} concept={allowed_res.get('output')}")

    logs.append_evidence(
        step=step,
        evidence={
            "schema_version": 1,
            "concept_id": concept_id,
            "csg_hash": csg_hash,
            "source": {
                "kind": "smoke_v130",
                "trace_sig": str(trace_sig),
                "concept_calls_total": int(
                    len((allowed_res.get("trace") or {}).get("concept_calls") or [])
                    if isinstance(allowed_res.get("trace"), dict)
                    else 0
                ),
                "window": {"start": 0, "end": int(len(steps) - 1)},
            },
            "verify": {"ok": True, "reason": "replay_equivalent"},
        },
    )
    step += 1

    # Disallowed execution (match closed, explicit).
    disallowed_res = engine.execute_concept_csv(
        concept_act_id=concept_id,
        inputs=dict(inputs),
        goal_kind=disallowed_goal_kind,
        expected="12",
        step=0,
        max_depth=6,
        max_events=256,
        validate_output=True,
    )
    _assert_match_disallowed(
        exec_res=disallowed_res,
        want_concept_id=str(concept_id),
        want_goal_kind=str(disallowed_goal_kind),
    )
    dis_meta = disallowed_res.get("meta") if isinstance(disallowed_res.get("meta"), dict) else {}
    dis_trace = disallowed_res.get("trace") if isinstance(disallowed_res.get("trace"), dict) else {}
    dis_trace_sig = sha256_canon(dis_trace)

    logs.append_telemetry(
        step=step,
        telemetry={
            "schema_version": 1,
            "event": "CALL",
            "concept_id": concept_id,
            "csg_hash": csg_hash,
            "goal_kind": disallowed_goal_kind,
            "ok": False,
            "reason": str(dis_meta.get("reason") or ""),
            "trace_sig": str(dis_trace_sig),
            "call_site": "smoke_v130:disallowed",
            "blocked": True,
            "blocked_reason": "match_disallowed",
        },
    )
    step += 1

    chains = logs.verify_chains()
    if not (bool(chains.get("concepts_chain_ok")) and bool(chains.get("evidence_chain_ok")) and bool(chains.get("telemetry_chain_ok"))):
        _fail(f"ERROR: hash-chain verification failed: {chains}")

    eval_obj: Dict[str, Any] = {
        "schema_version": 1,
        "seed": int(seed),
        "csg_hash": str(csg_hash),
        "concept_id": str(concept_id),
        "allowed_goal_kind": str(allowed_goal_kind),
        "disallowed_goal_kind": str(disallowed_goal_kind),
        "allowed_output": str(allowed_res.get("output") or ""),
        "replay_output": str(replay_res.get("output") or ""),
        "replay_ok": True,
        "disallowed_reason": str(dis_meta.get("reason") or ""),
        "trace_sig_allowed": str(trace_sig),
        "trace_sig_disallowed": str(dis_trace_sig),
        "chains": dict(chains),
    }
    eval_sha = write_json(os.path.join(out_dir, "eval.json"), eval_obj)

    core = {
        "schema_version": 1,
        "seed": int(seed),
        "csg_hash": str(csg_hash),
        "concept_id": str(concept_id),
        "allowed_output": str(eval_obj.get("allowed_output") or ""),
        "disallowed_reason": str(eval_obj.get("disallowed_reason") or ""),
        "replay_ok": bool(eval_obj.get("replay_ok", False)),
        "chains": dict(chains),
        "sha256_eval_json": str(eval_sha),
    }
    summary_sha256 = sha256_text(canonical_json_dumps(core))
    smoke_summary = dict(core, summary_sha256=str(summary_sha256))
    write_json(os.path.join(out_dir, "smoke_summary.json"), smoke_summary)

    return {
        "out_dir": str(out_dir),
        "eval": dict(eval_obj),
        "smoke_summary": dict(smoke_summary),
        "summary_sha256": str(summary_sha256),
        "sha256": {
            "eval_json": str(eval_sha),
            "concepts_jsonl": sha256_file(logs.concepts_path),
            "concept_evidence_jsonl": sha256_file(logs.evidence_path),
            "concept_telemetry_jsonl": sha256_file(logs.telemetry_path),
            "smoke_summary_json": sha256_file(os.path.join(out_dir, "smoke_summary.json")),
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_base", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out_base = str(args.out_base)
    seed = int(args.seed)

    out1 = f"{out_base}_try1"
    out2 = f"{out_base}_try2"
    ensure_absent(out1)
    ensure_absent(out2)

    r1 = smoke_try(out_dir=out1, seed=seed)
    r2 = smoke_try(out_dir=out2, seed=seed)

    if str(r1.get("summary_sha256") or "") != str(r2.get("summary_sha256") or ""):
        _fail(f"ERROR: determinism failed: try1={r1.get('summary_sha256')} try2={r2.get('summary_sha256')}")

    out = {
        "ok": True,
        "seed": int(seed),
        "determinism": {"ok": True, "summary_sha256": str(r1.get("summary_sha256") or "")},
        "try1": {"out_dir": str(out1), "sha256": dict(r1.get("sha256") or {})},
        "try2": {"out_dir": str(out2), "sha256": dict(r2.get("sha256") or {})},
    }
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

