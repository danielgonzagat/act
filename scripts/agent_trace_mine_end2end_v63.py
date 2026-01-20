#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Instruction, Patch, canonical_json_dumps, deterministic_iso, sha256_hex
from atos_core.agent_v63 import build_v63_tasks, build_v63_toolbox, make_concept_act, make_goal_act, plan_task_steps
from atos_core.csv_miner import materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import validate_act_for_promotion
from atos_core.ledger import Ledger
from atos_core.proof import act_body_sha256_placeholder, build_concept_pcc_certificate_v2, verify_concept_pcc_v2
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


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"ERROR: path already exists: {path}")


def write_jsonl(path: str, rows: Sequence[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def write_promoted_acts_preserve_order(
    *, base_acts_path: str, out_acts_path: str, appended_acts: Sequence[Act]
) -> str:
    with open(base_acts_path, "rb") as f:
        base_bytes = f.read()
    if base_bytes and not base_bytes.endswith(b"\n"):
        base_bytes += b"\n"
    tail = b"".join(canonical_json_dumps(a.to_dict()).encode("utf-8") + b"\n" for a in appended_acts)
    tmp = out_acts_path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(base_bytes)
        f.write(tail)
    os.replace(tmp, out_acts_path)
    return sha256_file(out_acts_path)


def _unique_vectors_from_examples(examples: Sequence[Dict[str, Any]], *, min_vectors: int = 3) -> List[Dict[str, Any]]:
    uniq: set = set()
    out: List[Dict[str, Any]] = []
    for ex in examples:
        if not isinstance(ex, dict):
            continue
        sig = str(ex.get("expected_sig") or "")
        if not sig or sig in uniq:
            continue
        uniq.add(sig)
        out.append(
            {
                "inputs": dict(ex.get("inputs") or {}),
                "expected": ex.get("expected"),
                "expected_output_text": str(ex.get("expected_output_text") or ""),
            }
        )
        if len(out) >= int(min_vectors):
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=512)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--max_events_per_step", type=int, default=128)
    ap.add_argument("--top_k_candidates", type=int, default=6)
    ap.add_argument("--max_total_overhead_bits", type=int, default=4096)
    ap.add_argument("--concept_overhead_bits", type=int, default=1024)
    ap.add_argument("--patch_diff", default="")
    ap.add_argument("--freeze_path", default="")
    args = ap.parse_args()

    ensure_absent(args.out)
    if args.freeze_path:
        ensure_absent(args.freeze_path)
    os.makedirs(args.out, exist_ok=False)
    traces_dir = os.path.join(args.out, "traces")
    os.makedirs(traces_dir, exist_ok=False)

    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"ERROR: missing base acts.jsonl: {base_acts}")
    base_acts_sha256 = sha256_file(base_acts)
    run_id = f"agent_trace_mine_v63␟acts={base_acts_sha256}␟seed={int(args.seed)}"

    store = ActStore.load_jsonl(base_acts)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])

    toolbox = build_v63_toolbox(step=1, store_hash_excl_semantic=store_hash_excl, overhead_bits=1024)
    for act in toolbox.values():
        if store.get(act.id) is None:
            store.add(act)
    engine = Engine(store, seed=int(args.seed), config=EngineConfig())

    # (1) Agent loop (deterministic) producing miner-ready traces (INS events).
    tasks = build_v63_tasks()
    trace_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    by_cat_total: Dict[str, int] = {}
    by_cat_ok: Dict[str, int] = {}
    tasks_ok = 0
    steps_total = 0
    ethics_passed = 0
    ic_count = 0

    for task in tasks:
        cat = str(task.category)
        by_cat_total[cat] = by_cat_total.get(cat, 0) + 1

        plan = plan_task_steps(task)
        plan_view = [{"step_name": s["step_name"], "concept_key": s["concept_key"]} for s in plan]

        task_ok = True
        last_out = ""
        last_exp = ""
        for si, step in enumerate(plan):
            if steps_total >= int(args.max_steps):
                _fail("ERROR: max_steps exceeded")

            concept_key = str(step["concept_key"])
            concept = toolbox.get(concept_key)
            if concept is None:
                _fail(f"ERROR: missing toolbox concept: {concept_key}")

            inputs = dict(step["inputs"])
            expected = step["expected"]
            expected_text = str(step["expected_output_text"] or "")

            # Create a goal act (deterministic id) and execute via Engine.execute_goal.
            g = make_goal_act(
                step=steps_total,
                store_hash_excl_semantic=store_hash_excl,
                title=f"agent_v63:{task.task_id}:{step['step_name']}",
                concept_id=str(concept.id),
                inputs=dict(inputs),
                expected=expected,
                priority=10,
                overhead_bits=0,
            )
            if store.get(g.id) is None:
                store.add(g)

            r = engine.execute_goal(goal_act_id=str(g.id), step=int(steps_total), max_depth=int(args.max_depth))
            tr = r.get("trace") if isinstance(r, dict) else {}
            tr = tr if isinstance(tr, dict) else {}
            meta = tr.get("concept_meta") if isinstance(tr.get("concept_meta"), dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            eth = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
            unc = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
            if bool(eth.get("ok", True)):
                ethics_passed += 1
            if str(unc.get("mode_out") or "") == "IC":
                ic_count += 1

            out_text = str(meta.get("output_text") or "")
            ok = bool(r.get("ok", False))
            reason = str(r.get("reason") or "")
            selected_concept_id = str(tr.get("selected_concept_id") or "")

            events_full = r.get("events") if isinstance(r, dict) else []
            events_full = events_full if isinstance(events_full, list) else []
            events: List[Dict[str, Any]] = []
            for ev in events_full[: int(args.max_events_per_step)]:
                if isinstance(ev, dict):
                    events.append(dict(ev))
            events_truncated = len(events_full) > len(events)

            row = {
                "run_id": str(run_id),
                "ctx_sig": f"agent_v63␟task={task.task_id}␟step={si}",
                "task_id": str(task.task_id),
                "category": str(cat),
                "prompt_text": str(task.prompt_text),
                "plan": plan_view if si == 0 else None,
                "step_id": int(steps_total),
                "goal_id": str(g.id),
                "step_name": str(step["step_name"]),
                "inputs": dict(inputs),
                "output_text": str(out_text),
                "expected_output_text": str(expected_text),
                "ok": bool(ok),
                "reason": str(reason),
                "selected_concept_id": str(selected_concept_id),
                "events_sig": sha256_hex(canonical_json_dumps(events).encode("utf-8")),
                "events_truncated": bool(events_truncated),
                "events": events,
            }
            trace_rows.append(row)

            last_out = out_text
            last_exp = expected_text
            if (not ok) or (out_text != expected_text):
                task_ok = False

            steps_total += 1

        if task_ok:
            tasks_ok += 1
            by_cat_ok[cat] = by_cat_ok.get(cat, 0) + 1

        task_rows.append(
            {
                "task_id": str(task.task_id),
                "category": str(cat),
                "ok": bool(task_ok),
                "final_output_text": str(last_out),
                "final_expected_output_text": str(last_exp),
            }
        )

    agent_trace_path = os.path.join(traces_dir, "agent_trace_v63.jsonl")
    agent_trace_sha256 = write_jsonl(agent_trace_path, trace_rows)

    # (2) Mine candidates from agent traces.
    candidates = mine_csv_candidates(
        agent_trace_path,
        min_ops=2,
        max_ops=6,
        bits_per_op=128,
        overhead_bits=int(args.concept_overhead_bits),
    )
    if len(candidates) < 2:
        _fail(f"ERROR: expected >=2 candidates, got {len(candidates)}")
    mined_candidates_path = os.path.join(args.out, "mined_candidates.json")
    ensure_absent(mined_candidates_path)
    with open(mined_candidates_path, "w", encoding="utf-8") as f:
        f.write(json.dumps([c.to_dict() for c in candidates], ensure_ascii=False, indent=2))

    # (3) Materialize top-K candidates and attach PCC v2 (deterministic).
    top_k = max(1, int(args.top_k_candidates))
    top = candidates[:top_k]
    materialized: List[Act] = []
    rejected: List[Dict[str, Any]] = []

    for idx, cand in enumerate(top):
        concept = materialize_concept_act_from_candidate(
            cand,
            step=200 + idx,
            store_content_hash_excluding_semantic=store_hash_excl,
            title=f"mined_concept_v63_rank{idx:02d}",
            overhead_bits=int(args.concept_overhead_bits),
            meta={
                "builder": "agent_trace_v63",
                "trace_file_sha256": str(agent_trace_sha256),
                "gain_bits_est": int(cand.gain_bits_est),
            },
        )
        vecs = _unique_vectors_from_examples(cand.examples, min_vectors=3)
        if len(vecs) < 3:
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "not_enough_vectors"})
            continue

        ethics = validate_act_for_promotion(concept)
        if not bool(ethics.ok):
            rejected.append(
                {"candidate_sig": str(cand.candidate_sig), "reason": "ethics_fail_closed", "ethics": ethics.to_dict()}
            )
            continue

        cert = build_concept_pcc_certificate_v2(
            concept,
            store=store,
            mined_from={
                "trace_file_sha256": str(agent_trace_sha256),
                "store_hash_excluding_semantic": str(store_hash_excl),
                "seed": int(args.seed),
                "candidate_sig": str(cand.candidate_sig),
            },
            test_vectors=list(vecs),
            ethics_verdict=ethics.to_dict(),
            uncertainty_policy="no_ic",
        )
        concept.evidence.setdefault("certificate_v2", cert)
        try:
            concept.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept)
        except Exception:
            pass

        pv = verify_concept_pcc_v2(concept, store)
        if not bool(pv.ok):
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "pcc_verify_failed", "pcc": pv.to_dict()})
            continue
        materialized.append(concept)

    if len(materialized) < 2:
        _fail(f"ERROR: expected >=2 materialized concepts, got {len(materialized)}")

    # (4) Create one wrapper concept with CSV_CALL to the first mined int concept.
    callee_int: Optional[Act] = None
    for c in materialized:
        iface = c.evidence.get("interface") if isinstance(c.evidence, dict) else {}
        iface = iface if isinstance(iface, dict) else {}
        if str(iface.get("validator_id") or "") == "int_value_exact":
            callee_int = c
            break
    if callee_int is None:
        _fail("ERROR: missing int concept for wrapper")

    callee_iface = callee_int.evidence.get("interface") if isinstance(callee_int.evidence, dict) else {}
    callee_iface = callee_iface if isinstance(callee_iface, dict) else {}
    in_schema = callee_iface.get("input_schema") if isinstance(callee_iface.get("input_schema"), dict) else {}
    in_schema = dict(in_schema)
    out_schema = callee_iface.get("output_schema") if isinstance(callee_iface.get("output_schema"), dict) else {}
    out_schema = dict(out_schema)
    validator_id = str(callee_iface.get("validator_id") or "")
    in_keys = sorted(list(in_schema.keys()))

    wrapper_prog: List[Instruction] = []
    bind: Dict[str, str] = {}
    for idx, name in enumerate(in_keys):
        wrapper_prog.append(Instruction("CSV_GET_INPUT", {"name": str(name), "out": f"in{idx}"}))
        bind[str(name)] = f"in{idx}"
    wrapper_prog.append(Instruction("CSV_CALL", {"concept_id": str(callee_int.id), "out": "out0", "bind": dict(bind)}))
    wrapper_prog.append(Instruction("CSV_RETURN", {"var": "out0"}))

    wrapper = make_concept_act(
        step=400,
        store_hash_excl_semantic=store_hash_excl,
        title="wrapper_call_mined_int_v63",
        program=wrapper_prog,
        interface={"input_schema": dict(in_schema), "output_schema": dict(out_schema), "validator_id": str(validator_id)},
        overhead_bits=int(args.concept_overhead_bits),
        meta={"builder": "wrapper_v63", "callee_id": str(callee_int.id)},
    )

    callee_cert = callee_int.evidence.get("certificate_v2") if isinstance(callee_int.evidence, dict) else {}
    callee_cert = callee_cert if isinstance(callee_cert, dict) else {}
    callee_vecs = callee_cert.get("test_vectors") if isinstance(callee_cert.get("test_vectors"), list) else []
    callee_vecs = list(callee_vecs)
    wrapper_vecs = callee_vecs[:3]
    if len(wrapper_vecs) < 3:
        _fail("ERROR: callee missing vectors for wrapper")

    wrapper_ethics = validate_act_for_promotion(wrapper)
    if not bool(wrapper_ethics.ok):
        _fail(f"ERROR: wrapper ethics failed: {wrapper_ethics.reason}:{wrapper_ethics.violated_laws}")

    store_for_wrapper = ActStore(acts=dict(store.acts), next_id_int=int(store.next_id_int))
    for c in materialized:
        store_for_wrapper.add(c)
    store_for_wrapper.add(wrapper)

    wrapper_cert = build_concept_pcc_certificate_v2(
        wrapper,
        store=store_for_wrapper,
        mined_from={
            "trace_file_sha256": str(agent_trace_sha256),
            "store_hash_excluding_semantic": str(store_hash_excl),
            "seed": int(args.seed),
            "kind": "wrapper_manual_v63",
            "callee_id": str(callee_int.id),
        },
        test_vectors=list(wrapper_vecs),
        ethics_verdict=wrapper_ethics.to_dict(),
        uncertainty_policy="no_ic",
    )
    wrapper.evidence.setdefault("certificate_v2", wrapper_cert)
    try:
        wrapper.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(wrapper)
    except Exception:
        pass
    wv = verify_concept_pcc_v2(wrapper, store_for_wrapper)
    if not bool(wv.ok):
        _fail(f"ERROR: wrapper PCC v2 failed: {wv.reason}:{wv.details}")

    # (5) Budgeted promotion under max_total_overhead_bits (deterministic).
    max_bits = max(0, int(args.max_total_overhead_bits))
    used_bits = 0
    promoted_concepts: List[Act] = []
    promotion_rejections: List[Dict[str, Any]] = []

    for c in list(materialized) + [wrapper]:
        ob = int(c.cost.get("overhead_bits", 1024))
        if used_bits + ob > max_bits:
            promotion_rejections.append({"act_id": str(c.id), "kind": "concept_csv", "reason": "budget_exceeded", "overhead_bits": ob})
            continue
        used_bits += ob
        promoted_concepts.append(c)

    if len(promoted_concepts) < 2:
        _fail("ERROR: budget caused <2 concepts to be promoted")

    # (6) Goals: >=3 per promoted concept.
    goals: List[Act] = []
    goal_rows_expected: List[Dict[str, Any]] = []
    gstep = 500
    for c in promoted_concepts:
        cert = c.evidence.get("certificate_v2") if isinstance(c.evidence, dict) else {}
        cert = cert if isinstance(cert, dict) else {}
        vecs = cert.get("test_vectors") if isinstance(cert.get("test_vectors"), list) else []
        vecs = list(vecs)
        if len(vecs) < 3:
            _fail(f"ERROR: promoted concept missing vectors: {c.id}")
        for vi, v in enumerate(vecs[:3]):
            if not isinstance(v, dict):
                continue
            inputs = v.get("inputs") if isinstance(v.get("inputs"), dict) else {}
            expected = v.get("expected")
            expected_text = str(v.get("expected_output_text") or "")
            g = make_goal_act(
                step=gstep,
                store_hash_excl_semantic=store_hash_excl,
                title=f"goal_v63_{c.id}_v{vi}",
                concept_id=str(c.id),
                inputs=dict(inputs),
                expected=expected,
                priority=10,
                overhead_bits=1024,
            )
            gstep += 1
            goals.append(g)
            goal_rows_expected.append(
                {
                    "goal_id": str(g.id),
                    "concept_id": str(c.id),
                    "inputs": dict(inputs),
                    "expected": expected,
                    "expected_output_text": expected_text,
                }
            )

    # (7) Promote append-only: base + concepts + goals; preserve base order; hash-chained promotion ledger.
    promo_dir = os.path.join(args.out, "promotion")
    os.makedirs(promo_dir, exist_ok=False)
    acts_promoted = os.path.join(promo_dir, "acts_promoted.jsonl")
    appended = list(promoted_concepts) + list(goals)
    promoted_sha256 = write_promoted_acts_preserve_order(base_acts_path=base_acts, out_acts_path=acts_promoted, appended_acts=appended)

    promotion_ledger_path = os.path.join(promo_dir, "promotion_ledger.jsonl")
    ledger = Ledger(path=promotion_ledger_path)
    for idx, a in enumerate(appended):
        patch = Patch(kind="ADD_ACT", payload={"act_id": str(a.id), "kind": str(a.kind)})
        ledger.append(step=int(idx), patch=patch, acts_hash=str(promoted_sha256), metrics={"promotion": True, "act_id": str(a.id), "kind": str(a.kind)}, snapshot_path=None)
    promotion_chain_ok = bool(ledger.verify_chain())

    promotion_manifest = {
        "base_acts_path": str(base_acts),
        "base_acts_sha256": str(base_acts_sha256),
        "store_hash_excluding_semantic": str(store_hash_excl),
        "agent_trace_path": str(agent_trace_path),
        "agent_trace_sha256": str(agent_trace_sha256),
        "mined_candidates_total": int(len(candidates)),
        "top_k_candidates": int(top_k),
        "materialized_concepts_total": int(len(materialized)),
        "rejected_materialization": list(rejected),
        "promotion_budget": {"max_total_overhead_bits": int(max_bits), "used_bits": int(used_bits)},
        "promoted_concept_ids": [str(c.id) for c in promoted_concepts],
        "promotion_rejections": list(promotion_rejections),
        "goal_ids": [str(g.id) for g in goals],
        "promotion_chain_ok": bool(promotion_chain_ok),
    }
    promotion_manifest_path = os.path.join(promo_dir, "promotion_manifest.json")
    ensure_absent(promotion_manifest_path)
    with open(promotion_manifest_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(promotion_manifest, ensure_ascii=False, indent=2))

    # (8) From-store: execute goals and compare expected outputs.
    store2 = ActStore.load_jsonl(acts_promoted)
    engine2 = Engine(store2, seed=int(args.seed), config=EngineConfig())

    mismatch_goals = 0
    call_depth_max = 0
    ethics_passed2 = 0
    ic_count2 = 0
    from_store_rows: List[Dict[str, Any]] = []

    for i, row in enumerate(goal_rows_expected):
        gid = str(row["goal_id"])
        r = engine2.execute_goal(goal_act_id=gid, step=i, max_depth=8)
        tr = r.get("trace") if isinstance(r, dict) else {}
        tr = tr if isinstance(tr, dict) else {}
        meta = tr.get("concept_meta") if isinstance(tr.get("concept_meta"), dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        eth2 = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
        unc2 = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
        if bool(eth2.get("ok", True)):
            ethics_passed2 += 1
        if str(unc2.get("mode_out") or "") == "IC":
            ic_count2 += 1
        evs = r.get("events") if isinstance(r, dict) else []
        if isinstance(evs, list):
            for ev in evs:
                if isinstance(ev, dict):
                    call_depth_max = max(call_depth_max, int(ev.get("depth", 0) or 0))
        out_text = str(meta.get("output_text") or "")
        expected_text = str(row.get("expected_output_text") or "")
        if out_text != expected_text:
            mismatch_goals += 1
        from_store_rows.append(
            {
                "goal_id": str(gid),
                "ok": bool(r.get("ok", False)),
                "output_text": out_text,
                "expected_output_text": expected_text,
                "selected_concept_id": str(tr.get("selected_concept_id") or ""),
                "ethics": eth2,
                "uncertainty": unc2,
            }
        )

    reuse = sum(1 for r in from_store_rows if str(r.get("selected_concept_id") or ""))
    reuse_rate = float(reuse / max(1, len(from_store_rows)))

    gain_bits_est_total = 0
    for c in promoted_concepts:
        meta = c.evidence.get("meta") if isinstance(c.evidence, dict) else {}
        meta = meta if isinstance(meta, dict) else {}
        try:
            gain_bits_est_total += int(meta.get("gain_bits_est") or 0)
        except Exception:
            pass

    summary = {
        "seed": int(args.seed),
        "tasks_total": int(len(tasks)),
        "tasks_ok": int(tasks_ok),
        "pass_rate": float(tasks_ok / max(1, len(tasks))),
        "by_category_total": dict(sorted(by_cat_total.items(), key=lambda kv: str(kv[0]))),
        "by_category_ok": dict(sorted(by_cat_ok.items(), key=lambda kv: str(kv[0]))),
        "steps_total": int(steps_total),
        "agent_trace_sha256": str(agent_trace_sha256),
        "mined_candidates_total": int(len(candidates)),
        "promoted_concepts_total": int(len(promoted_concepts)),
        "gain_bits_est_total": int(gain_bits_est_total),
        "goals_total": int(len(goal_rows_expected)),
        "mismatch_goals": int(mismatch_goals),
        "reuse_rate": float(reuse_rate),
        "call_depth_max": int(call_depth_max),
        "ethics_checks_passed": int(ethics_passed2),
        "uncertainty_ic_count": int(ic_count2),
        "promotion_chain_ok": bool(promotion_chain_ok),
    }

    summary_csv = os.path.join(args.out, "summary.csv")
    ensure_absent(summary_csv)
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write(
            "seed,tasks_total,tasks_ok,pass_rate,steps_total,mined_candidates_total,promoted_concepts_total,gain_bits_est_total,goals_total,mismatch_goals,reuse_rate,call_depth_max,ethics_checks_passed,uncertainty_ic_count,promotion_chain_ok\n"
        )
        f.write(
            f"{summary['seed']},{summary['tasks_total']},{summary['tasks_ok']},{summary['pass_rate']},{summary['steps_total']},{summary['mined_candidates_total']},{summary['promoted_concepts_total']},{summary['gain_bits_est_total']},{summary['goals_total']},{summary['mismatch_goals']},{summary['reuse_rate']},{summary['call_depth_max']},{summary['ethics_checks_passed']},{summary['uncertainty_ic_count']},{int(summary['promotion_chain_ok'])}\n"
        )

    summary_json = os.path.join(args.out, "summary.json")
    ensure_absent(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary, "promotion_manifest": promotion_manifest, "from_store": from_store_rows, "tasks": task_rows}, ensure_ascii=False, indent=2))

    if args.freeze_path:
        sha: Dict[str, str] = {
            "base_acts_jsonl": str(base_acts_sha256),
            "patch_diff": str(sha256_file(args.patch_diff)) if args.patch_diff and os.path.exists(args.patch_diff) else "",
            "agent_trace_jsonl": str(agent_trace_sha256),
            "mined_candidates_json": str(sha256_file(mined_candidates_path)),
            "acts_promoted_jsonl": str(promoted_sha256),
            "promotion_ledger_jsonl": str(sha256_file(promotion_ledger_path)),
            "promotion_manifest_json": str(sha256_file(promotion_manifest_path)),
            "summary_csv": str(sha256_file(summary_csv)),
            "summary_json": str(sha256_file(summary_json)),
        }
        sha_paths = {
            "base_acts_jsonl": str(os.path.join(args.acts_run, "acts.jsonl")),
            "patch_diff": str(args.patch_diff),
            "agent_trace_jsonl": str(agent_trace_path),
            "mined_candidates_json": str(mined_candidates_path),
            "acts_promoted_jsonl": str(acts_promoted),
            "promotion_ledger_jsonl": str(promotion_ledger_path),
            "promotion_manifest_json": str(promotion_manifest_path),
            "summary_csv": str(summary_csv),
            "summary_json": str(summary_json),
        }
        freeze = {
            "name": "V63_AGENT_TRACE_MINE_END2END",
            "acts_source_run": str(args.acts_run),
            "out_dir": str(args.out),
            "commands": [" ".join(sys.argv)],
            "verify_chain": bool(promotion_chain_ok),
            "sha256": sha,
            "sha256_paths": sha_paths,
            "summary": summary,
        }
        with open(args.freeze_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(freeze, ensure_ascii=False, indent=2))

    print(json.dumps({"summary": summary, "out_dir": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
