#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import Act, Patch, canonical_json_dumps
from atos_core.agent_v63 import make_goal_act
from atos_core.csv_miner import materialize_concept_act_from_candidate, mine_csv_candidates
from atos_core.engine import Engine, EngineConfig
from atos_core.ethics import validate_act_for_promotion
from atos_core.ledger import Ledger
from atos_core.proof import act_body_sha256_placeholder, build_concept_pcc_certificate_v2, verify_concept_pcc_v2
from atos_core.store import ActStore
from atos_core.toc import detect_duplicate, toc_eval, verify_concept_toc_v1


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


def _domain_from_ctx_sig(ctx_sig: str) -> str:
    s = str(ctx_sig)
    if "task=v63_parse_" in s or "task=v63_json_" in s or "task=v65_dialog_" in s:
        return "A"
    if "task=v63_math_" in s or "task=v63_plan_" in s or "task=v65_math_" in s or "task=v65_plan_" in s:
        return "B"
    return "B"


def _unique_vectors_by_domain(
    examples: Sequence[Dict[str, Any]],
    *,
    domain: str,
    min_vectors: int = 3,
) -> List[Dict[str, Any]]:
    want = str(domain)
    exs = [e for e in examples if isinstance(e, dict)]
    exs.sort(key=lambda e: (str(e.get("ctx_sig") or ""), str(e.get("expected_sig") or "")))
    uniq: set = set()
    out: List[Dict[str, Any]] = []
    for ex in exs:
        if _domain_from_ctx_sig(str(ex.get("ctx_sig") or "")) != want:
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
    ap.add_argument("--planner_max_depth", type=int, default=8)
    ap.add_argument("--planner_max_expansions", type=int, default=8000)
    ap.add_argument("--max_events_per_step", type=int, default=128)
    ap.add_argument("--top_k_candidates", type=int, default=12)
    ap.add_argument("--max_total_overhead_bits", type=int, default=8192)
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
    run_id = f"agent_trace_mine_v65␟acts={base_acts_sha256}␟seed={int(args.seed)}"

    store_base = ActStore.load_jsonl(base_acts)
    store_hash_excl = store_base.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])

    # (1) Agent loop v65 → INS traces.
    agent_loop_out = os.path.join(args.out, "agent_loop")
    ensure_absent(agent_loop_out)
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "agent_loop_v65.py"),
        "--acts_run",
        str(args.acts_run),
        "--out",
        str(agent_loop_out),
        "--seed",
        str(int(args.seed)),
        "--planner_max_depth",
        str(int(args.planner_max_depth)),
        "--planner_max_expansions",
        str(int(args.planner_max_expansions)),
        "--max_events_per_step",
        str(int(args.max_events_per_step)),
    ]
    subprocess.run(cmd, check=True)

    agent_trace_src = os.path.join(agent_loop_out, "traces", "agent_trace_v65.jsonl")
    agent_summary_src = os.path.join(agent_loop_out, "summary.json")
    if not os.path.exists(agent_trace_src) or not os.path.exists(agent_summary_src):
        _fail("ERROR: agent_loop_v65 did not produce expected artifacts")

    agent_trace_path = os.path.join(traces_dir, "agent_trace_v65.jsonl")
    ensure_absent(agent_trace_path)
    with open(agent_trace_src, "rb") as f:
        b = f.read()
    with open(agent_trace_path + ".tmp", "wb") as f:
        f.write(b)
    os.replace(agent_trace_path + ".tmp", agent_trace_path)
    agent_trace_sha256 = sha256_file(agent_trace_path)

    with open(agent_summary_src, "r", encoding="utf-8") as f:
        agent_summary_obj = json.load(f)
    agent_summary = agent_summary_obj.get("summary") if isinstance(agent_summary_obj, dict) else {}
    agent_tasks_total = int(agent_summary.get("tasks_total", 0) or 0) if isinstance(agent_summary, dict) else 0
    agent_tasks_ok = int(agent_summary.get("tasks_ok", 0) or 0) if isinstance(agent_summary, dict) else 0
    agent_pass_rate = float(agent_tasks_ok / max(1, agent_tasks_total))
    if agent_tasks_total < 40:
        _fail(f"ERROR: expected >=40 tasks, got {agent_tasks_total}")
    if agent_pass_rate < 0.80:
        _fail(f"ERROR: agent_loop_v65 pass_rate too low: {agent_pass_rate}")
    if int(agent_summary.get("uncertainty_ic_count", 0) or 0) != 0:
        _fail("ERROR: agent_loop_v65 uncertainty_ic_count must be 0")

    # (2) Mine candidates from agent traces.
    candidates = mine_csv_candidates(
        agent_trace_path,
        min_ops=2,
        max_ops=6,
        bits_per_op=128,
        overhead_bits=int(args.concept_overhead_bits),
        max_examples_per_candidate=200,
    )
    if len(candidates) < 2:
        _fail(f"ERROR: expected >=2 candidates, got {len(candidates)}")
    mined_candidates_path = os.path.join(args.out, "mined_candidates.json")
    ensure_absent(mined_candidates_path)
    with open(mined_candidates_path, "w", encoding="utf-8") as f:
        f.write(json.dumps([c.to_dict() for c in candidates], ensure_ascii=False, indent=2, sort_keys=True))

    # (3) Materialize top-K and attach PCC v2 + ToC v1 (deterministic).
    top_k = max(1, int(args.top_k_candidates))
    materialized: List[Act] = []
    rejected: List[Dict[str, Any]] = []

    store_exec = ActStore(acts=dict(store_base.acts), next_id_int=int(store_base.next_id_int))
    existing_for_duplicates: List[Act] = list(store_exec.concept_acts())

    for idx, cand in enumerate(candidates[:top_k]):
        concept = materialize_concept_act_from_candidate(
            cand,
            step=200 + idx,
            store_content_hash_excluding_semantic=store_hash_excl,
            title=f"mined_concept_v65_rank{idx:02d}",
            overhead_bits=int(args.concept_overhead_bits),
            meta={"builder": "agent_trace_v65", "trace_file_sha256": str(agent_trace_sha256), "gain_bits_est": int(cand.gain_bits_est)},
        )

        dup = detect_duplicate(concept, existing=existing_for_duplicates + materialized)
        if dup is not None:
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "duplicate", "dup": dict(dup)})
            continue

        vecA = _unique_vectors_by_domain(cand.examples, domain="A", min_vectors=3)
        vecB = _unique_vectors_by_domain(cand.examples, domain="B", min_vectors=3)
        if len(vecA) < 3 or len(vecB) < 3:
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "not_enough_toc_vectors", "got_A": len(vecA), "got_B": len(vecB)})
            continue

        ethics = validate_act_for_promotion(concept)
        if not bool(ethics.ok):
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "ethics_fail_closed", "ethics": ethics.to_dict()})
            continue

        store_tmp = ActStore(acts=dict(store_exec.acts), next_id_int=int(store_exec.next_id_int))
        store_tmp.add(concept)
        toc = toc_eval(
            concept_act=concept,
            vectors_A=list(vecA),
            vectors_B=list(vecB),
            store=store_tmp,
            domain_A="A",
            domain_B="B",
            min_vectors_per_domain=3,
        )
        if not bool(toc.get("pass_A", False)) or not bool(toc.get("pass_B", False)):
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "toc_eval_failed", "toc": toc})
            continue

        cert = build_concept_pcc_certificate_v2(
            concept,
            store=store_tmp,
            mined_from={
                "trace_file_sha256": str(agent_trace_sha256),
                "store_hash_excluding_semantic": str(store_hash_excl),
                "seed": int(args.seed),
                "candidate_sig": str(cand.candidate_sig),
            },
            test_vectors=list(vecA[:3]),
            ethics_verdict=ethics.to_dict(),
            uncertainty_policy="no_ic",
            toc_v1=dict(toc),
        )
        concept.evidence.setdefault("certificate_v2", cert)
        concept.evidence["certificate_v2"]["hashes"]["act_body_sha256"] = act_body_sha256_placeholder(concept)

        pv = verify_concept_pcc_v2(concept, store_tmp)
        if not bool(pv.ok):
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "pcc_verify_failed", "pcc": pv.to_dict()})
            continue
        tv = verify_concept_toc_v1(concept, store=store_tmp)
        if not bool(tv.ok):
            rejected.append({"candidate_sig": str(cand.candidate_sig), "reason": "toc_verify_failed", "toc": tv.to_dict()})
            continue

        materialized.append(concept)

    if len(materialized) < 2:
        _fail(f"ERROR: expected >=2 materialized concepts, got {len(materialized)}")

    # (4) Promotion under budget (deterministic ordering).
    max_bits = int(args.max_total_overhead_bits)
    used_bits = 0
    promoted_concepts: List[Act] = []
    promotion_rejections: List[Dict[str, Any]] = []
    for c in materialized:
        ob = int(c.cost.get("overhead_bits", int(args.concept_overhead_bits)) or 0)
        if used_bits + ob > max_bits:
            promotion_rejections.append({"act_id": str(c.id), "reason": "budget_exceeded", "used_bits": used_bits, "need_bits": ob})
            continue
        promoted_concepts.append(c)
        used_bits += ob

    if len(promoted_concepts) < 2:
        _fail(f"ERROR: expected >=2 promoted concepts, got {len(promoted_concepts)}")

    # (5) Create goals from ToC vectors (A+B) for each promoted concept.
    goals: List[Act] = []
    goal_rows_expected: List[Dict[str, Any]] = []
    goal_step = 1000
    for c in promoted_concepts:
        cert = c.evidence.get("certificate_v2") if isinstance(c.evidence, dict) else {}
        cert = cert if isinstance(cert, dict) else {}
        toc = cert.get("toc_v1") if isinstance(cert.get("toc_v1"), dict) else {}
        vecA = toc.get("vectors_A") if isinstance(toc.get("vectors_A"), list) else []
        vecB = toc.get("vectors_B") if isinstance(toc.get("vectors_B"), list) else []
        vecs = list(vecA)[:3] + list(vecB)[:3]
        if len(vecs) < 6:
            _fail(f"ERROR: promoted concept missing ToC vectors: {c.id}")
        for vi, v in enumerate(vecs):
            inputs = v.get("inputs") if isinstance(v.get("inputs"), dict) else {}
            expected = v.get("expected")
            expected_text = str(v.get("expected_output_text") or "")
            g = make_goal_act(
                step=int(goal_step),
                store_hash_excl_semantic=store_hash_excl,
                title=f"v65_goal:{c.id}:{vi}",
                concept_id=str(c.id),
                inputs=dict(inputs),
                expected=expected,
                priority=10,
                overhead_bits=0,
            )
            goal_step += 1
            goals.append(g)
            goal_rows_expected.append({"goal_id": str(g.id), "expected_output_text": str(expected_text)})

    promo_dir = os.path.join(args.out, "promotion")
    os.makedirs(promo_dir, exist_ok=False)
    acts_promoted = os.path.join(promo_dir, "acts_promoted.jsonl")
    appended = list(promoted_concepts) + list(goals)
    promoted_sha256 = write_promoted_acts_preserve_order(base_acts_path=base_acts, out_acts_path=acts_promoted, appended_acts=appended)

    promotion_ledger_path = os.path.join(promo_dir, "promotion_ledger.jsonl")
    ledger = Ledger(path=promotion_ledger_path)
    for i, a in enumerate(appended):
        patch = Patch(kind="ADD_ACT", payload={"act_id": str(a.id), "kind": str(a.kind)})
        ledger.append(step=int(i), patch=patch, acts_hash=str(promoted_sha256), metrics={"promotion": True, "act_id": str(a.id), "kind": str(a.kind)}, snapshot_path=None)
    promotion_chain_ok = bool(ledger.verify_chain())

    promotion_manifest = {
        "run_id": str(run_id),
        "base_acts_path": str(base_acts),
        "base_acts_sha256": str(base_acts_sha256),
        "store_hash_excluding_semantic": str(store_hash_excl),
        "agent_trace_path": str(agent_trace_path),
        "agent_trace_sha256": str(agent_trace_sha256),
        "agent_pass_rate": float(agent_pass_rate),
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
        f.write(json.dumps(promotion_manifest, ensure_ascii=False, indent=2, sort_keys=True))

    # (6) From-store: execute goals and compare expected outputs.
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
        "agent_tasks_total": int(agent_tasks_total),
        "agent_tasks_ok": int(agent_tasks_ok),
        "agent_pass_rate": float(agent_pass_rate),
        "agent_trace_sha256": str(agent_trace_sha256),
        "mined_candidates_total": int(len(candidates)),
        "promoted_concepts_total": int(len(promoted_concepts)),
        "gain_bits_est_total": int(gain_bits_est_total),
        "goals_total": int(len(goals)),
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
            "seed,agent_tasks_total,agent_tasks_ok,agent_pass_rate,mined_candidates_total,promoted_concepts_total,gain_bits_est_total,goals_total,mismatch_goals,reuse_rate,call_depth_max,ethics_checks_passed,uncertainty_ic_count,promotion_chain_ok\n"
        )
        f.write(
            f"{summary['seed']},{summary['agent_tasks_total']},{summary['agent_tasks_ok']},{summary['agent_pass_rate']},{summary['mined_candidates_total']},{summary['promoted_concepts_total']},{summary['gain_bits_est_total']},{summary['goals_total']},{summary['mismatch_goals']},{summary['reuse_rate']},{summary['call_depth_max']},{summary['ethics_checks_passed']},{summary['uncertainty_ic_count']},{int(summary['promotion_chain_ok'])}\n"
        )

    summary_json = os.path.join(args.out, "summary.json")
    ensure_absent(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary, "promotion_manifest": promotion_manifest, "from_store": from_store_rows}, ensure_ascii=False, indent=2, sort_keys=True))

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
            "name": "V65_DIALOG_TOC_PLANNER_AGENT_LOOP",
            "acts_source_run": str(args.acts_run),
            "out_dir": str(args.out),
            "commands": [" ".join(sys.argv)],
            "verify_chain": bool(promotion_chain_ok),
            "sha256": sha,
            "sha256_paths": sha_paths,
            "summary": summary,
        }
        with open(args.freeze_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(freeze, ensure_ascii=False, indent=2, sort_keys=True))

    print(json.dumps({"summary": summary, "out_dir": str(args.out)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

