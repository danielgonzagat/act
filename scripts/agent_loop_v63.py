#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.agent_v63 import build_v63_tasks, build_v63_toolbox, plan_task_steps, stable_act_id
from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.proof import program_sha256
from atos_core.act import Act


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


def write_jsonl(path: str, rows: List[Dict[str, Any]]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ensure_absent(path)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")
    os.replace(tmp, path)
    return sha256_file(path)


def write_text(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    ensure_absent(path)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return sha256_file(path)


def _events_sig(events: List[Dict[str, Any]]) -> str:
    return sha256_hex(canonical_json_dumps(events).encode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True, help="Base run dir containing acts.jsonl")
    ap.add_argument("--out", required=True, help="WORM out dir (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_steps", type=int, default=512)
    ap.add_argument("--max_depth", type=int, default=8)
    ap.add_argument("--max_events_per_step", type=int, default=128)
    args = ap.parse_args()

    ensure_absent(args.out)
    os.makedirs(args.out, exist_ok=False)
    traces_dir = os.path.join(args.out, "traces")
    os.makedirs(traces_dir, exist_ok=False)

    base_acts = os.path.join(args.acts_run, "acts.jsonl")
    if not os.path.exists(base_acts):
        _fail(f"ERROR: missing base acts.jsonl: {base_acts}")
    base_acts_sha256 = sha256_file(base_acts)
    run_id = f"agent_loop_v63␟acts={base_acts_sha256}␟seed={int(args.seed)}"

    store = ActStore.load_jsonl(base_acts)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])

    toolbox = build_v63_toolbox(step=1, store_hash_excl_semantic=store_hash_excl, overhead_bits=1024)
    for act in toolbox.values():
        if store.get(act.id) is None:
            store.add(act)

    engine = Engine(store, seed=int(args.seed), config=EngineConfig())

    tasks = build_v63_tasks()
    trace_rows: List[Dict[str, Any]] = []
    task_results: List[Dict[str, Any]] = []
    steps_total = 0
    tasks_ok = 0

    by_cat_total: Dict[str, int] = {}
    by_cat_ok: Dict[str, int] = {}

    ethics_passed = 0
    uncertainty_ic_count = 0

    for ti, task in enumerate(tasks):
        cat = str(task.category)
        by_cat_total[cat] = by_cat_total.get(cat, 0) + 1

        plan = plan_task_steps(task)
        plan_view = [{"step_name": s["step_name"], "concept_key": s["concept_key"]} for s in plan]

        task_ok = True
        last_output_text = ""
        last_expected_text = ""

        for si, step in enumerate(plan):
            if steps_total >= int(args.max_steps):
                _fail("ERROR: max_steps exceeded")

            concept_key = str(step["concept_key"])
            concept = toolbox.get(concept_key)
            if concept is None:
                _fail(f"ERROR: missing toolbox concept: {concept_key}")

            inputs = step["inputs"]
            expected = step["expected"]
            expected_text = str(step["expected_output_text"] or "")

            goal_ev = {
                "name": "goal_v0",
                "meta": {"title": f"agent_v63:{task.task_id}:{step['step_name']}"},
                "goal": {
                    "priority": 10,
                    "concept_id": str(concept.id),
                    "inputs": dict(inputs),
                    "expected": expected,
                },
            }
            goal_body = {
                "kind": "goal",
                "version": 1,
                "match": {},
                "program": [],
                "evidence": goal_ev,
                "deps": [],
                "active": True,
            }
            goal_id = stable_act_id("act_goal_", goal_body)
            if store.get(goal_id) is None:
                store.add(
                    Act(
                        id=str(goal_id),
                        version=1,
                        created_at="1970-01-01T00:00:00+00:00",
                        kind="goal",
                        match={},
                        program=[],
                        evidence=goal_ev,
                        cost={"overhead_bits": 0},
                        deps=[],
                        active=True,
                    )
                )

            r = engine.execute_goal(goal_act_id=str(goal_id), step=int(steps_total), max_depth=int(args.max_depth))
            tr = r.get("trace") if isinstance(r, dict) else {}
            tr = tr if isinstance(tr, dict) else {}
            meta = tr.get("concept_meta") if isinstance(tr.get("concept_meta"), dict) else {}
            meta = meta if isinstance(meta, dict) else {}

            out_text = str(meta.get("output_text") or "")
            ok = bool(r.get("ok", False))
            reason = str(r.get("reason") or "")
            selected_concept_id = str(tr.get("selected_concept_id") or "")

            eth = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
            unc = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
            if bool(eth.get("ok", True)):
                ethics_passed += 1
            if str(unc.get("mode_out") or "") == "IC":
                uncertainty_ic_count += 1

            events_full = r.get("events") if isinstance(r, dict) else []
            events_full = events_full if isinstance(events_full, list) else []
            events: List[Dict[str, Any]] = []
            for ev in events_full[: int(args.max_events_per_step)]:
                if isinstance(ev, dict):
                    events.append(dict(ev))
            events_truncated = len(events_full) > len(events)

            ctx_sig = f"agent_v63␟task={task.task_id}␟step={si}"
            row = {
                "run_id": str(run_id),
                "ctx_sig": str(ctx_sig),
                "task_id": str(task.task_id),
                "category": str(cat),
                "prompt_text": str(task.prompt_text),
                "plan": plan_view if si == 0 else None,
                "step_id": int(steps_total),
                "goal_id": str(goal_id),
                "step_name": str(step["step_name"]),
                "inputs": dict(inputs),
                "output_text": str(out_text),
                "expected_output_text": str(expected_text),
                "ok": bool(ok),
                "reason": str(reason),
                "selected_concept_id": str(selected_concept_id),
                "program_sig": str(program_sha256(concept)),
                "events_sig": str(_events_sig(events)),
                "events_truncated": bool(events_truncated),
                "events": events,
            }
            trace_rows.append(row)

            last_output_text = out_text
            last_expected_text = expected_text
            if (not ok) or (out_text != expected_text):
                task_ok = False

            steps_total += 1

        if task_ok:
            tasks_ok += 1
            by_cat_ok[cat] = by_cat_ok.get(cat, 0) + 1

        task_results.append(
            {
                "task_id": str(task.task_id),
                "category": str(cat),
                "ok": bool(task_ok),
                "final_output_text": str(last_output_text),
                "final_expected_output_text": str(last_expected_text),
            }
        )

    # Persist trace + summaries (WORM).
    trace_path = os.path.join(traces_dir, "agent_trace_v63.jsonl")
    trace_sha256 = write_jsonl(trace_path, trace_rows)

    summary = {
        "seed": int(args.seed),
        "tasks_total": int(len(tasks)),
        "tasks_ok": int(tasks_ok),
        "pass_rate": float(tasks_ok / max(1, len(tasks))),
        "steps_total": int(steps_total),
        "by_category_total": dict(sorted(by_cat_total.items(), key=lambda kv: str(kv[0]))),
        "by_category_ok": dict(sorted(by_cat_ok.items(), key=lambda kv: str(kv[0]))),
        "ethics_checks_passed": int(ethics_passed),
        "uncertainty_ic_count": int(uncertainty_ic_count),
        "agent_trace_sha256": str(trace_sha256),
    }

    summary_csv = os.path.join(args.out, "summary.csv")
    ensure_absent(summary_csv)
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("seed,tasks_total,tasks_ok,pass_rate,steps_total,ethics_checks_passed,uncertainty_ic_count,agent_trace_sha256\n")
        f.write(
            f"{summary['seed']},{summary['tasks_total']},{summary['tasks_ok']},{summary['pass_rate']},{summary['steps_total']},{summary['ethics_checks_passed']},{summary['uncertainty_ic_count']},{summary['agent_trace_sha256']}\n"
        )

    summary_json = os.path.join(args.out, "summary.json")
    write_text(
        summary_json,
        json.dumps({"summary": summary, "tasks": task_results}, ensure_ascii=False, indent=2, sort_keys=True),
    )

    print(json.dumps({"summary": summary, "out_dir": str(args.out)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
