#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import canonical_json_dumps, sha256_hex
from atos_core.agent_v63 import extract_int
from atos_core.agent_v65 import build_v65_tasks, build_v65_toolbox
from atos_core.engine import Engine, EngineConfig
from atos_core.planner_v64 import search_plan
from atos_core.proof import program_sha256
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


def _events_sig(events: List[Dict[str, Any]]) -> str:
    return sha256_hex(canonical_json_dumps(events).encode("utf-8"))


def _task_spec_from_prompt(prompt_text: str) -> Dict[str, Any]:
    """
    Minimal deterministic parser: expects prompt_text to contain a first line:
      V65_SPEC=<json>
    """
    first = (prompt_text.splitlines() or [""])[0].strip()
    m = re.fullmatch(r"V65_SPEC=(\{.*\})", first)
    if not m:
        raise ValueError("bad_prompt_spec")
    return json.loads(m.group(1))


def _domain_for_category(cat: str) -> str:
    return "A" if str(cat) in {"parse", "json", "dialog"} else "B"


def _make_v65_prompts(tasks) -> List[Tuple[str, str, Dict[str, Any]]]:
    """
    Returns [(task_id, category, spec)] where prompt_text encodes the spec.
    """
    out: List[Tuple[str, str, Dict[str, Any]]] = []
    for t in tasks:
        kind = str(t.kind)
        cat = str(t.category)

        if kind == "extract_int":
            text = str(t.args.get("text") or "")
            strip0 = bool(t.args.get("strip0", False))
            n = int(extract_int(text, strip_one_zero=strip0))
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"text": str(text)},
                "target_output_type": "int",
                "validator_id": "int_value_exact",
                "expected": int(n),
                "expected_output_text": str(int(n)),
            }
        elif kind == "json_ab":
            a = int(t.args.get("a", 0) or 0)
            b = int(t.args.get("b", 0) or 0)
            exp = {"a": int(a), "b": int(b)}
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"a": int(a), "b": int(b)},
                "target_output_type": "str",
                "validator_id": "json_ab_int_exact",
                "expected": dict(exp),
                "expected_output_text": canonical_json_dumps(exp),
            }
        elif kind == "sum_two_texts":
            ta = str(t.args.get("text_a") or "")
            tb = str(t.args.get("text_b") or "")
            strip0_a = bool(t.args.get("strip0_a", False))
            strip0_b = bool(t.args.get("strip0_b", False))
            a = int(extract_int(ta, strip_one_zero=strip0_a))
            b = int(extract_int(tb, strip_one_zero=strip0_b))
            s = int(a) + int(b)
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"text_a": str(ta), "text_b": str(tb)},
                "target_output_type": "int",
                "validator_id": "int_value_exact",
                "expected": int(s),
                "expected_output_text": str(int(s)),
            }
        elif kind == "plan_json_sum":
            ta = str(t.args.get("text_a") or "")
            tb = str(t.args.get("text_b") or "")
            strip0_a = bool(t.args.get("strip0_a", False))
            strip0_b = bool(t.args.get("strip0_b", False))
            a = int(extract_int(ta, strip_one_zero=strip0_a))
            b = int(extract_int(tb, strip_one_zero=strip0_b))
            s = int(a) + int(b)
            exp = {"a": int(s), "b": int(b)}
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"text_a": str(ta), "text_b": str(tb)},
                "target_output_type": "str",
                "validator_id": "json_ab_int_exact",
                "expected": dict(exp),
                "expected_output_text": canonical_json_dumps(exp),
            }
        elif kind == "dialog_greet":
            prefix = str(t.args.get("prefix") or "")
            name = str(t.args.get("name") or "")
            suffix = str(t.args.get("suffix") or "")
            out_text = prefix + name + suffix
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"prefix": str(prefix), "name": str(name), "suffix": str(suffix)},
                "target_output_type": "str",
                "validator_id": "text_exact",
                "expected": str(out_text),
                "expected_output_text": str(out_text),
            }
        elif kind == "dialog_format_int":
            prefix = str(t.args.get("prefix") or "")
            suffix = str(t.args.get("suffix") or "")
            n = int(t.args.get("n", 0) or 0)
            out_text = prefix + str(int(n)) + suffix
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"prefix": str(prefix), "n": int(n), "suffix": str(suffix)},
                "target_output_type": "str",
                "validator_id": "text_exact",
                "expected": str(out_text),
                "expected_output_text": str(out_text),
            }
        elif kind == "sum_two_texts_sentence":
            ta = str(t.args.get("text_a") or "")
            tb = str(t.args.get("text_b") or "")
            strip0_a = bool(t.args.get("strip0_a", False))
            strip0_b = bool(t.args.get("strip0_b", False))
            prefix = str(t.args.get("prefix") or "")
            suffix = str(t.args.get("suffix") or "")
            a = int(extract_int(ta, strip_one_zero=strip0_a))
            b = int(extract_int(tb, strip_one_zero=strip0_b))
            s = int(a) + int(b)
            out_text = prefix + str(int(s)) + suffix
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"text_a": str(ta), "text_b": str(tb), "prefix": str(prefix), "suffix": str(suffix)},
                "target_output_type": "str",
                "validator_id": "text_exact",
                "expected": str(out_text),
                "expected_output_text": str(out_text),
            }
        elif kind == "sum_two_ints_sentence":
            a = int(t.args.get("a", 0) or 0)
            b = int(t.args.get("b", 0) or 0)
            prefix = str(t.args.get("prefix") or "")
            suffix = str(t.args.get("suffix") or "")
            s = int(a) + int(b)
            out_text = prefix + str(int(s)) + suffix
            spec = {
                "task_id": str(t.task_id),
                "category": str(cat),
                "domain": _domain_for_category(cat),
                "inputs": {"a": int(a), "b": int(b), "prefix": str(prefix), "suffix": str(suffix)},
                "target_output_type": "str",
                "validator_id": "text_exact",
                "expected": str(out_text),
                "expected_output_text": str(out_text),
            }
        else:
            raise ValueError(f"unknown_kind:{kind}")

        prompt_text = "V65_SPEC=" + canonical_json_dumps(spec)
        spec2 = _task_spec_from_prompt(prompt_text)
        out.append((str(t.task_id), str(cat), dict(spec2)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--acts_run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner_max_depth", type=int, default=8)
    ap.add_argument("--planner_max_expansions", type=int, default=8000)
    ap.add_argument("--limit_tasks", type=int, default=0)
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
    run_id = f"agent_loop_v65␟acts={base_acts_sha256}␟seed={int(args.seed)}"

    store = ActStore.load_jsonl(base_acts)
    store_hash_excl = store.content_hash(exclude_kinds=["gate_table_ctxsig", "concept_csv", "goal"])
    toolbox = build_v65_toolbox(step=1, store_hash_excl_semantic=store_hash_excl, overhead_bits=1024)
    for act in toolbox.values():
        if store.get(act.id) is None:
            store.add(act)

    engine = Engine(store, seed=int(args.seed), config=EngineConfig())
    toolbox_ids = {v.id for v in toolbox.values()}
    concept_acts = [a for a in store.concept_acts() if str(a.id) in toolbox_ids]
    concept_acts.sort(key=lambda a: str(a.id))

    tasks = build_v65_tasks()
    if int(args.limit_tasks) > 0:
        tasks = tasks[: int(args.limit_tasks)]
    prompts = _make_v65_prompts(tasks)

    trace_rows: List[Dict[str, Any]] = []
    task_rows: List[Dict[str, Any]] = []

    by_cat_total: Dict[str, int] = {}
    by_cat_ok: Dict[str, int] = {}
    tasks_ok = 0
    steps_total = 0
    ethics_passed = 0
    ic_count = 0

    for task_id, cat, spec in prompts:
        by_cat_total[cat] = by_cat_total.get(cat, 0) + 1

        available_inputs = spec.get("inputs") if isinstance(spec.get("inputs"), dict) else {}
        target_type = str(spec.get("target_output_type") or "")
        validator_id = str(spec.get("validator_id") or "")
        expected = spec.get("expected")
        expected_text = str(spec.get("expected_output_text") or "")

        plan_res = search_plan(
            engine=engine,
            concept_acts=concept_acts,
            available_inputs=dict(available_inputs),
            target_output_type=target_type,
            validator_id=validator_id,
            expected=expected,
            expected_output_text=expected_text,
            max_depth=int(args.planner_max_depth),
            max_expansions=int(args.planner_max_expansions),
        )

        task_ok = bool(plan_res.ok)
        if not task_ok:
            task_rows.append(
                {"task_id": str(task_id), "category": str(cat), "ok": False, "reason": str(plan_res.reason)}
            )
            continue

        env: Dict[str, Any] = dict(available_inputs)
        plan_dict = plan_res.to_dict()
        last_out_text = ""
        for si, step in enumerate(plan_res.plan):
            act = store.get(step.concept_id)
            if act is None:
                _fail(f"ERROR: missing concept in store: {step.concept_id}")
            concept_inputs = {k: env[v] for k, v in step.bind.items()}

            is_last = si == (len(plan_res.plan) - 1)
            r = engine.execute_concept_csv(
                concept_act_id=str(act.id),
                inputs=dict(concept_inputs),
                expected=(expected if is_last else None),
                step=int(steps_total),
                max_depth=8,
                validate_output=bool(is_last),
            )
            meta = r.get("meta") if isinstance(r, dict) else {}
            meta = meta if isinstance(meta, dict) else {}
            eth = meta.get("ethics") if isinstance(meta.get("ethics"), dict) else {}
            unc = meta.get("uncertainty") if isinstance(meta.get("uncertainty"), dict) else {}
            if bool(eth.get("ok", True)):
                ethics_passed += 1
            if str(unc.get("mode_out") or "") == "IC":
                ic_count += 1

            out_val = r.get("output")
            out_text = str(meta.get("output_text") or "")
            ok = bool(meta.get("ok", False))
            reason = str(meta.get("reason") or "")

            events_full = r.get("events") if isinstance(r, dict) else []
            events_full = events_full if isinstance(events_full, list) else []
            events: List[Dict[str, Any]] = []
            for ev in events_full[: int(args.max_events_per_step)]:
                if isinstance(ev, dict):
                    events.append(dict(ev))
            events_truncated = len(events_full) > len(events)

            env[str(step.out_var)] = out_val
            last_out_text = out_text

            trace_rows.append(
                {
                    "run_id": str(run_id),
                    "ctx_sig": f"agent_v65␟task={task_id}␟step={si}",
                    "task_id": str(task_id),
                    "category": str(cat),
                    "prompt_text": "V65_SPEC=" + canonical_json_dumps(spec),
                    "plan": plan_dict if si == 0 else None,
                    "step_id": int(steps_total),
                    "goal_id": "",
                    "inputs": dict(concept_inputs),
                    "output_text": str(out_text),
                    "expected_output_text": str(expected_text if is_last else ""),
                    "ok": bool(ok),
                    "reason": str(reason),
                    "selected_concept_id": str(act.id),
                    "program_sig": str(program_sha256(act)),
                    "events_sig": str(_events_sig(events)),
                    "events_truncated": bool(events_truncated),
                    "events": events,
                }
            )

            steps_total += 1
            if (not ok) or (is_last and out_text != expected_text):
                task_ok = False

        if task_ok:
            tasks_ok += 1
            by_cat_ok[cat] = by_cat_ok.get(cat, 0) + 1
        task_rows.append(
            {
                "task_id": str(task_id),
                "category": str(cat),
                "ok": bool(task_ok),
                "final_output_text": str(last_out_text),
                "final_expected_output_text": str(expected_text),
            }
        )

    trace_path = os.path.join(traces_dir, "agent_trace_v65.jsonl")
    trace_sha = write_jsonl(trace_path, trace_rows)

    summary = {
        "seed": int(args.seed),
        "tasks_total": int(len(prompts)),
        "tasks_ok": int(tasks_ok),
        "pass_rate": float(tasks_ok / max(1, len(prompts))),
        "steps_total": int(steps_total),
        "by_category_total": dict(sorted(by_cat_total.items(), key=lambda kv: str(kv[0]))),
        "by_category_ok": dict(sorted(by_cat_ok.items(), key=lambda kv: str(kv[0]))),
        "ethics_checks_passed": int(ethics_passed),
        "uncertainty_ic_count": int(ic_count),
        "agent_trace_sha256": str(trace_sha),
    }

    summary_csv = os.path.join(args.out, "summary.csv")
    ensure_absent(summary_csv)
    with open(summary_csv, "w", encoding="utf-8") as f:
        f.write("seed,tasks_total,tasks_ok,pass_rate,steps_total,ethics_checks_passed,uncertainty_ic_count,agent_trace_sha256\n")
        f.write(
            f"{summary['seed']},{summary['tasks_total']},{summary['tasks_ok']},{summary['pass_rate']},{summary['steps_total']},{summary['ethics_checks_passed']},{summary['uncertainty_ic_count']},{summary['agent_trace_sha256']}\n"
        )

    summary_json = os.path.join(args.out, "summary.json")
    ensure_absent(summary_json)
    with open(summary_json, "w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": summary, "tasks": task_rows}, ensure_ascii=False, indent=2, sort_keys=True))

    print(json.dumps({"summary": summary, "out_dir": str(args.out)}, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
