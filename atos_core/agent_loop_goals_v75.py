from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .act import canonical_json_dumps, deterministic_iso, sha256_hex
from .agent_loop_v72 import run_goal_spec_v72
from .goal_act_v75 import goal_sig_v75, goal_v75_is_satisfied, goal_v75_update_from_run, list_goal_acts_v75
from .goal_spec_v72 import GoalSpecV72
from .store import ActStore
from .trace_v73 import TraceV73, trace_from_agent_loop_v72


def _fail(msg: str) -> None:
    raise ValueError(msg)


def ensure_absent(path: str) -> None:
    if os.path.exists(path):
        _fail(f"path_exists:{path}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_canon(obj: Any) -> str:
    return sha256_hex(canonical_json_dumps(obj).encode("utf-8"))


def run_dir_sha256_v75(*, run_dir: str) -> str:
    mg_dir = os.path.join(str(run_dir), "mind_graph")
    nodes_path = os.path.join(mg_dir, "mind_nodes.jsonl")
    edges_path = os.path.join(mg_dir, "mind_edges.jsonl")
    nodes_sha = sha256_file(nodes_path)
    edges_sha = sha256_file(edges_path)
    return _sha256_canon({"mind_nodes_sha256": str(nodes_sha), "mind_edges_sha256": str(edges_sha)})


def _append_jsonl(path: str, row: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(canonical_json_dumps(row))
        f.write("\n")


def _write_json(path: str, obj: Any) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def goal_spec_v72_from_goal_act_v75(goal_act) -> Tuple[Optional[GoalSpecV72], str]:
    if goal_act is None or str(getattr(goal_act, "kind", "")) != "goal_v75":
        return None, "not_goal_v75_act"
    ev = goal_act.evidence if isinstance(goal_act.evidence, dict) else {}
    goal = ev.get("goal") if isinstance(ev.get("goal"), dict) else {}

    goal_kind = str(goal.get("goal_kind") or "")
    bindings = goal.get("bindings") if isinstance(goal.get("bindings"), dict) else {}
    output_key = str(goal.get("output_key") or "")
    expected = goal.get("expected")
    validator_id = str(goal.get("validator_id") or "text_exact")
    if not output_key:
        return None, "missing_output_key"
    return (
        GoalSpecV72(
            goal_kind=str(goal_kind),
            bindings={str(k): bindings.get(k) for k in sorted(bindings.keys(), key=str)},
            output_key=str(output_key),
            expected=expected,
            validator_id=str(validator_id),
            created_step=0,
        ),
        "ok",
    )


def run_goals_v75(
    *,
    store: ActStore,
    seed: int,
    out_dir: str,
    max_rounds: int = 5,
    max_goals_per_round: int = 3,
    enable_promotion: bool = True,
    promotion_budget_bits: int = 1024,
) -> Dict[str, Any]:
    """
    Deterministic goals scheduler loop (MVP, audit-first):
      - selects active goal_v75 acts by id ASC
      - attempts satisfaction via run_goal_spec_v72
      - updates goal act evidence deterministically
      - emits append-only goals_v75_events.jsonl and traces_v75.json

    Note: enable_promotion/promotion_budget_bits are reserved for future integration; V75 smoke
    performs mining/promotion explicitly using V74 modules.
    """
    _ = bool(enable_promotion)
    _ = int(promotion_budget_bits)

    ensure_absent(out_dir)
    os.makedirs(out_dir, exist_ok=False)

    events_path = os.path.join(out_dir, "goals_v75_events.jsonl")
    ensure_absent(events_path)

    traces: List[TraceV73] = []
    attempts_total = 0

    step_ctr = 0
    store_hash_init = str(store.content_hash())

    for r in range(0, int(max_rounds)):
        goals = list_goal_acts_v75(store)
        pending = []
        for g in goals:
            if not goal_v75_is_satisfied(g):
                pending.append(g)

        if not pending:
            break
        pending.sort(key=lambda a: str(getattr(a, "id", "")))

        to_run = pending[: int(max_goals_per_round)]
        skipped = pending[int(max_goals_per_round) :]
        for g in skipped:
            store_hash_before = str(store.content_hash())
            ev_row = {
                "created_at": deterministic_iso(step=int(step_ctr)),
                "round": int(r),
                "goal_id": str(getattr(g, "id", "")),
                "goal_sig": str(goal_sig_v75(g)),
                "decision": "skipped",
                "reason": "max_goals_per_round",
                "plan_sig": "",
                "trace_sig": "",
                "steps_total": 0,
                "got": "",
                "expected": "",
                "goal_active": True,
                "goal_progress": 0.0,
                "goal_satisfied": False,
                "store_hash_before": str(store_hash_before),
                "store_hash_after": str(store_hash_before),
            }
            _append_jsonl(events_path, ev_row)
            step_ctr += 1

        for g in to_run:
            store_hash_before = str(store.content_hash())

            goal_spec, reason = goal_spec_v72_from_goal_act_v75(g)
            if goal_spec is None:
                ev_row = {
                    "created_at": deterministic_iso(step=int(step_ctr)),
                    "round": int(r),
                    "goal_id": str(getattr(g, "id", "")),
                    "goal_sig": str(goal_sig_v75(g)),
                    "decision": "failed",
                    "reason": str(reason),
                    "plan_sig": "",
                    "trace_sig": "",
                    "steps_total": 0,
                    "got": "",
                    "expected": "",
                    "goal_active": True,
                    "goal_progress": 0.0,
                    "goal_satisfied": False,
                    "store_hash_before": str(store_hash_before),
                    "store_hash_after": str(store_hash_before),
                }
                _append_jsonl(events_path, ev_row)
                step_ctr += 1
                continue

            gid_short = str(getattr(g, "id", ""))[:24]
            run_subdir = os.path.join(out_dir, f"round{int(r):02d}", f"goal_{gid_short}")
            ensure_absent(run_subdir)
            os.makedirs(run_subdir, exist_ok=False)

            res = run_goal_spec_v72(goal_spec=goal_spec, store=store, seed=int(seed), out_dir=run_subdir)
            tr = trace_from_agent_loop_v72(goal_spec=goal_spec, result=res)
            tr_sig = str(tr.trace_sig())
            traces.append(tr)
            attempts_total += 1

            run_sha = run_dir_sha256_v75(run_dir=run_subdir)
            updated = goal_v75_update_from_run(
                act=g,
                run_res=res,
                trace_sig=str(tr_sig),
                run_dir_sha256=str(run_sha),
                step=int(step_ctr),
            )
            store.add(updated)

            plan = res.get("plan") if isinstance(res.get("plan"), dict) else {}
            plan = plan if isinstance(plan, dict) else {}
            raw_steps = plan.get("steps") if isinstance(plan.get("steps"), list) else []
            final = res.get("final") if isinstance(res.get("final"), dict) else {}
            final = final if isinstance(final, dict) else {}

            ok = bool(res.get("ok", False))
            got = str(final.get("got") or "")
            expected = goal_spec.expected
            plan_sig = str(plan.get("plan_sig") or "")
            steps_total = int(len(raw_steps))
            satisfied = goal_v75_is_satisfied(updated)

            decision = "satisfied" if satisfied else "failed"
            reason2 = ""
            if not satisfied:
                reason2 = str(res.get("reason") or "")
                if not reason2:
                    v = final.get("validator") if isinstance(final.get("validator"), dict) else {}
                    reason2 = str(v.get("reason") or "")

            store_hash_after = str(store.content_hash())
            ev_row = {
                "created_at": deterministic_iso(step=int(step_ctr)),
                "round": int(r),
                "goal_id": str(updated.id),
                "goal_sig": str(goal_sig_v75(updated)),
                "decision": str(decision),
                "reason": str(reason2),
                "plan_sig": str(plan_sig),
                "trace_sig": str(tr_sig),
                "steps_total": int(steps_total),
                "got": str(got),
                "expected": str(expected),
                "goal_active": bool(not satisfied),
                "goal_progress": 1.0 if satisfied else 0.0,
                "goal_satisfied": bool(satisfied),
                "store_hash_before": str(store_hash_before),
                "store_hash_after": str(store_hash_after),
            }
            _append_jsonl(events_path, ev_row)
            step_ctr += 1

    # Persist traces for mining.
    traces_sorted = sorted([t.to_canonical_dict(include_sig=True) for t in traces], key=lambda d: str(d.get("trace_sig") or ""))
    traces_path = os.path.join(out_dir, "traces_v75.json")
    _write_json(traces_path, {"schema_version": 1, "traces": list(traces_sorted)})

    return {
        "schema_version": 1,
        "seed": int(seed),
        "store_hash_init": str(store_hash_init),
        "store_hash_final": str(store.content_hash()),
        "goals_total": int(len(list_goal_acts_v75(store))),
        "goals_satisfied": int(len([g for g in list_goal_acts_v75(store) if goal_v75_is_satisfied(g)])),
        "attempts_total": int(attempts_total),
        "traces_total": int(len(traces_sorted)),
        "trace_sigs": [str(t.get("trace_sig") or "") for t in traces_sorted],
        "artifacts": {"traces_v75_json_sha256": sha256_file(traces_path), "goals_v75_events_jsonl_sha256": sha256_file(events_path)},
        "__internal_traces": list(traces),
    }
