#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.act import estimate_act_cost_bits
from atos_core.engine import Engine, EngineConfig
from atos_core.ledger import Ledger
from atos_core.store import ActStore
from atos_core.suite import (
    run_skill_suite,
    skill_suite_tasks_for_pack,
    suite_metrics_from_transcripts,
)


_SNAP_RE = re.compile(r"^step(\d+)_acts\.jsonl$")


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _find_latest_snapshot(run_dir: str) -> Optional[str]:
    snaps = os.path.join(run_dir, "snapshots")
    if not os.path.isdir(snaps):
        return None
    best: Tuple[int, str] = (-1, "")
    for name in os.listdir(snaps):
        m = _SNAP_RE.match(name)
        if not m:
            continue
        try:
            step = int(m.group(1))
        except Exception:
            continue
        path = os.path.join(snaps, name)
        if step > best[0]:
            best = (step, path)
    return best[1] if best[0] >= 0 else None


def _resolve_acts_path(*, run: Optional[str], acts: Optional[str]) -> str:
    if acts:
        p = str(acts)
        if not os.path.exists(p):
            raise SystemExit(f"acts_not_found:{p}")
        return p
    if not run:
        raise SystemExit("must_pass --run or --acts")
    run_dir = str(run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"run_dir_not_found:{run_dir}")
    # Prefer deterministic snapshot.
    snap = _find_latest_snapshot(run_dir)
    if snap:
        return snap
    p = os.path.join(run_dir, "acts.jsonl")
    if os.path.exists(p):
        return p
    raise SystemExit(f"no_acts_found_in_run:{run_dir}")


def _ledger_chain_ok(run_dir: str) -> Optional[bool]:
    path = os.path.join(str(run_dir), "ledger.jsonl")
    if not os.path.exists(path):
        return None
    try:
        return bool(Ledger(path).verify_chain())
    except Exception:
        return False


def _print_sample_transcripts(transcripts: List[Dict[str, Any]], *, limit: int) -> None:
    shown = 0
    for rec in transcripts:
        if shown >= int(limit):
            break
        task_id = str(rec.get("task_id") or "")
        turns = rec.get("turns", [])
        if not isinstance(turns, list) or not turns:
            continue
        shown += 1
        print(f"\n=== {task_id or f'task_{shown}'} ===")
        for t in turns:
            if not isinstance(t, dict):
                continue
            user = str(t.get("user", "")).strip()
            assistant = str(t.get("system", "")).strip()
            if user:
                print(f"User: {user}")
            if assistant:
                print(f"ACT:  {assistant}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="Training run dir (uses latest snapshot by default).")
    ap.add_argument("--acts", help="Path to acts snapshot jsonl.")
    ap.add_argument("--pack", default="sota_v8", help="Skill-suite pack (e.g., sota_v8).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=200)
    ap.add_argument("--prompt_history_k", type=int, default=0)
    ap.add_argument("--show_transcripts", type=int, default=3, help="Print N sample transcripts to stdout.")
    ap.add_argument("--json", action="store_true", help="Print full JSON report (machine-readable).")
    args = ap.parse_args()

    acts_path = _resolve_acts_path(run=args.run, acts=args.acts)
    store = ActStore.load_jsonl(acts_path)
    engine = Engine(store, seed=int(args.seed), config=EngineConfig(enable_contracts=False))

    tasks = skill_suite_tasks_for_pack(str(args.pack))

    t0 = time.time()
    transcripts, metrics = run_skill_suite(
        engine,
        tasks=tasks,
        max_new_tokens=int(args.max_new_tokens),
        prompt_history_k=int(args.prompt_history_k),
    )
    dt = max(1e-9, time.time() - t0)
    cost = suite_metrics_from_transcripts(transcripts)

    trace_tokens = int(cost.get("trace_tokens_total") or 0)
    tokens_per_s = float(trace_tokens) / float(dt) if trace_tokens > 0 else None
    model_cost_bits = int(sum(estimate_act_cost_bits(a) for a in store.active()))

    run_dir = str(args.run or "")
    report: Dict[str, Any] = {
        "pack": str(args.pack),
        "acts_path": str(acts_path),
        "sha256_acts": _sha256_file(str(acts_path)),
        "ledger_chain_ok": _ledger_chain_ok(run_dir) if run_dir else None,
        "seed": int(args.seed),
        "elapsed_s": float(dt),
        "trace_tokens_total": int(trace_tokens),
        "tokens_per_s_est": float(tokens_per_s) if tokens_per_s is not None else None,
        "model_cost_bits": int(model_cost_bits),
        "metrics": dict(metrics),
        "cost_metrics": dict(cost),
        "cross_context_reuse_example": metrics.get("concept_cross_context_reuse_example"),
    }

    if bool(args.json):
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    # Human-friendly summary (for demos / leigos).
    m = report["metrics"]
    print(f"pack={report['pack']} acts={report['acts_path']}")
    print(
        "utility_pass_rate="
        f"{m.get('pass_rate')} (plan={m.get('plan_pass_rate')}, json={m.get('json_pass_rate')}, "
        f"math={m.get('math_pass_rate')}, state={m.get('state_pass_rate')}, memory={m.get('memory_pass_rate')})"
    )
    print(
        "concept_used_rate="
        f"{m.get('concept_used_rate')} deep_rate={m.get('concept_deep_rate')} very_deep_rate={m.get('concept_very_deep_rate')} "
        f"calls_max_depth_mean={m.get('concept_calls_max_depth_mean')} static_depth_max={m.get('concept_static_depth_max')}"
    )
    print(f"tokens/s≈{report.get('tokens_per_s_est')}")
    ex = report.get("cross_context_reuse_example") or {}
    if isinstance(ex, dict) and ex.get("concept_id"):
        print(
            "cross_context_example="
            f"{ex.get('concept_id')} birth={ex.get('birth_tags')} used={ex.get('used_tags')} extra={ex.get('extra_used_tags')}"
        )
    if int(args.show_transcripts) > 0:
        _print_sample_transcripts(transcripts, limit=int(args.show_transcripts))


if __name__ == "__main__":
    main()
