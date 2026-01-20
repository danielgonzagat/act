#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from atos_core.goal_act_v75 import make_goal_act_v75
from atos_core.goal_objective_v88 import evaluate_goal_success_v88
from atos_core.objective_v88 import make_objective_eq_text_act_v88
from atos_core.store import ActStore


def _fail(msg: str, *, code: int = 2) -> None:
    print(msg, file=sys.stderr)
    raise SystemExit(code)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed = int(args.seed)
    store = ActStore()

    objective = make_objective_eq_text_act_v88(created_step=0, match={})
    store.add(objective)

    goal = make_goal_act_v75(
        goal_kind="v88_demo",
        bindings={"x": "0004", "y": "0008"},
        output_key="sum",
        expected="12",
        validator_id="text_exact",
        created_step=0,
    )
    # Opt-in V88: attach objective_act_id.
    goal.evidence["goal"]["objective_act_id"] = str(objective.id)
    store.add(goal)

    goal_output = "12"
    verdict, dbg = evaluate_goal_success_v88(
        store=store,
        seed=seed,
        goal_act=goal,
        goal_output=goal_output,
        step=0,
    )

    out = {
        "ok": True,
        "goal_id": str(goal.id),
        "objective_id": str(objective.id),
        "goal_output": str(goal_output),
        "verdict": verdict.to_dict(),
        "debug": dict(dbg),
    }
    if not bool(verdict.ok):
        _fail(f"ERROR: expected objective verdict ok=true, got={out}")
    print(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

