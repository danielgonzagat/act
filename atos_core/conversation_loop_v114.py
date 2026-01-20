from __future__ import annotations

from typing import Any, Dict, Sequence

from .conversation_loop_v110 import run_conversation_v110
from .goal_plan_eval_gate_v114 import verify_goal_plan_eval_law_v114


def run_conversation_v114(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
    max_replans_per_turn: int = 3,
) -> Dict[str, Any]:
    """
    V114 wrapper around V110 runtime enforcing:
      INPUT -> GOAL(ATO) -> PLAN(ATO) -> EVAL(ATO satisfies==true) before render.

    Implementation is audit-first:
      - run V110 as-is (keeps prior baselines stable),
      - then construct MindGraph nodes/edges + verify GOAL/PLAN/EVAL law,
      - write `goal_plan_eval_summary_v114.json` and `mind_graph_v114/`.
    """
    res = run_conversation_v110(user_turn_texts=list(user_turn_texts), out_dir=str(out_dir), seed=int(seed))
    gate = verify_goal_plan_eval_law_v114(
        run_dir=str(out_dir),
        max_replans_per_turn=int(max_replans_per_turn),
        write_mind_graph=True,
    )
    out = dict(res)
    out["gate_v114_ok"] = bool(gate.ok)
    out["gate_v114_reason"] = str(gate.reason)
    out["gate_v114"] = dict(gate.details)
    return out

