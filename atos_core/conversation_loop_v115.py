from __future__ import annotations

import json
import os
from typing import Any, Dict, Sequence

from .conversation_loop_v110 import run_conversation_v110
from .goal_persistence_v115 import render_fail_response_v115
from .goal_plan_eval_gate_v115 import verify_goal_plan_eval_law_v115


def _write_once_json(path: str, obj: Any) -> None:
    if os.path.exists(path):
        raise ValueError(f"worm_exists:{path}")
    tmp = path + ".tmp"
    if os.path.exists(tmp):
        raise ValueError(f"tmp_exists:{tmp}")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True))
        f.write("\n")
    os.replace(tmp, path)


def run_conversation_v115(
    *,
    user_turn_texts: Sequence[str],
    out_dir: str,
    seed: int,
    max_replans_per_turn: int = 3,
) -> Dict[str, Any]:
    """
    V115 wrapper around V110 runtime enforcing the law as a *runtime gate*:
      - run the base deterministic pipeline (V110),
      - verify GOAL->PLAN->EVAL law + nontriviality,
      - if gate fails: return FAIL_RESPONSE (do not surface the generated assistant output).

    This keeps V90–V114 baselines intact while making "render without SATISFIES" impossible for V115 consumers.
    """
    res = run_conversation_v110(user_turn_texts=list(user_turn_texts), out_dir=str(out_dir), seed=int(seed))
    gate = verify_goal_plan_eval_law_v115(
        run_dir=str(out_dir),
        max_replans_per_turn=int(max_replans_per_turn),
        write_mind_graph=True,
        write_snapshots=True,
    )

    final_ok = bool(gate.ok)
    final_reason = str(gate.reason)
    final_text = ""
    if not final_ok:
        final_text = render_fail_response_v115(str(final_reason))

    final_obj = {
        "schema_version": 115,
        "kind": "final_response_v115",
        "ok": bool(final_ok),
        "reason": str(final_reason if not final_ok else "ok"),
        "fail_response_text": str(final_text),
    }
    _write_once_json(os.path.join(str(out_dir), "final_response_v115.json"), dict(final_obj))

    out = dict(res)
    out["gate_v115_ok"] = bool(gate.ok)
    out["gate_v115_reason"] = str(gate.reason)
    out["gate_v115"] = dict(gate.details)
    out["final_response_v115"] = dict(final_obj)
    return out

