from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

from atos_core.act import canonical_json_dumps
from atos_core.goal_plan_eval_gate_v114 import (
    FAIL_REASON_EXHAUSTED_PLANS_V114,
    FAIL_REASON_MISSING_GOAL_V114,
    FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V114,
    verify_goal_plan_eval_law_v114,
)


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(canonical_json_dumps(r))
            f.write("\n")


class TestGoalPlanEvalGateV114(unittest.TestCase):
    def test_missing_goal_creates_failure(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            _write_jsonl(
                run_dir / "conversation_turns.jsonl",
                [
                    {"payload": {"conversation_id": "c", "role": "user", "created_step": 0, "turn_index": 0, "turn_id": ""}},
                    {"payload": {"conversation_id": "c", "role": "assistant", "created_step": 1, "turn_index": 1, "turn_id": "t_a"}},
                ],
            )
            _write_jsonl(run_dir / "action_plans.jsonl", [])
            _write_jsonl(run_dir / "objective_evals.jsonl", [])

            gate = verify_goal_plan_eval_law_v114(run_dir=str(run_dir), max_replans_per_turn=3, write_mind_graph=False)
            self.assertFalse(gate.ok)
            self.assertEqual(gate.reason, FAIL_REASON_MISSING_GOAL_V114)

    def test_missing_eval_satisfies_blocks_render(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            turn_id = "turn_u"
            eval_id = "eval_x"
            _write_jsonl(
                run_dir / "conversation_turns.jsonl",
                [
                    {"payload": {"conversation_id": "c", "role": "user", "created_step": 0, "turn_index": 0, "turn_id": turn_id, "refs": {}}},
                    {"payload": {"conversation_id": "c", "role": "assistant", "created_step": 1, "turn_index": 1, "turn_id": "turn_a", "refs": {"eval_id": eval_id}}},
                ],
            )
            _write_jsonl(
                run_dir / "action_plans.jsonl",
                [
                    {
                        "user_turn_id": turn_id,
                        "user_turn_index": 0,
                        "created_step": 1,
                        "plan_id": "plan_1",
                        "chosen_action_id": "act_x",
                        "chosen_eval_id": eval_id,
                        "chosen_ok": True,
                        "ranked_candidates": [{"act_id": "act_x", "expected_success": 1.0, "expected_cost": 1.0}],
                        "attempted_actions": [{"act_id": "act_x", "eval_id": eval_id, "ok": False}],
                    }
                ],
            )
            _write_jsonl(
                run_dir / "objective_evals.jsonl",
                [
                    {
                        "eval_id": eval_id,
                        "turn_id": turn_id,
                        "step": 1,
                        "objective_kind": "COMM_RESPOND",
                        "objective_id": "objective_v90_comm_respond",
                        "action_concept_id": "act_x",
                        "expected_text_sig": "e",
                        "output_text_sig": "o",
                        "verdict": {"ok": False, "reason": "x", "score": 0},
                    }
                ],
            )
            gate = verify_goal_plan_eval_law_v114(run_dir=str(run_dir), max_replans_per_turn=3, write_mind_graph=False)
            self.assertFalse(gate.ok)
            self.assertEqual(gate.reason, FAIL_REASON_RENDER_BLOCKED_NO_EVAL_SATISFIES_V114)

    def test_satisfied_eval_allows_render(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            turn_id = "turn_u"
            eval_id = "eval_ok"
            _write_jsonl(
                run_dir / "conversation_turns.jsonl",
                [
                    {"payload": {"conversation_id": "c", "role": "user", "created_step": 0, "turn_index": 0, "turn_id": turn_id, "refs": {}}},
                    {"payload": {"conversation_id": "c", "role": "assistant", "created_step": 1, "turn_index": 1, "turn_id": "turn_a", "refs": {"eval_id": eval_id}}},
                ],
            )
            _write_jsonl(
                run_dir / "action_plans.jsonl",
                [
                    {
                        "user_turn_id": turn_id,
                        "user_turn_index": 0,
                        "created_step": 1,
                        "plan_id": "plan_1",
                        "chosen_action_id": "act_x",
                        "chosen_eval_id": eval_id,
                        "chosen_ok": True,
                        "ranked_candidates": [{"act_id": "act_x", "expected_success": 1.0, "expected_cost": 1.0}],
                        "attempted_actions": [{"act_id": "act_x", "eval_id": eval_id, "ok": True}],
                    }
                ],
            )
            _write_jsonl(
                run_dir / "objective_evals.jsonl",
                [
                    {
                        "eval_id": eval_id,
                        "turn_id": turn_id,
                        "step": 1,
                        "objective_kind": "COMM_RESPOND",
                        "objective_id": "objective_v90_comm_respond",
                        "action_concept_id": "act_x",
                        "expected_text_sig": "e",
                        "output_text_sig": "e",
                        "verdict": {"ok": True, "reason": "", "score": 1},
                    }
                ],
            )
            gate = verify_goal_plan_eval_law_v114(run_dir=str(run_dir), max_replans_per_turn=3, write_mind_graph=False)
            self.assertTrue(gate.ok)
            self.assertEqual(gate.reason, "ok")

    def test_replanning_budget_marks_exhausted(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            run_dir = Path(td)
            turn_id = "turn_u"
            eval_id = "eval_fail"
            _write_jsonl(
                run_dir / "conversation_turns.jsonl",
                [
                    {"payload": {"conversation_id": "c", "role": "user", "created_step": 0, "turn_index": 0, "turn_id": turn_id, "refs": {}}},
                    {"payload": {"conversation_id": "c", "role": "assistant", "created_step": 1, "turn_index": 1, "turn_id": "turn_a", "refs": {"eval_id": eval_id}}},
                ],
            )
            _write_jsonl(
                run_dir / "action_plans.jsonl",
                [
                    {
                        "user_turn_id": turn_id,
                        "user_turn_index": 0,
                        "created_step": 1,
                        "plan_id": "plan_1",
                        "chosen_action_id": "act_x",
                        "chosen_eval_id": eval_id,
                        "chosen_ok": False,
                        "ranked_candidates": [
                            {"act_id": "act_x", "expected_success": 1.0, "expected_cost": 1.0},
                            {"act_id": "act_y", "expected_success": 1.0, "expected_cost": 1.0},
                            {"act_id": "act_z", "expected_success": 1.0, "expected_cost": 1.0},
                        ],
                        "attempted_actions": [
                            {"act_id": "act_x", "eval_id": "e1", "ok": False},
                            {"act_id": "act_y", "eval_id": "e2", "ok": False},
                        ],
                    }
                ],
            )
            _write_jsonl(
                run_dir / "objective_evals.jsonl",
                [
                    {
                        "eval_id": eval_id,
                        "turn_id": turn_id,
                        "step": 1,
                        "objective_kind": "COMM_RESPOND",
                        "objective_id": "objective_v90_comm_respond",
                        "action_concept_id": "act_x",
                        "expected_text_sig": "e",
                        "output_text_sig": "o",
                        "verdict": {"ok": False, "reason": "x", "score": 0},
                    }
                ],
            )
            gate = verify_goal_plan_eval_law_v114(run_dir=str(run_dir), max_replans_per_turn=2, write_mind_graph=False)
            self.assertFalse(gate.ok)
            self.assertEqual(gate.reason, FAIL_REASON_EXHAUSTED_PLANS_V114)


if __name__ == "__main__":
    unittest.main()

