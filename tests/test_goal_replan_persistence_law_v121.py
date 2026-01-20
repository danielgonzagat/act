import json
import os
import tempfile
import unittest

from atos_core.goal_replan_persistence_law_v121 import (
    FAIL_REASON_EXHAUSTION_WITHOUT_PROOF_V121,
    FAIL_REASON_GOAL_DIED_ON_FAIL_V121,
    OK_REASON_GOAL_EXHAUSTED_V121,
    verify_goal_replan_persistence_law_v121,
)


def _write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            f.write("\n")


class TestGoalReplanPersistenceLawV121(unittest.TestCase):
    def test_fail_without_replan_when_remaining_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(
                os.path.join(td, "conversation_turns.jsonl"),
                [{"payload": {"role": "user", "turn_id": "turn_u0", "turn_index": 0, "created_step": 0}}],
            )
            _write_jsonl(
                os.path.join(td, "action_plans.jsonl"),
                [
                    {
                        "user_turn_id": "turn_u0",
                        "objective_kind": "COMM_RESPOND",
                        "chosen_ok": False,
                        "ranked_candidates": [{"act_id": "A"}, {"act_id": "B"}],
                        "attempted_actions": [{"act_id": "A", "eval_id": "e0", "ok": False}],
                    }
                ],
            )
            _write_jsonl(os.path.join(td, "objective_evals.jsonl"), [{"eval_id": "e0", "verdict": {"ok": False, "reason": "x"}}])
            res = verify_goal_replan_persistence_law_v121(run_dir=str(td), max_replans_per_turn=3)
            self.assertFalse(res.ok)
            self.assertEqual(res.reason, FAIL_REASON_GOAL_DIED_ON_FAIL_V121)

    def test_pass_when_waiting_for_user(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(
                os.path.join(td, "conversation_turns.jsonl"),
                [{"payload": {"role": "user", "turn_id": "turn_u0", "turn_index": 0, "created_step": 0}}],
            )
            _write_jsonl(
                os.path.join(td, "action_plans.jsonl"),
                [
                    {
                        "user_turn_id": "turn_u0",
                        "objective_kind": "COMM_ASK_CLARIFY",
                        "chosen_ok": True,
                        "ranked_candidates": [{"act_id": "ask"}],
                        "attempted_actions": [{"act_id": "ask", "eval_id": "e0", "ok": True}],
                    }
                ],
            )
            _write_jsonl(os.path.join(td, "objective_evals.jsonl"), [{"eval_id": "e0", "verdict": {"ok": True, "reason": ""}}])
            res = verify_goal_replan_persistence_law_v121(run_dir=str(td), max_replans_per_turn=3)
            self.assertTrue(res.ok)
            self.assertEqual(res.reason, "ok")

    def test_exhaustion_declared_without_proof_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(
                os.path.join(td, "conversation_turns.jsonl"),
                [{"payload": {"role": "user", "turn_id": "turn_u0", "turn_index": 0, "created_step": 0}}],
            )
            _write_jsonl(
                os.path.join(td, "action_plans.jsonl"),
                [
                    {
                        "user_turn_id": "turn_u0",
                        "objective_kind": "COMM_RESPOND",
                        "chosen_ok": False,
                        "ranked_candidates": [{"act_id": "A"}, {"act_id": "B"}],
                        "attempted_actions": [
                            {"act_id": "A", "eval_id": "e0", "ok": False},
                            {"act_id": "B", "eval_id": "e1", "ok": False},
                        ],
                    }
                ],
            )
            _write_jsonl(os.path.join(td, "objective_evals.jsonl"), [{"eval_id": "e0", "verdict": {"ok": False, "reason": "x"}}])
            res = verify_goal_replan_persistence_law_v121(run_dir=str(td), max_replans_per_turn=3)
            self.assertFalse(res.ok)
            self.assertEqual(res.reason, FAIL_REASON_EXHAUSTION_WITHOUT_PROOF_V121)

    def test_exhaustion_with_proof_passes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            _write_jsonl(
                os.path.join(td, "conversation_turns.jsonl"),
                [{"payload": {"role": "user", "turn_id": "turn_u0", "turn_index": 0, "created_step": 0}}],
            )
            _write_jsonl(
                os.path.join(td, "action_plans.jsonl"),
                [
                    {
                        "user_turn_id": "turn_u0",
                        "objective_kind": "COMM_RESPOND",
                        "chosen_ok": False,
                        "ranked_candidates": [{"act_id": "A"}, {"act_id": "B"}],
                        "attempted_actions": [
                            {"act_id": "A", "eval_id": "e0", "ok": False},
                            {"act_id": "B", "eval_id": "e1", "ok": False},
                        ],
                    }
                ],
            )
            _write_jsonl(
                os.path.join(td, "objective_evals.jsonl"),
                [
                    {"eval_id": "e0", "verdict": {"ok": False, "reason": "x"}},
                    {"eval_id": "e1", "verdict": {"ok": False, "reason": "y"}},
                ],
            )
            res = verify_goal_replan_persistence_law_v121(run_dir=str(td), max_replans_per_turn=3)
            self.assertTrue(res.ok)
            self.assertEqual(res.reason, OK_REASON_GOAL_EXHAUSTED_V121)

