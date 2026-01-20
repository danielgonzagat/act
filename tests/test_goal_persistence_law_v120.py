import json
import os
import tempfile
import unittest

from atos_core.conversation_v96 import append_chained_jsonl_v96
from atos_core.goal_ledger_v99 import GoalEventV99, goal_id_v99
from atos_core.goal_persistence_law_v120 import (
    FAIL_REASON_GOAL_DONE_BEFORE_HORIZON_V120,
    FAIL_REASON_GOAL_DONE_MISSING_PROGRESS_PROOF_V120,
    FAIL_REASON_MISSING_GOAL_EVENTS_V120,
    verify_goal_persistence_law_v120,
)


def _write_jsonl(path: str, rows) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True))
            f.write("\n")


def _write_goal_events(*, run_dir: str, goal_id: str, done_ts_turn_index: int, created_step: int) -> None:
    goal_path = os.path.join(str(run_dir), "goal_events.jsonl")
    prev_hash = None
    prev_sig = ""

    ev_add = GoalEventV99(
        conversation_id="conv_test",
        ts_turn_index=0,
        op="GOAL_ADD",
        goal_id=str(goal_id),
        parent_goal_id="",
        priority=100,
        status="active",
        text="demo",
        cause_type="user_intent",
        cause_id="turn0",
        created_step=int(created_step),
        prev_event_sig=str(prev_sig),
    ).to_dict()
    prev_hash = append_chained_jsonl_v96(
        goal_path,
        {"time": ev_add.get("created_at"), "step": int(created_step), "event": "GOAL_EVENT", "payload": dict(ev_add)},
        prev_hash=prev_hash,
    )
    prev_sig = str(ev_add.get("event_sig") or "")

    ev_done = GoalEventV99(
        conversation_id="conv_test",
        ts_turn_index=int(done_ts_turn_index),
        op="GOAL_DONE",
        goal_id=str(goal_id),
        parent_goal_id="",
        priority=100,
        status="done",
        text="demo",
        cause_type="system",
        cause_id="turn_done",
        created_step=int(created_step + 1),
        prev_event_sig=str(prev_sig),
    ).to_dict()
    append_chained_jsonl_v96(
        goal_path,
        {"time": ev_done.get("created_at"), "step": int(created_step + 1), "event": "GOAL_EVENT", "payload": dict(ev_done)},
        prev_hash=prev_hash,
    )


class TestGoalPersistenceLawV120(unittest.TestCase):
    def test_missing_goal_events_fails(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            res = verify_goal_persistence_law_v120(run_dir=str(td), expected_autopilot_total_steps=5)
            self.assertFalse(res.ok)
            self.assertEqual(res.reason, FAIL_REASON_MISSING_GOAL_EVENTS_V120)

    def test_ok_with_valid_progress_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            goal_id = goal_id_v99(conversation_id="conv_test", ts_turn_index=0, text="demo", parent_goal_id="")
            _write_goal_events(run_dir=str(td), goal_id=str(goal_id), done_ts_turn_index=0, created_step=0)
            turns_path = os.path.join(str(td), "conversation_turns.jsonl")
            _write_jsonl(
                turns_path,
                [
                    {"payload": {"role": "user", "turn_index": 0, "text": "ok"}},
                    {"payload": {"role": "assistant", "turn_index": 1, "text": "AVANÇO 5/5: demo"}},
                ],
            )
            res = verify_goal_persistence_law_v120(run_dir=str(td), expected_autopilot_total_steps=5)
            self.assertTrue(res.ok)
            self.assertEqual(res.reason, "ok")

    def test_fail_when_step_less_than_total(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            goal_id = goal_id_v99(conversation_id="conv_test", ts_turn_index=0, text="demo", parent_goal_id="")
            _write_goal_events(run_dir=str(td), goal_id=str(goal_id), done_ts_turn_index=0, created_step=0)
            turns_path = os.path.join(str(td), "conversation_turns.jsonl")
            _write_jsonl(
                turns_path,
                [
                    {"payload": {"role": "user", "turn_index": 0, "text": "ok"}},
                    {"payload": {"role": "assistant", "turn_index": 1, "text": "AVANÇO 4/5: demo"}},
                ],
            )
            res = verify_goal_persistence_law_v120(run_dir=str(td), expected_autopilot_total_steps=5)
            self.assertFalse(res.ok)
            self.assertEqual(res.reason, FAIL_REASON_GOAL_DONE_BEFORE_HORIZON_V120)

    def test_fail_when_missing_progress_marker(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            goal_id = goal_id_v99(conversation_id="conv_test", ts_turn_index=0, text="demo", parent_goal_id="")
            _write_goal_events(run_dir=str(td), goal_id=str(goal_id), done_ts_turn_index=0, created_step=0)
            turns_path = os.path.join(str(td), "conversation_turns.jsonl")
            _write_jsonl(
                turns_path,
                [
                    {"payload": {"role": "user", "turn_index": 0, "text": "ok"}},
                    {"payload": {"role": "assistant", "turn_index": 1, "text": "done"}},
                ],
            )
            res = verify_goal_persistence_law_v120(run_dir=str(td), expected_autopilot_total_steps=5)
            self.assertFalse(res.ok)
            self.assertEqual(res.reason, FAIL_REASON_GOAL_DONE_MISSING_PROGRESS_PROOF_V120)

