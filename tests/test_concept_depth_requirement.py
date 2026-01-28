from __future__ import annotations

import unittest
from typing import Any, Dict, Optional

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso
from atos_core.engine import Engine, EngineConfig
from atos_core.store import ActStore
from atos_core.suite import run_skill_suite


def _plan_expected_spec() -> Dict[str, Any]:
    inputs = {"goal_id": "G", "plan": "P", "a": 3, "b": 2}
    expected_obj = dict(inputs)
    expected_text = canonical_json_dumps(expected_obj)
    return {
        "input_keys": ["goal_id", "plan", "a", "b"],
        "inputs": dict(inputs),
        "ops": [
            {"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", "in2", "in3"], "out": "v0"},
            {"fn": "json_canonical", "in": ["v0"], "out": "v1"},
        ],
        "return_var": "v1",
        "expected_output_text": str(expected_text),
        "required_keys": ["goal_id", "plan", "a", "b"],
        "expected_values": dict(expected_obj),
    }


class _FakePlanEngine:
    def __init__(self, *, response: str, concept_calls_max_depth: int) -> None:
        self._response = str(response)
        self._depth = int(concept_calls_max_depth)

    def generate(  # noqa: ANN201
        self,
        *,
        prompt: str,
        max_new_tokens: int,  # noqa: ARG002
        mode: str,  # noqa: ARG002
        dialogue_id: int,  # noqa: ARG002
        turn: int,  # noqa: ARG002
        plan_trace: Optional[Dict[str, Any]] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401,ARG002
    ) -> Dict[str, Any]:
        tr: Dict[str, Any] = {}
        tr["concept_executor"] = {
            "used": True,
            "ok": True,
            "concept_calls_total": 2 if self._depth >= 1 else 1,
            "concept_calls_max_depth": int(self._depth),
        }
        return {
            "text": str(prompt) + str(self._response),
            "trace": tr,
            "mode": "default",
            "mode_source": "router",
            "mode_policy_action": "",
            "policy_coverage": 0.0,
            "user_sig": "",
        }


class _FakePlanEngineCsg:
    def __init__(self, *, store: ActStore, response: str, concept_calls_max_depth: int, concept_id: str) -> None:
        self.store = store
        self._response = str(response)
        self._depth = int(concept_calls_max_depth)
        self._concept_id = str(concept_id)

    def generate(  # noqa: ANN201
        self,
        *,
        prompt: str,
        max_new_tokens: int,  # noqa: ARG002
        mode: str,  # noqa: ARG002
        dialogue_id: int,  # noqa: ARG002
        turn: int,  # noqa: ARG002
        plan_trace: Optional[Dict[str, Any]] = None,  # noqa: ARG002
        **kwargs: Any,  # noqa: ANN401,ARG002
    ) -> Dict[str, Any]:
        tr: Dict[str, Any] = {}
        tr["concept_executor"] = {
            "used": True,
            "ok": True,
            "concept_id": str(self._concept_id),
            "concept_calls_total": 2 if self._depth >= 1 else 1,
            "concept_calls_max_depth": int(self._depth),
        }
        return {
            "text": str(prompt) + str(self._response),
            "trace": tr,
            "mode": "default",
            "mode_source": "router",
            "mode_policy_action": "",
            "policy_coverage": 0.0,
            "user_sig": "",
        }


class ConceptDepthRequirementTests(unittest.TestCase):
    def test_skill_suite_plan_validator_enforces_min_depth_when_set(self) -> None:
        expected_spec = _plan_expected_spec()
        tasks = (
            {
                "task_id": "plan_task",
                "dialogue": ("Objetivo: retornar JSON canônico com goal_id, plan, a, b.",),
                "validator_id": "plan_validator",
                "expected_spec": dict(expected_spec),
                "concept_min_depth": 1,
                "tags": ["plan", "state", "json"],
            },
        )

        # Correct output + concept ok, but no nested calls => fail (depth too shallow).
        engine = _FakePlanEngine(response=str(expected_spec["expected_output_text"]), concept_calls_max_depth=0)
        _t, m = run_skill_suite(engine, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m.get("pass_count") or 0), 0)
        failures = m.get("failures") or []
        self.assertTrue(any(isinstance(f, dict) and f.get("reason") == "concept_depth_too_shallow" for f in failures))

        # With nested calls (depth>=1), the same output can pass.
        engine2 = _FakePlanEngine(response=str(expected_spec["expected_output_text"]), concept_calls_max_depth=1)
        _t2, m2 = run_skill_suite(engine2, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m2.get("pass_count") or 0), 1)
        self.assertAlmostEqual(float(m2.get("concept_pass_rate") or 0.0), 1.0, places=9)

    def test_skill_suite_plan_validator_enforces_min_depth_2(self) -> None:
        expected_spec = _plan_expected_spec()
        tasks = (
            {
                "task_id": "plan_task_depth2",
                "dialogue": ("Objetivo: retornar JSON canônico com goal_id, plan, a, b.",),
                "validator_id": "plan_validator",
                "expected_spec": dict(expected_spec),
                "concept_min_depth": 2,
                "tags": ["plan", "state", "json"],
            },
        )

        # Depth 1 is not enough for min_depth=2.
        engine = _FakePlanEngine(response=str(expected_spec["expected_output_text"]), concept_calls_max_depth=1)
        _t, m = run_skill_suite(engine, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m.get("pass_count") or 0), 0)
        failures = m.get("failures") or []
        self.assertTrue(any(isinstance(f, dict) and f.get("reason") == "concept_depth_too_shallow" for f in failures))

        # Depth 2 can pass.
        engine2 = _FakePlanEngine(response=str(expected_spec["expected_output_text"]), concept_calls_max_depth=2)
        _t2, m2 = run_skill_suite(engine2, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m2.get("pass_count") or 0), 1)
        self.assertAlmostEqual(float(m2.get("concept_pass_rate") or 0.0), 1.0, places=9)

    def test_engine_generate_prefers_composed_concept_when_min_depth_required(self) -> None:
        expected_spec = _plan_expected_spec()

        store = ActStore()
        inner = Act(
            id="concept_inner_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "in0"}),
                Instruction("CSV_GET_INPUT", {"name": "plan", "out": "in1"}),
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "in2"}),
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "in3"}),
                Instruction(
                    "CSV_PRIMITIVE",
                    {"fn": "make_dict_goal_plan_ab", "in": ["in0", "in1", "in2", "in3"], "out": "v0"},
                ),
                Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["v0"], "out": "v1"}),
                Instruction("CSV_RETURN", {"var": "v1"}),
            ],
            evidence={
                "name": "concept_test_inner",
                "interface": {
                    "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        outer = Act(
            id="concept_outer_composed_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "goal_id"}),
                Instruction("CSV_GET_INPUT", {"name": "plan", "out": "plan"}),
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
                Instruction(
                    "CSV_CALL",
                    {
                        "concept_id": str(inner.id),
                        "bind": {"goal_id": "goal_id", "plan": "plan", "a": "a", "b": "b"},
                        "out": "out",
                    },
                ),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            evidence={
                "name": "concept_test_outer",
                "interface": {
                    "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
            },
            cost={"overhead_bits": 0},
            deps=[str(inner.id)],
            active=True,
        )
        store.add(inner)
        store.add(outer)

        engine = Engine(store, seed=0, config=EngineConfig(enable_contracts=False))
        plan_trace = {
            "validator_id": "plan_validator",
            "expected_format": "plan",
            "constraints": ["plan_validator"],
            "expected_spec": dict(expected_spec),
            "concept_min_depth": 1,
        }
        out = engine.generate(
            prompt="X",
            max_new_tokens=64,
            mode="greedy",
            dialogue_id=0,
            turn=0,
            plan_trace=plan_trace,
        )
        resp = out["text"][len("X") :]
        self.assertEqual(resp, str(expected_spec["expected_output_text"]))
        tr = out.get("trace") or {}
        cm = tr.get("concept_executor") if isinstance(tr.get("concept_executor"), dict) else {}
        self.assertEqual(str(cm.get("concept_id") or ""), str(outer.id))
        self.assertGreaterEqual(int(cm.get("concept_calls_max_depth") or 0), 1)

    def test_skill_suite_plan_validator_enforces_min_csg_complexity_when_set(self) -> None:
        expected_spec = _plan_expected_spec()
        tasks = (
            {
                "task_id": "plan_task_csg_complexity",
                "dialogue": ("Objetivo: retornar JSON canônico com goal_id, plan, a, b.",),
                "validator_id": "plan_validator",
                "expected_spec": dict(expected_spec),
                "concept_min_depth": 1,
                "concept_csg_min_nodes": 2,
                "concept_csg_min_edges": 1,
                "tags": ["plan", "state", "json"],
            },
        )

        store = ActStore()
        shallow = Act(
            id="concept_shallow_csg_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "goal_id"}),
                Instruction(
                    "CSV_CALL",
                    {"concept_id": "concept_leaf_v0", "bind": {"goal_id": "goal_id"}, "out": "v0"},
                ),
                Instruction("CSV_RETURN", {"var": "v0"}),
            ],
            evidence={
                "name": "concept_test_shallow",
                "interface": {
                    "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        rich = Act(
            id="concept_rich_csg_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "goal_id"}),
                Instruction("CSV_GET_INPUT", {"name": "plan", "out": "plan"}),
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
                Instruction(
                    "CSV_CALL",
                    {
                        "concept_id": "concept_leaf_v0",
                        "bind": {"goal_id": "goal_id", "plan": "plan", "a": "a", "b": "b"},
                        "out": "v0",
                    },
                ),
                Instruction(
                    "CSV_CALL",
                    {
                        "concept_id": "concept_leaf_v0",
                        "bind": {"obj": "v0"},
                        "out": "v1",
                    },
                ),
                Instruction("CSV_RETURN", {"var": "v1"}),
            ],
            evidence={
                "name": "concept_test_rich",
                "interface": {
                    "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        store.add(shallow)
        store.add(rich)

        # Correct output + concept ok, but CSG too shallow (nodes=1, edges=0) => fail.
        engine = _FakePlanEngineCsg(
            store=store,
            response=str(expected_spec["expected_output_text"]),
            concept_calls_max_depth=1,
            concept_id=str(shallow.id),
        )
        _t, m = run_skill_suite(engine, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m.get("pass_count") or 0), 0)
        failures = m.get("failures") or []
        self.assertTrue(any(isinstance(f, dict) and f.get("reason") == "concept_csg_too_shallow" for f in failures))

        # With rich CSG (nodes>=2, edges>=1) the same output can pass.
        engine2 = _FakePlanEngineCsg(
            store=store,
            response=str(expected_spec["expected_output_text"]),
            concept_calls_max_depth=1,
            concept_id=str(rich.id),
        )
        _t2, m2 = run_skill_suite(engine2, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m2.get("pass_count") or 0), 1)
        self.assertAlmostEqual(float(m2.get("concept_csg_required_pass_rate") or 0.0), 1.0, places=9)

    def test_engine_generate_prefers_csg_rich_concept_when_required(self) -> None:
        expected_spec = _plan_expected_spec()

        store = ActStore()
        op_make_dict = Act(
            id="concept_plan_op_make_dict_goal_plan_ab_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "goal_id"}),
                Instruction("CSV_GET_INPUT", {"name": "plan", "out": "plan"}),
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
                Instruction(
                    "CSV_PRIMITIVE",
                    {"fn": "make_dict_goal_plan_ab", "in": ["goal_id", "plan", "a", "b"], "out": "value_out"},
                ),
                Instruction("CSV_RETURN", {"var": "value_out"}),
            ],
            evidence={
                "name": "concept_test_plan_op_make_dict",
                "interface": {
                    "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                    "output_schema": {"value_out": "dict"},
                    "validator_id": "",
                },
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        op_json = Act(
            id="concept_plan_op_json_canonical_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "obj", "out": "obj"}),
                Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["obj"], "out": "value_out"}),
                Instruction("CSV_RETURN", {"var": "value_out"}),
            ],
            evidence={
                "name": "concept_test_plan_op_json_canonical",
                "interface": {"input_schema": {"obj": "dict"}, "output_schema": {"value_out": "str"}, "validator_id": ""},
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        shallow_leaf = Act(
            id="concept_plan_validator_leaf_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "goal_id"}),
                Instruction("CSV_GET_INPUT", {"name": "plan", "out": "plan"}),
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
                Instruction(
                    "CSV_PRIMITIVE",
                    {"fn": "make_dict_goal_plan_ab", "in": ["goal_id", "plan", "a", "b"], "out": "d"},
                ),
                Instruction("CSV_PRIMITIVE", {"fn": "json_canonical", "in": ["d"], "out": "j"}),
                Instruction("CSV_RETURN", {"var": "j"}),
            ],
            evidence={
                "name": "concept_test_plan_validator_leaf",
                "interface": {
                    "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        rich_csg = Act(
            id="concept_plan_validator_csg_rich_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "goal_id", "out": "goal_id"}),
                Instruction("CSV_GET_INPUT", {"name": "plan", "out": "plan"}),
                Instruction("CSV_GET_INPUT", {"name": "a", "out": "a"}),
                Instruction("CSV_GET_INPUT", {"name": "b", "out": "b"}),
                Instruction(
                    "CSV_CALL",
                    {
                        "concept_id": str(op_make_dict.id),
                        "bind": {"goal_id": "goal_id", "plan": "plan", "a": "a", "b": "b"},
                        "out": "d",
                    },
                ),
                Instruction(
                    "CSV_CALL",
                    {"concept_id": str(op_json.id), "bind": {"obj": "d"}, "out": "j"},
                ),
                Instruction("CSV_RETURN", {"var": "j"}),
            ],
            evidence={
                "name": "concept_test_plan_validator_csg_rich",
                "interface": {
                    "input_schema": {"goal_id": "str", "plan": "str", "a": "int", "b": "int"},
                    "output_schema": {"value": "str"},
                    "validator_id": "plan_validator",
                },
            },
            cost={"overhead_bits": 0},
            deps=[str(op_make_dict.id), str(op_json.id)],
            active=True,
        )

        store.add(op_make_dict)
        store.add(op_json)
        store.add(shallow_leaf)
        store.add(rich_csg)

        engine = Engine(store, seed=0, config=EngineConfig(enable_contracts=False))
        plan_trace = {
            "validator_id": "plan_validator",
            "expected_format": "plan",
            "constraints": ["plan_validator"],
            "expected_spec": dict(expected_spec),
            "concept_min_depth": 1,
            "concept_csg_min_nodes": 2,
            "concept_csg_min_edges": 1,
        }
        out = engine.generate(
            prompt="X",
            max_new_tokens=64,
            mode="greedy",
            dialogue_id=0,
            turn=0,
            plan_trace=plan_trace,
        )
        resp = out["text"][len("X") :]
        self.assertEqual(resp, str(expected_spec["expected_output_text"]))
        tr = out.get("trace") or {}
        cm = tr.get("concept_executor") if isinstance(tr.get("concept_executor"), dict) else {}
        self.assertEqual(str(cm.get("concept_id") or ""), str(rich_csg.id))
        self.assertTrue(bool(cm.get("selection_csg_met", False)))


if __name__ == "__main__":
    unittest.main()
