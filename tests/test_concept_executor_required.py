from __future__ import annotations

import tempfile
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
    def __init__(self, *, response: str, concept_used: bool, concept_ok: bool) -> None:
        self._response = str(response)
        self._concept_used = bool(concept_used)
        self._concept_ok = bool(concept_ok)

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
        tr["concept_executor"] = {"used": bool(self._concept_used), "ok": bool(self._concept_ok)}
        return {
            "text": str(prompt) + str(self._response),
            "trace": tr,
            "mode": "default",
            "mode_source": "router",
            "mode_policy_action": "",
            "policy_coverage": 0.0,
            "user_sig": "",
        }


class ConceptExecutorRequiredTests(unittest.TestCase):
    def test_skill_suite_plan_validator_requires_concept_executor(self) -> None:
        expected_spec = _plan_expected_spec()
        tasks = (
            {
                "task_id": "plan_task",
                "dialogue": ("Objetivo: retornar JSON canônico com goal_id, plan, a, b.",),
                "validator_id": "plan_validator",
                "expected_spec": dict(expected_spec),
                "tags": ["plan", "state", "json"],
            },
        )

        # Even if the output text is perfect, missing concept usage must fail.
        engine = _FakePlanEngine(response=str(expected_spec["expected_output_text"]), concept_used=False, concept_ok=False)
        _t, m = run_skill_suite(engine, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m.get("pass_count") or 0), 0)
        self.assertEqual(int(m.get("concept_total") or 0), 1)
        self.assertAlmostEqual(float(m.get("concept_pass_rate") or 0.0), 0.0, places=9)
        failures = m.get("failures") or []
        self.assertTrue(any(isinstance(f, dict) and f.get("reason") == "missing_concept_executor" for f in failures))

        # With concept usage, the same output can pass.
        engine2 = _FakePlanEngine(response=str(expected_spec["expected_output_text"]), concept_used=True, concept_ok=True)
        _t2, m2 = run_skill_suite(engine2, tasks=tasks, max_new_tokens=64)
        self.assertEqual(int(m2.get("pass_count") or 0), 1)
        self.assertAlmostEqual(float(m2.get("concept_pass_rate") or 0.0), 1.0, places=9)

    def test_engine_generate_executes_matching_plan_concept(self) -> None:
        expected_spec = _plan_expected_spec()

        store = ActStore()
        concept = Act(
            id="concept_plan_wrap_v0",
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
                "name": "concept_test_v0",
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
        store.add(concept)

        engine = Engine(store, seed=0, config=EngineConfig(enable_contracts=False))
        plan_trace = {
            "validator_id": "plan_validator",
            "expected_format": "plan",
            "constraints": ["plan_validator"],
            "expected_spec": dict(expected_spec),
        }
        out = engine.generate(prompt="X", max_new_tokens=64, mode="greedy", dialogue_id=0, turn=0, plan_trace=plan_trace)
        resp = out["text"][len("X") :]
        self.assertEqual(resp, str(expected_spec["expected_output_text"]))
        tr = out.get("trace") or {}
        cm = tr.get("concept_executor") if isinstance(tr.get("concept_executor"), dict) else {}
        self.assertTrue(bool(cm.get("used", False)))
        self.assertTrue(bool(cm.get("ok", False)))
        self.assertEqual(str(cm.get("concept_id") or ""), "concept_plan_wrap_v0")

    def test_skill_suite_generic_concept_expected_spec_requires_concept_executor(self) -> None:
        tasks = (
            {
                "task_id": "mem_task",
                "dialogue": ("Memória: armazene o segredo. Responda exatamente: OK",),
                "validator_id": "exact_match",
                "expected_spec": {"text": "OK", "collapse_ws": True, "case_sensitive": True},
                "concept_expected_spec": {
                    "inputs": {"key": "k", "value": "V"},
                    "input_keys": ["key", "value"],
                    "expected_for_validator": "OK",
                },
                "concept_validator_id": "text_exact",
                "tags": ["memory"],
            },
        )

        # Even if the output text is perfect, missing concept usage must fail.
        engine = _FakePlanEngine(response="OK", concept_used=False, concept_ok=False)
        _t, m = run_skill_suite(engine, tasks=tasks, max_new_tokens=16)
        self.assertEqual(int(m.get("pass_count") or 0), 0)
        self.assertEqual(int(m.get("concept_total") or 0), 1)
        failures = m.get("failures") or []
        self.assertTrue(
            any(isinstance(f, dict) and f.get("reason") == "missing_concept_executor" for f in failures)
        )

        # With concept usage, the same output can pass.
        engine2 = _FakePlanEngine(response="OK", concept_used=True, concept_ok=True)
        _t2, m2 = run_skill_suite(engine2, tasks=tasks, max_new_tokens=16)
        self.assertEqual(int(m2.get("pass_count") or 0), 1)
        self.assertAlmostEqual(float(m2.get("concept_pass_rate") or 0.0), 1.0, places=9)

    def test_engine_generate_executes_generic_concept_expected_spec(self) -> None:
        store = ActStore()
        store.add(
            Act(
                id="fact_memory_v0",
                version=1,
                created_at=deterministic_iso(step=0),
                kind="memory_facts",
                match={"type": "always"},
                program=[],
                evidence={"name": "fact_memory_v0", "enabled": True, "table": {}},
                cost={"overhead_bits": 0},
                deps=[],
                active=True,
            )
        )
        concept_set = Act(
            id="concept_fact_set_test_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "key", "out": "key"}),
                Instruction("CSV_GET_INPUT", {"name": "value", "out": "value"}),
                Instruction("CSV_FACT_SET", {"key_var": "key", "value_var": "value"}),
                Instruction("CSV_CONST", {"value": "OK", "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            evidence={
                "name": "concept_test_v0",
                "interface": {
                    "input_schema": {"key": "str", "value": "str"},
                    "output_schema": {"out": "str"},
                    "validator_id": "text_exact",
                },
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        concept_get = Act(
            id="concept_fact_get_test_v0",
            version=1,
            created_at=deterministic_iso(step=0),
            kind="concept_csv",
            match={},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "key", "out": "key"}),
                Instruction("CSV_FACT_GET", {"key_var": "key", "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            evidence={
                "name": "concept_test_v0",
                "interface": {
                    "input_schema": {"key": "str"},
                    "output_schema": {"out": "str"},
                    "validator_id": "text_exact",
                },
            },
            cost={"overhead_bits": 0},
            deps=[],
            active=True,
        )
        store.add(concept_set)
        store.add(concept_get)

        engine = Engine(store, seed=0, config=EngineConfig(enable_contracts=False))

        plan_trace_set = {
            "concept_validator_id": "text_exact",
            "concept_expected_spec": {
                "inputs": {"key": "k", "value": "DELTA8"},
                "input_keys": ["key", "value"],
                "expected_for_validator": "OK",
            },
        }
        out1 = engine.generate(prompt="X", max_new_tokens=16, mode="greedy", dialogue_id=0, turn=0, plan_trace=plan_trace_set)
        self.assertEqual(out1["text"][len("X") :], "OK")

        plan_trace_get = {
            "concept_validator_id": "text_exact",
            "concept_expected_spec": {
                "inputs": {"key": "k"},
                "input_keys": ["key"],
                "expected_for_validator": "DELTA8",
            },
        }
        out2 = engine.generate(prompt="Y", max_new_tokens=16, mode="greedy", dialogue_id=0, turn=1, plan_trace=plan_trace_get)
        self.assertEqual(out2["text"][len("Y") :], "DELTA8")


if __name__ == "__main__":
    unittest.main()
