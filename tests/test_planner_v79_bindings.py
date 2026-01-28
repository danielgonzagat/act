from __future__ import annotations

from atos_core.act import Act, Instruction, deterministic_iso
from atos_core.goal_spec_v72 import GoalSpecV72
from atos_core.planner_v79 import PlannerV79
from atos_core.store import ActStore


def _concept_act(*, act_id: str, input_schema: dict, output_schema: dict, validator_id: str = "text_exact") -> Act:
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match={},
        program=[Instruction("CSV_RETURN", {"var": "value"})],
        evidence={
            "name": "test_concept",
            "interface": {
                "input_schema": dict(input_schema),
                "output_schema": dict(output_schema),
                "validator_id": str(validator_id),
            },
        },
        cost={},
        deps=[],
        active=True,
    )


def test_planner_v79_supports_bind_maps_and_output_aliasing() -> None:
    store = ActStore()
    store.add(
        _concept_act(
            act_id="concept_make_dict_ab",
            input_schema={"a": "int", "b": "int"},
            output_schema={"dict": "dict"},
            validator_id="json_obj_exact",
        )
    )
    store.add(
        _concept_act(
            act_id="concept_json_canonical",
            input_schema={"obj": "dict"},
            output_schema={"json": "str"},
            validator_id="plan_validator",
        )
    )

    goal = GoalSpecV72(
        goal_kind="test_planner_v79",
        bindings={"x": 1, "y": 2},
        output_key="answer",
        expected={"text": "{}"},
        validator_id="plan_validator",
        created_step=0,
    )
    planner = PlannerV79(max_depth=4, max_expansions=256)
    plan, dbg = planner.plan(goal_spec=goal, store=store)
    assert plan is not None, dbg
    assert len(plan.steps) == 2, plan.to_dict()

    s0 = plan.steps[0]
    assert s0.concept_id == "concept_make_dict_ab"
    assert set(s0.bind_map.keys()) == {"a", "b"}
    assert set(s0.bind_map.values()) == {"x", "y"}
    assert s0.produces != goal.output_key

    s1 = plan.steps[1]
    assert s1.concept_id == "concept_json_canonical"
    assert s1.bind_map == {"obj": s0.produces}
    assert s1.produces == goal.output_key

