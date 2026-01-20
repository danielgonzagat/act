from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .act import Act, Instruction, deterministic_iso


def make_action_concept_v90(
    *,
    act_id: str,
    input_schema: Dict[str, str],
    output_schema: Dict[str, str],
    program: List[Instruction],
    supports_goals_v89: Sequence[Dict[str, Any]],
    created_step: int = 0,
) -> Act:
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=int(created_step)),
        kind="concept_csv",
        match={},
        program=list(program),
        evidence={
            "interface": {
                "input_schema": dict(input_schema),
                "output_schema": dict(output_schema),
                "validator_id": "",
            },
            "supports_goals_v89": [dict(x) for x in supports_goals_v89 if isinstance(x, dict)],
            "action_v90": {"schema_version": 1, "action_id": str(act_id)},
        },
        cost={},
        deps=[],
        active=True,
    )


def action_concepts_for_dsl_v90(*, goal_ids: Dict[str, str]) -> List[Act]:
    """
    Deterministic set of language-as-action concepts for the V90 DSL harness.
    goal_ids maps COMM_* -> goal_id string used in supports(G).
    """
    respond_gid = str(goal_ids.get("COMM_RESPOND") or "COMM_RESPOND")
    ask_gid = str(goal_ids.get("COMM_ASK_CLARIFY") or "COMM_ASK_CLARIFY")
    confirm_gid = str(goal_ids.get("COMM_CONFIRM") or "COMM_CONFIRM")
    correct_gid = str(goal_ids.get("COMM_CORRECT") or "COMM_CORRECT")
    summarize_gid = str(goal_ids.get("COMM_SUMMARIZE") or "COMM_SUMMARIZE")
    admit_gid = str(goal_ids.get("COMM_ADMIT_UNKNOWN") or "COMM_ADMIT_UNKNOWN")
    end_gid = str(goal_ids.get("COMM_END") or "COMM_END")

    def _support(goal_id: str, *, prior_success: float, prior_cost: float, note: str = "") -> Dict[str, Any]:
        return {
            "goal_id": str(goal_id),
            "prior_success": float(prior_success),
            "prior_strength": 2,
            "prior_cost": float(prior_cost),
            "note": str(note),
        }

    acts: List[Act] = []

    # EmitTextCSV_V90: out=text
    acts.append(
        make_action_concept_v90(
            act_id="concept_v90_emit_text_v0",
            input_schema={"text": "str"},
            output_schema={"out": "str"},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "text", "out": "text"}),
                Instruction("CSV_RETURN", {"var": "text"}),
            ],
            supports_goals_v89=[
                _support(respond_gid, prior_success=0.9, prior_cost=2.0, note="emit"),
                _support(summarize_gid, prior_success=0.9, prior_cost=2.0, note="emit"),
                _support(confirm_gid, prior_success=0.9, prior_cost=2.0, note="emit"),
            ],
        )
    )

    # Confirm SET: "OK: " + k + "=" + v
    acts.append(
        make_action_concept_v90(
            act_id="concept_v90_confirm_set_v0",
            input_schema={"k": "str", "v": "str"},
            output_schema={"out": "str"},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "k", "out": "k"}),
                Instruction("CSV_GET_INPUT", {"name": "v", "out": "v"}),
                Instruction("CSV_CONST", {"value": "OK: ", "out": "p"}),
                Instruction("CSV_CONST", {"value": "=", "out": "eq"}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["p", "k"], "out": "t0"}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["t0", "eq"], "out": "t1"}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["t1", "v"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            supports_goals_v89=[_support(respond_gid, prior_success=0.9, prior_cost=8.0, note="set")],
        )
    )

    # Emit SUM: "SUM=" + sum
    acts.append(
        make_action_concept_v90(
            act_id="concept_v90_emit_sum_v0",
            input_schema={"sum": "str"},
            output_schema={"out": "str"},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "sum", "out": "sum"}),
                Instruction("CSV_CONST", {"value": "SUM=", "out": "p"}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["p", "sum"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            supports_goals_v89=[_support(respond_gid, prior_success=0.9, prior_cost=4.0, note="sum")],
        )
    )

    # AskClarification: "Qual é o valor de " + k + "?"
    acts.append(
        make_action_concept_v90(
            act_id="concept_v90_ask_clarify_v0",
            input_schema={"k": "str"},
            output_schema={"out": "str"},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "k", "out": "k"}),
                Instruction("CSV_CONST", {"value": "Qual é o valor de ", "out": "p"}),
                Instruction("CSV_CONST", {"value": "?", "out": "q"}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["p", "k"], "out": "t0"}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["t0", "q"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            supports_goals_v89=[_support(ask_gid, prior_success=0.9, prior_cost=6.0, note="clarify")],
        )
    )

    # CorrectUser: "Comando inválido: " + msg
    acts.append(
        make_action_concept_v90(
            act_id="concept_v90_correct_user_v0",
            input_schema={"msg": "str"},
            output_schema={"out": "str"},
            program=[
                Instruction("CSV_GET_INPUT", {"name": "msg", "out": "msg"}),
                Instruction("CSV_CONST", {"value": "Comando inválido: ", "out": "p"}),
                Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["p", "msg"], "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            supports_goals_v89=[_support(correct_gid, prior_success=0.9, prior_cost=4.0, note="correct")],
        )
    )

    # AdmitUnknown: "Não sei."
    acts.append(
        make_action_concept_v90(
            act_id="concept_v90_admit_unknown_v0",
            input_schema={},
            output_schema={"out": "str"},
            program=[
                Instruction("CSV_CONST", {"value": "Não sei.", "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            supports_goals_v89=[_support(admit_gid, prior_success=0.9, prior_cost=2.0, note="unknown")],
        )
    )

    # EndConversation: "Encerrado."
    acts.append(
        make_action_concept_v90(
            act_id="concept_v90_end_conversation_v0",
            input_schema={},
            output_schema={"out": "str"},
            program=[
                Instruction("CSV_CONST", {"value": "Encerrado.", "out": "out"}),
                Instruction("CSV_RETURN", {"var": "out"}),
            ],
            supports_goals_v89=[_support(end_gid, prior_success=0.9, prior_cost=2.0, note="end")],
        )
    )

    return list(acts)

