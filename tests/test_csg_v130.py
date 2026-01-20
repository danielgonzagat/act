from __future__ import annotations

import json
import os
import tempfile
import unittest
from typing import Any, Dict, List

from atos_core.act import Act, Instruction, canonical_json_dumps, deterministic_iso
from atos_core.csg_v130 import (
    append_chained_jsonl_v130,
    canonicalize_csg_v130,
    csg_expand_v130,
    csg_hash_v130,
    csg_to_concept_program_v130,
    verify_chained_jsonl_v130,
)
from atos_core.engine_v80 import EngineV80
from atos_core.store import ActStore


def _make_concept_act(*, act_id: str, program: List[Instruction], match: Dict[str, Any]) -> Act:
    return Act(
        id=str(act_id),
        version=1,
        created_at=deterministic_iso(step=0),
        kind="concept_csv",
        match=dict(match),
        program=list(program),
        evidence={"interface": {"input_schema": {}, "output_schema": {}, "validator_id": "text_exact"}},
        cost={},
        deps=[],
        active=True,
    )


def _execute_inline_replay(
    *, engine: EngineV80, steps: List[Dict[str, Any]], inputs: Dict[str, Any], goal_kind: str
) -> Dict[str, Any]:
    env: Dict[str, Any] = dict(inputs)
    for st in steps:
        act_id = str(st.get("act_id") or "")
        bind = st.get("bind") if isinstance(st.get("bind"), dict) else {}
        produces = str(st.get("produces") or "")
        sub_inputs: Dict[str, Any] = {}
        for slot, var in bind.items():
            sub_inputs[str(slot)] = env.get(str(var))
        res = engine.execute_concept_csv(
            concept_act_id=str(act_id),
            inputs=sub_inputs,
            goal_kind=str(goal_kind),
            expected=None,
            step=0,
            max_depth=6,
            max_events=256,
            validate_output=False,
        )
        meta = res.get("meta") if isinstance(res.get("meta"), dict) else {}
        if not bool(meta.get("ok", False)):
            return {"ok": False, "reason": "inline_step_failed", "meta": dict(meta)}
        env[produces] = res.get("output")
    last_out = steps[-1].get("produces") if steps else ""
    return {"ok": True, "output": env.get(str(last_out))}


class TestCsgV130(unittest.TestCase):
    def test_canonicalize_and_hash_stable(self) -> None:
        # Same logical CSG with different node order and bind order must hash the same.
        csg1 = {
            "schema_version": 1,
            "nodes": [
                {"act_id": "c", "bind": {"nx": "nx", "ny": "ny"}, "produces": "sum"},
                {"act_id": "a", "bind": {"x": "x"}, "produces": "nx"},
                {"act_id": "b", "bind": {"y": "y"}, "produces": "ny"},
            ],
            "interface": {"match": {"goal_kinds": ["k2", "k1", "k1"]}},
        }
        csg2 = {
            "schema_version": 1,
            "nodes": [
                {"act_id": "b", "bind": {"y": "y"}, "produces": "ny"},
                {"act_id": "c", "bind": {"ny": "ny", "nx": "nx"}, "produces": "sum"},
                {"act_id": "a", "bind": {"x": "x"}, "produces": "nx"},
            ],
            "interface": {"match": {"goal_kinds": ["k1", "k2"]}},
        }
        h1 = csg_hash_v130(csg1)
        h2 = csg_hash_v130(csg2)
        self.assertEqual(h1, h2)

        canon = canonicalize_csg_v130(csg1)
        iface = canon.get("interface")
        self.assertIsInstance(iface, dict)
        match = iface.get("match")
        self.assertIsInstance(match, dict)
        self.assertEqual(match.get("goal_kinds"), ["k1", "k2"])

    def test_hash_chained_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "x.jsonl")
            prev = None
            prev = append_chained_jsonl_v130(path, {"a": 1}, prev_hash=prev)
            prev = append_chained_jsonl_v130(path, {"b": 2}, prev_hash=prev)
            self.assertTrue(verify_chained_jsonl_v130(path))

            # Tamper: change payload without recomputing entry_hash.
            rows = []
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    rows.append(json.loads(line))
            rows[1]["b"] = 999
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(canonical_json_dumps(row))
                    f.write("\n")
            self.assertFalse(verify_chained_jsonl_v130(path))

    def test_expand_and_replay_equivalence(self) -> None:
        store = ActStore()
        goal_kind = "v130_test"

        a_id = "concept_v130_a_v0"
        b_id = "concept_v130_b_v0"
        c_id = "concept_v130_c_v0"

        store.add(
            _make_concept_act(
                act_id=a_id,
                match={"goal_kinds": [goal_kind]},
                program=[
                    Instruction("CSV_GET_INPUT", {"name": "x", "out": "x"}),
                    Instruction("CSV_RETURN", {"var": "x"}),
                ],
            )
        )
        store.add(
            _make_concept_act(
                act_id=b_id,
                match={"goal_kinds": [goal_kind]},
                program=[
                    Instruction("CSV_GET_INPUT", {"name": "y", "out": "y"}),
                    Instruction("CSV_RETURN", {"var": "y"}),
                ],
            )
        )
        store.add(
            _make_concept_act(
                act_id=c_id,
                match={"goal_kinds": [goal_kind]},
                program=[
                    Instruction("CSV_GET_INPUT", {"name": "nx", "out": "nx"}),
                    Instruction("CSV_GET_INPUT", {"name": "ny", "out": "ny"}),
                    Instruction("CSV_PRIMITIVE", {"fn": "str_concat", "in": ["nx", "ny"], "out": "out"}),
                    Instruction("CSV_RETURN", {"var": "out"}),
                ],
            )
        )

        csg = {
            "schema_version": 1,
            "nodes": [
                {"act_id": a_id, "bind": {"x": "x"}, "produces": "nx"},
                {"act_id": b_id, "bind": {"y": "y"}, "produces": "ny"},
                {"act_id": c_id, "bind": {"nx": "nx", "ny": "ny"}, "produces": "out"},
            ],
            "interface": {"inputs": ["x", "y"], "outputs": ["out"], "match": {"goal_kinds": [goal_kind]}},
        }
        canon = canonicalize_csg_v130(csg)
        prog = csg_to_concept_program_v130(canon)
        composed_id = "concept_v130_composed_v0"
        store.add(_make_concept_act(act_id=composed_id, match={"goal_kinds": [goal_kind]}, program=prog))

        engine = EngineV80(store, seed=0)
        inputs = {"x": "AA", "y": "BB"}
        direct = engine.execute_concept_csv(
            concept_act_id=composed_id,
            inputs=dict(inputs),
            goal_kind=goal_kind,
            expected=None,
            step=0,
            max_depth=6,
            max_events=256,
            validate_output=False,
        )
        meta = direct.get("meta") if isinstance(direct.get("meta"), dict) else {}
        self.assertTrue(bool(meta.get("ok", False)))
        out_direct = direct.get("output")

        steps = csg_expand_v130(canon, store)
        replay = _execute_inline_replay(engine=engine, steps=steps, inputs=dict(inputs), goal_kind=goal_kind)
        self.assertTrue(bool(replay.get("ok", False)))
        self.assertEqual(out_direct, replay.get("output"))


if __name__ == "__main__":
    unittest.main()
