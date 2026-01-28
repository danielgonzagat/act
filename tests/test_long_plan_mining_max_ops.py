import json
import os
import tempfile

from atos_core.csv_miner import mine_csv_candidates
from atos_core.suite import _v69_long_sum_task


def test_mine_csv_candidates_can_extract_full_long_plan_segment() -> None:
    task = _v69_long_sum_task(
        task_id="v69_long_sum_test_25",
        values=list(range(1, 26)),
        b=0,
        plan="long_sum_chain_test_25",
        concept_min_depth=2,
    )
    expected_spec = task["expected_spec"]
    inputs = expected_spec["inputs"]
    input_keys = expected_spec["input_keys"]
    ops = expected_spec["ops"]
    ret = expected_spec["return_var"]

    # Synthetic engine-style CSV exec trace row (v61 format): t:"INS" with op:"CSV_*".
    events = []
    for idx, key in enumerate(input_keys):
        events.append(
            {
                "t": "INS",
                "op": "CSV_GET_INPUT",
                "name": str(key),
                "out": f"in{idx}",
            }
        )
    for op in ops:
        events.append(
            {
                "t": "INS",
                "op": "CSV_PRIMITIVE",
                "fn": str(op["fn"]),
                "in": list(op["in"]),
                "out": str(op["out"]),
            }
        )
    events.append({"t": "INS", "op": "CSV_RETURN", "var": str(ret)})

    row = {
        "ctx_sig": "plan␟v69_long_sum_test_25",
        "inputs": dict(inputs),
        "utility_passed": True,
        "events": events,
    }

    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "trace.jsonl")
        with open(p, "w", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
            f.write("\n")

        cands = mine_csv_candidates(
            p,
            min_ops=len(ops),
            max_ops=len(ops),
            bits_per_op=128,
            overhead_bits=1024,
            max_examples_per_candidate=1,
        )

    assert cands, "expected at least one candidate"
    cand = cands[0]
    assert cand.validator_id == "plan_validator"
    # Full plan must depend on all original inputs.
    assert set(cand.input_schema.keys()) == set(inputs.keys())
    assert len(cand.ops) == len(ops)

