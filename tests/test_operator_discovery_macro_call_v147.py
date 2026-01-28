from __future__ import annotations

import importlib.util
from pathlib import Path

from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141
from atos_core.grid_v124 import grid_from_list_v124


def _load_miner_module():
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "arc_mine_operator_templates_v147.py"
    spec = importlib.util.spec_from_file_location("arc_mine_operator_templates_v147", str(path))
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_operator_discovery_enables_macro_call_under_tight_depth() -> None:
    # Construct a tiny task whose natural solution is:
    #   bbox_by_color -> crop_bbox -> commit_patch
    # but we run with max_depth=1 so it cannot be solved without a macro/operator closure.
    train_in = grid_from_list_v124(
        [
            [2, 2, 2, 3],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 2, 2, 2],
        ]
    )
    train_out = grid_from_list_v124([[1, 1], [1, 1]])
    test_in = grid_from_list_v124(
        [
            [2, 2, 2, 3],
            [2, 1, 1, 2],
            [2, 1, 1, 2],
            [2, 2, 2, 2],
        ]
    )

    # Base solve (no macros) must fail under depth=1, but still emit trace_programs
    # containing partial programs like bbox_by_color.
    cfg0 = SolveConfigV141(
        max_depth=1,
        max_programs=200,
        trace_program_limit=80,
        max_ambiguous_outputs=8,
        enable_reachability_pruning=False,  # allow trace exploration under tight caps
    )
    r0 = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg0)
    assert r0.get("status") == "FAIL"
    trace = r0.get("trace") if isinstance(r0.get("trace"), dict) else {}
    tps = trace.get("trace_programs") if isinstance(trace.get("trace_programs"), list) else []
    assert any(
        isinstance(tp, dict)
        and isinstance(tp.get("steps"), list)
        and tp.get("steps")
        and isinstance(tp["steps"][0], dict)
        and str(tp["steps"][0].get("op_id") or "") == "bbox_by_color"
        for tp in tps
    )

    # Operator discovery: close a single-step trace into a reusable grid->grid closure.
    miner = _load_miner_module()
    closed = miner._close_operator_seq(("bbox_by_color",), max_len=5)
    assert closed == ("bbox_by_color", "crop_bbox", "commit_patch")

    macro_row = {
        "kind": "arc_operator_template_v147",
        "schema_version": 147,
        "operator_id": "op_test_bbox_crop_commit",
        "op_ids": list(closed),
        "support": 3,
    }

    # With the mined operator available as a macro_call, the solver can solve under depth=1.
    cfg1 = SolveConfigV141(
        max_depth=1,
        max_programs=200,
        trace_program_limit=80,
        max_ambiguous_outputs=8,
        enable_reachability_pruning=False,
        macro_try_on_fail_only=True,
        macro_propose_max_depth=0,
        macro_templates=(macro_row,),
    )
    r1 = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg1)
    assert r1.get("status") == "SOLVED"
    steps = r1.get("program_steps")
    assert isinstance(steps, list) and steps
    assert str(steps[0].get("op_id") or "") == "macro_call"
    args = steps[0].get("args") if isinstance(steps[0].get("args"), dict) else {}
    assert str(args.get("macro_id") or "") == "op_test_bbox_crop_commit"
    inner = args.get("steps")
    assert isinstance(inner, list)
    assert [str(s.get("op_id") or "") for s in inner if isinstance(s, dict)] == [
        "bbox_by_color",
        "crop_bbox",
        "commit_patch",
    ]

