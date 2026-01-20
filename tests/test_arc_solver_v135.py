from __future__ import annotations

import re
from pathlib import Path

from atos_core.arc_solver_v135 import SolveConfigV135, solve_arc_task_v135
from atos_core.grid_v124 import GridV124, grid_from_list_v124


def _g(rows) -> GridV124:
    return grid_from_list_v124([[int(x) for x in row] for row in rows])


def test_overlay_self_translate_solved_unique() -> None:
    train_in = _g([[0, 1, 0], [0, 0, 0], [0, 0, 0]])
    train_out = _g([[0, 1, 1], [0, 0, 0], [0, 0, 0]])
    test_in = _g([[0, 2, 0], [0, 0, 0], [0, 0, 0]])
    want = _g([[0, 2, 2], [0, 0, 0], [0, 0, 0]])
    res = solve_arc_task_v135(train_pairs=[(train_in, train_out)], test_in=test_in, config=SolveConfigV135(max_depth=2, max_programs=500))
    assert res["status"] == "SOLVED"
    pred = _g(res["predicted_grid"])
    assert pred == want


def test_ambiguous_rule_fail_closed() -> None:
    # Train: rotate90 and reflect_h both map inp->out with equal base cost, but differ on test_in.
    train_in = _g([[1, 2], [2, 3]])
    train_out = _g([[2, 1], [3, 2]])
    test_in = _g([[4, 5], [6, 7]])
    res = solve_arc_task_v135(train_pairs=[(train_in, train_out)], test_in=test_in, config=SolveConfigV135(max_depth=1, max_programs=500))
    assert res["status"] == "UNKNOWN"
    fr = res.get("failure_reason") or {}
    assert fr.get("kind") == "AMBIGUOUS_RULE"


def test_anti_hack_scan_solver_source() -> None:
    src = Path("atos_core/arc_solver_v135.py").read_text(encoding="utf-8")
    assert "task_id" not in src
    assert "Path(" not in src
    assert "glob(" not in src
    assert "rglob(" not in src
    assert re.search(r"\\b[0-9a-f]{8}\\.json\\b", src) is None

