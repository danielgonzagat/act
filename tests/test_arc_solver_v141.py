from __future__ import annotations

import re
import unittest
from pathlib import Path

from atos_core.arc_ops_v141 import apply_op_v141
from atos_core.arc_ops_v132 import StateV132
from atos_core.arc_solver_v141 import SolveConfigV141, solve_arc_task_v141
from atos_core.grid_v124 import grid_from_list_v124


class TestArcSolverV141(unittest.TestCase):
    def test_determinism_same_seed(self) -> None:
        train_in = grid_from_list_v124([[1, 2], [3, 4]])
        train_out = grid_from_list_v124([[2, 1], [4, 3]])  # reflect_h
        test_in = grid_from_list_v124([[5, 6], [7, 8]])
        cfg = SolveConfigV141(max_depth=2, max_programs=200, trace_program_limit=5, max_ambiguous_outputs=8)
        r1 = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        r2 = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(r1.get("status"), "SOLVED")
        self.assertEqual(r1.get("predicted_grid_hash"), r2.get("predicted_grid_hash"))
        self.assertEqual(r1.get("program_sig"), r2.get("program_sig"))

    def test_ambiguous_rule_fail_closed(self) -> None:
        # reflect_h and reflect_v coincide on train, but diverge on test.
        train_in = grid_from_list_v124([[1, 2], [2, 1]])
        train_out = grid_from_list_v124([[2, 1], [1, 2]])  # both reflect_h and reflect_v
        test_in = grid_from_list_v124([[1, 2], [3, 4]])
        cfg = SolveConfigV141(max_depth=1, max_programs=100, trace_program_limit=5, max_ambiguous_outputs=8)
        r = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "UNKNOWN")
        self.assertIsInstance(r.get("predicted_grids"), list)
        self.assertGreaterEqual(len(r.get("predicted_grids") or []), 2)

    def test_invariant_violation_classification(self) -> None:
        # Use raw tuple-of-tuples to bypass grid_from_list_v124 validation and
        # exercise solver's own invariant checks.
        train_in = ((10,),)  # type: ignore[assignment]
        train_out = grid_from_list_v124([[0]])
        test_in = grid_from_list_v124([[0]])
        cfg = SolveConfigV141(max_depth=1, max_programs=10, trace_program_limit=1, max_ambiguous_outputs=2)
        r = solve_arc_task_v141(train_pairs=[(train_in, train_out)], test_in=test_in, config=cfg)
        self.assertEqual(r.get("status"), "FAIL")
        fr = r.get("failure_reason") or {}
        self.assertEqual(fr.get("kind"), "INVARIANT_VIOLATION")

    def test_cc4_nonbg_multicolor_groups_component(self) -> None:
        g = grid_from_list_v124([[1, 2], [0, 0]])
        st = StateV132(grid=g)
        st2 = apply_op_v141(state=st, op_id="cc4_nonbg_multicolor", args={"bg": 0})
        self.assertIsNotNone(st2.objset)
        objs = list(st2.objset.objects) if st2.objset is not None else []
        self.assertEqual(len(objs), 1)
        self.assertEqual(int(objs[0].area), 2)
        self.assertEqual(objs[0].bbox.to_tuple(), (0, 0, 1, 2))

    def test_static_anti_hack_scan(self) -> None:
        # Solver/ops must not contain dataset/path/task_id conditionals.
        root = Path(__file__).resolve().parent.parent
        solver_src = (root / "atos_core" / "arc_solver_v141.py").read_text(encoding="utf-8")
        ops_src = (root / "atos_core" / "arc_ops_v141.py").read_text(encoding="utf-8")
        for src in (solver_src, ops_src):
            self.assertNotRegex(src, r"\btask_id\b")
            self.assertNotIn("Path(", src)
            self.assertNotIn("glob(", src)
            self.assertNotIn("rglob(", src)
            self.assertIsNone(re.search(r"\b[0-9a-f]{8}(?:\.json)?\b", src))
