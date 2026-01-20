from __future__ import annotations

import unittest

from atos_core.arc_dsl_v129 import apply_op_v129
from atos_core.grid_v124 import grid_from_list_v124


class TestArcDslV129(unittest.TestCase):
    def test_rotate90(self) -> None:
        g = grid_from_list_v124([[1, 0, 0], [0, 2, 0]])
        got = apply_op_v129(op_id="rotate90", inputs=[g], args={})
        self.assertEqual([list(r) for r in got], [[0, 1], [2, 0], [0, 0]])

    def test_reflect_h(self) -> None:
        g = grid_from_list_v124([[1, 2, 3], [4, 5, 6]])
        got = apply_op_v129(op_id="reflect_h", inputs=[g], args={})
        self.assertEqual([list(r) for r in got], [[3, 2, 1], [6, 5, 4]])

    def test_transpose(self) -> None:
        g = grid_from_list_v124([[1, 2, 3], [4, 5, 6]])
        got = apply_op_v129(op_id="transpose", inputs=[g], args={})
        self.assertEqual([list(r) for r in got], [[1, 4], [2, 5], [3, 6]])

    def test_scale_up(self) -> None:
        g = grid_from_list_v124([[1, 2], [3, 4]])
        got = apply_op_v129(op_id="scale_up", inputs=[g], args={"k": 2})
        self.assertEqual(
            [list(r) for r in got],
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
        )

    def test_tile(self) -> None:
        g = grid_from_list_v124([[1, 2], [3, 4]])
        got = apply_op_v129(op_id="tile", inputs=[g], args={"reps_h": 2, "reps_w": 3})
        self.assertEqual(
            [list(r) for r in got],
            [
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4],
                [1, 2, 1, 2, 1, 2],
                [3, 4, 3, 4, 3, 4],
            ],
        )

    def test_mask_outline_and_dilate(self) -> None:
        mask = grid_from_list_v124(
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ]
        )
        outline = apply_op_v129(op_id="mask_outline", inputs=[mask], args={})
        self.assertEqual(
            [list(r) for r in outline],
            [
                [0, 0, 0, 0, 0],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 1, 1, 1, 0],
                [0, 0, 0, 0, 0],
            ],
        )
        dil = apply_op_v129(op_id="mask_dilate", inputs=[outline], args={"steps": 1})
        self.assertEqual(int(dil[2][2]), 1)


if __name__ == "__main__":
    unittest.main()

