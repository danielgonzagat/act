from __future__ import annotations

import unittest

from atos_core.arc_dsl_v126 import BboxV126, ObjectV126
from atos_core.arc_dsl_v128 import mask_difference_v128, mask_of_object_v128, mask_rect_v128, paint_mask_v128
from atos_core.grid_v124 import grid_from_list_v124


class TestArcDslV128(unittest.TestCase):
    def test_mask_difference(self) -> None:
        a = grid_from_list_v124([[0, 1], [1, 0]])
        b = grid_from_list_v124([[0, 2], [1, 0]])
        m = mask_difference_v128(a, b)
        self.assertEqual([list(r) for r in m], [[0, 1], [0, 0]])

    def test_mask_rect(self) -> None:
        g = grid_from_list_v124([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        bb = BboxV126(r0=1, c0=1, r1=3, c1=3)
        m = mask_rect_v128(g, bb)
        self.assertEqual([list(r) for r in m], [[0, 0, 0], [0, 1, 1], [0, 1, 1]])

    def test_mask_of_object(self) -> None:
        g = grid_from_list_v124([[0, 0], [0, 0]])
        o = ObjectV126(color=1, bbox=BboxV126(r0=0, c0=0, r1=1, c1=2), cells=((0, 1),))
        m = mask_of_object_v128(g, o)
        self.assertEqual([list(r) for r in m], [[0, 1], [0, 0]])

    def test_paint_mask_modes(self) -> None:
        g = grid_from_list_v124([[0, 1], [0, 1]])
        mask = grid_from_list_v124([[1, 1], [0, 1]])

        over = paint_mask_v128(g, mask, color=2, mode="overwrite")
        self.assertEqual([list(r) for r in over], [[2, 2], [0, 2]])

        only_bg = paint_mask_v128(g, mask, color=2, mode="only_bg", bg=0)
        self.assertEqual([list(r) for r in only_bg], [[2, 1], [0, 1]])

        only_color = paint_mask_v128(g, mask, color=2, mode="only_color", only_color=1)
        self.assertEqual([list(r) for r in only_color], [[0, 2], [0, 2]])


if __name__ == "__main__":
    unittest.main()

