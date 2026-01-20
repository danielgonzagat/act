from __future__ import annotations

import unittest

from atos_core.arc_delta_v128 import compute_delta_v128
from atos_core.grid_v124 import grid_from_list_v124


class TestArcDeltaV128(unittest.TestCase):
    def test_delta_deterministic_sig(self) -> None:
        inp = grid_from_list_v124([[0, 1], [1, 0]])
        out = grid_from_list_v124([[0, 2], [1, 0]])
        d1 = compute_delta_v128(inp, out)
        d2 = compute_delta_v128(inp, out)
        self.assertEqual(d1.delta_sig(), d2.delta_sig())
        self.assertEqual(d1.to_dict(), d2.to_dict())

    def test_color_decomposition_masks(self) -> None:
        inp = grid_from_list_v124([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        out = grid_from_list_v124([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        d = compute_delta_v128(inp, out)
        self.assertFalse(d.shape_changed)
        self.assertEqual(d.changed_cells, 1)
        self.assertEqual(d.changed_bbox, (1, 1, 2, 2))
        self.assertEqual(list(d.palette_added), [2])
        self.assertEqual(list(d.out_colors_in_changed), [2])
        self.assertEqual(list(d.in_colors_in_changed), [1])

        self.assertEqual(len(d.changed_out_color_masks), 1)
        c_out, mask_out, bb_out, cnt_out = d.changed_out_color_masks[0]
        self.assertEqual(int(c_out), 2)
        self.assertEqual(int(cnt_out), 1)
        self.assertEqual(tuple(int(x) for x in bb_out), (1, 1, 2, 2))
        self.assertEqual([list(r) for r in mask_out], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])

        self.assertEqual(len(d.changed_in_color_masks), 1)
        c_in, mask_in, bb_in, cnt_in = d.changed_in_color_masks[0]
        self.assertEqual(int(c_in), 1)
        self.assertEqual(int(cnt_in), 1)
        self.assertEqual(tuple(int(x) for x in bb_in), (1, 1, 2, 2))
        self.assertEqual([list(r) for r in mask_in], [[0, 0, 0], [0, 1, 0], [0, 0, 0]])


if __name__ == "__main__":
    unittest.main()

