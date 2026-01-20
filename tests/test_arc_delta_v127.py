from __future__ import annotations

import unittest

from atos_core.arc_delta_v127 import compute_delta_v127
from atos_core.grid_v124 import grid_from_list_v124


class TestArcDeltaV127(unittest.TestCase):
    def test_delta_deterministic_sig(self) -> None:
        inp = grid_from_list_v124([[0, 1], [1, 0]])
        out = grid_from_list_v124([[0, 2], [1, 0]])
        d1 = compute_delta_v127(inp, out)
        d2 = compute_delta_v127(inp, out)
        self.assertEqual(d1.delta_sig(), d2.delta_sig())
        self.assertEqual(d1.to_dict(), d2.to_dict())

    def test_changed_bbox_and_palette(self) -> None:
        inp = grid_from_list_v124([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        out = grid_from_list_v124([[0, 0, 0], [0, 2, 0], [0, 0, 0]])
        d = compute_delta_v127(inp, out)
        self.assertFalse(d.shape_changed)
        self.assertEqual(d.changed_cells, 1)
        self.assertEqual(d.changed_bbox, (1, 1, 2, 2))
        self.assertEqual(list(d.palette_added), [2])


if __name__ == "__main__":
    unittest.main()

