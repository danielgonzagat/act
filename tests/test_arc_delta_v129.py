from __future__ import annotations

import unittest

from atos_core.arc_delta_v129 import compute_delta_v129
from atos_core.grid_v124 import grid_from_list_v124


class TestArcDeltaV129(unittest.TestCase):
    def test_delta_deterministic_sig(self) -> None:
        inp = grid_from_list_v124([[0, 1], [1, 0]])
        out = grid_from_list_v124([[0, 2], [1, 0]])
        d1 = compute_delta_v129(inp, out)
        d2 = compute_delta_v129(inp, out)
        self.assertEqual(d1.delta_sig(), d2.delta_sig())
        self.assertEqual(d1.to_dict(), d2.to_dict())

    def test_ratio_when_scaled(self) -> None:
        inp = grid_from_list_v124([[1, 2], [3, 4]])
        out = grid_from_list_v124(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ]
        )
        d = compute_delta_v129(inp, out)
        self.assertTrue(d.shape_changed)
        self.assertEqual(int(d.ratio_h or 0), 2)
        self.assertEqual(int(d.ratio_w or 0), 2)


if __name__ == "__main__":
    unittest.main()

