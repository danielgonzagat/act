from __future__ import annotations

import unittest

from atos_core.arc_selector_v127 import infer_selector_hypotheses_v127
from atos_core.grid_v124 import grid_from_list_v124


class TestArcSelectorV127(unittest.TestCase):
    def test_infer_smallest_area_selector(self) -> None:
        train_pairs = [
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 4, 4, 0, 0, 0],
                        [0, 4, 4, 0, 4, 0],
                        [0, 0, 0, 0, 4, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 4, 4, 0, 0, 0],
                        [0, 4, 4, 0, 9, 0],
                        [0, 0, 0, 0, 9, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
            (
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 4, 4, 4, 0, 0],
                        [0, 0, 0, 0, 4, 0],
                        [0, 0, 0, 0, 4, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
                grid_from_list_v124(
                    [
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 4, 4, 4, 0, 0],
                        [0, 0, 0, 0, 9, 0],
                        [0, 0, 0, 0, 9, 0],
                        [0, 0, 0, 0, 0, 0],
                    ]
                ),
            ),
        ]
        hyps = infer_selector_hypotheses_v127(train_pairs=train_pairs)
        self.assertTrue(hyps)
        # Must include smallest-area selection (color-filtered is ideal, but not strictly required).
        self.assertTrue(any(str(h.selector) == "smallest_area" for h in hyps))


if __name__ == "__main__":
    unittest.main()

