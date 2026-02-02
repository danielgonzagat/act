"""
Tests for arc_ops_v148.py - Extended ARC Operators.
"""

import unittest
from atos_core.arc_ops_v148 import (
    # Morphological
    _morpho_dilate_v148,
    _morpho_erode_v148,
    _morpho_open_v148,
    _morpho_close_v148,
    _morpho_skeleton_v148,
    # Symmetry
    _detect_symmetry_h_v148,
    _detect_symmetry_v_v148,
    _detect_symmetry_diag_v148,
    _detect_symmetry_rot180_v148,
    _detect_symmetry_rot90_v148,
    _get_symmetry_type_v148,
    _symmetry_complete_h_v148,
    _symmetry_complete_v_v148,
    _symmetry_complete_rot180_v148,
    # Scaling
    _scale_up_v148,
    _scale_down_v148,
    _resize_v148,
    # Object grouping
    _merge_adjacent_objects_v148,
    _fill_between_objects_v148,
    _group_by_color_v148,
    # Features
    _count_colors_v148,
    _most_frequent_color_v148,
    _least_frequent_color_v148,
    _count_objects_v148,
    _find_pattern_v148,
    _extract_unique_patterns_v148,
    # Advanced
    _fractal_tile_v148,
    _xor_grids_v148,
    _and_grids_v148,
    _or_grids_v148,
    # Operator defs
    OP_DEFS_V148,
    apply_op_v148,
    step_cost_bits_v148,
)
from atos_core.arc_ops_v132 import StateV132


class TestMorphologicalOps(unittest.TestCase):
    """Tests for morphological operations."""
    
    def test_dilate_expands_shape(self):
        """Dilation should expand non-bg regions."""
        g = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
        result = _morpho_dilate_v148(g, bg=0)
        
        # Center point should spread to 4-neighbors
        self.assertEqual(result[0][2], 1)  # above
        self.assertEqual(result[2][2], 1)  # below
        self.assertEqual(result[1][1], 1)  # left
        self.assertEqual(result[1][3], 1)  # right
    
    def test_erode_shrinks_shape(self):
        """Erosion should shrink non-bg regions."""
        g = [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
        result = _morpho_erode_v148(g, bg=0)
        
        # Edges should be eroded
        self.assertEqual(result[0][1], 0)
        self.assertEqual(result[2][1], 0)
        # Center should remain
        self.assertEqual(result[1][1], 1)
    
    def test_open_removes_noise(self):
        """Opening should remove small protrusions."""
        g = [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 1, 1],
        ]
        result = _morpho_open_v148(g, bg=0)
        
        # Single pixel (noise) should be removed
        self.assertEqual(result[0][1], 0)
    
    def test_close_fills_holes(self):
        """Closing should fill small holes."""
        g = [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
        result = _morpho_close_v148(g, bg=0)
        
        # The hole should be filled
        self.assertEqual(result[1][1], 1)
    
    def test_skeleton_preserves_structure(self):
        """Skeletonization should reduce shape but preserve connectivity."""
        g = [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
        result = _morpho_skeleton_v148(g, bg=0)
        
        # Result should have at least one non-bg cell (connectivity preserved)
        nonzero = sum(1 for row in result for c in row if c != 0)
        self.assertGreater(nonzero, 0)


class TestSymmetryDetection(unittest.TestCase):
    """Tests for symmetry detection."""
    
    def test_detect_h_symmetry(self):
        """Detect horizontal symmetry."""
        symmetric = [
            [1, 2, 1],
            [3, 4, 3],
        ]
        asymmetric = [
            [1, 2, 3],
            [4, 5, 6],
        ]
        
        self.assertTrue(_detect_symmetry_h_v148(symmetric))
        self.assertFalse(_detect_symmetry_h_v148(asymmetric))
    
    def test_detect_v_symmetry(self):
        """Detect vertical symmetry."""
        symmetric = [
            [1, 2],
            [3, 4],
            [1, 2],
        ]
        asymmetric = [
            [1, 2],
            [3, 4],
            [5, 6],
        ]
        
        self.assertTrue(_detect_symmetry_v_v148(symmetric))
        self.assertFalse(_detect_symmetry_v_v148(asymmetric))
    
    def test_detect_diag_symmetry(self):
        """Detect diagonal (transpose) symmetry."""
        symmetric = [
            [1, 2, 3],
            [2, 4, 5],
            [3, 5, 6],
        ]
        asymmetric = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        
        self.assertTrue(_detect_symmetry_diag_v148(symmetric))
        self.assertFalse(_detect_symmetry_diag_v148(asymmetric))
    
    def test_detect_rot180_symmetry(self):
        """Detect 180° rotational symmetry."""
        symmetric = [
            [1, 2, 1],
            [3, 4, 3],
            [1, 2, 1],
        ]
        
        self.assertTrue(_detect_symmetry_rot180_v148(symmetric))
    
    def test_detect_rot90_symmetry(self):
        """Detect 90° rotational symmetry (square only)."""
        symmetric = [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
        
        self.assertTrue(_detect_symmetry_rot90_v148(symmetric))
    
    def test_get_symmetry_type(self):
        """Get symmetry type classification."""
        hv = [
            [1, 2, 1],
            [3, 4, 3],
            [1, 2, 1],
        ]
        h_only = [
            [1, 2, 1],
            [3, 4, 3],
            [5, 6, 5],
        ]
        none = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        
        self.assertEqual(_get_symmetry_type_v148(hv), "hv")
        self.assertEqual(_get_symmetry_type_v148(h_only), "h")
        self.assertEqual(_get_symmetry_type_v148(none), "none")
    
    def test_symmetry_complete_h(self):
        """Complete to horizontal symmetry."""
        partial = [
            [1, 2, 0],
            [3, 4, 0],
        ]
        result = _symmetry_complete_h_v148(partial)
        
        self.assertEqual(result[0][2], 1)
        self.assertEqual(result[1][2], 3)
    
    def test_symmetry_complete_v(self):
        """Complete to vertical symmetry."""
        partial = [
            [1, 2],
            [3, 4],
            [0, 0],
        ]
        result = _symmetry_complete_v_v148(partial)
        
        self.assertEqual(result[2][0], 1)
        self.assertEqual(result[2][1], 2)


class TestScaling(unittest.TestCase):
    """Tests for scaling operations."""
    
    def test_scale_up_2x(self):
        """Scale up by 2x."""
        g = [
            [1, 2],
            [3, 4],
        ]
        result = _scale_up_v148(g, factor=2)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 1)
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][1], 1)
        self.assertEqual(result[2][2], 4)
    
    def test_scale_down_2x(self):
        """Scale down by 2x."""
        g = [
            [1, 1, 2, 2],
            [1, 1, 2, 2],
            [3, 3, 4, 4],
            [3, 3, 4, 4],
        ]
        result = _scale_down_v148(g, factor=2, mode="mode")
        
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 2)
        self.assertEqual(result[1][0], 3)
        self.assertEqual(result[1][1], 4)
    
    def test_resize_smaller(self):
        """Resize to smaller grid."""
        g = [
            [1, 2, 3],
            [4, 5, 6],
            [7, 8, 9],
        ]
        result = _resize_v148(g, new_h=2, new_w=2)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][1], 5)
    
    def test_resize_larger(self):
        """Resize to larger grid with padding."""
        g = [
            [1, 2],
            [3, 4],
        ]
        result = _resize_v148(g, new_h=4, new_w=4, bg=0)
        
        self.assertEqual(len(result), 4)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[2][2], 0)  # padded


class TestObjectGrouping(unittest.TestCase):
    """Tests for object grouping operations."""
    
    def test_merge_adjacent(self):
        """Merge adjacent objects."""
        g = [
            [1, 0, 2],
            [1, 2, 2],
            [0, 0, 0],
        ]
        result = _merge_adjacent_objects_v148(g, bg=0)
        
        # All non-bg cells should have same color (8-connected)
        nonbg_colors = set()
        for r in range(3):
            for c in range(3):
                if result[r][c] != 0:
                    nonbg_colors.add(result[r][c])
        
        self.assertEqual(len(nonbg_colors), 1)
    
    def test_fill_between(self):
        """Fill between objects in rows."""
        g = [
            [1, 0, 0, 0, 1],
            [0, 0, 0, 0, 0],
            [2, 0, 0, 2, 0],
        ]
        result = _fill_between_objects_v148(g, bg=0, fill_color=5)
        
        # Row 0: should fill between two 1s
        self.assertEqual(result[0][2], 5)
        # Row 2: should fill between two 2s
        self.assertEqual(result[2][2], 5)
    
    def test_group_by_color(self):
        """Group cells by color."""
        g = [
            [1, 2, 1],
            [2, 0, 2],
            [1, 2, 1],
        ]
        groups = _group_by_color_v148(g, bg=0)
        
        self.assertEqual(len(groups[1]), 4)  # 4 cells with color 1
        self.assertEqual(len(groups[2]), 4)  # 4 cells with color 2


class TestFeatureExtraction(unittest.TestCase):
    """Tests for feature extraction."""
    
    def test_count_colors(self):
        """Count color occurrences."""
        g = [
            [0, 1, 1],
            [2, 2, 2],
            [0, 0, 1],
        ]
        counts = _count_colors_v148(g)
        
        self.assertEqual(counts[0], 3)  # 3 zeros
        self.assertEqual(counts[1], 3)  # 3 ones
        self.assertEqual(counts[2], 3)  # 3 twos
    
    def test_most_frequent_color(self):
        """Find most frequent non-bg color."""
        g = [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 0],
        ]
        most = _most_frequent_color_v148(g, exclude_bg=True, bg=0)
        
        self.assertEqual(most, 1)
    
    def test_least_frequent_color(self):
        """Find least frequent non-bg color."""
        g = [
            [0, 0, 0],
            [1, 1, 1],
            [2, 0, 0],
        ]
        least = _least_frequent_color_v148(g, exclude_bg=True, bg=0)
        
        self.assertEqual(least, 2)
    
    def test_count_objects(self):
        """Count connected components."""
        g = [
            [1, 0, 2],
            [1, 0, 2],
            [0, 0, 0],
            [3, 3, 0],
        ]
        count = _count_objects_v148(g, bg=0)
        
        self.assertEqual(count, 3)  # 3 separate objects
    
    def test_find_pattern(self):
        """Find pattern occurrences."""
        g = [
            [1, 2, 1, 2],
            [3, 4, 3, 4],
            [1, 2, 1, 2],
        ]
        pattern = [
            [1, 2],
            [3, 4],
        ]
        positions = _find_pattern_v148(g, pattern)
        
        self.assertEqual(len(positions), 2)
        self.assertIn((0, 0), positions)
        self.assertIn((0, 2), positions)
    
    def test_extract_unique_patterns(self):
        """Extract unique 2x2 patterns."""
        g = [
            [1, 2, 1],
            [3, 4, 3],
            [1, 2, 1],
        ]
        patterns = _extract_unique_patterns_v148(g, size=2, bg=0)
        
        # Should find unique 2x2 patterns
        self.assertGreater(len(patterns), 0)


class TestAdvancedTransforms(unittest.TestCase):
    """Tests for advanced transformations."""
    
    def test_fractal_tile(self):
        """Fractal tiling expands pattern."""
        g = [
            [1, 0],
            [0, 1],
        ]
        result = _fractal_tile_v148(g, iterations=1)
        
        # 2x2 -> 4x4 after one iteration
        self.assertEqual(len(result), 4)
        self.assertEqual(len(result[0]), 4)
    
    def test_xor_grids(self):
        """XOR two grids."""
        g1 = [
            [1, 0],
            [0, 1],
        ]
        g2 = [
            [0, 1],
            [1, 0],
        ]
        result = _xor_grids_v148(g1, g2, bg=0)
        
        # XOR: only where one is non-bg
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 1)
        self.assertEqual(result[1][0], 1)
        self.assertEqual(result[1][1], 1)
    
    def test_and_grids(self):
        """AND two grids."""
        g1 = [
            [1, 1],
            [0, 1],
        ]
        g2 = [
            [1, 0],
            [1, 1],
        ]
        result = _and_grids_v148(g1, g2, bg=0)
        
        # AND: only where both are non-bg
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 0)
        self.assertEqual(result[1][0], 0)
        self.assertEqual(result[1][1], 1)
    
    def test_or_grids(self):
        """OR two grids."""
        g1 = [
            [1, 0],
            [0, 0],
        ]
        g2 = [
            [0, 2],
            [0, 0],
        ]
        result = _or_grids_v148(g1, g2, bg=0)
        
        # OR: where either is non-bg
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[0][1], 2)


class TestOperatorDefs(unittest.TestCase):
    """Tests for operator definitions."""
    
    def test_new_operators_exist(self):
        """New operators should be defined."""
        new_ops = [
            "morpho_dilate",
            "morpho_erode",
            "morpho_open",
            "morpho_close",
            "morpho_skeleton",
            "symmetry_complete_h",
            "symmetry_complete_v",
            "symmetry_complete_rot180",
            "scale_up",
            "scale_down",
            "merge_adjacent",
            "fill_between",
            "fractal_tile",
        ]
        
        for op in new_ops:
            self.assertIn(op, OP_DEFS_V148, f"Missing operator: {op}")
    
    def test_apply_op_morpho(self):
        """Apply morphological operator."""
        g = [[0, 1, 0], [0, 0, 0], [0, 0, 0]]
        state = StateV132(grid=g)
        
        result = apply_op_v148(state, "morpho_dilate", {"bg": 0})
        
        # Should have expanded
        self.assertNotEqual(result.grid, g)
    
    def test_apply_op_symmetry(self):
        """Apply symmetry completion operator."""
        g = [[1, 2, 0], [3, 4, 0]]
        state = StateV132(grid=g)
        
        result = apply_op_v148(state, "symmetry_complete_h", {})
        
        # Should be symmetric
        self.assertTrue(_detect_symmetry_h_v148(result.grid))
    
    def test_step_cost_bits(self):
        """Step cost should be defined for new ops."""
        self.assertGreater(step_cost_bits_v148("morpho_dilate", {}), 0)
        self.assertGreater(step_cost_bits_v148("fractal_tile", {}), 0)
    
    def test_total_operators_increased(self):
        """Should have more operators than V141."""
        from atos_core.arc_ops_v141 import OP_DEFS_V141
        
        self.assertGreater(len(OP_DEFS_V148), len(OP_DEFS_V141))


if __name__ == "__main__":
    unittest.main()
