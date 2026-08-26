import unittest
import cv2
import numpy as np

from puzzle_solver.core.features import (
    extract_borders,
    normalize_strip_2d,
    border_distance_2d,
    build_compatibility,
)


class TestFeatures(unittest.TestCase):
    def setUp(self):
        self.piece = np.zeros((32, 32, 3), dtype=np.uint8)
        self.piece[:16, :] = [255, 0, 0]
        self.piece[16:, :] = [0, 255, 0]

    def test_extract_borders(self):
        borders = extract_borders(self.piece, strip_width=2)
        self.assertEqual(set(borders.keys()), {0, 1, 2, 3})
        self.assertEqual(borders[0].shape, (2, 32, 6))
        self.assertEqual(borders[1].shape, (32, 2, 6))

    def test_normalize_strip_2d(self):
        strip = np.random.rand(4, 32, 6) * 100.0
        normalized = normalize_strip_2d(strip)
        for ch in range(6):
            self.assertAlmostEqual(float(normalized[..., ch].mean()), 0.0, places=4)
            self.assertAlmostEqual(float(normalized[..., ch].std()), 1.0, places=4)

    def test_border_distance_2d(self):
        borders = extract_borders(self.piece, strip_width=1)
        dist_self = border_distance_2d(borders[0], borders[0], 0, 0)
        self.assertAlmostEqual(dist_self, 0.0, places=4)

    def test_build_compatibility(self):
        pieces = [self.piece, self.piece.copy()]
        compat = build_compatibility(pieces, strip_width=1)
        self.assertEqual(set(compat.keys()), {0, 1, 2, 3})
        for side in range(4):
            self.assertEqual(compat[side].shape, (2, 2))
            self.assertEqual(compat[side][0, 0], 1e9)


if __name__ == "__main__":
    unittest.main()
