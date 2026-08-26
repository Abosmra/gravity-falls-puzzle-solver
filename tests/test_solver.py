import unittest
import numpy as np

from puzzle_solver.core.solver import (
    PuzzleSolver,
    opposite_side,
    best_partner_for,
    is_best_buddy,
    solve_bruteforce,
)
from puzzle_solver.core.features import build_compatibility


class TestSolver(unittest.TestCase):
    def test_opposite_side(self):
        self.assertEqual(opposite_side(0), 2)
        self.assertEqual(opposite_side(1), 3)
        self.assertEqual(opposite_side(2), 0)
        self.assertEqual(opposite_side(3), 1)

    def test_best_buddies_logic(self):
        compat = {s: np.full((3, 3), 10.0, dtype=np.float32) for s in range(4)}
        compat[1][0, 1] = 0.1
        compat[3][1, 0] = 0.1

        self.assertTrue(is_best_buddy(0, 1, 1, compat))
        self.assertTrue(is_best_buddy(1, 3, 0, compat))
        self.assertFalse(is_best_buddy(0, 1, 2, compat))

    def test_bruteforce_solver_2x2(self):
        # Create continuous 2D gradient pattern
        base = np.zeros((64, 64, 3), dtype=np.uint8)
        for r in range(64):
            for c in range(64):
                base[r, c] = [r * 4, c * 4, int((r + c) * 1.5)]

        # Extract tiles in order: top-left (0), top-right (1), bot-left (2), bot-right (3)
        tiles = [
            {"id": "0", "img": base[:32, :32].copy()},
            {"id": "1", "img": base[:32, 32:].copy()},
            {"id": "2", "img": base[32:, :32].copy()},
            {"id": "3", "img": base[32:, 32:].copy()},
        ]

        # Scramble order
        scrambled_indices = [3, 0, 2, 1]
        scrambled_tiles = [tiles[i] for i in scrambled_indices]
        pieces = [t["img"] for t in scrambled_tiles]

        compat = build_compatibility(pieces, strip_width=1)
        order, score = solve_bruteforce(pieces, compat, grid_n=2)

        # Map solved permutation back to original IDs
        solved_ids = [scrambled_tiles[idx]["id"] for idx in order]
        self.assertEqual(solved_ids, ["0", "1", "2", "3"])

    def test_puzzle_solver_class(self):
        tiles = [
            {"id": f"{i}", "img": np.full((16, 16, 3), i * 50, dtype=np.uint8)}
            for i in range(4)
        ]
        solver = PuzzleSolver(tiles, rows=2, cols=2)
        result = solver.solve(time_limit=5.0)

        self.assertIsNotNone(result)
        self.assertIn("placement_map", result)
        self.assertIn("score", result)
        self.assertEqual(len(result["order"]), 4)


if __name__ == "__main__":
    unittest.main()
