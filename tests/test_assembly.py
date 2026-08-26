import unittest
import numpy as np

from puzzle_solver.core.assembly import assemble_puzzle


class TestAssembly(unittest.TestCase):
    def test_assemble_puzzle(self):
        t0 = np.full((10, 10, 3), 10, dtype=np.uint8)
        t1 = np.full((10, 10, 3), 20, dtype=np.uint8)
        t2 = np.full((10, 10, 3), 30, dtype=np.uint8)
        t3 = np.full((10, 10, 3), 40, dtype=np.uint8)

        tiles = [
            {"id": "t0", "img": t0},
            {"id": "t1", "img": t1},
            {"id": "t2", "img": t2},
            {"id": "t3", "img": t3},
        ]

        placement = {"0_0": 0, "0_1": 1, "1_0": 2, "1_1": 3}
        canvas = assemble_puzzle(tiles, placement, rows=2, cols=2)

        self.assertEqual(canvas.shape, (20, 20, 3))
        self.assertEqual(canvas[0, 0, 0], 10)
        self.assertEqual(canvas[0, 15, 0], 20)
        self.assertEqual(canvas[15, 0, 0], 30)
        self.assertEqual(canvas[15, 15, 0], 40)


if __name__ == "__main__":
    unittest.main()
