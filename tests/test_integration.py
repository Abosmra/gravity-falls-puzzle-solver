import os
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from puzzle_solver.core.tiling import preprocess_image
from puzzle_solver.core.features import load_tiles_from_phase1
from puzzle_solver.core.solver import PuzzleSolver
from puzzle_solver.core.assembly import assemble_puzzle
from puzzle_solver.pipeline import solve_image


class TestIntegrationPipeline(unittest.TestCase):
    def test_synthetic_end_to_end_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            source_img_path = tmp_p / "input_test.png"
            phase1_out = tmp_p / "phase1_out"
            phase2_out = tmp_p / "phase2_out"

            img = np.zeros((64, 64, 3), dtype=np.uint8)
            for r in range(64):
                for c in range(64):
                    img[r, c] = [r * 4, c * 4, (r + c) * 2]
            cv2.imwrite(str(source_img_path), img)

            p1_dest = phase1_out / "puzzle_2x2" / "sample"
            preprocess_image(source_img_path, p1_dest, rows=2, cols=2)

            tiles = load_tiles_from_phase1(phase1_out, "puzzle_2x2", "sample")
            self.assertEqual(len(tiles), 4)

            solver = PuzzleSolver(tiles, rows=2, cols=2)
            result = solver.solve(time_limit=10.0)
            self.assertIsNotNone(result)

            assembled_path = phase2_out / "puzzle_2x2" / "sample.png"
            canvas = assemble_puzzle(tiles, result["placement_map"], 2, 2, output_path=assembled_path)

            self.assertTrue(assembled_path.exists())
            self.assertEqual(canvas.shape, (64, 64, 3))

    def test_solve_image_direct(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            source_img_path = tmp_p / "direct_input.png"
            out_img_path = tmp_p / "direct_solved.png"

            img = np.zeros((48, 48, 3), dtype=np.uint8)
            for r in range(48):
                for c in range(48):
                    img[r, c] = [r * 5, c * 5, (r + c)]
            cv2.imwrite(str(source_img_path), img)

            canvas, result = solve_image(source_img_path, rows=2, cols=2, output_path=out_img_path, time_limit=5.0)
            self.assertTrue(out_img_path.exists())
            self.assertEqual(canvas.shape, (48, 48, 3))
            self.assertIn("placement_map", result)


if __name__ == "__main__":
    unittest.main()
