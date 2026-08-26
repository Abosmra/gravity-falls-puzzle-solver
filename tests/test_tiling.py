import json
from pathlib import Path
import tempfile
import unittest
import cv2
import numpy as np

from puzzle_solver.core.tiling import (
    detect_grid_from_folder,
    smart_enhance,
    preprocess_image,
)


class TestTiling(unittest.TestCase):
    def test_detect_grid_from_folder(self):
        self.assertEqual(detect_grid_from_folder("puzzle_2x2"), (2, 2))
        self.assertEqual(detect_grid_from_folder("dataset/puzzle_4x4/0"), (4, 4))
        self.assertEqual(detect_grid_from_folder("PUZZLE_8x8"), (8, 8))
        self.assertEqual(detect_grid_from_folder("random_folder"), (None, None))

    def test_smart_enhance(self):
        img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        enhanced = smart_enhance(img, tile_size=32)
        self.assertEqual(enhanced.shape, (32, 32, 3))
        self.assertEqual(enhanced.dtype, np.uint8)

    def test_preprocess_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_p = Path(tmp_dir)
            img_path = tmp_p / "input.png"
            out_path = tmp_p / "output"

            test_img = np.zeros((65, 65, 3), dtype=np.uint8)
            cv2.rectangle(test_img, (0, 0), (32, 32), (255, 0, 0), -1)
            cv2.rectangle(test_img, (33, 33), (65, 65), (0, 255, 0), -1)
            cv2.imwrite(str(img_path), test_img)

            res_path, metadata = preprocess_image(img_path, out_path, rows=2, cols=2)
            self.assertTrue(res_path.exists())
            self.assertEqual(metadata["num_tiles_saved"], 4)
            self.assertEqual(metadata["rows"], 2)
            self.assertEqual(metadata["cols"], 2)

            tiles_dir = out_path / "tiles"
            for fn in metadata["tile_filenames"]:
                self.assertTrue((tiles_dir / fn).exists())


if __name__ == "__main__":
    unittest.main()
