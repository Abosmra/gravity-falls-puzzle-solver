import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def assemble_puzzle(
    tiles: List[Dict[str, Any]],
    placement: Dict[str, int],
    rows: int,
    cols: int,
    output_path: Optional[Union[str, Path]] = None
) -> np.ndarray:
    """Stitch puzzle tiles into a reconstructed composite image according to placement map.

    Args:
        tiles: List of tile dictionaries containing an 'img' ndarray.
        placement: Mapping of '{r}_{c}' grid coordinates to tile indices.
        rows: Number of grid rows.
        cols: Number of grid columns.
        output_path: Optional file path to save the assembled image.

    Returns:
        Assembled composite image as a BGR numpy ndarray.
    """
    sizes = [(t['img'].shape[0], t['img'].shape[1]) for t in tiles]
    size_counts: Dict[Tuple[int, int], int] = {}
    for s in sizes:
        size_counts[s] = size_counts.get(s, 0) + 1
    tile_h, tile_w = max(size_counts.items(), key=lambda kv: kv[1])[0]

    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            key = f"{r}_{c}"
            if key in placement:
                tile_idx = placement[key]
                if 0 <= tile_idx < len(tiles):
                    tile_img = tiles[tile_idx]['img']
                    h, w = tile_img.shape[:2]

                    if h != tile_h or w != tile_w:
                        tile_img = cv2.resize(tile_img, (tile_w, tile_h))

                    y_start = r * tile_h
                    x_start = c * tile_w
                    canvas[y_start : y_start + tile_h, x_start : x_start + tile_w] = tile_img

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_p), canvas)

    return canvas
