import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

MIN_TILE_SIDE: int = 8


def detect_grid_from_folder(folder_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Infer the grid dimensions (rows, cols) from a folder or path name.

    Args:
        folder_name: Directory or group name (e.g. 'puzzle_4x4' or 'puzzle_8x8').

    Returns:
        Tuple of (rows, cols) if matched, or (None, None) if undetermined.
    """
    name = folder_name.lower()
    for pattern in ["2x2", "4x4", "8x8"]:
        if pattern in name:
            r, c = pattern.split("x")
            return int(r), int(c)
    return None, None


def ensure_clean_dir(path: Path) -> None:
    """Ensure directory exists and is empty."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def smart_enhance(img: np.ndarray, tile_size: Optional[int] = None) -> np.ndarray:
    """Enhance tile image quality while preserving critical edge information.

    Applies bilateral filtering for noise reduction, guided smoothing for contour
    preservation, and unsharp masking for gradient enhancement.

    Args:
        img: Input BGR image tile as a numpy ndarray.
        tile_size: Average tile side length for adaptive kernel sizing.

    Returns:
        Enhanced image tile as a numpy ndarray.
    """
    original = img.copy()

    if tile_size is not None:
        scale = tile_size / 112.0
        d_bilateral = max(3, int(9 * scale))
        guided_radius = max(2, int(8 * scale))
        bilateral_fallback_d = max(3, int(7 * scale))
    else:
        d_bilateral = 9
        guided_radius = 8
        bilateral_fallback_d = 7

    # Edge-preserving bilateral denoising
    img_filtered = cv2.bilateralFilter(
        img,
        d=d_bilateral,
        sigmaColor=40,
        sigmaSpace=40
    )

    # Guided filtering with fallback
    try:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "guidedFilter"):
            img_smooth = cv2.ximgproc.guidedFilter(
                guide=img_filtered,
                src=img_filtered,
                radius=guided_radius,
                eps=1e-2
            )
        else:
            img_smooth = cv2.bilateralFilter(
                img_filtered,
                d=bilateral_fallback_d,
                sigmaColor=20,
                sigmaSpace=20
            )
    except Exception:
        img_smooth = cv2.bilateralFilter(
            img_filtered,
            d=bilateral_fallback_d,
            sigmaColor=20,
            sigmaSpace=20
        )

    # High-frequency sharpening via unsharp masking
    blur = cv2.GaussianBlur(img_smooth, (0, 0), sigmaX=1.1)
    sharp = cv2.addWeighted(img_smooth, 1.12, blur, -0.12, 0)

    # Weighted composite blend with original
    final = cv2.addWeighted(original, 0.55, sharp, 0.45, 0)
    return final


def preprocess_image(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    rows: int,
    cols: int
) -> Tuple[Path, Dict]:
    """Slice an input source image into regular grid tiles with metadata tracking.

    Args:
        in_path: Path to the input image file.
        out_path: Path to the destination directory for output tiles and metadata.
        rows: Number of grid rows.
        cols: Number of grid columns.

    Returns:
        Tuple containing the output Path and the generated metadata dictionary.

    Raises:
        RuntimeError: If the input image cannot be read.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)

    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {in_path}")

    ensure_clean_dir(out_path)

    h_img, w_img = img.shape[:2]

    # Distribute remainder pixels evenly across tiles
    base_w = w_img // cols
    rem_w = w_img % cols
    cell_ws = [base_w + (1 if c < rem_w else 0) for c in range(cols)]

    base_h = h_img // rows
    rem_h = h_img % rows
    cell_hs = [base_h + (1 if r < rem_h else 0) for r in range(rows)]

    # Compute coordinate offsets
    x_offsets = [0]
    for w_c in cell_ws[:-1]:
        x_offsets.append(x_offsets[-1] + w_c)

    y_offsets = [0]
    for h_r in cell_hs[:-1]:
        y_offsets.append(y_offsets[-1] + h_r)

    tiles_dir = out_path / "tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)

    filenames: List[str] = []
    saved_count = 0

    for r in range(rows):
        for c in range(cols):
            x1 = x_offsets[c]
            y1 = y_offsets[r]
            w_tile = cell_ws[c]
            h_tile = cell_hs[r]
            x2 = x1 + w_tile
            y2 = y1 + h_tile

            tile_raw = img[y1:y2, x1:x2].copy()
            if tile_raw.shape[0] < MIN_TILE_SIDE or tile_raw.shape[1] < MIN_TILE_SIDE:
                continue

            avg_tile_size = (w_tile + h_tile) // 2
            tile_enh = smart_enhance(tile_raw, tile_size=avg_tile_size)

            tile_name = f"tile_{r:02d}_{c:02d}.png"
            cv2.imwrite(str(tiles_dir / tile_name), tile_enh)

            filenames.append(tile_name)
            saved_count += 1

    metadata = {
        "source": str(in_path),
        "rows": int(rows),
        "cols": int(cols),
        "num_tiles_saved": int(saved_count),
        "tile_sizes": {
            "row_heights": [int(h) for h in cell_hs],
            "col_widths": [int(w) for w in cell_ws],
        },
        "tile_filenames": filenames,
    }

    with open(out_path / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    return out_path, metadata
