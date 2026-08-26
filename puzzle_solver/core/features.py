import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np


def extract_borders(piece: np.ndarray, strip_width: int = 1) -> Dict[int, np.ndarray]:
    """Extract border feature strips from each side (0:Top, 1:Right, 2:Bottom, 3:Left).

    Constructs a 6-channel feature tensor per border containing:
    - 3 Channels LAB Color Representation
    - 1 Channel Gradient Magnitude (Sobel)
    - 1 Channel Gradient Direction/Phase (Sobel)
    - 1 Channel Edge Roughness/Curvature (Laplacian)

    Args:
        piece: Tile image array (BGR format).
        strip_width: Boundary strip width in pixels.

    Returns:
        Dictionary mapping side indices [0, 1, 2, 3] to their respective feature arrays.
    """
    lab = cv2.cvtColor(piece, cv2.COLOR_BGR2LAB).astype(np.float32)
    h, w = lab.shape[:2]
    sw = max(1, min(strip_width, h // 2, w // 2))

    def make_grad_patch_enhanced(patch_lab: np.ndarray) -> np.ndarray:
        patch_bgr = cv2.cvtColor(patch_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
        patch_gray = cv2.cvtColor(patch_bgr.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
        patch_gray = cv2.GaussianBlur(patch_gray, (3, 3), 0)
        gx = cv2.Sobel(patch_gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(patch_gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = cv2.magnitude(gx, gy)[..., None]
        grad_dir = cv2.phase(gx, gy, angleInDegrees=True)[..., None]
        lap = cv2.Laplacian(patch_gray, cv2.CV_32F)[..., None]
        return np.concatenate([patch_lab, grad_mag, grad_dir, lap], axis=2)

    return {
        0: make_grad_patch_enhanced(lab[0:sw, :, :]),
        1: make_grad_patch_enhanced(lab[:, w - sw : w, :]),
        2: make_grad_patch_enhanced(lab[h - sw : h, :, :]),
        3: make_grad_patch_enhanced(lab[:, 0:sw, :]),
    }


def normalize_strip_2d(strip: np.ndarray) -> np.ndarray:
    """Normalize each channel of a border feature strip to zero mean and unit variance."""
    arr = strip.astype(np.float32).copy()
    for ch in range(arr.shape[2]):
        m, sd = arr[..., ch].mean(), arr[..., ch].std()
        arr[..., ch] = (arr[..., ch] - m) / (sd if sd > 1e-6 else 1.0)
    return arr


def border_distance_2d(
    strip_a: np.ndarray,
    strip_b: np.ndarray,
    side_a: int,
    side_b: int,
    p: float = 0.3,
    q: float = 1.0 / 16.0,
    w_color: float = 0.4,
    w_grad_mag: float = 0.2,
    w_grad_dir: float = 0.2,
    w_lap: float = 0.4,
) -> float:
    """Calculate the non-linear distance between two opposing border feature strips.

    Args:
        strip_a: Feature strip of piece A.
        strip_b: Feature strip of piece B.
        side_a: Border orientation index for piece A.
        side_b: Border orientation index for piece B.
        p: Non-linear Minkowski exponent for channel differences.
        q: Outer compression exponent.
        w_color: Weight for LAB color difference.
        w_grad_mag: Weight for gradient magnitude difference.
        w_grad_dir: Weight for gradient phase difference.
        w_lap: Weight for Laplacian second-derivative difference.

    Returns:
        Scalar compatibility distance (lower values denote higher compatibility).
    """
    def orient(strip: np.ndarray, side: int) -> np.ndarray:
        return normalize_strip_2d(np.transpose(strip, (1, 0, 2)) if side in (1, 3) else strip)

    a, b = orient(strip_a, side_a), orient(strip_b, side_b)
    if a.size == 0 or b.size == 0:
        return 1e9

    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_LINEAR)

    d_color = np.sum(np.abs(a[..., :3] - b[..., :3]) ** p)
    d_grad_mag = np.sum(np.abs(a[..., 3:4] - b[..., 3:4]) ** p)
    d_grad_dir = np.sum(np.abs(a[..., 4:5] - b[..., 4:5]) ** p)
    d_lap = np.sum(np.abs(a[..., 5:8] - b[..., 5:8]) ** p)

    total = w_color * d_color + w_grad_mag * d_grad_mag + w_grad_dir * d_grad_dir + w_lap * d_lap
    return float(total ** (q / p))


def build_compatibility(pieces: List[np.ndarray], strip_width: int = 1) -> Dict[int, np.ndarray]:
    """Compute pairwise border distance matrices across all spatial orientations.

    Returns a dictionary mapping side index [0: Top, 1: Right, 2: Bottom, 3: Left]
    to an (N x N) compatibility distance matrix.
    """
    n = len(pieces)
    borders = [extract_borders(p, strip_width) for p in pieces]
    compat = {s: np.full((n, n), 1e9, dtype=np.float32) for s in range(4)}

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            compat[0][i, j] = border_distance_2d(borders[i][0], borders[j][2], 0, 2)
            compat[1][i, j] = border_distance_2d(borders[i][1], borders[j][3], 1, 3)
            compat[2][i, j] = border_distance_2d(borders[i][2], borders[j][0], 2, 0)
            compat[3][i, j] = border_distance_2d(borders[i][3], borders[j][1], 3, 1)

    return compat


def load_tiles_from_phase1(source_root: Union[str, Path], category: str, identifier: str) -> List[Dict[str, Any]]:
    """Load enhanced tile images from Phase 1 directory for a given puzzle image.

    Args:
        source_root: Base path to Phase 1 outputs.
        category: Puzzle category directory (e.g. 'puzzle_4x4').
        identifier: Puzzle image ID (e.g. '0').

    Returns:
        List of dictionaries containing tile id, img (np.ndarray), and path.
    """
    source_root = Path(source_root)
    resource_dir = source_root / category / identifier / "tiles"
    if not resource_dir.is_dir():
        raise FileNotFoundError(f"Tiles directory not found: {resource_dir}")

    metadata_path = source_root / category / identifier / "metadata.json"
    meta: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except Exception:
            meta = {}

    def _load_one(img_path: Union[str, Path]) -> Optional[Dict[str, Any]]:
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        base = Path(img_path).stem
        return {"id": base, "img": img, "path": str(img_path)}

    if isinstance(meta, dict) and "tile_filenames" in meta:
        ordered = []
        for fn in meta["tile_filenames"]:
            if not isinstance(fn, str) or not fn.lower().endswith(".png"):
                continue
            if any(tag in fn for tag in ["_mask", "_contours", "_inv"]):
                continue
            candidate = resource_dir / fn
            if candidate.exists():
                item = _load_one(candidate)
                if item is not None:
                    ordered.append(item)
        if len(ordered) > 0:
            return ordered

    all_pngs = sorted([f for f in os.listdir(resource_dir) if f.lower().endswith(".png")])
    numeric_pattern = re.compile(r"tile_(\d+)_(\d+)\.png$", flags=re.IGNORECASE)

    tiles = []
    for fn in all_pngs:
        m = numeric_pattern.match(fn)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            tiles.append((r, c, fn))

    if tiles:
        tiles.sort(key=lambda x: (x[0], x[1]))
        loaded = []
        for r, c, fn in tiles:
            candidate = resource_dir / fn
            item = _load_one(candidate)
            if item is not None:
                loaded.append(item)
        if loaded:
            return loaded

    raise RuntimeError(f"No tiles found in {resource_dir}")
