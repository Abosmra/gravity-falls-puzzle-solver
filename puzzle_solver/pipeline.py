from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

from puzzle_solver.core.tiling import detect_grid_from_folder, preprocess_image, smart_enhance
from puzzle_solver.core.features import load_tiles
from puzzle_solver.core.solver import PuzzleSolver
from puzzle_solver.core.assembly import assemble_puzzle


def solve_image(
    image_path: Union[str, Path],
    rows: int,
    cols: int,
    output_path: Optional[Union[str, Path]] = None,
    time_limit: float = 60.0
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """End-to-end unified solver: slices an image into tiles, solves the puzzle, and reassembles it."""
    image_path = Path(image_path)
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Cannot read image: {image_path}")

    h_img, w_img = img.shape[:2]
    base_w = w_img // cols
    rem_w = w_img % cols
    cell_ws = [base_w + (1 if c < rem_w else 0) for c in range(cols)]

    base_h = h_img // rows
    rem_h = h_img % rows
    cell_hs = [base_h + (1 if r < rem_h else 0) for r in range(rows)]

    x_offsets = [0]
    for w_c in cell_ws[:-1]:
        x_offsets.append(x_offsets[-1] + w_c)

    y_offsets = [0]
    for h_r in cell_hs[:-1]:
        y_offsets.append(y_offsets[-1] + h_r)

    tiles = []
    for r in range(rows):
        for c in range(cols):
            x1, y1 = x_offsets[c], y_offsets[r]
            w_t, h_t = cell_ws[c], cell_hs[r]
            tile_raw = img[y1 : y1 + h_t, x1 : x1 + w_t].copy()
            avg_size = (w_t + h_t) // 2
            tile_enh = smart_enhance(tile_raw, tile_size=avg_size)
            tiles.append({"id": f"tile_{r:02d}_{c:02d}", "img": tile_enh, "path": ""})

    solver = PuzzleSolver(tiles, rows, cols)
    result = solver.solve(time_limit=time_limit)
    if result is None:
        raise RuntimeError(f"Solver failed to find a valid arrangement for {image_path}")

    assembled = assemble_puzzle(tiles, result["placement_map"], rows, cols, output_path=output_path)
    return assembled, result


def iter_groups(root_path: Union[str, Path], specific: Optional[str] = None) -> Generator[str, None, None]:
    root_p = Path(root_path)
    if specific:
        yield specific
        return
    if not root_p.is_dir():
        return
    for item in sorted(os.listdir(root_p)):
        full_path = root_p / item
        if full_path.is_dir() and item.startswith("puzzle_"):
            yield item


def iter_images(
    root_path: Union[str, Path],
    group_name: str,
    specific: Optional[str] = None
) -> Generator[str, None, None]:
    root_p = Path(root_path)
    group_path = root_p / group_name
    if specific:
        if (group_path / specific).is_dir():
            yield specific
        return

    if group_path.is_dir():
        for item in sorted(os.listdir(group_path), key=lambda x: int(x) if x.isdigit() else x):
            item_path = group_path / item
            if item_path.is_dir():
                yield item


def _extract_worker(image_path: Path, input_dataset_path: Path, output_base_path: Path):
    folder_name = image_path.parent.name
    grid_rows, grid_cols = detect_grid_from_folder(folder_name)
    if grid_rows is None:
        return ("skip", image_path, None, None, None)

    relative_path = image_path.parent.relative_to(input_dataset_path)
    output_directory = output_base_path / relative_path / image_path.stem

    output_path, metadata = preprocess_image(
        image_path,
        output_directory,
        grid_rows,
        grid_cols,
    )

    saved_tile_count = metadata.get("num_tiles_saved", metadata.get("num_pieces_saved", "unknown"))
    detected_tile_count = metadata.get("num_tiles_detected", metadata.get("num_pieces_detected", "unknown"))
    return ("done", image_path, output_path, detected_tile_count, saved_tile_count)


def extract_tiles(
    input_dataset_path: Union[str, Path] = "dataset_images",
    output_tiles_path: Union[str, Path] = "output/tiles",
    max_workers: Optional[int] = None
) -> Dict[str, int]:
    """Extract and enhance puzzle tiles from raw dataset images."""
    input_dataset_path = Path(input_dataset_path)
    output_tiles_path = Path(output_tiles_path)
    output_tiles_path.mkdir(parents=True, exist_ok=True)

    image_file_list = [
        fp for fp in input_dataset_path.rglob("*")
        if fp.suffix.lower() in (".png", ".jpg", ".jpeg")
    ]
    if not image_file_list:
        print(f"[ERROR] No images found in: {input_dataset_path}")
        return {"total": 0, "processed": 0, "skipped": 0, "failed": 0}

    workers = max_workers or max(1, os.cpu_count() or 1)
    print(f"[INFO] Using {workers} workers for tile extraction")

    processed = 0
    skipped = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_extract_worker, img_p, input_dataset_path, output_tiles_path): img_p
            for img_p in image_file_list
        }

        for fut in as_completed(futures):
            img_p = futures[fut]
            try:
                status, image_path, out_path, detected_count, saved_count = fut.result()
            except Exception as exc:
                print(f"[ERROR] {img_p} failed: {exc}")
                failed += 1
                continue

            if status == "skip":
                print(f"[SKIP] {image_path} (folder does not contain 2x2/4x4/8x8)")
                skipped += 1
                continue

            print(f"[DONE] -> {out_path} | detected={detected_count} saved={saved_count}")
            processed += 1

    return {"total": len(image_file_list), "processed": processed, "skipped": skipped, "failed": failed}


def _reassemble_worker(
    tiles_root: Union[str, Path],
    group: str,
    image: str,
    out_root: Union[str, Path],
    time_limit: float
) -> bool:
    try:
        rows, cols = detect_grid_from_folder(group)
        if rows is None:
            print(f"[{group}/{image}] SKIP - Cannot infer grid size")
            return False

        tiles = load_tiles(Path(tiles_root) / group / image)
        solver = PuzzleSolver(tiles, rows, cols)
        result = solver.solve(time_limit=time_limit)

        if result is None:
            print(f"[{group}/{image}] FAILED - No solution found")
            return False

        placement = result["placement_map"]
        score = result.get("score", 0.0)
        method = result.get("method", "unknown")

        output_dir = Path(out_root) / group
        output_dir.mkdir(parents=True, exist_ok=True)
        assembled_path = output_dir / f"{image}.png"

        assemble_puzzle(tiles, placement, rows, cols, assembled_path)
        print(f"[{group}/{image}] SUCCESS - Score: {score:.3f}, Method: {method}")
        return True
    except Exception as e:
        print(f"[{group}/{image}] ERROR - {e}")
        traceback.print_exc()
        return False


def reassemble_puzzles(
    tiles_root: Union[str, Path] = "output/tiles",
    out_root: Union[str, Path] = "output/solved",
    group: Optional[str] = None,
    image: Optional[str] = None,
    time_limit: float = 60.0,
    dataset_root: Union[str, Path] = "dataset_images",
    max_workers: Optional[int] = None
) -> Dict[str, int]:
    """Reassemble puzzles using Best-Buddies solver."""
    tiles_root = Path(tiles_root)
    # Check fallback legacy path if output/tiles is empty
    if not tiles_root.exists() and Path("phase1_outputs").exists():
        tiles_root = Path("phase1_outputs")

    out_root = Path(out_root)
    dataset_root = Path(dataset_root)

    tasks = [
        (grp, img)
        for grp in iter_groups(tiles_root, group)
        for img in iter_images(tiles_root, grp, image)
    ]

    if not tasks:
        print("[INFO] Tile outputs not found; extracting tiles now...")
        extract_tiles(dataset_root, tiles_root, max_workers=max_workers)
        tasks = [
            (grp, img)
            for grp in iter_groups(tiles_root, group)
            for img in iter_images(tiles_root, grp, image)
        ]

    if not tasks:
        print("[WARN] No puzzles found to process.")
        return {"total": 0, "successes": 0, "failures": 0}

    workers = max_workers or max(1, os.cpu_count() or 1)
    if os.environ.get("RUN_ALL_CONTEXT"):
        workers = max(1, workers - 2)

    print(f"[INFO] Using {workers} workers for puzzle reassembly")
    successes = 0
    failures = 0

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_reassemble_worker, tiles_root, grp, img, out_root, time_limit): (grp, img)
            for (grp, img) in tasks
        }

        for fut in as_completed(futures):
            grp, img = futures[fut]
            try:
                success = fut.result()
            except Exception as exc:
                print(f"[{grp}/{img}] ERROR - {exc}")
                traceback.print_exc()
                failures += 1
                continue

            if success:
                successes += 1
            else:
                failures += 1

    print(f"\n{'=' * 70}")
    print("REASSEMBLY SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total: {successes + failures} | Success: {successes} | Failures: {failures}")
    print(f"{'=' * 70}\n")

    return {"total": successes + failures, "successes": successes, "failures": failures}


def solve_dataset(
    dataset_path: Union[str, Path] = "dataset_images",
    output_dir: Union[str, Path] = "output",
    time_limit: float = 60.0,
    max_workers: Optional[int] = None
) -> Dict[str, Any]:
    """Unified pipeline runner: extracts tiles and reassembles all dataset puzzles."""
    output_p = Path(output_dir)
    tiles_p = output_p / "tiles"
    solved_p = output_p / "solved"

    print(f"[INFO] Step 1: Extracting tiles to {tiles_p}")
    extract_res = extract_tiles(dataset_path, tiles_p, max_workers=max_workers)

    print(f"[INFO] Step 2: Reassembling puzzles to {solved_p}")
    solve_res = reassemble_puzzles(tiles_p, solved_p, time_limit=time_limit, dataset_root=dataset_path, max_workers=max_workers)

    return {"extraction": extract_res, "reassembly": solve_res}
