import argparse
import os
from pathlib import Path
import sys
import threading
import time

from puzzle_solver.core.tiling import detect_grid_from_folder
from puzzle_solver.pipeline import solve_image, solve_dataset, extract_tiles, reassemble_puzzles
from puzzle_solver.ui.viewer import launch_gui


def parse_cli_args(args=None):
    parser = argparse.ArgumentParser(
        prog="gravity-falls-solver",
        description="Gravity Falls Image Puzzle Solver & Reassembly Pipeline"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: all
    subparsers.add_parser("all", help="Run background solver and launch visual inspector GUI")

    # Command: solve (Unified end-to-end command)
    solve_parser = subparsers.add_parser("solve", help="Solve an image or full dataset end-to-end")
    solve_parser.add_argument("--image", help="Path to a single image to slice and solve")
    solve_parser.add_argument("--grid", default="2x2", help="Grid dimensions for single image (e.g. 2x2, 4x4, 8x8)")
    solve_parser.add_argument("--dataset", default="dataset_images", help="Dataset directory if running batch solving")
    solve_parser.add_argument("--out", default="output", help="Output directory or output image path")
    solve_parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit per puzzle (seconds)")
    solve_parser.add_argument("--workers", type=int, default=None, help="Worker processes for batch execution")

    # Command: gui
    gui_parser = subparsers.add_parser("gui", help="Launch interactive visual inspector GUI")
    gui_parser.add_argument("--tiles-dir", default="output/tiles", help="Directory containing tiles")
    gui_parser.add_argument("--solved-dir", default="output/solved", help="Directory containing solved puzzles")

    # Command: extract (Tile extraction only)
    ext_parser = subparsers.add_parser("extract", help="Extract & enhance puzzle tiles")
    ext_parser.add_argument("--dataset", default="dataset_images", help="Path to input dataset directory")
    ext_parser.add_argument("--out", default="output/tiles", help="Output directory for tiles")
    ext_parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")

    # Command: reassemble (Reassembly from tiles)
    rea_parser = subparsers.add_parser("reassemble", help="Reassemble puzzle tiles using Best-Buddies solver")
    rea_parser.add_argument("--tiles-dir", default="output/tiles", help="Tiles directory")
    rea_parser.add_argument("--out", default="output/solved", help="Output directory for solved puzzles")
    rea_parser.add_argument("--group", required=False, help="Specific puzzle group (e.g. puzzle_2x2)")
    rea_parser.add_argument("--image", required=False, help="Specific image ID")
    rea_parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit per puzzle (seconds)")
    rea_parser.add_argument("--dataset", default="dataset_images", help="Raw dataset directory")
    rea_parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")

    return parser.parse_args(args)


def run_all_workflow(dataset_root: str = "dataset_images", output_root: str = "output"):
    print("=" * 60)
    print("Gravity Falls Puzzle Solver - Unified Launcher")
    print("=" * 60)

    out_p = Path(output_root)
    tiles_p = out_p / "tiles"
    solved_p = out_p / "solved"

    tiles_exist = tiles_p.is_dir() and any(tiles_p.iterdir())
    solved_exist = solved_p.is_dir() and any(solved_p.iterdir())

    if not tiles_exist:
        print("[INFO] Tiles not detected. Extracting in background...")
        t1 = threading.Thread(target=lambda: extract_tiles(dataset_root, tiles_p), daemon=True)
        t1.start()
    else:
        print("[INFO] Tiles already present.")

    if not solved_exist:
        print("[INFO] Solved puzzles not detected. Reassembling in background...")
        time.sleep(1)
        os.environ["RUN_ALL_CONTEXT"] = "1"
        t2 = threading.Thread(target=lambda: reassemble_puzzles(tiles_p, solved_p, dataset_root=dataset_root), daemon=True)
        t2.start()
    else:
        print("[INFO] Solved puzzles already present.")

    print("\n[INFO] Launching Visual Inspector GUI...")
    launch_gui(tiles_dir=str(tiles_p), solved_dir=str(solved_p))


def main(args=None):
    parsed = parse_cli_args(args)

    if parsed.command == "solve":
        if parsed.image:
            r, c = detect_grid_from_folder(parsed.grid)
            if r is None:
                r, c = 2, 2
            print(f"[INFO] Solving single image: {parsed.image} ({r}x{c})")
            _, res = solve_image(parsed.image, r, c, output_path=parsed.out, time_limit=parsed.time_limit)
            print(f"[SUCCESS] Solved in {res['time']:.2f}s | Score: {res['score']:.3f} | Method: {res['method']}")
        else:
            solve_dataset(
                dataset_path=parsed.dataset,
                output_dir=parsed.out,
                time_limit=parsed.time_limit,
                max_workers=parsed.workers
            )
    elif parsed.command == "extract":
        extract_tiles(
            input_dataset_path=parsed.dataset,
            output_tiles_path=parsed.out,
            max_workers=parsed.workers
        )
    elif parsed.command == "reassemble":
        reassemble_puzzles(
            tiles_root=parsed.tiles_dir,
            out_root=parsed.out,
            group=parsed.group,
            image=parsed.image,
            time_limit=parsed.time_limit,
            dataset_root=parsed.dataset,
            max_workers=parsed.workers
        )
    elif parsed.command == "gui":
        launch_gui(tiles_dir=parsed.tiles_dir, solved_dir=parsed.solved_dir)
    elif parsed.command == "all" or parsed.command is None:
        run_all_workflow()


if __name__ == "__main__":
    main()
