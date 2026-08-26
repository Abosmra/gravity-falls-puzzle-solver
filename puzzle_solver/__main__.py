import argparse
import os
from pathlib import Path
import sys
import threading
import time

from puzzle_solver.core.tiling import detect_grid_from_folder
from puzzle_solver.pipeline import run_phase1_pipeline, run_phase2_pipeline, solve_image
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
    solve_parser.add_argument("--out", default="phase2_outputs", help="Output path or directory")
    solve_parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit per puzzle (seconds)")
    solve_parser.add_argument("--workers", type=int, default=None, help="Worker processes for batch execution")

    # Command: gui
    gui_parser = subparsers.add_parser("gui", help="Launch interactive visual inspector GUI")
    gui_parser.add_argument("--phase1-dir", default="phase1_outputs", help="Tile directory")
    gui_parser.add_argument("--phase2-dir", default="phase2_outputs", help="Output directory")

    # Command: phase1 (Tile extraction)
    p1_parser = subparsers.add_parser("phase1", help="Extract & enhance puzzle tiles")
    p1_parser.add_argument("--dataset", default="dataset_images", help="Path to input dataset directory")
    p1_parser.add_argument("--out", default="phase1_outputs", help="Output directory for tiles")
    p1_parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")

    # Command: phase2 (Reassembly)
    p2_parser = subparsers.add_parser("phase2", help="Reassemble puzzle tiles using Best-Buddies solver")
    p2_parser.add_argument("--phase1-dir", default="phase1_outputs", help="Tiles directory")
    p2_parser.add_argument("--out", default="phase2_outputs", help="Output directory for solved puzzles")
    p2_parser.add_argument("--group", required=False, help="Specific puzzle group (e.g. puzzle_2x2)")
    p2_parser.add_argument("--image", required=False, help="Specific image ID")
    p2_parser.add_argument("--time-limit", type=float, default=60.0, help="Time limit per puzzle (seconds)")
    p2_parser.add_argument("--dataset", default="dataset_images", help="Raw dataset directory")
    p2_parser.add_argument("--workers", type=int, default=None, help="Number of worker processes")

    return parser.parse_args(args)


def run_all_workflow(phase1_root: str = "phase1_outputs", phase2_root: str = "phase2_outputs", dataset_root: str = "dataset_images"):
    print("=" * 60)
    print("Gravity Falls Puzzle Solver - Unified Launcher")
    print("=" * 60)

    p1_path = Path(phase1_root)
    p2_path = Path(phase2_root)

    p1_has_data = p1_path.is_dir() and any(p1_path.iterdir())
    p2_has_data = p2_path.is_dir() and any(p2_path.iterdir())

    if not p1_has_data:
        print("[INFO] Tile outputs not detected. Extracting tiles in background...")
        t1 = threading.Thread(target=lambda: run_phase1_pipeline(dataset_root, phase1_root), daemon=True)
        t1.start()
    else:
        print("[INFO] Tile outputs already present.")

    if not p2_has_data:
        print("[INFO] Solved puzzles not detected. Reassembling in background...")
        time.sleep(1)
        os.environ["RUN_ALL_CONTEXT"] = "1"
        t2 = threading.Thread(target=lambda: run_phase2_pipeline(phase1_root, phase2_root, dataset_root=dataset_root), daemon=True)
        t2.start()
    else:
        print("[INFO] Solved puzzles already present.")

    print("\n[INFO] Launching Visual Inspector GUI...")
    launch_gui(phase1_root=phase1_root, out_dir=phase2_root)


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
            run_phase2_pipeline(
                phase1_root="phase1_outputs",
                out_root=parsed.out,
                time_limit=parsed.time_limit,
                dataset_root=parsed.dataset,
                max_workers=parsed.workers
            )
    elif parsed.command == "phase1":
        run_phase1_pipeline(
            input_dataset_path=parsed.dataset,
            output_base_path=parsed.out,
            max_workers=parsed.workers
        )
    elif parsed.command == "phase2":
        run_phase2_pipeline(
            phase1_root=parsed.phase1_dir,
            out_root=parsed.out,
            group=parsed.group,
            image=parsed.image,
            time_limit=parsed.time_limit,
            dataset_root=parsed.dataset,
            max_workers=parsed.workers
        )
    elif parsed.command == "gui":
        launch_gui(phase1_root=parsed.phase1_dir, out_dir=parsed.phase2_dir)
    elif parsed.command == "all" or parsed.command is None:
        run_all_workflow()


if __name__ == "__main__":
    main()
