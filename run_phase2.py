#!/usr/bin/env python3
"""Phase 2 runner: Best-Buddies puzzle reassembly."""

import argparse
from pathlib import Path
from puzzle_solver.pipeline import run_phase2_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Phase 2 Best-Buddies Puzzle Solver")
    parser.add_argument("--phase1_root", default="phase1_outputs", help="Phase 1 outputs directory")
    parser.add_argument("--out_dir", default="phase2_outputs", help="Output directory for assembled puzzles")
    parser.add_argument("--group", required=False, help="Specific puzzle group (e.g. puzzle_2x2)")
    parser.add_argument("--image", required=False, help="Specific image ID")
    parser.add_argument("--time_limit", type=float, default=60.0, help="Time limit per puzzle (seconds)")
    parser.add_argument("--dataset_root", default="dataset_images", help="Raw dataset directory")
    parser.add_argument("--workers", type=int, default=None, help="Max worker processes")
    args = parser.parse_args()

    run_phase2_pipeline(
        phase1_root=Path(args.phase1_root),
        out_root=Path(args.out_dir),
        group=args.group,
        image=args.image,
        time_limit=args.time_limit,
        dataset_root=Path(args.dataset_root),
        max_workers=args.workers
    )


if __name__ == "__main__":
    main()
