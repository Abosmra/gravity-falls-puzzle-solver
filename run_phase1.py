#!/usr/bin/env python3
"""Phase 1 runner: Image preprocessing and tile slicing."""

import argparse
from pathlib import Path
from puzzle_solver.pipeline import run_phase1_pipeline


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 Preprocessing and Tile Extraction")
    parser.add_argument("--dataset", default="dataset_images", help="Path to input dataset directory")
    parser.add_argument("--out", default="phase1_outputs", help="Output directory for generated tiles")
    parser.add_argument("--workers", type=int, default=None, help="Max worker processes")
    args = parser.parse_args()

    print("[INFO] Running Phase 1")
    run_phase1_pipeline(
        input_dataset_path=Path(args.dataset),
        output_base_path=Path(args.out),
        max_workers=args.workers
    )
    print("[INFO] Phase 1 complete")


if __name__ == "__main__":
    main()
