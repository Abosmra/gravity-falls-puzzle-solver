#!/usr/bin/env python3
"""
Gravity Falls Puzzle Solver - Unified Entrypoint
Usage:
    python main.py all
    python main.py phase1 --dataset dataset_images --out phase1_outputs
    python main.py phase2 --phase1-dir phase1_outputs --out phase2_outputs --group puzzle_2x2
    python main.py gui
"""

import sys
from puzzle_solver.__main__ import main

if __name__ == "__main__":
    main(sys.argv[1:])
