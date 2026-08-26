from .config import GridDimension, SolverConfig, SolverResult, TileData
from .core.tiling import detect_grid_from_folder, smart_enhance, preprocess_image
from .core.features import (
    extract_borders,
    normalize_strip_2d,
    border_distance_2d,
    build_compatibility,
    load_tiles_from_phase1,
)
from .core.solver import (
    PuzzleSolver,
    opposite_side,
    best_partner_for,
    is_best_buddy,
    place_pieces,
    shift_placement,
    solve_bruteforce,
)
from .core.assembly import assemble_puzzle
from .pipeline import run_phase1_pipeline, run_phase2_pipeline, solve_image

__version__ = "1.0.0"

__all__ = [
    "GridDimension",
    "SolverConfig",
    "SolverResult",
    "TileData",
    "detect_grid_from_folder",
    "smart_enhance",
    "preprocess_image",
    "extract_borders",
    "normalize_strip_2d",
    "border_distance_2d",
    "build_compatibility",
    "load_tiles_from_phase1",
    "PuzzleSolver",
    "opposite_side",
    "best_partner_for",
    "is_best_buddy",
    "place_pieces",
    "shift_placement",
    "solve_bruteforce",
    "assemble_puzzle",
    "run_phase1_pipeline",
    "run_phase2_pipeline",
    "solve_image",
]
