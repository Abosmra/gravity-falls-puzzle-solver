from puzzle_solver.core.solver import (
    PuzzleSolver,
    opposite_side,
    best_partner_for,
    is_best_buddy,
    place_pieces,
    segment_placement,
    compute_best_buddies_score,
    shift_placement,
    solve_bruteforce,
)
from puzzle_solver.core.features import (
    extract_borders as _extract_borders,
    normalize_strip_2d as _normalize_strip_2d,
    border_distance_2d as _border_distance_2d,
    build_compatibility as _build_compatibility,
)

# Aliases for backward compatibility
_opposite = opposite_side
_best_partner_for = best_partner_for
_is_best_buddy = is_best_buddy
_placer = place_pieces
_segmenter = segment_placement
_compute_best_buddies_score = compute_best_buddies_score
_shifter = shift_placement
_solve_bruteforce = solve_bruteforce

__all__ = [
    'PuzzleSolver',
    'opposite_side',
    'best_partner_for',
    'is_best_buddy',
    'place_pieces',
    'segment_placement',
    'compute_best_buddies_score',
    'shift_placement',
    'solve_bruteforce',
    '_extract_borders',
    '_normalize_strip_2d',
    '_border_distance_2d',
    '_build_compatibility',
    '_opposite',
    '_best_partner_for',
    '_is_best_buddy',
    '_placer',
    '_segmenter',
    '_compute_best_buddies_score',
    '_shifter',
    '_solve_bruteforce',
]
