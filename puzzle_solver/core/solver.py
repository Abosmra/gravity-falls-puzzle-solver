from collections import deque
from itertools import permutations
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from puzzle_solver.config import SolverConfig, SolverResult
from .features import build_compatibility


def opposite_side(side: int) -> int:
    """Return the opposing border orientation index (0->2, 1->3, 2->0, 3->1)."""
    return (side + 2) % 4


def best_partner_for(i: int, side: int, compat: Dict[int, np.ndarray]) -> int:
    """Identify the piece with the minimum border distance on a specified side."""
    return int(np.argmin(compat[side][i]))


def is_best_buddy(i: int, side: int, j: int, compat: Dict[int, np.ndarray]) -> bool:
    """Evaluate whether piece i and piece j are mutually best matching neighbors."""
    if i == j:
        return False
    bj = best_partner_for(i, side, compat)
    if bj != j:
        return False
    opp = opposite_side(side)
    bi = best_partner_for(j, opp, compat)
    return bi == i


def place_pieces(
    n: int,
    grid_n: int,
    compat: Dict[int, np.ndarray],
    seed_placement: Optional[Dict[int, int]] = None,
    seed_center: bool = True
) -> List[int]:
    """Place pieces onto the grid prioritizing mutual best-buddy relationships."""
    placement = [-1] * n
    used = [False] * n

    if seed_placement:
        seed_pos = list(seed_placement.keys())
        rs = [p // grid_n for p in seed_pos]
        cs = [p % grid_n for p in seed_pos]
        rmin, rmax = min(rs), max(rs)
        cmin, cmax = min(cs), max(cs)
        seed_h = rmax - rmin + 1
        seed_w = cmax - cmin + 1
        top = (grid_n - seed_h) // 2 if seed_center else 0
        left = (grid_n - seed_w) // 2 if seed_center else 0
        for pos_old, pid in seed_placement.items():
            r_old, c_old = pos_old // grid_n, pos_old % grid_n
            r_new = top + (r_old - rmin)
            c_new = left + (c_old - cmin)
            if 0 <= r_new < grid_n and 0 <= c_new < grid_n:
                pos_new = r_new * grid_n + c_new
                placement[pos_new] = pid
                used[pid] = True
    else:
        seed_pid = np.random.randint(0, n)
        seed_pos = np.random.choice(range(n))
        placement[seed_pos] = seed_pid
        used[seed_pid] = True

    def get_neighbors(pos: int) -> List[Tuple[int, int]]:
        r, c = pos // grid_n, pos % grid_n
        neighbors = []
        if r > 0 and placement[pos - grid_n] != -1:
            neighbors.append((pos - grid_n, 2))
        if r < grid_n - 1 and placement[pos + grid_n] != -1:
            neighbors.append((pos + grid_n, 0))
        if c > 0 and placement[pos - 1] != -1:
            neighbors.append((pos - 1, 1))
        if c < grid_n - 1 and placement[pos + 1] != -1:
            neighbors.append((pos + 1, 3))
        return neighbors

    slots_filled = sum(1 for x in placement if x != -1)
    while slots_filled < n:
        empty_slots = []
        for pos in range(n):
            if placement[pos] != -1:
                continue
            neighs = get_neighbors(pos)
            if neighs:
                empty_slots.append((-len(neighs), pos, neighs))
        if not empty_slots:
            pos = placement.index(-1)
            empty_slots = [(0, pos, [])]

        empty_slots.sort()
        chosen = None

        for _, slot_pos, neighs in empty_slots:
            candidates = []
            for pid in range(n):
                if used[pid]:
                    continue
                bb_count = 0
                compat_sum = 0.0
                for neigh_pos, neigh_side in neighs:
                    neigh_pid = placement[neigh_pos]
                    if is_best_buddy(neigh_pid, neigh_side, pid, compat):
                        bb_count += 1
                    compat_sum += compat[neigh_side][neigh_pid, pid]
                if bb_count > 0:
                    candidates.append((bb_count, compat_sum, slot_pos, pid))
            if candidates:
                candidates.sort(key=lambda x: (-x[0], x[1]))
                chosen = candidates[0]
                break

        if chosen is None:
            _, slot_pos, neighs = empty_slots[0]
            best_val = 1e18
            best_pid = 0
            for pid in range(n):
                if used[pid]:
                    continue
                ssum = 0.0
                for neigh_pos, neigh_side in neighs:
                    neigh_pid = placement[neigh_pos]
                    ssum += compat[neigh_side][neigh_pid, pid]
                avg = ssum / max(1, len(neighs))
                if avg < best_val:
                    best_val = avg
                    best_pid = pid
            chosen = (0, best_val, slot_pos, best_pid)

        _, _, slot_pos, chosen_pid = chosen
        placement[slot_pos] = chosen_pid
        used[chosen_pid] = True
        slots_filled += 1

    return placement


def segment_placement(placement: List[int], grid_n: int, compat: Dict[int, np.ndarray]) -> List[List[int]]:
    """Partition placed pieces into connected components linked by mutual best-buddy relations."""
    n_slots = len(placement)
    visited = [False] * n_slots
    segments = []

    def neighbors(pos: int):
        r, c = pos // grid_n, pos % grid_n
        if c > 0:
            yield pos - 1, 3
        if c < grid_n - 1:
            yield pos + 1, 1
        if r > 0:
            yield pos - grid_n, 0
        if r < grid_n - 1:
            yield pos + grid_n, 2

    for pos in range(n_slots):
        if visited[pos]:
            continue
        queue = deque([pos])
        comp = []
        visited[pos] = True
        while queue:
            u = queue.popleft()
            comp.append(u)
            pu = placement[u]
            for v, side_of_u in neighbors(u):
                if visited[v]:
                    continue
                pv = placement[v]
                if is_best_buddy(pu, side_of_u, pv, compat):
                    visited[v] = True
                    queue.append(v)
        if comp:
            segments.append(comp)
    return segments


def compute_best_buddies_score(placement: List[int], grid_n: int, compat: Dict[int, np.ndarray]) -> float:
    """Calculate the ratio of adjacent grid edges satisfying mutual best-buddy pairing."""
    n_slots = len(placement)
    bb_count = 0
    total_adj = 0
    for pos in range(n_slots):
        r = pos // grid_n
        c = pos % grid_n
        pid = placement[pos]
        if c < grid_n - 1:
            np0 = pos + 1
            pid2 = placement[np0]
            total_adj += 1
            if is_best_buddy(pid, 1, pid2, compat):
                bb_count += 1
        if r < grid_n - 1:
            np1 = pos + grid_n
            pid2 = placement[np1]
            total_adj += 1
            if is_best_buddy(pid, 2, pid2, compat):
                bb_count += 1
    if total_adj == 0:
        return 0.0
    return float(bb_count / total_adj)


def shift_placement(
    initial_placement: List[int],
    grid_n: int,
    compat: Dict[int, np.ndarray],
    max_iters: int = 8,
    swap_pass: bool = True
) -> Tuple[List[int], float]:
    """Iteratively refine grid placement through segment re-anchoring and 2-opt swaps."""
    n_slots = len(initial_placement)
    current = initial_placement.copy()
    best_score = compute_best_buddies_score(current, grid_n, compat)

    for _ in range(max_iters):
        segments = segment_placement(current, grid_n, compat)
        if not segments:
            break
        segments.sort(key=lambda x: -len(x))
        improved = False

        for seg in segments:
            if not seg:
                continue
            seed_map = {pos: current[pos] for pos in seg}
            placement_new = place_pieces(n_slots, grid_n, compat, seed_placement=seed_map)
            score_new = compute_best_buddies_score(placement_new, grid_n, compat)
            if score_new > best_score + 1e-9:
                current = placement_new
                best_score = score_new
                improved = True
                break

        if not improved and swap_pass:
            for pos1 in range(n_slots):
                for pos2 in range(pos1 + 1, n_slots):
                    new_p = current.copy()
                    new_p[pos1], new_p[pos2] = new_p[pos2], new_p[pos1]
                    score_swap = compute_best_buddies_score(new_p, grid_n, compat)
                    if score_swap > best_score + 1e-9:
                        current = new_p
                        best_score = score_swap
                        improved = True
                        break
                if improved:
                    break

        if not improved:
            break

    return current, float(best_score)


def solve_bruteforce(pieces: List[np.ndarray], compat: Dict[int, np.ndarray], grid_n: int) -> Tuple[List[int], float]:
    """Perform exhaustive branch-and-bound permutation search for small grids (2x2)."""
    n = grid_n * grid_n
    best_perm = None
    best_score = 1e12
    for perm in permutations(range(n)):
        score = 0.0
        valid = True
        for pos, pid in enumerate(perm):
            r = pos // grid_n
            c = pos % grid_n
            if c > 0:
                left_pid = perm[pos - 1]
                score += compat[1][left_pid, pid]
                if score >= best_score:
                    valid = False
                    break
            if r > 0:
                top_pid = perm[pos - grid_n]
                score += compat[2][top_pid, pid]
                if score >= best_score:
                    valid = False
                    break
        if not valid:
            continue
        if score < best_score:
            best_score = score
            best_perm = perm
    return list(best_perm) if best_perm is not None else list(range(n)), float(best_score)


class PuzzleSolver:
    """Autonomous square jigsaw puzzle solver utilizing multi-cue border matching and best-buddies."""

    def __init__(
        self,
        tiles: List[Dict[str, Any]],
        rows: int,
        cols: int,
        config: Optional[SolverConfig] = None,
        strip_width: int = 1,
        seeds: Optional[int] = None,
        shifter_iters: int = 8,
        beam_width: Optional[int] = None
    ):
        self.tiles = tiles
        self.rows = rows
        self.cols = cols
        self.n = len(tiles)
        if rows * cols != self.n:
            raise ValueError(f"Grid {rows}x{cols} doesn't match {self.n} tiles")
        self.grid_n = rows

        self.config = config or SolverConfig(
            strip_width=strip_width,
            seeds=seeds,
            shifter_iters=shifter_iters,
            beam_width=beam_width
        )

        self.strip_width = self.config.strip_width
        if self.config.seeds is None:
            self.seeds = {2: 5, 4: 10, 8: 20}.get(self.grid_n, 10)
        else:
            self.seeds = self.config.seeds

        self.shifter_iters = self.config.shifter_iters
        if self.config.beam_width is None:
            self.beam_width = {2: 1, 4: 3, 8: 5}.get(self.grid_n, 3)
        else:
            self.beam_width = max(1, self.config.beam_width)

        self.pieces = [t["img"] for t in tiles]

    def solve(self, time_limit: Optional[float] = None, beam_width: int = 0) -> Optional[Dict[str, Any]]:
        """Execute puzzle solving and return optimal tile placement configuration."""
        start_time = time.time()
        effective_limit = time_limit if time_limit is not None else self.config.time_limit
        effective_beam = beam_width if beam_width > 0 else self.beam_width

        compat = build_compatibility(self.pieces, strip_width=self.strip_width)

        if self.grid_n == 2:
            order, _ = solve_bruteforce(self.pieces, compat, self.grid_n)
            placement = order
            best_bb = compute_best_buddies_score(order, self.grid_n, compat)
        else:
            candidates: List[Tuple[float, List[int]]] = []
            best_placement = None
            best_bb = -1.0
            seed_id = 0

            while seed_id < self.seeds and (time.time() - start_time) < effective_limit:
                init_placement = place_pieces(self.n, self.grid_n, compat, seed_placement=None)
                bb0 = compute_best_buddies_score(init_placement, self.grid_n, compat)
                placement_after_shifter, bb_sh = shift_placement(
                    init_placement, self.grid_n, compat, max_iters=self.shifter_iters
                )
                final_placement = placement_after_shifter if bb_sh >= bb0 else init_placement
                final_bb = bb_sh if bb_sh >= bb0 else bb0
                candidates.append((final_bb, final_placement))
                if final_bb > best_bb:
                    best_bb = final_bb
                    best_placement = final_placement
                seed_id += 1

            if best_bb < 0.60 or (time.time() - start_time) < effective_limit:
                for _ in range(5):
                    if (time.time() - start_time) >= effective_limit:
                        break
                    init_placement = place_pieces(self.n, self.grid_n, compat, seed_placement=None)
                    bb0 = compute_best_buddies_score(init_placement, self.grid_n, compat)
                    placement_after_shifter, bb_sh = shift_placement(
                        init_placement, self.grid_n, compat, max_iters=self.shifter_iters
                    )
                    final_placement = placement_after_shifter if bb_sh >= bb0 else init_placement
                    final_bb = bb_sh if bb_sh >= bb0 else bb0
                    candidates.append((final_bb, final_placement))
                    if final_bb > best_bb:
                        best_bb = final_bb
                        best_placement = final_placement

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                candidates = candidates[:effective_beam]
                best_bb, best_placement = candidates[0]

            placement = best_placement if best_placement is not None else init_placement

        placement_map: Dict[str, int] = {}
        for pos, pid in enumerate(placement):
            r = pos // self.grid_n
            c = pos % self.grid_n
            placement_map[f"{r}_{c}"] = pid

        elapsed = time.time() - start_time
        return {
            "placement_map": placement_map,
            "order": placement,
            "score": float(best_bb),
            "method": "best_buddies",
            "time": elapsed,
        }
