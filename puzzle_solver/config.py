from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


@dataclass
class GridDimension:
    """Represents the row and column dimensions of a puzzle grid."""
    rows: int
    cols: int

    @property
    def total_tiles(self) -> int:
        return self.rows * self.cols


@dataclass
class TileData:
    """Encapsulates image tile data and associated identifiers."""
    id: str
    img: np.ndarray
    path: str = ""

    def __post_init__(self):
        if self.img is None or not isinstance(self.img, np.ndarray):
            raise ValueError(f"Tile {self.id} must contain a valid numpy image ndarray.")


@dataclass
class SolverConfig:
    """Hyperparameters and configuration options for the puzzle solver."""
    strip_width: int = 1
    p: float = 0.3
    q: float = 1.0 / 16.0
    w_color: float = 0.4
    w_grad_mag: float = 0.2
    w_grad_dir: float = 0.2
    w_lap: float = 0.4
    seeds: Optional[int] = None
    shifter_iters: int = 8
    beam_width: Optional[int] = None
    time_limit: float = 60.0


@dataclass
class SolverResult:
    """Structured output returned after puzzle reassembly execution."""
    placement_map: Dict[str, int]
    order: List[int]
    score: float
    method: str
    time: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "placement_map": self.placement_map,
            "order": self.order,
            "score": float(self.score),
            "method": self.method,
            "time": float(self.time),
        }
