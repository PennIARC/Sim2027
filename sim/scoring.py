"""Official Mission 10 scorer (matches fexl.com/iarc/draw) and path text format.

Score = 150000 * W / ((1 + B) * L * (1 + 7*A + 100*N)), 0 if any mine lies on the centerline or the path is invalid.

Verified against fexl (see tests/fixtures/fexl/):
- The path starts OFF the field at (start_col, row -1); the first move (normally `U,n`) enters row 0. Every move
  lands on one square, so squares visited == total moves, L = 2 ft * moves, and finishing needs 150 up-moves.
- W = 2 * (1 + 2*min(G, 20)) ft: fexl caps G at 20, so W <= 82 ft (G=21 and G=30 give the same W and buffer as G=20). Buffer = Manhattan distance <= G of any centerline square.
- fexl's B also counts centerline mines, but that case scores 0 anyway; here B excludes the centerline.
Grids are [row, col] = [y, x] (see sim/geometry.py); fexl path text speaks (x, y) = (col, row).
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import distance_transform_cdt

from sim.geometry import SQ_COLS, SQ_FT, SQ_ROWS

_DIRS = {"U": (1, 0), "D": (-1, 0), "L": (0, -1), "R": (0, 1)}  # (drow, dcol)
SCORE_NUM = 150000.0
MAX_G = 20  # fexl caps the green half-width at 20 (G=20, 21, 30 all give W=82 ft and the same buffer)


@dataclass
class Path:
    start_col: int
    G: int
    moves: list = field(default_factory=list)  # [(dir, n), ...], dir in "UDLR"

    # ---- text format -------------------------------------------------------------------------------------
    def to_text(self) -> str:
        return f"S,{self.start_col},{self.G}\n" + "".join(f"{d},{n}\n" for d, n in self.moves)

    @classmethod
    def from_text(cls, text: str) -> "Path":
        """Strict parse of fexl's format. fexl is lenient: it silently ignores malformed lines (lowercase moves, spaces
        after commas, a bad S line falls back to x=0) and a `U,0` ends all processing, so a typo yields an empty path
        with score 0 instead of an error. We raise ValueError instead so bugs surface."""
        lines = [ln for ln in text.strip().splitlines() if ln.strip()]
        if not lines:
            raise ValueError("empty path")
        m = re.fullmatch(r"S,(\d+),(\d+)", lines[0].strip())
        if not m:
            raise ValueError(f"bad start line {lines[0]!r}")
        moves = []
        for ln in lines[1:]:
            m2 = re.fullmatch(r"([UDLR]),(\d+)", ln.strip())
            if not m2:
                raise ValueError(f"bad move line {ln!r}")
            moves.append((m2.group(1), int(m2.group(2))))
        return cls(int(m.group(1)), int(m.group(2)), moves)

    @classmethod
    def from_squares(cls, squares, G: int) -> "Path":
        """Build from a (row, col) centerline starting on row 0 (run-length encodes; first move is U,1)."""
        r0, c0 = squares[0]
        moves, prev = [("U", 1)], (r0, c0)
        for r, c in squares[1:]:
            d = {(1, 0): "U", (-1, 0): "D", (0, -1): "L", (0, 1): "R"}[(r - prev[0], c - prev[1])]
            if moves[-1][0] == d:
                moves[-1] = (d, moves[-1][1] + 1)
            else:
                moves.append((d, 1))
            prev = (r, c)
        return cls(int(c0), int(G), moves)

    # ---- geometry ----------------------------------------------------------------------------------------
    def trace(self):
        """Walk like fexl. Returns (squares, status): squares = in-bounds (row, col) visited in order, stopping
        before a revisit; status flags out_of_bounds / revisit / n_moves / ends_on_far_row."""
        r, c = -1, self.start_col
        seen, sq = set(), []
        oob = revisit = False
        n_moves = 0
        for d, n in self.moves:
            dr, dc = _DIRS[d]
            for _ in range(n):
                r, c = r + dr, c + dc
                n_moves += 1
                if not (0 <= r < SQ_ROWS and 0 <= c < SQ_COLS):
                    oob = True
                    continue
                if (r, c) in seen:
                    revisit = True
                    break
                seen.add((r, c))
                sq.append((r, c))
            if revisit:
                break
        ends = bool(sq) and sq[-1][0] == SQ_ROWS - 1 and not oob and not revisit
        return sq, dict(out_of_bounds=oob, revisit=revisit, n_moves=n_moves, ends_on_far_row=ends)

    def squares(self) -> list:
        return self.trace()[0]

    def validate(self) -> tuple[bool, str]:
        if not (0 <= self.start_col < SQ_COLS):
            return False, "start column out of bounds"
        if self.G < 0:
            return False, "negative G"
        if any(d not in _DIRS or n < 1 for d, n in self.moves):
            return False, "bad move"
        sq, st = self.trace()
        if st["out_of_bounds"]:
            return False, "out of bounds"
        if st["revisit"]:
            return False, "revisits a square"
        if not st["ends_on_far_row"]:
            return False, "does not end on the far row"
        return True, ""


def centerline_mask(centerline) -> np.ndarray:
    m = np.zeros((SQ_ROWS, SQ_COLS), bool)
    if len(centerline):
        a = np.asarray(centerline)
        m[a[:, 0], a[:, 1]] = True
    return m


def buffer_mask(centerline, G: int) -> np.ndarray:
    """Squares within Manhattan distance G of any centerline square, excluding the centerline. (150, 40) bool."""
    cm = centerline_mask(centerline)
    if not cm.any():
        return cm
    # chessboard=False/taxicab metric: distance of every cell to the nearest centerline cell
    dist = distance_transform_cdt(~cm, metric="taxicab")
    return (dist <= min(G, MAX_G)) & ~cm


def width_ft(G: int) -> float:
    return SQ_FT * (1 + 2 * min(G, MAX_G))


def score_formula(W, B, L, A=7, N=0) -> float:
    return SCORE_NUM * W / ((1 + B) * L * (1 + 7 * A + 100 * N))


def score(path: Path, mines: np.ndarray, A: float = 7, N: float = 0) -> dict:
    """Exact official score on a known (150, 40) bool mine grid."""
    ok, reason = path.validate()
    sq, st = path.trace()
    L = SQ_FT * len(sq)  # == 2 ft * moves when valid
    W = width_ft(path.G)
    cm = centerline_mask(sq)
    on_path = int((mines & cm).sum())
    B = int((mines & buffer_mask(sq, path.G)).sum()) if sq else 0
    if ok and on_path == 0 and L > 0:
        s = score_formula(W, B, L, A, N)
    else:
        s = 0.0
        if ok and on_path:
            reason = "mine on centerline"
    return dict(score=s, W=W, L=L, B=B, on_path_mines=on_path, valid=ok, reason=reason)


def fexl_url(path: Path, num_danger: int, seed: int, A: float = 7, N: float = 0) -> str:
    q = urllib.parse.urlencode({"num_danger": num_danger, "seed": seed, "scale": 5, "scan_time": A,
                                "overweight": N, "path_text": path.to_text()}, quote_via=urllib.parse.quote)
    return "https://fexl.com/iarc/draw/?" + q
