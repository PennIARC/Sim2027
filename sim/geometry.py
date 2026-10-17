"""Field geometry and coordinate conventions shared by every sim module. Matches fexl.com/iarc/draw.

World coordinates are in feet: x across the field (0..80, left to right), y along it (0..300, from the
start edge y=0 to the far edge y=300). Up/north = +y.

Grids are numpy arrays indexed [row, col] = [y, x], row 0 at the start edge:
- squares: the official 2 ft scoring grid, shape (SQ_ROWS, SQ_COLS) = (150, 40). One mine at most per square.
  fexl's path.txt uses (x, y) = (col, row) with (0, 0) the lower-left square.
- belief: the 1 ft sensor/belief grid, shape (BEL_ROWS, BEL_COLS) = (300, 80); BEL_PER_SQ x BEL_PER_SQ
  belief cells per square.
- blocks: the allocation grid the policy picks from, shape (BLK_ROWS, BLK_COLS) = (30, 8) of 10 ft blocks,
  flattened row-major: block id = row * BLK_COLS + col.
"""
import numpy as np

FIELD_W_FT, FIELD_L_FT = 80.0, 300.0
SQ_FT = 2.0
SQ_COLS, SQ_ROWS = 40, 150
BEL_PER_SQ = 2
BEL_FT = SQ_FT / BEL_PER_SQ
BEL_COLS, BEL_ROWS = SQ_COLS * BEL_PER_SQ, SQ_ROWS * BEL_PER_SQ
BLK_FT = 10.0
BLK_COLS, BLK_ROWS = int(FIELD_W_FT / BLK_FT), int(FIELD_L_FT / BLK_FT)
N_BLOCKS = BLK_COLS * BLK_ROWS


def square_of(x_ft: float, y_ft: float) -> tuple[int, int]:
    """World point -> (row, col) of its scoring square, clamped to the field."""
    return (int(np.clip(y_ft // SQ_FT, 0, SQ_ROWS - 1)), int(np.clip(x_ft // SQ_FT, 0, SQ_COLS - 1)))


def square_center(row: int, col: int) -> tuple[float, float]:
    """(row, col) -> world (x, y) of the square's center."""
    return ((col + 0.5) * SQ_FT, (row + 0.5) * SQ_FT)


def block_bounds(block: int) -> tuple[float, float, float, float]:
    """Block id -> world (x0, y0, x1, y1)."""
    r, c = divmod(block, BLK_COLS)
    return (c * BLK_FT, r * BLK_FT, (c + 1) * BLK_FT, (r + 1) * BLK_FT)


def belief_to_squares_prob(p_belief: np.ndarray) -> np.ndarray:
    """Per-belief-cell mine probabilities (300, 80) -> per-square probability (150, 40).

    A square holds at most one mine, so the belief cells of a square are mutually exclusive locations:
    P(square has a mine) = sum of its cells' probabilities, clipped to 1.
    """
    s = p_belief.reshape(SQ_ROWS, BEL_PER_SQ, SQ_COLS, BEL_PER_SQ).sum(axis=(1, 3))
    return np.minimum(s, 1.0)
