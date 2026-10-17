"""Mine-layout generator fitted to the organizers' fexl.com/iarc/draw generator.

Model (see tools/fit_mines.py for the evidence): a log-Gaussian Cox process. Per layout a random intensity field
    lambda(row, col) = profile(row) * exp(sigma * G - sigma^2 / 2),   G = unit-variance anisotropic Gaussian field
is drawn, then `n` distinct squares are sampled without replacement with probability proportional to lambda.
fexl's layouts for a given seed are deterministic and nested in num_danger (the first 60 mines of the 135 layout are
the 60-mine layout), i.e. the field is fixed by the seed and mines are drawn one by one from it.

Grid convention (sim/geometry.py): bool (150, 40) indexed [row=y, col=x], row 0 = start edge.
"""
import heapq
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, label

from sim.geometry import SQ_COLS, SQ_ROWS

_PARAMS_FILE = Path(__file__).with_name("mine_params.json")


def _load_params() -> dict:
    return json.loads(_PARAMS_FILE.read_text())


_P = _load_params()

# Fitted defaults (sim/mine_params.json is the source of truth; tools/fit_mines.py rewrites it).
SPEC_DEFAULTS: dict = {
    "mines_count": list(_P["count_range"]),        # [lo, hi] inclusive; uniform integer per episode
    "mines_field_sigma": _P["field_sigma"],        # log-intensity std of the random field (clumpiness)
    "mines_field_len_rows": _P["field_len_rows"],  # Gaussian-smoothing length of the field along y, squares
    "mines_field_len_cols": _P["field_len_cols"],  # ... along x, squares
    "mines_row_knots": list(_P["row_knots"]),      # relative density at rows 0, step, 2*step ... (linear interp)
    "mines_row_step": _P["row_step"],
    "mines_clear_path": True,                      # guarantee a mine-free 4-connected start->far edge path
    "mines_prior_samples": 2000,                   # Monte Carlo layouts for prior_density
}


def _spec(spec: dict | None) -> dict:
    return {**SPEC_DEFAULTS, **(spec or {})}


def row_profile(s: dict) -> np.ndarray:
    knots = np.asarray(s["mines_row_knots"], float)
    return np.interp(np.arange(SQ_ROWS), np.arange(len(knots)) * s["mines_row_step"], knots)


def intensity(rng: np.random.Generator, s: dict) -> np.ndarray:
    """Random per-square intensity field (150, 40), mean ~ row_profile."""
    ly, lx, sig = s["mines_field_len_rows"], s["mines_field_len_cols"], s["mines_field_sigma"]
    prof = row_profile(s)[:, None]
    if sig <= 0:
        return np.broadcast_to(prof, (SQ_ROWS, SQ_COLS)).copy()
    py, px = int(3 * ly) + 1, int(3 * lx) + 1
    z = rng.standard_normal((SQ_ROWS + 2 * py, SQ_COLS + 2 * px))
    g = gaussian_filter(z, (ly, lx), mode="wrap")[py:py + SQ_ROWS, px:px + SQ_COLS]
    g *= np.sqrt(4 * np.pi * ly * lx)  # unit variance
    return prof * np.exp(sig * g - 0.5 * sig * sig)


def _draw(rng, lam, n) -> np.ndarray:
    """n distinct squares ~ without-replacement sampling proportional to lam (Gumbel top-k)."""
    flat = lam.ravel()
    n = min(int(n), int((flat > 0).sum()))
    keys = np.log(np.maximum(flat, 1e-300)) + rng.gumbel(size=flat.size)
    idx = np.argpartition(-keys, n - 1)[:n] if n > 0 else np.empty(0, int)
    out = np.zeros(flat.size, bool)
    out[idx] = True
    return out.reshape(lam.shape)


def has_clear_path(mines: np.ndarray) -> bool:
    lab, _ = label(~mines)  # 4-connectivity by default
    a, b = lab[0][lab[0] > 0], lab[-1][lab[-1] > 0]
    return bool(a.size and b.size and np.intersect1d(a, b).size)


def _min_cost_path(mines: np.ndarray) -> list[tuple[int, int]]:
    """Dijkstra from row 0 to the last row, cost 1 per mine square entered (+tiny per step)."""
    R, C = mines.shape
    dist = np.full((R, C), np.inf)
    prev = {}
    pq = []
    for c in range(C):
        dist[0, c] = mines[0, c] + 1e-3
        heapq.heappush(pq, (dist[0, c], 0, c))
    while pq:
        d, r, c = heapq.heappop(pq)
        if d > dist[r, c]:
            continue
        if r == R - 1:
            path = [(r, c)]
            while (r, c) in prev:
                r, c = prev[(r, c)]
                path.append((r, c))
            return path
        for rr, cc in ((r + 1, c), (r - 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < R and 0 <= cc < C:
                nd = d + mines[rr, cc] + 1e-3
                if nd < dist[rr, cc]:
                    dist[rr, cc] = nd
                    prev[(rr, cc)] = (r, c)
                    heapq.heappush(pq, (nd, rr, cc))
    return []


def _repair(rng, mines, lam) -> np.ndarray:
    """Open the cheapest start->far path by moving its mines to other squares drawn from lam (count preserved)."""
    path = _min_cost_path(mines)
    on_path = np.zeros_like(mines)
    for r, c in path:
        on_path[r, c] = True
    moved = int((mines & on_path).sum())
    mines = mines & ~on_path
    lam2 = np.where(mines | on_path, 0.0, lam)
    return mines | _draw(rng, lam2, moved)


def generate(rng: np.random.Generator, spec: dict | None = None) -> np.ndarray:
    """One random minefield: bool (150, 40), True = mine. Fully determined by `rng` and `spec`."""
    s = _spec(spec)
    lo, hi = s["mines_count"]
    n = int(rng.integers(lo, hi + 1))
    lam = intensity(rng, s)
    mines = _draw(rng, lam, n)
    if s["mines_clear_path"] and not has_clear_path(mines):
        mines = _repair(rng, mines, lam)
    return mines


def _spec_key(s: dict) -> str:
    return json.dumps(s, sort_keys=True, default=list)


@lru_cache(maxsize=8)
def _prior(key: str) -> np.ndarray:
    s = json.loads(key)
    rng = np.random.default_rng(12345)
    acc = np.zeros((SQ_ROWS, SQ_COLS))
    k = int(s["mines_prior_samples"])
    for _ in range(k):
        acc += generate(rng, s)
    p = acc / k
    # smooth the MC noise lightly in x (profile is column-stationary by construction), keep exact row means
    p = np.broadcast_to(p.mean(axis=1, keepdims=True), p.shape).copy()
    p.setflags(write=False)
    return p


def prior_density(spec: dict | None = None) -> np.ndarray:
    """Expected P(mine) per square (150, 40) under the generator (Monte Carlo, cached per spec; read-only).

    The field is column-stationary, so the prior depends on the row only (averaged over columns to cut MC noise).
    """
    return _prior(_spec_key(_spec(spec)))


_EVAL_DIR = Path(__file__).resolve().parents[1] / "data" / "fexl_eval"


def load_eval_set(name: str = "fexl") -> list[tuple[str, np.ndarray]]:
    """Scraped fexl layouts as [(id, bool (150, 40))]; id = 'n<num_danger>_s<seed>'."""
    if name != "fexl":
        raise ValueError(f"unknown eval set {name!r}")
    z = np.load(_EVAL_DIR / "fexl_layouts.npz")
    return [(str(i), m.copy()) for i, m in zip(z["ids"], z["layouts"])]
