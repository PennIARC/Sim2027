"""Stateless baseline allocators with the same signature as sim.allocator_env.allocate-style callables:
    fn(inp: AllocatorInput) -> Allocation
and `run_episode` to drive the env with any such callable (no PPO).

Factories take (spec) and return the callable: random_alloc / lawnmower_alloc / greedy_alloc are the callables themselves
(spec = keyword argument, default env defaults), `make_baseline(name, spec)` binds the spec.
"""
import functools

import numpy as np

from sim import planner as planner_mod
from sim.geometry import BLK_COLS, BLK_ROWS, N_BLOCKS, SQ_COLS, SQ_ROWS, belief_to_squares_prob, block_bounds
from sim.interfaces import Allocation, AllocatorInput
from sim.scoring import centerline_mask, buffer_mask

_K = 4
# planner settings used inside greedy_alloc: a lighter sweep than the end-of-episode planner (speed; it only needs a rough path)
GREEDY_PLANNER = {"planner_g_max": 10, "planner_lam_buffer": (3.0, 20.0)}
DONE_SEEN = 0.4  # a block whose mean seen confidence exceeds this counts as covered (lawnmower)


def _k(spec) -> int:
    return int((spec or {}).get("alloc_k", _K))


def _block_mean(x: np.ndarray) -> np.ndarray:
    """(300, 80) belief-grid map -> (240,) mean per 10 ft block."""
    return x.reshape(BLK_ROWS, 10, BLK_COLS, 10).mean(axis=(1, 3)).reshape(-1)


def _block_centers() -> np.ndarray:
    r, c = np.divmod(np.arange(N_BLOCKS), BLK_COLS)
    return np.stack([(c + 0.5) * 10.0, (r + 0.5) * 10.0], 1)


_CENTERS = _block_centers()


def random_alloc(inp: AllocatorInput, spec: dict | None = None, seed: int = 0) -> Allocation:
    """K random unmasked blocks. Stateless: the RNG is derived from (seed, drone, time)."""
    rng = np.random.default_rng([seed, int(round(inp.t * 10)), inp.drone_id])
    free = np.flatnonzero(inp.block_mask)
    return Allocation(inp.drone_id, [int(b) for b in rng.choice(free, _k(spec), replace=False)])


def lawnmower_alloc(inp: AllocatorInput, spec: dict | None = None) -> Allocation:
    """Drone i owns block columns 2i, 2i+1 (for 4 drones) and sweeps them front (y=0) to back: the K lowest-row unmasked,
    not-yet-covered blocks of its lane. If its lane has fewer than K left it helps out on the least-covered blocks elsewhere."""
    k = _k(spec)
    n_drones = len(inp.drones)
    per = max(BLK_COLS // n_drones, 1)
    cols = np.arange(N_BLOCKS) % BLK_COLS
    rows = np.arange(N_BLOCKS) // BLK_COLS
    lane = (cols // per).clip(max=n_drones - 1) == inp.drone_id
    cover = _block_mean(inp.seen)
    ok = inp.block_mask & (cover < DONE_SEEN)
    mine = np.flatnonzero(ok & lane)
    order = mine[np.lexsort((cols[mine], rows[mine]))][:k]
    picks = [int(b) for b in order]
    if len(picks) < k:  # lane done: take the least-covered free blocks anywhere (front first on ties)
        rest = np.flatnonzero(inp.block_mask)
        rest = rest[~np.isin(rest, picks)]
        rest = rest[np.lexsort((rows[rest], cover[rest]))]
        picks += [int(b) for b in rest[:k - len(picks)]]
    return Allocation(inp.drone_id, picks)


def greedy_alloc(inp: AllocatorInput, spec: dict | None = None, dist_weight: float = 0.3, centerline_weight: float = 3.0,
                 planner_spec: dict | None = None) -> Allocation:
    """Look where the current safe path is least certain. Plan on the snapshot's belief (squares), weight every square in the
    path's green buffer by 1 (centerline and its 2-square surroundings by `centerline_weight`), and score each block by
    mean_over_its_squares[ weight * u ], u = 0.5 (1 - seen) + 0.5 min(1, P(1-P)/0.03) (unseen fraction + normalized uncertainty).
    Pick the top K unmasked blocks after subtracting dist_weight * (distance to the drone / 300 ft)."""
    k = _k(spec)
    P = belief_to_squares_prob(inp.p_mine)
    ps = {**(spec or {}), **GREEDY_PLANNER, **(planner_spec or {})}
    path, _ = planner_mod.plan(P, ps)
    sq, _ = path.trace()
    w = buffer_mask(sq, path.G).astype(np.float32)
    cm = centerline_mask(sq)
    near = np.zeros_like(cm)
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if abs(dr) + abs(dc) <= 2:
                near |= np.roll(np.roll(cm, dr, 0), dc, 1)
    w = np.maximum(w, 0.0) + (centerline_weight - 1.0) * near
    seen_sq = inp.seen.reshape(SQ_ROWS, 2, SQ_COLS, 2).mean(axis=(1, 3))
    u = 0.5 * (1.0 - seen_sq) + 0.5 * np.minimum(1.0, P * (1.0 - P) / 0.03)
    score = (w * u).reshape(BLK_ROWS, 5, BLK_COLS, 5).mean(axis=(1, 3)).reshape(-1)
    pos = np.asarray(inp.drones[inp.drone_id].pose_xy, float)
    dist = np.linalg.norm(_CENTERS - pos, axis=1)
    eff = np.where(inp.block_mask, score - dist_weight * dist / 300.0 * score.max(), -np.inf)
    return Allocation(inp.drone_id, [int(b) for b in np.argsort(-eff, kind="stable")[:k]])


BASELINES = {"random": random_alloc, "lawnmower": lawnmower_alloc, "greedy": greedy_alloc}


def make_baseline(name: str, spec: dict | None = None, **kw):
    """Baseline callable AllocatorInput -> Allocation with the spec (and e.g. seed=) bound."""
    return functools.partial(BASELINES[name], spec=spec, **kw)


def run_episode(allocator_fn, seed: int, spec: dict | None = None, layout: np.ndarray | None = None,
                progress: float | None = None) -> dict:
    """Run one full episode with any allocator callable AllocatorInput -> Allocation. Returns info["episode"].
    progress sets the GPS-drift curriculum point (None = leave at the env default 0.0; evaluation uses 1.0)."""
    from sim.allocator_env import AllocatorEnv
    env = AllocatorEnv(spec or {})
    if progress is not None:
        env.set_progress(progress)
    env.reset(seed=seed, layout=layout)
    while True:
        alloc = allocator_fn(env.last_input)
        _, _, term, trunc, info = env.step(np.array(alloc.blocks))
        if term or trunc:
            return info["episode"]
