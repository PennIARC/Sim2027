"""Order K blocks for one drone.

Every block is swept by one straight 10 ft pass through its center along +y, -y, +x or -x (option 0..3, see
OPT_DIRS). The router picks both the visiting order and each block's pass option. Cost of a route =
straight-line travel (current position -> entry, exit -> next entry) + turn_penalty * |turn angle| at each
transition (heading -> leg, leg -> pass direction); in-block pass length is constant and omitted.
Exact for K <= route_brute_max_k (all permutations, DP over the 4 options per block, vectorized, deterministic),
otherwise ant colony optimization over permutations followed by a 2-opt polish.
"""
from itertools import permutations

import numpy as np

from sim.geometry import BLK_FT, block_bounds

SPEC_DEFAULTS = {
    "route_turn_penalty_ft_per_rad": 10.0,
    "route_brute_max_k": 7,
    "route_aco_ants": 24,
    "route_aco_iters": 40,
    "route_aco_alpha": 1.0,
    "route_aco_beta": 3.0,
    "route_aco_rho": 0.15,
}

OPT_DIRS = np.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])


def block_center(block: int) -> np.ndarray:
    x0, y0, x1, y1 = block_bounds(int(block))
    return np.array([(x0 + x1) / 2, (y0 + y1) / 2])


def pass_points(block: int, opt: int):
    """(entry, exit, unit direction) of the in-block pass."""
    d = OPT_DIRS[opt]
    c = block_center(block)
    return c - d * BLK_FT / 2, c + d * BLK_FT / 2, d


def _ang(a, b):
    return np.arccos(np.clip((a * b).sum(-1), -1.0, 1.0))


def _matrices(blocks, pos, heading, pen):
    K = len(blocks)
    E = np.zeros((K, 4, 2)); X = np.zeros((K, 4, 2)); Dn = np.zeros((K, 4, 2))
    for i, b in enumerate(blocks):
        for o in range(4):
            E[i, o], X[i, o], Dn[i, o] = pass_points(b, o)
    E, X, Dn = E.reshape(-1, 2), X.reshape(-1, 2), Dn.reshape(-1, 2)
    leg = E[None, :, :] - X[:, None, :]
    dist = np.linalg.norm(leg, axis=-1)
    ld = leg / np.maximum(dist, 1e-9)[..., None]
    short = dist < 0.5
    po = np.where(short, _ang(Dn[:, None, :], Dn[None, :, :]), _ang(Dn[:, None, :], ld))
    pi = np.where(short, 0.0, _ang(ld, Dn[None, :, :]))
    D = (dist + pen * (po + pi)).reshape(K, 4, K, 4)
    legs = E - np.asarray(pos)[None, :]
    ds = np.linalg.norm(legs, axis=-1)
    lds = legs / np.maximum(ds, 1e-9)[:, None]
    if heading is None:
        S = ds + pen * np.where(ds < 0.5, 0.0, _ang(lds, Dn))
    else:
        h = np.array([np.cos(heading), np.sin(heading)])
        S = ds + pen * np.where(ds < 0.5, _ang(h[None, :], Dn), _ang(h[None, :], lds) + _ang(lds, Dn))
    return S.reshape(K, 4), D


def _perm_costs(S, D, perms):
    """perms (P, K) -> (P,) min cost over pass options."""
    o4 = np.arange(4)
    c = S[perms[:, 0]]                                   # (P, 4)
    for t in range(1, perms.shape[1]):
        a, b = perms[:, t - 1], perms[:, t]
        m = c[:, :, None] + D[a[:, None, None], o4[None, :, None], b[:, None, None], o4[None, None, :]]
        c = m.min(axis=1)
    return c.min(axis=1)


def _best_options(S, D, perm):
    K = len(perm)
    c = S[perm[0]].copy(); back = []
    for t in range(1, K):
        m = c[:, None] + D[perm[t - 1], :, perm[t], :]
        back.append(m.argmin(0)); c = m.min(0)
    o = int(c.argmin()); cost = float(c[o]); opts = [o]
    for bk in reversed(back):
        o = int(bk[o]); opts.append(o)
    return opts[::-1], cost


def _aco(S, D, spec, rng):
    K = S.shape[0]
    A, alpha, beta, rho = spec["route_aco_ants"], spec["route_aco_alpha"], spec["route_aco_beta"], spec["route_aco_rho"]
    eta = 1.0 / (1.0 + D.min(axis=(1, 3))); eta0 = 1.0 / (1.0 + S.min(axis=1))
    tau = np.ones((K, K))
    best, best_c = None, np.inf
    for _ in range(spec["route_aco_iters"]):
        perms = np.zeros((A, K), int)
        for a in range(A):
            left = np.ones(K, bool)
            pr = eta0 ** beta
            for t in range(K):
                pr = pr * left
                j = rng.choice(K, p=pr / pr.sum())
                perms[a, t] = j; left[j] = False
                pr = (tau[j] ** alpha) * (eta[j] ** beta)
        costs = _perm_costs(S, D, perms)
        k = int(costs.argmin())
        if costs[k] < best_c:
            best_c, best = costs[k], perms[k].copy()
        tau *= 1 - rho
        for p, c in ((perms[k], costs[k]), (best, best_c)):
            tau[p[:-1], p[1:]] += 1.0 / (1.0 + c / 100.0)
    # 2-opt polish: reverse every segment, keep improvements until none left
    improved = True
    while improved:
        improved = False
        cand = np.array([np.concatenate([best[:i], best[i:j + 1][::-1], best[j + 1:]])
                         for i in range(K) for j in range(i + 1, K)])
        cc = _perm_costs(S, D, cand)
        if cc.min() < best_c - 1e-9:
            best_c, best, improved = cc.min(), cand[int(cc.argmin())], True
    return best


def order_blocks(blocks, pos, heading=None, spec: dict | None = None, rng: np.random.Generator | None = None):
    """-> (route, cost) with route = [(block_id, pass_option), ...] in visiting order. rng only used when K > brute max."""
    sp = {**SPEC_DEFAULTS, **(spec or {})}
    blocks = list(dict.fromkeys(int(b) for b in blocks))
    K = len(blocks)
    if K == 0:
        return [], 0.0
    S, D = _matrices(blocks, pos, heading, sp["route_turn_penalty_ft_per_rad"])
    if K <= sp["route_brute_max_k"]:
        perms = np.array(list(permutations(range(K))))
        perm = perms[int(np.argmin(_perm_costs(S, D, perms)))]
    else:
        perm = _aco(S, D, sp, rng if rng is not None else np.random.default_rng(0))
    opts, cost = _best_options(S, D, [int(p) for p in perm])
    return [(blocks[int(p)], o) for p, o in zip(perm, opts)], cost
