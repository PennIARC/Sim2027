"""End-of-episode safe-path planner: expected-score-maximising iterative shortest path on a mine-probability grid.

Input P: per-square mine probability (150, 40) in [0, 1] (see sim/geometry.belief_to_squares_prob).
For each green half-width G and each risk weight lam_buffer, find a min-cost centerline from ANY row-0 square to ANY
row-149 square (4-connected, virtual source/sink), with additive per-step cost
    1                                   length (each step = 1 move = 2 ft of L)
  + lam_center * -log(1 - P[sq])        risk of stepping on a mine (exactly -log P(survive centerline), additive)
  + lam_buffer * newmass(move)          expected number of NEW mines swept into the green buffer by this move
Squares with P > p_block are impassable. Every candidate is then scored by its exact expected official score and the
best is returned.

Buffer-mass approximation: the buffer is the union of Manhattan diamonds (radius G) around centerline squares, which is
not additive. Moving one square in direction d adds exactly the diamond's leading edge (the 2G+1 cells at offsets
(a, b) with a+|b| = G on the side of d) when the path continues straight, so the cost of a move is the P-sum over that
edge, E_d[v], precomputed per direction with one correlation per (G, d). This is exact along straight runs; at turns
it double-counts roughly half an edge (cells already covered by earlier diamonds), and tight zig-zags/loops are not
corrected. First squares (entered from the virtual source) pay the full diamond sum. The sweep over lam_buffer absorbs
the bias, and the final choice uses the exact expectation, so approximation error only affects which candidates exist.

Expected score (exact under independent Bernoulli(P) squares): the buffer excludes the centerline, so
    E[score] = P(no centerline mine) * 150000*W / (L*(1+7A+100N)) * E[1/(1+B)],  B ~ PoissonBinomial(buffer p's),
    E[1/(1+B)] = integral_0^1 prod_s(1 - p_s*u) du   (probability generating function identity),
evaluated by Gauss-Legendre quadrature with a u = v^3 substitution on probabilities binned to 1/400 (bin-mean p).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import correlate
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sim.geometry import SQ_COLS, SQ_ROWS, SQ_FT
from sim.scoring import Path, buffer_mask

SPEC_DEFAULTS = {
    "planner_g_min": 0,
    "planner_g_max": 20,  # fexl's MAX_G
    "planner_p_block": 0.5,  # centerline squares with P above this are impassable
    "planner_lam_center": 200.0,  # ~ expected path length in steps: a centerline hit zeroes the whole score
    "planner_lam_buffer": (1.0, 3.0, 8.0, 20.0, 60.0, 200.0),  # ~ L/(1+B): value of one buffer mine in steps; sweep = tolerance sweep
    "planner_A": 7.0,
    "planner_N": 0.0,
}

NODES = SQ_ROWS * SQ_COLS
SRC = NODES
_NB = [(1, 0), (-1, 0), (0, 1), (0, -1)]  # U, D, R, L as (drow, dcol); index = direction id; id 4 = source edge


def _build_graph():
    r, c = np.divmod(np.arange(NODES), SQ_COLS)
    src, dst, dr = [], [], []
    for k, (a, b) in enumerate(_NB):
        r2, c2 = r + a, c + b
        ok = (r2 >= 0) & (r2 < SQ_ROWS) & (c2 >= 0) & (c2 < SQ_COLS)
        src.append(np.flatnonzero(ok))
        dst.append(r2[ok] * SQ_COLS + c2[ok])
        dr.append(np.full(ok.sum(), k))
    src.append(np.full(SQ_COLS, SRC))
    dst.append(np.arange(SQ_COLS))
    dr.append(np.full(SQ_COLS, 4))
    src, dst, dr = map(np.concatenate, (src, dst, dr))
    order = np.argsort(src, kind="stable")
    src, dst, dr = src[order], dst[order], dr[order]
    indptr = np.searchsorted(src, np.arange(NODES + 2))
    return dst.astype(np.int32), dr, indptr.astype(np.int32)


_DST, _DIR, _INDPTR = _build_graph()
_GL_X, _GL_W = np.polynomial.legendre.leggauss(96)
_V = 0.5 * (_GL_X + 1)  # nodes on [0,1]
_U = _V ** 3
_UW = 0.5 * _GL_W * 3 * _V ** 2  # weights for u = v^3
_BIN = 400


def _kernels(G: int):
    """Correlation kernels: leading edge for U, D, R, L and the full diamond."""
    n = 2 * G + 1
    ks = [np.zeros((n, n)) for _ in range(4)]
    full = np.zeros((n, n))
    for b in range(-G, G + 1):
        a = G - abs(b)
        ks[0][G + a, G + b] = 1
        ks[1][G - a, G + b] = 1
        ks[2][G + b, G + a] = 1  # offsets (a=b, b=a): right edge
        ks[3][G + b, G - a] = 1
    for a in range(-G, G + 1):
        for b in range(-G, G + 1):
            if abs(a) + abs(b) <= G:
                full[G + a, G + b] = 1
    return ks, full


def _edge_mass(P: np.ndarray, G: int) -> np.ndarray:
    """(5, NODES): new buffer mass of entering node v by direction id 0..3 (U,D,R,L), 4 = from source (full diamond)."""
    out = np.zeros((5, NODES))
    if G == 0:
        return out
    ks, full = _kernels(G)
    for k in range(4):
        out[k] = correlate(P, ks[k], mode="constant", cval=0.0).ravel()
    out[4] = correlate(P, full, mode="constant", cval=0.0).ravel()
    return out


def expected_inv_one_plus_b(p: np.ndarray) -> tuple[float, float]:
    """(E[1/(1+B)], E[B]) for B = sum of independent Bernoulli(p_i)."""
    p = np.asarray(p, float).ravel()
    p = p[p > 0]
    if p.size == 0:
        return 1.0, 0.0
    mean = float(p.sum())
    idx = np.minimum((p * _BIN).astype(int), _BIN)
    cnt = np.bincount(idx, minlength=_BIN + 1)
    psum = np.bincount(idx, weights=p, minlength=_BIN + 1)
    nz = cnt > 0
    pm = psum[nz] / cnt[nz]
    # log prod(1 - p u) = sum_k cnt_k log(1 - pm_k u); p == 1 gives log(1-u): integrable singularity at u=1 (v=1 node excluded)
    logs = np.log1p(-np.clip(np.outer(pm, _U), 0, 1 - 1e-12))
    val = float(np.exp(cnt[nz] @ logs) @ _UW)
    return val, mean


def expected_score(centerline, G: int, P: np.ndarray, A: float = 7.0, N: float = 0.0) -> dict:
    """Exact expected official score of a centerline (list of (row, col)) under independent Bernoulli(P) squares."""
    cl = np.asarray(centerline)
    p_c = np.clip(P[cl[:, 0], cl[:, 1]], 0, 1)
    p_surv = float(np.prod(1 - p_c))
    buf = buffer_mask(centerline, G)
    inv, eb = expected_inv_one_plus_b(P[buf])
    L = SQ_FT * len(cl)
    W = SQ_FT * (1 + 2 * G)
    base = 150000.0 * W / (L * (1 + 7 * A + 100 * N))
    return dict(expected_score=p_surv * base * inv, E_B=eb, p_survive=p_surv, E_inv=inv, L=L, W=W)


def plan(P: np.ndarray, spec: dict | None = None) -> tuple[Path, dict]:
    """Return (best Path, diagnostics). Raises RuntimeError only if every start->far-row route is blocked
    even after relaxing p_block (then falls back to the min-P route)."""
    t0 = time.perf_counter()
    s = {**SPEC_DEFAULTS, **(spec or {})}
    P = np.clip(np.asarray(P, float), 0.0, 1.0)
    assert P.shape == (SQ_ROWS, SQ_COLS)
    cc = (-np.log1p(-np.minimum(P, 1 - 1e-9))).ravel()
    blocked = (P > s["planner_p_block"]).ravel()
    lam_c = float(s["planner_lam_center"])
    lams = tuple(s["planner_lam_buffer"])
    A, N = s["planner_A"], s["planner_N"]
    BIG = 1e7
    last_row = np.arange((SQ_ROWS - 1) * SQ_COLS, NODES)
    seen, cands = {}, 0
    best = None
    for G in range(int(s["planner_g_min"]), int(s["planner_g_max"]) + 1):
        em = _edge_mass(P, G)
        em_edge = em[_DIR, _DST]  # per-edge new buffer mass
        base_w = 1.0 + lam_c * cc[_DST] + np.where(blocked[_DST], BIG, 0.0)
        for lam_b in (lams if G > 0 else lams[:1]):
            data = base_w + lam_b * em_edge
            g = csr_matrix((data, _DST, _INDPTR), shape=(NODES + 1, NODES + 1))
            dist, pred = dijkstra(g, directed=True, indices=SRC, return_predecessors=True)
            end = last_row[np.argmin(dist[last_row])]
            if dist[end] >= BIG:
                continue
            sq, v = [], int(end)
            while v != SRC:
                sq.append(divmod(v, SQ_COLS))
                v = int(pred[v])
            sq.reverse()
            cands += 1
            key = (G, tuple(sq))
            if key in seen:
                continue
            ev = expected_score(sq, G, P, A, N)
            seen[key] = ev["expected_score"]
            if best is None or ev["expected_score"] > best[0]["expected_score"]:
                best = (ev, G, sq, lam_b)
    if best is None:  # everything blocked: retry with nothing blocked
        return plan(P, {**s, "planner_p_block": 2.0})
    ev, G, sq, lam_b = best
    path = Path.from_squares(sq, G)
    diag = dict(ev, G=G, lam_buffer=lam_b, n_candidates=cands, n_unique=len(seen), runtime_s=time.perf_counter() - t0,
                n_moves=len(sq))
    return path, diag
