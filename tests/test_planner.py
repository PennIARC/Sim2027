import time

import numpy as np
import pytest
from scipy.ndimage import distance_transform_cdt
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from sim.planner import SPEC_DEFAULTS, expected_inv_one_plus_b, expected_score, plan
from sim.scoring import Path, score

R, C = 150, 40


def poisson_binomial_inv(ps):
    pmf = np.zeros(len(ps) + 1)
    pmf[0] = 1
    for q in ps:
        pmf[1:] = pmf[1:] * (1 - q) + pmf[:-1] * q
        pmf[0] *= 1 - q
    return float((pmf / (1 + np.arange(len(pmf)))).sum())


def gridtesting_reference(mines, widths=range(0, 9), tols=range(0, 3)):
    """GridTesting's iterative A*, corrected: any start/end column, plan on the known map, layered shortest path where
    a centerline square within Manhattan `width` of a mine is a violation (at most `tol` allowed), true official score."""
    n = R * C
    r, c = np.divmod(np.arange(n), C)
    es, ed = [], []
    for a, b in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        r2, c2 = r + a, c + b
        ok = (r2 >= 0) & (r2 < R) & (c2 >= 0) & (c2 < C)
        es.append(np.flatnonzero(ok))
        ed.append(r2[ok] * C + c2[ok])
    es, ed = np.concatenate(es), np.concatenate(ed)
    dmine = distance_transform_cdt(~mines, metric="taxicab").ravel()
    best = (0.0, None)
    for w in widths:
        viol = (dmine <= w).astype(int)
        for tol in tols:
            L = tol + 1
            src = [es + l * n for l in range(L)]
            dst = [ed + np.minimum(l + viol[ed], L) * n for l in range(L)]
            keep = [(l + viol[ed] <= tol) & ~mines.ravel()[ed] for l in range(L)]
            s_all = np.concatenate([s[k] for s, k in zip(src, keep)])
            d_all = np.concatenate([d[k] for d, k in zip(dst, keep)])
            S = L * n
            ok0 = ~mines[0] & (viol[:C] <= tol)
            s_all = np.concatenate([s_all, np.full(ok0.sum(), S)])
            d_all = np.concatenate([d_all, np.flatnonzero(ok0) + viol[:C][ok0] * n])
            g = csr_matrix((np.ones(len(s_all)), (s_all, d_all)), shape=(S + 1, S + 1))
            dist, pred = dijkstra(g, indices=S, return_predecessors=True)
            ends = np.array([l * n + (R - 1) * C + cc for l in range(L) for cc in range(C)])
            e = ends[np.argmin(dist[ends])]
            if not np.isfinite(dist[e]):
                continue
            sq, v = [], int(e)
            while v != S:
                sq.append(divmod(v % n, C))
                v = int(pred[v])
            sq.reverse()
            sc = score(Path.from_squares(sq, w), mines)["score"]
            if sc > best[0]:
                best = (sc, (w, tol))
    return best


def test_expected_inv_matches_exact_dp():
    rng = np.random.default_rng(1)
    for ps in [rng.random(300) * 0.05, np.full(1500, 0.02), rng.random(400), np.array([1.0, 0.3, 0.0, 0.5]), np.array([])]:
        exact = poisson_binomial_inv(ps) if len(ps) else 1.0
        got, mean = expected_inv_one_plus_b(ps)
        assert got == pytest.approx(exact, rel=1e-4, abs=1e-9)
        assert mean == pytest.approx(float(np.sum(ps)))


def test_zero_belief_gives_straight_wide_path_and_exact_expected_score():
    path, d = plan(np.zeros((R, C)))
    gmax = SPEC_DEFAULTS["planner_g_max"]
    assert path.G == gmax and len(path.squares()) == R
    assert len({c for _, c in path.squares()}) == 1  # straight
    assert path.validate()[0]
    exact = 150000 * 2 * (1 + 2 * gmax) / (1 * 300 * 50)
    assert d["expected_score"] == pytest.approx(exact)
    assert score(path, np.zeros((R, C), bool))["score"] == pytest.approx(exact)
    assert d["E_B"] == 0 and d["p_survive"] == 1


def test_expected_score_matches_monte_carlo():
    rng = np.random.default_rng(3)
    P = rng.random((R, C)) ** 4 * 0.1
    path, d = plan(P)
    sq = path.squares()
    ev = expected_score(sq, path.G, P)
    assert ev["expected_score"] == pytest.approx(d["expected_score"])
    sims = [score(path, rng.random((R, C)) < P)["score"] for _ in range(4000)]
    se = np.std(sims) / np.sqrt(len(sims))
    assert abs(np.mean(sims) - ev["expected_score"]) < 4 * se + 1e-9


@pytest.mark.parametrize("seed,n", [(0, 30), (1, 60), (2, 100)])
def test_known_mines_beats_gridtesting_reference(seed, n):
    rng = np.random.default_rng(seed)
    mines = np.zeros((R, C), bool)
    mines.ravel()[rng.choice(R * C, n, replace=False)] = True
    path, d = plan(mines.astype(float))
    true = score(path, mines)
    assert true["valid"] and true["on_path_mines"] == 0 and true["score"] > 0
    assert true["score"] == pytest.approx(d["expected_score"], rel=1e-6)  # P in {0,1} -> expectation is exact
    ref, cfg = gridtesting_reference(mines)
    assert true["score"] >= ref * 0.999, (true, ref, cfg)


def test_known_mines_zero_B_when_possible():
    mines = np.zeros((R, C), bool)
    mines[::10, ::7] = True
    mines[:, 0] = False
    path, d = plan(mines.astype(float))
    t = score(path, mines)
    assert t["valid"] and t["on_path_mines"] == 0 and t["score"] > 0


def test_uncertain_belief_prefers_low_p_corridor():
    P = np.full((R, C), 0.05)
    P[:, 30:34] = 0.001  # a safe corridor
    path, d = plan(P)
    cols = [c for _, c in path.squares()]
    assert all(28 <= c <= 35 for c in cols), cols
    # and beats the straight line through the high-P region
    straight = expected_score([(r, 5) for r in range(R)], 0, P)["expected_score"]
    assert d["expected_score"] > straight


def test_blocked_detection_is_avoided_and_total_block_falls_back():
    P = np.zeros((R, C))
    P[70, :] = 1.0
    P[70, 17] = 0.0
    path, d = plan(P)
    assert (70, 17) in path.squares()
    P[70, 17] = 1.0  # fully blocked -> still returns something valid-shaped
    path, d = plan(P)
    assert path.validate()[0]


def test_runtime():
    rng = np.random.default_rng(0)
    P = np.clip(rng.random((R, C)) ** 5 * 0.2 + 0.02, 0, 1)
    P[rng.random((R, C)) < 0.01] = 0.95
    plan(P)
    t = time.perf_counter()
    path, d = plan(P)
    assert time.perf_counter() - t < 0.5
    assert d["runtime_s"] < 0.5
