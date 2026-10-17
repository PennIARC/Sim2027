"""Characterise the fexl mine distribution and fit sim/mines.py's generator to it.

Findings encoded here (re-run `python3 tools/fit_mines.py` to reproduce; needs data/fexl_cache/layouts.json from
tools/scrape_fexl.py):
  * fexl layouts are deterministic per seed and NESTED in num_danger (60 subset 135 subset 250, same seed) -> a fixed
    per-seed random intensity field, mines drawn one at a time from it (a Cox process).
  * block-count CV^2 of that field ((VMR-1)/mean) is ~constant across num_danger -> Cox, not a cluster process whose
    clumpiness would scale with n.
  * mean row profile: flat-ish to ~70% of the field then tapering to ~0.15x at the far edge; columns uniform.

Modes: default = fit + write sim/mine_params.json + print table; `--table` = only print the comparison table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sim import mines as M  # noqa: E402

CACHE = ROOT / "data" / "fexl_cache" / "layouts.json"
PARAMS = ROOT / "sim" / "mine_params.json"
NUMS = (100, 135, 170)
STEP = 5  # row-profile knot spacing (squares)


def grid(m):
    g = np.zeros((150, 40), bool)
    for c, r in m:
        g[r, c] = True
    return g


def load_fexl():
    d = json.loads(CACHE.read_text())
    return {int(n): [grid(m) for m in L.values()] for n, L in d.items() if int(n) in NUMS}


def stats(G):
    """Dict of scalar statistics over a list of bool grids (all with the same mine count)."""
    G = np.asarray(G)
    out = {}
    for bs in (5, 10, 20):  # field CV^2 = (VMR-1)/mean on rows 0..99 (flat part of the profile)
        R = 100 // bs * bs
        b = G[:, :R, :].reshape(len(G), R // bs, bs, 40 // bs, bs).sum((2, 4)).astype(float)
        out[f"cv2_b{bs}"] = (b.var() / b.mean() - 1) / b.mean()
    b10 = G.reshape(len(G), 15, 10, 4, 10).sum((2, 4))
    out["vmr_b10"] = b10.var() / b10.mean()
    out["empty_b10"] = (b10 == 0).mean()
    out["max_b10"] = b10.max(axis=(1, 2)).mean()
    near = {1: [], 2: [], 3: [], 5: []}
    k = {4: [], 8: [], 12: []}
    nn = []
    n4 = []
    for g in G:
        ys, xs = np.nonzero(g)
        pts = np.c_[ys, xs].astype(float)
        t = cKDTree(pts)
        for r in near:  # fraction with another mine within Chebyshev r
            cnt = np.array([len(x) for x in t.query_ball_point(pts, r, p=np.inf)]) - 1
            near[r].append((cnt > 0).mean())
        for r in k:
            cnt = np.array([len(x) for x in t.query_ball_point(pts, r, p=2)]) - 1
            k[r].append(cnt.mean())
        nn.append(t.query(pts, 2)[0][:, 1].mean())
        up = g[1:] & g[:-1]
        lf = g[:, 1:] & g[:, :-1]
        n4.append(((up.sum() + lf.sum()) * 2) / g.sum())  # approx: mines-with-edge-neighbour upper bound
    for r in near:
        out[f"near_cheb{r}"] = np.mean(near[r])
    for r in k:
        out[f"K_r{r}"] = np.mean(k[r])
    out["nn_dist"] = np.mean(nn)
    out["edge_nbr_pairs"] = np.mean(n4) / 2
    rows = G.sum((0, 2)).astype(float)
    rows /= rows.sum()
    for i, v in enumerate(rows.reshape(5, 30).sum(1)):
        out[f"rowfrac_{i * 30}-{i * 30 + 29}"] = v
    return out


def row_knots(fexl):
    tot = np.zeros(150)
    for n, L in fexl.items():
        for g in L:
            tot += g.sum(1) / n  # per-layout normalised
    tot = gaussian_filter1d(tot, 4.0, mode="nearest")
    prof = tot / tot.mean()  # relative density, mean 1
    return [float(round(prof[i], 3)) for i in range(0, 150, STEP)]


def sim_layouts(spec, n, k, seed):
    rng = np.random.default_rng(seed)
    sp = {**spec, "mines_count": [n, n]}
    return [M.generate(rng, sp) for _ in range(k)]


def uniform_layouts(n, k, seed, profile=None):
    rng = np.random.default_rng(seed)
    p = np.ones(150) if profile is None else profile
    p = np.repeat(p, 40) / np.repeat(p, 40).sum()
    out = []
    for _ in range(k):
        g = np.zeros(6000, bool)
        g[rng.choice(6000, n, replace=False, p=p)] = True
        out.append(g.reshape(150, 40))
    return out


FIT_KEYS = ["cv2_b5", "cv2_b10", "cv2_b20", "near_cheb1", "near_cheb2", "near_cheb3", "K_r4", "K_r8", "empty_b10"]


def fit(fexl, knots):
    base = {**M.SPEC_DEFAULTS, "mines_row_knots": knots, "mines_row_step": STEP, "mines_clear_path": False}
    target = {n: stats(fexl[n]) for n in NUMS}
    best = (1e9, None)
    t0 = time.time()
    # coarse-to-fine grid over (sigma, len_rows, len_cols)
    def score(sig, ly, lx, k=60):
        sp = {**base, "mines_field_sigma": sig, "mines_field_len_rows": ly, "mines_field_len_cols": lx}
        e = 0.0
        for n in NUMS:
            s = stats(sim_layouts(sp, n, k, 7))
            for key in FIT_KEYS:
                e += ((s[key] - target[n][key]) / (abs(target[n][key]) + 1e-9)) ** 2
        return e
    grid_ = [(sg, ly, lx) for sg in (0.8, 1.0, 1.2, 1.4) for ly in (6.5, 9, 12, 16) for lx in (4.5, 6.5, 9, 12)]
    res = []
    for p in grid_:
        res.append((score(*p), p))
    res.sort()
    print("coarse top3:", res[:3], f"{time.time() - t0:.0f}s", flush=True)
    sg0, ly0, lx0 = res[0][1]
    for sg in np.linspace(sg0 * 0.85, sg0 * 1.15, 5):
        for ly in np.linspace(ly0 * 0.75, ly0 * 1.25, 5):
            for lx in np.linspace(lx0 * 0.75, lx0 * 1.25, 5):
                e = score(sg, ly, lx, 200)
                if e < best[0]:
                    best = (e, (float(sg), float(ly), float(lx)))
    print("fine best:", best, f"{time.time() - t0:.0f}s", flush=True)
    return best[1]


def table(fexl, spec):
    prof = np.array(M.row_profile(spec))
    print(f"{'stat':<22}{'n':>5}{'fexl':>10}{'fitted':>10}{'unif':>10}{'unif+prof':>11}")
    for n in NUMS:
        sf, ss = stats(fexl[n]), stats(sim_layouts({**spec, "mines_clear_path": False}, n, 300, 11))
        su, sp = stats(uniform_layouts(n, 100, 3)), stats(uniform_layouts(n, 100, 3, prof))
        for k in sf:
            print(f"{k:<22}{n:>5}{sf[k]:>10.3f}{ss[k]:>10.3f}{su[k]:>10.3f}{sp[k]:>11.3f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    fexl = load_fexl()
    if not a.table:
        knots = row_knots(fexl)
        sig, ly, lx = fit(fexl, knots)
        PARAMS.write_text(json.dumps({
            "count_range": [100, 170], "field_sigma": round(sig, 3), "field_len_rows": round(ly, 2),
            "field_len_cols": round(lx, 2), "row_knots": knots, "row_step": STEP,
            "note": "Fitted by tools/fit_mines.py to 150 scraped fexl layouts (num_danger 100/135/170 x 50 seeds). "
                    "Log-Gaussian Cox process: lambda=profile(row)*exp(sigma*G-sigma^2/2), G Gaussian field with "
                    "Gaussian smoothing lengths (rows, cols); mines drawn w/o replacement prop. to lambda. "
                    "count_range is user-specified, not fitted."}, indent=1))
        import importlib
        importlib.reload(M)
    table(fexl, M.SPEC_DEFAULTS)


if __name__ == "__main__":
    main()
