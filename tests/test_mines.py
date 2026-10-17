import numpy as np
import pytest

from sim import mines as M


def _stack(n, k, seed=0, **kw):
    rng = np.random.default_rng(seed)
    spec = {"mines_count": [n, n], "mines_clear_path": False, **kw}
    return np.array([M.generate(rng, spec) for _ in range(k)])


def _eval_by_n(n):
    return np.array([g for i, g in M.load_eval_set() if i.startswith(f"n{n}_")])


def _vmr10(G):
    b = G.reshape(len(G), 15, 10, 4, 10).sum((2, 4))
    return b.var() / b.mean()


def test_shape_dtype_count():
    assert M.SPEC_DEFAULTS["mines_count"] == [100, 170]
    rng = np.random.default_rng(0)
    lo, hi = M.SPEC_DEFAULTS["mines_count"]
    for _ in range(30):
        g = M.generate(rng)
        assert g.shape == (150, 40) and g.dtype == bool
        assert lo <= g.sum() <= hi


def test_determinism():
    a = M.generate(np.random.default_rng(5))
    b = M.generate(np.random.default_rng(5))
    c = M.generate(np.random.default_rng(6))
    assert (a == b).all() and not (a == c).all()


def test_speed():
    import time
    rng = np.random.default_rng(0)
    M.generate(rng)
    t = time.perf_counter()
    for _ in range(100):
        M.generate(rng)
    assert (time.perf_counter() - t) / 100 < 5e-3


def test_clear_path_guarantee():
    rng = np.random.default_rng(1)
    for n in (170, 900, 1500):  # absurdly dense: forces repairs
        for _ in range(10):
            g = M.generate(rng, {"mines_count": [n, n], "mines_clear_path": True})
            assert M.has_clear_path(g)
            assert g.sum() == n  # repair preserves the count
    dense = M.generate(rng, {"mines_count": [3000, 3000], "mines_clear_path": False})
    assert not M.has_clear_path(dense)


def test_stats_match_fexl():
    for n in (100, 135, 170):
        f, s = _eval_by_n(n), _stack(n, 150)
        assert abs(_vmr10(s) - _vmr10(f)) / _vmr10(f) < 0.2, n
        pf, ps = f.sum((0, 2)).reshape(5, 30).sum(1), s.sum((0, 2)).reshape(5, 30).sum(1)
        assert np.abs(pf / pf.sum() - ps / ps.sum()).max() < 0.02, n
    # clumped: far from the VMR ~1 of a uniform layout
    assert _vmr10(_stack(135, 100)) > 2.5


def test_prior_density():
    p = M.prior_density()
    assert p.shape == (150, 40) and (p >= 0).all()
    lo, hi = M.SPEC_DEFAULTS["mines_count"]
    assert abs(p.sum() - (lo + hi) / 2) < 5
    assert p[:70].mean() > 3 * p[130:].mean()
    assert M.prior_density() is p  # cached


def test_eval_set():
    ev = M.load_eval_set("fexl")
    assert len(ev) == 150
    ids = [i for i, _ in ev]
    assert len(set(ids)) == 150
    for i, g in ev:
        assert g.shape == (150, 40) and g.dtype == bool
        assert g.sum() == int(i.split("_")[0][1:]) and 100 <= g.sum() <= 170
    with pytest.raises(ValueError):
        M.load_eval_set("nope")
