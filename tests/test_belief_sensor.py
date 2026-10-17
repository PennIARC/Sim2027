import numpy as np

from sim.belief import Belief
from sim.drone import Drone
from sim.geometry import SQ_COLS, SQ_ROWS, belief_to_squares_prob
from sim.sensor import SPEC_DEFAULTS, Sensor, footprint_dims, p_detect_for


def make_belief(p=0.03):
    return Belief(np.full((SQ_ROWS, SQ_COLS), p))


def test_init_conservation():
    rng = np.random.default_rng(0)
    prior = rng.uniform(0.02, 0.05, (SQ_ROWS, SQ_COLS))
    b = Belief(prior)
    assert b.p_mine.dtype == np.float32 and b.p_mine.shape == (300, 80)
    assert np.allclose(belief_to_squares_prob(b.p_mine), prior, atol=1e-6)
    assert b.seen.sum() == 0 and (b.allocated == -1).all() and b.allocated.dtype == np.int8


def test_bayes_hand_case():
    b = make_belief(0.03)
    pd, pf = 0.9, 0.01
    b.apply_observation(10, 10, np.array([[False]]), np.array([[1.0]]), pd, pf)
    expect = 0.03 * 0.1 / (0.03 * 0.1 + 0.97 * 0.99)
    assert abs(b.p_squares()[10, 10] - expect) < 1e-6
    assert np.allclose(b.p_mine[20:22, 20:22], expect / 4, atol=1e-7)   # spread equally
    b.apply_observation(11, 10, np.array([[True]]), np.array([[1.0]]), pd, pf)
    expect = 0.03 * 0.9 / (0.03 * 0.9 + 0.97 * 0.01)
    assert abs(b.p_squares()[11, 10] - expect) < 1e-6 and expect > 0.7
    assert b.seen[22, 20] == np.float32(0.6)


def test_edge_tempering():
    b = make_belief(0.03)
    w = np.array([[1.0, 0.5, 0.0]])
    b.apply_observation(5, 5, np.zeros((1, 3), bool), w, 0.9, 0.01)
    p = b.p_squares()[5, 5:8]
    assert p[0] < p[1] < p[2] == np.float32(0.03) or abs(p[2] - 0.03) < 1e-7
    full = 0.03 * 0.1 / (0.03 * 0.1 + 0.97 * 0.99)
    half = 0.03 * 0.1 ** 0.5 / (0.03 * 0.1 ** 0.5 + 0.97 * 0.99 ** 0.5)
    assert abs(p[0] - full) < 1e-6 and abs(p[1] - half) < 1e-6
    assert b.seen[10, 10] > b.seen[10, 12] > b.seen[10, 14] == 0


def test_footprint_weights_and_allocation_channels():
    s = Sensor(1, np.random.default_rng(0))
    r0, c0, xs, ys, w = s.footprint_weights(40.0, 100.0, 0.0)   # heading +x: along x, across y
    assert w.max() == 1.0 and (w > 0).any()
    assert xs[w > 0].max() - xs[w > 0].min() <= s.along and ys[w > 0].max() - ys[w > 0].min() <= s.across
    assert w[w > 0].min() < 1.0                                  # blurred edge
    b = make_belief()
    b.mark_blocks(2, [0, 9]); b.mark_blocks(1, [9])
    assert b.mask_for_drone(2).sum() == 100 and b.mask_for_others(2).sum() == 100
    assert b.mask_for_drone(1).sum() == 100
    obs = b.observation(1)
    assert obs.shape == (4, 300, 80) and obs.dtype == np.float32 and obs[2].sum() == 100 and obs[3].sum() == 100
    b.clear_drone(1); assert b.mask_for_drone(1).sum() == 0


def test_drift_statistics():
    rng = np.random.default_rng(1)
    s = Sensor(1, rng, {"sensor_gps_sigma_ft": 1.5, "sensor_gps_tau_s": 2.0})
    e = np.array([s.step_drift(0, 0.1, rng).copy() for _ in range(60000)])
    assert abs(e.std() - 1.5) < 0.15 and abs(e.mean()) < 0.2
    ac = np.corrcoef(e[:-20, 0], e[20:, 0])[0, 1]          # lag 2 s = tau -> exp(-1)
    assert abs(ac - np.exp(-1)) < 0.08


def test_detection_rates():
    rng = np.random.default_rng(2)
    mine = np.zeros((SQ_ROWS, SQ_COLS), bool)
    mine[::3, ::3] = True
    b = make_belief()
    s = Sensor(1, rng, {"sensor_gps_sigma_ft": 0.0})
    d = Drone(0, (40.0, 150.0), 0.0)
    tp = fp = nm = nc = 0
    for k in range(3000):
        if s.scan(d, 0.5 * k, 0.5, mine, rng) is None:
            continue
        r = s.last_stats
        tp += r.n_tp; fp += r.n_fp; nm += r.n_mines_in_view; nc += r.n_clear_in_view
    assert abs(tp / nm - s.p_detect) < 0.02
    assert abs(fp / nc - 0.01) < 0.003


def test_scan_returns_frame_report_in_believed_frame():
    from sim.interfaces import FrameReport
    rng = np.random.default_rng(5)
    mine = np.zeros((SQ_ROWS, SQ_COLS), bool); mine[75, 20] = True
    xy = np.zeros((SQ_ROWS, SQ_COLS, 2)); xy[75, 20] = (41.3, 151.2)
    s = Sensor(1, rng, {"sensor_gps_sigma_ft": 1.0, "sensor_p_detect_max": 1.0, "sensor_px_for_half_detect": 1e-3})
    d = Drone(0, (40.0, 150.0), 0.0)
    rep = s.scan(d, 0.0, 0.1, mine, rng, xy)
    assert isinstance(rep, FrameReport) and rep.drone_id == 0 and rep.footprint_ft == (s.across, s.along)
    assert np.allclose(rep.pose_xy, d.pos + s.err[0]) and rep.pos_sigma_ft == 1.0
    tp = [x for x in rep.detections if np.hypot(x.xy[0] - 41.3 - s.err[0, 0], x.xy[1] - 151.2 - s.err[0, 1]) < 1e-9]
    assert len(tp) == 1                                        # the mine, reported at true position + GPS error


def test_frame_rate_not_per_tick():
    rng = np.random.default_rng(3)
    s = Sensor(1, rng)
    d = Drone(0, (40.0, 150.0), 0.0)
    b = make_belief(); mine = np.zeros((SQ_ROWS, SQ_COLS), bool)
    frames = sum(s.scan(d, 0.1 * k, 0.1, mine, rng) is not None for k in range(100))
    assert frames == 20   # 10 s at 0.5 s per frame


def test_scan_speed():
    import time
    rng = np.random.default_rng(4)
    s = Sensor(1, rng, {"sensor_frame_interval_s": 0.0})   # worst case: a frame every call
    d = Drone(0, (40.0, 150.0), 0.7)
    b = make_belief(); mine = rng.random((SQ_ROWS, SQ_COLS)) < 0.03
    t0 = time.perf_counter()
    for k in range(2000):
        s.scan(d, k * 0.1, 0.1, mine, rng)
    us = (time.perf_counter() - t0) / 2000 * 1e6
    print(f"scan with a frame every call: {us:.1f} us")
    assert us < 1000


def test_footprint_and_detection_vs_altitude():
    dims, pds = [], []
    for h in (10, 15, 20, 30):
        sp = {**SPEC_DEFAULTS, "sensor_altitude_ft": h}
        dims.append(footprint_dims(sp)); pds.append(p_detect_for(sp))
    assert all(a[0] < b[0] and a[1] < b[1] for a, b in zip(dims, dims[1:]))
    assert all(a > b for a, b in zip(pds, pds[1:]))
    d15 = footprint_dims({**SPEC_DEFAULTS})
    assert abs(d15[0] - 17.32) < 0.01 and abs(d15[1] - 12.43) < 0.01
    # half-detect pixel count gives exactly half of p_detect_max
    h_half = None
    sp = {**SPEC_DEFAULTS, "sensor_px_for_half_detect": 28.9}
    assert p_detect_for(sp) < 0.5 * 0.95 + 0.02 and p_detect_for(sp) > 0.5 * 0.95 - 0.02
    print("p_detect by altitude", dict(zip((10, 15, 20, 30), np.round(pds, 3))))


def test_gps_curriculum_hook():
    rng = np.random.default_rng(0)
    s = Sensor(2, rng)
    assert s.sigma == 0.5
    s.set_progress(0.25); assert abs(s.sigma - 1.0) < 1e-9
    s.set_progress(0.5); assert abs(s.sigma - 1.5) < 1e-9
    s.set_progress(1.0); assert abs(s.sigma - 1.5) < 1e-9
    s.set_progress(0.0); assert abs(s.sigma - 0.5) < 1e-9


def test_drift_spreading_update():
    det = np.array([[True]]); w = np.array([[1.0]])
    a, b = make_belief(), make_belief()
    a.apply_observation(50, 20, det, w, 0.9, 0.01, 0.0)
    b.apply_observation(50, 20, det, w, 0.9, 0.01, 0.6)
    pa, pb = a.p_squares(), b.p_squares()
    assert pa[50, 19] == np.float32(0.03) and pb[50, 19] > 0.03 and pb[51, 20] > 0.03   # neighbors gain
    assert pb[50, 20] < pa[50, 20] and pb[50, 20] > 0.03                               # center less sharp
    # no-detection: spreading weakens the veto at the center, lowers neighbors a little
    c, d = make_belief(), make_belief()
    c.apply_observation(50, 20, ~det, w, 0.9, 0.01, 0.0); d.apply_observation(50, 20, ~det, w, 0.9, 0.01, 0.6)
    assert d.p_squares()[50, 20] > c.p_squares()[50, 20] and d.p_squares()[50, 19] < 0.03
    # field borders: window clipped without error
    d.apply_observation(0, 0, det, w, 0.9, 0.01, 1.0); d.apply_observation(149, 39, det, w, 0.9, 0.01, 1.0)
    assert np.isfinite(d.p_mine).all() and d.p_mine.min() >= 0


def test_sweep_pass_covers_block_with_derived_footprint():
    from sim.geometry import block_bounds
    from sim.routing import pass_points
    s = Sensor(1, np.random.default_rng(0))
    for opt in range(4):
        e, x, d = pass_points(94, opt)       # an interior block
        h = np.arctan2(d[1], d[0])
        cov = np.zeros((150, 40))
        for f in (0.0, 0.5, 1.0):           # frames 5 ft apart along the pass
            p = e + (x - e) * f
            r0, c0, _, _, w = s.footprint_weights(p[0], p[1], h)
            cov[r0:r0 + w.shape[0], c0:c0 + w.shape[1]] = np.maximum(cov[r0:r0 + w.shape[0], c0:c0 + w.shape[1]], w)
        x0, y0, x1, y1 = block_bounds(94)
        blk = cov[int(y0 // 2):int(y1 // 2), int(x0 // 2):int(x1 // 2)]
        assert blk.min() > 0.99
        # even with 3 ft of lateral error the block is still covered at w >= 0.5
        shifted = np.zeros_like(cov)
        for f in (0.0, 0.5, 1.0):
            p = e + (x - e) * f + np.array([-d[1], d[0]]) * 3.0
            r0, c0, _, _, w = s.footprint_weights(p[0], p[1], h)
            shifted[r0:r0 + w.shape[0], c0:c0 + w.shape[1]] = np.maximum(shifted[r0:r0 + w.shape[0], c0:c0 + w.shape[1]], w)
        assert shifted[int(y0 // 2):int(y1 // 2), int(x0 // 2):int(x1 // 2)].min() >= 0.5
