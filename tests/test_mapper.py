import time

import numpy as np

from sim.belief import Belief
from sim.geometry import BEL_COLS, BEL_ROWS, N_BLOCKS, SQ_COLS, SQ_ROWS
from sim.interfaces import Allocation, AllocatorInput, Detection, DroneStatus, FrameReport
from sim.mapper import Mapper
from sim.sensor import footprint_dims, p_detect_at

PRIOR = np.full((SQ_ROWS, SQ_COLS), 0.03)


def frame(d=0, t=0.0, pose=(40.0, 100.0), yaw=0.0, sigma=0.0, dets=(), alt=15.0):
    return FrameReport(d, t, pose, yaw, alt, sigma, footprint_dims({"sensor_altitude_ft": alt, "sensor_fov_across_deg": 60.0,
                                                                    "sensor_fov_along_deg": 45.0}),
                       [Detection(tuple(x)) for x in dets])


def test_snapshot_shapes_and_blocks():
    m = Mapper(PRIOR, 3, None)
    m.record_allocation(Allocation(1, [5, 40]))
    m.record_allocation(Allocation(2, [41]))
    s = m.snapshot(0, 12.0, 420.0)
    assert isinstance(s, AllocatorInput) and s.drone_id == 0 and s.t == 12.0 and s.t_end == 420.0
    assert s.p_mine.shape == (300, 80) and s.p_mine.dtype == np.float32
    assert s.seen.shape == (300, 80) and s.seen.dtype == np.float32
    assert s.allocated.shape == (300, 80) and s.allocated.dtype == np.int8
    assert s.others_path_eta.shape == (300, 80) and s.others_path_eta.dtype == np.float32
    assert len(s.drones) == 3 and all(isinstance(x, DroneStatus) for x in s.drones)
    assert s.block_mask.shape == (N_BLOCKS,) and s.block_mask.dtype == bool
    assert not s.block_mask[[5, 40, 41]].any() and s.block_mask.sum() == N_BLOCKS - 3
    assert m.snapshot(1, 0, 1).block_mask[[5, 40]].all() and not m.snapshot(1, 0, 1).block_mask[41]
    # snapshot is a copy
    s.p_mine[:] = 0
    assert m.snapshot(0, 0, 1).p_mine.max() > 0


def test_allocation_replaces_and_release():
    m = Mapper(PRIOR, 2, None)
    m.record_allocation(Allocation(0, [1, 2, 3]))
    m.record_allocation(Allocation(0, [3, 4]))            # previous blocks released
    assert m.snapshot(1, 0, 1).block_mask[[1, 2]].all() and not m.snapshot(1, 0, 1).block_mask[[3, 4]].any()
    m.release_blocks(0, [3])
    assert m.snapshot(1, 0, 1).block_mask[3] and not m.snapshot(1, 0, 1).block_mask[4]
    m.record_allocation(Allocation(1, [4]))               # (re)taken by drone 1: drone 0 can't release it
    m.release_blocks(0, [4])
    assert not m.snapshot(0, 0, 1).block_mask[4]


def test_others_path_eta_raster():
    m = Mapper(PRIOR, 3, None)
    path = np.array([[10.0, 10.0], [10.0, 30.0], [30.0, 30.0]])
    eta = np.array([0.0, 2.0, 4.0])
    m.update_drone(DroneStatus(1, 0.0, (10.0, 10.0), 0.0, path, eta, False))
    m.update_drone(DroneStatus(2, 0.0, (50.0, 50.0), 0.0, path + 20, eta, True))        # idle: ignored
    own = DroneStatus(0, 0.0, (5.0, 5.0), 0.0, path - 5, eta, False)
    m.update_drone(own)
    r = m.snapshot(0, 0.0, 420.0).others_path_eta
    assert r[10, 10] == 1.0 and abs(r[30, 10] - 0.5) < 0.05 and r[30, 29] < 0.1 and r[30, 25] < 0.4
    assert r[0, 0] == 0 and r[50, 50] == 0 and r[5, 5] == 0                              # own and idle paths excluded
    assert 0 <= r.min() and r.max() <= 1
    assert m.snapshot(1, 0.0, 420.0).others_path_eta[10, 10] == 0                        # drone 1's own path excluded
    assert m.snapshot(2, 0.0, 420.0).others_path_eta[10, 10] == 1.0


def test_ingest_bayes_and_pd_from_altitude():
    pd = p_detect_at({}, 15.0)
    m = Mapper(PRIOR, 1, {"mapper_drift_mode": "none"})
    m.ingest_frame(frame(dets=[(41.0, 101.0)]))
    p = m.p_squares()[50, 20]
    exp = 0.03 * pd / (0.03 * pd + 0.97 * 0.01)
    assert abs(p - exp) < 1e-5
    assert m.p_squares()[50, 18] < 0.03 and m.snapshot(0, 0, 1).seen.max() > 0       # a miss elsewhere in view
    m2 = Mapper(PRIOR, 1, {"mapper_drift_mode": "none"})
    m2.ingest_frame(frame(dets=[(41.0, 101.0)], alt=25.0))                          # higher: lower p_detect -> weaker
    assert m2.p_squares()[50, 20] < p


def test_mapper_without_truth_and_other_modes():
    # nothing but reports/statuses/allocations are needed; every mode runs with empty and non-empty frames
    for mode in ("none", "independent", "correlated"):
        m = Mapper(PRIOR, 2, {"mapper_drift_mode": mode})
        m.set_pos_sigma(1.0)
        for k in range(10):
            m.ingest_frame(frame(d=k % 2, t=0.5 * k, pose=(40.0 + k, 100.0), sigma=1.0, dets=[(45.0, 100.0)] if k % 3 == 0 else []))
        assert np.isfinite(m.p_squares()).all() and m.p_squares().shape == (150, 40)
        assert m.snapshot(0, 5.0, 420.0).pos_sigma_ft == 1.0
    # edge of the field
    m.ingest_frame(frame(pose=(0.0, 0.0), yaw=1.0, sigma=1.0, dets=[(1.0, 1.0)]))
    m.ingest_frame(frame(pose=(79.0, 299.0), sigma=1.0))
    assert np.isfinite(m.belief.p_mine).all()


def test_kalman_converges_with_anchors():
    """Drone 0 (zero drift, stationary) flies over a few mines and makes them confident anchors; drone 1 has a constant
    offset E and sees them: its estimate must converge to E, with shrinking std. Mines are placed from truth in the test only."""
    rng = np.random.default_rng(0)
    E = np.array([0.8, -0.6])
    sigma = 1.0
    mines = np.array([[30.3, 100.7], [36.9, 103.1], [44.2, 98.4], [50.5, 101.9], [41.0, 106.0]])
    m = Mapper(PRIOR, 2, {"sensor_gps_sigma_ft": sigma})
    for k in range(3):                                                  # drone 0 sees each mine in 3 frames, no offset
        m.ingest_frame(frame(0, t=0.5 * k, pose=(40.0, 102.0), sigma=sigma, dets=mines + rng.normal(0, 0.05, mines.shape)))
    est0, std0 = m.offset_estimate(1)
    assert np.allclose(est0, 0)
    for k in range(6):
        m.ingest_frame(frame(1, t=100.0 + 0.5 * k, pose=(40.0 + E[0], 102.0 + E[1]), sigma=sigma,
                             dets=mines + E + rng.normal(0, 0.05, mines.shape)))
    est, std = m.offset_estimate(1)
    assert np.abs(est - E).max() < 0.2, est
    assert std < 0.5 * sigma
    # no anchor from the drone's own frames: a lone drone seeing the same mines twice never self-confirms
    m1 = Mapper(PRIOR, 1, {"sensor_gps_sigma_ft": sigma})
    for k in range(6):
        m1.ingest_frame(frame(0, t=0.5 * k, pose=(40.0, 102.0), sigma=sigma, dets=mines + E))
    e1, s1 = m1.offset_estimate(0)
    assert np.allclose(e1, 0) and abs(s1 - sigma) < 1e-9


def test_false_positive_does_not_anchor():
    m = Mapper(PRIOR, 2, {"sensor_gps_sigma_ft": 1.0})
    m.ingest_frame(frame(0, 0.0, sigma=1.0, dets=[(40.0, 100.0)]))                   # single-frame detection
    m.ingest_frame(frame(1, 100.0, sigma=1.0, dets=[(41.0, 100.5)]))
    e, s = m.offset_estimate(1)
    assert np.allclose(e, 0) and abs(s - 1.0) < 1e-9


def _leak(sigma, mode, seed):
    from tools.demo_scan import run_episode
    return run_episode(seed, sensor_spec={"sensor_gps_sigma_ft": sigma}, mapper_spec={"mapper_drift_mode": mode})


def test_leakage_table_thresholds_and_perf():
    t0 = time.perf_counter()
    r = _leak(0.5, "correlated", 1)
    wall = time.perf_counter() - t0
    assert wall < 6.0, wall                                 # ~2 s typical; loose for loaded machines
    assert r["mean_p_clear"] < 0.03 and r["mean_p_mine"] > 0.6
    ind = _leak(0.5, "independent", 1)
    assert ind["mean_p_clear"] > 1.5 * r["mean_p_clear"]    # independent-frame spreading leaks
    r15 = _leak(1.5, "correlated", 1)
    ind15 = _leak(1.5, "independent", 1)
    assert r15["mean_p_clear"] < 0.03 and ind15["mean_p_clear"] > 0.1
    assert r15["mean_p_mine"] > _leak(1.5, "none", 1)["mean_p_mine"]
    assert r["offset_rmse"] < r["drift_rms"]               # tracking beats assuming zero offset
    z = _leak(0.0, "correlated", 1)
    assert z["mean_p_mine"] > 0.95 and z["mean_p_clear"] < 0.001
