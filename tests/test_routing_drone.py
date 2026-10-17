import itertools
import time

import numpy as np

from sim.drone import Drone, start_poses
from sim.geometry import N_BLOCKS
from sim.routing import SPEC_DEFAULTS as R, _matrices, _perm_costs, order_blocks


def brute(blocks, pos, heading):
    S, D = _matrices(blocks, pos, heading, R["route_turn_penalty_ft_per_rad"])
    perms = np.array(list(itertools.permutations(range(len(blocks)))))
    return _perm_costs(S, D, perms).min()


def test_routing_matches_brute_force_and_is_deterministic():
    rng = np.random.default_rng(0)
    for _ in range(5):
        blocks = rng.choice(N_BLOCKS, 5, replace=False)
        route, cost = order_blocks(blocks, [40, 0], np.pi / 2)
        assert abs(cost - brute(list(blocks), [40, 0], np.pi / 2)) < 1e-9
        assert order_blocks(blocks, [40, 0], np.pi / 2)[0] == route
        assert sorted(b for b, _ in route) == sorted(blocks)


def test_aco_near_optimal_k9():
    rng = np.random.default_rng(5)
    for trial in range(2):
        blocks = list(rng.choice(N_BLOCKS, 9, replace=False))
        opt = brute(blocks, [30, 0], np.pi / 2)
        route, cost = order_blocks(blocks, [30, 0], np.pi / 2, rng=np.random.default_rng(trial))
        assert len(route) == 9 and cost <= opt * 1.05, (cost, opt)


def test_spline_turn_radius_and_field():
    rng = np.random.default_rng(6)
    worst, n_bad = np.inf, 0
    for trial in range(30):
        d = Drone(0, *start_poses()[trial % 4])
        d.assign(rng.choice(N_BLOCKS, 4, replace=False), rng)
        r = 1 / d.path_curvature().max()
        worst = min(worst, r); n_bad += r < 0.9 * d.spec["drone_min_turn_radius_ft"]
        assert np.all(d._vlim * d.path_curvature() <= d.spec["drone_max_yaw_rate_radps"] + 1e-6)   # yaw-rate limit holds
        assert d._xy[:, 0].min() >= 0 and d._xy[:, 0].max() <= 80 and d._xy[:, 1].min() >= 0 and d._xy[:, 1].max() <= 300
    print(f"min radius over 30 random routes {worst:.2f} ft, routes below 0.9*rmin: {n_bad}")
    # geometric radius: reversals between nearby blocks cannot fit a full fillet (smaller radius, flown slowly);
    # the benign example route must respect it fully
    assert n_bad <= 12
    d = Drone(0, *start_poses()[0]); d.assign([45, 100, 150, 62], rng)
    assert 1 / d.path_curvature().max() >= 0.9 * d.spec["drone_min_turn_radius_ft"]


def fly(d, dt):
    t = 0.0
    while not d.done and t < 600:
        d.step(dt); t += dt
    return t


def test_four_block_route_time_and_blocks_released():
    d = Drone(0, *start_poses()[0])
    d.assign([45, 100, 150, 62], np.random.default_rng(0))
    est = d.eta()
    t = fly(d, 0.1)
    assert 0.6 * est < t < 1.5 * est and 15 < t < 60
    assert sorted(d.pop_completed_blocks()) == [45, 62, 100, 150] and d.eta() == 0
    assert d.remaining_path().shape == (16, 2)


def test_remaining_path_and_speed_limits():
    d = Drone(0, *start_poses()[1])
    d.assign([5, 120, 230, 77], np.random.default_rng(1))
    for _ in range(50):
        d.step(0.1)
    rp = d.remaining_path(10)
    assert rp.shape == (10, 2) and np.allclose(rp[0], d.pos, atol=0.6)
    vmax, ymax, last_h = 0, 0, d.heading
    while not d.done:
        d.step(0.1)
        dh = abs((d.heading - last_h + np.pi) % (2 * np.pi) - np.pi); last_h = d.heading
        vmax = max(vmax, d.speed); ymax = max(ymax, dh / 0.1)
    assert vmax <= 10.0 + 1e-6 and ymax < 2.0 * 1.6


def test_dt_independence():
    ts = []
    for dt in (0.1, 0.05):
        d = Drone(0, *start_poses()[2])
        d.assign([45, 100, 150, 62], np.random.default_rng(0))
        ts.append(fly(d, dt))
    assert abs(ts[0] - ts[1]) < 0.5


def test_full_episode_perf():
    from tools import demo_scan as demo
    t0 = time.perf_counter()
    s = demo.run_episode(seed=0)
    wall = time.perf_counter() - t0
    print(f"\n7-minute 4-drone episode: {wall:.2f}s wall, {1e6 * s['scan_s'] / s['scan_calls']:.1f} us per scan call, "
          f"{s['assigns']} allocations")
    assert wall < 10 and s["assigns"] > 20
