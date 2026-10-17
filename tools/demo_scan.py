"""Random-allocation episode: 4 drones, K=4 random free blocks whenever a drone finishes; prints coverage stats.

Flow: Drone/Sensor (truth) -> FrameReport -> Mapper (belief, drift tracking) -> AllocatorInput (block mask).
Run from the repo root:  python3 -m tools.demo_scan [--seed N] [--density 0.03] [--table]
--table prints the leakage table (mean P on mine / clear squares) for sigma x drift handling mode.
Uses a local uniform random mine layout (does not depend on sim/mines.py)."""
import argparse
import time

import numpy as np

from sim.drone import Drone, start_poses
from sim.geometry import BEL_PER_SQ, SQ_COLS, SQ_FT, SQ_ROWS
from sim.interfaces import DroneStatus
from sim.mapper import Mapper
from sim.sensor import Sensor

EPISODE_S = 420.0


def random_mines(rng, density=0.03):
    mine_sq = rng.random((SQ_ROWS, SQ_COLS)) < density
    mine_xy = (np.stack(np.meshgrid(np.arange(SQ_COLS), np.arange(SQ_ROWS)), -1) + rng.random((SQ_ROWS, SQ_COLS, 2))) * SQ_FT
    return mine_sq, mine_xy


def drone_status(d: Drone, t: float) -> DroneStatus:
    """Believed pose + remaining path/eta of a sim Drone (what the flight controller would report)."""
    path = d.remaining_path()
    if d.has_path and not d.done:
        ss = np.linspace(d.s, d._L, len(path))
        eta = np.interp(ss, d._S, d._T) - np.interp(d.s, d._S, d._T)
    else:
        eta = np.zeros(len(path))
    b = d.believed_pos
    return DroneStatus(d.idx, t, (float(b[0]), float(b[1])), d.heading, path + d.gps_err, eta, bool(d.done))


def run_episode(seed=0, density=0.03, dt=0.1, k=4, episode_s=EPISODE_S, n_drones=4, prior=0.03, sensor_spec=None,
                mapper_spec=None):
    rng = np.random.default_rng(seed)
    mine_sq, mine_xy = random_mines(rng, density)
    sensor = Sensor(n_drones, rng, sensor_spec)
    mapper = Mapper(np.full((SQ_ROWS, SQ_COLS), prior), n_drones, {**(sensor_spec or {}), **(mapper_spec or {})})
    mapper.set_pos_sigma(sensor.sigma)
    drones = [Drone(i, p, h) for i, (p, h) in enumerate(start_poses())][:n_drones]
    stats = dict(tp=0, fp=0, frames=0, scan_s=0.0, map_s=0.0, scan_calls=0, assigns=0)
    off_err = []
    t = 0.0
    for _ in range(int(round(episode_s / dt))):
        for d in drones:
            if d.done:
                mapper.update_drone(drone_status(d, t))
                free = np.flatnonzero(mapper.snapshot(d.idx, t, episode_s).block_mask)
                blocks = rng.choice(free, size=k, replace=False)
                d.assign(blocks, rng)
                from sim.interfaces import Allocation
                mapper.record_allocation(Allocation(d.idx, [int(b) for b in blocks]))
                stats["assigns"] += 1
            d.step(dt)
            for b in d.pop_completed_blocks():
                mapper.release_blocks(d.idx, [b])
            t0 = time.perf_counter()
            rep = sensor.scan(d, t, dt, mine_sq, rng, mine_xy)
            stats["scan_s"] += time.perf_counter() - t0
            stats["scan_calls"] += 1
            if rep is not None:
                t0 = time.perf_counter()
                mapper.ingest_frame(rep)
                mapper.update_drone(drone_status(d, t))
                stats["map_s"] += time.perf_counter() - t0
                r = sensor.last_stats
                stats["tp"] += r.n_tp; stats["fp"] += r.n_fp; stats["frames"] += 1
                m, _ = mapper.offset_estimate(d.idx)
                off_err.append(np.r_[m - sensor.err[d.idx], sensor.err[d.idx]])
        t += dt
    psq = mapper.p_squares()
    seen = mapper.belief.seen
    seen_sq = seen.reshape(SQ_ROWS, BEL_PER_SQ, SQ_COLS, BEL_PER_SQ).mean(axis=(1, 3))
    oe = np.array(off_err)
    stats.update(frac_seen=float((seen > 0.5).mean()), mean_p_mine=float(psq[mine_sq].mean()),
                 mean_p_clear=float(psq[~mine_sq].mean()), n_mines=int(mine_sq.sum()),
                 mean_p_mine_seen=float(psq[mine_sq & (seen_sq > 0.5)].mean()) if (mine_sq & (seen_sq > 0.5)).any() else float("nan"),
                 mean_p_clear_seen=float(psq[~mine_sq & (seen_sq > 0.5)].mean()),
                 offset_rmse=float(np.sqrt((oe[:, :2] ** 2).mean())), drift_rms=float(np.sqrt((oe[:, 2:] ** 2).mean())))
    return stats


def leakage_table(seeds=(1, 2, 3, 4, 5), sigmas=(0.0, 0.5, 1.0, 1.5)):
    modes = (("none", "(a) no spreading"), ("independent", "(b) independent-frame spreading"), ("correlated", "(c) correlated tracking"))
    print("mean P on mine squares / clear squares (all squares; prior 0.030), averaged over seeds", list(seeds))
    print(f"{'sigma ft':>8}  " + "  ".join(f"{n:>34}" for _, n in modes) + "  offset RMSE (c) vs drift RMS")
    for sg in sigmas:
        cells, rm = [], ""
        for mode, _ in modes:
            rs = [run_episode(s, sensor_spec={"sensor_gps_sigma_ft": sg}, mapper_spec={"mapper_drift_mode": mode}) for s in seeds]
            cells.append(f"{np.mean([r['mean_p_mine'] for r in rs]):.3f} / {np.mean([r['mean_p_clear'] for r in rs]):.4f}")
            if mode == "correlated":
                rm = f"{np.mean([r['offset_rmse'] for r in rs]):.3f} vs {np.mean([r['drift_rms'] for r in rs]):.3f}"
        print(f"{sg:>8.1f}  " + "  ".join(f"{c:>34}" for c in cells) + "  " + rm)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--density", type=float, default=0.03)
    ap.add_argument("--table", action="store_true")
    a = ap.parse_args()
    if a.table:
        leakage_table()
        raise SystemExit
    t0 = time.perf_counter()
    s = run_episode(a.seed, a.density)
    wall = time.perf_counter() - t0
    print(f"episode wall time {wall:.2f}s, {s['assigns']} allocations, {s['frames']} frames, "
          f"sensor {1e6 * s['scan_s'] / s['scan_calls']:.1f} us/call avg, mapper {1e6 * s['map_s'] / s['frames']:.0f} us/frame")
    print(f"fraction of cells seen (>0.5): {s['frac_seen']:.3f}")
    print(f"mines: {s['n_mines']}; mean P(square) on mine squares {s['mean_p_mine']:.3f} (seen only {s['mean_p_mine_seen']:.3f}), "
          f"on clear squares {s['mean_p_clear']:.4f} (seen only {s['mean_p_clear_seen']:.4f})")
    print(f"detections: true positive {s['tp']}, false positive {s['fp']} (counted per frame, squares re-seen are re-counted)")
    print(f"offset estimate RMSE {s['offset_rmse']:.3f} ft vs true drift RMS {s['drift_rms']:.3f} ft")
