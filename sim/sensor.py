"""Downward camera + noisy GPS.

Footprint orientation: HEAD-ALIGNED (long side across-track, as a body-fixed camera would be). Field-aligned would be
simpler but would give 12 ft coverage only for travel along y; in-block passes also run along x, where only the 8 ft
side would be across-track and a single centered pass would not cover a 10 ft block. Cost of rotation is small: we
test square centers in the bounding box of the rotated rectangle (<= ~8x8 squares).

Look rate: a drone takes one "frame" every sensor_frame_interval_s (0.5 s), not every tick. Information then accrues per
distinct view (at 10 ft/s consecutive frames are 5 ft apart vs an 8 ft along-track footprint, so each square gets
~1-2 independent looks, edge squares are down-weighted by w). A hovering drone keeps adding one look per frame
interval (like a camera with independent frames). dt-independent: frames fire on time, not on tick count.
GPS: believed = true + e, e an Ornstein-Uhlenbeck process (exact discretization) per axis.

The Sensor is the "vision" stand-in: from ground truth (true pose, true mines, GPS error) it emits a FrameReport (believed
pose = true + e, detections = true mines seen with p_detect plus false positives, positions in the believed frame). It does
NOT touch the belief; sim/mapper.py fuses the reports.
"""
from typing import NamedTuple

import numpy as np

from sim.geometry import SQ_COLS, SQ_FT, SQ_ROWS
from sim.interfaces import Detection, FrameReport

SPEC_DEFAULTS = {
    "sensor_altitude_ft": 15.0,        # fixed per episode
    "sensor_fov_across_deg": 60.0,     # footprint = 2 h tan(fov/2): ~17.3 x 12.4 ft at defaults
    "sensor_fov_along_deg": 45.0,
    "sensor_edge_blur_ft": 1.5,        # weight ramps 0 -> 1 over this distance inside the footprint boundary
    "sensor_w_min": 0.02,              # squares with weight below this are ignored
    # pixels-on-target detection: p_detect = p_detect_max * px^k / (px^k + px_half^k),
    # px = target_size_ft / (footprint_across_ft / image_px_across). Equals 0.5 p_detect_max at px = px_for_half_detect,
    # saturates toward p_detect_max for large px, falls off like px^k for small px.
    "sensor_image_px_across": 1280,
    "sensor_target_size_in": 4.7,      # ~12 cm PFM-1
    "sensor_px_for_half_detect": 12.0,
    "sensor_detect_k": 3.0,
    "sensor_p_detect_max": 0.95,
    "sensor_p_fp": 0.01,
    "sensor_gps_sigma_ft": 0.5,        # stationary std per axis at progress 0
    "sensor_gps_sigma_end_ft": 1.5,    # std reached at progress >= sensor_gps_ramp_end (curriculum)
    "sensor_gps_ramp_end": 0.5,
    "sensor_gps_tau_s": 20.0,          # OU correlation time
    "sensor_frame_interval_s": 0.5,
}


def footprint_dims(spec: dict) -> tuple[float, float]:
    """(across, along) footprint in ft from altitude and FOV."""
    h = spec["sensor_altitude_ft"]
    return (2 * h * np.tan(np.radians(spec["sensor_fov_across_deg"]) / 2),
            2 * h * np.tan(np.radians(spec["sensor_fov_along_deg"]) / 2))


def p_detect_for(spec: dict) -> float:
    across, _ = footprint_dims(spec)
    px = (spec["sensor_target_size_in"] / 12.0) / (across / spec["sensor_image_px_across"])
    k = spec["sensor_detect_k"]
    return float(spec["sensor_p_detect_max"] * px**k / (px**k + spec["sensor_px_for_half_detect"] ** k))


def p_detect_at(spec: dict, altitude_ft: float) -> float:
    """p_detect for a given altitude (same model as p_detect_for)."""
    return p_detect_for({**SPEC_DEFAULTS, **spec, "sensor_altitude_ft": float(altitude_ft)})


def footprint_weights(bx: float, by: float, heading: float, across: float, along: float, blur: float):
    """Squares under a head-aligned footprint centered at (bx, by). Returns (r0, c0, xs, ys, w) with xs/ys the
    square-center coordinate grids (nr, nc) and w the edge weights (ramp 0 -> 1 over `blur` ft inside the boundary),
    or None if empty/out of field."""
    ha, hc = 0.5 * along, 0.5 * across
    c, sn = np.cos(heading), np.sin(heading)
    hx, hy = abs(c) * ha + abs(sn) * hc, abs(sn) * ha + abs(c) * hc
    c0 = max(int((bx - hx) // SQ_FT), 0); c1 = min(int((bx + hx) // SQ_FT), SQ_COLS - 1)
    r0 = max(int((by - hy) // SQ_FT), 0); r1 = min(int((by + hy) // SQ_FT), SQ_ROWS - 1)
    if c1 < c0 or r1 < r0:
        return None
    xs = ((np.arange(c0, c1 + 1) + 0.5) * SQ_FT)[None, :]
    ys = ((np.arange(r0, r1 + 1) + 0.5) * SQ_FT)[:, None]
    dx, dy = xs - bx, ys - by
    u = dx * c + dy * sn          # along-track
    v = -dx * sn + dy * c         # across-track
    w = np.clip((ha - np.abs(u)) / blur, 0.0, 1.0) * np.clip((hc - np.abs(v)) / blur, 0.0, 1.0)
    return r0, c0, np.broadcast_to(xs, w.shape), np.broadcast_to(ys, w.shape), w


class ScanResult(NamedTuple):
    """Ground-truth bookkeeping of the last frame (for tests/demos only; never given to the Mapper)."""
    xy: np.ndarray        # (N, 2) believed-frame positions of reported detections
    is_tp: np.ndarray     # (N,) bool
    n_tp: int
    n_fp: int
    n_mines_in_view: int  # squares with a mine in the footprint (w > w_min)
    n_clear_in_view: int


class Sensor:
    def __init__(self, n_drones: int, rng: np.random.Generator, spec: dict | None = None):
        self.spec = {**SPEC_DEFAULTS, **(spec or {})}
        self.n = n_drones
        self.last_frame = np.full(n_drones, -np.inf)
        self.across, self.along = footprint_dims(self.spec)
        self.p_detect = p_detect_for(self.spec)
        self.sigma = self.spec["sensor_gps_sigma_ft"]
        self.last_stats: ScanResult | None = None
        self.reset_drift(rng)

    def set_progress(self, p: float) -> None:
        """Curriculum hook: sigma ramps linearly from sensor_gps_sigma_ft (p=0) to sensor_gps_sigma_end_ft
        (p >= sensor_gps_ramp_end). Existing drift errors are rescaled so they stay consistent with the new std."""
        s = self.spec
        f = min(max(p / max(s["sensor_gps_ramp_end"], 1e-9), 0.0), 1.0)
        new = s["sensor_gps_sigma_ft"] + f * (s["sensor_gps_sigma_end_ft"] - s["sensor_gps_sigma_ft"])
        if self.sigma > 1e-9:
            self.err *= new / self.sigma
        self.sigma = new

    def reset_drift(self, rng):
        self.err = rng.normal(0.0, self.sigma, (self.n, 2))

    def step_drift(self, i: int, dt: float, rng) -> np.ndarray:
        s = self.spec
        a = np.exp(-dt / s["sensor_gps_tau_s"])
        self.err[i] = a * self.err[i] + self.sigma * np.sqrt(1.0 - a * a) * rng.standard_normal(2)
        return self.err[i]

    def footprint_weights(self, bx: float, by: float, heading: float):
        return footprint_weights(bx, by, heading, self.across, self.along, self.spec["sensor_edge_blur_ft"])

    def scan(self, drone, t: float, dt: float, mine_sq: np.ndarray, rng,
             mine_xy: np.ndarray | None = None) -> FrameReport | None:
        """Call every tick. Advances the drone's GPS error; sets drone.gps_err; returns a FrameReport if a frame is due
        (None otherwise, or if the footprint is off the field). Ground-truth stats of the frame go to self.last_stats.

        mine_sq (150, 40) bool ground truth; mine_xy optional (150, 40, 2) exact mine positions (default: square
        center). The camera sees the TRUE footprint; detections are reported at true position + GPS error."""
        s = self.spec
        i = drone.idx
        e = self.step_drift(i, dt, rng)
        drone.gps_err = e
        if t - self.last_frame[i] < s["sensor_frame_interval_s"] - 1e-9:
            return None
        self.last_frame[i] = t
        fp = self.footprint_weights(drone.pos[0], drone.pos[1], drone.heading)
        if fp is None:
            return None
        r0, c0, xs, ys, w = fp
        nr, nc = w.shape
        has_mine = mine_sq[r0:r0 + nr, c0:c0 + nc]
        pd, pfp = self.p_detect, s["sensor_p_fp"]
        det = rng.random(w.shape) < np.where(has_mine, pd, pfp)
        inview = w > s["sensor_w_min"]
        rep = det & inview
        tp = rep & has_mine
        if mine_xy is not None:
            true_xy = mine_xy[r0:r0 + nr, c0:c0 + nc]
        else:
            true_xy = np.stack([xs, ys], -1)
        fp_xy = np.stack([xs, ys], -1) + rng.uniform(-0.5 * SQ_FT, 0.5 * SQ_FT, (nr, nc, 2))
        rep_xy = np.where(tp[..., None], true_xy, fp_xy)[rep] + e
        self.last_stats = ScanResult(rep_xy, tp[rep], int(tp.sum()), int((rep & ~has_mine).sum()),
                                     int((has_mine & inview).sum()), int((~has_mine & inview).sum()))
        bel = drone.pos + e
        return FrameReport(
            drone_id=i, t=float(t), pose_xy=(float(bel[0]), float(bel[1])), yaw=float(drone.heading),
            altitude_ft=float(s["sensor_altitude_ft"]), pos_sigma_ft=float(self.sigma),
            footprint_ft=(float(self.across), float(self.along)),
            detections=[Detection((float(x), float(y)), 1.0) for x, y in rep_xy])
