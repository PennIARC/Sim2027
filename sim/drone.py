"""Drone kinematics, route building and path following.

Route: [straight leg from current pose] + per block a straight pass through the block center along one axis
(sim.routing). Pass length 10 ft with a 12 ft wide camera: every point of the block is within 5 ft of the
center line (<= 6 ft half-width), so one pass covers it (1 ft margin vs GPS drift). Each pass is extended by
drone_lead_ft outside the block so the turn arcs happen mostly outside it. The polyline's corners are
rounded with circular fillets of radius drone_fillet_radius_ft (> min turn radius), the result is interpolated with a
cubic B-spline (scipy splprep) and resampled by arc length. Speed along the path is limited by curvature
(v <= yaw_rate_max / curvature, floor drone_v_floor), by accel/decel limits (backward pass + clipped acceleration).
Known gaps: no drone-drone avoidance; corners sharper than the geometry allows (e.g. 180 deg reversal between very
close points) get a smaller fillet radius, and are flown slowly instead (yaw-rate limit still respected).
Heading convention: radians from +x axis (math convention); drones start at y=0 heading +y.
"""
import numpy as np
from scipy.interpolate import splev, splprep

from sim.geometry import FIELD_L_FT, FIELD_W_FT
from sim.routing import order_blocks, pass_points

SPEC_DEFAULTS = {
    "drone_dt_s": 0.1,
    "drone_cruise_speed_ftps": 10.0,
    "drone_max_accel_ftps2": 4.0,
    "drone_max_decel_ftps2": 6.0,
    "drone_max_yaw_rate_radps": 2.0,
    "drone_min_turn_radius_ft": 4.0,
    "drone_fillet_radius_ft": 5.0,
    "drone_lead_ft": 5.0,
    "drone_v_floor_ftps": 1.5,
    "drone_path_ds_ft": 0.5,
    "drone_start_xs": (10.0, 30.0, 50.0, 70.0),
    "drone_start_y": 0.0,
    "drone_remaining_pts": 16,
}


def start_poses(spec: dict | None = None):
    sp = {**SPEC_DEFAULTS, **(spec or {})}
    return [(np.array([x, sp["drone_start_y"]]), np.pi / 2) for x in sp["drone_start_xs"]]


def _clamp(p):
    return np.array([np.clip(p[0], 0.3, FIELD_W_FT - 0.3), np.clip(p[1], 0.0, FIELD_L_FT)])


def _fillet(pts: np.ndarray, r: float):
    """Round polyline corners with circular arcs of radius <= r. Returns dense point list (also returns min radius used)."""
    n = len(pts)
    seg = np.diff(pts, axis=0)
    ln = np.linalg.norm(seg, axis=1)
    dirs = seg / ln[:, None]
    turn = np.zeros(n)
    turn[1:-1] = np.arccos(np.clip((dirs[:-1] * dirs[1:]).sum(1), -1, 1))
    corner = turn > np.radians(1.0)
    out = [pts[0]]
    r_min_used = np.inf
    for i in range(1, n - 1):
        d1, d2 = dirs[i - 1], dirs[i]
        th = np.arccos(np.clip(d1 @ d2, -1, 1))
        if th < np.radians(1.0):
            out.append(pts[i]); continue
        b_in = ln[i - 1] * (0.5 if corner[i - 1] else 1.0)
        b_out = ln[i] * (0.5 if corner[i + 1] else 1.0)
        t = min(r * np.tan(th / 2), b_in, b_out)
        re = t / np.tan(th / 2)
        r_min_used = min(r_min_used, re)
        A = pts[i] - d1 * t
        sgn = np.sign(d1[0] * d2[1] - d1[1] * d2[0])
        C = A + sgn * np.array([-d1[1], d1[0]]) * re
        v = A - C
        m = max(3, int(np.ceil(th / np.radians(10))))
        for k in range(m + 1):
            a = sgn * th * k / m
            ca, sa = np.cos(a), np.sin(a)
            out.append(C + np.array([ca * v[0] - sa * v[1], sa * v[0] + ca * v[1]]))
    out.append(pts[-1])
    return out, r_min_used


class Drone:
    def __init__(self, idx: int, pos, heading: float, spec: dict | None = None):
        self.spec = {**SPEC_DEFAULTS, **(spec or {})}
        self.idx = idx
        self.pos = np.array(pos, float)
        self.heading = float(heading)
        self.speed = 0.0
        self.gps_err = np.zeros(2)       # set by Sensor.scan
        self.route: list[tuple[int, int]] = []
        self.has_path = False
        self.done = True                 # True when no route is being flown (allocator should assign one)
        self.path_min_fillet_r = np.inf
        self._xy = self._psi = self._kap = self._vlim = None
        self._pending: list[int] = []
        self._exit_s: list[float] = []
        self._reported = 0

    @property
    def believed_pos(self) -> np.ndarray:
        return self.pos + self.gps_err

    # ---- route building -----------------------------------------------------------------------------
    def assign(self, blocks, rng: np.random.Generator | None = None):
        """Order the blocks, build the smooth path starting from the current pose. Returns the route."""
        sp = self.spec
        route, _ = order_blocks(blocks, self.pos, self.heading, sp, rng)
        self.route = route
        lead = sp["drone_lead_ft"]
        pts = [self.pos.copy()]
        if self.speed > 0.5:
            pts.append(_clamp(self.pos + np.array([np.cos(self.heading), np.sin(self.heading)]) * sp["drone_fillet_radius_ft"]))
        exits = []
        for b, o in route:
            e, x, d = pass_points(b, o)
            for p in (e - d * lead, e, 0.5 * (e + x), x, x + d * lead):
                pts.append(_clamp(p))
            exits.append(x)
        clean = [pts[0]]
        for p in pts[1:]:
            if np.linalg.norm(p - clean[-1]) > 0.5:
                clean.append(p)
        pts = np.array(clean)
        self._build_spline(pts)
        # arc length at which each block's exit point is reached
        idx0, self._exit_s = 0, []
        for x in exits:
            j = idx0 + int(np.argmin(np.linalg.norm(self._xy[idx0:] - x, axis=1)))
            self._exit_s.append(self._S[j]); idx0 = j
        self._pending = [b for b, _ in route]
        self._reported = 0
        self.s, self.done, self.has_path = 0.0, len(pts) < 2 or self._L < 0.5, True
        return route

    def _build_spline(self, pts):
        sp = self.spec
        if len(pts) < 2:
            self._L = 0.0; self._S = np.zeros(1); self._xy = pts[:1]; self._psi = np.array([self.heading])
            self._kap = np.zeros(1); self._vlim = np.array([sp["drone_v_floor_ftps"]]); self._T = np.zeros(1)
            return
        ctrl, self.path_min_fillet_r = _fillet(pts, sp["drone_fillet_radius_ft"])
        dense = [ctrl[0]]
        for p in ctrl[1:]:
            prev = dense[-1]
            k = int(np.ceil(np.linalg.norm(p - prev) / 3.0))
            for f in range(1, k):
                dense.append(prev + (p - prev) * f / k)
            dense.append(p)
        dense = np.array(dense)
        keep = np.r_[True, np.linalg.norm(np.diff(dense, axis=0), axis=1) > 1e-3]
        dense = dense[keep]
        kdeg = min(3, len(dense) - 1)
        tck, _ = splprep([dense[:, 0], dense[:, 1]], s=0, k=kdeg)
        uf = np.linspace(0, 1, 12 * len(dense))
        xf, yf = splev(uf, tck)
        cum = np.r_[0, np.cumsum(np.hypot(np.diff(xf), np.diff(yf)))]
        L = cum[-1]
        ds = sp["drone_path_ds_ft"]
        S = np.r_[np.arange(0, L, ds), L] if L - (np.floor(L / ds) * ds) > 1e-6 else np.arange(0, L + 1e-9, ds)
        u = np.interp(S, cum, uf)
        x, y = splev(u, tck); dx, dy = splev(u, tck, der=1); ddx, ddy = splev(u, tck, der=2)
        kap = np.abs(dx * ddy - dy * ddx) / np.maximum(dx * dx + dy * dy, 1e-12) ** 1.5
        self._xy = np.clip(np.stack([x, y], 1), [0.0, 0.0], [FIELD_W_FT, FIELD_L_FT])
        self._S, self._L = S, float(L)
        self._psi = np.arctan2(dy, dx)
        self._kap = kap
        vl = np.clip(sp["drone_max_yaw_rate_radps"] / np.maximum(kap, 1e-9), sp["drone_v_floor_ftps"], sp["drone_cruise_speed_ftps"])
        for i in range(len(S) - 2, -1, -1):          # backward pass: be able to brake in time
            vl[i] = min(vl[i], np.sqrt(vl[i + 1] ** 2 + 2 * sp["drone_max_decel_ftps2"] * (S[i + 1] - S[i])))
        self._vlim = vl
        # time estimate with a forward (accel-limited) pass from the current speed
        vf = vl.copy(); vf[0] = min(vl[0], max(self.speed, 0.0)) if self.speed > 0 else 0.0
        for i in range(1, len(S)):
            vf[i] = min(vf[i], np.sqrt(vf[i - 1] ** 2 + 2 * sp["drone_max_accel_ftps2"] * (S[i] - S[i - 1])))
        vm = np.maximum(0.5 * (vf[1:] + vf[:-1]), 0.3)
        self._T = np.r_[0, np.cumsum(np.diff(S) / vm)]

    # ---- following ----------------------------------------------------------------------------------
    def _at(self, s):
        S = self._S
        i = min(max(int(np.searchsorted(S, s, "right")) - 1, 0), len(S) - 2)
        f = (s - S[i]) / max(S[i + 1] - S[i], 1e-9)
        return i, min(max(f, 0.0), 1.0)

    def step(self, dt: float) -> None:
        if not self.has_path or self.done:
            return
        sp = self.spec
        v, s = self.speed, self.s
        i, f = self._at(min(s + v * dt, self._L))
        vp = self._vlim[i] * (1 - f) + self._vlim[i + 1] * f
        vn = min(max(vp, v - sp["drone_max_decel_ftps2"] * dt), v + sp["drone_max_accel_ftps2"] * dt)
        s = s + 0.5 * (v + vn) * dt
        self.speed = vn
        if s >= self._L - 1e-9:
            s = self._L; self.done = True
        self.s = s
        i, f = self._at(s)
        self.pos = self._xy[i] * (1 - f) + self._xy[i + 1] * f
        a0, a1 = self._psi[i], self._psi[i + 1]
        self.heading = float(a0 + f * ((a1 - a0 + np.pi) % (2 * np.pi) - np.pi))

    def pop_completed_blocks(self) -> list[int]:
        """Blocks whose in-block pass has been flown since the last call (use to release allocations)."""
        out = []
        while self._reported < len(self._exit_s) and (self.s >= self._exit_s[self._reported] or self.done):
            out.append(self._pending[self._reported]); self._reported += 1
        return out

    def remaining_path(self, n: int | None = None) -> np.ndarray:
        """(n, 2) points evenly spaced by arc length from the current position to the route end (last point repeated
        if the route is finished). Excluding nothing: the first point is the current position."""
        n = n or self.spec["drone_remaining_pts"]
        if not self.has_path or self.done:
            return np.repeat(self.pos[None, :], n, 0)
        ss = np.linspace(self.s, self._L, n)
        return np.stack([np.interp(ss, self._S, self._xy[:, 0]), np.interp(ss, self._S, self._xy[:, 1])], 1)

    def eta(self) -> float:
        """Estimated seconds until the route is complete."""
        if not self.has_path or self.done:
            return 0.0
        return float(self._T[-1] - np.interp(self.s, self._S, self._T))

    def path_curvature(self) -> np.ndarray:
        return self._kap
