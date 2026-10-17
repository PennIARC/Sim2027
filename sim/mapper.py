"""Mapper: the single stateful component between vision/drones and the allocator (see sim/interfaces.py).

Fuses FrameReports into the belief maps, keeps per-drone state (latest DroneStatus, GPS-offset estimate, block
holdings) and builds AllocatorInput snapshots. It never sees ground truth: everything arrives via FrameReport /
DroneStatus, and the allocation bookkeeping via record_allocation / release_blocks.

Belief storage is sim.belief.Belief (1 ft grids + the Bayesian footprint update); the Mapper owns one instance.

Drift handling (mapper_drift_mode = "correlated", default)
----------------------------------------------------------
The GPS error of one drone is an OU process (tau ~ 20 s), so consecutive frames share nearly one offset e. Per drone the
Mapper runs a 2-axis Kalman filter on e (isotropic: mean m, scalar variance v per axis):
  predict   m <- a m, v <- a^2 v + s^2 (1 - a^2), a = exp(-dt / tau), s = reported pos_sigma_ft (stationary std);
  measure   a detection that is associated to a known mine at position M gives z = det - M = e + noise,
            noise variance R = var(M) + det_sigma^2.
Association: a detection is matched to an ANCHOR, a mine hypothesis (list of per-drone position contributions) whose
square currently has P > mapper_anchor_p, within a chi-square gate on (z - m)^2 / (v + R); ambiguous matches (two anchors
nearly equally close) are skipped. Anti-self-confirmation: a drone's own contributions are corrected with its own offset
estimate, so using them again would just return the residual it already absorbed. For drone d an anchor therefore only
uses contributions from OTHER drones, plus d's own contributions older than 2 tau (offset decorrelated, e^-2 ~ 0.14), and
needs >= 2 supporting frames overall (a single false positive never anchors anything). Each contribution carries the
variance (v_post + det_sigma^2) of the offset estimate that corrected it, so weakly-known anchors give weak measurements.
Where there is no revisit / no other drone, nothing is observable and the filter correctly stays at its prior (v = s^2).

Frame update: the footprint is placed at pose - m (corrected position), detections are mapped to squares at det - m, and
the spreading kernel uses the POSTERIOR std sqrt(v) instead of the full sigma.
Tempering (shared offset => repeated looks are not independent evidence about the neighbours): a per-drone, per-square
accumulator A of full-weight looks decays with tau; the spread (off-square) part of each new look is scaled by
1 - min(A, 1), i.e. the first look within a correlation time spreads fully, later looks only sharpen their own square.
Modes "independent" (full-sigma spreading, frames independent) and "none" (no spreading) exist for comparison.
"""
import numpy as np

from sim.belief import Belief
from sim.geometry import BEL_COLS, BEL_ROWS, BLK_COLS, N_BLOCKS, SQ_COLS, SQ_FT, SQ_ROWS, square_of
from sim.interfaces import Allocation, AllocatorInput, DroneStatus, FrameReport
from sim.sensor import SPEC_DEFAULTS as SENSOR_DEFAULTS
from sim.sensor import footprint_weights, p_detect_at

SPEC_DEFAULTS = {
    "mapper_drift_mode": "correlated",   # "correlated" | "independent" | "none"
    "mapper_det_sigma_ft": 0.15,         # detector localisation noise (per axis)
    "mapper_gate_chi2": 9.21,            # 99% chi-square gate (2 dof) on association innovations
    "mapper_anchor_p": 0.5,              # a square must have P above this to act as an anchor
    "mapper_anchor_min_frames": 2,       # supporting frames (any drone) before an anchor may be used
    "mapper_own_age_taus": 2.0,          # a drone's own contributions are reusable after this many tau
    "mapper_temper": True,               # temper spreading of repeated looks within a correlation time
}


class _Tracker:
    __slots__ = ("m", "v", "t", "sigma", "A")

    def __init__(self):
        self.m = np.zeros(2)
        self.v = 0.0
        self.t = None
        self.sigma = 0.0
        self.A = np.zeros((SQ_ROWS, SQ_COLS), np.float32)


class _Anchor:
    __slots__ = ("xy", "contribs")   # contribs: drone -> [x, y, var, n_frames, t_last]

    def __init__(self, xy):
        self.xy = np.asarray(xy, float)
        self.contribs: dict[int, list] = {}


class Mapper:
    def __init__(self, prior_sq: np.ndarray, n_drones: int, spec: dict | None = None):
        self.spec = {**SENSOR_DEFAULTS, **SPEC_DEFAULTS, **(spec or {})}
        self.n = n_drones
        self.belief = Belief(prior_sq, self.spec)
        self.sigma = float(self.spec["sensor_gps_sigma_ft"])
        self.tau = float(self.spec["sensor_gps_tau_s"])
        self.status: list[DroneStatus | None] = [None] * n_drones
        self._trk = [_Tracker() for _ in range(n_drones)]
        self._anchors: list[_Anchor] = []
        self._anc_xy = np.zeros((256, 2))
        self._pd_cache: dict[float, float] = {}
        self.mode = self.spec["mapper_drift_mode"]

    # ---- public API ----------------------------------------------------------------------------------
    def set_pos_sigma(self, sigma_ft: float) -> None:
        """Current drift level (curriculum in the sim; GPS/EKF covariance in reality)."""
        self.sigma = float(sigma_ft)

    def p_squares(self) -> np.ndarray:
        """(150, 40) per-square mine probability."""
        return self.belief.p_squares()

    def offset_estimate(self, drone_id: int) -> tuple[np.ndarray, float]:
        """Current (mean offset (2,), posterior std per axis) of a drone's GPS error. Diagnostics only."""
        t = self._trk[drone_id]
        return t.m.copy(), float(np.sqrt(max(t.v, 0.0)))

    def update_drone(self, status: DroneStatus) -> None:
        self.status[status.drone_id] = status

    def record_allocation(self, alloc: Allocation) -> None:
        """Blocks become held by alloc.drone_id; its previous blocks are released."""
        self.belief.clear_drone(alloc.drone_id)
        self.belief.mark_blocks(alloc.drone_id, alloc.blocks)

    def release_blocks(self, drone_id: int, blocks: list[int]) -> None:
        mine = [b for b in blocks if self._block_owner(b) == drone_id]
        self.belief.unmark_blocks(mine)

    def snapshot(self, drone_id: int, t: float, t_end: float) -> AllocatorInput:
        owner = self.belief.allocated[5::10, 5::10].ravel()      # (N_BLOCKS,) owner of each 10 ft block
        assert owner.size == N_BLOCKS
        return AllocatorInput(
            drone_id=drone_id, t=float(t), t_end=float(t_end),
            p_mine=self.belief.p_mine.copy(), seen=self.belief.seen.copy(),
            allocated=self.belief.allocated.copy(),
            others_path_eta=self._others_path_eta(drone_id),
            drones=[self._status_or_default(i, t) for i in range(self.n)],
            pos_sigma_ft=self.sigma,
            block_mask=(owner < 0) | (owner == drone_id))

    def ingest_frame(self, report: FrameReport) -> None:
        sp = self.spec
        d = report.drone_id
        tr = self._trk[d]
        sig = float(report.pos_sigma_ft)
        px, py = report.pose_xy
        dets = np.array([x.xy for x in report.detections], float).reshape(-1, 2)
        if self.mode == "correlated" and sig > 1e-6:
            self._predict(tr, report.t, sig)
            self._associate_and_update(tr, d, dets, report.t)
            m, v = tr.m, tr.v
        elif self.mode == "independent":
            m, v = np.zeros(2), sig * sig
        else:
            m, v = np.zeros(2), 0.0
        across, along = report.footprint_ft
        fp = footprint_weights(px - m[0], py - m[1], report.yaw, across, along, sp["sensor_edge_blur_ft"])
        if fp is not None:
            r0, c0, xs, ys, w = fp
            nr, nc = w.shape
            det = np.zeros((nr, nc), bool)
            if len(dets):
                pc = dets - m
                rr = np.floor(pc[:, 1] / SQ_FT).astype(int) - r0
                cc = np.floor(pc[:, 0] / SQ_FT).astype(int) - c0
                ok = (rr >= 0) & (rr < nr) & (cc >= 0) & (cc < nc)
                det[rr[ok], cc[ok]] = True
            w = np.where(w > sp["sensor_w_min"], w, 0.0)
            alt = round(float(report.altitude_ft), 3)
            pd = self._pd_cache.get(alt)
            if pd is None:
                pd = self._pd_cache[alt] = p_detect_at(sp, alt)
            gain = None
            spread = float(np.sqrt(v)) / SQ_FT
            if self.mode == "correlated" and sp["mapper_temper"] and spread > 0.05:
                sl = (slice(r0, r0 + nr), slice(c0, c0 + nc))
                gain = np.clip(1.0 - tr.A[sl], 0.0, 1.0)
            self.belief.apply_observation(r0, c0, det, w, pd, sp["sensor_p_fp"], spread, gain)
            if gain is not None:
                sl = (slice(r0, r0 + nr), slice(c0, c0 + nc))
                tr.A[sl] += w.astype(np.float32)
        if self.mode == "correlated" and sig > 1e-6 and len(dets):
            self._add_contributions(d, dets - tr.m, tr.v, report.t)

    # ---- drift tracking ------------------------------------------------------------------------------
    def _predict(self, tr: _Tracker, t: float, sig: float) -> None:
        if tr.t is None:
            tr.m[:] = 0.0
            tr.v = sig * sig
        else:
            if tr.sigma > 1e-9 and abs(sig - tr.sigma) > 1e-9:
                f = sig / tr.sigma
                tr.m *= f
                tr.v *= f * f
            dt = max(t - tr.t, 0.0)
            a = float(np.exp(-dt / self.tau))
            tr.m *= a
            tr.v = a * a * tr.v + sig * sig * (1.0 - a * a)
            tr.A *= np.float32(a)
        tr.t, tr.sigma = t, sig

    def _fused_anchor(self, anc: _Anchor, d: int, t: float):
        """(xy, var, n_frames) of the anchor as seen by drone d (own recent contributions excluded) or None."""
        sp = self.spec
        info, sx, n_tot, n_ok = 0.0, np.zeros(2), 0, 0
        for k, (x, y, var, n, tl) in anc.contribs.items():
            n_tot += n
            if k == d and t - tl < sp["mapper_own_age_taus"] * self.tau:
                continue
            n_ok += 1
            info += 1.0 / var
            sx += np.array([x, y]) / var
        if n_ok == 0 or n_tot < sp["mapper_anchor_min_frames"]:
            return None
        return sx / info, 1.0 / info

    def _associate_and_update(self, tr: _Tracker, d: int, dets: np.ndarray, t: float) -> None:
        if not self._anchors or not len(dets):
            return
        sp = self.spec
        r2 = sp["mapper_det_sigma_ft"] ** 2
        n = len(self._anchors)
        axy = self._anc_xy[:n]
        reach = 3.0 * np.sqrt(2.0 * tr.sigma ** 2 + r2) + 1.0
        for det in dets:
            dist = np.hypot(*(axy - (det - tr.m)).T)
            cand = []
            for j in np.flatnonzero(dist < reach):
                anc = self._anchors[j]
                pr, pc = square_of(*anc.xy)
                if self.belief._p4[pr, :, pc, :].sum() <= sp["mapper_anchor_p"]:
                    continue
                f = self._fused_anchor(anc, d, t)
                if f is None:
                    continue
                z = det - f[0]
                S = tr.v + f[1] + r2
                cand.append((float(((z - tr.m) ** 2).sum() / S), z, f[1] + r2))
            if not cand:
                continue
            cand.sort(key=lambda c: c[0])
            if cand[0][0] > sp["mapper_gate_chi2"] or (len(cand) > 1 and cand[1][0] < 2.0 * cand[0][0] + 1.0):
                continue
            _, z, R = cand[0]
            K = tr.v / (tr.v + R)
            tr.m += K * (z - tr.m)
            tr.v *= (1.0 - K)

    def _add_contributions(self, d: int, pcs: np.ndarray, v: float, t: float) -> None:
        r2 = self.spec["mapper_det_sigma_ft"] ** 2
        var = v + r2
        rm = max(1.2, 2.0 * np.sqrt(var))
        for p in pcs:
            n = len(self._anchors)
            j = -1
            if n:
                dist = np.hypot(*(self._anc_xy[:n] - p).T)
                k = int(np.argmin(dist))
                if dist[k] < rm:
                    j = k
            if j < 0:
                if n == len(self._anc_xy):
                    self._anc_xy = np.vstack([self._anc_xy, np.zeros_like(self._anc_xy)])
                self._anchors.append(_Anchor(p))
                self._anc_xy[n] = p
                j = n
            anc = self._anchors[j]
            c = anc.contribs.get(d)
            if c is None:
                anc.contribs[d] = [p[0], p[1], var, 1, t]
            else:
                if var < c[2] or t - c[4] > self.tau:
                    c[0], c[1], c[2] = p[0], p[1], var
                c[3] += 1
                c[4] = t
            w = np.array([[1.0 / c[2], c[0] / c[2], c[1] / c[2]] for c in anc.contribs.values()]).sum(0)
            anc.xy = np.array([w[1] / w[0], w[2] / w[0]])
            self._anc_xy[j] = anc.xy

    # ---- snapshot helpers ----------------------------------------------------------------------------
    def _block_owner(self, b: int) -> int:
        r, c = divmod(int(b), BLK_COLS)
        return int(self.belief.allocated[r * 10 + 5, c * 10 + 5])

    def _status_or_default(self, i: int, t: float) -> DroneStatus:
        s = self.status[i]
        if s is not None:
            return s
        return DroneStatus(i, float(t), (0.0, 0.0), 0.0, np.zeros((0, 2)), np.zeros(0), True)

    def _others_path_eta(self, drone_id: int) -> np.ndarray:
        out = np.zeros((BEL_ROWS, BEL_COLS), np.float32)
        for i, s in enumerate(self.status):
            if i == drone_id or s is None or s.idle or len(s.remaining_path) == 0:
                continue
            pts = np.asarray(s.remaining_path, float)
            eta = np.asarray(s.remaining_eta_s, float)
            seg = np.hypot(*np.diff(pts, axis=0).T) if len(pts) > 1 else np.zeros(0)
            S = np.r_[0.0, np.cumsum(seg)]
            ss = np.arange(0.0, S[-1], 0.5) if S[-1] > 0 else np.zeros(1)
            ss = np.r_[ss, S[-1]]
            x, y = np.interp(ss, S, pts[:, 0]), np.interp(ss, S, pts[:, 1])
            e = np.interp(ss, S, eta)
            emax = max(float(eta[-1]), 1e-6)
            val = np.clip(1.0 - e / emax, 0.0, 1.0).astype(np.float32)
            ri = np.clip(np.floor(y).astype(int), 0, BEL_ROWS - 1)
            ci = np.clip(np.floor(x).astype(int), 0, BEL_COLS - 1)
            np.maximum.at(out, (ri, ci), val)
        return out
