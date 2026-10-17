"""Belief maps over the 1 ft grid (300, 80), arrays indexed [row=y, col=x].

p_mine   float32  P(the square's mine is in this cell). The 4 cells of a square are mutually exclusive
                  locations, so P(square has mine) = sum of its 4 cells (geometry.belief_to_squares_prob).
seen     float32  soft 0..1 confidence the cell has been observed.
allocated int8    id of the drone whose assigned blocks contain the cell, -1 = nobody.
"""
import numpy as np
from scipy.special import ndtr

from sim.geometry import (BEL_COLS, BEL_PER_SQ, BEL_ROWS, BLK_COLS, BLK_FT, SQ_COLS, SQ_ROWS,
                          belief_to_squares_prob, block_bounds)

SPEC_DEFAULTS = {
    "belief_seen_gain": 0.6,   # each full-weight look closes this fraction of the (1 - seen) gap
}

# Order of channels returned by Belief.observation().
CHANNELS = ("p_mine", "seen", "allocated_self", "allocated_others")


class Belief:
    def __init__(self, prior_sq: np.ndarray, spec: dict | None = None):
        """prior_sq: (150, 40) per-square mine prior (~0.02-0.05). Split equally over the 4 cells."""
        self.spec = {**SPEC_DEFAULTS, **(spec or {})}
        prior_sq = np.asarray(prior_sq, dtype=np.float64)
        assert prior_sq.shape == (SQ_ROWS, SQ_COLS)
        self.prior_sq = prior_sq.astype(np.float32)
        self.p_mine = np.repeat(np.repeat(prior_sq / BEL_PER_SQ**2, BEL_PER_SQ, 0), BEL_PER_SQ, 1).astype(np.float32)
        self.seen = np.zeros((BEL_ROWS, BEL_COLS), np.float32)
        self.allocated = np.full((BEL_ROWS, BEL_COLS), -1, np.int8)
        # (150, 2, 40, 2) views so a square's 4 cells are axes 1 and 3
        self._p4 = self.p_mine.reshape(SQ_ROWS, BEL_PER_SQ, SQ_COLS, BEL_PER_SQ)
        self._s4 = self.seen.reshape(SQ_ROWS, BEL_PER_SQ, SQ_COLS, BEL_PER_SQ)

    # ---- sensing -------------------------------------------------------------------------------------
    def p_squares(self) -> np.ndarray:
        """(150, 40) P(square has a mine)."""
        return belief_to_squares_prob(self.p_mine)

    def apply_observation(self, r0: int, c0: int, detected: np.ndarray, w: np.ndarray,
                          p_detect: float, p_fp: float, spread_sq: float = 0.0,
                          spread_gain: np.ndarray | None = None) -> None:
        """Bayesian update from a look covering squares [r0:r0+nr, c0:c0+nc] (square indices, in bounds).

        detected (nr, nc) bool: detector fired for the square. w (nr, nc) in [0, 1]: edge weight; the likelihood is
        tempered as L**w, i.e. the log-odds shift is w * llr with llr = log(L_mine / L_none) of the outcome.

        Drift-honest form: the GPS error e ~ N(0, sigma^2) is shared by the whole frame, and the outcome seen at
        believed square j is really the outcome at true square j - e. Square k's mine state therefore only matters for
        the outcome at j = k + e, so its likelihood ratio is the EXPECTATION over e:
            LR_k = E_e[ r_(k+e) ] = 1 + sum_j g(k - j) * (r_j - 1),    r_j = (L_mine/L_none)(outcome_j) ** w_j,
        (r = 1 outside the footprint), g = Gaussian kernel of std spread_sq squares integrated per square (mass 1).
        This is one separable convolution of (r - 1) and can spill ~3 sigma beyond the footprint. A detection is
        smeared over the neighborhood; a miss weakens neighbors toward 1 instead of vetoing them. spread_sq = 0 gives
        g = delta, i.e. the plain tempered update (LR = r).
        spread_gain (nr, nc) in [0, 1] (default 1) scales how much of each square's evidence is allowed to spread to its
        neighbors (tempering of repeated looks under a shared offset): F_eff = blur(s*F) + (1-s)*g0*F, g0 = the kernel's
        centre mass, so a fully tempered look still updates its own square but no longer leaks outward.
        The 'seen' gain uses the convolved weight in the same way. Posterior is spread over the 4 cells by scaling."""
        nr, nc = detected.shape
        w = w.astype(np.float64)
        llr = np.where(detected, np.log(p_detect / p_fp), np.log((1.0 - p_detect) / (1.0 - p_fp)))
        F = np.expm1(w * llr)             # likelihood ratio minus 1 (0 where nothing is learned)
        W = w
        R = 0
        if spread_sq > 0.05:
            R = int(np.ceil(3 * spread_sq))
            d = np.arange(-R, R + 1)
            g = np.diff(ndtr((np.r_[d - 0.5, R + 0.5]) / spread_sq))
            g /= g.sum()
            if spread_gain is None:
                F, W = (self._blur(np.pad(x, R), g, R) for x in (F, W))
            else:
                sg, g0 = np.asarray(spread_gain, np.float64), g[R] ** 2
                F, W = (self._blur(np.pad(sg * x, R), g, R) + np.pad((1.0 - sg) * g0 * x, R) for x in (F, W))
        # clip the (possibly enlarged) window to the field
        ra, ca = r0 - R, c0 - R
        ra0, ca0 = max(ra, 0), max(ca, 0)
        rb, cb = min(ra + F.shape[0], SQ_ROWS), min(ca + F.shape[1], SQ_COLS)
        F = F[ra0 - ra:rb - ra, ca0 - ca:cb - ca]
        W = W[ra0 - ra:rb - ra, ca0 - ca:cb - ca]
        sub = self._p4[ra0:rb, :, ca0:cb, :]
        ps = sub.sum(axis=(1, 3), dtype=np.float64)
        a = ps * (1.0 + F)
        post = a / (a + (1.0 - ps))
        factor = np.where(ps > 1e-12, post / np.maximum(ps, 1e-12), 1.0)
        sub *= factor.astype(np.float32)[:, None, :, None]
        sv = self._s4[ra0:rb, :, ca0:cb, :]
        sv += (self.spec["belief_seen_gain"] * W).astype(np.float32)[:, None, :, None] * (1.0 - sv)

    @staticmethod
    def _blur(x, g, R):
        """Separable 'same' convolution of a zero-padded array (pad R each side) -> same shape."""
        out = np.zeros_like(x)
        n0, n1 = x.shape
        for k, gk in enumerate(g):
            sh = k - R
            lo, hi = max(0, -sh), min(n0, n0 - sh)
            out[lo:hi] += gk * x[lo + sh:hi + sh]
        x, out = out, np.zeros_like(x)
        for k, gk in enumerate(g):
            sh = k - R
            lo, hi = max(0, -sh), min(n1, n1 - sh)
            out[:, lo:hi] += gk * x[:, lo + sh:hi + sh]
        return out

    # ---- allocation ----------------------------------------------------------------------------------
    def _block_slice(self, block: int):
        x0, y0, x1, y1 = block_bounds(int(block))
        return slice(int(y0 / 1), int(y1 / 1)), slice(int(x0 / 1), int(x1 / 1))

    def mark_blocks(self, drone: int, blocks) -> None:
        """Assign blocks to a drone (overwrites any previous owner of those cells)."""
        for b in blocks:
            self.allocated[self._block_slice(b)] = drone

    def unmark_blocks(self, blocks) -> None:
        for b in blocks:
            self.allocated[self._block_slice(b)] = -1

    def clear_drone(self, drone: int) -> None:
        self.allocated[self.allocated == drone] = -1

    def mask_for_drone(self, drone: int) -> np.ndarray:
        return (self.allocated == drone).astype(np.float32)

    def mask_for_others(self, drone: int) -> np.ndarray:
        return ((self.allocated >= 0) & (self.allocated != drone)).astype(np.float32)

    # ---- policy observation --------------------------------------------------------------------------
    def observation(self, drone: int | None = None) -> np.ndarray:
        """float32 (4, 300, 80). Channels (see CHANNELS): 0 p_mine, 1 seen, 2 allocated to `drone`,
        3 allocated to any other drone. With drone=None channel 2 is zeros and channel 3 is any allocation."""
        out = np.empty((len(CHANNELS), BEL_ROWS, BEL_COLS), np.float32)
        out[0] = self.p_mine
        out[1] = self.seen
        if drone is None:
            out[2] = 0.0
            out[3] = self.allocated >= 0
        else:
            out[2] = self.allocated == drone
            out[3] = (self.allocated >= 0) & (self.allocated != drone)
        return out
