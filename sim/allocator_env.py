"""AllocatorEnv: the event-driven cell-allocation task for the PPO stack (satisfies rl/env.py).

One env step = one decision: the single central agent assigns K blocks (10 ft blocks, sim/geometry.py) to the ONE drone that
just went idle. At t=0 all drones are idle and are assigned in drone-index order; afterwards the sim advances at the drone dt
(drones fly, the sensor emits FrameReports into the Mapper, finished blocks are released) until some drone finishes its route.
The episode is a fixed `episode_s` of sim time; at the end the planner picks a path on the Mapper's belief and the official
scorer is run against the true mines: that score is the ONLY reward (terminal, terminated=True).

The policy sees only what sim.interfaces.AllocatorInput carries: `obs_from_input(inp, spec)` is the pure function that turns a
Mapper snapshot into the observation dict, and `allocate(policy, inp)` is the stateless allocator entry point.

Observation (dict):
  maps    float16 (5, 300, 80): p_mine (x alloc_pmine_scale, clipped to 1), seen, allocated-to-other-drones, others' remaining
          paths (time-weighted raster from the Mapper: 1 = reached now -> 0 at route end), self-position Gaussian blob
  scalars float32 (3,): time remaining fraction, mean seen fraction (progress proxy; the decision count is not in AllocatorInput),
          position sigma / 1.5 ft
  mask    bool (240,): False only for blocks held by OTHER drones
Action: MaskedChoice(n=240, k=alloc_k): block ids in pick order.
"""
import collections

import numpy as np

from rl.spaces import Box, MaskedChoice
from sim import belief as belief_mod
from sim import drone as drone_mod
from sim import mapper as mapper_mod
from sim import mines as mines_mod
from sim import planner as planner_mod
from sim import routing as routing_mod
from sim import sensor as sensor_mod
from sim.geometry import BEL_COLS, BEL_ROWS, BLK_COLS, N_BLOCKS
from sim.interfaces import Allocation, AllocatorInput, DroneStatus
from sim.scoring import score as official_score

N_CHANNELS = 5
N_SCALARS = 3
CHANNELS = ("p_mine", "seen", "allocated_others", "others_path_eta", "self_blob")

ENV_DEFAULTS = {
    "alloc_k": 4,                  # blocks per decision
    "episode_s": 420.0,            # fixed episode length, sim seconds
    "alloc_n_drones": 4,
    "alloc_pmine_scale": 8.0,      # p_mine channel multiplier (per-cell P is ~prior/4)
    "alloc_self_sigma_ft": 6.0,    # std of the self-position blob
    "alloc_status_interval_s": 1.0,  # how often DroneStatus is pushed to the Mapper between decisions
    "alloc_reward_scale": 1.0,     # terminal reward = score * this
}


def _merged_defaults() -> dict:
    """SPEC_DEFAULTS of every sim module the env drives, plus the env's own keys. Colliding keys must agree."""
    mods = [mines_mod, sensor_mod, drone_mod, routing_mod, planner_mod]
    out: dict = {}
    for name, d in [(m.__name__, m.SPEC_DEFAULTS) for m in mods] + [("allocator_env", ENV_DEFAULTS)] + _extra_defaults():
        for k, v in d.items():
            if k in out and out[k] != v:
                raise KeyError(f"spec key {k!r} defined with different defaults ({name})")
            out[k] = v
    return out


def _extra_defaults() -> list:
    """Defaults of the Mapper and of the Belief it owns."""
    return [("mapper", mapper_mod.SPEC_DEFAULTS), ("belief", belief_mod.SPEC_DEFAULTS)]


# ---------------------------------------------------------------------------------------------------------------------
# observation: the policy's view of an AllocatorInput
# ---------------------------------------------------------------------------------------------------------------------
_XC = (np.arange(BEL_COLS) + 0.5).astype(np.float32)
_YC = (np.arange(BEL_ROWS) + 0.5).astype(np.float32)


def obs_from_input(inp: AllocatorInput, spec: dict | None = None) -> dict:
    """AllocatorInput -> observation dict (see module docstring). Uses nothing but the fields of `inp`."""
    s = {**ENV_DEFAULTS, **(spec or {})}
    maps = np.empty((N_CHANNELS, BEL_ROWS, BEL_COLS), np.float16)
    maps[0] = np.minimum(inp.p_mine * s["alloc_pmine_scale"], 1.0)
    maps[1] = inp.seen
    maps[2] = (inp.allocated >= 0) & (inp.allocated != inp.drone_id)
    maps[3] = inp.others_path_eta
    px, py = inp.drones[inp.drone_id].pose_xy
    sg = s["alloc_self_sigma_ft"]
    maps[4] = np.exp(-0.5 * ((_YC - py) / sg) ** 2)[:, None] * np.exp(-0.5 * ((_XC - px) / sg) ** 2)[None, :]
    t_end = max(inp.t_end, 1e-9)
    scalars = np.array([max(inp.t_end - inp.t, 0.0) / t_end, float(inp.seen.mean()), inp.pos_sigma_ft / 1.5], np.float32)
    return {"maps": maps, "scalars": scalars, "mask": np.asarray(inp.block_mask, bool).copy()}


def allocate(policy, inp: AllocatorInput, deterministic: bool = True, spec: dict | None = None, norm=None,
             seed: int | None = None) -> Allocation:
    """Stateless allocator entry point: AllocatorInput -> Allocation using a trained rl.policy.Policy.
    spec defaults to the policy's own (policy.alloc_spec); norm = optional rl.norm.Normalizer; seed seeds torch's RNG when sampling."""
    import torch
    if seed is not None:
        torch.manual_seed(seed)
    obs = obs_from_input(inp, spec or getattr(policy, "alloc_spec", None))
    if norm is not None:
        obs = norm.apply(obs)
    action, _, _ = policy.act(obs, deterministic=deterministic)
    return Allocation(inp.drone_id, [int(b) for b in action])


# ---------------------------------------------------------------------------------------------------------------------
# environment
# ---------------------------------------------------------------------------------------------------------------------
class AllocatorEnv:
    SPEC_DEFAULTS = _merged_defaults()

    def __init__(self, spec: dict):
        self.spec = {**self.SPEC_DEFAULTS, **spec}
        s = self.spec
        self.k, self.n_drones = int(s["alloc_k"]), int(s["alloc_n_drones"])
        self.dt, self.t_end = float(s["drone_dt_s"]), float(s["episode_s"])
        self.obs_space = {"maps": Box((N_CHANNELS, BEL_ROWS, BEL_COLS), 0.0, 1.0, np.float16), "scalars": Box((N_SCALARS,))}
        self.action_space = MaskedChoice(N_BLOCKS, self.k)
        self.rng = np.random.default_rng(0)
        self.progress = 0.0
        mspec = {k: v for k, v in s.items() if k.startswith("mines_")}
        self.prior_sq = np.asarray(mines_mod.prior_density(mspec))
        self.last_input: AllocatorInput | None = None
        self.mapper = self.sensor = None
        self._zero_obs = {"maps": np.zeros(self.obs_space["maps"].shape, np.float16), "scalars": np.zeros(N_SCALARS, np.float32),
                          "mask": np.ones(N_BLOCKS, bool)}

    # ---- helpers --------------------------------------------------------------------------------------------------
    def set_progress(self, progress: float) -> None:
        """Curriculum hook: forwards to the sensor's GPS-drift ramp and tells the Mapper the new sigma."""
        self.progress = float(progress)
        if self.sensor is not None:
            self.sensor.set_progress(self.progress)
            self.mapper.set_pos_sigma(self.sensor.sigma)

    def _status(self, d) -> DroneStatus:
        n = self.spec["drone_remaining_pts"]
        if d.has_path and not d.done:
            path = d.remaining_path(n) + d.gps_err
            ss = np.linspace(d.s, d._L, n)
            eta = np.interp(ss, d._S, d._T) - np.interp(d.s, d._S, d._T)
        else:
            path, eta = np.zeros((0, 2)), np.zeros(0)
        return DroneStatus(d.idx, self.t, tuple(d.believed_pos), d.heading, path, eta, bool(d.done))

    def _push_status(self, which=None) -> None:
        for d in self.drones:
            if which is None or d.idx != which:
                self.mapper.update_drone(self._status(d))

    def _observe(self, drone_id: int) -> dict:
        self._push_status()
        self.last_input = self.mapper.snapshot(drone_id, self.t, self.t_end)
        return obs_from_input(self.last_input, self.spec)

    # ---- Env API --------------------------------------------------------------------------------------------------
    def reset(self, seed: int | None = None, layout: np.ndarray | None = None) -> dict:
        """New episode. seed (re)seeds the env's RNG (mines, sensor, drift); later resets continue the stream.
        layout: optional fixed (150, 40) bool mine grid (evaluation); the RNG is still used for the sensor and mine positions."""
        s = self.spec
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        rng = self.rng
        self.mines = (mines_mod.generate(rng, s) if layout is None else np.asarray(layout, bool))
        self.mine_xy = (np.stack(np.meshgrid(np.arange(self.mines.shape[1]), np.arange(self.mines.shape[0])), -1)
                        + rng.random((*self.mines.shape, 2))) * 2.0
        self.mapper = mapper_mod.Mapper(self.prior_sq, self.n_drones, s)
        self.sensor = sensor_mod.Sensor(self.n_drones, rng, s)
        self.drones = [drone_mod.Drone(i, p, h, s) for i, (p, h) in enumerate(drone_mod.start_poses(s))][:self.n_drones]
        self.t, self.tick, self.decisions = 0.0, 0, 0
        self._next_status = 0.0
        self.set_progress(self.progress)
        self.pending = collections.deque(range(self.n_drones))
        return self._observe(self.pending[0])

    def _advance(self) -> None:
        """Run ticks until some drone is idle or the episode time is up."""
        s, dt = self.spec, self.dt
        n_ticks = int(round(self.t_end / dt))
        while self.tick < n_ticks and not self.pending:
            for d in self.drones:
                d.step(dt)
                done_blocks = d.pop_completed_blocks()
                if done_blocks:
                    self.mapper.release_blocks(d.idx, done_blocks)
                frame = self.sensor.scan(d, self.t, dt, self.mines, rng=self.rng, mine_xy=self.mine_xy)
                if frame is not None:
                    self.mapper.ingest_frame(frame)
            self.tick += 1
            self.t = self.tick * dt
            if self.t >= self._next_status:
                self._push_status()
                self._next_status = self.t + s["alloc_status_interval_s"]
            self.pending.extend(d.idx for d in self.drones if d.done)

    def step(self, action) -> tuple:
        action = [int(a) for a in np.asarray(action).reshape(-1)]
        inp = self.last_input
        if len(action) != self.k or len(set(action)) != self.k or not inp.block_mask[action].all():
            raise ValueError(f"invalid action {action} (k={self.k}, masked or repeated block)")
        did = self.pending.popleft()
        assert did == inp.drone_id
        d = self.drones[did]
        d.assign(action, self.rng)
        self.mapper.record_allocation(Allocation(did, action))
        self.mapper.update_drone(self._status(d))
        self.decisions += 1
        if self.pending:
            return self._observe(self.pending[0]), 0.0, False, False, {}
        self._advance()
        if self.tick < int(round(self.t_end / self.dt)):
            return self._observe(self.pending[0]), 0.0, False, False, {}
        return self._finish()

    def _finish(self) -> tuple:
        s = self.spec
        psq = self.mapper.p_squares()
        path, diag = planner_mod.plan(psq, s)
        res = official_score(path, self.mines, s["planner_A"], s["planner_N"])
        seen = self.mapper.snapshot(0, self.t, self.t_end).seen
        ep = {"score": float(res["score"]), "B": float(res["B"]), "L": float(res["L"]), "W": float(res["W"]), "G": float(path.G),
              "on_path_mines": float(res["on_path_mines"]), "died": float(res["on_path_mines"] > 0),
              "expected_score": float(diag["expected_score"]), "decisions": float(self.decisions),
              "seen_frac": float((seen > 0.5).mean()), "n_mines": float(self.mines.sum()), "planner_s": float(diag["runtime_s"])}
        self.last_input = None
        return self._zero_obs, float(res["score"]) * s["alloc_reward_scale"], True, False, {"episode": ep}
