"""Tiny envs that prove learning end-to-end (used by the smoke tests)."""
import numpy as np

from rl.spaces import Box, MaskedChoice


class PickBestEnv:
    """Pick k of n options per decision for a few decisions; reward only at the end = sum of the chosen true values.

    Hidden values ~ N(0,1) per episode; the agent sees a noisy copy as a (1, rows, n/rows) "map" plus a scalars
    vector (progress, constant). Each decision has a random mask (already-picked options are always masked out).
    A random policy scores ~0; a good one picks the high-value options at every step (credit spans decisions, gamma=1).
    With probability pick_trunc_prob the episode is cut short (truncated, no reward) after a random earlier decision,
    which exercises critic bootstrapping. info["episode"] = {"score": true values picked so far, "decisions": count}.
    """
    SPEC_DEFAULTS = {"pick_n": 20, "pick_k": 3, "pick_decisions": 3, "pick_noise": 0.3, "pick_mask_keep": 0.7, "pick_trunc_prob": 0.2}

    def __init__(self, spec: dict):
        self.n, self.k, self.horizon = spec["pick_n"], spec["pick_k"], spec["pick_decisions"]
        self.noise, self.keep, self.trunc_prob = spec["pick_noise"], spec["pick_mask_keep"], spec["pick_trunc_prob"]
        rows = 4 if self.n % 4 == 0 else 1
        self.obs_space = {"maps": Box((1, rows, self.n // rows)), "scalars": Box((2,))}
        self.action_space = MaskedChoice(self.n, self.k)
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.values = self.rng.normal(size=self.n)
        self.view = (self.values + self.noise * self.rng.normal(size=self.n)).astype(np.float32)
        self.taken, self.t, self.score = np.zeros(self.n, bool), 0, 0.0
        # truncation point: after decision `cut` (1-based) if the episode is to be cut short
        self.cut = int(self.rng.integers(1, self.horizon)) if self.horizon > 1 and self.rng.random() < self.trunc_prob else None
        return self._obs()

    def _obs(self) -> dict:
        free = ~self.taken
        mask = free & (self.rng.random(self.n) < self.keep)
        if mask.sum() < self.k:  # top up so at least k options are allowed
            extra = self.rng.permutation(np.flatnonzero(free & ~mask))[: self.k - mask.sum()]
            mask[extra] = True
        shape = self.obs_space["maps"].shape
        return {"maps": self.view.reshape(shape), "scalars": np.array([self.t / self.horizon, 1.0], np.float32), "mask": mask}

    def step(self, action):
        action = np.asarray(action)
        assert len(set(action.tolist())) == self.k and not self.taken[action].any(), "invalid pick"
        self.taken[action] = True
        self.score += float(self.values[action].sum())
        self.t += 1
        terminated = self.t >= self.horizon
        truncated = not terminated and self.cut == self.t
        info = {"episode": {"score": self.score, "decisions": self.t}} if terminated or truncated else {}
        return self._obs(), (self.score if terminated else 0.0), terminated, truncated, info


class TargetBoxEnv:
    """Move a 2-D point to a random target with a velocity action in [-1,1]^2 (scaled by max_step; the env clips).
    Reward per step = -distance after the move; terminates on reaching the target; truncated after `target_steps`."""
    SPEC_DEFAULTS = {"target_steps": 15, "target_max_step": 0.3}

    def __init__(self, spec: dict):
        self.steps, self.max_step = spec["target_steps"], spec["target_max_step"]
        self.obs_space = {"pos": Box((2,)), "target": Box((2,))}
        self.action_space = Box((2,), -1.0, 1.0)
        self.rng = np.random.default_rng(0)

    def reset(self, seed: int | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.pos, self.target = self.rng.uniform(-1, 1, 2), self.rng.uniform(-1, 1, 2)
        self.t, self.ret = 0, 0.0
        return self._obs()

    def _obs(self) -> dict:
        return {"pos": self.pos.astype(np.float32), "target": self.target.astype(np.float32)}

    def step(self, action):
        self.pos = self.pos + self.max_step * np.clip(action, -1, 1)
        self.t += 1
        dist = float(np.linalg.norm(self.pos - self.target))
        reward = -dist
        self.ret += reward
        terminated = dist < 0.05
        truncated = not terminated and self.t >= self.steps
        info = {"episode": {"score": self.ret, "length": self.t}} if terminated or truncated else {}
        return self._obs(), reward, terminated, truncated, info
