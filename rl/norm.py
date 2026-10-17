"""Running observation / reward normalization."""
import numpy as np

NORM_CLIP = 10.0


class RunningNorm:
    """Running mean and variance over everything seen (parallel/batched update), per feature."""

    def __init__(self, shape=()):
        self.mean, self.var, self.count = np.zeros(shape), np.ones(shape), 1e-4

    def update(self, x) -> None:  # x: [batch, *shape]
        x = np.asarray(x, dtype=np.float64)
        n = len(x)
        delta = x.mean(0) - self.mean
        total = self.count + n
        self.var = (self.var * self.count + x.var(0) * n + delta ** 2 * self.count * n / total) / total
        self.mean = self.mean + delta * n / total
        self.count = total

    def normalize(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -NORM_CLIP, NORM_CLIP)


class Normalizer:
    """Per-key RunningNorms for the obs keys in spec["obs_norm_keys"] (other keys, e.g. maps and mask, pass through),
    plus optional reward scaling by the running std of the discounted return (spec["reward_scaling"])."""

    def __init__(self, spec: dict, obs_space: dict):
        self.keys = list(spec["obs_norm_keys"])
        missing = [k for k in self.keys if k not in obs_space]
        if missing:
            raise ValueError(f"obs_norm_keys not in obs_space: {missing}")
        self.obs = {k: RunningNorm(obs_space[k].shape) for k in self.keys}
        self.reward_scaling, self.gamma = bool(spec["reward_scaling"]), spec["gamma"]
        self.freeze_obs = bool(spec["freeze_obs_norm"])
        self.ret = RunningNorm(())
        self.disc_return = None  # [N] running discounted return of each env

    def apply(self, obs: dict) -> dict:
        """Normalized copy of an obs dict (single or batched; float32)."""
        out = dict(obs)
        for k in self.keys:
            out[k] = self.obs[k].normalize(obs[k]).astype(np.float32)
        return out

    def update_obs(self, obs: dict) -> None:
        """obs: dict of arrays [batch, *shape] of raw observations."""
        if self.freeze_obs:
            return
        for k in self.keys:
            self.obs[k].update(np.asarray(obs[k]).reshape(-1, *self.obs[k].mean.shape))

    def scale_rewards(self, rewards: np.ndarray, done: np.ndarray) -> np.ndarray:
        """rewards, done: [T, N]. Returns rewards / running std of the discounted return (clipped), updating the stats."""
        if not self.reward_scaling:
            return rewards.astype(np.float32)
        T, N = rewards.shape
        if self.disc_return is None:
            self.disc_return = np.zeros(N)
        returns = np.zeros((T, N))
        for t in range(T):
            self.disc_return = self.disc_return * self.gamma + rewards[t]
            returns[t] = self.disc_return
            self.disc_return[done[t]] = 0.0
        self.ret.update(returns.reshape(-1))
        return np.clip(rewards / np.sqrt(self.ret.var + 1e-8), -NORM_CLIP, NORM_CLIP).astype(np.float32)

    def state(self) -> dict:
        s = {}
        for name, rn in [*((f"obs__{k}", self.obs[k]) for k in self.keys), ("ret", self.ret)]:
            s[f"{name}__mean"], s[f"{name}__var"], s[f"{name}__count"] = rn.mean, rn.var, np.float64(rn.count)
        return s

    def load_state(self, s) -> None:
        for name, rn in [*((f"obs__{k}", self.obs[k]) for k in self.keys), ("ret", self.ret)]:
            if f"{name}__mean" in s:
                rn.mean, rn.var, rn.count = np.array(s[f"{name}__mean"]), np.array(s[f"{name}__var"]), float(s[f"{name}__count"])

    def save(self, path) -> None:
        np.savez(path, **self.state())

    def load(self, path) -> None:
        with np.load(path) as z:
            self.load_state(dict(z))
