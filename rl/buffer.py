"""Rollout stacking and GAE."""
import numpy as np


def compute_gae(rewards, values, terminated, truncated, final_values, last_values, gamma, lam):
    """Generalized advantage estimation over [T, N] arrays.

    values[t] = V(obs_t). The value after step t is values[t+1], except: at the end of the rollout it is
    last_values (V of the carried-over obs, the episode continues next epoch); at a step that ended an episode
    it is final_values[t] = V(final obs) on truncation (bootstrap) and 0 on termination. The GAE trace is cut
    at every episode end. Returns (advantages, returns = advantages + values), float32.
    """
    T, N = rewards.shape
    done = terminated | truncated
    next_val = np.empty((T, N))
    next_val[:-1], next_val[-1] = values[1:], last_values
    next_val = np.where(done, np.where(terminated, 0.0, final_values), next_val)
    adv, last = np.zeros((T, N)), np.zeros(N)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * next_val[t] - values[t]
        last = delta + gamma * lam * (~done[t]) * last
        adv[t] = last
    return adv.astype(np.float32), (adv + values).astype(np.float32)


def stack_rollouts(results: list) -> dict:
    """Per-worker rollout dicts (leading axis T) -> one dict with [T, N, ...] arrays (obs is a dict of them),
    last_values [N], and the concatenated list of episode-stat dicts."""
    out = {"obs": {k: np.stack([r["obs"][k] for r in results], 1) for k in results[0]["obs"]}}
    for key in ("actions", "logp", "values", "rewards", "terminated", "truncated", "final_values"):
        out[key] = np.stack([r[key] for r in results], 1)
    out["last_values"] = np.array([r["last_value"] for r in results])
    out["episodes"] = [e for r in results for e in r["episodes"]]
    return out
