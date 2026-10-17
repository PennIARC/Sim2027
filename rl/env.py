"""The contract between a task (environment) and the generic PPO stack. Torch-free.

A task is a class satisfying `Env`. The trainer builds it inside every worker process from the
spec key "env" ("package.module:ClassName") as `EnvClass(spec)`, so it must be importable and
picklable-free (construct everything inside __init__).

One env = one agent making a sequence of decisions. A "step" is one decision, however much
simulated time passes between decisions (event-driven tasks advance their sim inside step()).
"""
import importlib
from typing import Any, Protocol

import numpy as np

from rl.spaces import Box, MaskedChoice


class Env(Protocol):
    # Task-specific spec keys and their defaults; merged into the spec by rl.spec.make_spec.
    SPEC_DEFAULTS: dict

    # obs_space: name -> Box. Every observation is a dict with exactly these keys (float32 arrays of
    # those shapes), plus "mask" (bool, shape (n,)) when action_space is MaskedChoice.
    obs_space: dict
    action_space: Box | MaskedChoice

    def __init__(self, spec: dict): ...

    def reset(self, seed: int | None = None) -> dict:
        """Start a new episode and return the first observation."""

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        """Apply one decision. Returns (obs, reward, terminated, truncated, info).

        terminated: the episode really ended (no bootstrap). truncated: cut off by a limit (bootstrap
        from the critic's value of the returned obs). When either is True, the returned obs is the
        final observation; the caller then calls reset().
        info: free-form. On the last step of an episode, info["episode"] may hold a flat dict of
        numeric per-episode stats (e.g. {"score": 41.2, "mines_missed": 1}); the trainer averages
        each key per epoch and logs it to epochs.csv.
        """

    def set_progress(self, progress: float) -> None:
        """Optional curriculum hook: training progress in [0, 1], called before every rollout."""


def load_object(path: str) -> Any:
    """'package.module:Name' -> the object."""
    module, _, name = path.partition(":")
    return getattr(importlib.import_module(module), name)


def make_env(spec: dict):
    return load_object(spec["env"])(spec)
