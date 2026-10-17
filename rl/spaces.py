"""Observation and action space descriptions. Plain dataclasses, no gymnasium."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Box:
    """Continuous array space (observations, or continuous actions sampled from a diagonal Gaussian)."""
    shape: tuple
    low: float = -np.inf
    high: float = np.inf
    dtype: type = np.float32


@dataclass(frozen=True)
class MaskedChoice:
    """Discrete action: pick k distinct options out of n, without replacement.

    The observation dict must carry a bool array obs["mask"] of shape (n,), True = option allowed.
    The action is an int64 array of shape (k,) holding the chosen option indices in pick order.
    log_prob(action) = sum over picks of log softmax(logits masked by the original mask minus the earlier picks).
    k=1 is an ordinary masked categorical.
    """
    n: int
    k: int = 1
