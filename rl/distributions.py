"""Action distributions: masked k-without-replacement choice and a diagonal Gaussian.

Both share one interface: sample(), mode() (deterministic: argmax picks / mean), log_prob(a), entropy(a=None).
"""
import torch
from torch.distributions import Normal

NEG = -1e9  # logit of a disallowed option (finite, so softmax/log_softmax never produce NaN)


class MaskedKChoice:
    """Pick k distinct options of n, sequentially without replacement, from per-option logits.

    Pick j is a categorical over the options that are in `mask` and not already picked, so
    log_prob(a) = sum_j log softmax(logits | remaining allowed)[a_j]  (the Plackett-Luce / sequential model).
    Requires mask.sum(-1) >= k for every row; otherwise ValueError (we raise rather than pad: the env owns
    the mask and a short mask is a task bug that padding would silently hide).
    entropy(a): sum over picks of the entropy of each pick's categorical, evaluated along the sequence `a`
    (given, or the last sampled / log_prob'd one). This is an approximation: the exact entropy of the
    ordered-subset distribution is intractable, but this sum is differentiable, zero iff every pick is
    deterministic, and what we need as an exploration bonus.
    """

    def __init__(self, logits: torch.Tensor, mask: torch.Tensor, k: int):
        if (mask.sum(-1) < k).any():
            raise ValueError(f"MaskedKChoice needs at least k={k} allowed options per row")
        self.logits, self.mask, self.k = logits, mask.bool(), k
        self._last = None  # actions of the most recent sample()/log_prob()/mode()

    def _run(self, actions=None, deterministic=False):
        """Walk the k picks. Returns (picks (B,k), logp (B,), entropy (B,)). With actions=None, picks are drawn."""
        avail = self.mask.clone()
        picks, logp, ent = [], 0.0, 0.0
        for j in range(self.k):
            lp_all = torch.log_softmax(self.logits.masked_fill(~avail, NEG), dim=-1)
            if actions is None:
                with torch.no_grad():
                    pick = lp_all.argmax(-1, keepdim=True) if deterministic else torch.multinomial(lp_all.exp(), 1)
            else:
                pick = actions[:, j:j + 1]
            logp = logp + lp_all.gather(1, pick).squeeze(1)
            ent = ent - (lp_all.exp() * lp_all.masked_fill(~avail, 0.0)).sum(-1)
            picks.append(pick)
            avail = avail.scatter(1, pick, False)
        return torch.cat(picks, dim=1), logp, ent

    def sample(self) -> torch.Tensor:
        self._last = self._run()[0]
        return self._last

    def mode(self) -> torch.Tensor:
        """Greedy: each pick is the argmax of the remaining allowed options."""
        self._last = self._run(deterministic=True)[0]
        return self._last

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        self._last = actions
        return self._run(actions)[1]

    def entropy(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        actions = self._last if actions is None else actions
        if actions is None:
            actions = self.sample()
        return self._run(actions)[2]


class DiagGaussian:
    """Independent Gaussian per action component with a (state-independent) log_std; log_prob/entropy sum over all action dims."""

    def __init__(self, mean: torch.Tensor, log_std: torch.Tensor):
        self.mean, self.normal = mean, Normal(mean, log_std.exp().expand_as(mean))

    def _sum(self, x: torch.Tensor) -> torch.Tensor:
        return x.flatten(1).sum(-1)

    def sample(self) -> torch.Tensor:
        return self.normal.sample()

    def mode(self) -> torch.Tensor:
        return self.mean

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self._sum(self.normal.log_prob(actions))

    def entropy(self, actions: torch.Tensor | None = None) -> torch.Tensor:
        return self._sum(self.normal.entropy())
