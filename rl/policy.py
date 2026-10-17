"""Policy base class (actor + critic over dict observations) and reusable builders / network blocks.

A builder is any callable `builder(spec, obs_space, action_space) -> Policy` with a `SPEC_DEFAULTS` dict
attribute holding its own spec keys; rl.spec.make_spec merges it via getattr(load_object(path), "SPEC_DEFAULTS").
"""
import numpy as np
import torch
import torch.nn as nn

from rl.distributions import DiagGaussian, MaskedKChoice
from rl.spaces import Box, MaskedChoice


def action_dim(action_space) -> int:
    """Size of the actor head: n logits for MaskedChoice, prod(shape) means for Box."""
    return action_space.n if isinstance(action_space, MaskedChoice) else int(np.prod(action_space.shape))


class Policy(nn.Module):
    """Subclasses implement forward(obs) -> (dist_params, value). obs: dict of batched tensors; value: (B,).

    dist_params: (B, n) logits for MaskedChoice (the mask comes from obs["mask"]), (B, *shape) means for Box.
    Parameter groups for PPO: names starting with "critic"/"value" form the critic group (vf_lr), everything
    else (incl. a trunk shared with the actor) the actor group (pi_lr). Override the two methods to change that.
    """

    def __init__(self, obs_space: dict, action_space, init_log_std=-0.5):
        super().__init__()
        self.obs_space, self.action_space = obs_space, action_space
        if isinstance(action_space, Box):  # state-independent log std, one value or one per component
            init = torch.as_tensor(init_log_std, dtype=torch.float32).expand(action_space.shape).clone()
            self.log_std = nn.Parameter(init)

    def forward(self, obs: dict):
        raise NotImplementedError

    def dist_from_params(self, params: torch.Tensor, obs: dict):
        if isinstance(self.action_space, MaskedChoice):
            return MaskedKChoice(params, obs["mask"], self.action_space.k)
        return DiagGaussian(params, self.log_std)

    def dist_and_value(self, obs: dict):
        params, value = self(obs)
        return self.dist_from_params(params, obs), value

    def dist(self, obs: dict):
        return self.dist_and_value(obs)[0]

    def critic_parameters(self) -> list:
        return [p for n, p in self.named_parameters() if n.startswith(("critic", "value"))]

    def actor_parameters(self) -> list:
        ids = {id(p) for p in self.critic_parameters()}
        return [p for p in self.parameters() if id(p) not in ids]

    @staticmethod
    def _to_batch(obs: dict) -> dict:
        out = {}
        for k, v in obs.items():
            t = torch.as_tensor(np.asarray(v)).unsqueeze(0)
            out[k] = t.float() if t.is_floating_point() else t
        return out

    @torch.no_grad()
    def act(self, obs: dict, deterministic: bool = False):
        """Single numpy observation dict -> (action ndarray, logp float, value float). No grad."""
        batch = self._to_batch(obs)
        d, value = self.dist_and_value(batch)
        a = d.mode() if deterministic else d.sample()
        return a[0].numpy(), float(d.log_prob(a)[0]), float(value[0])

    @torch.no_grad()
    def value(self, obs: dict) -> float:
        return float(self(self._to_batch(obs))[1][0])


def mlp(in_dim: int, out_dim: int, hidden, out_gain: float = 1.0) -> nn.Sequential:
    """Tanh MLP with orthogonal init; out_gain scales the last layer (0.01 for policy heads)."""
    sizes, layers = [in_dim, *hidden], []
    for n_in, n_out in zip(sizes, sizes[1:]):
        layers += [nn.Linear(n_in, n_out), nn.Tanh()]
    layers.append(nn.Linear(sizes[-1], out_dim))
    for m in layers:
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, 2 ** 0.5 if m is not layers[-1] else out_gain)
            nn.init.zeros_(m.bias)
    return nn.Sequential(*layers)


def conv_trunk(in_channels: int, channels=(16, 32, 64), kernel: int = 3, stride: int = 2) -> nn.Sequential:
    """Stack of Conv2d+ReLU blocks (padding keeps size / stride). For task-specific CNN policies to build on."""
    layers, c_in = [], in_channels
    for c_out in channels:
        layers += [nn.Conv2d(c_in, c_out, kernel, stride, padding=kernel // 2), nn.ReLU()]
        c_in = c_out
    return nn.Sequential(*layers)


def conv_out_dim(trunk: nn.Module, in_shape: tuple) -> int:
    """Flattened feature size of `trunk` for an input of shape in_shape (C, H, W)."""
    with torch.no_grad():
        return int(trunk(torch.zeros(1, *in_shape)).numel())


class MLPPolicy(Policy):
    """Separate actor / critic tanh MLPs on all obs keys (except "mask") flattened and concatenated."""

    def __init__(self, obs_space, action_space, hidden, init_log_std):
        super().__init__(obs_space, action_space, init_log_std)
        self.keys = [k for k in obs_space if k != "mask"]
        in_dim = sum(int(np.prod(obs_space[k].shape)) for k in self.keys)
        self.actor = mlp(in_dim, action_dim(action_space), hidden, out_gain=0.01)
        self.critic = mlp(in_dim, 1, hidden)

    def forward(self, obs):
        x = torch.cat([obs[k].flatten(1) for k in self.keys], dim=-1)
        params = self.actor(x)
        if isinstance(self.action_space, Box):
            params = params.view(-1, *self.action_space.shape)
        return params, self.critic(x).squeeze(-1)


def mlp_policy(spec: dict, obs_space: dict, action_space) -> Policy:
    return MLPPolicy(obs_space, action_space, spec["hidden"], spec["init_log_std"])


mlp_policy.SPEC_DEFAULTS = {"hidden": [64, 64]}
