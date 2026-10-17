"""CNN policy for the cell allocator (builder `cnn_policy`, see rl/policy.py for the convention).

Input maps (C, 300, 80) on the 1 ft belief grid, plus two fixed coordinate channels (y/300, x/80) so the net knows where on the
field it is. Actor: 3x3 stride-2 conv (-> 150x40), 5x5 stride-5 conv (-> 30x8, exactly the 10 ft block grid; total
stride 10), then 3x3 convs on the block grid with the scalars injected FiLM-style (per-channel scale/shift), and a 1x1 conv to
one logit per block (row-major, = block id). Critic: its own lighter trunk of the same shape, global avg+max pool, concat the
scalars, MLP -> value. Actor params are everything not under `critic` (rl/policy.py grouping convention).
"""
import numpy as np
import torch
import torch.nn as nn

from rl.policy import Policy
from sim.geometry import BLK_COLS, BLK_ROWS


def _conv(c_in, c_out, k, stride=1, gain=2 ** 0.5):
    m = nn.Conv2d(c_in, c_out, k, stride, padding=k // 2)
    nn.init.orthogonal_(m.weight, gain)
    nn.init.zeros_(m.bias)
    return m


def _to_block_grid(c_in, stem, width):
    """(c_in, 300, 80) -> (width, 30, 8): stride 2 then stride 5."""
    return nn.Sequential(_conv(c_in, stem, 3, 2), nn.ReLU(), _conv(stem, width, 5, 5), nn.ReLU())


class CNNPolicy(Policy):
    def __init__(self, obs_space, action_space, spec):
        super().__init__(obs_space, action_space, spec.get("init_log_std", -0.5))
        c, h, w = obs_space["maps"].shape
        n_sc = obs_space["scalars"].shape[0]
        assert (h // 10, w // 10) == (BLK_ROWS, BLK_COLS) and action_space.n == BLK_ROWS * BLK_COLS
        stem, wa, wc, nb = spec["cnn_stem"], spec["cnn_width"], spec["cnn_critic_width"], spec["cnn_blocks"]
        yy, xx = torch.meshgrid(torch.linspace(0, 1, h), torch.linspace(0, 1, w), indexing="ij")
        self.register_buffer("coords", torch.stack([yy, xx])[None], persistent=False)
        # actor
        self.actor_stem = _to_block_grid(c + 2, stem, wa)
        self.actor_blocks = nn.ModuleList([_conv(wa, wa, 3) for _ in range(nb)])
        self.actor_film = nn.ModuleList([nn.Linear(n_sc, 2 * wa) for _ in range(nb + 1)])
        for f in self.actor_film:
            nn.init.zeros_(f.weight), nn.init.zeros_(f.bias)
        self.actor_head = _conv(wa, 1, 1, gain=0.01)
        # critic
        cs = max(stem // 2, 8)
        self.critic_stem = _to_block_grid(c + 2, cs, wc)
        self.critic_block = nn.Sequential(_conv(wc, wc, 3), nn.ReLU())
        self.critic_mlp = nn.Sequential(nn.Linear(2 * wc + n_sc, 64), nn.Tanh(), nn.Linear(64, 1))
        nn.init.orthogonal_(self.critic_mlp[0].weight, 2 ** 0.5), nn.init.zeros_(self.critic_mlp[0].bias)
        nn.init.orthogonal_(self.critic_mlp[2].weight, 1.0), nn.init.zeros_(self.critic_mlp[2].bias)

    def _film(self, layer, x, sc):
        g, b = layer(sc).chunk(2, dim=-1)
        return x * (1 + g[:, :, None, None]) + b[:, :, None, None]

    def forward(self, obs):
        maps, sc = obs["maps"].float(), obs["scalars"].float()
        x = torch.cat([maps, self.coords.expand(len(maps), -1, -1, -1)], 1)
        a = torch.relu(self._film(self.actor_film[0], self.actor_stem(x), sc))
        for conv, film in zip(self.actor_blocks, self.actor_film[1:]):
            a = a + torch.relu(self._film(film, conv(a), sc))  # residual
        logits = self.actor_head(a).flatten(1)  # (B, 240), row-major = block id
        z = self.critic_block(self.critic_stem(x))
        z = torch.cat([z.mean((2, 3)), z.amax((2, 3)), sc], 1)
        return logits, self.critic_mlp(z).squeeze(-1)


def cnn_policy(spec: dict, obs_space: dict, action_space) -> Policy:
    policy = CNNPolicy(obs_space, action_space, spec)
    policy.alloc_spec = {k: v for k, v in spec.items() if k.startswith("alloc_")}  # lets allocate() rebuild the observation
    return policy


cnn_policy.SPEC_DEFAULTS = {"cnn_stem": 16, "cnn_width": 48, "cnn_critic_width": 32, "cnn_blocks": 2}
