"""Time one PPO update of the allocator CNN on a synthetic batch, per device / thread count, and check CPU-MPS parity.

    .venv/bin/python -m tools.bench_update [--batch 512] [--devices cpu1,cpu4,mps]
"""
import argparse
import copy
import time

import numpy as np
import torch

from rl.policy import Policy  # noqa: F401  (import order: rl before sim)
from rl.ppo import make_optimizer, ppo_update
from rl.spec import make_spec
from sim.allocator_env import AllocatorEnv
from sim.allocator_policy import cnn_policy


def synthetic_batch(obs_space, action_space, B, rng):
    obs = {k: torch.as_tensor(rng.random((B, *s.shape), dtype=np.float32)).to(torch.float16 if k == "maps" else torch.float32)
           for k, s in obs_space.items()}
    obs["mask"] = torch.ones(B, action_space.n, dtype=torch.bool)
    actions = torch.as_tensor(np.stack([rng.choice(action_space.n, action_space.k, replace=False) for _ in range(B)]))
    return {"obs": obs, "actions": actions, "logp": torch.full((B,), -20.0), "values": torch.zeros(B),
            "adv": torch.as_tensor(rng.standard_normal(B), dtype=torch.float32), "ret": torch.as_tensor(rng.standard_normal(B), dtype=torch.float32)}


def to(batch, device):
    return {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device)) for k, v in batch.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--devices", default="cpu1,cpu4,mps")
    args = ap.parse_args()
    spec = make_spec({"env": "sim.allocator_env:AllocatorEnv", "policy": "sim.allocator_policy:cnn_policy"})
    env = AllocatorEnv(spec)
    torch.manual_seed(0)
    base = cnn_policy(spec, env.obs_space, env.action_space)
    batch = synthetic_batch(env.obs_space, env.action_space, args.batch, np.random.default_rng(0))

    # parity: one forward/backward on identical weights and data
    grads = {}
    for dev in ("cpu", "mps") if torch.backends.mps.is_available() else ("cpu",):
        p = copy.deepcopy(base).to(dev)
        b = to(batch, dev)
        obs = {k: (v[:64].float() if v.is_floating_point() else v[:64]) for k, v in b["obs"].items()}
        dist, value = p.dist_and_value(obs)
        loss = -(dist.log_prob(b["actions"][:64]) * b["adv"][:64]).mean() + ((value - b["ret"][:64]) ** 2).mean()
        loss.backward()
        grads[dev] = (loss.item(), torch.cat([q.grad.flatten().cpu() for q in p.parameters() if q.grad is not None]))
    if "mps" in grads:
        (lc, gc), (lm, gm) = grads["cpu"], grads["mps"]
        print(f"parity: loss cpu {lc:.6f} mps {lm:.6f} | grad max abs diff {(gc - gm).abs().max():.2e} (grad scale {gc.abs().max():.2e})")

    for name in args.devices.split(","):
        dev, threads = ("mps", 4) if name == "mps" else ("cpu", int(name[3:]))
        torch.set_num_threads(threads)
        p = copy.deepcopy(base).to(dev)
        opt, _ = make_optimizer(p, spec)
        b = to(batch, dev)
        ppo_update(p, opt, b, spec)  # warm-up (kernel compilation on mps)
        t = time.time()
        ppo_update(p, opt, b, spec)
        if dev == "mps":
            torch.mps.synchronize()
        print(f"{name:5s} threads={threads}: {time.time() - t:6.2f} s per update (batch {args.batch}, "
              f"{spec['update_epochs']} epochs x {spec['num_minibatches']} minibatches)")


if __name__ == "__main__":
    main()
