import numpy as np
import pytest
import torch

from rl.spec import make_spec
from sim.allocator_env import AllocatorEnv
from sim.allocator_policy import cnn_policy
from sim.baselines import BASELINES, greedy_alloc, lawnmower_alloc, make_baseline, random_alloc, run_episode
from sim.geometry import N_BLOCKS

SHORT = {"episode_s": 50.0}


@pytest.fixture(scope="module")
def spec():
    return make_spec({"env": "sim.allocator_env:AllocatorEnv", "policy": "sim.allocator_policy:cnn_policy"})


def test_policy_shapes_params_and_groups(spec):
    env = AllocatorEnv(SHORT)
    pol = cnn_policy(spec, env.obs_space, env.action_space)
    n = sum(p.numel() for p in pol.parameters())
    assert 50_000 <= n <= 150_000
    actor, critic = pol.actor_parameters(), pol.critic_parameters()
    assert len(actor) + len(critic) == len(list(pol.parameters())) and critic and actor
    B = 3
    obs = {"maps": torch.rand(B, 5, 300, 80).half(), "scalars": torch.rand(B, 3), "mask": torch.ones(B, N_BLOCKS, dtype=torch.bool)}
    logits, value = pol(obs)
    assert logits.shape == (B, N_BLOCKS) and value.shape == (B,) and logits.dtype == torch.float32
    (logits.sum() + value.sum()).backward()
    obs1 = {k: v[:1] for k, v in obs.items()}
    a, lp, v = pol.act({k: x.numpy()[0] for k, x in obs1.items()}, deterministic=True)
    assert a.shape == (4,) and len(set(a.tolist())) == 4


def test_actor_isolated_from_critic(spec):
    env = AllocatorEnv(SHORT)
    pol = cnn_policy(spec, env.obs_space, env.action_space)
    obs = {"maps": torch.rand(2, 5, 300, 80), "scalars": torch.rand(2, 3), "mask": torch.ones(2, N_BLOCKS, dtype=torch.bool)}
    _, value = pol(obs)
    value.sum().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in pol.actor_parameters())


@pytest.mark.parametrize("name", ["random", "lawnmower", "greedy"])
def test_baseline_full_episode(name):
    ep = run_episode(make_baseline(name), 11, SHORT)
    assert ep["score"] >= 0 and ep["decisions"] > 4 and 0 <= ep["seen_frac"] <= 1


def test_baselines_valid_and_deterministic():
    env = AllocatorEnv(SHORT)
    env.reset(seed=0)
    env.step(np.array([0, 1, 2, 3]))
    inp = env.last_input
    for fn in (random_alloc, lawnmower_alloc, greedy_alloc):
        a, b = fn(inp), fn(inp)
        assert a.blocks == b.blocks and a.drone_id == inp.drone_id
        assert len(set(a.blocks)) == 4 and inp.block_mask[a.blocks].all()
    # lawnmower: drone 1 owns block columns 2, 3 and starts at the front
    env.step(np.array(lawnmower_alloc(inp).blocks))
    al = lawnmower_alloc(env.last_input)
    assert al.drone_id == 2 and all(b % 8 in (4, 5) for b in al.blocks) and sorted(b // 8 for b in al.blocks) == [0, 0, 1, 1]


def test_smoke_train_two_workers(tmp_path):
    from rl.trainer import load_policy, train
    spec = {"env": "sim.allocator_env:AllocatorEnv", "policy": "sim.allocator_policy:cnn_policy", "seed": 0, "episode_s": 40.0,
            "epochs": 2, "num_envs": 2, "steps_per_epoch": 8, "num_minibatches": 2, "update_epochs": 2, "reward_scaling": True}
    out = train(spec, tmp_path)
    assert out["epochs"] == 2 and (tmp_path / "policy.pt").exists()
    policy, norm = load_policy(tmp_path)
    assert next(iter(policy.parameters())).dtype == torch.float32
    from sim.evaluate import evaluate
    res = evaluate(tmp_path, procs=2, n=3, with_baselines=False)  # 3 layouts, deterministic policy, spawn pool
    assert res["n"] == 3 and len(res["per_layout"]) == 3 and res["mean_score"] >= 0 and "onpath_death_rate" in res
