import itertools
import math

import numpy as np
import pytest
import torch

from rl.buffer import compute_gae
from rl.distributions import MaskedKChoice
from rl.norm import Normalizer, RunningNorm
from rl.spec import make_spec

PICK = "rl.toy_envs:PickBestEnv"
MLP = "rl.policy:mlp_policy"


def test_masked_k_choice_sample():
    torch.manual_seed(0)
    B, n, k = 200, 8, 3
    logits = torch.randn(B, n)
    mask = torch.rand(B, n) < 0.7
    mask[:, :k] = True
    d = MaskedKChoice(logits, mask, k)
    a = d.sample()
    assert a.shape == (B, k) and a.dtype == torch.int64
    assert all(len(set(r)) == k for r in a.tolist())
    assert mask.gather(1, a).all()
    assert torch.equal(d.mode(), MaskedKChoice(logits, mask, k).mode())


def test_masked_k_choice_logprob_bruteforce():
    n, k = 5, 3
    logits = torch.randn(2, n)
    mask = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 0, 1, 1]], dtype=torch.bool)
    d = MaskedKChoice(logits, mask, k)
    total = torch.zeros(2)
    for perm in itertools.permutations(range(n), k):
        a = torch.tensor([perm, perm])
        lp = d.log_prob(a)
        for b in range(2):  # brute force sequential softmax
            allowed, expect = [i for i in range(n) if mask[b, i]], 0.0
            if any(p not in allowed for p in perm):
                continue
            for p in perm:
                expect += math.log(math.exp(logits[b, p]) / sum(math.exp(logits[b, i]) for i in allowed))
                allowed.remove(p)
            assert abs(lp[b].item() - expect) < 1e-5
            total[b] += math.exp(expect)
    assert torch.allclose(total, torch.ones(2), atol=1e-5)  # a proper distribution over ordered picks


def test_masked_k_choice_deterministic_entropy_errors():
    logits = torch.tensor([[0.1, 5.0, 3.0, 4.0]])
    mask = torch.tensor([[True, True, False, True]])
    d = MaskedKChoice(logits, mask, 2)
    assert d.mode().tolist() == [[1, 3]]  # masked option 2 never chosen despite being third best
    assert d.entropy(d.mode()).item() > 0
    with pytest.raises(ValueError):
        MaskedKChoice(logits, mask, 4)
    logits.requires_grad_()
    MaskedKChoice(logits, mask, 2).log_prob(torch.tensor([[1, 3]])).sum().backward()
    assert torch.isfinite(logits.grad).all()


def gae(r, v, term, trunc, fv, lv, gamma=1.0, lam=1.0):
    col = lambda x, dt=float: np.array(x, dtype=dt).reshape(-1, 1)
    adv, ret = compute_gae(col(r), col(v), col(term, bool), col(trunc, bool), col(fv), np.array([lv]), gamma, lam)
    return adv[:, 0], ret[:, 0]


def test_gae_carry_over_termination_truncation():
    # carry-over: no episode end, bootstrap from last value 5
    adv, _ = gae([0, 0, 1], [0, 0, 0], [0] * 3, [0] * 3, [0] * 3, 5.0)
    assert np.allclose(adv, [6, 6, 6])
    # termination at t=1: nothing after it counts, even a huge value from the next episode
    adv, ret = gae([0, 1, 0], [0.5, 0.5, 0.0], [0, 1, 0], [0] * 3, [0] * 3, 9.0)
    assert np.allclose(adv[:2], [0.5, 0.5]) and np.allclose(ret[:2], [1.0, 1.0])
    assert np.isclose(adv[2], 9.0)
    # truncation at t=1: bootstrap from final value 4 (not zero), trace still cut
    adv, _ = gae([0, 1, 0], [0.5, 0.5, 0.0], [0] * 3, [0, 1, 0], [0, 4.0, 0], 0.0)
    assert np.allclose(adv[:2], [5.0 - 0.5, 5.0 - 0.5])  # t=1: 1+4-0.5=4.5, t=0: 0+0.5-0.5+4.5 (lam=1)
    # gamma<1, lam<1 hand check: t=0: delta=0+0.5*v1-v0
    adv, _ = gae([0, 2], [1.0, 1.0], [0, 1], [0, 0], [0, 0], 0.0, gamma=0.5, lam=0.5)
    d1, d0 = 2 - 1.0, 0.5 * 1.0 - 1.0
    assert np.allclose(adv, [d0 + 0.25 * d1, d1])


def test_running_norm():
    rng = np.random.default_rng(0)
    x = rng.normal(3, 2, size=(1000, 4))
    rn = RunningNorm((4,))
    for chunk in np.array_split(x, 7):
        rn.update(chunk)
    assert np.allclose(rn.mean, x.mean(0), atol=1e-3) and np.allclose(rn.var, x.var(0), rtol=1e-3)
    assert np.abs(rn.normalize(np.full(4, 1e6))).max() == 10.0


def test_normalizer_keys_and_reward_scaling():
    from rl.spaces import Box
    spec = {"obs_norm_keys": ["s"], "reward_scaling": True, "gamma": 1.0, "freeze_obs_norm": False}
    norm = Normalizer(spec, {"s": Box((2,)), "maps": Box((1, 2, 2))})
    norm.update_obs({"s": np.random.randn(100, 2) * 5 + 1})
    out = norm.apply({"s": np.ones(2, np.float32), "maps": np.ones((1, 2, 2), np.float32), "mask": np.ones(3, bool)})
    assert out["maps"].max() == 1 and out["mask"].all() and out["s"].dtype == np.float32
    r = norm.scale_rewards(np.random.randn(20, 3) * 10, np.zeros((20, 3), bool))
    assert r.std() < 5
    norm2 = Normalizer(spec, {"s": Box((2,)), "maps": Box((1, 2, 2))})
    norm2.load_state(norm.state())
    assert np.allclose(norm2.obs["s"].mean, norm.obs["s"].mean)
    with pytest.raises(ValueError):
        Normalizer({**spec, "obs_norm_keys": ["nope"]}, {"s": Box((2,))})


def test_spec_merging():
    spec = make_spec({"env": PICK, "policy": MLP})
    assert spec["hidden"] == [64, 64] and spec["pick_n"] == 20 and spec["save_every"] > 0
    with pytest.raises(ValueError):
        make_spec({"env": PICK, "policy": MLP, "hiden": [8]})
