import json

import numpy as np
import pytest

from rl.toy_envs import PickBestEnv

from rl.trainer import load_policy, train

PICK = {"env": "rl.toy_envs:PickBestEnv", "policy": "rl.policy:mlp_policy", "seed": 1, "epochs": 15, "steps_per_epoch": 600,
        "num_envs": 0, "hidden": [32, 32], "update_epochs": 10, "obs_norm_keys": ["scalars"], "pi_lr": 3e-3, "vf_lr": 3e-3, "lr_decay_end": 0.3,
        "gae_lambda": 0.95, "num_minibatches": 6, "entropy_coef": 0.0, "save_every": 5}


def run(tmp_path, spec):
    out = train(spec, tmp_path)
    for f in ("spec.json", "policy.pt", "norm.npz", "epochs.csv", "done.json"):
        assert (tmp_path / f).exists(), f
    rows = (tmp_path / "epochs.csv").read_text().splitlines()
    assert len(rows) == spec["epochs"] + 1
    return out, rows


def test_pick_best_inline(tmp_path):
    out, rows = run(tmp_path, PICK)
    header = rows[0].split(",")
    scores = [float(r.split(",")[header.index("score")]) for r in rows[1:]]
    print("PickBest inline: first", np.mean(scores[:2]), "last", np.mean(scores[-3:]))
    assert np.mean(scores[:2]) < 1.0               # random is ~0
    assert np.mean(scores[-3:]) > 2.0              # clearly better than random
    assert json.loads((tmp_path / "done.json").read_text())["final"]["score"] > 1.5
    policy, norm = load_policy(tmp_path)           # evaluation path, greedy
    from rl.toy_envs import PickBestEnv
    from rl.spec import make_spec
    env = PickBestEnv(make_spec(PICK))
    tot = []
    for ep in range(100):
        obs = env.reset(seed=ep)
        done = False
        while not done:
            a, _, _ = policy.act(norm.apply(obs), deterministic=True)
            obs, _, term, trunc, info = env.step(a)
            done = term or trunc
        tot.append(info["episode"]["score"])
    assert np.mean(tot) > 2.5  # greedy; oracle on the noisy view is ~6.4


def test_pick_best_workers(tmp_path):
    spec = {**PICK, "num_envs": 2, "steps_per_epoch": 300}
    out, rows = run(tmp_path, spec)
    header = rows[0].split(",")
    scores = [float(r.split(",")[header.index("score")]) for r in rows[1:]]
    print("PickBest 2 workers: first", np.mean(scores[:2]), "last", np.mean(scores[-3:]))
    assert np.mean(scores[-3:]) > 1.8 and np.mean(scores[-3:]) > np.mean(scores[:2]) + 1.2


def test_resume_init_from(tmp_path):
    train({**PICK, "epochs": 2}, tmp_path / "a")
    out, _ = run(tmp_path / "b", {**PICK, "epochs": 2, "init_from": str(tmp_path / "a"), "freeze_obs_norm": True, "critic_warmup_epochs": 1})
    assert out["epochs"] == 2


@pytest.mark.parametrize("workers", [0, 2])
def test_target_box(tmp_path, workers):
    spec = {"env": "rl.toy_envs:TargetBoxEnv", "policy": "rl.policy:mlp_policy", "seed": 0, "epochs": 15, "steps_per_epoch": 400 // max(workers, 1),
            "num_envs": workers, "pi_lr": 2e-3, "vf_lr": 3e-3, "num_minibatches": 4, "obs_norm_keys": ["pos", "target"], "reward_scaling": True, "init_log_std": -0.5}
    out, rows = run(tmp_path, spec)
    header = rows[0].split(",")
    scores = [float(r.split(",")[header.index("score")]) for r in rows[1:]]
    print(f"TargetBox workers={workers}: first {np.mean(scores[:2]):.2f} last {np.mean(scores[-3:]):.2f}")
    assert np.mean(scores[-3:]) > np.mean(scores[:2]) + 2.0


class BoomEnv(PickBestEnv):
    """Fails on its first step, to check that worker errors surface with the traceback."""
    def step(self, action):
        raise ZeroDivisionError("boom")


def test_worker_error_surfaces(tmp_path):
    with pytest.raises(RuntimeError, match="boom"):
        train({**PICK, "env": "tests.test_train_smoke:BoomEnv", "num_envs": 1, "epochs": 1}, tmp_path)
