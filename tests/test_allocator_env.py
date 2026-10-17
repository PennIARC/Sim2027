import numpy as np
import pytest

from rl.spaces import MaskedChoice
from sim.allocator_env import AllocatorEnv, N_CHANNELS, allocate, obs_from_input
from sim.geometry import N_BLOCKS
from sim.interfaces import Allocation
from sim.mines import load_eval_set
from sim.planner import plan
from sim.scoring import score

SHORT = {"episode_s": 50.0}


def run_random(env, seed, layout=None, rng_seed=7):
    rng = np.random.default_rng(rng_seed)
    obs = env.reset(seed=seed, layout=layout)
    trace = []
    while True:
        a = rng.choice(np.flatnonzero(obs["mask"]), env.k, replace=False)
        obs, r, term, trunc, info = env.step(a)
        trace.append((a.tolist(), r))
        if term or trunc:
            return obs, r, info, trace


@pytest.fixture(scope="module")
def env():
    return AllocatorEnv(SHORT)


def check_obs(env, obs):
    assert set(obs) == {"maps", "scalars", "mask"}
    for k, box in env.obs_space.items():
        assert obs[k].shape == box.shape and obs[k].dtype == box.dtype, k
    assert obs["mask"].dtype == bool and obs["mask"].shape == (N_BLOCKS,)
    assert obs["mask"].sum() >= env.k
    assert np.isfinite(obs["maps"].astype(np.float32)).all() and np.isfinite(obs["scalars"]).all()


def test_spaces_and_first_obs(env):
    assert isinstance(env.action_space, MaskedChoice) and env.action_space.n == N_BLOCKS and env.action_space.k == 4
    assert env.obs_space["maps"].shape == (N_CHANNELS, 300, 80)
    obs = env.reset(seed=0)
    check_obs(env, obs)
    assert obs["mask"].all()                      # nothing allocated yet
    assert obs["maps"][2].max() == 0 and obs["maps"][3].max() == 0
    assert obs["maps"][4].max() > 0.99            # self blob sits on the drone


def test_episode_contract_and_terminal_reward():
    env = AllocatorEnv(SHORT)
    obs = env.reset(seed=3)
    n, others_seen = 0, False
    rng = np.random.default_rng(0)
    while True:
        check_obs(env, obs)
        if n >= 1:
            others_seen |= bool(obs["maps"][2].max() > 0)
            assert obs["mask"].sum() < N_BLOCKS   # blocks held by the other drones are masked
        a = rng.choice(np.flatnonzero(obs["mask"]), 4, replace=False)
        obs, r, term, trunc, info = env.step(a)
        n += 1
        if term or trunc:
            break
        assert r == 0.0 and not term and not trunc
    assert term and not trunc and others_seen
    ep = info["episode"]
    assert r == pytest.approx(ep["score"]) and ep["decisions"] == n
    for k in ("score", "B", "L", "W", "G", "on_path_mines", "expected_score", "decisions", "seen_frac"):
        assert k in ep
    # reward = scorer output for the planner's path on the mapper's belief
    path, _ = plan(env.mapper.p_squares(), env.spec)
    assert score(path, env.mines, 7, 0)["score"] == pytest.approx(r)


def test_first_four_decisions_go_to_drones_in_order(env):
    obs = env.reset(seed=1)
    rng = np.random.default_rng(0)
    for i in range(4):
        assert env.last_input.drone_id == i
        obs, r, term, _, _ = env.step(rng.choice(np.flatnonzero(obs["mask"]), 4, replace=False))
        assert r == 0 and not term


def test_invalid_actions_rejected(env):
    obs = env.reset(seed=1)
    env.step(np.array([0, 1, 2, 3]))
    held = np.flatnonzero(~env.last_input.block_mask)
    assert len(held) == 4
    with pytest.raises(ValueError):
        env.step(np.array([*held[:1], 100, 101, 102]))
    with pytest.raises(ValueError):
        env.step(np.array([100, 100, 101, 102]))


def test_deterministic_given_seed():
    a = run_random(AllocatorEnv(SHORT), 5)
    b = run_random(AllocatorEnv(SHORT), 5)
    strip = lambda ep: {k: v for k, v in ep.items() if k != "planner_s"}  # planner_s is wall time
    assert strip(a[2]["episode"]) == strip(b[2]["episode"]) and a[3] == b[3]
    c = run_random(AllocatorEnv(SHORT), 6)
    assert c[2]["episode"]["n_mines"] != a[2]["episode"]["n_mines"] or c[3] != a[3]


def test_fixed_layout_and_progress():
    layout = load_eval_set()[0][1]
    env = AllocatorEnv(SHORT)
    env.set_progress(1.0)
    env.reset(seed=0, layout=layout)
    assert (env.mines == layout).all()
    assert env.mapper.sigma == pytest.approx(1.5) and env.sensor.sigma == pytest.approx(1.5)
    env.set_progress(0.0)
    assert env.mapper.sigma == pytest.approx(0.5)


def test_obs_from_input_uses_only_allocator_input(env):
    env.reset(seed=2)
    env.step(np.array([0, 1, 2, 3]))
    inp = env.last_input
    obs = obs_from_input(inp, env.spec)
    again = obs_from_input(inp, env.spec)
    for k in obs:
        assert (obs[k] == again[k]).all()
    # perturbing any AllocatorInput field changes the observation; the function needs nothing else (no env, no mapper)
    import copy
    changed = copy.deepcopy(inp)
    changed.p_mine = changed.p_mine * 0.5
    assert (obs_from_input(changed, env.spec)["maps"][0] != obs["maps"][0]).any()
    changed = copy.deepcopy(inp)
    changed.drones[inp.drone_id].pose_xy = (70.0, 250.0)
    assert (obs_from_input(changed, env.spec)["maps"][4] != obs["maps"][4]).any()
    changed = copy.deepcopy(inp)
    changed.t = inp.t + 10
    assert obs_from_input(changed, env.spec)["scalars"][0] < obs["scalars"][0]


def test_allocate_stateless_entry_point(env):
    from sim.allocator_policy import cnn_policy
    from rl.spec import make_spec
    spec = make_spec({"env": "sim.allocator_env:AllocatorEnv", "policy": "sim.allocator_policy:cnn_policy"})
    policy = cnn_policy(spec, env.obs_space, env.action_space).eval()
    env.reset(seed=4)
    inp = env.last_input
    a1, a2 = allocate(policy, inp), allocate(policy, inp)
    assert isinstance(a1, Allocation) and a1.blocks == a2.blocks and len(set(a1.blocks)) == 4
    assert all(inp.block_mask[b] for b in a1.blocks)
    s1, s2 = allocate(policy, inp, deterministic=False, seed=9), allocate(policy, inp, deterministic=False, seed=9)
    assert s1.blocks == s2.blocks
