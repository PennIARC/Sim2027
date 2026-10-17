"""Run specs: flat JSON dicts. Generic PPO keys live here; each task adds its own via Env.SPEC_DEFAULTS
and each policy network via its builder's SPEC_DEFAULTS. Unknown keys are rejected so typos fail loudly."""
from rl.env import load_object

DEFAULTS = {
    # identity / bookkeeping (not hyperparameters)
    "name": "run", "seed": 0, "blurb": None, "group": None, "tags": [], "parent": None, "kind": "ppo",
    "primary_stat": "score",  # per-episode stat (an epochs.csv column) used for curve summaries and `list`
    "eval": None,             # optional "package.module:function": called as fn(run_dir) -> dict of metrics (see rl/runner.py)
    # what to train: "package.module:Name" paths
    "env": None,     # Env class, built as Env(spec) in every worker
    "policy": None,  # policy builder, called as builder(spec, obs_space, action_space) -> rl.policy.Policy
    # rollout
    "epochs": 300, "num_envs": 8,  # worker processes; 0 = one env in-process (tests/debugging)
    "steps_per_epoch": 512,  # decisions per env per epoch
    "torch_threads": 1, "save_every": 50,  # checkpoint (policy.pt, norm.npz, epochs.csv) every N epochs and at the end
    # PPO
    "gamma": 1.0, "gae_lambda": 0.95, "clip": 0.2, "update_epochs": 4, "num_minibatches": 8,
    "pi_lr": 3e-4, "vf_lr": 1e-3, "lr_schedule": "linear", "lr_decay_end": 0.1,
    "entropy_coef": 0.0, "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": None,
    "init_log_std": -0.5,  # Box actions only
    # normalization
    "obs_norm_keys": [],    # obs dict keys normalized with a running mean/std (e.g. small scalar vectors; not maps)
    "reward_scaling": False,  # divide rewards by the running std of the discounted return
    # warm starts
    "init_from": None, "critic_warmup_epochs": 0, "freeze_obs_norm": False,
}
NON_HYPER_KEYS = ("name", "seed", "blurb", "group", "tags", "parent", "kind", "primary_stat", "eval")


def all_defaults(spec: dict) -> dict:
    """Generic defaults plus those of the task and policy named in the spec."""
    out = dict(DEFAULTS)
    for key in ("env", "policy"):
        path = spec.get(key)
        if path:
            out.update(getattr(load_object(path), "SPEC_DEFAULTS", {}))
    return out


def make_spec(spec: dict) -> dict:
    """Spec with all defaults filled in."""
    defaults = all_defaults(spec)
    unknown = set(spec) - set(defaults)
    if unknown:
        raise ValueError(f"unknown spec keys: {sorted(unknown)}")
    if not spec.get("env") or not spec.get("policy"):
        raise ValueError("spec needs 'env' and 'policy'")
    return {**defaults, **spec}
