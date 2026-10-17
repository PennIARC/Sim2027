"""train(spec, run_dir): the whole PPO loop. See the module docstrings of workers/buffer/ppo for the pieces.

run_dir layout: spec.json (full spec), policy.pt (state_dict), norm.npz, epochs.csv (rewritten at each checkpoint),
done.json (written last). policy.pt/norm.npz/epochs.csv are refreshed every spec["save_every"] epochs and at the end.
"""
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from rl.buffer import compute_gae, stack_rollouts
from rl.env import load_object, make_env
from rl.norm import Normalizer
from rl.ppo import make_optimizer, ppo_update, set_lrs
from rl.spec import make_spec
from rl.workers import make_workers

RESERVED = ("epoch", "time", "steps", "episodes", "pi_loss", "v_loss", "entropy", "approx_kl", "clip_frac", "explained_var", "pi_lr", "vf_lr")


def _build(spec: dict):
    """Probe env (in-process) -> (policy, normalizer). The probe is discarded."""
    env = make_env(spec)
    obs_space, action_space = env.obs_space, env.action_space
    if hasattr(env, "close"):
        env.close()
    return load_object(spec["policy"])(spec, obs_space, action_space), Normalizer(spec, obs_space)


def _weights(policy) -> dict:
    return {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}


def resolve_device(name: str) -> torch.device:
    """spec["device"] -> torch.device; "auto" picks cuda, then mps, then cpu."""
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("device 'mps' requested but MPS is not available")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device 'cuda' requested but CUDA is not available")
    return torch.device(name)


def _write_csv(path: Path, rows: list) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="nan")
        w.writeheader()
        w.writerows(rows)


def _save(run_dir: Path, policy, norm, rows: list, optimizer=None) -> None:
    torch.save({k: v.detach().cpu() for k, v in policy.state_dict().items()}, run_dir / "policy.pt")
    norm.save(run_dir / "norm.npz")
    _write_csv(run_dir / "epochs.csv", rows)
    if optimizer is not None:  # lets an interrupted run resume with its Adam moments
        torch.save(optimizer.state_dict(), run_dir / "optimizer.pt")


def _read_rows(path: Path) -> list:
    """epochs.csv back into row dicts (numbers as int/float)."""
    def num(v):
        f = float(v)
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    with open(path, newline="") as f:
        return [{k: num(v) for k, v in r.items() if v != ""} for r in csv.DictReader(f)]


def _episode_means(episodes: list) -> dict:
    keys = list(dict.fromkeys(k for e in episodes for k in e))
    return {(f"ep_{k}" if k in RESERVED else k): float(np.mean([e[k] for e in episodes if k in e])) for k in keys}


def train(spec: dict, run_dir, resume: bool = False) -> dict:
    """Train one run and write its files into run_dir. Returns {"epochs", "steps", "wall_seconds", "final", "best"}:
    final = per-episode-stat mean over the last 10 epochs, best = per-stat max of the epoch means.

    resume=True continues an interrupted run in the same run_dir from its last checkpoint (policy.pt, norm.npz,
    epochs.csv, and optimizer.pt when present): epoch numbering, LR schedule and curriculum progress carry on
    where the checkpoint left off. Epochs after the last checkpoint are redone."""
    spec = make_spec(spec)
    run_dir = Path(run_dir)
    if resume and not ((run_dir / "policy.pt").exists() and (run_dir / "epochs.csv").exists()):
        raise FileNotFoundError(f"nothing to resume in {run_dir} (needs policy.pt and epochs.csv)")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "spec.json").write_text(json.dumps(spec, indent=1))
    random.seed(spec["seed"]), np.random.seed(spec["seed"]), torch.manual_seed(spec["seed"])
    torch.set_num_threads(int(spec["trainer_threads"]))
    device = resolve_device(spec["device"])

    policy, norm = _build(spec)
    if spec["init_from"]:  # warm start: policy (actor and critic) and observation statistics
        src = Path(spec["init_from"])
        policy.load_state_dict(torch.load(src / "policy.pt", map_location="cpu", weights_only=True))
        if (src / "norm.npz").exists():
            norm.load(src / "norm.npz")
    rows, steps, first_epoch, t_offset = [], 0, 1, 0.0
    if resume:
        policy.load_state_dict(torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=True))
        norm.load(run_dir / "norm.npz")
        rows = _read_rows(run_dir / "epochs.csv")
        if rows:
            steps, first_epoch, t_offset = int(rows[-1]["steps"]), int(rows[-1]["epoch"]) + 1, float(rows[-1]["time"])
        print(f"resuming at epoch {first_epoch}/{spec['epochs']} from {run_dir}", flush=True)
    policy.to(device)  # before the optimizer is built so its state lives on the same device
    optimizer, base_lrs = make_optimizer(policy, spec)
    if resume and (run_dir / "optimizer.pt").exists():
        optimizer.load_state_dict(torch.load(run_dir / "optimizer.pt", map_location=device, weights_only=True))
    T, epochs, start = spec["steps_per_epoch"], spec["epochs"], time.time() - t_offset
    workers = make_workers(spec)
    try:
        N = workers.n
        if norm.keys and not spec["init_from"] and not resume:  # a short throwaway rollout to get first observation statistics
            for r in workers.collect(_weights(policy), norm.state(), min(128, T), 0.0):
                norm.update_obs(r["obs"])
        for epoch in range(first_epoch, epochs + 1):
            progress = (epoch - 1) / max(epochs - 1, 1)
            set_lrs(optimizer, base_lrs, progress, epoch, spec)
            results = workers.collect(_weights(policy), norm.state(), T, progress)
            ro = stack_rollouts(results)  # [T, N, ...]
            if not (np.isfinite(ro["rewards"]).all() and np.isfinite(ro["values"]).all()):
                raise FloatingPointError("NaN/inf in rewards or values")
            done = ro["terminated"] | ro["truncated"]
            rewards = norm.scale_rewards(ro["rewards"], done)
            adv, ret = compute_gae(rewards, ro["values"], ro["terminated"], ro["truncated"], ro["final_values"],
                                   ro["last_values"], spec["gamma"], spec["gae_lambda"])
            flat = lambda x: torch.as_tensor(np.ascontiguousarray(x.reshape(T * N, *x.shape[2:]))).to(device)
            batch = {"obs": {k: flat(v) for k, v in norm.apply(ro["obs"]).items()},  # same stats the workers acted with
                     "actions": flat(ro["actions"]), "logp": flat(ro["logp"]), "values": flat(ro["values"]),
                     "adv": flat(adv), "ret": flat(ret)}
            stats = ppo_update(policy, optimizer, batch, spec)
            norm.update_obs({k: v.reshape(T * N, *v.shape[2:]) for k, v in ro["obs"].items()})
            steps += T * N
            rows.append({"epoch": epoch, "time": round(time.time() - start, 2), "steps": steps, "episodes": len(ro["episodes"]),
                         **_episode_means(ro["episodes"]), **stats,
                         "pi_lr": optimizer.param_groups[0]["lr"], "vf_lr": optimizer.param_groups[1]["lr"]})
            if epoch % 10 == 0 or epoch == epochs:
                ep = " | ".join(f"{k} {v:.3g}" for k, v in rows[-1].items() if k not in RESERVED)
                print(f"epoch {epoch}/{epochs} {rows[-1]['time']:.0f}s | eps {len(ro['episodes'])} | {ep} | "
                      f"v {stats['v_loss']:.3g} | kl {stats['approx_kl']:.3g} | ent {stats['entropy']:.3g}", flush=True)
            if epoch % spec["save_every"] == 0 and epoch != epochs:
                _save(run_dir, policy, norm, rows, optimizer)
    finally:
        workers.close()
    _save(run_dir, policy, norm, rows, optimizer)

    ep_keys = [k for k in dict.fromkeys(k for r in rows for k in r) if k not in RESERVED]
    tail = lambda k: [r[k] for r in rows[-10:] if k in r and not math.isnan(r[k])]
    allv = lambda k: [r[k] for r in rows if k in r and not math.isnan(r[k])]
    summary = {"epochs": epochs, "steps": steps, "wall_seconds": round(time.time() - start, 1),
               "final": {k: float(np.mean(tail(k))) for k in ep_keys if tail(k)},
               "best": {k: float(max(allv(k))) for k in ep_keys if allv(k)}}
    (run_dir / "done.json").write_text(json.dumps(summary, indent=1))
    return summary


def load_policy(run_dir):
    """Rebuild a trained policy for evaluation. Returns (policy in eval mode, Normalizer).
    Act with: policy.act(norm.apply(raw_obs), deterministic=True)."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    policy, norm = _build(spec)
    policy.load_state_dict(torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=True))
    norm.load(run_dir / "norm.npz")
    return policy.eval(), norm
