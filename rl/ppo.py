"""PPO update: clipped surrogate + vf_coef * value MSE - entropy_coef * entropy, minibatch epochs, LR schedules."""
import numpy as np
import torch
import torch.nn as nn


def lr_factor(progress: float, spec: dict) -> float:
    """Multiplier on the base LRs: 1 at progress 0, lr_decay_end at progress 1 (linear / exponential / constant)."""
    kind, end = spec["lr_schedule"], spec["lr_decay_end"]
    if kind == "linear":
        return 1 - (1 - end) * progress
    if kind == "exponential":
        return end ** progress
    if kind == "constant":
        return 1.0
    raise ValueError(f"unknown lr_schedule {kind!r}")


def make_optimizer(policy, spec: dict):
    """One Adam with separate actor (pi_lr) and critic (vf_lr) groups. Returns (optimizer, base_lrs)."""
    base = [spec["pi_lr"], spec["vf_lr"]]
    groups = [{"params": policy.actor_parameters(), "lr": base[0]}, {"params": policy.critic_parameters(), "lr": base[1]}]
    return torch.optim.Adam(groups), base


def set_lrs(optimizer, base_lrs, progress: float, epoch: int, spec: dict) -> None:
    """Schedule the LRs for this epoch; the actor LR is 0 during the first critic_warmup_epochs (epoch is 1-based)."""
    f = lr_factor(progress, spec)
    for group, lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = lr * f
    if epoch <= spec["critic_warmup_epochs"]:
        optimizer.param_groups[0]["lr"] = 0.0


def explained_variance(pred: torch.Tensor, target: torch.Tensor) -> float:
    var = target.var()
    return float("nan") if var < 1e-12 else float(1 - (target - pred).var() / var)


def ppo_update(policy, optimizer, batch: dict, spec: dict) -> dict:
    """One PPO update on a flat batch of tensors: obs (dict), actions, logp, adv, ret, values (old critic values).
    Returns the mean over all minibatch steps of pi_loss, v_loss, entropy, approx_kl, clip_frac, plus explained_var."""
    B = len(batch["adv"])
    adv_all = (batch["adv"] - batch["adv"].mean()) / (batch["adv"].std() + 1e-8)
    mb_size, clip, target_kl = max(B // spec["num_minibatches"], 1), spec["clip"], spec["target_kl"]
    acc = {k: [] for k in ("pi_loss", "v_loss", "entropy", "approx_kl", "clip_frac")}
    policy.train()
    stop = False
    for _ in range(spec["update_epochs"]):
        for idx in torch.randperm(B).split(mb_size):
            # batch obs may be stored in a compact dtype (e.g. float16 maps): only the minibatch is cast to float32
            obs = {k: (v[idx].float() if v.is_floating_point() else v[idx]) for k, v in batch["obs"].items()}
            act, adv = batch["actions"][idx], adv_all[idx]
            dist, value = policy.dist_and_value(obs)
            log_ratio = dist.log_prob(act) - batch["logp"][idx]
            ratio = log_ratio.exp()
            with torch.no_grad():
                kl = ((ratio - 1) - log_ratio).mean().item()
            if target_kl is not None and kl > 1.5 * target_kl:
                stop = True  # policy already moved too far this epoch; skip this step
                break
            pi_loss = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
            v_loss = ((value - batch["ret"][idx]) ** 2).mean()
            entropy = dist.entropy(act).mean()
            loss = pi_loss + spec["vf_coef"] * v_loss - spec["entropy_coef"] * entropy
            optimizer.zero_grad()
            loss.backward()
            if spec["max_grad_norm"]:
                nn.utils.clip_grad_norm_(policy.parameters(), spec["max_grad_norm"])
            optimizer.step()
            for k, v in zip(acc, (pi_loss.item(), v_loss.item(), entropy.item(), kl, ((ratio - 1).abs() > clip).float().mean().item())):
                acc[k].append(v)
        if stop:
            break
    out = {k: float(np.mean(v)) if v else float("nan") for k, v in acc.items()}
    out["explained_var"] = explained_variance(batch["values"], batch["ret"])
    if not all(np.isfinite(out[k]) for k in ("pi_loss", "v_loss")) and acc["pi_loss"]:
        raise FloatingPointError(f"non-finite loss: {out}")
    return out
