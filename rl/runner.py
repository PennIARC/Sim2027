"""Creating a run folder before training and finishing its meta.json afterwards.

prepare_run(spec)  -> (run_dir, full_spec): validates, fills defaults, writes spec_in.json and meta.json (status "running").
train_in_process() -> calls rl.trainer.train (lazy import: this module never imports torch at load time).
run_eval()         -> the optional eval hook (see resolve_eval_hook).
finalize_run()     -> fills meta.json from the files the run left behind (done.json, epochs.csv, model.json, eval.json).

Eval hook: a run is evaluated only if it has one. `spec["eval"] = "package.module:function"` names it; otherwise the env class
(spec["env"]) may expose a callable attribute `evaluate`. It is called as fn(run_dir) (a pathlib.Path) -> dict of metrics, or
fn(run_dir, procs=N) if its signature has a `procs` parameter. Scalars of the returned dict are merged into meta["metrics"]
and the dict is saved as eval.json. No hook means evaluation is skipped."""
import csv
import inspect
import json
from pathlib import Path

import numpy as np

from rl import registry
from rl.registry import safe_float

CURVE_POINTS = (0.25, 0.5, 0.75, 1.0)


def training_summary(epochs_csv, primary="score"):
    """Curve statistics of the primary stat (an epochs.csv column): value at 25/50/75/100% of training, final mean/std (last 10 epochs),
    best smoothed value and epoch, epochs to cover half of the start-to-final gain, biggest drop from the running best; plus counts."""
    with open(epochs_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}

    def col(key):
        return np.array([float(r[key]) if r.get(key) not in (None, "", "nan") else np.nan for r in rows])

    n = len(rows)
    summary = {"epochs_logged": n, "primary_stat": primary}
    if "steps" in rows[0]:
        summary["total_steps"] = safe_float(col("steps")[-1])
    if "episodes" in rows[0]:
        summary["episodes_per_epoch"] = safe_float(np.nanmean(col("episodes")))
    if "time" in rows[0]:
        summary["last_time_s"] = safe_float(col("time")[-1])
    if primary not in rows[0]:
        summary["warning"] = f"primary stat {primary!r} is not a column of epochs.csv"
        return summary
    vals = col(primary)
    smooth = np.convolve(vals, np.ones(5) / 5, mode="valid") if n >= 5 else vals
    final = float(np.nanmean(vals[-10:]))
    start = float(smooth[0])
    gain = np.flatnonzero(smooth >= start + 0.5 * (final - start)) if final > start else []
    summary.update({"final": safe_float(final), "final_std": safe_float(np.nanstd(vals[-10:])), "best": safe_float(np.nanmax(smooth)),
                    "best_epoch": int(np.nanargmax(smooth)) + 1 + (n - len(smooth)) // 2,
                    "epochs_to_half_gain": int(gain[0]) + 1 if len(gain) else None,
                    "biggest_drop": safe_float(np.nanmax(np.maximum.accumulate(smooth) - smooth))})
    for q in CURVE_POINTS:
        end = max(1, int(q * n))
        summary[f"at_{int(q * 100)}pct"] = safe_float(np.nanmean(vals[max(0, end - 5):end]))
    return summary


def prepare_run(spec, batch=None, root=None, when=None):
    """Creates runs/<id>/ for a new run and writes spec_in.json and a first meta.json (status 'running'). Returns (run_dir, full spec)."""
    from rl.spec import make_spec
    registry.check_run_spec(spec)
    spec = dict(spec)
    if spec.get("parent"):
        try:
            spec["parent"] = registry.find_run(spec["parent"], root).name
        except KeyError:
            pass  # parent given as a free-form label
    full = make_spec(spec)
    run_dir = registry.new_run_dir(full["name"], when, root)
    title, text = registry.split_blurb(full["blurb"])
    meta = {"id": run_dir.name, "name": full["name"], "title": title, "blurb_text": text, "group": full["group"], "parent": full["parent"],
            "tags": list(full["tags"]), "kind": full["kind"], "status": "running", "created": registry.now_iso(), "finished": None,
            "batch": batch, "model": {}, "spec": {k: v for k, v in full.items() if k != "blurb"}, "training": {}, "metrics": {}}
    registry.dump_json(run_dir / "spec_in.json", full)
    registry.write_meta(run_dir, meta)
    return run_dir, full


def train_in_process(run_dir, resume=False):
    """Trains the run in this process from run_dir/spec_in.json; also records the model's parameter count in model.json.
    resume=True continues an interrupted run from its last checkpoint (see rl.trainer.train)."""
    from rl.trainer import train
    run_dir = Path(run_dir)
    summary = train(json.loads((run_dir / "spec_in.json").read_text()), run_dir, resume=resume)
    registry.dump_json(run_dir / "train_summary.json", summary)
    try:
        from rl.trainer import load_policy
        policy, _ = load_policy(run_dir)
        registry.dump_json(run_dir / "model.json", {"params": int(sum(p.numel() for p in policy.parameters()))})
    except Exception as e:  # the count is optional
        print(f"could not count model parameters: {e!r}", flush=True)
    return summary


def resolve_eval_hook(spec):
    """The eval callable for a spec (spec["eval"], else the env class's `evaluate`), or None."""
    from rl.env import load_object
    if spec.get("eval"):
        return load_object(spec["eval"])
    if spec.get("env"):
        hook = getattr(load_object(spec["env"]), "evaluate", None)
        return hook if callable(hook) else None
    return None


def run_eval(run_dir, procs=None):
    """Runs the eval hook on a finished run, writes eval.json and returns the metrics; None if the run has no hook."""
    run_dir = Path(run_dir)
    spec = registry.read_meta(run_dir)["spec"]
    hook = resolve_eval_hook(spec)
    if hook is None:
        return None
    kwargs = {"procs": procs} if procs and "procs" in inspect.signature(hook).parameters else {}
    metrics = dict(hook(run_dir, **kwargs))
    registry.dump_json(run_dir / "eval.json", metrics)
    return metrics


def has_eval_hook(spec):
    try:
        return resolve_eval_hook(spec) is not None
    except Exception:
        return False


def finalize_run(run_dir, status="ok", error=None):
    """Fills meta.json from the files training and evaluation left in the run folder; returns the meta."""
    run_dir = Path(run_dir)
    meta = registry.read_meta(run_dir)
    changes = {"status": status}
    if meta.get("finished") is None:
        changes["finished"] = registry.now_iso()  # set once; re-evaluating later must not move it
    if error:
        changes["error"] = error[-600:]
    training = {}
    for fname in ("train_summary.json", "done.json"):
        if (run_dir / fname).exists():
            d = json.loads((run_dir / fname).read_text())
            training.update({"epochs": d.get("epochs"), "wall_s": d.get("wall_s", d.get("wall_seconds"))})
            break
    if (run_dir / "epochs.csv").exists():
        training["summary"] = training_summary(run_dir / "epochs.csv", meta["spec"].get("primary_stat") or "score")
    if training:
        changes["training"] = training
    if (run_dir / "model.json").exists():
        changes["model"] = json.loads((run_dir / "model.json").read_text())
    if (run_dir / "eval.json").exists():
        metrics = json.loads((run_dir / "eval.json").read_text())
        changes["metrics"] = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool)) or v is None}
    return registry.update_meta(run_dir, **changes)
