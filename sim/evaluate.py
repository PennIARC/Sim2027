"""Evaluation of the allocator on the 150 fexl layouts (100/135/170 mines), policy and baselines.

    evaluate(run_dir, procs=4) -> dict        the eval hook for spec key "eval": "sim.evaluate:evaluate"
    evaluate_baselines(spec, procs, ...)      random / lawnmower / greedy, cached in data/eval_baselines/
CLI:  python3 -m sim.evaluate baselines [--n 30] [--procs 4] [--force]
      python3 -m sim.evaluate run RUN_DIR [--n 30] [--procs 4]

Protocol: every layout gets the fixed seed EVAL_SEED0 + its index (sensor noise, mine positions, drift, random baseline), the GPS
drift is at its final level (progress 1.0) and the policy acts deterministically. Episodes run in a spawn Pool of module functions.
Baseline results are cached per layout id under data/eval_baselines/<name>_<hash of the env spec>.json (delete or pass force=True
after changing sim code).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np

EVAL_SEED0 = 1_000_000
CACHE_DIR = Path(__file__).resolve().parents[1] / "data" / "eval_baselines"
BASELINE_NAMES = ("random", "lawnmower", "greedy")
_G: dict = {}  # per-worker-process globals


# ---------------------------------------------------------------------------------------------------------------------
# pool workers (module-level so the spawn context can import them)
# ---------------------------------------------------------------------------------------------------------------------
def _init_worker(run_dir, spec) -> None:
    os.environ["OMP_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    _G["spec"] = spec
    if run_dir is not None:
        from rl.trainer import load_policy
        _G["policy"], _G["norm"] = load_policy(run_dir)


def _policy_episode(job) -> dict:
    from sim.allocator_env import AllocatorEnv
    idx, lid, layout = job
    env = AllocatorEnv(_G["spec"])
    env.set_progress(1.0)
    obs = env.reset(seed=EVAL_SEED0 + idx, layout=layout)
    policy, norm = _G["policy"], _G["norm"]
    while True:
        action, _, _ = policy.act(norm.apply(obs), deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        if term or trunc:
            return {"id": lid, **info["episode"]}


def _baseline_episode(job) -> dict:
    from sim.baselines import make_baseline, run_episode
    name, idx, lid, layout = job
    seed = EVAL_SEED0 + idx
    kw = {"seed": seed} if name == "random" else {}
    return {"id": lid, **run_episode(make_baseline(name, _G["spec"], **kw), seed, _G["spec"], layout, progress=1.0)}


def _pool_map(fn, jobs, procs, run_dir, spec) -> list:
    if not jobs:
        return []
    ctx = mp.get_context("spawn")
    saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")  # inherited by the spawned children before they import torch
    try:
        with ctx.Pool(max(1, min(procs, len(jobs))), initializer=_init_worker, initargs=(run_dir, spec)) as pool:
            return pool.map(fn, jobs, chunksize=1)
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)


# ---------------------------------------------------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------------------------------------------------
def summarize(rows: list, prefix: str = "") -> dict:
    """Flat metrics from per-layout episode dicts (each has 'id' like 'n135_s7' and the env's episode stats); per mine count
    (100/135/170) the score / death / B / W subset gets an _n<count> suffix."""
    def block(rs):
        m = lambda k: float(np.mean([r[k] for r in rs]))
        return {"mean_score": m("score"), "median_score": float(np.median([r["score"] for r in rs])), "mean_B": m("B"),
                "mean_W": m("W"), "mean_L": m("L"), "onpath_death_rate": m("died"), "mean_G": m("G"),
                "mean_seen_frac": m("seen_frac"), "mean_decisions": m("decisions"), "n": len(rs)}
    out = {f"{prefix}{k}": v for k, v in block(rows).items()}
    count = lambda r: int(r["id"].split("_")[0][1:])
    for n in sorted({count(r) for r in rows}):
        sub = block([r for r in rows if count(r) == n])
        out.update({f"{prefix}{k}_n{n}": sub[k] for k in ("mean_score", "median_score", "onpath_death_rate", "mean_B", "mean_W")})
    return out


def _layouts(n=None) -> list:
    from sim.mines import load_eval_set
    ev = load_eval_set()
    if n is not None and n < len(ev):  # evenly spread subset so every mine count is represented
        ev = [ev[i] for i in np.linspace(0, len(ev) - 1, n).astype(int)]
    return ev


def _full_spec(spec: dict | None) -> dict:
    from sim.allocator_env import AllocatorEnv
    return {**AllocatorEnv.SPEC_DEFAULTS, **(spec or {})}


def _env_key(spec: dict) -> str:
    from sim.allocator_env import AllocatorEnv
    sub = {k: spec[k] for k in sorted(AllocatorEnv.SPEC_DEFAULTS)}
    return hashlib.sha1(json.dumps(sub, sort_keys=True, default=list).encode()).hexdigest()[:10]


# ---------------------------------------------------------------------------------------------------------------------
# baselines
# ---------------------------------------------------------------------------------------------------------------------
def evaluate_baselines(spec: dict | None = None, procs: int = 4, names=BASELINE_NAMES, n: int | None = None,
                       force: bool = False) -> dict:
    """{name: flat metrics dict, ...} plus "rows": {name: [per-layout dict]}; results cached in data/eval_baselines/."""
    spec = _full_spec(spec)
    key, layouts = _env_key(spec), _layouts(n)
    index = {lid: i for i, (lid, _) in enumerate(_layouts())}
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {"rows": {}}
    for name in names:
        path = CACHE_DIR / f"{name}_{key}.json"
        cache = {} if force or not path.exists() else json.loads(path.read_text())
        todo = [(name, index[lid], lid, lay) for lid, lay in layouts if lid not in cache]
        if todo:
            t0 = time.time()
            for r in _pool_map(_baseline_episode, todo, procs, None, spec):
                cache[r["id"]] = r
            path.write_text(json.dumps(cache))
            print(f"baseline {name}: {len(todo)} episodes in {time.time() - t0:.0f}s", flush=True)
        rows = [cache[lid] for lid, _ in layouts]
        out["rows"][name] = rows
        out[name] = summarize(rows)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# the eval hook
# ---------------------------------------------------------------------------------------------------------------------
def evaluate(run_dir, procs: int = 4, n: int | None = None, with_baselines: bool = True) -> dict:
    """Eval hook: deterministic policy on the fexl layouts (+ cached baselines). Returns a flat dict of metrics
    (policy: score_mean..., per mine count suffix _n100/_n135/_n170; baselines under the prefix baseline_<name>_)
    plus nested "per_layout" for eval.json."""
    run_dir = Path(run_dir)
    spec = json.loads((run_dir / "spec.json").read_text())
    layouts = _layouts(n)
    index = {lid: i for i, (lid, _) in enumerate(_layouts())}
    t0 = time.time()
    rows = _pool_map(_policy_episode, [(index[lid], lid, lay) for lid, lay in layouts], procs, run_dir, spec)
    out = summarize(rows)
    out["eval_wall_s"] = time.time() - t0
    if with_baselines:
        base = evaluate_baselines(spec, procs, n=n)
        for name in BASELINE_NAMES:
            out.update({f"baseline_{name}_{k}": v for k, v in base[name].items()})
        best = max(base[nm]["mean_score"] for nm in BASELINE_NAMES)
        out["score_minus_best_baseline"] = out["mean_score"] - best
    out["per_layout"] = rows
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baselines")
    r = sub.add_parser("run")
    r.add_argument("run_dir")
    for p in (b, r):
        p.add_argument("--n", type=int, default=None, help="evaluate an evenly spread subset of n layouts (default all 150)")
        p.add_argument("--procs", type=int, default=4)
    b.add_argument("--force", action="store_true")
    b.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="override env spec keys")
    a = ap.parse_args(argv)
    if a.cmd == "baselines":
        spec = {k: json.loads(v) for k, v in (s.split("=", 1) for s in a.set)}
        res = evaluate_baselines(spec, a.procs, n=a.n, force=a.force)
        print(f"{'allocator':<10} {'mean':>7} {'median':>7} {'B':>6} {'W':>6} {'L':>6} {'death':>6} {'seen':>6} {'dec':>5}")
        for name in BASELINE_NAMES:
            m = res[name]
            print(f"{name:<10} {m['mean_score']:7.2f} {m['median_score']:7.2f} {m['mean_B']:6.2f} {m['mean_W']:6.1f} {m['mean_L']:6.0f} "
                  f"{m['onpath_death_rate']:6.2f} {m['mean_seen_frac']:6.3f} {m['mean_decisions']:5.0f}   n={m['n']}")
    else:
        res = evaluate(a.run_dir, a.procs, a.n)
        print(json.dumps({k: v for k, v in res.items() if k != "per_layout"}, indent=1))


if __name__ == "__main__":
    from sim.evaluate import main as _main  # import as a real module so pool workers resolve sim.evaluate.* functions
    _main()
