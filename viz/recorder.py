"""Record one AllocatorEnv episode (policy run dir or baseline) into a Recording; save/load as .npz (+ .json metadata).

No sim/* edits: snapshots are taken by wrapping two methods on the env *instance* (env._push_status, which _advance calls every
`alloc_status_interval_s` of sim time, and mapper.ingest_frame for detections). So the snapshot granularity is
max(interval_s, alloc_status_interval_s) (1 s by default); decisions are additionally recorded exactly (t, drone, blocks).
Pure numpy: no pygame import.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from sim import planner as planner_mod
from sim.allocator_env import AllocatorEnv, allocate
from sim.baselines import BASELINES, make_baseline
from sim.evaluate import EVAL_SEED0
from sim.geometry import N_BLOCKS
from sim.mines import load_eval_set
from sim.scoring import Path as FexlPath, buffer_mask, centerline_mask, fexl_url, score as official_score
from sim.sensor import footprint_dims

ARRAYS = ("t", "pos", "bel", "head", "rem", "idle", "block_owner", "pmine", "seen", "dets", "dec", "dec_blocks", "mines")


@dataclass
class Recording:
    t: np.ndarray            # (T,) snapshot times, s
    pos: np.ndarray          # (T, nd, 2) true drone xy, ft
    bel: np.ndarray          # (T, nd, 2) believed (GPS) xy
    head: np.ndarray         # (T, nd) heading, rad
    rem: np.ndarray          # (T, nd, M, 2) remaining path (true frame), NaN when idle
    idle: np.ndarray         # (T, nd) bool
    block_owner: np.ndarray  # (T, 240) int8 drone holding each block, -1 none
    pmine: np.ndarray        # (T, 300, 80) float16
    seen: np.ndarray         # (T, 300, 80) float16
    dets: np.ndarray         # (N, 4) float32: t, drone, x, y (believed frame) of every reported detection
    dec: np.ndarray          # (D, 2) float32: t, drone
    dec_blocks: np.ndarray   # (D, k) int16
    mines: np.ndarray        # (150, 40) bool ground truth
    meta: dict = field(default_factory=dict)  # seed, layout, allocator, score dict, path_text, fexl_url, footprint, ...

    def save(self, path) -> Path:
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, meta=np.array(json.dumps(self.meta)), **{k: getattr(self, k) for k in ARRAYS})
        path.with_suffix(".json").write_text(json.dumps(self.meta, indent=1))
        return path

    @classmethod
    def load(cls, path) -> "Recording":
        z = np.load(path, allow_pickle=False)
        return cls(meta=json.loads(str(z["meta"])), **{k: z[k] for k in ARRAYS})

    # derived final-path masks (squares)
    def final_path(self) -> FexlPath:
        return FexlPath.from_text(self.meta["path_text"])

    def masks(self):
        sq, _ = self.final_path().trace()
        return centerline_mask(sq), buffer_mask(sq, self.final_path().G)


def resolve_layout(layout_id: str):
    """fexl layout id like 'n135_s0.05' -> (eval index, bool grid)."""
    for i, (lid, grid) in enumerate(load_eval_set()):
        if lid == layout_id:
            return i, grid
    raise KeyError(f"unknown layout {layout_id!r}")


def record_episode(run_dir=None, baseline: str | None = None, seed: int | None = None, layout_id: str | None = None,
                   progress: float = 1.0, interval_s: float = 1.0, spec: dict | None = None) -> Recording:
    """Run one deterministic episode and record it. Exactly one of run_dir / baseline.
    layout_id (a fexl eval layout) uses the evaluation seed EVAL_SEED0 + its index unless seed is given."""
    if (run_dir is None) == (baseline is None):
        raise ValueError("give exactly one of run_dir or baseline")
    layout = None
    if layout_id is not None:
        idx, layout = resolve_layout(layout_id)
        seed = EVAL_SEED0 + idx if seed is None else seed
    elif seed is None:
        seed = 0
    if run_dir is not None:
        from rl.trainer import load_policy
        run_dir = Path(run_dir)
        spec = {**json.loads((run_dir / "spec.json").read_text()), **(spec or {})}
        policy, norm = load_policy(run_dir)
        name = f"policy:{run_dir.name}"
    else:
        if baseline not in BASELINES:
            raise KeyError(f"unknown baseline {baseline!r}; choose from {sorted(BASELINES)}")
        policy = norm = None
        name = f"baseline:{baseline}"

    env = AllocatorEnv(spec or {})
    if baseline is not None:
        fn = make_baseline(baseline, env.spec, **({"seed": seed} if baseline == "random" else {}))
    else:
        fn = lambda inp: allocate(policy, inp, deterministic=True, spec=env.spec, norm=norm)
    env.set_progress(progress)

    nd, M = env.n_drones, int(env.spec["drone_remaining_pts"])
    snaps: list = []
    dets: list = []
    state = {"next": 0.0, "last": -1.0}

    def snapshot():
        T = env.t
        rem = np.full((nd, M, 2), np.nan, np.float32)
        for d in env.drones:
            if d.has_path and not d.done:
                rem[d.idx] = d.remaining_path(M)
        b = env.mapper.belief
        snaps.append((T, np.array([d.pos for d in env.drones], np.float32), np.array([d.believed_pos for d in env.drones], np.float32),
                      np.array([d.heading for d in env.drones], np.float32), rem, np.array([d.done for d in env.drones], bool),
                      b.allocated[5::10, 5::10].ravel().astype(np.int8), b.p_mine.astype(np.float16), b.seen.astype(np.float16)))
        state["last"] = T

    orig_push = env._push_status

    def push(which=None):
        orig_push(which)
        if env.t >= state["next"] - 1e-9 and env.t > state["last"] + 1e-9:
            snapshot()
            state["next"] = env.t + interval_s

    env._push_status = push
    env.reset(seed=seed, layout=layout)
    orig_ingest = env.mapper.ingest_frame

    def ingest(report):
        for det in report.detections:
            dets.append((report.t, report.drone_id, det.xy[0], det.xy[1]))
        orig_ingest(report)

    env.mapper.ingest_frame = ingest
    decisions: list = []
    while True:
        inp = env.last_input
        alloc = fn(inp)
        decisions.append((env.t, inp.drone_id, list(alloc.blocks)))
        _, _, term, trunc, info = env.step(np.array(alloc.blocks))
        if term or trunc:
            break
    snapshot()  # final state (t = episode end) — a duplicate time is fine for the last frame
    ep = info["episode"]

    # the final path: re-plan on the same belief (deterministic) and score with the official scorer
    path, diag = planner_mod.plan(env.mapper.p_squares(), env.spec)
    res = official_score(path, env.mines, env.spec["planner_A"], env.spec["planner_N"])
    assert abs(res["score"] - ep["score"]) < 1e-6 * max(1.0, abs(ep["score"])), "re-planned path differs from the env's"
    n_mines = int(env.mines.sum())
    cols = list(zip(*snaps))
    st = lambda i, dt=None: np.stack(cols[i]).astype(dt) if dt else np.stack(cols[i])
    across, along = footprint_dims(env.spec)
    meta = {"allocator": name, "seed": int(seed), "layout": layout_id, "progress": float(progress), "interval_s": float(interval_s),
            "n_drones": nd, "k": env.k, "episode_s": env.t_end, "n_mines": n_mines, "footprint_ft": [float(across), float(along)],
            "score": {k: float(res[k]) for k in ("score", "B", "L", "W", "on_path_mines")} | {"G": int(path.G), "valid": bool(res["valid"])},
            "episode": ep, "path_text": path.to_text(),
            "fexl_url": fexl_url(path, n_mines, 0) if layout_id is None else _layout_url(path, layout_id),
            "granularity": f"snapshot every ~{float(np.median(np.diff(cols[0]))):.2f}s sim (env status cadence) + exact decision events"}
    return Recording(t=st(0, np.float32), pos=st(1), bel=st(2), head=st(3), rem=st(4), idle=st(5), block_owner=st(6), pmine=st(7),
                     seen=st(8), dets=np.array(dets, np.float32).reshape(-1, 4), dec=np.array([(a, b) for a, b, _ in decisions], np.float32),
                     dec_blocks=np.array([c for _, _, c in decisions], np.int16), mines=env.mines.copy(), meta=meta)


def _layout_url(path: FexlPath, layout_id: str) -> str:
    """Permalink for a fexl layout id 'n<num_danger>_s<seed>' (the layout's own seed)."""
    n, s = layout_id.split("_")
    return fexl_url(path, int(n[1:]), float(s[1:]))
