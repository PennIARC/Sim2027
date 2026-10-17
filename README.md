# Sim2027

PPO cell allocator for the IARC Mission 10 drone swarm. This repo holds a generic torch PPO stack (no gymnasium) plus an
experiment-management layer (run registry, batch runner, CLI); the task-specific simulator in `sim/` is still to come.

## Setup

```
pip install -r requirements.txt
python3 -m pytest -q
```

## Quick start

```
python3 -m rl train-one specs/one.json --name try1 --group scratch --tags toy --blurb $'short name\nwhat it tests'
python3 -m rl batch specs/example.json --parallel 2     # resumable; add --train-timeout 60 (minutes)
python3 -m rl list --sort final                         # recent runs, best first (also: --status ok --tag env:PickBestEnv)
python3 -m rl show <run>                                # run = id, id prefix or unique name; --spec for the full spec
python3 -m rl eval --missing                            # run the eval hook on runs that lack metrics
python3 -m rl index                                     # rebuild runs/index.json
python3 -m rl fav <run>                                 # toggle in runs/favorites.json
```

The runs directory is `<repo>/runs`, overridden by `$SIM2027_RUNS` or `--root DIR`.

## Specs and batches

A spec is a flat JSON dict; generic PPO keys are in `rl/spec.py` (`DEFAULTS`), and the env class and policy builder add their
own via `SPEC_DEFAULTS`. Unknown keys are rejected. Every run needs `name`, `blurb` (two lines: short title, then what it
tests), `group` and non-empty `tags`; `env` and `policy` are `"package.module:Name"` paths.

Extra keys: `primary_stat` (default `"score"`) names the per-episode stat in `epochs.csv` used for curve summaries and `list`;
`eval` (optional `"package.module:function"`) is called as `fn(run_dir) -> dict` after training (or `fn(run_dir, procs=N)` if it
takes `procs`); scalar results land in `meta["metrics"]` and `eval.json`. If `eval` is unset, an `evaluate` attribute on the env
class is used; with neither, evaluation is skipped.

A batch file is `{"batch": "b001", "group": ..., "tags": [...], "env": ..., "policy": ..., "specs": [{...}, ...]}`. Every
top-level key except `batch`/`specs` is a default for all specs; a spec's own keys win. Each spec runs as its own subprocess
(own process group, killed whole on timeout). A spec whose run (same batch + name) already finished `ok` is skipped, so
re-running a batch resumes it. See `specs/example.json`.

## Run folder layout

```
runs/<YYYY-MM-DD_HHMM>_<name>/   (-2, -3, ... on collision)
  meta.json       source of truth: id, name, title, blurb_text, group, parent, tags, kind, status
                  (running|ok|train_failed|eval_failed|timeout), created, finished, batch, model, spec, training, metrics
  spec_in.json    spec as submitted (defaults filled);  spec.json  full spec written by the trainer
  train.log       captured stdout/stderr of training and eval
  epochs.csv, policy.pt, norm.npz, checkpoints, done.json, eval.json   written by the trainer / eval hook
runs/index.json     flat table of all runs (rebuilt by `index`, and after each batch)
runs/favorites.json pinned run ids
```

## Running heavy jobs quietly (macOS)

Use `nice -n 10` (or `taskpolicy -b -p <pid>` on running processes, which moves them to the efficiency cores), set
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` (the batch runner already does this for its children), and wrap long runs in
`caffeinate -i` so the machine doesn't sleep:

```
caffeinate -i nice -n 10 python3 -m rl batch specs/example.json --parallel 4
```

## Replay viewer

`pip install -r requirements-viewer.txt` (adds pygame). Record one episode with a trained run or a baseline and watch it:

```
python3 -m viz.replay --run alloc_base --layout n135_s0.05 --save viz_out/a.npz   # run = id, unique name or id prefix
python3 -m viz.replay --baseline greedy --seed 3 --progress 1.0
python3 -m viz.replay --load viz_out/a.npz                                        # --no-view records/saves only
```

A layout id uses the evaluation seed of that layout (as `sim.evaluate` does); `--progress` is the GPS-drift curriculum point
(default 1.0 = evaluation). Snapshots are taken at the env's status cadence (about every 1.1 s of sim time) plus exact decision
events; recordings are `.npz` + `.json` metadata (put `viz_out/` in `.gitignore`). `viz.recorder.record_episode(...)` is the
pure-numpy API (no pygame). Keys: space play/pause, left/right step (shift = 10), home/end, up/down playback speed, click or drag
the slider, P p_mine, S seen, M true mines, A allocations, D drones/footprints (filled = true position, hollow = believed),
R remaining paths, F final path (blue) + buffer (green) with on-path mines orange and buffer mines yellow, E detections,
C copy fexl link, H help, Esc/Q quit.
