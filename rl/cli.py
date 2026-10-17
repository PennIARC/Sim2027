"""Command line: python3 -m rl <command>. Heavy modules (torch, the trainer) are imported only by the commands that need them."""
import argparse
import json
import sys
from pathlib import Path

from rl import registry

COMMANDS = {
    "train-one": "train a single spec (JSON path or inline JSON) into a new run folder",
    "resume": "continue an interrupted run in place from its last checkpoint, then eval + finalize",
    "batch": "run a batch file: train (+ eval hook) every spec in it, resumable",
    "eval": "run the eval hook on finished runs: eval <run>... | --all | --missing",
    "index": "rebuild runs/index.json from the runs' meta.json files",
    "list": "compact table of recent runs (sortable by a stat)",
    "show": "print one run's meta summary",
    "fav": "toggle a run in favorites.json",
}


def build_parser():
    root = argparse.ArgumentParser(prog="python3 -m rl", description="PPO experiment management")
    sub = root.add_subparsers(dest="cmd", metavar="<command>", help="; ".join(f"{k}: {v}" for k, v in COMMANDS.items()))
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=None, help="runs directory (default $SIM2027_RUNS or <repo>/runs)")

    p = sub.add_parser("train-one", parents=[common], help=COMMANDS["train-one"])
    p.add_argument("spec", help="path to a spec JSON file, or inline JSON starting with '{'")
    p.add_argument("--run-dir", default=None, help="train into this already-prepared run (used by batch; the caller finalizes)")
    p.add_argument("--name"), p.add_argument("--group"), p.add_argument("--blurb", help="two lines: short name, then description (use \\n)")
    p.add_argument("--tags", help="comma-separated"), p.add_argument("--parent"), p.add_argument("--batch")

    p = sub.add_parser("resume", parents=[common], help=COMMANDS["resume"])
    p.add_argument("run", help="run id / name / prefix / folder path")

    p = sub.add_parser("batch", parents=[common], help=COMMANDS["batch"])
    p.add_argument("specs_file")
    p.add_argument("--parallel", type=int, default=2)
    p.add_argument("--num-envs", type=int, default=None, help="num_envs for specs that don't set it")
    p.add_argument("--batch", default=None, help="override the batch name")
    p.add_argument("--train-timeout", type=float, default=120, help="minutes per training run (default 120)")
    p.add_argument("--eval-timeout", type=float, default=30, help="minutes per evaluation (default 30)")

    p = sub.add_parser("eval", parents=[common], help=COMMANDS["eval"])
    p.add_argument("runs", nargs="*", help="run ids / names / prefixes / folder paths")
    p.add_argument("--all", action="store_true", help="every finished run that has an eval hook")
    p.add_argument("--missing", action="store_true", help="only finished runs with an eval hook and no metrics yet")
    p.add_argument("--procs", type=int, default=None)

    sub.add_parser("index", parents=[common], help=COMMANDS["index"])

    p = sub.add_parser("list", parents=[common], help=COMMANDS["list"])
    p.add_argument("--sort", default=None, help="stat to sort by, best first: 'primary', a metric, or a curve stat (final, best, ...)")
    p.add_argument("--asc", action="store_true", help="ascending instead of best-first")
    p.add_argument("--limit", type=int, default=20)
    p.add_argument("--status"), p.add_argument("--group"), p.add_argument("--tag")

    p = sub.add_parser("show", parents=[common], help=COMMANDS["show"])
    p.add_argument("run")
    p.add_argument("--spec", action="store_true", help="print the full spec")

    p = sub.add_parser("fav", parents=[common], help=COMMANDS["fav"])
    p.add_argument("run")
    return root


def _load_spec(arg):
    text = arg if arg.lstrip().startswith("{") else Path(arg).read_text()
    return json.loads(text)


def cmd_train_one(a):
    spec = _load_spec(a.spec)
    for key in ("name", "group", "blurb", "parent"):
        if getattr(a, key) is not None:
            spec[key] = getattr(a, key).replace("\\n", "\n") if key == "blurb" else getattr(a, key)
    if a.tags:
        spec["tags"] = [t.strip() for t in a.tags.split(",") if t.strip()]
    from rl import runner
    if a.run_dir:  # batch child: the parent prepared the folder and finalizes it
        done = runner.train_in_process(a.run_dir)
        print("done", json.dumps(done, default=str), flush=True)
        return 0
    run_dir, full = runner.prepare_run(spec, batch=a.batch, root=a.root)
    print(f"run {run_dir}", flush=True)
    status, error = "ok", None
    try:
        runner.train_in_process(run_dir)
        if runner.has_eval_hook(full):
            runner.run_eval(run_dir)
    except Exception as e:
        status, error = "train_failed", repr(e)
        print(f"failed: {error}", file=sys.stderr)
    meta = runner.finalize_run(run_dir, status, error)
    registry.build_index(a.root)
    print(f"{meta['name']}: {status} {meta['training'].get('summary', {}).get('final')}")
    return 0 if status == "ok" else 1


def cmd_resume(a):
    from rl import runner
    run_dir = Path(a.run) if Path(a.run, registry.META).exists() else registry.find_run(a.run, a.root)
    if (run_dir / "done.json").exists():
        raise ValueError(f"{run_dir.name} already finished training")
    registry.update_meta(run_dir, status="running")
    status, error = "ok", None
    try:
        runner.train_in_process(run_dir, resume=True)
        if runner.has_eval_hook(registry.read_meta(run_dir)["spec"]):
            runner.run_eval(run_dir)
    except Exception as e:
        status, error = "train_failed", repr(e)
        print(f"failed: {error}", file=sys.stderr)
    meta = runner.finalize_run(run_dir, status, error)
    registry.build_index(a.root)
    print(f"{meta['name']}: {status} {meta['training'].get('summary', {}).get('final')}")
    return 0 if status == "ok" else 1


def cmd_batch(a):
    from rl.batch import run_batch
    results = run_batch(a.specs_file, a.parallel, a.root, a.batch, a.num_envs, a.train_timeout * 60, a.eval_timeout * 60)
    return 0 if all(s == "ok" for s in results.values()) else 1


def cmd_eval(a):
    from rl import runner
    if a.all or a.missing:
        dirs = [d for d, m in registry.iter_runs(a.root) if m.get("status") in ("ok", "eval_failed") and runner.has_eval_hook(m["spec"])
                and not (a.missing and m.get("metrics"))]
    else:
        dirs = [Path(r) if Path(r, registry.META).exists() else registry.find_run(r, a.root) for r in a.runs]
    if not dirs:
        print("nothing to evaluate")
    failed = 0
    for d in dirs:
        meta = registry.read_meta(d)
        try:
            metrics = runner.run_eval(d, a.procs)
        except Exception as e:
            print(f"{d.name}: eval failed {e!r}", file=sys.stderr)
            runner.finalize_run(d, "eval_failed", repr(e))
            failed += 1
            continue
        if metrics is None:
            print(f"{d.name}: no eval hook (set spec 'eval' or give the env an 'evaluate' function)")
            continue
        runner.finalize_run(d, "ok" if meta.get("status") in ("ok", "eval_failed") else meta.get("status"))
        print(f"{d.name}: {_fmt_metrics(metrics)}", flush=True)
    registry.build_index(a.root)
    return 1 if failed else 0


def cmd_index(a):
    rows = registry.build_index(a.root)
    print(f"indexed {len(rows)} runs -> {registry.runs_root(a.root) / 'index.json'}")
    return 0


def _fmt(v):
    if isinstance(v, float):
        return f"{v:.4g}"
    return "-" if v is None else str(v)


def _fmt_metrics(m, limit=8):
    return ", ".join(f"{k}={_fmt(v)}" for k, v in list(m.items())[:limit] if isinstance(v, (int, float, str, bool)) or v is None)


def stat_value(row, stat):
    """Value of a stat in an index row: 'primary', or a row key, or curve.<stat>."""
    for key in (stat, f"curve.{stat}"):
        if key in row:
            return row[key]
    return None


def cmd_list(a):
    rows = [registry.index_row(d, m) for d, m in registry.iter_runs(a.root)]
    rows = [r for r in rows if (not a.status or r["status"] == a.status) and (not a.group or r["group"] == a.group) and (not a.tag or a.tag in r["tags"])]
    if a.sort:
        have = [r for r in rows if isinstance(stat_value(r, a.sort), (int, float))]
        rows = sorted(have, key=lambda r: stat_value(r, a.sort), reverse=not a.asc) + [r for r in rows if r not in have]
    else:
        rows.sort(key=lambda r: r.get("created") or "", reverse=True)
    rows = rows[:a.limit]
    header = ["name", "status", "primary", "created"] + ([a.sort] if a.sort and a.sort != "primary" else [])
    table = [header]
    for r in rows:
        prim = f"{r['primary_stat']}={_fmt(r['primary'])}" if r.get("primary") is not None else "-"
        line = [str(r["name"]), str(r["status"]), prim, (r.get("created") or "-")[:16].replace("T", " ")]
        if len(header) > 4:
            line.append(_fmt(stat_value(r, a.sort)))
        table.append(line)
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)).rstrip())
    return 0


def cmd_show(a):
    from rl.spec import DEFAULTS
    d = registry.find_run(a.run, a.root)
    m = registry.read_meta(d)
    print(f"{m['name']}  [{m['status']}]  {m.get('title') or ''}")
    for key in ("id", "group", "parent", "kind", "batch", "created", "finished"):
        print(f"  {key:9s} {_fmt(m.get(key))}")
    print(f"  tags      {', '.join(registry.all_tags(m))}")
    if m.get("blurb_text"):
        print(f"  blurb     {m['blurb_text']}")
    if m.get("model"):
        print(f"  model     {_fmt_metrics(m['model'])}")
    t = m.get("training", {})
    if t:
        print(f"  training  epochs={_fmt(t.get('epochs'))} wall_s={_fmt(t.get('wall_s'))}")
        if t.get("summary"):
            print(f"  curve     {_fmt_metrics(t['summary'], 20)}")
    if m.get("metrics"):
        print(f"  metrics   {_fmt_metrics(m['metrics'], 20)}")
    if m.get("error"):
        print(f"  error     {m['error']}")
    spec = m.get("spec", {})
    shown = spec if a.spec else {k: v for k, v in spec.items() if k not in DEFAULTS or DEFAULTS[k] != v}
    print(f"  spec{'' if a.spec else ' (non-default)'}  {json.dumps(shown, default=str)}")
    return 0


def cmd_fav(a):
    name = registry.find_run(a.run, a.root).name
    favs = registry.toggle_favorite(name, a.root)
    print(f"{name}: {'favorite' if name in favs else 'removed from favorites'}")
    return 0


def main(argv=None):
    parser = build_parser()
    a = parser.parse_args(argv)
    if not a.cmd:
        parser.print_help()
        sys.exit(0)
    try:
        code = globals()["cmd_" + a.cmd.replace("-", "_")](a)
    except (KeyError, ValueError, FileNotFoundError) as e:
        print(f"error: {e.args[0] if isinstance(e, KeyError) and e.args else e}", file=sys.stderr)
        sys.exit(2)
    sys.exit(code or 0)
