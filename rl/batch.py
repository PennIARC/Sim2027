"""Runs a batch of training (+ optional evaluation) sessions: python3 -m rl batch specs/<file>.json [--parallel P]

A batch file is {"batch": "b001", "group": "...", "tags": [...], "env": "...", "specs": [spec, ...]}
(a bare list of specs also works). every other top-level key (group, tags, parent, env, policy, epochs, ...) is a default for every spec; a spec's own keys win.
A spec without a name is called <batch>_<index>. Every spec needs name, two-line blurb, group, non-empty tags (registry.check_run_spec).
Each run is a subprocess (`python3 -m rl train-one --run-dir DIR DIR/spec_in.json`) in its own session, so on timeout the whole process
group, env workers included, is killed. Resumable: a spec whose run (same batch + name) already finished ok is skipped."""
import json
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rl import registry, runner

TRAIN_TIMEOUT_S, EVAL_TIMEOUT_S = 120 * 60, 30 * 60
_print_lock = threading.Lock()


def log_print(*a):
    with _print_lock:
        print(*a, flush=True)


def run_subprocess(cmd, log, timeout):
    """Runs cmd with output appended to log in a new session; kills the whole process group on timeout or failure. Returns (ok, reason)."""
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "PYTHONUNBUFFERED": "1"}
    with open(log, "a") as f:
        f.write(f"$ {' '.join(map(str, cmd))}\n")
        f.flush()
        proc = subprocess.Popen([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=registry.REPO, env=env, start_new_session=True)
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _killpg(proc.pid)
            proc.wait()
            f.write(f"\nTIMEOUT after {timeout}s\n")
            return False, "timeout"
        _killpg(proc.pid)  # reap any worker the command left behind
        if code != 0:
            return False, f"exit code {code}"
    return True, ""


def _killpg(pid):
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def tail(log, n=6):
    try:
        return " | ".join(l.strip() for l in Path(log).read_text().strip().splitlines()[-n:])[-600:]
    except OSError:
        return ""


def default_train_cmd(run_dir, spec_file):
    return [sys.executable, "-m", "rl", "train-one", "--run-dir", run_dir, spec_file]


def default_eval_cmd(run_dir):
    return [sys.executable, "-m", "rl", "eval", "--root", Path(run_dir).parent, run_dir]


def run_one(batch, spec, root, train_timeout=TRAIN_TIMEOUT_S, eval_timeout=EVAL_TIMEOUT_S, train_cmd=default_train_cmd, eval_cmd=default_eval_cmd):
    """Prepare, train (subprocess), evaluate if the spec has a hook (subprocess), finalize. Returns (run_dir, meta)."""
    t0 = time.time()
    run_dir, full = runner.prepare_run(spec, batch=batch, root=root)
    log = run_dir / "train.log"
    ok, why = run_subprocess(train_cmd(run_dir, run_dir / "spec_in.json"), log, train_timeout)
    status = "ok" if ok else "timeout" if why == "timeout" else "train_failed"
    if ok and runner.has_eval_hook(full):
        ok, why = run_subprocess(eval_cmd(run_dir), log, eval_timeout)
        if not ok:
            status = "eval_failed"
    meta = runner.finalize_run(run_dir, status, None if status == "ok" else f"{why}: {tail(log)}")
    meta["wall"] = round(time.time() - t0, 1)
    return run_dir, meta


def load_specs(path, batch_override=None):
    """(batch name, [spec dicts]) with the top-level defaults merged in."""
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):
        data = {"specs": data}
    batch = batch_override or data.get("batch") or Path(path).stem
    defaults = {k: v for k, v in data.items() if k not in ("batch", "specs")}
    specs = []
    for i, s in enumerate(data["specs"]):
        s = {**defaults, **s}
        s.setdefault("name", f"{batch}_{i:03d}")
        specs.append(s)
    return batch, specs


def already_done(batch, name, root):
    return any(m.get("name") == name and m.get("batch") == batch and m.get("status") == "ok" for _, m in registry.iter_runs(root))


def run_batch(specs_file, parallel=2, root=None, batch=None, num_envs=None, train_timeout=TRAIN_TIMEOUT_S, eval_timeout=EVAL_TIMEOUT_S,
              train_cmd=default_train_cmd, eval_cmd=default_eval_cmd):
    """Runs a batch file. Returns {run name: status} for the runs started (skipped specs are not included)."""
    batch, specs = load_specs(specs_file, batch)
    root = registry.runs_root(root)
    root.mkdir(parents=True, exist_ok=True)
    todo, names = [], set()
    for s in specs:
        if num_envs and "num_envs" not in s:
            s["num_envs"] = num_envs
        if s["name"] in names:
            raise ValueError(f"duplicate run name {s['name']}")
        names.add(s["name"])
        registry.check_run_spec(s)  # fail before anything starts
        if not already_done(batch, s["name"], root):
            todo.append(s)
    log_print(f"[{batch}] {len(todo)} to run ({len(specs) - len(todo)} skipped), parallel {parallel}")
    t0, results = time.time(), {}
    with ThreadPoolExecutor(max(1, parallel)) as pool:
        futures = [(s, pool.submit(run_one, batch, s, root, train_timeout, eval_timeout, train_cmd, eval_cmd)) for s in todo]
        for s, fut in futures:
            try:
                _, meta = fut.result()
            except Exception as e:  # never stop the batch
                results[s["name"]] = "runner_error"
                log_print(f"[{batch}] {s['name']}: runner error {e!r}")
                continue
            results[meta["name"]] = meta["status"]
            summ = meta.get("training", {}).get("summary", {})
            if meta["status"] == "ok":
                log_print(f"[{batch}] {meta['name']}: {summ.get('primary_stat')} {summ.get('final')} ({meta['wall']:.0f}s)")
            else:
                log_print(f"[{batch}] {meta['name']}: {meta['status']} {meta.get('error', '')[-140:]}")
    registry.build_index(root)
    log_print(f"[{batch}] finished in {(time.time() - t0) / 60:.1f} min: {sorted(set(results.values()))}")
    return results
