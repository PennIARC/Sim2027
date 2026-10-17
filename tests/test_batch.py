import json
import os
import sys
import time
from pathlib import Path

import pytest

from rl import batch, registry

FAKE = str(Path(__file__).parent / "fake_trainer.py")
ENV, POL = "tests.fake_task:FakeEnv", "tests.fake_task:fake_policy"


def fake_cmd(run_dir, spec_file):
    return [sys.executable, FAKE, "--run-dir", run_dir, spec_file]


def write_batch(tmp_path, specs, **top):
    p = tmp_path / "b.json"
    p.write_text(json.dumps({"batch": "bt", "group": "g", "tags": ["t"], "env": ENV, "policy": POL, **top, "specs": specs}))
    return p


def blurb(n):
    return f"{n}\ndescription of {n}"


def test_load_specs_merges_defaults(tmp_path):
    p = write_batch(tmp_path, [{"name": "a", "blurb": blurb("a")}, {"blurb": blurb("b"), "group": "other"}], parent="par")
    name, specs = batch.load_specs(p)
    assert name == "bt"
    assert specs[0]["group"] == "g" and specs[0]["tags"] == ["t"] and specs[0]["parent"] == "par"
    assert specs[1]["group"] == "other" and specs[1]["name"] == "bt_001"
    assert batch.load_specs(p, "zz")[0] == "zz"


def test_batch_runs_and_resumes(tmp_path):
    root = tmp_path / "runs"
    p = write_batch(tmp_path, [{"name": "a", "blurb": blurb("a")}, {"name": "b", "blurb": blurb("b"), "fake_mode": "fail"}, {"name": "c", "blurb": blurb("c"), "eval": "tests.fake_task:fake_eval"}])
    res = batch.run_batch(p, parallel=2, root=root, train_cmd=fake_cmd)
    assert res == {"a": "ok", "b": "train_failed", "c": "ok"}
    metas = {m["name"]: m for _, m in registry.iter_runs(root)}
    assert metas["a"]["training"]["summary"]["final"] > 0 and metas["a"]["batch"] == "bt"
    assert "boom" in metas["b"]["error"]
    assert metas["c"]["metrics"]["win_rate"] == 0.75  # real `python3 -m rl eval` subprocess
    assert (root / "index.json").exists()
    assert "trained" in (root / metas["a"]["id"] / "train.log").read_text()
    # resume: ok runs skipped, failed run retried
    res2 = batch.run_batch(p, parallel=2, root=root, train_cmd=fake_cmd)
    assert res2 == {"b": "train_failed"}
    assert sum(1 for _, m in registry.iter_runs(root) if m["name"] == "a") == 1
    assert sum(1 for _, m in registry.iter_runs(root) if m["name"] == "b") == 2


def test_batch_validates_before_starting(tmp_path):
    root = tmp_path / "runs"
    p = write_batch(tmp_path, [{"name": "a", "blurb": blurb("a")}, {"name": "x", "blurb": "one line"}])
    with pytest.raises(ValueError):
        batch.run_batch(p, root=root, train_cmd=fake_cmd)
    assert registry.iter_runs(root) == []
    p = write_batch(tmp_path, [{"name": "a", "blurb": blurb("a")}, {"name": "a", "blurb": blurb("a")}])
    with pytest.raises(ValueError, match="duplicate"):
        batch.run_batch(p, root=root, train_cmd=fake_cmd)


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


def test_timeout_kills_process_group(tmp_path):
    root = tmp_path / "runs"
    p = write_batch(tmp_path, [{"name": "h", "blurb": blurb("h"), "fake_mode": "hang"}])
    t0 = time.time()
    res = batch.run_batch(p, root=root, train_timeout=3, train_cmd=fake_cmd)
    assert res == {"h": "timeout"} and time.time() - t0 < 20
    d, meta = registry.iter_runs(root)[0]
    assert meta["status"] == "timeout" and "TIMEOUT" in (d / "train.log").read_text()
    pid = int((d / "child.pid").read_text())
    for _ in range(20):
        if not alive(pid):
            break
        time.sleep(0.1)
    assert not alive(pid)


def test_run_subprocess_env_and_exit(tmp_path):
    log = tmp_path / "l.log"
    code = "import os; print(os.environ['OMP_NUM_THREADS'], os.environ['MKL_NUM_THREADS'])"
    assert batch.run_subprocess([sys.executable, "-c", code], log, 20) == (True, "")
    assert "1 1" in log.read_text()
    assert batch.run_subprocess([sys.executable, "-c", "raise SystemExit(2)"], log, 20) == (False, "exit code 2")
