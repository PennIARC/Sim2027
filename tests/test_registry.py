import json
import sys
from datetime import datetime

import pytest

from rl import registry, runner

ENV, POL = "tests.fake_task:FakeEnv", "tests.fake_task:fake_policy"


def spec(name="a", **kw):
    return {"name": name, "blurb": "short title\nwhat it tests.", "group": "g", "tags": ["t"], "env": ENV, "policy": POL, **kw}


def test_no_torch_import():
    import subprocess
    code = "import sys, rl.registry, rl.runner, rl.cli, rl.batch; assert 'torch' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, cwd=registry.REPO)


def test_ids_and_collisions(tmp_path):
    when = datetime(2026, 3, 4, 5, 6)
    assert registry.run_id("my run/x", when) == "2026-03-04_0506_my-run-x"
    d1 = registry.new_run_dir("r", when, tmp_path)
    d2 = registry.new_run_dir("r", when, tmp_path)
    d3 = registry.new_run_dir("r", when, tmp_path)
    assert [d.name for d in (d1, d2, d3)] == ["2026-03-04_0506_r", "2026-03-04_0506_r-2", "2026-03-04_0506_r-3"]


def test_runs_root_env(tmp_path, monkeypatch):
    monkeypatch.setenv("SIM2027_RUNS", str(tmp_path / "x"))
    assert registry.runs_root() == tmp_path / "x"
    assert registry.runs_root(tmp_path / "y") == tmp_path / "y"
    monkeypatch.delenv("SIM2027_RUNS")
    assert registry.runs_root() == registry.REPO / "runs"


@pytest.mark.parametrize("change,msg", [
    ({"name": ""}, "name"), ({"blurb": None}, "blurb"), ({"group": None}, "group"), ({"tags": []}, "tags"),
    ({"tags": "x"}, "list of strings"), ({"blurb": "only one line"}, "two lines")])
def test_check_run_spec_errors(change, msg):
    with pytest.raises(ValueError, match=msg):
        registry.check_run_spec({**spec(), **change})
    registry.check_run_spec(spec())


def test_prepare_and_find(tmp_path):
    d, full = runner.prepare_run(spec("alpha"), batch="b1", root=tmp_path, when=datetime(2026, 1, 1, 10, 0))
    assert full["fake_mode"] == "ok" and full["primary_stat"] == "score"
    meta = registry.read_meta(d)
    assert meta["status"] == "running" and meta["title"] == "short title" and meta["batch"] == "b1" and "blurb" not in meta["spec"]
    assert json.loads((d / "spec_in.json").read_text())["name"] == "alpha"
    d2, _ = runner.prepare_run(spec("beta"), root=tmp_path, when=datetime(2026, 1, 1, 11, 0))
    assert registry.find_run(d.name, tmp_path) == d
    assert registry.find_run("alpha", tmp_path) == d
    assert registry.find_run("2026-01-01_1100", tmp_path) == d2
    with pytest.raises(KeyError):
        registry.find_run("2026-01-01", tmp_path)  # ambiguous prefix
    with pytest.raises(KeyError):
        registry.find_run("nope", tmp_path)
    d3, _ = runner.prepare_run(spec("alpha"), root=tmp_path, when=datetime(2026, 1, 2))
    with pytest.raises(KeyError, match="ambiguous"):
        registry.find_run("alpha", tmp_path)
    assert registry.find_run(d3.name, tmp_path) == d3


def test_prepare_rejects_bad_specs(tmp_path):
    with pytest.raises(ValueError, match="unknown spec keys"):
        runner.prepare_run(spec(bogus=1), root=tmp_path)
    with pytest.raises(ValueError):
        runner.prepare_run(spec(group=None), root=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_parent_resolved_to_run_id(tmp_path):
    p, _ = runner.prepare_run(spec("p"), root=tmp_path)
    c, _ = runner.prepare_run(spec("c", parent="p"), root=tmp_path)
    assert registry.read_meta(c)["parent"] == p.name


def test_auto_tags(tmp_path):
    d, _ = runner.prepare_run(spec("a"), batch="b9", root=tmp_path)
    tags = registry.all_tags(registry.read_meta(d))
    assert tags == ["t", "kind:ppo", "batch:b9", "env:FakeEnv", "policy:fake_policy"]
    assert registry.auto_tags({"spec": {}}) == ["kind:ppo"]


def write_csv(path, values):
    path.write_text("epoch,time,steps,episodes,score\n" + "\n".join(f"{i + 1},{i},{i * 10},2,{v}" for i, v in enumerate(values)) + "\n")


def test_training_summary(tmp_path):
    write_csv(tmp_path / "e.csv", [i / 10 for i in range(20)])
    s = runner.training_summary(tmp_path / "e.csv")
    assert s["epochs_logged"] == 20 and s["primary_stat"] == "score"
    assert s["final"] == pytest.approx(sum(range(10, 20)) / 100)
    assert s["at_100pct"] > s["at_50pct"] > s["at_25pct"]
    assert s["epochs_to_half_gain"] is not None and s["total_steps"] == 190
    assert "reaches" not in json.dumps(s)
    assert "warning" in runner.training_summary(tmp_path / "e.csv", primary="other")


def test_finalize_and_index(tmp_path):
    d, _ = runner.prepare_run(spec("a", eval="tests.fake_task:fake_eval"), batch="b", root=tmp_path)
    write_csv(d / "epochs.csv", [1, 2, 3])
    (d / "done.json").write_text(json.dumps({"epochs": 3, "wall_s": 2.0}))
    (d / "model.json").write_text(json.dumps({"params": 123}))
    assert runner.run_eval(d) == {"win_rate": 0.75, "detail": {"nested": 1}, "n": 3}
    meta = runner.finalize_run(d, "ok")
    assert meta["status"] == "ok" and meta["finished"] and meta["model"] == {"params": 123}
    assert meta["metrics"] == {"win_rate": 0.75, "n": 3}  # nested dict dropped from flat metrics
    assert meta["training"]["epochs"] == 3 and meta["training"]["wall_s"] == 2.0
    first = meta["finished"]
    assert runner.finalize_run(d, "ok")["finished"] == first
    rows = registry.build_index(tmp_path)
    assert len(rows) == 1 and (tmp_path / "index.json").exists()
    r = rows[0]
    assert r["name"] == "a" and r["model.params"] == 123 and r["win_rate"] == 0.75 and r["primary"] == r["curve.final"]
    assert r["hp.pi_lr"] == 3e-4 and "env:FakeEnv" in r["tags"]
    assert registry.load_index(tmp_path) == rows


def test_eval_hook_resolution(tmp_path):
    assert runner.resolve_eval_hook({"env": ENV}) is None
    assert runner.has_eval_hook({"env": "tests.fake_task:EvalEnv"})
    d, _ = runner.prepare_run(spec("e", env="tests.fake_task:EvalEnv"), root=tmp_path)
    assert runner.run_eval(d, procs=4) == {"procs_seen": 4}
    d2, _ = runner.prepare_run(spec("n"), root=tmp_path)
    assert runner.run_eval(d2) is None


def test_favorites(tmp_path):
    assert registry.favorites(tmp_path) == []
    assert registry.toggle_favorite("x", tmp_path) == ["x"]
    assert registry.toggle_favorite("y", tmp_path) == ["x", "y"]
    assert registry.toggle_favorite("x", tmp_path) == ["y"]
    assert registry.favorites(tmp_path) == ["y"]
