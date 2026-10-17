import json
import subprocess
import sys

import pytest

from rl import cli, registry, runner

ENV, POL = "tests.fake_task:FakeEnv", "tests.fake_task:fake_policy"


def make(root, name, scores, status="ok", when=None, **kw):
    d, _ = runner.prepare_run({"name": name, "blurb": f"{name}\ndesc", "group": "g", "tags": ["t"], "env": ENV, "policy": POL, **kw}, root=root, when=when)
    (d / "epochs.csv").write_text("epoch,time,steps,episodes,score\n" + "\n".join(f"{i},{i},{i},1,{v}" for i, v in enumerate(scores, 1)) + "\n")
    runner.finalize_run(d, status)
    return d


def run(argv, root, capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([argv[0], "--root", str(root), *argv[1:]])
    out = capsys.readouterr()
    return e.value.code, out.out, out.err


def test_parser_train_one_and_batch():
    p = cli.build_parser()
    a = p.parse_args(["train-one", "{}", "--name", "n", "--tags", "a,b", "--root", "/x"])
    assert (a.name, a.tags, a.root, a.run_dir) == ("n", "a,b", "/x", None)
    a = p.parse_args(["batch", "f.json", "--parallel", "3", "--train-timeout", "0.5"])
    assert (a.parallel, a.train_timeout, a.eval_timeout) == (3, 0.5, 30)
    a = p.parse_args(["eval", "--missing"])
    assert a.missing and a.runs == []
    with pytest.raises(SystemExit):
        p.parse_args(["imitate"])


def test_list_and_sort(tmp_path, capsys):
    make(tmp_path, "low", [0.1] * 12)
    make(tmp_path, "high", [0.9] * 12)
    make(tmp_path, "bad", [0.5] * 12, status="train_failed")
    code, out, _ = run(["list"], tmp_path, capsys)
    assert code == 0
    lines = out.strip().splitlines()
    assert lines[0].split() == ["name", "status", "primary", "created"] and len(lines) == 4
    assert "score=0.9" in out and "train_failed" in out
    _, out, _ = run(["list", "--sort", "final"], tmp_path, capsys)
    assert [l.split()[0] for l in out.strip().splitlines()[1:]] == ["high", "bad", "low"]
    _, out, _ = run(["list", "--sort", "primary", "--asc", "--status", "ok"], tmp_path, capsys)
    assert [l.split()[0] for l in out.strip().splitlines()[1:]] == ["low", "high"]
    _, out, _ = run(["list", "--limit", "1"], tmp_path, capsys)
    assert len(out.strip().splitlines()) == 2


def test_show(tmp_path, capsys):
    make(tmp_path, "shown", [1.0] * 12, pi_lr=0.01)
    code, out, _ = run(["show", "shown"], tmp_path, capsys)
    assert code == 0 and "shown  [ok]" in out and "env:FakeEnv" in out and "final=1" in out
    assert '"pi_lr": 0.01' in out and '"gamma"' not in out
    _, out, _ = run(["show", "shown", "--spec"], tmp_path, capsys)
    assert '"gamma"' in out
    code, _, err = run(["show", "missing"], tmp_path, capsys)
    assert code != 0 and "matches no run" in err


def test_index_fav_eval(tmp_path, capsys):
    make(tmp_path, "one", [1] * 3)
    make(tmp_path, "ev", [1] * 3, eval="tests.fake_task:fake_eval")
    _, out, _ = run(["index"], tmp_path, capsys)
    assert "indexed 2 runs" in out and json.loads((tmp_path / "index.json").read_text())[0]["id"]
    _, out, _ = run(["fav", "one"], tmp_path, capsys)
    assert "favorite" in out and len(registry.favorites(tmp_path)) == 1
    _, out, _ = run(["eval", "--missing"], tmp_path, capsys)
    assert "win_rate=0.75" in out and "one" not in out
    assert registry.read_meta(registry.find_run("ev", tmp_path))["metrics"]["win_rate"] == 0.75
    _, out, _ = run(["eval", "--missing"], tmp_path, capsys)
    assert "nothing to evaluate" in out
    _, out, _ = run(["eval", "one"], tmp_path, capsys)
    assert "no eval hook" in out


def test_module_entry_and_root_env(tmp_path):
    make(tmp_path, "viaenv", [1] * 3)
    r = subprocess.run([sys.executable, "-m", "rl", "list"], capture_output=True, text=True, cwd=registry.REPO,
                       env={**__import__("os").environ, "SIM2027_RUNS": str(tmp_path)})
    assert r.returncode == 0 and "viaenv" in r.stdout
    r = subprocess.run([sys.executable, "-m", "rl", "--help"], capture_output=True, text=True, cwd=registry.REPO)
    assert "train-one" in r.stdout
