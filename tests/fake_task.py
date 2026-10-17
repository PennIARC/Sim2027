"""Tiny stand-ins used by the experiment-management tests (no torch)."""
import json


class FakeEnv:
    SPEC_DEFAULTS = {"fake_mode": "ok", "fake_sleep": 0.0}


def fake_policy(spec, obs_space, action_space):
    raise NotImplementedError


def fake_eval(run_dir):
    return {"win_rate": 0.75, "detail": {"nested": 1}, "n": 3}


class EvalEnv(FakeEnv):
    @staticmethod
    def evaluate(run_dir, procs=1):
        return {"procs_seen": procs}
