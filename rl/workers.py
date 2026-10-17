"""Rollout collection. Each worker owns one env + a copy of the policy and, on request, runs a whole stretch of
decisions alone (auto-reset, carry-over across epochs). Subprocess workers use the spawn context and Pipes; any
exception comes back as ("error", traceback) so the trainer fails with the real cause instead of hanging."""
import multiprocessing as mp
import os
import random
import traceback

import numpy as np
import torch

from rl.env import load_object, make_env
from rl.norm import Normalizer


class Collector:
    """One env + policy copy. collect() returns numpy arrays for n_steps decisions (see keys below)."""

    def __init__(self, spec: dict, rank: int, seed_global: bool = True):
        self.spec, self.seed = spec, spec["seed"] + rank
        if seed_global:  # subprocesses own their RNGs; inline mode leaves the trainer's global RNGs alone
            random.seed(self.seed), np.random.seed(self.seed), torch.manual_seed(self.seed)
        self.env = make_env(spec)
        self.policy = load_object(spec["policy"])(spec, self.env.obs_space, self.env.action_space).eval()
        self.norm = Normalizer(spec, self.env.obs_space)
        self.obs = self.env.reset(seed=self.seed)  # seeded once; later resets continue the env's own RNG stream
        self.ep_len = 0

    def collect(self, weights: dict, norm_state: dict, n_steps: int, progress: float) -> dict:
        """Act with `weights` (numpy state_dict) and normalizer stats. Returns, for T=n_steps:
        obs (dict of raw [T,...] arrays, incl. mask), actions, logp, values (V of obs_t), rewards, terminated,
        truncated, final_values (V of the final obs on truncation, else 0), last_value (V of the carried obs),
        episodes (list of info["episode"] dicts)."""
        self.policy.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
        self.norm.load_state(norm_state)
        if hasattr(self.env, "set_progress"):
            self.env.set_progress(progress)
        obs_buf = {k: [] for k in self.obs}
        acts, logp, vals, rew = [], np.zeros(n_steps, np.float32), np.zeros(n_steps, np.float32), np.zeros(n_steps, np.float32)
        term, trunc = np.zeros(n_steps, bool), np.zeros(n_steps, bool)
        final_values, episodes = np.zeros(n_steps, np.float32), []
        for t in range(n_steps):
            for k, v in self.obs.items():
                obs_buf[k].append(v)
            a, logp[t], vals[t] = self.policy.act(self.norm.apply(self.obs))
            acts.append(a)
            self.obs, rew[t], term[t], trunc[t], info = self.env.step(a)
            term[t], trunc[t] = bool(term[t]), bool(trunc[t]) and not term[t]
            if term[t] or trunc[t]:
                if trunc[t]:
                    final_values[t] = self.policy.value(self.norm.apply(self.obs))
                if "episode" in info:
                    episodes.append(info["episode"])
                self.obs = self.env.reset()
        return {"obs": {k: np.stack(v) for k, v in obs_buf.items()}, "actions": np.stack(acts), "logp": logp, "values": vals,
                "rewards": rew, "terminated": term, "truncated": trunc, "final_values": final_values,
                "last_value": self.policy.value(self.norm.apply(self.obs)), "episodes": episodes}


def _worker_main(conn, spec: dict, rank: int) -> None:
    try:
        torch.set_num_threads(int(spec["torch_threads"]))
        collector = Collector(spec, rank)
        while True:
            command, data = conn.recv()
            if command != "collect":
                break
            conn.send(("ok", collector.collect(*data)))
    except BaseException:
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


class InlineWorkers:
    """One env in the calling process (num_envs=0): for tests and debugging."""

    def __init__(self, spec: dict):
        self.n = 1
        self.collector = Collector(spec, 0, seed_global=False)

    def collect(self, weights, norm_state, n_steps, progress) -> list:
        return [self.collector.collect(weights, norm_state, n_steps, progress)]

    def close(self) -> None:
        pass


class SubprocWorkers:
    """spec["num_envs"] worker processes, each running its stretch of steps without waiting on the others."""

    def __init__(self, spec: dict):
        self.n = spec["num_envs"]
        ctx = mp.get_context("spawn")
        saved = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")}
        os.environ.update(OMP_NUM_THREADS=str(spec["torch_threads"]), MKL_NUM_THREADS=str(spec["torch_threads"]))  # inherited by spawned children, before they import torch
        self.conns, self.procs = [], []
        try:
            for rank in range(self.n):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_worker_main, args=(child, spec, rank), daemon=True)
                proc.start()
                child.close()
                self.conns.append(parent)
                self.procs.append(proc)
        finally:
            for k, v in saved.items():
                os.environ.pop(k, None) if v is None else os.environ.__setitem__(k, v)

    def collect(self, weights, norm_state, n_steps, progress) -> list:
        for conn in self.conns:
            try:
                conn.send(("collect", (weights, norm_state, n_steps, progress)))
            except (BrokenPipeError, OSError):
                pass  # the worker died; its error message (if any) is read below
        results = []
        for conn, proc in zip(self.conns, self.procs):
            while not conn.poll(0.5):
                if not proc.is_alive() and not conn.poll(0):
                    raise RuntimeError(f"env worker died (exit code {proc.exitcode})")
            status, payload = conn.recv()
            if status == "error":
                raise RuntimeError(f"env worker failed:\n{payload}")
            results.append(payload)
        return results

    def close(self) -> None:
        for conn in self.conns:
            try:
                conn.send(("close", None))
            except Exception:
                pass
        for proc in self.procs:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.terminate()


def make_workers(spec: dict, inline: bool = False):
    """Workers for the spec: inline when `inline` or num_envs == 0, else num_envs subprocesses."""
    return InlineWorkers(spec) if inline or spec["num_envs"] == 0 else SubprocWorkers(spec)
