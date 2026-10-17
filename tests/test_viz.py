import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pytest

from viz.recorder import Recording, record_episode


@pytest.fixture(scope="module")
def rec():
    return record_episode(baseline="lawnmower", seed=3, spec={"episode_s": 60.0})


def test_recording_shapes(rec):
    T = len(rec.t)
    assert T > 30 and np.all(np.diff(rec.t) >= 0)
    assert rec.pos.shape == (T, 4, 2) and rec.pmine.shape == (T, 300, 80) and rec.pmine.dtype == np.float16
    assert len(rec.dec) == len(rec.dec_blocks) >= 4
    assert rec.meta["score"]["score"] >= 0 and rec.meta["fexl_url"].startswith("https://fexl.com/iarc/draw/")


def test_save_load_roundtrip(rec, tmp_path):
    p = rec.save(tmp_path / "r")
    assert p.suffix == ".npz" and p.with_suffix(".json").exists()
    r2 = Recording.load(p)
    assert r2.meta == rec.meta
    for k in ("t", "pos", "pmine", "seen", "dets", "dec_blocks", "mines", "block_owner"):
        assert np.array_equal(getattr(r2, k), getattr(rec, k), equal_nan=True)


def test_viewer_renders_offscreen(rec):
    import pygame
    from viz.replay import LAYERS, Viewer
    pygame.init()
    surf = pygame.Surface((1000, 760))
    v = Viewer(rec)
    for name in v.layers:
        v.layers[name] = True
    v.seek(rec.t[-1] / 2)
    v.draw(surf)
    v.seek(rec.t[-1])
    v.draw(surf)
    assert v.index() == len(rec.t) - 1
    assert np.asarray(pygame.surfarray.array3d(surf)).std() > 5  # something was drawn
    pygame.quit()
