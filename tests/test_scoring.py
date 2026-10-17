import hashlib
import json
from pathlib import Path as FsPath

import numpy as np
import pytest

from sim.scoring import Path, buffer_mask, centerline_mask, fexl_url, score

FIX = json.loads((FsPath(__file__).parent / "fixtures" / "fexl" / "scorer_cases.json").read_text())


def grid(*mines_rc):
    m = np.zeros((150, 40), bool)
    for r, c in mines_rc:
        m[r, c] = True
    return m


def test_text_roundtrip_and_parse():
    p = Path(3, 2, [("U", 10), ("R", 2), ("U", 140)])
    assert p.to_text() == "S,3,2\nU,10\nR,2\nU,140\n"
    assert Path.from_text(p.to_text()) == p
    for bad in ["", "S, 3,2\nU,1", "S,3,2\nu,1", "S,3,2\nU, 1", "S,3\nU,1"]:
        with pytest.raises(ValueError):
            Path.from_text(bad)


def test_straight_path_basic_numbers():
    p = Path(5, 0, [("U", 150)])
    r = score(p, grid())
    assert r["valid"] and r["L"] == 300 and r["W"] == 2 and r["B"] == 0  # G=0 -> W=2, L = 2*moves
    assert r["score"] == pytest.approx(150000 * 2 / (1 * 300 * 50))
    assert len(p.squares()) == 150 and p.squares()[0] == (0, 5)


def test_width_and_length_scaling():
    r = score(Path(20, 3, [("U", 150)]), grid())
    assert r["W"] == 14 and r["L"] == 300
    r = score(Path(20, 3, [("U", 10), ("R", 2), ("U", 140)]), grid())
    assert r["L"] == 304  # 152 moves


def test_buffer_is_manhattan_diamond():
    sq = [(0, 10), (1, 10), (1, 11), (2, 11)]
    b = buffer_mask(sq, 2)
    cm = centerline_mask(sq)
    assert not (b & cm).any()
    assert b[1, 13] and b[3, 12] and b[0, 8]  # distance exactly 2
    assert not b[3, 13]  # (3,13): dist to (2,11) = 3, to (1,11) = 4
    # brute force check
    rr, cc = np.mgrid[0:150, 0:40]
    d = np.min([abs(rr - r) + abs(cc - c) for r, c in sq], axis=0)
    assert (b == ((d <= 2) & ~cm)).all()


def test_mine_in_buffer_counts_B_and_centerline_mine_zeroes():
    p = Path(10, 2, [("U", 150)])
    r = score(p, grid((50, 12)))  # inside buffer
    assert r["B"] == 1 and r["on_path_mines"] == 0
    assert r["score"] == pytest.approx(150000 * 10 / (2 * 300 * 50))
    assert score(p, grid((50, 13)))["B"] == 0
    r = score(p, grid((50, 10)))
    assert r["score"] == 0 and r["on_path_mines"] == 1
    # diagonal corner of the diamond is outside for Chebyshev-type squares
    assert score(p, grid((50, 11), (51, 12)))["B"] == 2
    assert score(Path(10, 1, [("U", 150)]), grid((50, 12)))["B"] == 0


def test_validity_rules():
    m = grid()
    assert not score(Path(3, 0, [("U", 149)]), m)["valid"]  # stops short of far row
    assert not score(Path(3, 0, [("U", 151)]), m)["valid"]  # out of bounds
    assert not score(Path(40, 0, [("U", 150)]), m)["valid"]
    assert not score(Path(3, 0, [("R", 1), ("U", 150)]), m)["valid"]  # first move off the start line
    assert not score(Path(3, 0, [("U", 150), ("L", 1), ("D", 1)]), m)["valid"]  # ends off the far row
    assert not score(Path(3, 0, [("U", 10), ("L", 1), ("D", 3)]), m)["valid"]
    assert not score(Path(3, 0, [("U", 10), ("R", 1), ("D", 2), ("L", 1), ("U", 1), ("U", 200)]), m)["valid"]  # revisit
    ok = score(Path(3, 0, [("U", 150), ("L", 2)]), m)  # may keep moving along the far row
    assert ok["valid"] and ok["L"] == 304
    assert score(Path(3, 0, [("U", 149)]), m)["score"] == 0


def test_fexl_url():
    u = fexl_url(Path(3, 2, [("U", 150)]), 30, 1)
    assert u.startswith("https://fexl.com/iarc/draw/?num_danger=30&seed=1&scale=5&scan_time=7&overweight=0&path_text=")
    assert u.endswith("S%2C3%2C2%0AU%2C150%0A")


@pytest.mark.parametrize("case", FIX["cases"], ids=lambda c: c["path_text"].replace("\n", "|"))
def test_parity_with_fexl(case):
    mines = grid(*[(r, c) for c, r in FIX["scenarios"][case["scenario"]]])
    p = Path.from_text(case["path_text"])
    r = score(p, mines, case["A"], case["N"])
    assert r["L"] == case["fexl_L"] and r["W"] == case["fexl_W"]
    assert r["score"] == pytest.approx(case["fexl_score"], abs=6e-4)
    assert r["B"] + r["on_path_mines"] == case["fexl_B"]  # fexl's B includes centerline mines
    assert r["on_path_mines"] == case["n_crossed"]
    sq = p.squares()
    assert len(sq) == case["n_blue"]
    lg = sorted((int(c), int(r_)) for r_, c in np.argwhere(buffer_mask(sq, p.G) | centerline_mask(sq)))
    assert len(lg) == case["n_lightgreen"]
    assert hashlib.sha1(json.dumps(lg).encode()).hexdigest()[:12] == case["lightgreen_sha1"]
    if case["fexl_score"] > 0:
        assert r["valid"]
