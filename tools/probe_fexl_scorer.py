"""Fetch + parse the fexl.com/iarc/draw scorer page (untrusted HTML: parsed with regex only, never executed).

Cached under data/fexl_cache/scorer/. Usage: python3 tools/probe_fexl_scorer.py  (see main for demo)
"""
import hashlib
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "data" / "fexl_cache" / "scorer"
BASE = "https://fexl.com/iarc/draw/"
_last = [0.0]


def url_for(path_text, num_danger, seed, A=7, N=0):
    q = urllib.parse.urlencode({"num_danger": num_danger, "seed": seed, "scale": 5, "scan_time": A,
                                "overweight": N, "path_text": path_text}, quote_via=urllib.parse.quote)
    return BASE + "?" + q


def fetch(path_text, num_danger=30, seed=1, A=7, N=0):
    url = url_for(path_text, num_danger, seed, A, N)
    f = CACHE / (hashlib.sha1(url.encode()).hexdigest()[:16] + ".html")
    if f.exists():
        return f.read_text()
    CACHE.mkdir(parents=True, exist_ok=True)
    wait = 1.2 - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    _last[0] = time.time()
    h = urllib.request.urlopen(url, timeout=30).read().decode("utf-8", "replace")
    f.write_text(h)
    return h


def parse(h):
    """-> dict with rects by color as sets of (col,row), plus reported numbers/messages."""
    out = {"colors": {}}
    for x, y, c in re.findall(r'<rect x=(\d+) y=(\d+) width=10 height=10 fill="(\w+)"', h):
        out["colors"].setdefault(c, set()).add(((int(x) - 30) // 10, (1490 - int(y)) // 10))
    def num(label):
        m = re.search(re.escape(label) + r"[^<]*</td>\s*<td align=right>\s*(-?[\d.]+)", h)
        return float(m.group(1)) if m else None
    out["B"] = num("B: Number of missed mines")
    out["L"] = num("L: Length of path")
    out["W"] = num("W: Width of path")
    out["score"] = num("Assessed score")
    m = re.search(r"Current location x = (-?\d+) y = (-?\d+)", h)
    out["loc"] = (int(m.group(1)), int(m.group(2))) if m else None
    # human-readable messages (results section)
    res = h.split("Results:", 1)[-1].split("A: Minutes", 1)[0]
    out["msgs"] = [re.sub(r"\s+", " ", t).strip() for t in re.findall(r">([^<>]*[A-Za-z][^<>]*)<", res) if t.strip()]
    return out
