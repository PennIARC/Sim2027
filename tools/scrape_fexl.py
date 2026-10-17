"""Polite scraper for fexl.com/iarc/draw mine layouts. Raw HTML + parsed layouts cached in data/fexl_cache/.

Usage: python3 tools/scrape_fexl.py [--nums 60 135 250] [--n 50] [--check]
Seeds are 0.05 + i*0.0471 (same list for every num_danger so nesting can be tested).
Downloaded HTML is untrusted: parsed with regex only.
"""
import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

CACHE = Path(__file__).resolve().parents[1] / "data" / "fexl_cache"
URL = "https://fexl.com/iarc/draw/?num_danger={num}&seed={seed}&scale=5&scan_time=7&overweight=0&path_text="
UA = "Sim2027-research-scraper/1.0 (IARC Mission 10 sim; personal use; 1 req/s)"
MIN_GAP_S = 1.0
RECT = re.compile(r'<rect x=(\d+) y=(\d+) width=10 height=10 fill="red"')
_last = [0.0]
requests_made = [0]


def seeds(n):
    return [str(round(0.05 + i * 0.0471, 4)) for i in range(n)]


def parse(html):
    return sorted(((int(x) - 30) // 10, (1490 - int(y)) // 10) for x, y in RECT.findall(html))


def fetch(num, seed, force=False):
    """Return list of (col, row) mines. Uses cache; at most one network request per uncached (num, seed)."""
    f = CACHE / f"html_{num}_{seed}.html"
    if f.exists() and not force:
        return parse(f.read_text())
    wait = MIN_GAP_S - (time.time() - _last[0])
    if wait > 0:
        time.sleep(wait)
    req = urllib.request.Request(URL.format(num=num, seed=seed), headers={"User-Agent": UA})
    try:
        html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8", "replace")
    except Exception as e:  # back off and stop rather than hammer
        print(f"fexl error for num={num} seed={seed}: {e}; stopping", file=sys.stderr)
        sys.exit(2)
    finally:
        _last[0] = time.time()
        requests_made[0] += 1
    f.write_text(html)
    return parse(html)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nums", type=int, nargs="+", default=[60, 135, 250])
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--import-old", default="/tmp/fexl_layouts/m135.json",
                    help="reuse earlier probe (rebuilds minimal html cache) to save requests")
    ap.add_argument("--check", action="store_true", help="refetch one seed to test determinism")
    a = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)
    old = Path(a.import_old)
    if old.exists():
        for s, mines in json.loads(old.read_text()).items():
            f = CACHE / f"html_135_{s}.html"
            if not f.exists():  # synthesize minimal html in the same rect format
                f.write_text("".join(f'<rect x={30 + 10 * c} y={1490 - 10 * r} width=10 height=10 fill="red" />\n'
                                     for c, r in mines))
    out = {}
    for num in a.nums:
        out[str(num)] = {s: fetch(num, s) for s in seeds(a.n)}
        print(num, "done; requests so far", requests_made[0], flush=True)
    lf = CACHE / "layouts.json"
    merged = json.loads(lf.read_text()) if lf.exists() else {}
    merged.update(out)  # keep previously scraped counts
    lf.write_text(json.dumps(merged))
    if a.check:
        s = seeds(1)[0]
        again = fetch(135, s, force=True)
        print("deterministic:", again == merged["135"][s], "requests", requests_made[0])


if __name__ == "__main__":
    main()
