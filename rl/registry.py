"""The run registry: every run lives in runs/<YYYY-MM-DD_HHMM>_<name>/ with a meta.json that is the single source of truth.
meta.json holds id, name, title, blurb_text, group, parent, tags, kind, status, created/finished, batch, model (param count),
the full spec, the training summary and flat evaluation metrics. runs/index.json is a generated flat table of all runs
(`python3 -m rl index`); runs/favorites.json holds pinned run ids.

The runs directory is never a module global: pass `root=` or set the SIM2027_RUNS environment variable (see runs_root)."""
import json
import math
import os
import re
import tempfile
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
META = "meta.json"
STATUSES = ("running", "ok", "train_failed", "eval_failed", "timeout")
REQUIRED_SPEC_FIELDS = ("name", "blurb", "group", "tags")


def runs_root(root=None) -> Path:
    """Explicit root > $SIM2027_RUNS > <repo>/runs."""
    return Path(root or os.environ.get("SIM2027_RUNS") or REPO / "runs")


def slug(text):
    return re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")


def run_id(name, when):
    return f"{when.strftime('%Y-%m-%d_%H%M')}_{slug(name)}"


def new_run_dir(name, when=None, root=None) -> Path:
    """Creates and returns runs/<id>/ (a numeric suffix -2, -3, ... is added if that id already exists)."""
    root = runs_root(root)
    base = run_id(name, when or datetime.now())
    path, n = root / base, 1
    while True:
        try:
            path.mkdir(parents=True)
            return path
        except FileExistsError:
            n += 1
            path = root / f"{base}-{n}"


def check_run_spec(spec):
    """Raises ValueError unless a spec has a name, a two-line blurb ('short name' line, then a description), a group and non-empty tags."""
    missing = [k for k in REQUIRED_SPEC_FIELDS if not spec.get(k)]
    if missing:
        raise ValueError(f"run {spec.get('name', '?')!r} is missing {missing}: every run needs a name, a blurb "
                         "('short name\\nwhat it tests, what changed from the parent'), a group and tags")
    if not isinstance(spec["tags"], (list, tuple)) or not all(isinstance(t, str) for t in spec["tags"]):
        raise ValueError(f"run {spec['name']!r}: tags must be a list of strings")
    if len([l for l in str(spec["blurb"]).strip().splitlines() if l.strip()]) < 2:
        raise ValueError(f"run {spec['name']!r}: the blurb needs two lines, a short name and then the description")


def split_blurb(blurb):
    """('short name', 'description') from the two-line blurb text."""
    lines = [l.strip() for l in str(blurb or "").strip().splitlines() if l.strip()]
    return (lines[0] if lines else ""), " ".join(lines[1:])


def _json_default(obj):
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"cannot serialise {type(obj)}")


def dump_json(path, data, indent=1):
    """Atomic write (temp file + rename) so parallel readers never see half a file."""
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        f.write(json.dumps(data, indent=indent, default=_json_default) + "\n")
    os.replace(tmp, path)


def read_meta(run_dir):
    return json.loads((Path(run_dir) / META).read_text())


def write_meta(run_dir, meta):
    dump_json(Path(run_dir) / META, meta)


def update_meta(run_dir, **changes):
    """Top-level keys are replaced, except dict values which are merged into an existing dict."""
    meta = read_meta(run_dir)
    for key, value in changes.items():
        if isinstance(value, dict) and isinstance(meta.get(key), dict):
            meta[key].update(value)
        else:
            meta[key] = value
    write_meta(run_dir, meta)
    return meta


def now_iso():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def short_name(path):
    """'rl.toy_envs:PickBestEnv' -> 'PickBestEnv'."""
    return str(path).rpartition(":")[2] if path else "?"


def auto_tags(meta):
    """Tags that are facts about the run: kind:, batch:, env:<class>, policy:<builder>."""
    spec = meta.get("spec", {})
    tags = [f"kind:{meta.get('kind') or spec.get('kind') or 'ppo'}"]
    if meta.get("batch"):
        tags.append(f"batch:{meta['batch']}")
    if spec.get("env"):
        tags.append(f"env:{short_name(spec['env'])}")
    if spec.get("policy"):
        tags.append(f"policy:{short_name(spec['policy'])}")
    return tags


def all_tags(meta):
    seen, out = set(), []
    for tag in list(meta.get("tags", [])) + auto_tags(meta):
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def iter_runs(root=None):
    """(run_dir, meta) for every run folder with a readable meta.json, oldest first."""
    root = runs_root(root)
    if not root.exists():
        return []
    found = []
    for d in sorted(root.iterdir()):
        if d.is_dir() and (d / META).exists():
            try:
                found.append((d, read_meta(d)))
            except (OSError, ValueError):
                continue
    return found


def find_run(ref, root=None) -> Path:
    """Run dir for an exact id, a unique name, or a unique id prefix. Raises KeyError on no match or ambiguity."""
    root = runs_root(root)
    if (root / ref / META).exists():
        return root / ref
    runs = iter_runs(root)
    for hits in ([d for d, m in runs if m.get("name") == ref], [d for d, _ in runs if d.name.startswith(ref)]):
        if len(hits) == 1:
            return hits[0]
        if len(hits) > 1:
            raise KeyError(f"{ref!r} is ambiguous: matches {[h.name for h in hits]}")
    raise KeyError(f"{ref!r} matches no run")


def index_row(run_dir, meta):
    """One flat row: identity, grouping, timing, model size, hyperparameters (hp.*), curve summary (curve.*) and every scalar metric."""
    spec = meta.get("spec", {})
    training = meta.get("training", {})
    curve = training.get("summary", {})
    row = {"id": Path(run_dir).name, "name": meta.get("name"), "title": meta.get("title"), "blurb": meta.get("blurb_text"),
           "group": meta.get("group"), "parent": meta.get("parent"), "kind": meta.get("kind"), "batch": meta.get("batch"),
           "status": meta.get("status"), "created": meta.get("created"), "finished": meta.get("finished"), "tags": all_tags(meta),
           "primary_stat": spec.get("primary_stat"), "primary": curve.get("final")}
    row.update({f"model.{k}": v for k, v in meta.get("model", {}).items()})
    row.update({f"train.{k}": v for k, v in training.items() if not isinstance(v, (dict, list))})
    row.update({f"curve.{k}": v for k, v in curve.items()})
    row.update({f"hp.{k}": spec[k] for k in sorted(spec) if k not in ("blurb", "tags")})
    row.update({k: v for k, v in meta.get("metrics", {}).items() if isinstance(v, (int, float, str, bool)) or v is None})
    return row


def build_index(root=None):
    """Writes runs/index.json (flat rows, newest first) and returns the rows."""
    root = runs_root(root)
    root.mkdir(parents=True, exist_ok=True)
    rows = [index_row(d, m) for d, m in iter_runs(root)]
    rows.sort(key=lambda r: r.get("created") or "", reverse=True)
    dump_json(root / "index.json", rows)
    return rows


def load_index(root=None):
    path = runs_root(root) / "index.json"
    return json.loads(path.read_text()) if path.exists() else build_index(root)


def favorites(root=None):
    path = runs_root(root) / "favorites.json"
    return json.loads(path.read_text()) if path.exists() else []


def toggle_favorite(run, root=None):
    """Adds the run id to favorites.json, or removes it if already there. Returns the new list."""
    root = runs_root(root)
    favs = favorites(root)
    favs = [f for f in favs if f != run] if run in favs else favs + [run]
    root.mkdir(parents=True, exist_ok=True)
    dump_json(root / "favorites.json", favs)
    return favs


def safe_float(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None
