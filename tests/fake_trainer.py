"""Stand-in for `python3 -m rl train-one`: fake_trainer.py --run-dir DIR SPEC. Behaviour comes from spec key fake_mode:
ok (write epochs.csv + done.json), fail (exit 3), hang (start a child sleeper, record its pid, sleep forever)."""
import json
import subprocess
import sys
import time
from pathlib import Path

run_dir = Path(sys.argv[sys.argv.index("--run-dir") + 1])
spec = json.loads((run_dir / "spec_in.json").read_text())
mode = spec["fake_mode"]
if mode == "fail":
    print("boom", flush=True)
    sys.exit(3)
if mode == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    (run_dir / "child.pid").write_text(str(child.pid))
    time.sleep(120)
time.sleep(spec["fake_sleep"])
lines = ["epoch,time,steps,episodes,score"] + [f"{i},{i * 0.5},{i * 100},4,{i / 10}" for i in range(1, 21)]
(run_dir / "epochs.csv").write_text("\n".join(lines) + "\n")
(run_dir / "done.json").write_text(json.dumps({"epochs": 20, "wall_s": 1.5}))
print("trained", flush=True)
