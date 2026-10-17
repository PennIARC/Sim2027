"""Pack data/fexl_cache/layouts.json into the committed eval set data/fexl_eval/fexl_layouts.npz."""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
d = json.loads((ROOT / "data" / "fexl_cache" / "layouts.json").read_text())
ids, grids, nums, seeds = [], [], [], []
for num, L in d.items():
    if int(num) not in (100, 135, 170):
        continue  # 60/250 stay in the gitignored cache only
    for seed, mines in L.items():
        g = np.zeros((150, 40), bool)
        for c, r in mines:
            g[r, c] = True
        ids.append(f"n{num}_s{seed}"); grids.append(g); nums.append(int(num)); seeds.append(float(seed))
np.savez_compressed(ROOT / "data" / "fexl_eval" / "fexl_layouts.npz", ids=np.array(ids), layouts=np.array(grids),
                    num_danger=np.array(nums), seed=np.array(seeds))
print(len(ids), "layouts")
