# Ideas to test later

## Partial-block scanning (sub-block targets)
Today an allocation is K whole 10 ft blocks and the drone sweeps each one through its center. Instead,
let the allocator output a point inside a block (a corner, an edge midpoint, or a finer sub-grid
position) so it can deliberately scan only part of a block.

Why: if an earlier pass already saw a mine in one corner of a block, that block is ruled out for the
safe path anyway. The policy could act greedily on that knowledge and send the next drone along the
sides of the neighboring blocks, skipping the rest of the known-mine block and spending that time
elsewhere.

What it needs:
- Action space: per block, a choice of sub-target (e.g. center + 4 corners + 4 edge midpoints = 9
  options -> 240 x 9 masked choices), or a finer allocation grid.
- Drone side: in-block pass geometry from the chosen sub-target (sim/routing.py pass_points), and the
  ordering cost must account for it.
- Masking: still only blocks held by other drones.

## Other deferred items
- Intermediate (shaped) rewards; currently terminal rule score only.
- Ending the run early (learned "land now" action, A < 7).
- Variable K (a stop choice in the pick sequence).
- Drone-drone avoidance.
- Imitation from the greedy baseline as a warm start.
- Allocator chooses altitude per assignment (survey high, confirm low).
- Favor overlapping passes at block seams so the Mapper gets more drift anchors.
