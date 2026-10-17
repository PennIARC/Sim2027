# fexl eval set

150 mine layouts (50 seeds x 100/135/170 mines) scraped 2026-10-17 (rebuilt for the 100-170 mine range) from https://fexl.com/iarc/draw/ (the organizers' path scorer) by
`tools/scrape_fexl.py`, packed by `tools/build_eval_set.py` into `fexl_layouts.npz`:

- `layouts` bool (150, 150, 40) indexed [layout, row, col], row 0 = start edge (sim/geometry.py convention)
- `ids` e.g. `n135_s0.0971` (num_danger, seed); `num_danger` int and `seed` float arrays

Request params: `num_danger` in {100, 135, 170}, `seed` = round(0.05 + i*0.0471, 4) for i in 0..49 (same 50 seeds for
each count; layouts nest across num_danger for equal seed), `scale=5&scan_time=7&overweight=0&path_text=`.
Mines parsed from `<rect ... fill="red">` at x = 30 + 10*col, y = 1490 - 10*row. Rendered mine count always equaled
num_danger. Load with `sim.mines.load_eval_set("fexl")`.
