"""Pygame replay viewer for a recorded episode (see viz/recorder.py).

    python3 -m viz.replay --run RUN_REF [--seed S | --layout n135_s0.05] [--progress 1.0] [--save viz_out/x.npz]
    python3 -m viz.replay --baseline greedy --layout n135_s0.05
    python3 -m viz.replay --load viz_out/x.npz

Field is drawn vertically (start edge y=0 at the bottom, 40x150 squares), like fexl. Keys: see HELP below (H toggles it in-app).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import numpy as np
import pygame

from viz.recorder import Recording, record_episode

FIELD_W, FIELD_L = 80.0, 300.0
BG = (24, 26, 30)
TEXT = (230, 230, 230)
DIM = (150, 155, 160)
# fexl colors
PATH_BLUE, BUFFER_GREEN = (30, 60, 255), (144, 238, 144)
MINE_RED, ONPATH_ORANGE, MISSED_YELLOW = (210, 20, 20), (255, 140, 0), (255, 220, 0)
DRONE_COLORS = [(230, 40, 40), (40, 120, 255), (30, 170, 60), (170, 60, 220), (240, 160, 0), (0, 180, 190), (200, 90, 140), (120, 120, 120)]

# key -> (layer attr, label, default)
LAYERS = {
    pygame.K_p: ("pmine", "P p_mine heatmap", True),
    pygame.K_s: ("seen", "S seen heatmap", False),
    pygame.K_m: ("mines", "M true mines (hidden)", False),
    pygame.K_a: ("alloc", "A allocations", True),
    pygame.K_d: ("drones", "D drones + footprints", True),
    pygame.K_r: ("paths", "R remaining paths", True),
    pygame.K_f: ("final", "F final path + buffer", False),
    pygame.K_e: ("dets", "E detections", False),
}
HELP = ["space play/pause   left/right step (shift x10)", "home/end jump   up/down speed   click slider to seek",
        "H help   C copy fexl link   Esc/Q quit"]


def _lerp(a, b, f):
    return a + (b - a) * f


class _Font:
    """Default-font wrapper (no font files). Uses pygame._freetype directly with its bundled default font, because pygame.font and
    pygame.freetype fail to import in some wheels (py3.14: circular import via sysfont). Falls back to pygame.font.
    API: render(text, antialias, color) -> Surface, get_height()."""

    def __init__(self, size):
        try:
            from pygame import _freetype
            _freetype.init()
            self._f, self._free = _freetype.Font(None, size * 0.72), True
        except Exception:
            self._f, self._free = pygame.font.Font(None, size), False

    def render(self, text, aa, color):
        return self._f.render(text, fgcolor=color)[0] if self._free else self._f.render(text, aa, color)

    def get_height(self):
        return int(self._f.get_sized_height()) + 2 if self._free else self._f.get_height()


class Viewer:
    def __init__(self, rec: Recording):
        self.rec = rec
        self.layers = {name: default for name, _, default in LAYERS.values()}
        self.cur_t = 0.0
        self.playing = False
        self.speed = 10.0  # sim seconds per real second
        self.show_help = True
        self.t_end = float(rec.t[-1])
        self.nd = rec.pos.shape[1]
        self.cl_mask, self.buf_mask = rec.masks()
        self._base_key = None
        self._base_surf = None
        self.fonts: dict = {}
        self.slider = pygame.Rect(0, 0, 1, 1)
        self.url_rect = pygame.Rect(0, 0, 0, 0)
        self.notice = ""

    # ---- helpers -----------------------------------------------------------------------------------------------
    def font(self, size):
        if size not in self.fonts:
            self.fonts[size] = _Font(size)
        return self.fonts[size]

    def index(self) -> int:
        return int(np.clip(np.searchsorted(self.rec.t, self.cur_t + 1e-6, side="right") - 1, 0, len(self.rec.t) - 1))

    def seek(self, t):
        self.cur_t = float(np.clip(t, 0.0, self.t_end))

    def step(self, n):
        i = int(np.clip(self.index() + n, 0, len(self.rec.t) - 1))
        self.cur_t = float(self.rec.t[i])

    def advance(self, dt_real):
        if self.playing:
            self.cur_t += dt_real * self.speed
            if self.cur_t >= self.t_end:
                self.cur_t, self.playing = self.t_end, False

    def interp(self, arr, i):
        """Linear interpolation of a (T, ...) array at cur_t (between snapshots i and i+1)."""
        if i + 1 >= len(self.rec.t):
            return arr[i]
        t0, t1 = self.rec.t[i], self.rec.t[i + 1]
        f = float(np.clip((self.cur_t - t0) / max(t1 - t0, 1e-9), 0, 1))
        return _lerp(arr[i], arr[i + 1], f)

    # ---- events ------------------------------------------------------------------------------------------------
    def on_event(self, ev) -> bool:
        """Returns False when the user asked to quit."""
        if ev.type == pygame.QUIT:
            return False
        if ev.type == pygame.KEYDOWN:
            if ev.key in (pygame.K_ESCAPE, pygame.K_q):
                return False
            if ev.key == pygame.K_SPACE:
                if self.cur_t >= self.t_end:
                    self.cur_t = 0.0
                self.playing = not self.playing
            elif ev.key in (pygame.K_LEFT, pygame.K_RIGHT):
                self.playing = False
                n = 10 if ev.mod & pygame.KMOD_SHIFT else 1
                self.step(-n if ev.key == pygame.K_LEFT else n)
            elif ev.key == pygame.K_HOME:
                self.seek(0.0)
            elif ev.key == pygame.K_END:
                self.seek(self.t_end)
            elif ev.key == pygame.K_UP:
                self.speed = min(self.speed * 2, 320.0)
            elif ev.key == pygame.K_DOWN:
                self.speed = max(self.speed / 2, 0.5)
            elif ev.key == pygame.K_h:
                self.show_help = not self.show_help
            elif ev.key == pygame.K_c:
                self.copy_url()
            elif ev.key in LAYERS:
                name = LAYERS[ev.key][0]
                self.layers[name] = not self.layers[name]
        elif ev.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            down = ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1 or ev.type == pygame.MOUSEMOTION and ev.buttons[0]
            if down and self.slider.inflate(0, 16).collidepoint(ev.pos):
                self.seek((ev.pos[0] - self.slider.x) / max(self.slider.w, 1) * self.t_end)
        return True

    def copy_url(self):
        try:
            pygame.scrap.init()
            pygame.scrap.put_text(self.rec.meta["fexl_url"])
            self.notice = "fexl link copied"
        except Exception:
            self.notice = "clipboard unavailable; link printed to terminal"
            print(self.rec.meta["fexl_url"])

    # ---- drawing -----------------------------------------------------------------------------------------------
    def _base_array(self, i) -> np.ndarray:
        """(300, 80, 3) uint8 [row=y, col=x] heatmaps + final path + mines at 1 ft resolution."""
        rec, L = self.rec, self.layers
        img = np.full((300, 80, 3), 255.0)
        if L["seen"]:
            img = _lerp(img, np.array([120.0, 190.0, 235.0]), rec.seen[i].astype(np.float32)[..., None])
        if L["pmine"]:
            a = np.minimum(rec.pmine[i].astype(np.float32) * 8.0, 1.0)[..., None]
            img = _lerp(img, np.array([200.0, 0.0, 0.0]), 0.9 * a)
        up = lambda m: np.repeat(np.repeat(m, 2, 0), 2, 1)  # squares -> 1 ft cells
        mines = up(rec.mines)
        if L["final"]:
            img[up(self.buf_mask)] = BUFFER_GREEN
            img[up(self.cl_mask)] = PATH_BLUE
        if L["mines"]:
            img[mines] = MINE_RED
        if L["final"]:
            img[mines & up(self.buf_mask)] = MISSED_YELLOW
            img[mines & up(self.cl_mask)] = ONPATH_ORANGE
        return img.clip(0, 255).astype(np.uint8)

    def base_surface(self, i, size):
        key = (i, size, self.layers["pmine"], self.layers["seen"], self.layers["mines"], self.layers["final"])
        if key != self._base_key:
            arr = self._base_array(i)[::-1]                  # row 0 (start edge) at the bottom
            surf = pygame.surfarray.make_surface(np.ascontiguousarray(arr.transpose(1, 0, 2)))
            self._base_surf = pygame.transform.scale(surf, size)
            self._base_key = key
        return self._base_surf

    def draw(self, surf: pygame.Surface):
        rec, L = self.rec, self.layers
        W, H = surf.get_size()
        surf.fill(BG)
        bar_h = 62
        s = max(min((H - bar_h - 20) / FIELD_L, (W * 0.45) / FIELD_W), 0.5)
        fw, fh = int(FIELD_W * s), int(FIELD_L * s)
        fx, fy = 10, 10
        i = self.index()
        surf.blit(self.base_surface(i, (fw, fh)), (fx, fy))
        w2s = lambda x, y: (float(fx + x * s), float(fy + (FIELD_L - y) * s))
        if L["alloc"]:
            ov = pygame.Surface((fw, fh), pygame.SRCALPHA)
            bw = 10 * s
            for b, o in enumerate(rec.block_owner[i]):
                if o >= 0:
                    r, c = divmod(b, 8)
                    pygame.draw.rect(ov, (*DRONE_COLORS[o % 8], 85), (c * bw, (FIELD_L / 10 - 1 - r) * bw, bw, bw))
            surf.blit(ov, (fx, fy))
        pygame.draw.rect(surf, DIM, (fx, fy, fw, fh), 1)
        if L["dets"]:
            d = rec.dets
            for t, dr, x, y in d[d[:, 0] <= self.cur_t]:
                pygame.draw.circle(surf, DRONE_COLORS[int(dr) % 8], w2s(x, y), 1)
        if L["paths"]:
            for k in range(self.nd):
                pts = rec.rem[i, k]
                pts = pts[~np.isnan(pts[:, 0])]
                if len(pts) > 1:
                    pygame.draw.lines(surf, DRONE_COLORS[k % 8], False, [w2s(*p) for p in pts], 1)
        if L["drones"]:
            across, along = rec.meta["footprint_ft"]
            pos, bel, head = self.interp(rec.pos, i), self.interp(rec.bel, i), rec.head[i]
            for k in range(self.nd):
                col = DRONE_COLORS[k % 8]
                h = head[k]
                f, n = np.array([np.cos(h), np.sin(h)]), np.array([-np.sin(h), np.cos(h)])
                corners = [pos[k] + sa * along / 2 * f + sb * across / 2 * n for sa, sb in ((1, 1), (1, -1), (-1, -1), (-1, 1))]
                pygame.draw.polygon(surf, col, [w2s(*c) for c in corners], 1)
                pygame.draw.line(surf, col, w2s(*pos[k]), w2s(*bel[k]), 1)
                pygame.draw.circle(surf, col, w2s(*bel[k]), max(int(1.5 * s), 3), 1)   # believed: hollow
                pygame.draw.circle(surf, col, w2s(*pos[k]), max(int(1.5 * s), 3))      # true: filled
                lab = self.font(18).render(str(k), True, TEXT)
                surf.blit(lab, (w2s(*pos[k])[0] + 5, w2s(*pos[k])[1] - 14))
        self.draw_hud(surf, i, fx + fw + 20, 10, W - (fx + fw + 30))
        self.draw_slider(surf, H - bar_h + 10, W)

    def draw_slider(self, surf, y, W):
        rec = self.rec
        self.slider = pygame.Rect(20, y + 12, W - 40, 6)
        pygame.draw.rect(surf, (70, 74, 80), self.slider, border_radius=3)
        for t in rec.dec[:, 0]:  # decision ticks
            x = float(self.slider.x + t / max(self.t_end, 1e-9) * self.slider.w)
            pygame.draw.line(surf, (110, 114, 120), (x, y + 6), (x, y + 24))
        x = float(self.slider.x + self.cur_t / max(self.t_end, 1e-9) * self.slider.w)
        pygame.draw.rect(surf, (80, 160, 255), (self.slider.x, self.slider.y, x - self.slider.x, 6), border_radius=3)
        pygame.draw.circle(surf, TEXT, (x, self.slider.centery), 8)
        state = "playing" if self.playing else "paused"
        surf.blit(self.font(20).render(f"t = {self.cur_t:6.1f} / {self.t_end:.0f} s   {state}   {self.speed:g}x", True, TEXT), (20, y + 28))

    def draw_hud(self, surf, i, x, y, w):
        rec, m = self.rec, self.rec.meta
        f22, f18 = self.font(24), self.font(20)
        lines = []
        add = lambda txt, col=TEXT, f=f18: lines.append((txt, col, f))
        add(f"{m['allocator']}", TEXT, f22)
        add(f"seed {m['seed']}  layout {m.get('layout') or 'generated'}  drift progress {m['progress']:g}  mines {m['n_mines']}", DIM)
        n_dec = int((rec.dec[:, 0] <= self.cur_t + 1e-6).sum())
        add(f"t {self.cur_t:.1f} s   snapshot {i + 1}/{len(rec.t)}   decisions {n_dec}/{len(rec.dec)}")
        add("")
        for k in range(self.nd):
            owned = np.flatnonzero(rec.block_owner[i] == k)
            moved = rec.dec[:n_dec, 1] == k
            last = rec.dec_blocks[:n_dec][moved][-1].tolist() if moved.any() else "-"
            idle = bool(rec.idle[i, k])
            add(f"drone {k}  {'IDLE' if idle else 'flying'}  holds {len(owned)} blk  last {last}", DRONE_COLORS[k % 8])
        add("")
        sc = m["score"]
        fin = self.cur_t >= self.t_end - 1e-6
        add("final score" + ("" if fin else " (at episode end)"), TEXT if fin else DIM, f22)
        add(f"score {sc['score']:.2f}   B {sc['B']:.0f}   L {sc['L']:.0f} ft   W {sc['W']:.0f} ft   G {sc['G']}", TEXT if fin else DIM)
        if sc["on_path_mines"] or not sc["valid"]:
            add(f"on-path mines {sc['on_path_mines']:.0f}  valid {sc['valid']}", MINE_RED)
        add("")
        add("layers (key):", DIM)
        for key, (name, label, _) in LAYERS.items():
            add(("[x] " if self.layers[name] else "[ ] ") + label, TEXT if self.layers[name] else DIM)
        if self.layers["final"] or fin:
            add("path blue, buffer green; on-path mine orange, buffer mine yellow, others red (M)", DIM)
        add("")
        if self.show_help:
            for h in HELP:
                add(h, DIM)
            add("")
        add("fexl link:", DIM)
        cw = max(int(w / 7.5), 20)
        for ln in textwrap.wrap(m["fexl_url"], cw, break_on_hyphens=False, break_long_words=True)[:14]:
            add(ln, (140, 180, 255), self.font(16))
        if self.notice:
            add(self.notice, (255, 220, 120))
        for txt, col, f in lines:
            if txt:
                surf.blit(f.render(txt, True, col), (x, y))
            y += f.get_height() + 2


# ---------------------------------------------------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="Replay viewer for Sim2027 episodes")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--run", help="trained run: id, unique name or id prefix")
    src.add_argument("--baseline", choices=["random", "lawnmower", "greedy"])
    src.add_argument("--load", help="a saved recording .npz")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--layout", default=None, help="fexl eval layout id, e.g. n135_s0.05 (default seed = evaluation seed of that layout)")
    ap.add_argument("--progress", type=float, default=1.0, help="GPS-drift curriculum point 0..1 (eval uses 1.0)")
    ap.add_argument("--interval", type=float, default=1.0, help="snapshot interval, sim s (min = env status interval, 1 s)")
    ap.add_argument("--save", default=None, help="save the recording to this .npz path (+ .json metadata)")
    ap.add_argument("--no-view", action="store_true", help="record/save only")
    a = ap.parse_args(argv)
    if a.load:
        rec = Recording.load(a.load)
    else:
        run_dir = None
        if a.run:
            from rl.registry import find_run
            run_dir = find_run(a.run)
        print("recording episode ...", flush=True)
        rec = record_episode(run_dir=run_dir, baseline=a.baseline, seed=a.seed, layout_id=a.layout, progress=a.progress,
                             interval_s=a.interval)
    if a.save:
        print("saved", rec.save(a.save))
    print(rec.meta["allocator"], rec.meta["score"], "\n" + rec.meta["fexl_url"])
    if a.no_view:
        return
    pygame.init()
    screen = pygame.display.set_mode((1100, 820), pygame.RESIZABLE)
    pygame.display.set_caption("Sim2027 replay")
    v, clock, running = Viewer(rec), pygame.time.Clock(), True
    while running:
        for ev in pygame.event.get():
            running = v.on_event(ev) and running
        v.advance(clock.tick(60) / 1000.0)
        v.draw(screen)
        pygame.display.flip()
    pygame.quit()


if __name__ == "__main__":
    main()
