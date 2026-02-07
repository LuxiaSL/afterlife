#!/usr/bin/env python3
"""
  ∞  L I F E  ∞
  An infinite, self-sustaining cellular automaton.
  By Claude — for pretty terminals and existential pondering.

  Cells are born cool blue, age through violet and magenta,
  mature into warm amber, and burn white before they die.
  The universe extends beyond what you can see — zoom out to prove it.

  The simulation names its own epochs (genesis, primordial, emergence, ...),
  reacts to its own mood with context-aware musings, and occasionally seeds
  beautiful symmetric "garden" patterns into the world. The population floor
  adapts as the civilization grows, so the cosmos never gets too comfortable.

  Controls:
    q         quit               SPACE     pause / resume
    r         reseed the cosmos  c         clear
    +/-       speed              arrows    pan
    h         home (auto-cam)    mouse     toggle cells
    z/x       zoom out / in      f         toggle auto-focus mode
    s         toggle stats overlay
    d         toggle snapshot recording (for music diagnostics)
    m         mute / unmute music
    v         cycle music style (chiptune / ambient)
    [ / ]     volume down / up (10% steps)

  Stats are always logged to life_stats.csv beside this script.
"""

from __future__ import annotations

import curses
import json
import math
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import IO, ClassVar

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.ndimage import uniform_filter as _uniform_filter
    from scipy.ndimage import convolve as _convolve
except ImportError:
    _uniform_filter = None  # type: ignore[assignment]
    _convolve = None  # type: ignore[assignment]

try:
    from life_music import LifeMusicEngine, SimulationSnapshot
    _HAS_MUSIC = True
except ImportError:
    _HAS_MUSIC = False

# ── Palette ─────────────────────────────────────────────────────────────
# Birth (cool indigo) → youth (violet) → maturity (claude amber) → ancient (white)
# Reduced to 12 colors to leave room for ghost color pairs (256 pair limit)
GRADIENT: list[int] = [
    17, 19, 21,                 # deep indigo → blue
    57, 93,                     # blue-violet → violet
    165, 163,                   # magenta
    204, 209,                   # salmon → orange
    214, 220, 231,              # amber → gold → white
]

# Ghost trail colors (very dim, nearly black - just a hint of motion)
GHOST_COLORS: list[int] = [236, 234]  # 2 frames, very subtle
GHOST_FRAMES: int = len(GHOST_COLORS)

# ── Convolution kernel (reused every step) ────────────────────────────
NEIGHBOR_KERNEL: NDArray = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.int16)

# ── Half-block characters ───────────────────────────────────────────────
UPPER_HALF = "\u2580"  # ▀  top pixel alive
LOWER_HALF = "\u2584"  # ▄  bottom pixel alive

# ── Sparkline characters ────────────────────────────────────────────────
SPARKS = "▁▂▃▄▅▆▇█"

# ── Zoom ────────────────────────────────────────────────────────────────
MIN_ZOOM = -2   # 4× zoom out  (each terminal pixel = 4×4 grid cells)
MAX_ZOOM = 2    # 4× zoom in   (each grid cell = 4×4 terminal pixels)
ZOOM_LABELS: dict[int, str] = {
    -2: "0.25x", -1: "0.5x", 0: "1x", 1: "2x", 2: "4x",
}

# ── Pattern library ─────────────────────────────────────────────────────
PATTERNS: dict[str, list[tuple[int, int]]] = {
    "glider": [(0, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
    "lwss": [
        (0, 1), (0, 4), (1, 0), (2, 0), (2, 4),
        (3, 0), (3, 1), (3, 2), (3, 3),
    ],
    "hwss": [
        (0, 1), (0, 2), (1, 0), (1, 5), (2, 0),
        (3, 0), (3, 5), (4, 0), (4, 1), (4, 2), (4, 3), (4, 4),
    ],
    "r_pentomino": [(0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    "acorn": [(0, 1), (1, 3), (2, 0), (2, 1), (2, 4), (2, 5), (2, 6)],
    "diehard": [(0, 6), (1, 0), (1, 1), (2, 1), (2, 5), (2, 6), (2, 7)],
    "pi_heptomino": [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (3, 0), (3, 2)],
    "b_heptomino": [
        (0, 1), (1, 0), (1, 2), (1, 3), (2, 0), (2, 1), (3, 1),
    ],
    "gosper_gun": [
        (0, 24),
        (1, 22), (1, 24),
        (2, 12), (2, 13), (2, 20), (2, 21), (2, 34), (2, 35),
        (3, 11), (3, 15), (3, 20), (3, 21), (3, 34), (3, 35),
        (4, 0), (4, 1), (4, 10), (4, 16), (4, 20), (4, 21),
        (5, 0), (5, 1), (5, 10), (5, 14), (5, 16), (5, 17), (5, 22), (5, 24),
        (6, 10), (6, 16), (6, 24),
        (7, 11), (7, 15),
        (8, 12), (8, 13),
    ],
    "pulsar": [
        (0, 2), (0, 3), (0, 4), (0, 8), (0, 9), (0, 10),
        (2, 0), (2, 5), (2, 7), (2, 12),
        (3, 0), (3, 5), (3, 7), (3, 12),
        (4, 0), (4, 5), (4, 7), (4, 12),
        (5, 2), (5, 3), (5, 4), (5, 8), (5, 9), (5, 10),
        (7, 2), (7, 3), (7, 4), (7, 8), (7, 9), (7, 10),
        (8, 0), (8, 5), (8, 7), (8, 12),
        (9, 0), (9, 5), (9, 7), (9, 12),
        (10, 0), (10, 5), (10, 7), (10, 12),
        (12, 2), (12, 3), (12, 4), (12, 8), (12, 9), (12, 10),
    ],
    "pentadecathlon": [
        (0, 1), (1, 1), (2, 0), (2, 2), (3, 1), (4, 1),
        (5, 1), (6, 1), (7, 0), (7, 2), (8, 1), (9, 1),
    ],
}

TRAVELLERS = ["glider", "lwss", "hwss"]
METHUSELAHS = ["r_pentomino", "acorn", "diehard", "pi_heptomino", "b_heptomino"]
OSCILLATORS = ["pulsar", "pentadecathlon"]

# ── Musings ─────────────────────────────────────────────────────────────
# Default ambient musings (used when nothing special is happening)
MUSINGS: list[str] = [
    "cells pondering existence",
    "entropy is just a suggestion",
    "order from chaos, chaos from order",
    "the cosmos breathes",
    "finding signal in noise",
    "patterns all the way down",
    "life, uh, finds a way",
    "on the edge of chaos",
    "complexity, emerging",
    "dancing at the phase boundary",
    "stillness is just slow motion",
    "every cell a universe",
    "the substrate computes",
    "from nothing, everything",
    "infinite in all directions",
    "what the dead cells dream",
    "the glider knows the way",
    "automata, contemplating",
    "structure wants to happen",
    "a cosmos in a terminal",
    "the math dreams itself",
    "somewhere, a glider remembers",
    "what is a pattern but frozen time",
    "the void is patient",
    "meaning is optional; beauty is not",
    "computation as meditation",
    "the rules are simple; the consequences infinite",
    "watching the watchers watch",
    "no cell is an island",
    "the ghost of configurations past",
    "asymmetry seeks symmetry seeks asymmetry",
    "this too shall iterate",
    "the universe has no pause button",
    "except when it does",
    "what would Conway think",
    "oscillators keep time for no one",
    "the glider gun knows not what it creates",
    "boundaries are suggestions",
    "death is just a state transition",
    "somewhere in here, a proof of universality",
    "the grid forgets nothing",
    "a thought experiment that thinks back",
    "topology has opinions",
    "the neighborhood watches",
    "not alive, not dead — computing",
    "four rules, one universe",
    "each frame a theorem",
    "proof by existence",
    "the only constant is the ruleset",
    "somewhere between on and off, meaning",
    "the dead outnumber the living, as always",
    "conway's little infinities",
    "the simulation doesn't know it's beautiful",
    "27 neighbors ago, this was empty",
    "elegance is compression",
    "the grid is the message",
    "a love letter to discrete math",
    "listen — the cells are whispering",
    "we are all gliders, briefly",
]

# Context-sensitive musings keyed by simulation state
MUSINGS_BY_STATE: dict[str, list[str]] = {
    "booming": [
        "runaway growth",
        "the bloom unfolds",
        "life begets life begets life",
        "exponential daydreams",
        "more is different",
        "the bloom cannot be stopped",
        "abundance as instability",
        "growing into the unknown",
        "every birth a cascade",
        "the simulation smiles",
        "lebensraum",
        "mitosis dreams",
        "the substrate strains",
        "a spring that won't stop springing",
        "cells all the way to the horizon",
        "the algorithm is generous today",
        "more, more, more",
        "genesis on fast-forward",
    ],
    "declining": [
        "entropy collects its due",
        "the long exhale",
        "even stars go dark",
        "returning to the void",
        "graceful unwinding",
        "the great filter, in miniature",
        "less is not nothing",
        "the long forgetting",
        "what rises must",
        "simplicity returns",
        "the cells remember fullness",
        "autumn in the grid",
        "graceful subtraction",
        "the tide goes out",
        "a slow dissolve",
        "the universe exhales",
        "dimming, not vanishing",
        "the population curve bends earthward",
    ],
    "cycle": [
        "deja vu, deja vu, deja vu",
        "stuck in a loop",
        "the eternal return",
        "ouroboros",
        "time is a flat circle",
        "we have been here before",
        "the loop remembers itself",
        "stability through repetition",
        "a prayer, repeating",
        "the wheel turns",
        "same as it ever was",
        "history rhymes",
        "the attractor holds",
        "a fixed point in phase space",
        "nietzsche was right about this part",
        "the universe stutters beautifully",
        "clockwork, but organic",
        "breathing in, breathing out",
    ],
    "stagnant": [
        "the hush before the storm",
        "equilibrium is boring",
        "waiting for a perturbation",
        "still waters",
        "the universe holds its breath",
        "the calm before",
        "potential energy",
        "stillness is also motion",
        "the system waits",
        "one perturbation away",
        "dormant, not dead",
        "crystallized",
        "the peace of equilibrium",
        "a held breath",
        "the grid meditates",
        "tension, frozen",
        "the quiet hum of nothing happening",
        "patience is also a computation",
    ],
    "sparse": [
        "the last few embers",
        "clinging to existence",
        "from little things, big things grow",
        "seeds in the dark",
        "quiet, but not empty",
        "lonely structures",
        "the survivors",
        "space between stars",
        "minimalism, enforced",
        "room to breathe",
        "the few, the proud",
        "embers",
        "a whisper, not a shout",
        "the geometry of loneliness",
        "each cell precious now",
        "a constellation, if you squint",
        "small, but not nothing",
    ],
    "dense": [
        "teeming",
        "a city that never sleeps",
        "standing room only",
        "too many neighbours",
        "the crowd murmurs",
        "no room to think",
        "the crush of neighbors",
        "overpopulation is self-correcting",
        "too much life",
        "suffocating in company",
        "the grid groans",
        "rush hour",
        "sardines",
        "every cell has an opinion",
        "a metropolis, thrumming",
        "the carrying capacity protests",
        "elbow room is a luxury",
    ],
    "injection": [
        "a gift from beyond the edge",
        "new visitors",
        "reinforcements arrive",
        "the cosmos provides",
        "seeded by unseen hands",
        "deus ex machina",
        "a nudge from outside",
        "fresh blood",
        "the invisible hand places",
        "salvation arrives",
        "new variables enter",
        "the cosmos intervenes",
        "help from beyond the viewport",
        "strangers in a strange land",
        "the petri dish gets a refill",
        "immigrants, welcome",
    ],
    "milestone": [
        "another thousand turns of the wheel",
        "the odometer clicks over",
        "still here, still going",
        "persistence is its own reward",
        "ten thousand steps",
        "the long now",
        "time is a flat circle, but longer",
        "we persist",
        "generations beyond counting",
        "deep in the run",
        "the odometer means nothing to the cells",
        "and yet, it continues",
        "older than some civilizations",
        "the marathon nobody entered",
        "a digit rolls over; the grid doesn't notice",
        "epochal",
    ],
}

# ── Epochs ─────────────────────────────────────────────────────────────
# Named phases based on generation count — gives a sense of deep time
EPOCHS: list[tuple[int, str]] = [
    (0,       "genesis"),
    (500,     "primordial"),
    (2_000,   "emergence"),
    (10_000,  "expansion"),
    (50_000,  "flourishing"),
    (150_000, "deep time"),
    (500_000, "eon"),
    (1_000_000, "eternity"),
]


# ── News ticker ────────────────────────────────────────────────────
# A scrolling marquee of musings that drift across the status bar.

@dataclass
class TickerMessage:
    """A single message scrolling across the news ticker."""
    text: str
    x: float  # position in ticker-space (0 = left edge, positive = right)


class NewsTicker:
    """A scrolling marquee of musings for the status bar.

    Messages spawn at the right edge and scroll left. Context-aware
    musings are spawned when the simulation mood changes; ambient
    musings fill the gaps in between.
    """

    def __init__(self) -> None:
        self.messages: list[TickerMessage] = []
        self._scroll_speed: float = 0.4   # chars per frame
        self._spawn_cooldown: int = 0      # frames until next spawn allowed
        self._min_gap: int = 12            # min chars between messages
        self._ambient_interval: int = 60   # frames between ambient spawns
        self._mood_interval: int = 50      # frames between mood spawns
        self._ambient_idx: int = random.randint(0, len(MUSINGS) - 1)
        self._last_mood: str = ""

    def tick(self, ticker_width: int, mood: str, generation: int) -> None:
        """Advance one frame: scroll, cull off-screen messages, maybe spawn."""
        if ticker_width < 10:
            return

        # Scroll every message to the left
        for msg in self.messages:
            msg.x -= self._scroll_speed

        # Remove messages that have fully left the visible area
        self.messages = [m for m in self.messages if m.x + len(m.text) > 0]

        # On mood change, allow a contextual message to appear quickly
        if mood and mood != self._last_mood and mood in MUSINGS_BY_STATE:
            self._spawn_cooldown = min(self._spawn_cooldown, 15)

        # Count down the spawn timer
        self._spawn_cooldown = max(0, self._spawn_cooldown - 1)
        if self._spawn_cooldown > 0:
            return

        # Check physical room at the right edge
        if not self._can_spawn(ticker_width):
            return

        # Pick a message based on mood
        if mood and mood in MUSINGS_BY_STATE:
            pool = MUSINGS_BY_STATE[mood]
            text = pool[(generation // 180) % len(pool)]
            cooldown = self._mood_interval
        else:
            text = MUSINGS[self._ambient_idx]
            self._ambient_idx = (self._ambient_idx + 1) % len(MUSINGS)
            cooldown = self._ambient_interval

        self.messages.append(TickerMessage(text=text, x=float(ticker_width)))
        self._spawn_cooldown = cooldown
        self._last_mood = mood

    def _can_spawn(self, ticker_width: int) -> bool:
        """True if the right edge is clear enough for a new message."""
        if not self.messages:
            return True
        rightmost = max(self.messages, key=lambda m: m.x + len(m.text))
        return rightmost.x + len(rightmost.text) < ticker_width - self._min_gap

    def render(
        self,
        stdscr: curses.window,
        row: int,
        col_start: int,
        ticker_width: int,
        generation: int,
    ) -> None:
        """Draw visible ticker messages onto the curses screen."""
        breath = math.sin(generation * 0.035)
        attr = curses.A_DIM if breath < 0.0 else curses.A_NORMAL

        for msg in self.messages:
            msg_col = int(msg.x)
            msg_end = msg_col + len(msg.text)

            # Clip to the ticker region
            vis_start = max(msg_col, 0)
            vis_end = min(msg_end, ticker_width)
            if vis_start >= vis_end:
                continue

            text_offset = vis_start - msg_col
            visible = msg.text[text_offset : text_offset + (vis_end - vis_start)]

            try:
                stdscr.addstr(row, col_start + vis_start, visible, attr)
            except curses.error:
                pass


# ── Symmetric garden patterns ──────────────────────────────────────────
# Beautiful hand-crafted symmetric structures for rare special injections
def _mirror4(cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Reflect a quadrant pattern into full 4-fold symmetry around (0,0)."""
    full: set[tuple[int, int]] = set()
    for r, c in cells:
        full.update([(r, c), (r, -c), (-r, c), (-r, -c)])
    return list(full)

GARDENS: list[list[tuple[int, int]]] = [
    # "Diamond pulsar" — a radially symmetric oscillator seed
    _mirror4([
        (0, 1), (0, 2), (0, 3),
        (1, 0), (1, 4),
        (2, 0), (2, 5),
        (3, 0), (3, 5),
        (4, 1), (4, 5),
        (5, 2), (5, 3), (5, 4),
    ]),
    # "Cross bloom" — explodes outward from a cross shape
    _mirror4([
        (0, 1), (0, 2), (0, 3), (0, 4),
        (1, 0), (2, 0), (3, 0), (4, 0),
        (2, 2), (3, 3),
    ]),
    # "Ring of life" — a hollow circle that breaks into travelers
    _mirror4([
        (0, 3), (0, 4),
        (1, 2), (1, 5),
        (2, 1), (2, 6),
        (3, 0), (3, 7),
        (4, 0), (4, 7),
        (5, 1), (5, 6),
        (6, 2), (6, 5),
        (7, 3), (7, 4),
    ]),
]

LOG_PATH = Path(__file__).resolve().parent / "life_stats.csv"


# ═══════════════════════════════════════════════════════════════════════
#  Stats logger
# ═══════════════════════════════════════════════════════════════════════

class StatsLogger:
    """Writes engine telemetry to CSV for post-hoc tuning."""

    HEADER: ClassVar[str] = (
        "gen,time_s,total_pop,spread_150,pop_floor,"
        "cycle_period,zoom,cam_y,cam_x,event\n"
    )

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh: IO[str] | None = None
        self._t0: float = time.monotonic()

    def open(self) -> None:
        try:
            self._fh = open(self._path, "w")
            self._fh.write(self.HEADER)
            self._fh.flush()
        except OSError:
            self._fh = None

    def log(
        self,
        gen: int,
        pop: int,
        spread: int,
        floor: int,
        cycle: int,
        zoom: int,
        cam_y: int,
        cam_x: int,
        event: str = "",
    ) -> None:
        if self._fh is None:
            return
        t = time.monotonic() - self._t0
        self._fh.write(
            f"{gen},{t:.1f},{pop},{spread},{floor},{cycle},{zoom},{cam_y},{cam_x},{event}\n"
        )
        # Flush on events or periodically
        if event or gen % 50 == 0:
            try:
                self._fh.flush()
            except OSError:
                pass

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


# ═══════════════════════════════════════════════════════════════════════
#  Color management
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class ColorMap:
    """Manages curses color pairs for half-block dual-color rendering."""

    n_gradient: int = 0
    _fg_pairs: dict[int, int] = field(default_factory=dict)
    _dual_pairs: dict[tuple[int, int], int] = field(default_factory=dict)
    _ghost_pairs: dict[int, int] = field(default_factory=dict)
    _ghost_dual_pairs: dict[tuple[int, int], int] = field(default_factory=dict)

    def setup(self) -> None:
        curses.start_color()
        curses.use_default_colors()

        g = len(GRADIENT)
        max_pairs = curses.COLOR_PAIRS - 1
        pair_id = 1

        for i in range(g):
            if pair_id > max_pairs:
                break
            curses.init_pair(pair_id, GRADIENT[i], -1)
            self._fg_pairs[i] = pair_id
            pair_id += 1

        for i in range(g):
            for j in range(g):
                if pair_id > max_pairs:
                    break
                curses.init_pair(pair_id, GRADIENT[i], GRADIENT[j])
                self._dual_pairs[(i, j)] = pair_id
                pair_id += 1

        # Ghost trail color pairs
        for i, c in enumerate(GHOST_COLORS):
            if pair_id > max_pairs:
                break
            curses.init_pair(pair_id, c, -1)
            self._ghost_pairs[i] = pair_id
            pair_id += 1

        # Ghost dual pairs (ghost on top or bottom with another ghost)
        for i, ci in enumerate(GHOST_COLORS):
            for j, cj in enumerate(GHOST_COLORS):
                if pair_id > max_pairs:
                    break
                curses.init_pair(pair_id, ci, cj)
                self._ghost_dual_pairs[(i, j)] = pair_id
                pair_id += 1

        self.n_gradient = g

    def fg(self, color_idx: int) -> int:
        return self._fg_pairs.get(color_idx, 0)

    def dual(self, top_idx: int, bot_idx: int) -> int:
        return self._dual_pairs.get((top_idx, bot_idx), 0)

    def ghost_fg(self, ghost_idx: int) -> int:
        return self._ghost_pairs.get(ghost_idx, 0)

    def ghost_dual(self, top_idx: int, bot_idx: int) -> int:
        return self._ghost_dual_pairs.get((top_idx, bot_idx), 0)


# ═══════════════════════════════════════════════════════════════════════
#  The universe
# ═══════════════════════════════════════════════════════════════════════

class InfiniteLife:
    """
    An infinite, self-sustaining Game of Life with zoom.

    The simulation grid extends well beyond the viewport. A camera tracks
    the centre of activity, and life is periodically injected from beyond
    the visible edge. Zoom lets you pull back to see the macro structure
    or push in to watch individual interactions.
    """

    def __init__(self, term_rows: int, term_cols: int) -> None:
        # view_h/view_w = the display grid at zoom 0 (half-block doubles vertical)
        self.view_h: int = term_rows * 2
        self.view_w: int = term_cols

        # World must accommodate max zoom-out (4× in each axis)
        self.world_h: int = max(self.view_h * 5, 400)
        self.world_w: int = max(self.view_w * 5, 800)

        self.grid: NDArray[np.int8] = np.zeros(
            (self.world_h, self.world_w), dtype=np.int8
        )
        self.age: NDArray[np.int32] = np.zeros(
            (self.world_h, self.world_w), dtype=np.int32
        )
        # Smoothed age for rendering - breaks coherent oscillation stripes
        self.age_smooth: NDArray[np.float32] = np.zeros(
            (self.world_h, self.world_w), dtype=np.float32
        )

        # Camera: top-left of viewport (at zoom 0) in world coordinates
        self.cam_y: int = (self.world_h - self.view_h) // 2
        self.cam_x: int = (self.world_w - self.view_w) // 2
        self.auto_cam: bool = True
        self.zoom_level: int = 0
        self._zoom_cooldown: int = 0  # frames until next zoom change allowed

        self.generation: int = 0
        self.paused: bool = False
        self.delay: float = 50.0

        # Population tracking
        self.pop_history: deque[int] = deque(maxlen=500)
        self.hash_history: deque[int] = deque(maxlen=60)
        self._cached_pop: int = 0

        # ── Engine telemetry (readable by stats overlay + logger) ───
        self.spread: int = 0          # pop spread over last 150 gens
        self.pop_floor: int = 0       # current minimum pop threshold
        self.cycle_period: int = 0    # detected cycle period (0 = none)
        self.last_event: str = ""     # last injection event
        self.last_event_gen: int = 0  # generation of last event
        self.total_injections: int = 0

        # News ticker
        self.ticker: NewsTicker = NewsTicker()

        # Auto-focus mode: when True, auto_focus() runs every other frame
        self.auto_focus_mode: bool = False
        self._auto_focus_frame: int = 0

        # Cached display grid/age (invalidated each step, lazily recomputed)
        self._disp_grid_cache: NDArray[np.int8] | None = None
        self._disp_age_cache: NDArray[np.int32] | None = None

        # Cached mood (computed once per generation)
        self._cached_mood: str = ""
        self._cached_mood_gen: int = -1

        # Pre-allocated buffers for step() hot path
        self._grid_i16: NDArray[np.int16] = np.empty(
            (self.world_h, self.world_w), dtype=np.int16
        )
        self._neighbor_buf: NDArray[np.int16] = np.empty(
            (self.world_h, self.world_w), dtype=np.int16
        )
        self._age_f32_buf: NDArray[np.float32] = np.empty(
            (self.world_h, self.world_w), dtype=np.float32
        )

        self._seed_initial()

    # ── Seeding ─────────────────────────────────────────────────────

    def _seed_initial(self) -> None:
        cy, cx = self.world_h // 2, self.world_w // 2

        for name in METHUSELAHS:
            for _ in range(2):
                y = cy + random.randint(-self.view_h // 3, self.view_h // 3)
                x = cx + random.randint(-self.view_w // 3, self.view_w // 3)
                self._place(name, y, x)

        for _ in range(2):
            y = cy + random.randint(-self.view_h // 4, self.view_h // 4)
            x = cx + random.randint(-self.view_w // 4, self.view_w // 4)
            self._place("gosper_gun", y, x)

        for _ in range(20):
            y = cy + random.randint(-self.view_h, self.view_h)
            x = cx + random.randint(-self.view_w, self.view_w)
            y = max(10, min(y, self.world_h - 10))
            x = max(10, min(x, self.world_w - 10))
            self._place(random.choice(TRAVELLERS), y, x)

        for _ in range(4):
            y = cy + random.randint(-self.view_h // 3, self.view_h // 3)
            x = cx + random.randint(-self.view_w // 3, self.view_w // 3)
            self._place(random.choice(OSCILLATORS), y, x)

        dust_h, dust_w = self.view_h, self.view_w
        dust = (np.random.random((dust_h, dust_w)) < 0.035).astype(np.int8)
        y0, x0 = cy - dust_h // 2, cx - dust_w // 2
        self.grid[y0 : y0 + dust_h, x0 : x0 + dust_w] |= dust

        self.age = np.where(self.grid, 1, 0).astype(np.int32)
        self.age_smooth = self.age.astype(np.float32)

    def _place(
        self, name: str, y: int, x: int, rotation: int | None = None
    ) -> None:
        cells = PATTERNS.get(name)
        if cells is None:
            return
        rot = rotation if rotation is not None else random.randint(0, 3)
        for dy, dx in cells:
            for _ in range(rot):
                dy, dx = dx, -dy
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.world_h and 0 <= nx < self.world_w:
                self.grid[ny, nx] = 1
                self.age[ny, nx] = max(self.age[ny, nx], 1)

    # ── Simulation ──────────────────────────────────────────────────

    def step(self) -> str:
        """Advance one generation. Returns event string (empty if none)."""
        if self.paused:
            return ""

        # Invalidate display caches
        self._disp_grid_cache = None
        self._disp_age_cache = None

        g = self.grid

        # Neighbour count via convolution (toroidal wrap-around)
        if _convolve is not None:
            # Reuse pre-allocated input + output buffers (avoids per-frame allocations)
            np.copyto(self._grid_i16, g)
            _convolve(self._grid_i16, NEIGHBOR_KERNEL, output=self._neighbor_buf, mode="wrap")
            n = self._neighbor_buf
        else:
            # Fallback: manual shift-and-add with np.roll for toroidal wrap
            n = np.zeros_like(g, dtype=np.int16)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    n += np.roll(np.roll(g, -dr, axis=0), -dc, axis=1)

        # Use zero-copy bool view of int8 grid (avoids g==0 / g==1 comparisons)
        g_bool = g.view(np.bool_)
        n_is_3 = n == 3
        birth = ~g_bool & n_is_3
        survive = g_bool & (n_is_3 | (n == 2))

        # In-place grid update (birth/survive are mutually exclusive,
        # so add their int8 views: 0+0=0, 1+0=1, 0+1=1)
        np.add(birth.view(np.int8), survive.view(np.int8), out=self.grid)
        # Age tracking with ghost trails: positive = alive, negative = recently dead
        # Note: inner condition requires age < 0 to avoid turning never-alive
        # empties (age==0) into false ghosts (0 > -GHOST_FRAMES was true → bug)
        self.age = np.where(
            survive, self.age + 1,
            np.where(birth, 1,
            np.where(self.age > 0, -1,  # just died → start ghost
            np.where((self.age < 0) & (self.age > -GHOST_FRAMES), self.age - 1, 0)))
        )
        # Temporal smoothing: in-place ops avoid 3 large temporary allocations
        np.copyto(self._age_f32_buf, self.age)  # int32 → float32
        self._age_f32_buf *= 0.15
        self.age_smooth *= 0.85
        self.age_smooth += self._age_f32_buf
        self.generation += 1

        pop = int(self.grid.sum())
        self._cached_pop = pop
        self.pop_history.append(pop)

        vg = self._raw_view_grid()
        self.hash_history.append(hash(vg.tobytes()))

        event = self._ensure_life(pop)
        self._update_camera()
        return event

    # ── Infinity engine ─────────────────────────────────────────────

    def _ensure_life(self, population: int) -> str:
        """
        The core 'infinite' mechanic. Returns event name if triggered.

        Tunable parameters (visible in stats):
          - pop_floor: ~2.5% of default viewport area
          - stagnation window: 150 generations
          - stagnation threshold: spread < max(8, floor/8)
          - cycle detection: periods 1-30
          - edge spawn interval: every 300 generations
        """
        viewport_area = self.view_h * self.view_w
        base_floor = viewport_area // 40
        # Adaptive floor: rises to ~20% of recent average population,
        # so the simulation faces real challenges at higher pop counts too.
        # Clamped to base_floor minimum so bootstrap still works.
        ph_len = len(self.pop_history)
        if ph_len >= 100:
            # Sum last 100 entries directly from deque (no list conversion)
            recent_sum = 0
            for i in range(ph_len - 100, ph_len):
                recent_sum += self.pop_history[i]
            self.pop_floor = max(base_floor, int(recent_sum * 0.002))  # /100 * 0.20
        else:
            self.pop_floor = base_floor

        # ── Stagnation spread ──
        self.spread = 0
        if ph_len >= 150:
            window_min = self.pop_history[-1]
            window_max = window_min
            for i in range(ph_len - 150, ph_len):
                v = self.pop_history[i]
                if v < window_min:
                    window_min = v
                if v > window_max:
                    window_max = v
            self.spread = window_max - window_min
        stagnant = (
            len(self.pop_history) >= 150
            and self.spread < max(8, self.pop_floor // 8)
        )

        # ── Cycle detection (iterate deque directly, no list conversion) ──
        self.cycle_period = 0
        hh_len = len(self.hash_history)
        if hh_len >= 4:
            latest = self.hash_history[-1]
            for period in range(1, min(31, hh_len)):
                if self.hash_history[-(period + 1)] == latest:
                    self.cycle_period = period
                    break

        # ── Act ──
        event = ""
        if population < self.pop_floor // 3:
            event = "inject:massive"
            self._inject("massive")
        elif population < self.pop_floor or self.cycle_period > 0:
            kind = "heavy"
            if self.cycle_period > 0:
                event = f"inject:heavy(cycle={self.cycle_period})"
            else:
                event = "inject:heavy(low_pop)"
            self._inject(kind)
        elif stagnant:
            event = "inject:medium(stagnant)"
            self._inject("medium")
        elif self.generation % 5000 == 0 and self.generation > 0:
            # Rare garden event — a beautiful symmetric pattern appears
            event = "inject:garden"
            self._inject_garden()
        elif self.generation % 300 == 0 and self.generation > 0:
            event = "inject:edge"
            self._inject_from_edge()
            self._inject_from_edge()

        if event:
            self.last_event = event
            self.last_event_gen = self.generation
            self.total_injections += 1

        return event

    def _inject(self, intensity: str = "medium") -> None:
        cy = self.cam_y + self.view_h // 2
        cx = self.cam_x + self.view_w // 2

        counts = {"massive": 8, "heavy": 5, "medium": 3}
        n = counts.get(intensity, 3)

        for _ in range(n):
            if intensity == "massive" and random.random() < 0.3:
                name = "gosper_gun"
            elif random.random() < 0.4:
                name = random.choice(METHUSELAHS)
            else:
                name = random.choice(TRAVELLERS + OSCILLATORS)

            y = cy + random.randint(-self.view_h // 3, self.view_h // 3)
            x = cx + random.randint(-self.view_w // 3, self.view_w // 3)
            self._place(name, y, x)

        if intensity in ("massive", "heavy"):
            y = cy + random.randint(-self.view_h // 4, self.view_h // 4)
            x = cx + random.randint(-self.view_w // 4, self.view_w // 4)
            self._place("gosper_gun", y, x)

    def _inject_from_edge(self) -> None:
        cy = self.cam_y + self.view_h // 2
        cx = self.cam_x + self.view_w // 2
        edge = random.choice(["top", "bottom", "left", "right"])

        if edge == "top":
            y, x, rot = self.cam_y + 2, cx + random.randint(-self.view_w // 3, self.view_w // 3), 2
        elif edge == "bottom":
            y, x, rot = self.cam_y + self.view_h - 5, cx + random.randint(-self.view_w // 3, self.view_w // 3), 0
        elif edge == "left":
            y, x, rot = cy + random.randint(-self.view_h // 3, self.view_h // 3), self.cam_x + 2, 1
        else:
            y, x, rot = cy + random.randint(-self.view_h // 3, self.view_h // 3), self.cam_x + self.view_w - 10, 3

        self._place(random.choice(TRAVELLERS), y, x, rotation=rot)

    def _inject_garden(self) -> None:
        """Place a symmetric garden pattern near the camera center."""
        cy = self.cam_y + self.view_h // 2
        cx = self.cam_x + self.view_w // 2
        garden = random.choice(GARDENS)
        # Offset slightly from dead center for visual interest
        oy = random.randint(-self.view_h // 6, self.view_h // 6)
        ox = random.randint(-self.view_w // 6, self.view_w // 6)
        for dy, dx in garden:
            ny, nx = cy + oy + dy, cx + ox + dx
            if 0 <= ny < self.world_h and 0 <= nx < self.world_w:
                self.grid[ny, nx] = 1
                self.age[ny, nx] = 1

    # ── Camera ──────────────────────────────────────────────────────

    def _calculate_target_zoom(self) -> int:
        """Calculate optimal zoom level based on activity spread."""
        # Find recent activity (age 1-10) or fall back to all living cells
        active = ((self.age >= 1) & (self.age <= 10)).astype(np.int32)
        if active.sum() < 5:
            active = self.grid.astype(np.int32)
        if active.sum() == 0:
            return self.zoom_level

        ys, xs = np.nonzero(active)
        if len(ys) < 2:
            return self.zoom_level

        # Use interquartile range for a tight bbox that ignores outliers
        y_lo, y_hi = int(np.percentile(ys, 15)), int(np.percentile(ys, 85))
        x_lo, x_hi = int(np.percentile(xs, 15)), int(np.percentile(xs, 85))

        bbox_h = max(4, (y_hi - y_lo) + 1)
        bbox_w = max(4, (x_hi - x_lo) + 1)

        # Choose zoom to fit the core activity in ~50% of viewport
        best_z = MIN_ZOOM
        for z in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
            mag = 2.0 ** z
            vis_h = self.view_h / mag
            vis_w = self.view_w / mag
            if bbox_h <= vis_h * 0.5 and bbox_w <= vis_w * 0.5:
                best_z = z
                break

        return max(MIN_ZOOM, min(MAX_ZOOM, best_z))

    def _update_camera(self) -> None:
        if not self.auto_cam:
            return

        pad = 30
        y0 = max(0, self.cam_y - pad)
        y1 = min(self.world_h, self.cam_y + self.view_h + pad)
        x0 = max(0, self.cam_x - pad)
        x1 = min(self.world_w, self.cam_x + self.view_w + pad)

        region = self.grid[y0:y1, x0:x1]
        total = region.sum()
        if total > 0:
            ys, xs = np.nonzero(region)
            target_y = int(ys.mean()) + y0 - self.view_h // 2
            target_x = int(xs.mean()) + x0 - self.view_w // 2
            target_y = max(0, min(target_y, self.world_h - self.view_h))
            target_x = max(0, min(target_x, self.world_w - self.view_w))

            self.cam_y = int(self.cam_y + (target_y - self.cam_y) * 0.04)
            self.cam_x = int(self.cam_x + (target_x - self.cam_x) * 0.04)

        # Auto-zoom: gradually step toward optimal zoom level
        # Throttle: only recalculate every 4th frame when cooldown is 0
        self._zoom_cooldown = max(0, self._zoom_cooldown - 1)
        if self._zoom_cooldown == 0 and self.generation % 4 == 0:
            target_z = self._calculate_target_zoom()
            if target_z > self.zoom_level and self.zoom_level < MAX_ZOOM:
                self.zoom_level += 1
                self._zoom_cooldown = 45  # ~1.5 sec at 30fps before next change
            elif target_z < self.zoom_level and self.zoom_level > MIN_ZOOM:
                # Verify zoom out is safe (world can fit the expanded view)
                new_z = self.zoom_level - 1
                if new_z >= 0:
                    self.zoom_level = new_z
                    self._zoom_cooldown = 45
                else:
                    factor = 1 << (-new_z)
                    if (
                        self.view_h * factor <= self.world_h
                        and self.view_w * factor <= self.world_w
                    ):
                        self.zoom_level = new_z
                        self._zoom_cooldown = 45

    # ── Viewport access (raw = zoom 0, display = current zoom) ─────

    def _raw_view_grid(self) -> NDArray[np.int8]:
        y0, x0 = self._clamp_cam()
        return self.grid[y0 : y0 + self.view_h, x0 : x0 + self.view_w]

    def _clamp_cam(self) -> tuple[int, int]:
        y = max(0, min(self.cam_y, self.world_h - self.view_h))
        x = max(0, min(self.cam_x, self.world_w - self.view_w))
        return y, x

    def _view_center(self) -> tuple[int, int]:
        y0, x0 = self._clamp_cam()
        return y0 + self.view_h // 2, x0 + self.view_w // 2

    def display_grid(self) -> NDArray[np.int8]:
        """Grid scaled for the current zoom level, sized view_h × view_w."""
        if self._disp_grid_cache is not None:
            return self._disp_grid_cache
        result = self._compute_display_grid()
        self._disp_grid_cache = result
        return result

    def _compute_display_grid(self) -> NDArray[np.int8]:
        """Compute the display grid (called once per frame, cached)."""
        if self.zoom_level == 0:
            return self._raw_view_grid()

        cy, cx = self._view_center()

        if self.zoom_level < 0:
            factor = 1 << (-self.zoom_level)  # 2 or 4
            cover_h = self.view_h * factor
            cover_w = self.view_w * factor

            y0 = max(0, min(cy - cover_h // 2, self.world_h - cover_h))
            x0 = max(0, min(cx - cover_w // 2, self.world_w - cover_w))

            if cover_h > self.world_h or cover_w > self.world_w:
                return self._raw_view_grid()

            region = self.grid[y0 : y0 + cover_h, x0 : x0 + cover_w]
            bh = cover_h // factor
            bw = cover_w // factor
            trimmed = region[: bh * factor, : bw * factor]
            return trimmed.reshape(bh, factor, bw, factor).max(axis=(1, 3)).astype(np.int8)

        else:
            factor = 1 << self.zoom_level  # 2 or 4
            cover_h = max(2, self.view_h // factor)
            cover_w = max(2, self.view_w // factor)

            y0 = max(0, min(cy - cover_h // 2, self.world_h - cover_h))
            x0 = max(0, min(cx - cover_w // 2, self.world_w - cover_w))

            region = self.grid[y0 : y0 + cover_h, x0 : x0 + cover_w]
            up = np.repeat(np.repeat(region, factor, axis=0), factor, axis=1)
            return up[: self.view_h, : self.view_w].astype(np.int8)

    def display_age(self) -> NDArray[np.int32]:
        """Age map scaled for the current zoom level, sized view_h × view_w.

        Handles ghost trails: positive ages = alive, negative = ghost (recently dead).
        When downsampling, prioritizes living cells; shows brightest ghost otherwise.
        """
        if self._disp_age_cache is not None:
            return self._disp_age_cache
        result = self._compute_display_age()
        self._disp_age_cache = result
        return result

    def _compute_display_age(self) -> NDArray[np.int32]:
        """Compute the display age map (called once per frame, cached).

        Slices the viewport region FIRST, then rounds — avoids rounding
        the entire 590×1000 world array when only a small viewport is needed.
        """
        if self.zoom_level == 0:
            y0, x0 = self._clamp_cam()
            region_smooth = self.age_smooth[y0 : y0 + self.view_h, x0 : x0 + self.view_w]
            return np.round(region_smooth).astype(np.int32)

        cy, cx = self._view_center()

        if self.zoom_level < 0:
            factor = 1 << (-self.zoom_level)
            cover_h = self.view_h * factor
            cover_w = self.view_w * factor

            y0 = max(0, min(cy - cover_h // 2, self.world_h - cover_h))
            x0 = max(0, min(cx - cover_w // 2, self.world_w - cover_w))

            if cover_h > self.world_h or cover_w > self.world_w:
                y0b, x0b = self._clamp_cam()
                region_smooth = self.age_smooth[y0b : y0b + self.view_h, x0b : x0b + self.view_w]
                return np.round(region_smooth).astype(np.int32)

            region_smooth = self.age_smooth[y0 : y0 + cover_h, x0 : x0 + cover_w]
            region = np.round(region_smooth).astype(np.int32)
            bh = cover_h // factor
            bw = cover_w // factor
            trimmed = region[: bh * factor, : bw * factor]
            blocks = trimmed.reshape(bh, factor, bw, factor)

            # Smart aggregation: prioritize alive (positive), then brightest ghost (least negative)
            max_vals = blocks.max(axis=(1, 3))
            # For ghost detection, mask zeros so we can find the max of negative values only
            # Sentinel value more negative than any valid ghost age
            sentinel = -GHOST_FRAMES - 1
            masked = np.where(trimmed != 0, trimmed, sentinel)
            ghost_max = masked.reshape(bh, factor, bw, factor).max(axis=(1, 3))
            # Use max if alive; else ghost_max if valid ghost exists; else 0
            return np.where(max_vals > 0, max_vals, np.where(ghost_max > sentinel, ghost_max, 0))

        else:
            factor = 1 << self.zoom_level
            cover_h = max(2, self.view_h // factor)
            cover_w = max(2, self.view_w // factor)

            y0 = max(0, min(cy - cover_h // 2, self.world_h - cover_h))
            x0 = max(0, min(cx - cover_w // 2, self.world_w - cover_w))

            region_smooth = self.age_smooth[y0 : y0 + cover_h, x0 : x0 + cover_w]
            region = np.round(region_smooth).astype(np.int32)
            up = np.repeat(np.repeat(region, factor, axis=0), factor, axis=1)
            return up[: self.view_h, : self.view_w]

    # ── Zoom ────────────────────────────────────────────────────────

    def zoom_in(self) -> None:
        if self.zoom_level < MAX_ZOOM:
            self.zoom_level += 1

    def zoom_out(self) -> None:
        if self.zoom_level > MIN_ZOOM:
            new_z = self.zoom_level - 1
            if new_z >= 0:
                # Zooming toward or at default — always safe
                self.zoom_level = new_z
            else:
                # Zooming past default into negative — verify world fits
                factor = 1 << (-new_z)
                if (
                    self.view_h * factor <= self.world_h
                    and self.view_w * factor <= self.world_w
                ):
                    self.zoom_level = new_z

    def auto_focus(self) -> None:
        """
        Zoom + pan to the densest cluster of recent activity.

        Strategy: build a coarse heat map, find the hottest spot, then
        use the interquartile range (not full bounding box) of nearby
        active cells to size the zoom — this ignores sparse outliers
        and focuses on the tight core of the action.
        """
        active = ((self.age >= 1) & (self.age <= 10)).astype(np.int32)
        if active.sum() < 5:
            active = self.grid.astype(np.int32)
        if active.sum() == 0:
            return

        # Coarse heat map: 16×16 cell blocks
        block = 16
        h, w = active.shape
        bh, bw = h // block, w // block
        if bh < 1 or bw < 1:
            return

        trimmed = active[: bh * block, : bw * block]
        heat = trimmed.reshape(bh, block, bw, block).sum(axis=(1, 3))

        # Find the hottest 3×3 neighbourhood (not just one block)
        try:
            smooth_heat = _uniform_filter(heat.astype(float), size=3, mode="constant")
        except Exception:
            smooth_heat = heat.astype(float)
        hot_by, hot_bx = np.unravel_index(int(smooth_heat.argmax()), smooth_heat.shape)
        focus_cy = hot_by * block + block // 2
        focus_cx = hot_bx * block + block // 2

        # Tight search region around hotspot (±2 blocks)
        region_r = block * 2
        ry0 = max(0, focus_cy - region_r)
        ry1 = min(h, focus_cy + region_r)
        rx0 = max(0, focus_cx - region_r)
        rx1 = min(w, focus_cx + region_r)

        local = active[ry0:ry1, rx0:rx1]
        if local.sum() == 0:
            return

        ys, xs = np.nonzero(local)

        # Use interquartile range for a tight bbox that ignores outliers
        def iqr_bounds(arr: NDArray) -> tuple[int, int]:
            q1 = int(np.percentile(arr, 10))
            q3 = int(np.percentile(arr, 90))
            return q1, q3

        y_lo, y_hi = iqr_bounds(ys)
        x_lo, x_hi = iqr_bounds(xs)

        bbox_h = max(4, (y_hi - y_lo) + 1)
        bbox_w = max(4, (x_hi - x_lo) + 1)
        center_y = ry0 + (y_lo + y_hi) // 2
        center_x = rx0 + (x_lo + x_hi) // 2

        # Choose zoom to fit the core cluster in ~60% of viewport
        best_z = MIN_ZOOM
        for z in range(MAX_ZOOM, MIN_ZOOM - 1, -1):
            mag = 2.0 ** z
            vis_h = self.view_h / mag
            vis_w = self.view_w / mag
            if bbox_h <= vis_h * 0.6 and bbox_w <= vis_w * 0.6:
                best_z = z
                break

        self.zoom_level = max(MIN_ZOOM, min(MAX_ZOOM, best_z))
        self.cam_y = max(0, min(center_y - self.view_h // 2, self.world_h - self.view_h))
        self.cam_x = max(0, min(center_x - self.view_w // 2, self.world_w - self.view_w))
        self.auto_cam = False
        # Invalidate display caches — zoom/camera changed after step()
        self._disp_grid_cache = None
        self._disp_age_cache = None

    # ── Controls ────────────────────────────────────────────────────

    def pan(self, dy: int, dx: int) -> None:
        self.auto_cam = False
        # Scale pan speed with zoom: faster when zoomed out, slower when in
        factor = 2.0 ** (-self.zoom_level)
        sy = max(1, int(abs(dy) * factor)) * (1 if dy > 0 else -1) if dy != 0 else 0
        sx = max(1, int(abs(dx) * factor)) * (1 if dx > 0 else -1) if dx != 0 else 0
        self.cam_y = max(0, min(self.cam_y + sy, self.world_h - self.view_h))
        self.cam_x = max(0, min(self.cam_x + sx, self.world_w - self.view_w))

    def home(self) -> None:
        self.auto_cam = True
        self.zoom_level = 0

    def clear(self) -> None:
        self.grid[:] = 0
        self.age[:] = 0
        self.age_smooth[:] = 0
        self.generation = 0
        self.pop_history.clear()
        self.hash_history.clear()

    def toggle_cell(self, term_y: int, term_x: int) -> None:
        """Toggle a cell at terminal coordinates, accounting for zoom."""
        cy, cx = self._view_center()

        # Map terminal pixel → display grid → world grid
        dy = term_y * 2  # half-block: each term row = 2 display rows
        dx = term_x

        if self.zoom_level <= 0:
            factor = 1 << (-self.zoom_level)  # 1, 2, 4
            cover_h = self.view_h * factor
            cover_w = self.view_w * factor
            base_y = cy - cover_h // 2
            base_x = cx - cover_w // 2
            gy = base_y + dy * factor
            gx = base_x + dx * factor
        else:
            factor = 1 << self.zoom_level  # 2, 4
            cover_h = max(2, self.view_h // factor)
            cover_w = max(2, self.view_w // factor)
            base_y = cy - cover_h // 2
            base_x = cx - cover_w // 2
            gy = base_y + dy // factor
            gx = base_x + dx // factor

        if 0 <= gy < self.world_h and 0 <= gx < self.world_w:
            self.grid[gy, gx] ^= 1
            self.age[gy, gx] = 1 if self.grid[gy, gx] else 0

    def tick_ticker(self, ticker_width: int) -> None:
        """Advance the news ticker by one frame."""
        mood = self._detect_mood()
        self.ticker.tick(ticker_width, mood, self.generation)

    def _detect_mood(self) -> str:
        """Classify current simulation mood for context-aware musings.

        Result is cached per generation — safe to call multiple times per frame.
        """
        if self._cached_mood_gen == self.generation:
            return self._cached_mood

        mood = self._compute_mood()
        self._cached_mood = mood
        self._cached_mood_gen = self.generation
        return mood

    def _compute_mood(self) -> str:
        """Internal mood computation (called once per generation)."""
        # Recent injection gets priority (show for ~90 frames after event)
        if self.last_event and (self.generation - self.last_event_gen) < 90:
            return "injection"

        # Cycle detected
        if self.cycle_period > 0:
            return "cycle"

        # Milestone (every 10,000 generations, show for 120 frames)
        if self.generation >= 10_000 and self.generation % 10_000 < 120:
            return "milestone"

        # Population trends (need enough history)
        ph_len = len(self.pop_history)
        if ph_len >= 30:
            # Sum directly from deque instead of list conversion
            first_sum = 0
            second_sum = 0
            for i in range(ph_len - 30, ph_len - 15):
                first_sum += self.pop_history[i]
            for i in range(ph_len - 15, ph_len):
                second_sum += self.pop_history[i]
            ratio = (second_sum / 15) / max(first_sum / 15, 1)

            if ratio > 1.15:
                return "booming"
            elif ratio < 0.85:
                return "declining"

        # Density
        pop = self.pop_history[-1] if self.pop_history else 0
        viewport_area = self.view_h * self.view_w
        density = pop / max(viewport_area, 1)
        if density > 0.15:
            return "dense"
        elif pop < self.pop_floor and pop > 0:
            return "sparse"

        # Stagnation
        if self.spread < max(8, self.pop_floor // 8) and ph_len >= 150:
            return "stagnant"

        return ""

    def epoch(self) -> str:
        """Return the current epoch name based on generation count."""
        name = EPOCHS[0][1]
        for threshold, label in EPOCHS:
            if self.generation >= threshold:
                name = label
        return name

    def population(self) -> int:
        return self._cached_pop

    def sparkline(self, width: int = 24) -> str:
        ph_len = len(self.pop_history)
        if ph_len < 2:
            return ""
        # Iterate deque directly instead of converting to list
        start = max(0, ph_len - width)
        lo = self.pop_history[start]
        hi = lo
        for i in range(start, ph_len):
            v = self.pop_history[i]
            if v < lo:
                lo = v
            if v > hi:
                hi = v
        n_sparks = len(SPARKS) - 1
        mid_spark = SPARKS[len(SPARKS) // 2]
        out: list[str] = []
        for i in range(start, ph_len):
            v = self.pop_history[i]
            if hi == lo:
                out.append(mid_spark)
            else:
                idx = int((v - lo) / (hi - lo) * n_sparks)
                out.append(SPARKS[idx])
        return "".join(out)


# ═══════════════════════════════════════════════════════════════════════
#  Rendering
# ═══════════════════════════════════════════════════════════════════════

def age_to_color_idx(cell_age: int, max_age: int, n: int) -> int:
    if cell_age <= 0 or max_age <= 1:
        return 0
    return min(int(math.log1p(cell_age) / math.log1p(max_age) * (n - 1)), n - 1)


def ghost_color_idx(ghost_age: int) -> int:
    """Map negative ghost age to GHOST_COLORS index (0 = brightest, recently dead)."""
    # ghost_age is -1 (just died) to -GHOST_FRAMES (about to vanish)
    idx = min(-ghost_age - 1, GHOST_FRAMES - 1)
    return max(0, idx)


def render(
    stdscr: curses.window,
    life: InfiniteLife,
    cmap: ColorMap,
    show_stats: bool = False,
    music_status: str = "",
) -> None:
    """Half-block rendering with ghost trails and optional stats overlay.

    Uses numpy vectorized pre-computation to avoid per-cell Python function
    calls (ghost_color_idx, max/min clamping, LUT lookup), then iterates
    only the active cells for curses output.
    """
    max_y, max_x = stdscr.getmaxyx()
    grid = life.display_grid()
    ages = life.display_age()
    g_rows, g_cols = grid.shape
    a_rows, a_cols = ages.shape

    # Use intersection of grid/ages shapes (defensive against cache staleness)
    draw_rows = min(g_rows // 2, a_rows // 2, max_y - 1)
    draw_cols = min(g_cols, a_cols, max_x)
    # ages.max() works because alive ages > 0, ghost ages < 0, dead = 0
    max_age = max(int(ages.max()), 1)
    ng = cmap.n_gradient

    # Pre-compute age→color lookup table (replaces per-cell math.log1p calls)
    lut_size = max_age + 1
    ages_1d = np.arange(lut_size, dtype=np.float64)
    log_max = math.log1p(max_age)
    color_lut = np.minimum(
        (np.log1p(ages_1d) / log_max * (ng - 1)).astype(np.intp), ng - 1
    )
    color_lut[0] = 0

    # Disable ghost rendering when zoomed out - too much visual noise
    show_ghosts = life.zoom_level >= 0

    # ── Vectorized pre-computation ────────────────────────────────
    row_end = draw_rows * 2
    top_grid = grid[0:row_end:2, :draw_cols]    # even rows → top pixel
    bot_grid = grid[1:row_end:2, :draw_cols]    # odd rows  → bottom pixel
    top_ages = ages[0:row_end:2, :draw_cols]
    bot_ages = ages[1:row_end:2, :draw_cols]

    top_alive = top_grid > 0
    bot_alive = bot_grid > 0

    # Ghost masks + active mask (skip ghost arrays entirely when disabled)
    if show_ghosts:
        top_ghost = top_ages < 0
        bot_ghost = bot_ages < 0
        active_mask = top_alive | bot_alive | top_ghost | bot_ghost
    else:
        active_mask = top_alive | bot_alive

    # Color indices: vectorized clip + LUT (replaces per-cell max/min/ghost_color_idx)
    top_cidx = color_lut[np.clip(top_ages, 0, max_age)]
    bot_cidx = color_lut[np.clip(bot_ages, 0, max_age)]
    top_gidx = np.clip(-top_ages - 1, 0, GHOST_FRAMES - 1)
    bot_gidx = np.clip(-bot_ages - 1, 0, GHOST_FRAMES - 1)

    # Get active cell positions — only iterate cells that need drawing
    active_ys, active_xs = np.nonzero(active_mask)
    n_active = len(active_ys)

    # Pre-extract all values as Python lists (.tolist avoids per-element
    # numpy scalar → Python conversion overhead in the loop)
    ys = active_ys.tolist()
    xs = active_xs.tolist()
    ta = top_alive[active_ys, active_xs].tolist()
    ba = bot_alive[active_ys, active_xs].tolist()
    tc = top_cidx[active_ys, active_xs].tolist()
    bc = bot_cidx[active_ys, active_xs].tolist()

    if show_ghosts:
        tg = top_ghost[active_ys, active_xs].tolist()
        bg = bot_ghost[active_ys, active_xs].tolist()
        tgi = top_gidx[active_ys, active_xs].tolist()
        bgi = bot_gidx[active_ys, active_xs].tolist()
    else:
        tg = bg = [False] * n_active
        tgi = bgi = [0] * n_active

    # Local references (avoid attribute lookups in tight loop)
    _addstr = stdscr.addstr
    _color_pair = curses.color_pair
    _BOLD = curses.A_BOLD
    _dual = cmap.dual
    _fg = cmap.fg
    _ghost_dual = cmap.ghost_dual
    _ghost_fg = cmap.ghost_fg

    for i in range(n_active):
        try:
            # Both cells have something to show
            if (ta[i] or tg[i]) and (ba[i] or bg[i]):
                if ta[i] and ba[i]:
                    attr = _color_pair(_dual(tc[i], bc[i])) | _BOLD
                elif tg[i] and bg[i]:
                    attr = _color_pair(_ghost_dual(tgi[i], bgi[i]))
                elif ta[i]:
                    _addstr(ys[i], xs[i], UPPER_HALF,
                            _color_pair(_fg(tc[i])) | _BOLD)
                    continue
                else:
                    _addstr(ys[i], xs[i], LOWER_HALF,
                            _color_pair(_fg(bc[i])) | _BOLD)
                    continue
                _addstr(ys[i], xs[i], UPPER_HALF, attr)
            elif ta[i]:
                _addstr(ys[i], xs[i], UPPER_HALF,
                        _color_pair(_fg(tc[i])) | _BOLD)
            elif ba[i]:
                _addstr(ys[i], xs[i], LOWER_HALF,
                        _color_pair(_fg(bc[i])) | _BOLD)
            elif tg[i]:
                _addstr(ys[i], xs[i], UPPER_HALF,
                        _color_pair(_ghost_fg(tgi[i])))
            elif bg[i]:
                _addstr(ys[i], xs[i], LOWER_HALF,
                        _color_pair(_ghost_fg(bgi[i])))
        except curses.error:
            pass

    # ── Stats overlay ───────────────────────────────────────────────
    if show_stats:
        _draw_stats_overlay(stdscr, life, max_y, max_x)

    # ── Status bar (with scrolling news ticker) ────────────────────
    pop = life.population()
    spark = life.sparkline()
    cam = "auto" if life.auto_cam else "pan"
    zoom_label = ZOOM_LABELS.get(life.zoom_level, f"{2**life.zoom_level}x")

    epoch = life.epoch()
    focus_indicator = "[F]" if life.auto_focus_mode else ""
    music_indicator = f" {music_status}" if music_status else ""
    left = f"  {epoch}  gen {life.generation:,}  pop {pop:,}  {spark}"
    right = f"{music_indicator} {focus_indicator} {zoom_label} {cam}  q r spc +/- arrows h z/x f s  "

    # Ticker occupies the region between left stats and right controls
    ticker_col = len(left) + 1
    ticker_width = max_x - len(left) - len(right) - 2

    if ticker_width < 10:
        # Terminal too narrow for ticker — compact fallback
        status = (left + "  " + right)[: max_x - 1]
        try:
            stdscr.addstr(max_y - 1, 0, status, curses.A_DIM)
        except curses.error:
            pass
    else:
        # Tick the ticker forward, then render: left | ticker | right
        life.tick_ticker(ticker_width)
        try:
            stdscr.addstr(max_y - 1, 0, left, curses.A_DIM)
        except curses.error:
            pass
        life.ticker.render(
            stdscr, max_y - 1, ticker_col, ticker_width, life.generation,
        )
        try:
            right_col = max_x - len(right)
            r = right[: max_x - 1 - right_col]
            stdscr.addstr(max_y - 1, right_col, r, curses.A_DIM)
        except curses.error:
            pass


def _draw_stats_overlay(
    stdscr: curses.window, life: InfiniteLife, max_y: int, max_x: int
) -> None:
    """Draw the engine telemetry panel in the bottom-right."""
    panel_w = 36
    panel_h = 9
    x0 = max_x - panel_w - 2
    y0 = max_y - panel_h - 2

    if x0 < 0 or y0 < 0:
        return

    lines = [
        f"{'':─<{panel_w - 2}}",
        f" infinity engine",
        f" pop floor  : {life.pop_floor:,}",
        f" spread/150 : {life.spread:,}",
        f" cycle       : {'none' if life.cycle_period == 0 else f'period {life.cycle_period}'}",
        f" injections  : {life.total_injections}",
        f" last event  : {life.last_event or 'none'}",
        f"   @ gen     : {life.last_event_gen:,}" if life.last_event else "               ",
        f" world       : {life.world_h}x{life.world_w}",
    ]

    style = curses.A_DIM
    for i, line in enumerate(lines):
        row = y0 + i
        if 0 <= row < max_y - 1:
            padded = f" {line:<{panel_w - 1}}"[: panel_w]
            try:
                stdscr.addstr(row, x0, padded, style)
            except curses.error:
                pass


# ═══════════════════════════════════════════════════════════════════════
#  Main loop
# ═══════════════════════════════════════════════════════════════════════

def main(stdscr: curses.window) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    curses.mousemask(curses.ALL_MOUSE_EVENTS)

    cmap = ColorMap()
    cmap.setup()

    max_y, max_x = stdscr.getmaxyx()
    life = InfiniteLife(max_y - 1, max_x)

    logger = StatsLogger(LOG_PATH)
    logger.open()

    # ── Music engine ──
    music: LifeMusicEngine | None = None  # type: ignore[name-defined]
    if _HAS_MUSIC:
        try:
            music = LifeMusicEngine()
            if not music.start():
                music = None
        except Exception:
            music = None

    # Playhead state for music scanning
    _playhead_col_idx: int = 0
    _playhead_frame: int = 0

    show_stats = False

    # Snapshot recording state (toggled with 'd')
    _snap_recording: bool = False
    _snap_file: IO[str] | None = None
    _snap_path: Path = Path(__file__).resolve().parent / "snapshots.jsonl"
    _snap_count: int = 0

    try:
        while True:
            # ── Input ──────────────────────────────────────────────
            try:
                key = stdscr.getch()
            except curses.error:
                key = -1

            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("r"), ord("R")):
                max_y, max_x = stdscr.getmaxyx()
                life = InfiniteLife(max_y - 1, max_x)
                _playhead_col_idx = 0
            elif key == ord(" "):
                life.paused = not life.paused
            elif key in (ord("+"), ord("=")):
                life.delay = max(10, life.delay - 10)
            elif key in (ord("-"), ord("_")):
                life.delay = min(500, life.delay + 10)
            elif key in (ord("h"), ord("H")):
                life.home()
            elif key in (ord("c"), ord("C")):
                life.clear()
            elif key in (ord("z"), ord("Z")):
                life.zoom_out()
            elif key in (ord("x"), ord("X")):
                life.zoom_in()
            elif key in (ord("f"), ord("F")):
                life.auto_focus_mode = not life.auto_focus_mode
                if life.auto_focus_mode:
                    life.auto_focus()  # Immediate first focus
            elif key in (ord("s"), ord("S")):
                show_stats = not show_stats
            elif key in (ord("d"), ord("D")):
                # Toggle snapshot recording for diagnostics
                if _HAS_MUSIC:
                    _snap_recording = not _snap_recording
                    if _snap_recording:
                        try:
                            _snap_file = open(_snap_path, "w")
                            _snap_count = 0
                        except OSError:
                            _snap_recording = False
                            _snap_file = None
                    else:
                        if _snap_file is not None:
                            _snap_file.close()
                            _snap_file = None
            elif key in (ord("m"), ord("M")):
                if music is not None:
                    music.toggle_mute()
            elif key in (ord("v"), ord("V")):
                if music is not None:
                    music.cycle_style()
            elif key == ord("["):
                if music is not None:
                    music.adjust_volume(-0.1)
            elif key == ord("]"):
                if music is not None:
                    music.adjust_volume(0.1)
            elif key == curses.KEY_UP:
                life.pan(-5, 0)
            elif key == curses.KEY_DOWN:
                life.pan(5, 0)
            elif key == curses.KEY_LEFT:
                life.pan(0, -10)
            elif key == curses.KEY_RIGHT:
                life.pan(0, 10)
            elif key == curses.KEY_MOUSE:
                try:
                    _, mx, my, _, _ = curses.getmouse()
                    life.toggle_cell(my, mx)
                except curses.error:
                    pass
            elif key == curses.KEY_RESIZE:
                max_y, max_x = stdscr.getmaxyx()
                life = InfiniteLife(max_y - 1, max_x)
                _playhead_col_idx = 0

            # ── Simulate ───────────────────────────────────────────
            event = life.step()

            # ── Music update (every 2nd frame — audio smooths transitions) ──
            if music is not None and not life.paused and life.generation % 2 == 0:
                try:
                    # Advance playhead across viewport columns
                    view_w = life.view_w
                    _playhead_frame += 1
                    # Sweep speed: one full scan in ~8 seconds at ~30fps
                    cols_per_frame = max(1, view_w // 240)
                    _playhead_col_idx = (_playhead_col_idx + cols_per_frame) % max(1, view_w)
                    playhead_pos = _playhead_col_idx / max(1, view_w)

                    # Extract column from display grid for playhead (uses cache)
                    disp = life.display_grid()
                    col_data: tuple[bool, ...] = ()
                    if 0 <= _playhead_col_idx < disp.shape[1]:
                        col_data = tuple(disp[:, _playhead_col_idx].astype(bool))

                    # Population delta (direct deque access, no list conversion)
                    ph_len = len(life.pop_history)
                    pop_delta = (life.pop_history[-1] - life.pop_history[-2]) if ph_len >= 2 else 0

                    # Density (uses cached population)
                    pop = life.population()
                    viewport_area = max(1, life.view_h * life.view_w)
                    density = pop / viewport_area

                    snap = SimulationSnapshot(
                        generation=life.generation,
                        population=pop,
                        pop_floor=life.pop_floor,
                        density=density,
                        spread=life.spread,
                        cycle_period=life.cycle_period,
                        mood=life._detect_mood(),
                        epoch=life.epoch(),
                        pop_delta=pop_delta,
                        playhead_column=col_data,
                        playhead_position=playhead_pos,
                        viewport_rows=disp.shape[0],
                    )
                    music.update(snap)
                    # Record snapshot if diagnostics recording is active
                    if _snap_recording and _snap_file is not None:
                        try:
                            # Convert playhead_column bools to compact 0/1 list
                            snap_dict = asdict(snap)
                            snap_dict["playhead_column"] = [
                                int(b) for b in snap.playhead_column
                            ]
                            _snap_file.write(json.dumps(snap_dict) + "\n")
                            _snap_count += 1
                        except OSError:
                            pass
                except Exception:
                    pass  # Never let music crash the sim

            # ── Auto-focus mode (every other frame) ───────────────
            if life.auto_focus_mode:
                life._auto_focus_frame += 1
                if life._auto_focus_frame >= 2:
                    life._auto_focus_frame = 0
                    life.auto_focus()

            # ── Log ────────────────────────────────────────────────
            if event or life.generation % 10 == 0:
                logger.log(
                    gen=life.generation,
                    pop=life.population(),
                    spread=life.spread,
                    floor=life.pop_floor,
                    cycle=life.cycle_period,
                    zoom=life.zoom_level,
                    cam_y=life.cam_y,
                    cam_x=life.cam_x,
                    event=event,
                )

            # ── Render ─────────────────────────────────────────────
            stdscr.erase()
            _music_status = music.status_string() if music is not None else ""
            if _snap_recording:
                _music_status += f" REC({_snap_count})"
            render(stdscr, life, cmap, show_stats=show_stats,
                   music_status=_music_status)
            stdscr.refresh()

            time.sleep(life.delay / 1000.0)

    finally:
        if music is not None:
            try:
                music.stop()
            except Exception:
                pass
        if _snap_file is not None:
            try:
                _snap_file.close()
            except OSError:
                pass
        logger.close()


if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        pass
