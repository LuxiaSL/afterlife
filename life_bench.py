#!/usr/bin/env python3
"""
Profiling harness for the Life screensaver.

Runs the simulation + rendering pipeline headlessly under cProfile,
then prints a ranked breakdown of where time is spent.

Usage:
  python3 life_bench.py                  # 500 frames, summary
  python3 life_bench.py -n 1000          # 1000 frames
  python3 life_bench.py --line-timing    # per-frame component timing
  python3 life_bench.py --dump prof.out  # dump cProfile binary for snakeviz etc.
"""

from __future__ import annotations

import argparse
import cProfile
import math
import pstats
import time
from io import StringIO
from typing import TextIO

import numpy as np

# Import the actual simulation and rendering components
from life import (
    GHOST_FRAMES,
    SPARKS,
    ColorMap,
    InfiniteLife,
    age_to_color_idx,
    ghost_color_idx,
)

try:
    from life_music import LifeMusicEngine, SimulationSnapshot
    _HAS_MUSIC = True
except ImportError:
    _HAS_MUSIC = False


# ── Fake curses stubs for headless rendering ────────────────────────────

class FakeWindow:
    """Minimal curses.window stub that absorbs addstr calls."""

    def __init__(self, rows: int, cols: int) -> None:
        self._rows = rows
        self._cols = cols
        self._calls = 0

    def getmaxyx(self) -> tuple[int, int]:
        return self._rows, self._cols

    def addstr(self, *args: object) -> None:
        self._calls += 1

    def erase(self) -> None:
        pass

    def refresh(self) -> None:
        pass


def simulate_render_work(life: InfiniteLife, ng: int = 12) -> dict[str, float]:
    """
    Simulate the render() hot path without curses, timing each component.

    Returns a dict of component → seconds.
    """
    timings: dict[str, float] = {}

    # 1. display_grid + display_age (zoom scaling)
    t0 = time.perf_counter()
    grid = life.display_grid()
    ages = life.display_age()
    timings["display_grid+age"] = time.perf_counter() - t0

    g_rows, g_cols = grid.shape
    a_rows, a_cols = ages.shape
    draw_rows = min(g_rows // 2, a_rows // 2, 60)  # typical terminal height
    draw_cols = min(g_cols, a_cols, 200)  # typical terminal width

    # 2. max_age + LUT computation
    t0 = time.perf_counter()
    max_age = max(int(ages.max()), 1)
    lut_size = max_age + 1
    ages_1d = np.arange(lut_size, dtype=np.float64)
    log_max = math.log1p(max_age)
    color_lut = np.minimum(
        (np.log1p(ages_1d) / log_max * (ng - 1)).astype(np.intp), ng - 1
    )
    color_lut[0] = 0
    timings["lut_build"] = time.perf_counter() - t0

    # 3. Vectorized render pre-computation + active-cell loop
    t0 = time.perf_counter()
    show_ghosts = life.zoom_level >= 0

    row_end = draw_rows * 2
    top_grid = grid[0:row_end:2, :draw_cols]
    bot_grid = grid[1:row_end:2, :draw_cols]
    top_ages = ages[0:row_end:2, :draw_cols]
    bot_ages = ages[1:row_end:2, :draw_cols]

    top_alive = top_grid > 0
    bot_alive = bot_grid > 0

    if show_ghosts:
        top_ghost = top_ages < 0
        bot_ghost = bot_ages < 0
        active_mask = top_alive | bot_alive | top_ghost | bot_ghost
    else:
        active_mask = top_alive | bot_alive

    top_cidx = color_lut[np.clip(top_ages, 0, max_age)]
    bot_cidx = color_lut[np.clip(bot_ages, 0, max_age)]
    top_gidx = np.clip(-top_ages - 1, 0, GHOST_FRAMES - 1)
    bot_gidx = np.clip(-bot_ages - 1, 0, GHOST_FRAMES - 1)

    active_ys, active_xs = np.nonzero(active_mask)
    n_active = len(active_ys)

    ys = active_ys.tolist()
    xs = active_xs.tolist()
    ta = top_alive[active_ys, active_xs].tolist()
    ba = bot_alive[active_ys, active_xs].tolist()
    tc = top_cidx[active_ys, active_xs].tolist()
    bc = bot_cidx[active_ys, active_xs].tolist()

    if show_ghosts:
        tg_list = (top_ages < 0)[active_ys, active_xs].tolist()
        bg_list = (bot_ages < 0)[active_ys, active_xs].tolist()
        tgi_list = top_gidx[active_ys, active_xs].tolist()
        bgi_list = bot_gidx[active_ys, active_xs].tolist()
    else:
        tg_list = bg_list = [False] * n_active
        tgi_list = bgi_list = [0] * n_active

    # Simulate the inner loop (branching + fake addstr calls)
    char_calls = 0
    for i in range(n_active):
        char_calls += 1
        if (ta[i] or tg_list[i]) and (ba[i] or bg_list[i]):
            if ta[i] and ba[i]:
                _ = tc[i], bc[i]
            elif tg_list[i] and bg_list[i]:
                _ = tgi_list[i], bgi_list[i]
            elif ta[i]:
                _ = tc[i]
            else:
                _ = bc[i]
        elif ta[i]:
            _ = tc[i]
        elif ba[i]:
            _ = bc[i]
        elif tg_list[i]:
            _ = tgi_list[i]
        elif bg_list[i]:
            _ = bgi_list[i]

    timings["render_loop"] = time.perf_counter() - t0
    timings["_char_calls"] = float(char_calls)

    # 4. Sparkline + status bar text
    t0 = time.perf_counter()
    _ = life.sparkline()
    _ = life.epoch()
    _ = life.population()
    _ = life._detect_mood()
    timings["status_bar"] = time.perf_counter() - t0

    return timings


def run_benchmark(
    n_frames: int,
    term_rows: int = 60,
    term_cols: int = 200,
    line_timing: bool = False,
    dump_path: str | None = None,
) -> None:
    """Run the benchmark for n_frames and report results."""

    life = InfiniteLife(term_rows - 1, term_cols)
    ng = len([17, 19, 21, 57, 93, 165, 163, 204, 209, 214, 220, 231])

    print(f"World: {life.world_h}x{life.world_w}  "
          f"View: {life.view_h}x{life.view_w}  "
          f"Frames: {n_frames}")
    print(f"Terminal: {term_rows}x{term_cols}")
    print()

    # ── Per-frame component timing ─────────────────────────────────
    if line_timing:
        step_times: list[float] = []
        render_times: dict[str, list[float]] = {}
        total_times: list[float] = []

        for frame in range(n_frames):
            frame_t0 = time.perf_counter()

            # Step
            t0 = time.perf_counter()
            life.step()
            step_dt = time.perf_counter() - t0
            step_times.append(step_dt)

            # Render components
            rt = simulate_render_work(life, ng)
            for k, v in rt.items():
                render_times.setdefault(k, []).append(v)

            total_times.append(time.perf_counter() - frame_t0)

            if (frame + 1) % 100 == 0:
                avg_ms = sum(total_times[-100:]) / 100 * 1000
                print(f"  frame {frame + 1}/{n_frames}  "
                      f"avg {avg_ms:.1f}ms/frame  "
                      f"pop {life.population():,}")

        print()
        print("=== Per-Frame Component Breakdown (ms) ===")
        print(f"{'Component':<25} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Max':>8}")
        print("-" * 73)

        def stats_line(name: str, data: list[float]) -> str:
            arr = np.array(data) * 1000  # to ms
            return (f"{name:<25} {arr.mean():8.2f} {np.median(arr):8.2f} "
                    f"{np.percentile(arr, 95):8.2f} {np.percentile(arr, 99):8.2f} "
                    f"{arr.max():8.2f}")

        print(stats_line("step()", step_times))
        for k in sorted(render_times.keys()):
            if k.startswith("_"):
                continue
            print(stats_line(k, render_times[k]))
        print(stats_line("TOTAL (step+render)", total_times))

        char_calls = render_times.get("_char_calls", [])
        if char_calls:
            arr = np.array(char_calls)
            print(f"\naddstr calls/frame: mean={arr.mean():.0f}  "
                  f"max={arr.max():.0f}")

        budget_ms = 1000.0 / 30.0
        total_arr = np.array(total_times) * 1000
        over_budget = (total_arr > budget_ms).sum()
        print(f"\n30fps budget: {budget_ms:.1f}ms/frame")
        print(f"Frames over budget: {over_budget}/{n_frames} "
              f"({100 * over_budget / n_frames:.1f}%)")
        print(f"Headroom (mean): {budget_ms - total_arr.mean():.1f}ms")
        return

    # ── cProfile run ───────────────────────────────────────────────
    def profiled_run() -> None:
        for _ in range(n_frames):
            life.step()
            simulate_render_work(life, ng)

    profiler = cProfile.Profile()
    wall_t0 = time.perf_counter()
    profiler.runctx("profiled_run()", globals(), locals())
    wall_dt = time.perf_counter() - wall_t0

    print(f"Wall time: {wall_dt:.2f}s  ({wall_dt / n_frames * 1000:.1f}ms/frame)")
    print(f"Effective FPS: {n_frames / wall_dt:.1f}")
    print()

    # Dump binary if requested
    if dump_path:
        profiler.dump_stats(dump_path)
        print(f"Profile data saved to: {dump_path}")
        print(f"  View with: python3 -m pstats {dump_path}")
        print()

    # Print top functions by cumulative time
    buf = StringIO()
    ps = pstats.Stats(profiler, stream=buf)
    ps.sort_stats("cumulative")
    ps.print_stats(40)
    print(buf.getvalue())

    # Also show by tottime (self-time, no subcalls)
    buf2 = StringIO()
    ps2 = pstats.Stats(profiler, stream=buf2)
    ps2.sort_stats("tottime")
    ps2.print_stats(30)
    print("\n=== By Self-Time (tottime) ===")
    print(buf2.getvalue())


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the Life screensaver")
    parser.add_argument("-n", "--frames", type=int, default=500,
                        help="Number of frames to simulate (default: 500)")
    parser.add_argument("--rows", type=int, default=60,
                        help="Simulated terminal rows (default: 60)")
    parser.add_argument("--cols", type=int, default=200,
                        help="Simulated terminal cols (default: 200)")
    parser.add_argument("--line-timing", action="store_true",
                        help="Per-frame component timing instead of cProfile")
    parser.add_argument("--dump", type=str, default=None,
                        help="Dump cProfile binary to this path")
    args = parser.parse_args()

    run_benchmark(
        n_frames=args.frames,
        term_rows=args.rows,
        term_cols=args.cols,
        line_timing=args.line_timing,
        dump_path=args.dump,
    )


if __name__ == "__main__":
    main()
