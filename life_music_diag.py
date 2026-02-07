#!/usr/bin/env python3
"""Offline diagnostic renderer for the Game of Life music engine.

Renders test scenarios to WAV files and analyzes spectral/temporal
characteristics of each layer for tuning purposes.

Usage:
    python3 life_music_diag.py                         # full run (crafted scenarios)
    python3 life_music_diag.py --scenarios sparse booming
    python3 life_music_diag.py --styles ambient
    python3 life_music_diag.py --no-wav                # report only
    python3 life_music_diag.py --duration 2.0          # shorter renders
    python3 life_music_diag.py --simulate 10           # headless sim for 10 seconds
    python3 life_music_diag.py --replay snapshots.jsonl # replay recorded session
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.io import wavfile as scipy_wavfile
    _HAS_SCIPY_WAV = True
except ImportError:
    _HAS_SCIPY_WAV = False

from life_music import (
    BUFFER_SIZE,
    SAMPLE_RATE,
    STYLES,
    STYLE_AMBIENT,
    STYLE_CHIPTUNE,
    LayeredRender,
    LifeMusicEngine,
    SimulationSnapshot,
)


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

DEFAULT_DURATION: float = 4.0
DEFAULT_OUTPUT_DIR: str = "diag_output"
VIEWPORT_ROWS: int = 80
POP_FLOOR: int = 411

# Frequency band boundaries for spectral analysis (Hz)
BAND_LOW: tuple[float, float] = (20.0, 200.0)
BAND_MID: tuple[float, float] = (200.0, 1000.0)
BAND_HIGH: tuple[float, float] = (1000.0, 4000.0)
BAND_PRESENCE: tuple[float, float] = (4000.0, 10000.0)

BANDS: dict[str, tuple[float, float]] = {
    "low": BAND_LOW,
    "mid": BAND_MID,
    "high": BAND_HIGH,
    "presence": BAND_PRESENCE,
}

LAYER_NAMES: list[str] = ["drone", "melody", "arp", "noise", "mix"]


# ═══════════════════════════════════════════════════════════════════════
#  Scenario Definitions
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ScenarioDefinition:
    """A named simulation scenario for diagnostic rendering."""
    name: str
    description: str
    base_snapshot: SimulationSnapshot
    # Number of alive cells in the playhead column (rest are dead)
    playhead_alive_count: int = 5


def _make_playhead_column(alive_count: int, total: int = VIEWPORT_ROWS,
                          rng: np.random.Generator | None = None) -> tuple[bool, ...]:
    """Generate a playhead column with the specified number of alive cells."""
    col = [False] * total
    if rng is None:
        rng = np.random.default_rng(42)
    if alive_count > 0:
        indices = rng.choice(total, size=min(alive_count, total), replace=False)
        for idx in indices:
            col[int(idx)] = True
    return tuple(col)


SCENARIOS: dict[str, ScenarioDefinition] = {
    "sparse": ScenarioDefinition(
        name="sparse",
        description="Low population, below floor, few melody notes",
        base_snapshot=SimulationSnapshot(
            generation=100,
            population=80,
            pop_floor=POP_FLOOR,
            density=0.02,
            spread=30,
            cycle_period=0,
            mood="sparse",
            epoch="genesis",
            pop_delta=2,
            playhead_column=_make_playhead_column(3),
            playhead_position=0.0,
            viewport_rows=VIEWPORT_ROWS,
        ),
        playhead_alive_count=3,
    ),
    "booming": ScenarioDefinition(
        name="booming",
        description="Population growing rapidly, high melody density",
        base_snapshot=SimulationSnapshot(
            generation=500,
            population=700,
            pop_floor=POP_FLOOR,
            density=0.08,
            spread=200,
            cycle_period=0,
            mood="booming",
            epoch="primordial",
            pop_delta=25,
            playhead_column=_make_playhead_column(14),
            playhead_position=0.0,
            viewport_rows=VIEWPORT_ROWS,
        ),
        playhead_alive_count=14,
    ),
    "declining": ScenarioDefinition(
        name="declining",
        description="Population falling, melancholic",
        base_snapshot=SimulationSnapshot(
            generation=1200,
            population=350,
            pop_floor=POP_FLOOR,
            density=0.04,
            spread=150,
            cycle_period=0,
            mood="declining",
            epoch="emergence",
            pop_delta=-20,
            playhead_column=_make_playhead_column(7),
            playhead_position=0.0,
            viewport_rows=VIEWPORT_ROWS,
        ),
        playhead_alive_count=7,
    ),
    "cycle": ScenarioDefinition(
        name="cycle",
        description="Oscillating population, triggers fast arpeggio",
        base_snapshot=SimulationSnapshot(
            generation=800,
            population=500,
            pop_floor=POP_FLOOR,
            density=0.06,
            spread=100,
            cycle_period=8,
            mood="cycle",
            epoch="expansion",
            pop_delta=5,
            playhead_column=_make_playhead_column(9),
            playhead_position=0.0,
            viewport_rows=VIEWPORT_ROWS,
        ),
        playhead_alive_count=9,
    ),
    "dense": ScenarioDefinition(
        name="dense",
        description="High density, lots of melody notes",
        base_snapshot=SimulationSnapshot(
            generation=2000,
            population=1200,
            pop_floor=POP_FLOOR,
            density=0.18,
            spread=300,
            cycle_period=0,
            mood="dense",
            epoch="flourishing",
            pop_delta=3,
            playhead_column=_make_playhead_column(22),
            playhead_position=0.0,
            viewport_rows=VIEWPORT_ROWS,
        ),
        playhead_alive_count=22,
    ),
    "injection": ScenarioDefinition(
        name="injection",
        description="Sudden population spike, triggers noise bursts",
        base_snapshot=SimulationSnapshot(
            generation=600,
            population=650,
            pop_floor=POP_FLOOR,
            density=0.08,
            spread=250,
            cycle_period=0,
            mood="injection",
            epoch="expansion",
            pop_delta=60,
            playhead_column=_make_playhead_column(11),
            playhead_position=0.0,
            viewport_rows=VIEWPORT_ROWS,
        ),
        playhead_alive_count=11,
    ),
}


# ═══════════════════════════════════════════════════════════════════════
#  Snapshot Sequence Generation
# ═══════════════════════════════════════════════════════════════════════

def generate_snapshot_sequence(
    scenario: ScenarioDefinition,
    duration_secs: float,
    fps: float = 30.0,
) -> list[SimulationSnapshot]:
    """Generate a time-varying sequence of snapshots for realistic rendering.

    Adds natural variation to population, playhead position, and pop_delta
    while keeping mood/epoch/density stable.
    """
    n_frames = max(1, int(duration_secs * fps))
    base = scenario.base_snapshot
    rng = np.random.default_rng(seed=42)
    snapshots: list[SimulationSnapshot] = []

    for i in range(n_frames):
        # Playhead sweeps left-to-right across half the viewport
        playhead_pos = (i / max(1, n_frames)) * 0.5

        # Slight population variation (+-2%)
        pop_jitter = int(rng.normal(0, max(1, base.population * 0.02)))

        # Pop delta varies naturally
        delta_jitter = int(rng.normal(0, 3))

        # Regenerate playhead column with slight cell toggling
        col = list(base.playhead_column)
        for j in range(len(col)):
            if rng.random() < 0.05:  # 5% chance each cell flips
                col[j] = not col[j]

        snapshots.append(SimulationSnapshot(
            generation=base.generation + i,
            population=max(1, base.population + pop_jitter),
            pop_floor=base.pop_floor,
            density=base.density,
            spread=base.spread,
            cycle_period=base.cycle_period,
            mood=base.mood,
            epoch=base.epoch,
            pop_delta=base.pop_delta + delta_jitter,
            playhead_column=tuple(col),
            playhead_position=playhead_pos,
            viewport_rows=base.viewport_rows,
        ))

    return snapshots


# ═══════════════════════════════════════════════════════════════════════
#  Offline Renderer
# ═══════════════════════════════════════════════════════════════════════

class OfflineRenderer:
    """Renders audio offline by driving LifeMusicEngine without PyAudio."""

    def __init__(self, style: str) -> None:
        self.engine = LifeMusicEngine()
        # Force desired style directly (no crossfade)
        try:
            self.engine._style_idx = STYLES.index(style)
        except ValueError:
            self.engine._style_idx = 0
        self.engine._style = style
        self.engine._crossfade_remaining = 0
        # Update voice styles to match
        for v in self.engine._melody_voices:
            v.style = style
        self.engine._arp_voice.style = style

    def render_scenario(
        self,
        scenario: ScenarioDefinition,
        duration_secs: float = DEFAULT_DURATION,
    ) -> dict[str, NDArray[np.float32]]:
        """Render a full scenario, returning per-layer and mix arrays.

        Returns dict with keys: drone, melody, arp, noise, mix
        """
        snapshots = generate_snapshot_sequence(scenario, duration_secs)
        total_samples = int(duration_secs * SAMPLE_RATE)
        total_buffers = (total_samples + BUFFER_SIZE - 1) // BUFFER_SIZE

        # Audio update cadence: update() called every ~1.4 buffers (30fps sim / 43 buffers/sec)
        samples_per_update = SAMPLE_RATE / 30.0  # ~1470 samples
        snap_idx = 0
        samples_since_update = 0.0

        # Accumulate per-layer buffers
        drone_bufs: list[NDArray[np.float32]] = []
        melody_bufs: list[NDArray[np.float32]] = []
        arp_bufs: list[NDArray[np.float32]] = []
        noise_bufs: list[NDArray[np.float32]] = []
        mix_bufs: list[NDArray[np.float32]] = []

        for buf_idx in range(total_buffers):
            # Feed snapshots at simulation frame rate
            if samples_since_update >= samples_per_update or buf_idx == 0:
                if snap_idx < len(snapshots):
                    self.engine.update(snapshots[snap_idx])
                    snap_idx += 1
                samples_since_update = 0.0

            # Render one buffer with per-layer separation
            result: LayeredRender = self.engine.render_buffer_separated(BUFFER_SIZE)

            drone_bufs.append(result.drone)
            melody_bufs.append(result.melody)
            arp_bufs.append(result.arp)
            noise_bufs.append(result.noise)
            mix_bufs.append(result.mix)

            samples_since_update += BUFFER_SIZE

        # Concatenate and trim to exact duration
        return {
            "drone": np.concatenate(drone_bufs)[:total_samples],
            "melody": np.concatenate(melody_bufs)[:total_samples],
            "arp": np.concatenate(arp_bufs)[:total_samples],
            "noise": np.concatenate(noise_bufs)[:total_samples],
            "mix": np.concatenate(mix_bufs)[:total_samples],
        }


# ═══════════════════════════════════════════════════════════════════════
#  Audio Analysis
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class LayerMetrics:
    """Analysis metrics for a single audio layer."""
    rms: float
    rms_db: float
    spectral_centroid_hz: float
    crest_factor: float
    onset_sharpness_ms: float  # -1.0 if not applicable
    band_energy: dict[str, float]  # band name → dB


@dataclass(frozen=True)
class ScenarioMetrics:
    """Full analysis for one scenario x style combination."""
    scenario_name: str
    style: str
    layers: dict[str, LayerMetrics]
    drone_to_highlight_ratio: float
    frequency_masking_score: float


class AudioAnalyzer:
    """Compute diagnostic metrics from rendered audio arrays."""

    def __init__(self, sample_rate: int = SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate

    def analyze_scenario(
        self,
        scenario_name: str,
        style: str,
        layers: dict[str, NDArray[np.float32]],
    ) -> ScenarioMetrics:
        """Analyze all layers for a single scenario × style combination."""
        layer_metrics: dict[str, LayerMetrics] = {}

        for name in LAYER_NAMES:
            signal = layers.get(name)
            if signal is None:
                continue
            measure_onset = (name == "melody")
            layer_metrics[name] = self._analyze_layer(signal, measure_onset)

        # Cross-layer metrics
        drone_rms = layer_metrics.get("drone", LayerMetrics(0, -100, 0, 0, -1, {})).rms
        melody_rms = layer_metrics.get("melody", LayerMetrics(0, -100, 0, 0, -1, {})).rms
        arp_rms = layer_metrics.get("arp", LayerMetrics(0, -100, 0, 0, -1, {})).rms
        highlight_rms = melody_rms + arp_rms

        dh_ratio = drone_rms / max(highlight_rms, 1e-10)

        masking = self._frequency_masking(
            layers.get("drone", np.zeros(1, dtype=np.float32)),
            layers.get("melody", np.zeros(1, dtype=np.float32)),
            layers.get("arp", np.zeros(1, dtype=np.float32)),
        )

        return ScenarioMetrics(
            scenario_name=scenario_name,
            style=style,
            layers=layer_metrics,
            drone_to_highlight_ratio=dh_ratio,
            frequency_masking_score=masking,
        )

    def _analyze_layer(
        self, signal: NDArray[np.float32], measure_onset: bool = False,
    ) -> LayerMetrics:
        """Compute all metrics for a single layer."""
        rms = self._rms(signal)
        rms_db = 20.0 * math.log10(max(rms, 1e-10))
        centroid = self._spectral_centroid(signal)
        crest = self._crest_factor(signal, rms)
        onset = self._onset_sharpness(signal) if measure_onset else -1.0
        bands = self._band_energy(signal)

        return LayerMetrics(
            rms=rms,
            rms_db=rms_db,
            spectral_centroid_hz=centroid,
            crest_factor=crest,
            onset_sharpness_ms=onset,
            band_energy=bands,
        )

    @staticmethod
    def _rms(signal: NDArray[np.float32]) -> float:
        """Root mean square of the signal."""
        if len(signal) == 0:
            return 0.0
        return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))

    def _spectral_centroid(self, signal: NDArray[np.float32]) -> float:
        """Frequency-domain brightness: weighted mean of frequency bins."""
        if len(signal) < 2:
            return 0.0
        windowed = signal * np.hanning(len(signal)).astype(np.float32)
        fft_mag = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)
        total = float(np.sum(fft_mag))
        if total < 1e-10:
            return 0.0
        return float(np.sum(freqs * fft_mag) / total)

    @staticmethod
    def _crest_factor(signal: NDArray[np.float32], rms: float) -> float:
        """Peak / RMS — measures punchiness vs compression."""
        if len(signal) == 0 or rms < 1e-10:
            return 0.0
        peak = float(np.max(np.abs(signal)))
        return peak / rms

    def _onset_sharpness(self, signal: NDArray[np.float32]) -> float:
        """Median time from onset threshold to peak, in milliseconds.

        Returns -1.0 if no clear onsets are detected.
        """
        if len(signal) < 200:
            return -1.0

        # Compute amplitude envelope via short moving average
        window = min(100, len(signal) // 4)
        if window < 2:
            return -1.0
        kernel = np.ones(window, dtype=np.float32) / window
        envelope = np.convolve(np.abs(signal), kernel, mode="same")

        peak_env = float(np.max(envelope))
        if peak_env < 1e-8:
            return -1.0

        threshold = 0.2 * peak_env
        onset_times: list[float] = []

        # Find rising crossings
        above = envelope > threshold
        crossings = np.where(np.diff(above.astype(np.int8)) > 0)[0]

        for cross_idx in crossings:
            # Find the next local peak after this crossing
            search_end = min(cross_idx + int(0.1 * self.sample_rate), len(envelope))
            if search_end <= cross_idx + 1:
                continue
            segment = envelope[cross_idx:search_end]
            local_peak_offset = int(np.argmax(segment))
            if local_peak_offset > 0:
                onset_ms = (local_peak_offset / self.sample_rate) * 1000.0
                onset_times.append(onset_ms)

        if not onset_times:
            return -1.0

        return float(np.median(onset_times))

    def _band_energy(self, signal: NDArray[np.float32]) -> dict[str, float]:
        """RMS energy in frequency bands, reported in dB."""
        if len(signal) < 2:
            return {name: -100.0 for name in BANDS}

        fft_mag = np.abs(np.fft.rfft(signal.astype(np.float64)))
        freqs = np.fft.rfftfreq(len(signal), d=1.0 / self.sample_rate)

        result: dict[str, float] = {}
        for name, (lo, hi) in BANDS.items():
            mask = (freqs >= lo) & (freqs < hi)
            energy = float(np.sum(fft_mag[mask] ** 2))
            result[name] = 10.0 * math.log10(max(energy, 1e-10))

        return result

    def _frequency_masking(
        self,
        drone: NDArray[np.float32],
        melody: NDArray[np.float32],
        arp: NDArray[np.float32],
    ) -> float:
        """Quantify spectral overlap between drone and highlights in mid+high bands.

        Returns 0.0 (no masking) to 1.0 (complete masking).
        """
        min_len = min(len(drone), len(melody), len(arp))
        if min_len < 2:
            return 0.0

        # Combine melody + arp as the "highlight" signal
        highlight = melody[:min_len].astype(np.float64) + arp[:min_len].astype(np.float64)
        drone_sig = drone[:min_len].astype(np.float64)

        drone_fft = np.abs(np.fft.rfft(drone_sig)) ** 2
        highlight_fft = np.abs(np.fft.rfft(highlight)) ** 2
        freqs = np.fft.rfftfreq(min_len, d=1.0 / self.sample_rate)

        # Focus on mid + high bands (where masking matters most)
        mask = (freqs >= 200.0) & (freqs < 4000.0)
        drone_band = drone_fft[mask]
        highlight_band = highlight_fft[mask]

        if len(drone_band) == 0:
            return 0.0

        # Masking score: proportion of bins where drone dominates highlight
        total_bins = len(drone_band)
        if total_bins == 0:
            return 0.0

        # For each bin, compute how much drone exceeds highlight
        drone_dominant = drone_band > highlight_band
        if not np.any(drone_dominant):
            return 0.0

        # Weight by how much louder the drone is in those bins
        ratio = np.where(
            drone_dominant & (highlight_band > 1e-20),
            drone_band / np.maximum(highlight_band, 1e-20),
            0.0,
        )
        # Normalize: masking = fraction of bins dominated, weighted by severity
        score = float(np.mean(np.minimum(ratio, 10.0)) / 10.0)
        return min(1.0, score)


# ═══════════════════════════════════════════════════════════════════════
#  Report Formatting
# ═══════════════════════════════════════════════════════════════════════

def format_report(all_metrics: list[ScenarioMetrics], duration: float = DEFAULT_DURATION) -> str:
    """Format analysis results into a structured text report."""
    lines: list[str] = []
    sep = "=" * 79

    lines.append(sep)
    lines.append("LIFE MUSIC DIAGNOSTIC REPORT")
    lines.append(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"Duration per scenario: {duration}s @ {SAMPLE_RATE} Hz")
    lines.append(sep)
    lines.append("")

    # Group by scenario
    by_scenario: dict[str, list[ScenarioMetrics]] = {}
    for m in all_metrics:
        by_scenario.setdefault(m.scenario_name, []).append(m)

    for scenario_name, metrics_list in by_scenario.items():
        scenario_def = SCENARIOS.get(scenario_name)
        desc = scenario_def.description if scenario_def else ""

        for sm in metrics_list:
            lines.append(f"SCENARIO: {scenario_name} ({sm.style})")
            lines.append(f"  {desc}")
            lines.append("")

            # Main metrics table
            lines.append("  Layer     RMS (dB)  Centroid   Crest   Onset ms")
            lines.append("  " + "-" * 52)

            for layer_name in LAYER_NAMES:
                lm = sm.layers.get(layer_name)
                if lm is None:
                    continue
                onset_str = f"{lm.onset_sharpness_ms:7.1f}" if lm.onset_sharpness_ms >= 0 else "     --"
                lines.append(
                    f"  {layer_name:<8s}  {lm.rms_db:7.1f}   {lm.spectral_centroid_hz:6.0f} Hz"
                    f"  {lm.crest_factor:5.1f}   {onset_str}"
                )

            lines.append("")

            # D/H ratio and masking
            dh_flag = " [!]" if sm.drone_to_highlight_ratio > 2.5 else ""
            lines.append(
                f"  D/H Ratio: {sm.drone_to_highlight_ratio:.2f}{dh_flag}   "
                f"Masking: {sm.frequency_masking_score:.3f}"
            )
            lines.append("")

            # Band energy table
            lines.append("  Band Energy (dB):")
            lines.append("  Layer       Low      Mid     High  Presence")
            lines.append("  " + "-" * 48)

            for layer_name in ["drone", "melody", "arp", "noise"]:
                lm = sm.layers.get(layer_name)
                if lm is None:
                    continue
                be = lm.band_energy
                lines.append(
                    f"  {layer_name:<8s}  {be.get('low', -100):7.1f}  {be.get('mid', -100):7.1f}"
                    f"  {be.get('high', -100):7.1f}  {be.get('presence', -100):8.1f}"
                )

            lines.append("")
            lines.append("")

    # Cross-scenario summary
    lines.append(sep)
    lines.append("CROSS-SCENARIO SUMMARY")
    lines.append(sep)
    lines.append("")

    # Collect all scenario names that appear in the metrics
    seen_scenarios: list[str] = []
    for sm in all_metrics:
        if sm.scenario_name not in seen_scenarios:
            seen_scenarios.append(sm.scenario_name)

    # D/H ratio comparison
    lines.append("Drone-to-Highlight Ratio by scenario:")
    lines.append(f"  {'Scenario':<16s}  {'Chiptune':>10s}  {'Ambient':>10s}")
    lines.append("  " + "-" * 40)

    for scenario_name in seen_scenarios:
        chip_dh = "--"
        amb_dh = "--"
        for sm in all_metrics:
            if sm.scenario_name == scenario_name:
                val = f"{sm.drone_to_highlight_ratio:.2f}"
                flag = " [!]" if sm.drone_to_highlight_ratio > 2.5 else ""
                if sm.style == STYLE_CHIPTUNE:
                    chip_dh = val + flag
                elif sm.style == STYLE_AMBIENT:
                    amb_dh = val + flag
        lines.append(f"  {scenario_name:<16s}  {chip_dh:>10s}  {amb_dh:>10s}")

    lines.append("")

    # Masking comparison
    lines.append("Frequency Masking Score by scenario:")
    lines.append(f"  {'Scenario':<16s}  {'Chiptune':>10s}  {'Ambient':>10s}")
    lines.append("  " + "-" * 40)

    for scenario_name in seen_scenarios:
        chip_mask = "--"
        amb_mask = "--"
        for sm in all_metrics:
            if sm.scenario_name == scenario_name:
                val = f"{sm.frequency_masking_score:.3f}"
                if sm.style == STYLE_CHIPTUNE:
                    chip_mask = val
                elif sm.style == STYLE_AMBIENT:
                    amb_mask = val
        lines.append(f"  {scenario_name:<16s}  {chip_mask:>10s}  {amb_mask:>10s}")

    lines.append("")

    # Spectral centroid comparison (melody layer)
    lines.append("Melody Spectral Centroid (Hz) — brightness comparison:")
    lines.append(f"  {'Scenario':<16s}  {'Chiptune':>10s}  {'Ambient':>10s}")
    lines.append("  " + "-" * 40)

    for scenario_name in seen_scenarios:
        chip_cent = "--"
        amb_cent = "--"
        for sm in all_metrics:
            if sm.scenario_name == scenario_name:
                mel = sm.layers.get("melody")
                if mel:
                    val = f"{mel.spectral_centroid_hz:.0f}"
                    if sm.style == STYLE_CHIPTUNE:
                        chip_cent = val
                    elif sm.style == STYLE_AMBIENT:
                        amb_cent = val
        lines.append(f"  {scenario_name:<16s}  {chip_cent:>10s}  {amb_cent:>10s}")

    lines.append("")

    # RMS comparison (melody layer)
    lines.append("Melody RMS (dB) — loudness comparison:")
    lines.append(f"  {'Scenario':<16s}  {'Chiptune':>10s}  {'Ambient':>10s}")
    lines.append("  " + "-" * 40)

    for scenario_name in seen_scenarios:
        chip_rms = "--"
        amb_rms = "--"
        for sm in all_metrics:
            if sm.scenario_name == scenario_name:
                mel = sm.layers.get("melody")
                if mel:
                    val = f"{mel.rms_db:.1f}"
                    if sm.style == STYLE_CHIPTUNE:
                        chip_rms = val
                    elif sm.style == STYLE_AMBIENT:
                        amb_rms = val
        lines.append(f"  {scenario_name:<16s}  {chip_rms:>10s}  {amb_rms:>10s}")

    lines.append("")
    lines.append("[!] = drone-to-highlight ratio > 2.5 (drone dominance detected)")
    lines.append("")
    lines.append(sep)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
#  Replay from Recorded Snapshots
# ═══════════════════════════════════════════════════════════════════════

def load_recorded_snapshots(path: Path) -> list[SimulationSnapshot]:
    """Load snapshots from a .jsonl file recorded by life.py (d key)."""
    snapshots: list[SimulationSnapshot] = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                # Convert playhead_column from [0,1,...] back to tuple[bool,...]
                col = d.get("playhead_column", [])
                d["playhead_column"] = tuple(bool(x) for x in col)
                snapshots.append(SimulationSnapshot(**d))
            except (json.JSONDecodeError, TypeError) as e:
                print(f"  Warning: skipping line {line_num}: {e}", file=sys.stderr)
    return snapshots


def render_from_snapshots(
    snapshots: list[SimulationSnapshot],
    style: str,
) -> dict[str, NDArray[np.float32]]:
    """Render audio from a sequence of pre-recorded snapshots."""
    if not snapshots:
        empty = np.zeros(0, dtype=np.float32)
        return {name: empty for name in LAYER_NAMES}

    renderer = OfflineRenderer(style)

    # Each snapshot represents one simulation frame (~1/30th second).
    # Render audio buffers at the callback cadence between snapshot updates.
    samples_per_frame = SAMPLE_RATE / 30.0  # ~1470 samples per sim frame
    total_samples = int(len(snapshots) * samples_per_frame)
    total_buffers = (total_samples + BUFFER_SIZE - 1) // BUFFER_SIZE

    snap_idx = 0
    samples_since_update = 0.0

    drone_bufs: list[NDArray[np.float32]] = []
    melody_bufs: list[NDArray[np.float32]] = []
    arp_bufs: list[NDArray[np.float32]] = []
    noise_bufs: list[NDArray[np.float32]] = []
    mix_bufs: list[NDArray[np.float32]] = []

    for buf_idx in range(total_buffers):
        if samples_since_update >= samples_per_frame or buf_idx == 0:
            if snap_idx < len(snapshots):
                renderer.engine.update(snapshots[snap_idx])
                snap_idx += 1
            samples_since_update = 0.0

        result: LayeredRender = renderer.engine.render_buffer_separated(BUFFER_SIZE)
        drone_bufs.append(result.drone)
        melody_bufs.append(result.melody)
        arp_bufs.append(result.arp)
        noise_bufs.append(result.noise)
        mix_bufs.append(result.mix)
        samples_since_update += BUFFER_SIZE

    return {
        "drone": np.concatenate(drone_bufs)[:total_samples],
        "melody": np.concatenate(melody_bufs)[:total_samples],
        "arp": np.concatenate(arp_bufs)[:total_samples],
        "noise": np.concatenate(noise_bufs)[:total_samples],
        "mix": np.concatenate(mix_bufs)[:total_samples],
    }


# ═══════════════════════════════════════════════════════════════════════
#  Headless Simulation
# ═══════════════════════════════════════════════════════════════════════

def run_headless_simulation(
    duration_secs: float,
    term_rows: int = 40,
    term_cols: int = 120,
) -> list[SimulationSnapshot]:
    """Run InfiniteLife headlessly and extract real snapshots.

    Returns a list of SimulationSnapshot at ~30fps, just like the live engine.
    """
    # Import InfiniteLife from life.py (avoid circular import at module level)
    try:
        from life import InfiniteLife
    except ImportError:
        print("Error: cannot import InfiniteLife from life.py", file=sys.stderr)
        sys.exit(1)

    life = InfiniteLife(term_rows, term_cols)
    fps = 30.0
    n_frames = int(duration_secs * fps)
    snapshots: list[SimulationSnapshot] = []

    playhead_col_idx = 0

    for frame in range(n_frames):
        # Step the simulation
        life.step()

        # Build snapshot exactly like the main loop in life.py
        # (music updates happen every 2nd frame in real code, but for
        # diagnostics we capture every frame for maximum fidelity)
        disp = life.display_grid()

        # Advance playhead
        view_w = life.view_w
        cols_per_frame = max(1, view_w // 240)
        playhead_col_idx = (playhead_col_idx + cols_per_frame) % max(1, view_w)
        playhead_pos = playhead_col_idx / max(1, view_w)

        # Extract playhead column
        col_data: tuple[bool, ...] = ()
        if 0 <= playhead_col_idx < disp.shape[1]:
            col_data = tuple(disp[:, playhead_col_idx].astype(bool))

        # Population delta
        ph_len = len(life.pop_history)
        pop_delta = (life.pop_history[-1] - life.pop_history[-2]) if ph_len >= 2 else 0

        # Density
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
        snapshots.append(snap)

    return snapshots


# ═══════════════════════════════════════════════════════════════════════
#  WAV Output
# ═══════════════════════════════════════════════════════════════════════

def write_wavs(
    output_dir: Path,
    scenario_name: str,
    style: str,
    layers: dict[str, NDArray[np.float32]],
) -> list[Path]:
    """Write per-layer and mix WAVs. Returns list of written paths."""
    if not _HAS_SCIPY_WAV:
        print("  [skip WAV] scipy.io.wavfile not available", file=sys.stderr)
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for layer_name, audio in layers.items():
        filename = f"{scenario_name}_{style}_{layer_name}.wav"
        path = output_dir / filename

        # Normalize to prevent clipping
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-8:
            normalized = (audio / peak * 0.95).astype(np.float32)
        else:
            normalized = audio.astype(np.float32)

        try:
            scipy_wavfile.write(str(path), SAMPLE_RATE, normalized)
            written.append(path)
        except Exception as e:
            print(f"  [error] Failed to write {path}: {e}", file=sys.stderr)

    return written


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """Run the full diagnostic pipeline."""
    parser = argparse.ArgumentParser(
        description="Life music engine diagnostic renderer and analyzer",
    )
    parser.add_argument(
        "--scenarios", nargs="*", default=None,
        help=f"Specific scenarios to run (default: all). Choices: {', '.join(SCENARIOS.keys())}",
    )
    parser.add_argument(
        "--styles", nargs="*", default=None,
        help=f"Specific styles to run (default: all). Choices: {', '.join(STYLES)}",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(DEFAULT_OUTPUT_DIR),
        help=f"Directory for WAV output (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-wav", action="store_true",
        help="Skip WAV output, only print report",
    )
    parser.add_argument(
        "--duration", type=float, default=DEFAULT_DURATION,
        help=f"Seconds per scenario (default: {DEFAULT_DURATION})",
    )
    parser.add_argument(
        "--simulate", type=float, default=None, metavar="SECONDS",
        help="Run headless InfiniteLife simulation for N seconds and analyze the real output",
    )
    parser.add_argument(
        "--replay", type=Path, default=None, metavar="FILE",
        help="Replay recorded snapshots from a .jsonl file (use 'd' in life.py to record)",
    )
    args = parser.parse_args()

    styles: list[str] = args.styles if args.styles else list(STYLES)
    for style in styles:
        if style not in STYLES:
            print(f"Error: unknown style '{style}'. Choose from: {', '.join(STYLES)}",
                  file=sys.stderr)
            sys.exit(1)

    duration = args.duration
    analyzer = AudioAnalyzer()
    all_metrics: list[ScenarioMetrics] = []

    report_duration = duration
    has_extra_sources = False  # replay/simulate were added

    # ── Replay recorded snapshots ────────────────────────────────
    if args.replay is not None:
        if not args.replay.exists():
            print(f"Error: file not found: {args.replay}", file=sys.stderr)
            sys.exit(1)

        print(f"Loading snapshots from {args.replay}...", end="", flush=True)
        snapshots = load_recorded_snapshots(args.replay)
        snap_duration = len(snapshots) / 30.0
        print(f" {len(snapshots)} snapshots ({snap_duration:.1f}s)")
        report_duration = snap_duration

        for style in styles:
            label = f"replay ({style})"
            print(f"  Rendering: {label}...", end="", flush=True)
            layers = render_from_snapshots(snapshots, style)
            print(" analyzing...", end="", flush=True)
            metrics = analyzer.analyze_scenario("replay", style, layers)
            all_metrics.append(metrics)
            if not args.no_wav:
                written = write_wavs(args.output_dir, "replay", style, layers)
                print(f" wrote {len(written)} WAVs.", flush=True)
            else:
                print(" done.", flush=True)
        has_extra_sources = True

    # ── Headless simulation ──────────────────────────────────────
    if args.simulate is not None:
        sim_duration = args.simulate
        print(f"Running headless simulation for {sim_duration:.1f}s...", end="", flush=True)
        snapshots = run_headless_simulation(sim_duration)
        print(f" {len(snapshots)} frames captured.")
        report_duration = max(report_duration, sim_duration)

        for style in styles:
            label = f"simulate ({style})"
            print(f"  Rendering: {label}...", end="", flush=True)
            layers = render_from_snapshots(snapshots, style)
            print(" analyzing...", end="", flush=True)
            metrics = analyzer.analyze_scenario("simulate", style, layers)
            all_metrics.append(metrics)
            if not args.no_wav:
                written = write_wavs(args.output_dir, "simulate", style, layers)
                print(f" wrote {len(written)} WAVs.", flush=True)
            else:
                print(" done.", flush=True)
        has_extra_sources = True

    # ── Crafted scenarios ────────────────────────────────────────
    # Run crafted scenarios unless user only asked for replay/simulate
    # (i.e., --scenarios was not explicitly provided and we have extra sources)
    run_crafted = args.scenarios is not None or not has_extra_sources
    if run_crafted:
        scenario_names: list[str] = args.scenarios if args.scenarios else list(SCENARIOS.keys())
        for name in scenario_names:
            if name not in SCENARIOS:
                print(f"Error: unknown scenario '{name}'. Choose from: {', '.join(SCENARIOS.keys())}",
                      file=sys.stderr)
                sys.exit(1)

        for scenario_name in scenario_names:
            scenario = SCENARIOS[scenario_name]

            for style in styles:
                print(f"  Rendering: {scenario_name} ({style})...",
                      end="", flush=True)

                renderer = OfflineRenderer(style)
                layers = renderer.render_scenario(scenario, duration_secs=duration)
                print(" analyzing...", end="", flush=True)

                metrics = analyzer.analyze_scenario(scenario_name, style, layers)
                all_metrics.append(metrics)

                if not args.no_wav:
                    written = write_wavs(args.output_dir, scenario_name, style, layers)
                    print(f" wrote {len(written)} WAVs.", flush=True)
                else:
                    print(" done.", flush=True)

    # Print report
    print()
    report = format_report(all_metrics, duration=report_duration)
    print(report)

    # Also write report to file
    if not args.no_wav:
        report_path = args.output_dir / "analysis_report.txt"
        try:
            report_path.write_text(report)
            print(f"\nReport saved to: {report_path}")
        except Exception as e:
            print(f"\nFailed to save report: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
