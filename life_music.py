"""
Generative music engine for the Life screensaver.

Produces an ethereal, continuously-evolving soundscape where the music
emerges from the simulation state itself. Supports two switchable styles:
pure chiptune (NES) and ambient synth.

Architecture:
  Main thread builds a frozen SimulationSnapshot each frame, swaps it
  atomically (GIL guarantees). The PyAudio callback reads the latest
  snapshot and renders 4 voice layers: drone, melody, arpeggio, noise.

Audio: 44100 Hz, mono, float32, 1024 frames/buffer (~23ms latency).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

try:
    from scipy.signal import lfilter
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

try:
    import pyaudio
    _HAS_PYAUDIO = True
except ImportError:
    pyaudio = None  # type: ignore[assignment]
    _HAS_PYAUDIO = False


# ═══════════════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════════════

SAMPLE_RATE: int = 44100
BUFFER_SIZE: int = 2048
TWO_PI: float = 2.0 * math.pi

# Styles
STYLE_CHIPTUNE: str = "chiptune"
STYLE_AMBIENT: str = "ambient"
STYLES: list[str] = [STYLE_CHIPTUNE, STYLE_AMBIENT]

# ── Scales (semitone intervals from root) ──────────────────────────────
SCALES: dict[str, tuple[int, ...]] = {
    "lydian":           (0, 2, 4, 6, 7, 9, 11),
    "aeolian":          (0, 2, 3, 5, 7, 8, 10),
    "whole_tone":       (0, 2, 4, 6, 8, 10),
    "dorian":           (0, 2, 3, 5, 7, 9, 10),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "mixolydian":       (0, 2, 4, 5, 7, 9, 10),
    "major_pentatonic": (0, 2, 4, 7, 9),
}

# Mood → scale name
MOOD_SCALE: dict[str, str] = {
    "booming":   "lydian",
    "declining": "aeolian",
    "cycle":     "whole_tone",
    "stagnant":  "dorian",
    "sparse":    "minor_pentatonic",
    "dense":     "mixolydian",
    "injection": "major_pentatonic",
}
DEFAULT_SCALE: str = "major_pentatonic"

# Epoch → MIDI note number for root
EPOCH_ROOT: dict[str, int] = {
    "genesis":    48,   # C3
    "primordial": 46,   # Bb2
    "emergence":  50,   # D3
    "expansion":  53,   # F3
    "flourishing": 55,  # G3
    "deep time":  51,   # Eb3
    "eon":        48,   # C3
    "eternity":   45,   # A2
}
DEFAULT_ROOT: int = 48  # C3

# ADSR presets (attack, decay, sustain_level, release) in seconds
ADSR_CHIPTUNE: tuple[float, float, float, float] = (0.005, 0.05, 0.7, 0.02)
ADSR_AMBIENT: tuple[float, float, float, float] = (0.06, 0.25, 0.75, 0.5)

# Maximum polyphony for melody playhead
MAX_MELODY_VOICES: int = 4

# Playhead sweep duration in seconds (one full L→R scan)
PLAYHEAD_SWEEP_SECS: float = 8.0


# ═══════════════════════════════════════════════════════════════════════
#  Data transfer: frozen snapshot from sim → audio
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SimulationSnapshot:
    """Immutable state transfer from simulation to audio engine."""
    generation: int = 0
    population: int = 0
    pop_floor: int = 100
    density: float = 0.0
    spread: int = 0
    cycle_period: int = 0
    mood: str = ""
    epoch: str = "genesis"
    pop_delta: int = 0
    playhead_column: tuple[bool, ...] = ()
    playhead_position: float = 0.0
    viewport_rows: int = 40


# ═══════════════════════════════════════════════════════════════════════
#  Derived musical state
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MusicalState:
    """Derived musical parameters, recomputed each frame in update()."""
    root_midi: int = DEFAULT_ROOT
    target_root_midi: int = DEFAULT_ROOT
    scale_name: str = DEFAULT_SCALE
    scale_intervals: tuple[int, ...] = SCALES[DEFAULT_SCALE]
    prev_scale_intervals: tuple[int, ...] = SCALES[DEFAULT_SCALE]
    scale_crossfade: float = 1.0  # 0=old scale, 1=new scale
    tempo_bpm: float = 80.0

    # Per-layer target volumes (0.0-1.0)
    drone_volume: float = 0.3
    melody_volume: float = 0.2
    arp_volume: float = 0.15
    noise_volume: float = 0.05

    # Arpeggio
    arp_speed: float = 4.0  # notes per second
    arp_pattern: tuple[int, ...] = (0, 2, 4, 7)  # scale degree offsets

    # Melody playhead notes (Hz values for active notes)
    playhead_notes: tuple[float, ...] = ()


@dataclass(frozen=True)
class LayeredRender:
    """Per-layer audio arrays from a single render pass (diagnostic use).

    Each layer array has volume scaling applied, matching real-time behavior.
    """
    drone: NDArray[np.float32]
    melody: NDArray[np.float32]
    arp: NDArray[np.float32]
    noise: NDArray[np.float32]
    mix: NDArray[np.float32]
    drone_vol: float
    melody_vol: float
    arp_vol: float
    noise_vol: float


# ═══════════════════════════════════════════════════════════════════════
#  Utility functions
# ═══════════════════════════════════════════════════════════════════════

def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def scale_pitches(root_midi: int, intervals: tuple[int, ...], octaves: int = 2) -> list[float]:
    """Build a list of Hz pitches spanning the given number of octaves."""
    pitches: list[float] = []
    for octave in range(octaves):
        for interval in intervals:
            pitches.append(midi_to_hz(root_midi + interval + 12 * octave))
    return pitches


def soft_clip(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Soft clipping (tanh-based) to prevent harsh digital distortion.

    Operates in-place to avoid allocations on the audio callback thread.
    """
    np.tanh(x, out=x)
    return x


def lerp(a: float, b: float, t: float) -> float:
    """Linear interpolation from a to b by factor t (0.0-1.0)."""
    return a + (b - a) * max(0.0, min(1.0, t))


class CachedLPF:
    """One-pole low-pass filter with cached coefficients and persistent state.

    Avoids recomputing filter coefficients every call, and carries filter
    state (zi) across audio buffers for smooth continuous filtering.
    """

    def __init__(self, cutoff_hz: float, sample_rate: int = SAMPLE_RATE) -> None:
        self._cutoff = cutoff_hz
        rc = 1.0 / (TWO_PI * cutoff_hz)
        dt = 1.0 / sample_rate
        alpha = dt / (rc + dt)
        self._b = np.array([alpha], dtype=np.float64)
        self._a = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
        self._alpha = float(alpha)
        self._zi = np.zeros(1, dtype=np.float64)

    def apply(self, signal: NDArray[np.float32]) -> NDArray[np.float32]:
        """Filter signal in-place, carrying state across calls."""
        if _HAS_SCIPY:
            out, self._zi = lfilter(self._b, self._a,
                                    signal.astype(np.float64), zi=self._zi)
            return out.astype(np.float32)
        # Manual fallback
        out = np.empty_like(signal)
        prev = float(self._zi[0]) if len(self._zi) > 0 else 0.0
        a = self._alpha
        for i in range(len(signal)):
            prev = a * float(signal[i]) + (1.0 - a) * prev
            out[i] = prev
        self._zi[0] = prev
        return out

    def reset(self) -> None:
        """Reset filter state (e.g., on style change)."""
        self._zi[:] = 0.0


def one_pole_lp(signal: NDArray[np.float32], cutoff_hz: float,
                sample_rate: int = SAMPLE_RATE) -> NDArray[np.float32]:
    """One-pole low-pass filter (stateless, for one-off use)."""
    if cutoff_hz >= sample_rate / 2:
        return signal
    rc = 1.0 / (TWO_PI * cutoff_hz)
    dt = 1.0 / sample_rate
    alpha = dt / (rc + dt)
    b_coeff = np.array([alpha], dtype=np.float64)
    a_coeff = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    if _HAS_SCIPY:
        result = lfilter(b_coeff, a_coeff, signal.astype(np.float64))
        return result.astype(np.float32)
    # Manual fallback
    out = np.empty_like(signal)
    prev = 0.0
    a_f = float(alpha)
    for i in range(len(signal)):
        prev = a_f * float(signal[i]) + (1.0 - a_f) * prev
        out[i] = prev
    return out


# ═══════════════════════════════════════════════════════════════════════
#  Oscillator primitives (vectorized numpy)
# ═══════════════════════════════════════════════════════════════════════

def _sine_wave(phase: NDArray[np.float64]) -> NDArray[np.float32]:
    """Pure sine wave from phase array (in radians)."""
    return np.sin(phase).astype(np.float32)


def _triangle_wave(phase: NDArray[np.float64]) -> NDArray[np.float32]:
    """Triangle wave from phase array (in radians)."""
    # Normalize phase to [0, 2*pi)
    p = np.mod(phase, TWO_PI) / TWO_PI
    return (2.0 * np.abs(2.0 * p - 1.0) - 1.0).astype(np.float32)


def _square_wave(phase: NDArray[np.float64], duty: float = 0.5,
                  freq_hz: float = 0.0) -> NDArray[np.float32]:
    """Square wave with polyBLEP antialiasing."""
    p = np.mod(phase, TWO_PI) / TWO_PI
    raw = np.where(p < duty, 1.0, -1.0).astype(np.float64)

    # PolyBLEP correction at transitions for antialiasing
    # Use freq/sample_rate directly instead of expensive np.diff+np.mean
    dt = freq_hz / SAMPLE_RATE if freq_hz > 0.0 else (np.mean(np.diff(p)) if len(p) > 1 else 0.01)
    if dt > 0:
        # Transition at p=0 (rising edge)
        mask_rise = p < dt
        raw[mask_rise] += _polyblep(p[mask_rise] / dt)
        mask_rise_wrap = p > 1.0 - dt
        raw[mask_rise_wrap] += _polyblep((p[mask_rise_wrap] - 1.0) / dt)

        # Transition at p=duty (falling edge)
        t2 = (p - duty) / dt
        mask_fall = np.abs(t2) < 1.0
        raw[mask_fall] -= _polyblep(t2[mask_fall])

    return raw.astype(np.float32)


def _polyblep(t: NDArray[np.float64] | float) -> NDArray[np.float64] | float:
    """PolyBLEP residual for antialiased square wave edges."""
    t = np.asarray(t, dtype=np.float64)
    result = np.zeros_like(t)
    mask1 = (t >= 0) & (t < 1)
    mask2 = (t >= -1) & (t < 0)
    result[mask1] = 2.0 * t[mask1] - t[mask1] * t[mask1] - 1.0
    result[mask2] = t[mask2] * t[mask2] + 2.0 * t[mask2] + 1.0
    return result


# ═══════════════════════════════════════════════════════════════════════
#  ADSR Envelope
# ═══════════════════════════════════════════════════════════════════════

class ADSREnvelope:
    """Simple ADSR envelope generator."""

    def __init__(self, attack: float, decay: float,
                 sustain: float, release: float) -> None:
        self.attack = max(0.001, attack)
        self.decay = max(0.001, decay)
        self.sustain = sustain
        self.release = max(0.001, release)

    def generate(self, n_samples: int, gate: bool, elapsed: float,
                 release_time: float = -1.0,
                 out: NDArray[np.float64] | None = None) -> NDArray[np.float32]:
        """Generate envelope shape for n_samples starting at elapsed time."""
        dt = n_samples / SAMPLE_RATE
        t = np.linspace(elapsed, elapsed + dt,
                        n_samples, endpoint=False, dtype=np.float64)
        if out is not None and len(out) >= n_samples:
            env = out[:n_samples]
            env[:] = 0.0
        else:
            env = np.zeros(n_samples, dtype=np.float64)

        if gate:
            # Attack phase
            attack_mask = t < self.attack
            env[attack_mask] = t[attack_mask] / self.attack

            # Decay phase
            decay_mask = (t >= self.attack) & (t < self.attack + self.decay)
            decay_t = (t[decay_mask] - self.attack) / self.decay
            env[decay_mask] = 1.0 - (1.0 - self.sustain) * decay_t

            # Sustain phase
            sustain_mask = t >= self.attack + self.decay
            env[sustain_mask] = self.sustain
        else:
            if release_time >= 0:
                rel_elapsed = t - release_time
                env = np.where(
                    rel_elapsed < 0, self.sustain,
                    self.sustain * np.maximum(0.0, 1.0 - rel_elapsed / self.release)
                )
            # else: stay at zero

        return env.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════
#  Voice: a single pitched sound with envelope
# ═══════════════════════════════════════════════════════════════════════

class Voice:
    """A single oscillator voice with phase accumulator and envelope."""

    def __init__(self, freq: float = 440.0, style: str = STYLE_AMBIENT) -> None:
        self.freq: float = freq
        self.target_freq: float = freq
        self.style: str = style
        self.phase: float = 0.0
        self.phase_det: float = 0.0    # Detuned chorus oscillator phase
        self.active: bool = False
        self.gate: bool = False
        self.elapsed: float = 0.0
        self.release_time: float = -1.0
        # Pre-allocated envelope buffer (reused each render call)
        self._env_buf: NDArray[np.float64] = np.empty(BUFFER_SIZE, dtype=np.float64)

    def note_on(self, freq: float) -> None:
        """Start a new note — full ADSR restart."""
        self.target_freq = freq
        self.freq = freq
        self.gate = True
        self.active = True
        self.elapsed = 0.0
        self.release_time = -1.0

    def retrigger(self, freq: float) -> None:
        """Slide to a new pitch without restarting the ADSR envelope.

        Use this when a voice is already active and the pitch change is
        small enough that a smooth portamento sounds better than a hard
        retrigger (avoids 30 Hz envelope flutter from per-frame updates).
        """
        self.target_freq = freq
        if not self.active:
            self.note_on(freq)
        elif not self.gate:
            # Voice was releasing — re-gate it without resetting elapsed
            self.gate = True
            self.release_time = -1.0

    def note_off(self) -> None:
        self.gate = False
        self.release_time = self.elapsed

    def render(self, n_samples: int) -> NDArray[np.float32]:
        """Render n_samples of audio for this voice."""
        if not self.active:
            return np.zeros(n_samples, dtype=np.float32)

        dt = 1.0 / SAMPLE_RATE

        # Per-sample frequency ramp for buffer-size-independent portamento
        # Time constant in seconds: ambient slides slowly, chiptune snaps fast
        slide_tc = 0.15 if self.style == STYLE_AMBIENT else 0.02
        alpha = 1.0 - math.exp(-dt / slide_tc)

        # Vectorized exponential decay for frequency ramp
        # f[n] = target + (start - target) * (1 - alpha)^n
        decay = (1.0 - alpha) ** np.arange(n_samples, dtype=np.float64)
        freqs = self.target_freq + (self.freq - self.target_freq) * decay
        self.freq = float(freqs[-1])

        # Phase accumulation from per-sample frequencies
        phase_incs = TWO_PI * freqs * dt
        phases = self.phase + np.cumsum(phase_incs)
        self.phase = float(np.mod(phases[-1], TWO_PI)) if n_samples > 0 else self.phase

        # Oscillator
        if self.style == STYLE_CHIPTUNE:
            osc = _square_wave(phases, duty=0.25, freq_hz=self.freq)
            adsr = ADSREnvelope(*ADSR_CHIPTUNE)
        else:
            # Ambient: sine + chorus with dedicated detuned phase accumulator
            # (avoids micro-discontinuities from `phases * factor` at phase wraps)
            phase_incs_det = TWO_PI * freqs * 1.003 * dt
            phases_det = self.phase_det + np.cumsum(phase_incs_det)
            self.phase_det = float(np.mod(phases_det[-1], TWO_PI)) if n_samples > 0 else self.phase_det
            osc = _sine_wave(phases) * 0.7
            osc += _sine_wave(phases_det) * 0.3  # chorus (own phase)
            adsr = ADSREnvelope(*ADSR_AMBIENT)

        env = adsr.generate(n_samples, self.gate, self.elapsed, self.release_time,
                            out=self._env_buf)
        self.elapsed += n_samples * dt

        # Check if voice has fully released
        if not self.gate and self.release_time >= 0:
            release_dur = adsr.release
            if self.elapsed - self.release_time > release_dur:
                self.active = False

        return osc * env


# ═══════════════════════════════════════════════════════════════════════
#  The Music Engine
# ═══════════════════════════════════════════════════════════════════════

class LifeMusicEngine:
    """
    Generative music engine driven by Game of Life simulation state.

    Call start() to begin audio output, update(snapshot) each frame
    from the main thread, and stop() on shutdown.
    """

    def __init__(self) -> None:
        # Public state
        self._muted: bool = False
        self._master_volume: float = 0.5
        self._style_idx: int = 1  # Start with ambient
        self._style: str = STYLES[self._style_idx]

        # Musical state (read by audio callback, written by update())
        self._musical: MusicalState = MusicalState()
        self._snapshot: SimulationSnapshot = SimulationSnapshot()

        # Audio engine
        self._pa: pyaudio.PyAudio | None = None  # type: ignore[name-defined]
        self._stream: pyaudio.Stream | None = None  # type: ignore[name-defined]
        self._running: bool = False

        # Phase accumulators for drone layer
        self._drone_phase1: float = 0.0
        self._drone_phase1_det: float = 0.0   # Detuned chorus oscillator
        self._drone_phase2: float = 0.0
        self._drone_phase2_det: float = 0.0   # Detuned fifth oscillator
        self._drone_phase_sub: float = 0.0
        self._drone_lfo_phase: float = 0.0

        # Melody voices
        self._melody_voices: list[Voice] = [Voice(style=self._style) for _ in range(MAX_MELODY_VOICES)]

        # Arpeggio state
        self._arp_phase: float = 0.0
        self._arp_voice: Voice = Voice(style=self._style)
        self._arp_note_idx: int = 0
        self._arp_samples_in_note: int = 0  # discrete counter, no float drift

        # Noise state: triggered burst system
        self._noise_level: float = 0.0
        self._noise_burst_amp: float = 0.0       # current burst amplitude
        self._noise_burst_phase: float = 1.0     # 0=attack start, 1=idle
        self._noise_burst_duration: float = 0.15  # seconds per burst
        self._noise_burst_threshold: int = 20    # min pop_delta to trigger
        self._noise_burst_cooldown: float = 0.0  # seconds remaining before next burst allowed

        # Smoothed volumes (to avoid clicks)
        self._smooth_drone_vol: float = 0.3
        self._smooth_melody_vol: float = 0.0
        self._smooth_arp_vol: float = 0.0
        self._smooth_noise_vol: float = 0.0

        # Root frequency portamento
        self._current_root_hz: float = midi_to_hz(DEFAULT_ROOT)
        self._target_root_hz: float = midi_to_hz(DEFAULT_ROOT)

        # Frame counter for buffer-level operations
        self._buffer_count: int = 0
        self._underrun_count: int = 0

        # Crossfade buffer for anti-click when switching styles
        self._crossfade_remaining: int = 0
        self._crossfade_total: int = SAMPLE_RATE // 4  # 250ms crossfade
        self._old_style: str = self._style

        # Pre-allocated reusable buffers for the audio callback
        self._buf_n: NDArray[np.float64] = np.arange(BUFFER_SIZE, dtype=np.float64)
        self._buf_t: NDArray[np.float64] = self._buf_n / SAMPLE_RATE  # time offsets
        self._noise_pool: NDArray[np.float32] = np.random.randn(SAMPLE_RATE).astype(np.float32)
        self._noise_pool_idx: int = 0

        # Cached low-pass filters (pre-computed coefficients, persistent state)
        self._lpf_drone: CachedLPF = CachedLPF(800.0)
        self._lpf_melody: CachedLPF = CachedLPF(3500.0)
        self._lpf_arp: CachedLPF = CachedLPF(4500.0)
        self._lpf_noise: CachedLPF = CachedLPF(600.0)

    # ── Public properties ──────────────────────────────────────────────

    @property
    def is_muted(self) -> bool:
        return self._muted

    @property
    def style(self) -> str:
        return self._style

    @property
    def master_volume(self) -> float:
        return self._master_volume

    @property
    def volume_percent(self) -> int:
        return round(self._master_volume * 100)

    # ── Controls ───────────────────────────────────────────────────────

    def toggle_mute(self) -> None:
        self._muted = not self._muted

    def adjust_volume(self, delta: float) -> None:
        self._master_volume = max(0.0, min(1.0, self._master_volume + delta))

    def cycle_style(self) -> None:
        self._old_style = self._style
        self._style_idx = (self._style_idx + 1) % len(STYLES)
        self._style = STYLES[self._style_idx]
        self._crossfade_remaining = self._crossfade_total
        # Update voice styles
        for v in self._melody_voices:
            v.style = self._style
        self._arp_voice.style = self._style

    # ── Lifecycle ──────────────────────────────────────────────────────

    def start(self) -> bool:
        """Start audio output. Returns True on success, False on failure."""
        if not _HAS_PYAUDIO:
            return False

        try:
            self._pa = pyaudio.PyAudio()
            self._stream = self._pa.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=BUFFER_SIZE,
                stream_callback=self._audio_callback,
            )
            self._stream.start_stream()
            self._running = True
            return True
        except Exception:
            self._cleanup_audio()
            return False

    def stop(self) -> None:
        """Stop audio output and clean up resources."""
        self._running = False
        self._cleanup_audio()

    def _cleanup_audio(self) -> None:
        """Safely tear down PyAudio resources."""
        try:
            if self._stream is not None:
                if self._stream.is_active():
                    self._stream.stop_stream()
                self._stream.close()
        except Exception:
            pass
        self._stream = None
        try:
            if self._pa is not None:
                self._pa.terminate()
        except Exception:
            pass
        self._pa = None

    # ── State update (called from main thread each frame) ──────────────

    def update(self, snapshot: SimulationSnapshot) -> None:
        """Map simulation state to musical parameters. Main thread only."""
        self._snapshot = snapshot
        ms = self._musical

        # ── Root note from epoch ──
        ms.target_root_midi = EPOCH_ROOT.get(snapshot.epoch, DEFAULT_ROOT)
        # Smooth portamento for root changes
        self._target_root_hz = midi_to_hz(ms.target_root_midi)

        # ── Scale from mood ──
        new_scale_name = MOOD_SCALE.get(snapshot.mood, DEFAULT_SCALE)
        if new_scale_name != ms.scale_name:
            ms.prev_scale_intervals = ms.scale_intervals
            ms.scale_name = new_scale_name
            ms.scale_intervals = SCALES[new_scale_name]
            ms.scale_crossfade = 0.0  # Start crossfade

        # Advance crossfade (over ~2 seconds at ~30fps ≈ 60 frames)
        if ms.scale_crossfade < 1.0:
            ms.scale_crossfade = min(1.0, ms.scale_crossfade + 0.017)

        # ── Drone volume: proportional to population ──
        if snapshot.pop_floor > 0:
            pop_ratio = min(2.0, snapshot.population / max(1, snapshot.pop_floor))
        else:
            pop_ratio = 0.5
        ms.drone_volume = lerp(0.13, 0.38, min(1.0, pop_ratio))

        # ── Melody volume: moderate, scales with density ──
        ms.melody_volume = lerp(0.08, 0.35, min(1.0, snapshot.density * 10.0))

        # ── Arpeggio: speed from cycle detection, volume from spread ──
        if snapshot.cycle_period > 0:
            # Faster arpeggios when cycles are short
            ms.arp_speed = lerp(3.0, 12.0, min(1.0, 10.0 / max(1, snapshot.cycle_period)))
            ms.arp_volume = 0.2
        else:
            ms.arp_speed = lerp(2.0, 6.0, min(1.0, snapshot.spread / 200.0))
            ms.arp_volume = lerp(0.05, 0.15, min(1.0, snapshot.spread / 100.0))

        # Build arpeggio pattern from scale
        if len(ms.scale_intervals) >= 3:
            ms.arp_pattern = (
                ms.scale_intervals[0],
                ms.scale_intervals[min(2, len(ms.scale_intervals) - 1)],
                ms.scale_intervals[min(4, len(ms.scale_intervals) - 1)],
                ms.scale_intervals[0] + 12,
            )
        else:
            ms.arp_pattern = (0, 4, 7, 12)

        # ── Noise: driven by population delta (births/deaths) ──
        abs_delta = abs(snapshot.pop_delta)
        ms.noise_volume = lerp(0.0, 0.15, min(1.0, abs_delta / 50.0))

        # ── Melody playhead notes ──
        self._compute_playhead_notes(snapshot, ms)

        # ── Tempo ──
        ms.tempo_bpm = lerp(60.0, 120.0, min(1.0, snapshot.density * 8.0))

    def _compute_playhead_notes(self, snap: SimulationSnapshot,
                                ms: MusicalState) -> None:
        """Map alive cells in the playhead column to pitches."""
        col = snap.playhead_column
        if not col or snap.viewport_rows < 2:
            ms.playhead_notes = ()
            return

        n_rows = len(col)
        # Build pitch range: 2 octaves of current scale
        root = ms.target_root_midi
        pitches = scale_pitches(root, ms.scale_intervals, octaves=2)
        if not pitches:
            ms.playhead_notes = ()
            return

        # Find alive rows
        alive_rows: list[int] = [i for i, alive in enumerate(col) if alive]

        # Density gate: skip if too sparse and overall density is low
        if len(alive_rows) < 2 and snap.density < 0.03:
            ms.playhead_notes = ()
            return

        # Map row index → pitch (top = high, bottom = low)
        notes: list[float] = []
        for row in alive_rows:
            # Invert: row 0 (top) → highest pitch
            frac = 1.0 - (row / max(1, n_rows - 1))
            pitch_idx = int(frac * (len(pitches) - 1))
            pitch_idx = max(0, min(pitch_idx, len(pitches) - 1))
            notes.append(pitches[pitch_idx])

        # Cap polyphony: pick most spread-out if too many
        if len(notes) > MAX_MELODY_VOICES:
            notes.sort()
            step = len(notes) / MAX_MELODY_VOICES
            notes = [notes[int(i * step)] for i in range(MAX_MELODY_VOICES)]

        ms.playhead_notes = tuple(notes)

    # ── Audio callback (runs on PyAudio thread) ────────────────────────

    def _audio_callback(
        self,
        in_data: bytes | None,
        frame_count: int,
        time_info: dict[str, float],
        status_flags: int,
    ) -> tuple[bytes, int]:
        """PyAudio stream callback. Generates audio samples."""
        if not self._running:
            silence = b'\x00' * (frame_count * 4)
            return (silence, pyaudio.paComplete)

        # Track buffer underruns (output underflow = PortAudio ran out of data)
        if _HAS_PYAUDIO and (status_flags & pyaudio.paOutputUnderflow):
            self._underrun_count += 1

        try:
            samples = self._render_buffer(frame_count)
        except Exception:
            samples = np.zeros(frame_count, dtype=np.float32)

        # Apply master volume and mute
        if self._muted:
            samples[:] = 0.0
        else:
            samples *= self._master_volume

        # Soft clip to prevent harsh distortion
        samples = soft_clip(samples)

        self._buffer_count += 1
        return (samples.tobytes(), pyaudio.paContinue)

    def _render_buffer(self, n_samples: int) -> NDArray[np.float32]:
        """Mix all voice layers into a single buffer."""
        # Read snapshot atomically (GIL guarantees)
        ms = self._musical
        snap = self._snapshot

        dt = n_samples / SAMPLE_RATE

        # Smooth volume transitions — per-sample linear ramp to avoid
        # step discontinuities at buffer boundaries (audible as ~43 Hz buzz)
        smooth = 0.15
        prev_drone_vol = self._smooth_drone_vol
        prev_melody_vol = self._smooth_melody_vol
        prev_arp_vol = self._smooth_arp_vol
        prev_noise_vol = self._smooth_noise_vol

        self._smooth_drone_vol = lerp(prev_drone_vol, ms.drone_volume, smooth)
        self._smooth_melody_vol = lerp(prev_melody_vol, ms.melody_volume, smooth)
        self._smooth_arp_vol = lerp(prev_arp_vol, ms.arp_volume, smooth)
        self._smooth_noise_vol = lerp(prev_noise_vol, ms.noise_volume, smooth)

        drone_ramp = np.linspace(prev_drone_vol, self._smooth_drone_vol,
                                 n_samples, dtype=np.float32)
        melody_ramp = np.linspace(prev_melody_vol, self._smooth_melody_vol,
                                  n_samples, dtype=np.float32)
        arp_ramp = np.linspace(prev_arp_vol, self._smooth_arp_vol,
                               n_samples, dtype=np.float32)
        noise_ramp = np.linspace(prev_noise_vol, self._smooth_noise_vol,
                                 n_samples, dtype=np.float32)

        # Root portamento
        root_smooth = 0.05
        self._current_root_hz = lerp(self._current_root_hz, self._target_root_hz, root_smooth)

        # Render layers
        if self._crossfade_remaining > 0:
            # During style crossfade: render with both styles and blend
            fade_progress = 1.0 - (self._crossfade_remaining / max(1, self._crossfade_total))
            self._crossfade_remaining = max(0, self._crossfade_remaining - n_samples)

            # Render with new style (current)
            drone_new = self._render_drone(n_samples, dt)

            # Render drone with old style by temporarily swapping
            saved_style = self._style
            self._style = self._old_style
            drone_old = self._render_drone(n_samples, dt)
            self._style = saved_style

            # Crossfade: equal-power curve
            fade_new = np.float32(math.sqrt(fade_progress))
            fade_old = np.float32(math.sqrt(1.0 - fade_progress))
            drone = drone_old * fade_old + drone_new * fade_new
        else:
            drone = self._render_drone(n_samples, dt)

        melody = self._render_melody(n_samples, dt)
        arp = self._render_arpeggio(n_samples, dt)
        noise = self._render_noise(n_samples, dt)

        # Mix with per-sample volume ramps (click-free transitions)
        mix = (
            drone * drone_ramp
            + melody * melody_ramp
            + arp * arp_ramp
            + noise * noise_ramp
        )

        return mix

    def render_buffer_separated(self, n_samples: int) -> LayeredRender:
        """Render all layers and return them separately (diagnostic use).

        Performs identical processing to _render_buffer (volume smoothing,
        root portamento, style crossfade, LPF) but preserves per-layer
        arrays before mixing.  Each layer is post-volume-scaling.
        """
        ms = self._musical
        dt = n_samples / SAMPLE_RATE

        # Volume smoothing — per-sample ramp (same as _render_buffer)
        smooth = 0.15
        prev_drone_vol = self._smooth_drone_vol
        prev_melody_vol = self._smooth_melody_vol
        prev_arp_vol = self._smooth_arp_vol
        prev_noise_vol = self._smooth_noise_vol

        self._smooth_drone_vol = lerp(prev_drone_vol, ms.drone_volume, smooth)
        self._smooth_melody_vol = lerp(prev_melody_vol, ms.melody_volume, smooth)
        self._smooth_arp_vol = lerp(prev_arp_vol, ms.arp_volume, smooth)
        self._smooth_noise_vol = lerp(prev_noise_vol, ms.noise_volume, smooth)

        # Root portamento
        root_smooth = 0.05
        self._current_root_hz = lerp(self._current_root_hz, self._target_root_hz, root_smooth)

        # Render layers
        if self._crossfade_remaining > 0:
            fade_progress = 1.0 - (self._crossfade_remaining / max(1, self._crossfade_total))
            self._crossfade_remaining = max(0, self._crossfade_remaining - n_samples)
            drone_new = self._render_drone(n_samples, dt)
            saved_style = self._style
            self._style = self._old_style
            drone_old = self._render_drone(n_samples, dt)
            self._style = saved_style
            fade_new = np.float32(math.sqrt(fade_progress))
            fade_old = np.float32(math.sqrt(1.0 - fade_progress))
            drone_raw = drone_old * fade_old + drone_new * fade_new
        else:
            drone_raw = self._render_drone(n_samples, dt)

        melody_raw = self._render_melody(n_samples, dt)
        arp_raw = self._render_arpeggio(n_samples, dt)
        noise_raw = self._render_noise(n_samples, dt)

        # Apply per-sample volume ramps (diagnostic uses endpoint for LayeredRender metadata)
        d_vol = self._smooth_drone_vol
        m_vol = self._smooth_melody_vol
        a_vol = self._smooth_arp_vol
        n_vol = self._smooth_noise_vol

        drone_scaled = drone_raw * np.linspace(prev_drone_vol, d_vol,
                                               n_samples, dtype=np.float32)
        melody_scaled = melody_raw * np.linspace(prev_melody_vol, m_vol,
                                                 n_samples, dtype=np.float32)
        arp_scaled = arp_raw * np.linspace(prev_arp_vol, a_vol,
                                           n_samples, dtype=np.float32)
        noise_scaled = noise_raw * np.linspace(prev_noise_vol, n_vol,
                                               n_samples, dtype=np.float32)

        mix = drone_scaled + melody_scaled + arp_scaled + noise_scaled

        return LayeredRender(
            drone=drone_scaled,
            melody=melody_scaled,
            arp=arp_scaled,
            noise=noise_scaled,
            mix=mix,
            drone_vol=d_vol,
            melody_vol=m_vol,
            arp_vol=a_vol,
            noise_vol=n_vol,
        )

    # ── Drone layer ────────────────────────────────────────────────────

    def _render_drone(self, n_samples: int, dt: float) -> NDArray[np.float32]:
        """Continuous drone bed — always present, breathes with LFO."""
        root_hz = self._current_root_hz
        fifth_hz = root_hz * 1.5  # Perfect fifth

        t_inc = 1.0 / SAMPLE_RATE
        # Reuse pre-allocated buffer (slice to n_samples for safety)
        t_arr = self._buf_t[:n_samples]

        # LFO for breathing (very slow: ~0.1 Hz)
        lfo_freq = 0.08
        lfo_phases = self._drone_lfo_phase + TWO_PI * lfo_freq * t_arr
        lfo = 0.15 * np.sin(lfo_phases).astype(np.float32)
        self._drone_lfo_phase = float(np.mod(lfo_phases[-1] + TWO_PI * lfo_freq * t_inc, TWO_PI))

        # Phase increments (reuse pre-allocated index buffer)
        buf_n = self._buf_n[:n_samples]
        p1_inc = TWO_PI * root_hz * t_inc
        p1d_inc = TWO_PI * (root_hz * 1.003) * t_inc   # Detuned root (chorus)
        p2_inc = TWO_PI * fifth_hz * t_inc
        p2d_inc = TWO_PI * (fifth_hz * 1.002) * t_inc   # Detuned fifth
        ps_inc = TWO_PI * (root_hz * 0.5) * t_inc       # Sub-octave

        phases1 = self._drone_phase1 + buf_n * p1_inc
        phases1_det = self._drone_phase1_det + buf_n * p1d_inc
        phases2 = self._drone_phase2 + buf_n * p2_inc
        phases2_det = self._drone_phase2_det + buf_n * p2d_inc
        phases_sub = self._drone_phase_sub + buf_n * ps_inc

        # Keep phases bounded to preserve float precision
        self._drone_phase1 = float(np.mod(phases1[-1] + p1_inc, TWO_PI))
        self._drone_phase1_det = float(np.mod(phases1_det[-1] + p1d_inc, TWO_PI))
        self._drone_phase2 = float(np.mod(phases2[-1] + p2_inc, TWO_PI))
        self._drone_phase2_det = float(np.mod(phases2_det[-1] + p2d_inc, TWO_PI))
        self._drone_phase_sub = float(np.mod(phases_sub[-1] + ps_inc, TWO_PI))

        if self._style == STYLE_CHIPTUNE:
            # Detuned square waves + triangle sub
            osc1 = _square_wave(phases1, duty=0.5, freq_hz=root_hz) * 0.35
            osc2 = _square_wave(phases2_det, duty=0.5, freq_hz=fifth_hz * 1.002) * 0.25
            sub = _triangle_wave(phases_sub) * 0.4
            drone = osc1 + osc2 + sub
        else:
            # Ambient: detuned sines + filtered fifth
            osc1 = _sine_wave(phases1) * 0.4
            osc1_detune = _sine_wave(phases1_det) * 0.15  # Chorus (own phase)
            osc2 = _sine_wave(phases2) * 0.25
            sub = _sine_wave(phases_sub) * 0.2
            drone = osc1 + osc1_detune + osc2 + sub
            # Low-pass filter for warmth (cached coefficients + persistent state)
            drone = self._lpf_drone.apply(drone)

        # Apply LFO breathing
        drone = drone * (0.85 + lfo)

        return drone

    # ── Melody layer ───────────────────────────────────────────────────

    def _render_melody(self, n_samples: int, dt: float) -> NDArray[np.float32]:
        """Grid-scanning playhead: alive cells at current column → pitches."""
        ms = self._musical
        target_notes = ms.playhead_notes

        # Update voice assignments — use portamento for small pitch changes
        # to avoid ADSR retriggering at 30Hz (causes audible flutter).
        # Only hard-retrigger for large intervals (> ~1 semitone = 6% freq change).
        for i, voice in enumerate(self._melody_voices):
            if i < len(target_notes):
                freq = target_notes[i]
                if not voice.active:
                    voice.note_on(freq)
                elif abs(voice.target_freq - freq) > voice.target_freq * 0.06:
                    # Large pitch jump — full retrigger for articulation
                    voice.note_on(freq)
                else:
                    # Small change or same note — slide via portamento
                    voice.retrigger(freq)
            else:
                if voice.gate:
                    voice.note_off()

        # Render all melody voices (ambient gets a per-voice boost to
        # compensate for sine waves having ~3dB less RMS than square waves)
        voice_gain = 0.5 if self._style == STYLE_AMBIENT else 0.3
        buf = np.zeros(n_samples, dtype=np.float32)
        for voice in self._melody_voices:
            if voice.active:
                buf += voice.render(n_samples) * voice_gain

        # Chiptune melody benefits from a slight LPF to tame harsh square wave harmonics
        if self._style == STYLE_CHIPTUNE:
            buf = self._lpf_melody.apply(buf)

        return buf

    # ── Arpeggio layer ─────────────────────────────────────────────────

    def _render_arpeggio(self, n_samples: int, dt: float) -> NDArray[np.float32]:
        """Rhythmic arpeggiated texture, speed tied to cycle detection."""
        ms = self._musical
        pattern = ms.arp_pattern
        if not pattern:
            return np.zeros(n_samples, dtype=np.float32)

        root_midi = ms.target_root_midi
        note_duration_samples = max(1, int(SAMPLE_RATE / max(0.5, ms.arp_speed)))

        buf = np.zeros(n_samples, dtype=np.float32)
        samples_rendered = 0

        while samples_rendered < n_samples:
            # Samples remaining in current note
            remaining_in_note = note_duration_samples - self._arp_samples_in_note
            chunk = min(remaining_in_note, n_samples - samples_rendered)
            chunk = max(1, chunk)

            # Current note
            semitone_offset = pattern[self._arp_note_idx % len(pattern)]
            freq = midi_to_hz(root_midi + semitone_offset)

            if not self._arp_voice.active or abs(self._arp_voice.target_freq - freq) > 1.0:
                self._arp_voice.note_on(freq)
            self._arp_voice.style = self._style

            rendered = self._arp_voice.render(chunk)
            arp_gain = 0.55 if self._style == STYLE_AMBIENT else 0.4
            buf[samples_rendered:samples_rendered + chunk] = rendered * arp_gain

            self._arp_samples_in_note += chunk
            samples_rendered += chunk

            # Advance to next note when current note is complete
            if self._arp_samples_in_note >= note_duration_samples:
                self._arp_samples_in_note = 0
                self._arp_note_idx = (self._arp_note_idx + 1) % len(pattern)
                next_offset = pattern[self._arp_note_idx % len(pattern)]
                self._arp_voice.note_on(midi_to_hz(root_midi + next_offset))

        # Tame chiptune square wave harmonics; ambient sines don't need filtering
        if self._style == STYLE_CHIPTUNE:
            buf = self._lpf_arp.apply(buf)

        return buf

    # ── Noise layer ────────────────────────────────────────────────────

    def _render_noise(self, n_samples: int, dt: float) -> NDArray[np.float32]:
        """Percussive noise bursts triggered by population delta spikes."""
        ms = self._musical
        snap = self._snapshot

        # Trigger a new burst when pop_delta exceeds threshold.
        # Require full completion + cooldown to prevent chain-firing flutter.
        abs_delta = abs(snap.pop_delta)
        if self._noise_burst_cooldown > 0:
            self._noise_burst_cooldown = max(0.0, self._noise_burst_cooldown - dt)
        if (abs_delta >= self._noise_burst_threshold
                and self._noise_burst_phase >= 1.0
                and self._noise_burst_cooldown <= 0):
            self._noise_burst_phase = 0.0
            self._noise_burst_amp = lerp(0.1, 0.5, min(1.0, abs_delta / 80.0))
            # Bigger events = longer bursts
            self._noise_burst_duration = lerp(0.08, 0.25, min(1.0, abs_delta / 100.0))
            # Cooldown: at least the burst duration + 100ms breathing room
            self._noise_burst_cooldown = self._noise_burst_duration + 0.1

        # If no burst active, return silence
        if self._noise_burst_phase >= 1.0:
            return np.zeros(n_samples, dtype=np.float32)

        # Read from pre-generated noise pool (circular buffer, no allocation)
        pool_len = len(self._noise_pool)
        idx = self._noise_pool_idx % pool_len
        if idx + n_samples <= pool_len:
            noise = self._noise_pool[idx:idx + n_samples]
        else:
            # Wrap around
            part1 = self._noise_pool[idx:]
            part2 = self._noise_pool[:n_samples - len(part1)]
            noise = np.concatenate((part1, part2))
        self._noise_pool_idx = (idx + n_samples) % pool_len

        # Build per-sample envelope for the burst
        burst_dur = max(0.01, self._noise_burst_duration)
        t_arr = np.linspace(
            self._noise_burst_phase * burst_dur,
            self._noise_burst_phase * burst_dur + n_samples / SAMPLE_RATE,
            n_samples, endpoint=False, dtype=np.float32,
        )
        # Attack (first 10%) then exponential decay
        attack_end = burst_dur * 0.1
        env = np.where(
            t_arr < attack_end,
            t_arr / max(0.001, attack_end),  # linear attack
            np.exp(-6.0 * (t_arr - attack_end) / burst_dur),  # exponential decay
        ).astype(np.float32)

        # Advance burst phase
        self._noise_burst_phase += dt / burst_dur
        if self._noise_burst_phase >= 1.0:
            self._noise_burst_phase = 1.0

        if self._style == STYLE_CHIPTUNE:
            # NES-style: downsample for lo-fi crunch
            downsample = 8
            coarse = np.repeat(noise[::downsample], downsample)[:n_samples]
            if len(coarse) < n_samples:
                coarse = np.pad(coarse, (0, n_samples - len(coarse)))
            return coarse * env * self._noise_burst_amp
        else:
            # Ambient: low-pass filtered wash (cached filter)
            filtered = self._lpf_noise.apply(noise)
            return filtered * env * self._noise_burst_amp * 0.6

    # ── Status string for display ──────────────────────────────────────

    def status_string(self) -> str:
        """Return a short status string for the status bar."""
        if self._muted:
            return "[MUTE]"
        style_short = "NES" if self._style == STYLE_CHIPTUNE else "AMB"
        base = f"{style_short} {self.volume_percent}%"
        if self._underrun_count > 0:
            base += f" XR:{self._underrun_count}"
        return base
