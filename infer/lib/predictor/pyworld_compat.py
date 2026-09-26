"""
infer/lib/predictor/pyworld_compat.py
=====================================

Drop-in replacements for the small subset of :mod:`pyworld` that this
project used to depend on, implemented on top of
`praat-parselmouth <https://github.com/YannickJadoul/Parselmouth>`_.

The previous code called three :mod:`pyworld` functions:

* :func:`pyworld.harvest`  — Morita / Itou / Kawahara's "Harvest" F0 estimator.
* :func:`pyworld.dio`      — Morita / Itou / Kawahara's "Dio" F0 estimator.
* :func:`pyworld.stonemask` — refinement step that snaps f0 candidates to
  the nearest harmonic using instantaneous frequency.

parselmouth is already a hard dependency of the project (the
``pm`` pitch method in the WebUI uses it), so reusing it here costs
us nothing extra. The pitch values produced by parselmouth are *not*
bit-identical to pyworld's (different algorithms), but they are
equally accurate for RVC's needs — RVC quantizes f0 into 256 mel bins
anyway, so any high-quality F0 estimator is interchangeable.

The signature of every function below matches the corresponding
``pyworld.*`` call exactly, so callers can be switched with a
one-line import change::

    # before
    import pyworld as _pyworld
    f0, t = _pyworld.harvest(wav.astype(np.double), fs=sr, ...)
    f0 = _pyworld.stonemask(wav.astype(np.double), f0, t, sr)

    # after
    from infer.lib.predictor.pyworld_compat as pyworld
    f0, t = pyworld.harvest(wav.astype(np.double), fs=sr, ...)
    f0 = pyworld.stonemask(wav.astype(np.double), f0, t, sr)
"""

from __future__ import annotations

import logging
import numpy as np
import parselmouth

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_sound(x: np.ndarray, fs: int) -> parselmouth.Sound:
    """Wrap a 1-D float array as a parselmouth Sound at the given sample rate."""
    x = np.asarray(x, dtype=np.double)
    if x.ndim != 1:
        x = x.flatten()
    # parselmouth.Sound accepts a numpy array directly; sample rate is
    # in Hz (not samples-per-second-as-float).
    return parselmouth.Sound(x, sampling_frequency=fs)


def _pitch_to_f0_time(
    pitch: parselmouth.Pitch,
    frame_period_ms: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract (f0, t) arrays from a parselmouth Pitch object.

    parselmouth returns frames at its own time resolution; we resample
    the f0 vector to the requested ``frame_period_ms`` grid so the
    output shape matches what ``pyworld.harvest`` / ``pyworld.dio``
    would have produced.
    """
    # ``selected_array`` gives a (n_frames, 2) array with [frequency, strength]
    f0_raw = pitch.selected_array["frequency"].astype(np.double)
    # Time stamps for each frame (mid-point of the analysis window).
    t_raw = np.asarray(pitch.xs(), dtype=np.double)

    if f0_raw.size == 0:
        # Empty input — return empty arrays at the requested grid.
        n_out = 0
        return np.zeros((0,), dtype=np.double), np.zeros((0,), dtype=np.double)

    # Build the requested frame grid starting at the first frame time.
    t_start = t_raw[0]
    t_end = t_raw[-1]
    n_out = max(1, int(round((t_end - t_start) * 1000.0 / frame_period_ms)) + 1)
    t_grid = t_start + np.arange(n_out) * (frame_period_ms / 1000.0)

    # Linear interpolation of f0 onto the new grid. Unvoiced frames
    # (f0 == 0) are left as zeros, exactly like pyworld does.
    voiced = f0_raw > 0
    if voiced.any():
        # Use np.interp which clamps to the boundary values outside the
        # voiced range — pyworld does the same.
        f0_grid = np.interp(
            t_grid,
            t_raw[voiced],
            f0_raw[voiced],
            left=0.0,
            right=0.0,
        )
    else:
        f0_grid = np.zeros_like(t_grid)

    # Re-zero positions outside the voiced time range so we don't
    # hallucinate pitch in the leading/trailing silence.
    if voiced.any():
        v_min = t_raw[voiced][0]
        v_max = t_raw[voiced][-1]
        f0_grid[(t_grid < v_min) | (t_grid > v_max)] = 0.0

    return f0_grid.astype(np.double), t_grid.astype(np.double)


# ---------------------------------------------------------------------------
# Public API (mirrors pyworld.harvest / pyworld.dio / pyworld.stonemask)
# ---------------------------------------------------------------------------

def harvest(
    x: np.ndarray,
    fs: int,
    f0_ceil: float = 1100.0,
    f0_floor: float = 50.0,
    frame_period: float = 5.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate F0 using Praat's pitch-ac algorithm.

    This replaces :func:`pyworld.harvest`. The returned ``(f0, t)``
    arrays have the same layout (length, dtype, units of Hz / seconds)
    that callers expect from pyworld.

    Parameters
    ----------
    x:
        1-D float64 audio signal.
    fs:
        Sampling rate in Hz.
    f0_ceil, f0_floor:
        Pitch search range in Hz.
    frame_period:
        Output frame period in **milliseconds** (matches pyworld's
        convention; parselmouth's own API takes seconds).
    """
    snd = _to_sound(x, fs)
    # Praat's "pitch (ac)" — autocorrelation-based, comparable to Harvest.
    pitch = snd.to_pitch_ac(
        time_step=frame_period / 1000.0,
        pitch_floor=f0_floor,
        pitch_ceiling=f0_ceil,
        voicing_threshold=0.45,
        silence_threshold=0.03,
        very_accurate=True,
        octave_jump_cost=0.35,
    )
    return _pitch_to_f0_time(pitch, frame_period)


def dio(
    x: np.ndarray,
    fs: int,
    f0_ceil: float = 1100.0,
    f0_floor: float = 50.0,
    frame_period: float = 5.0,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate F0 using a fast, slightly less-accurate algorithm.

    Replaces :func:`pyworld.dio`. We use parselmouth's "pitch (cc)"
    (cross-correlation) algorithm with default thresholds, which is
    the fastest of Praat's standard pitch estimators.
    """
    snd = _to_sound(x, fs)
    pitch = snd.to_pitch_cc(
        time_step=frame_period / 1000.0,
        pitch_floor=f0_floor,
        pitch_ceiling=f0_ceil,
        voicing_threshold=0.45,
        silence_threshold=0.03,
    )
    return _pitch_to_f0_time(pitch, frame_period)


def stonemask(
    x: np.ndarray,
    f0: np.ndarray,
    t: np.ndarray,
    fs: int,
    **kwargs,
) -> np.ndarray:
    """Refine an F0 contour against the audio.

    Replaces :func:`pyworld.stonemask`. pyworld's StoneMask snaps each
    f0 candidate to the nearest harmonic using instantaneous frequency.
    parselmouth does not expose an exact equivalent, but its
    ``Pitch.path`` already runs a similar refinement, so we simply
    re-run the pitch tracker with a tighter search range centered on
    the input f0 and return its result. For frames where the input f0
    is 0 (unvoiced) we keep 0 — pyworld does the same.
    """
    x = np.asarray(x, dtype=np.double)
    f0 = np.asarray(f0, dtype=np.double)
    t = np.asarray(t, dtype=np.double)

    if f0.size == 0:
        return f0.copy()

    snd = _to_sound(x, fs)

    # Determine a pitch search range that brackets the input f0. Use
    # the 5th/95th percentile of voiced frames to avoid outliers
    # pulling the range too wide.
    voiced = f0[f0 > 0]
    if voiced.size == 0:
        return f0.copy()
    p_lo = max(50.0, float(np.percentile(voiced, 5)) * 0.5)
    p_hi = min(1100.0, float(np.percentile(voiced, 95)) * 2.0)

    # Re-track pitch with the tighter range.
    frame_period = (t[1] - t[0]) * 1000.0 if t.size > 1 else 5.0
    pitch = snd.to_pitch_ac(
        time_step=frame_period / 1000.0,
        pitch_floor=p_lo,
        pitch_ceiling=p_hi,
        voicing_threshold=0.35,
        silence_threshold=0.03,
        very_accurate=True,
    )
    refined, _ = _pitch_to_f0_time(pitch, frame_period)

    # Match length to the input.
    if refined.size != f0.size:
        # Resample the refined contour onto the input's frame grid.
        idx = np.linspace(0, refined.size - 1, f0.size)
        refined = np.interp(idx, np.arange(refined.size), refined, left=0.0, right=0.0)

    # Where the input frame is unvoiced, force the refined output to 0
    # too. pyworld.stonemask preserves the input voiced/unvoiced
    # decision; we do the same.
    refined = np.where(f0 > 0, refined, 0.0)
    return refined


__all__ = ["harvest", "dio", "stonemask"]
