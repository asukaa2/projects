"""
Audio I/O helpers for RVC.

This module replaces the previous PyAV-based implementation with one
built on top of :mod:`pydub` (which itself wraps FFmpeg under the hood).

Public API (kept identical to the previous version so callers do not
need to change):

* :func:`load_audio(file, sr)` — read any audio file and return a 1-D
  float32 numpy array resampled to ``sr`` Hz.
* :func:`wav2(i, o, format)` — transcode an input audio stream (file
  path or file-like) to a target container ``format`` (wav, flac, mp3,
  m4a, ogg, aac, ...).
* :func:`audio2(i, o, format, sr)` — decode ``i`` and re-encode it to
  ``o`` with the requested sample rate and a single (mono) channel.
  Used by :func:`load_audio` to stream f32le PCM into a numpy buffer.

Why pydub?
----------
PyAV is a thin Python wrapper around FFmpeg's libav* libraries that
loads via ``import av`` and pulls in a sizable native dependency
tree. Pydub is a much smaller pure-Python library that shells out to
the ``ffmpeg`` / ``ffprobe`` binaries — same codecs, much smaller
install surface, and easier to ship. It also has a friendlier API
for the common decode/encode/conversion operations RVC needs.
"""

from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import IO, Union

import numpy as np
import librosa
from pydub import AudioSegment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

#: Anything that :func:`pydub.AudioSegment.from_file` accepts for input —
#: either a filesystem path (str / bytes / os.PathLike) or a file-like
#: object with a ``read`` method.
FileLike = Union[str, bytes, os.PathLike, IO[bytes]]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

#: Map our external container names to the format strings pydub/FFmpeg
#: understand. The keys are the values that the WebUI's "导出文件格式"
#: dropdown produces ("wav", "flac", "mp3", "m4a") plus a few legacy
#: aliases ("ogg", "aac") that older callers may still pass.
_FORMAT_MAP: dict[str, str] = {
    "wav": "wav",
    "flac": "flac",
    "mp3": "mp3",
    "m4a": "ipod",   # pydub's "ipod" container == m4a / MP4 audio
    "mp4": "ipod",
    "ogg": "ogg",
    "aac": "adts",
    "f32le": "raw",
    "f64le": "raw",
}

#: Codec overrides. pydub's defaults are usually fine, but for some
#: containers we want to be explicit so that the right audio codec is
#: selected (e.g. AAC inside an MP4 / m4a container).
_CODEC_MAP: dict[str, str] = {
    "ogg": "libvorbis",
    "m4a": "aac",
    "mp4": "aac",
    "ipod": "aac",
    "aac": "aac",
    # For the raw f32le / f64le cases we use the PCM codec name that
    # FFmpeg understands; pydub passes it straight through.
    "f32le": "pcm_f32le",
    "f64le": "pcm_f64le",
}

#: Pydub sample-format -> numpy dtype + scale factor used by
#: :func:`_audio_segment_to_float32` to normalize integer samples to
#: the [-1.0, 1.0] float32 range RVC expects.
_SAMPLE_WIDTH_TO_DTYPE = {
    1: (np.int16, 32768.0),  # 8-bit: pydub upconverts to int16 internally
    2: (np.int16, 32768.0),
    3: (np.int32, 8388608.0),  # 24-bit packed into 32-bit
    4: (np.int32, 2147483648.0),
}


def _resolve_format(format: str) -> tuple[str, str]:
    """Return ``(pydub_format, codec)`` for a user-facing container name."""
    fmt = (format or "wav").lower()
    pydub_format = _FORMAT_MAP.get(fmt, fmt)
    codec = _CODEC_MAP.get(fmt) or _CODEC_MAP.get(pydub_format)
    return pydub_format, codec


def _audio_segment_to_float32(seg: AudioSegment) -> np.ndarray:
    """Convert a :class:`pydub.AudioSegment` to a 1-D float32 numpy array."""
    # ``get_array_of_samples`` returns signed integers; the bit depth is
    # given by ``seg.sample_width`` (1, 2, 3 or 4 bytes).
    samples = np.array(seg.get_array_of_samples(), dtype=np.int32)
    if seg.channels > 1:
        # (channels, frames) -> average to mono
        samples = samples.reshape(-1, seg.channels).mean(axis=1)
    dtype, scale = _SAMPLE_WIDTH_TO_DTYPE.get(seg.sample_width, (np.int16, 32768.0))
    samples = samples.astype(np.float32) / scale
    return samples.flatten()


def _coerce_segment(seg: AudioSegment, sr: int | None,
                    mono: bool = True) -> AudioSegment:
    """Apply sample-rate and channel-layout conversions to a segment."""
    if mono and seg.channels > 1:
        seg = seg.set_channels(1)
    if sr is not None and seg.frame_rate != sr:
        # Pydub's resampler uses FFmpeg's libswresample; quality is
        # good enough for content-vector extraction at 16 kHz.
        seg = seg.set_frame_rate(sr)
    return seg


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def wav2(i: FileLike, o: FileLike, format: str) -> None:
    """Transcode an audio stream to a target container format.

    Parameters
    ----------
    i:
        Source — a file path or a readable binary file-like object.
    o:
        Destination — a file path or a writable binary file-like object.
    format:
        User-facing container name. One of: ``wav``, ``flac``, ``mp3``,
        ``m4a``, ``ogg``, ``aac``. Unknown names are passed through
        to pydub as-is (which will raise a clear error if FFmpeg doesn't
        know the format).
    """
    pydub_format, codec = _resolve_format(format)
    audio = AudioSegment.from_file(i)
    export_kwargs = {"format": pydub_format}
    if codec:
        export_kwargs["codec"] = codec
    audio.export(o, **export_kwargs)


def audio2(i: FileLike, o: FileLike, format: str, sr: int) -> None:
    """Decode ``i`` and re-encode to ``o`` with sample-rate ``sr`` and mono.

    Used by :func:`load_audio` to stream raw f32le PCM into a numpy
    buffer; also useful in its own right for sample-rate conversion.

    Notes
    -----
    For the special case ``format="f32le"`` (raw 32-bit little-endian
    float PCM), we bypass pydub's export path because FFmpeg does not
    expose a "raw with f32le codec" muxer. Instead we decode the input
    to an :class:`AudioSegment`, resample to ``sr`` Hz, downmix to mono,
    convert the integer samples to float32, and write the raw bytes
    directly. This preserves the exact byte layout that the previous
    PyAV-based implementation produced.
    """
    seg = AudioSegment.from_file(i)
    seg = _coerce_segment(seg, sr=sr, mono=True)

    fmt = (format or "").lower()

    if fmt in ("f32le", "f64le"):
        # Raw PCM float output — bypass the FFmpeg "raw" muxer entirely.
        samples = _audio_segment_to_float32(seg)
        if fmt == "f64le":
            samples = samples.astype(np.float64)
        else:
            samples = samples.astype(np.float32)
        _write_bytes(o, samples.tobytes())
        return

    pydub_format, codec = _resolve_format(format)
    export_kwargs = {"format": pydub_format}
    if codec:
        export_kwargs["codec"] = codec
    seg.export(o, **export_kwargs)


def _write_bytes(o: FileLike, data: bytes) -> None:
    """Write ``data`` to either a file path or a writable file-like object."""
    if isinstance(o, (str, bytes, os.PathLike)):
        with open(o, "wb") as f:
            f.write(data)
    else:
        o.write(data)


def load_audio(file, sr: int) -> np.ndarray:
    """Load any audio file and return a 1-D float32 array at sample-rate ``sr``.

    Parameters
    ----------
    file:
        Either a filesystem path (``str`` / ``bytes`` / ``os.PathLike``)
        or a tuple ``(sample_rate, numpy_array)`` for in-memory audio
        that needs to be resampled to ``sr``.
    sr:
        Target sample rate in Hz (typically 16000 for content-vector
        extraction).

    Returns
    -------
    numpy.ndarray
        1-D ``float32`` array, mono, sampled at ``sr`` Hz.

    Notes
    -----
    The previous ``av``-based implementation streamed raw f32le PCM
    through a buffer and then ran ``np.frombuffer``. The pydub version
    decodes once into an :class:`AudioSegment` and converts the integer
    samples to float32 in one numpy step — semantically equivalent but
    with a smaller memory peak because we don't double-buffer the raw
    PCM bytes.
    """
    # Tuple input: in-memory (sample_rate, numpy_array) shortcut.
    # The previous code had this fallback branch and several call
    # sites (e.g. tests) rely on it.
    if isinstance(file, tuple):
        try:
            in_sr, audio = file
            audio = np.asarray(audio, dtype=np.float32)
            # Normalize integer-encoded audio (e.g. int16 PCM) to float32.
            # If the input already looks float-like (max <= 1.0) we leave
            # it alone; otherwise divide by 32768 (the int16 max).
            if np.max(np.abs(audio)) > 1.0:
                audio = audio / 32768.0
            if audio.ndim == 2:
                audio = audio.mean(axis=-1)
            if in_sr != sr:
                audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sr)
            return audio.flatten()
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from tuple: {e}")

    try:
        if isinstance(file, str):
            file = (
                file.strip(" ").strip('"').strip("\n").strip('"').strip(" ")
            )
        seg = AudioSegment.from_file(file)
        seg = _coerce_segment(seg, sr=sr, mono=True)
        return _audio_segment_to_float32(seg)
    except Exception as e:
        raise RuntimeError(f"Failed to load audio: {e}")
