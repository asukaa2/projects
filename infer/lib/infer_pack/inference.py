"""
infer/lib/infer_pack/inference.py
=================================

High-level inference API for RVC (Retrieval-based Voice Conversion) checkpoints.

This module lives under ``infer/lib/infer_pack`` so it can be imported as a
self-contained package (``from infer.lib.infer_pack.inference import RVCInferer``)
without dragging in the Gradio web UI.

Key features
------------
* **Format-agnostic loading** — pass either a ``.pth`` or a ``.safetensors``
  checkpoint; the right loader is picked automatically via
  :func:`infer.lib.infer_pack.safetensors_utils.load_rvc_checkpoint`.
* **Auto architecture selection** — the synthesizer class is chosen from the
  checkpoint's ``version`` (v1 / v2) and ``f0`` (pitch-guided or not) flags.
* **Single-file & batch conversion** — :meth:`RVCInferer.convert` for one
  file, :meth:`RVCInferer.convert_batch` for a folder.
* **Zero boilerplate** — sensible defaults for f0 method, index rate,
  protection, etc., matching the WebUI's defaults.

Quick example
-------------
.. code-block:: python

    from infer.lib.infer_pack.inference import RVCInferer

    inferer = RVCInferer("assets/weights/singer_v2.safetensors", device="cuda:0")
    inferer.convert("in.wav", "out.wav", f0up_key=12)

Or, equivalently, from the command line:

.. code-block:: bash

    python -m infer.lib.infer_pack.inference \\
        --model assets/weights/singer_v2.safetensors \\
        --input in.wav --output out.wav --f0up_key 12
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.io import wavfile

# Make the project root importable when this module is run as a script.
_NOW_DIR = os.getcwd()
if _NOW_DIR not in sys.path:
    sys.path.append(_NOW_DIR)

from infer.lib.audio import load_audio
from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)
from infer.lib.infer_pack.safetensors_utils import load_rvc_checkpoint

# NOTE: ``infer.modules.vc.pipeline.Pipeline`` and
# ``infer.modules.vc.utils.load_hubert`` are imported lazily inside the
# methods that need them. This keeps ``import infer.lib.infer_pack.inference``
# cheap (only needs torch + safetensors), which matters for users who
# want to inspect checkpoint metadata without pulling in librosa /
# torchaudio / faiss.
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

#: Audio file extensions recognized by the batch converter.
AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac", ".opus")

#: Map ``(version, if_f0) -> synthesizer class``. Keep in sync with
#: ``infer/modules/vc/modules.py``.
SYNTHESIZER_CLASSES: Dict[Tuple[str, int], type] = {
    ("v1", 1): SynthesizerTrnMs256NSFsid,
    ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
    ("v2", 1): SynthesizerTrnMs768NSFsid,
    ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
}

#: Default f0 extraction method. ``rmvpe`` is the best quality / speed
#: trade-off on a GPU; ``harvest`` is a CPU-friendly fallback.
DEFAULT_F0_METHOD = "rmvpe"

#: Allowed f0 extraction methods.
F0_METHODS = ("harvest", "pm", "crepe", "rmvpe", "fcpe")


def _resolve_device(device: Optional[str]) -> str:
    """Pick a sensible default device if the caller didn't specify one."""
    if device is not None:
        return device
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return "xpu"
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_is_half(device: str, is_half: Optional[bool]) -> bool:
    """Decide whether FP16 should be used on the chosen device."""
    if is_half is not None:
        return is_half
    # CPU/MPS don't always support FP16 well — keep them on FP32 by default.
    return device.startswith("cuda")


class _ConfigShim:
    """Minimal stand-in for ``configs.config.Config``.

    The full :class:`Config` parses argv and probes the GPU at import time,
    which is undesirable for a library API. We expose just the attributes
    read by :class:`Pipeline` and :func:`load_hubert`.
    """

    def __init__(self, device: str, is_half: bool):
        self.device = device
        self.is_half = is_half
        # x_pad / x_query / x_center / x_max control the inference chunk
        # sizes used by :class:`Pipeline`. The values below mirror the
        # defaults set by ``configs.config.Config.device_config()`` for the
        # corresponding dtype/device combination.
        if is_half and ("cuda" in device or "xpu" in device):
            self.x_pad = 3
            self.x_query = 10
            self.x_center = 60
            self.x_max = 65
        else:
            self.x_pad = 3
            self.x_query = 12
            self.x_center = 38
            self.x_max = 41


# ---------------------------------------------------------------------------
# RVCInferer
# ---------------------------------------------------------------------------

class RVCInferer:
    """High-level one-call voice-conversion wrapper.

    Parameters
    ----------
    model_path:
        Path to an RVC checkpoint, either ``.pth`` or ``.safetensors``.
        The format is auto-detected.
    device:
        Torch device, e.g. ``"cuda:0"``, ``"cpu"``, ``"mps"``. If
        ``None``, the best available device is chosen.
    is_half:
        Whether to run the synthesizer in FP16. If ``None``, FP16 is
        enabled on CUDA and disabled elsewhere.
    index_path:
        Optional path to a trained ``.index`` retrieval file. Greatly
        improves timbre fidelity for models trained with retrieval.
    hubert_cache_dir:
        Optional directory to look for ``hubert_base.pt``. Defaults to
        ``assets/hubert``.
    """

    def __init__(
        self,
        model_path: str,
        *,
        device: Optional[str] = None,
        is_half: Optional[bool] = None,
        index_path: str = "",
        hubert_cache_dir: str = "assets/hubert",
    ):
        self.model_path = os.path.abspath(model_path)
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        self.device = _resolve_device(device)
        self.is_half = _resolve_is_half(self.device, is_half)
        self.index_path = index_path
        self.hubert_cache_dir = hubert_cache_dir

        self.config = _ConfigShim(self.device, self.is_half)

        # State populated by ``_load_model``.
        self.ckpt: Optional[Dict] = None
        self.net_g: Optional[torch.nn.Module] = None
        self.pipeline: Optional[Pipeline] = None
        self.hubert_model = None
        self.tgt_sr: int = 0
        self.if_f0: int = 1
        self.version: str = "v1"
        self.n_spk: int = 1

        self._load_model()

    # ------------------------------------------------------------------
    # Internal loading
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Read the checkpoint and build the synthesizer + pipeline."""
        logger.info("Loading RVC checkpoint: %s", self.model_path)
        self.ckpt = load_rvc_checkpoint(self.model_path)

        weight = self.ckpt["weight"]
        config = self.ckpt["config"]
        if not config:
            raise ValueError(
                f"Checkpoint {self.model_path!r} has an empty `config` list — "
                "cannot build the synthesizer."
            )

        self.tgt_sr = config[-1]
        self.if_f0 = int(self.ckpt.get("f0", 1))
        self.version = self.ckpt.get("version", "v1")
        # ``config[-3]`` is the speaker-embedding dim, but the actual
        # number of speakers lives in ``emb_g.weight``. Override before
        # building the synthesizer so the WebUI's `n_spk` slider matches.
        try:
            self.n_spk = weight["emb_g.weight"].shape[0]
            config[-3] = self.n_spk
        except (KeyError, IndexError):
            # Some checkpoints (e.g. single-speaker .safetensors with
            # stripped metadata) won't have ``emb_g.weight``; default to 1.
            self.n_spk = 1

        synth_cls = SYNTHESIZER_CLASSES.get(
            (self.version, self.if_f0),
            SynthesizerTrnMs256NSFsid,
        )
        try:
            self.net_g = synth_cls(*config, is_half=self.is_half)
        except TypeError:
            # The ``_nono`` variants don't accept ``is_half``.
            self.net_g = synth_cls(*config)

        # The posterior encoder (``enc_q``) is only used for training —
        # drop it before load_state_dict to avoid shape mismatches.
        if hasattr(self.net_g, "enc_q"):
            del self.net_g.enc_q

        missing, unexpected = self.net_g.load_state_dict(weight, strict=False)
        if missing:
            logger.warning("Missing keys when loading model: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys when loading model: %s", unexpected[:5])

        self.net_g.eval().to(self.device)
        if self.is_half:
            self.net_g = self.net_g.half()
        else:
            self.net_g = self.net_g.float()

        # ``self.pipeline`` is built lazily by :meth:`_ensure_pipeline` so
        # that constructing an :class:`RVCInferer` (and just inspecting
        # its metadata via :meth:`info`) doesn't pull in the full audio
        # stack (librosa / torchaudio / faiss / parselmouth).

        logger.info(
            "Model ready: version=%s, if_f0=%s, tgt_sr=%d, n_spk=%d, device=%s, half=%s",
            self.version, self.if_f0, self.tgt_sr, self.n_spk,
            self.device, self.is_half,
        )

    def _ensure_pipeline(self) -> None:
        """Lazy-build the audio inference pipeline on first use."""
        if self.pipeline is None:
            from infer.modules.vc.pipeline import Pipeline
            self.pipeline = Pipeline(self.tgt_sr, self.config)

    def _ensure_hubert(self) -> None:
        """Lazy-load the HuBERT content encoder on first inference."""
        if self.hubert_model is None:
            from infer.modules.vc.utils import load_hubert
            logger.info("Loading HuBERT content encoder...")
            self.hubert_model = load_hubert(self.config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        input_path: str,
        output_path: Optional[str] = None,
        *,
        f0up_key: int = 0,
        f0_method: str = DEFAULT_F0_METHOD,
        index_rate: float = 0.75,
        filter_radius: int = 3,
        resample_sr: int = 0,
        rms_mix_rate: float = 0.25,
        protect: float = 0.33,
        sid: int = 0,
        f0_file: Optional[str] = None,
    ) -> Tuple[int, np.ndarray]:
        """Convert a single audio file.

        Parameters
        ----------
        input_path:
            Path to the source audio file (wav / mp3 / flac / m4a / ogg / aac).
        output_path:
            Where to write the converted WAV. If ``None``, nothing is
            written and the (sample_rate, audio_array) tuple is returned
            for the caller to handle.
        f0up_key:
            Pitch shift in semitones (default ``0``). Use ``12`` for one
            octave up, ``-12`` for one octave down.
        f0_method:
            Pitch-extraction algorithm. One of :data:`F0_METHODS`.
        index_rate:
            Retrieval-index blend rate, ``0.0``–``1.0``. ``0`` disables
            the index entirely; ``1`` fully trusts the retrieval results.
        filter_radius:
            Median-filter radius applied to the extracted f0 curve
            (default ``3``). Higher values smooth more aggressively.
        resample_sr:
            Target sample rate for the output. ``0`` keeps the model's
            native sample rate (``self.tgt_sr``).
        rms_mix_rate:
            How much of the source's envelope to keep vs. let the model
            produce its own (``0`` = keep source, ``1`` = use model).
        protect:
            Voiceless-frame protection (``0.0``–``0.5``). Lower values
            protect more against artifacts on unvoiced segments.
        sid:
            Speaker ID for multi-speaker models (0-indexed).
        f0_file:
            Optional path to a pre-computed f0 ``.pits`` file. If given,
            this overrides the algorithmic pitch extractor.

        Returns
        -------
        (sample_rate, audio_array)
            The converted audio as a 1-D float32 numpy array.
        """
        if f0_method not in F0_METHODS:
            raise ValueError(
                f"Unknown f0_method {f0_method!r}. Expected one of {F0_METHODS}."
            )
        if not os.path.isfile(input_path):
            raise FileNotFoundError(f"Input audio not found: {input_path}")

        self._ensure_pipeline()
        self._ensure_hubert()

        audio = load_audio(input_path, 16000)
        audio_max = np.abs(audio).max() / 0.95
        if audio_max > 1:
            audio /= audio_max

        times = [0.0, 0.0, 0.0]
        audio_opt = self.pipeline.pipeline(
            self.hubert_model,
            self.net_g,
            sid,
            audio,
            input_path,
            times,
            int(f0up_key),
            f0_method,
            self.index_path or None,
            float(index_rate),
            self.if_f0,
            int(filter_radius),
            self.tgt_sr,
            int(resample_sr),
            float(rms_mix_rate),
            self.version,
            float(protect),
            f0_file,
        )

        sr_out = self.tgt_sr if not (self.tgt_sr != resample_sr >= 16000) else resample_sr

        if output_path is not None:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
            wavfile.write(output_path, sr_out, audio_opt)
            logger.info(
                "Wrote %s  (npy=%.2fs, f0=%.2fs, infer=%.2fs)",
                output_path, *times,
            )

        return sr_out, audio_opt

    def convert_batch(
        self,
        input_dir: str,
        output_dir: str,
        *,
        recursive: bool = False,
        overwrite: bool = False,
        **convert_kwargs,
    ) -> Dict[str, int]:
        """Convert every audio file under ``input_dir`` to ``output_dir``.

        The relative folder structure is preserved when ``recursive=True``.

        Returns
        -------
        dict
            ``{"total": N, "ok": K, "skipped": S, "failed": F}``.
        """
        if not os.path.isdir(input_dir):
            raise NotADirectoryError(f"Input folder not found: {input_dir}")
        os.makedirs(output_dir, exist_ok=True)

        files = self._collect_audio_files(input_dir, recursive)
        if not files:
            logger.warning("No audio files found under %s", input_dir)
            return {"total": 0, "ok": 0, "skipped": 0, "failed": 0}

        total, ok, skipped, failed = len(files), 0, 0, 0
        self._ensure_pipeline()
        self._ensure_hubert()

        for i, in_path in enumerate(files, 1):
            rel = os.path.relpath(in_path, input_dir)
            stem = Path(rel).with_suffix("")
            out_path = os.path.join(output_dir, f"{stem}.wav")
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

            if os.path.exists(out_path) and not overwrite:
                logger.info("[%d/%d] skip (exists): %s", i, total, out_path)
                skipped += 1
                continue

            t0 = time.time()
            try:
                self.convert(in_path, out_path, **convert_kwargs)
                ok += 1
                logger.info(
                    "[%d/%d] ok in %.2fs: %s",
                    i, total, time.time() - t0, out_path,
                )
            except Exception as exc:  # noqa: BLE001
                logger.exception("[%d/%d] FAILED %s: %s", i, total, in_path, exc)
                failed += 1

        logger.info(
            "Batch complete: %d ok, %d skipped, %d failed (of %d).",
            ok, skipped, failed, total,
        )
        return {"total": total, "ok": ok, "skipped": skipped, "failed": failed}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _collect_audio_files(folder: str, recursive: bool) -> List[str]:
        out: List[str] = []
        if recursive:
            for root, _dirs, files in os.walk(folder):
                for name in files:
                    if name.lower().endswith(AUDIO_EXTS):
                        out.append(os.path.join(root, name))
        else:
            for name in os.listdir(folder):
                full = os.path.join(folder, name)
                if os.path.isfile(full) and name.lower().endswith(AUDIO_EXTS):
                    out.append(full)
        return sorted(out)

    def info(self) -> Dict[str, object]:
        """Return a short info dict about the loaded model."""
        return {
            "path": self.model_path,
            "version": self.version,
            "if_f0": self.if_f0,
            "tgt_sr": self.tgt_sr,
            "n_spk": self.n_spk,
            "device": self.device,
            "is_half": self.is_half,
            "info": self.ckpt.get("info", "") if self.ckpt else "",
        }

    def __repr__(self) -> str:
        return (
            f"RVCInferer(model={os.path.basename(self.model_path)!r}, "
            f"version={self.version!r}, if_f0={self.if_f0}, "
            f"tgt_sr={self.tgt_sr}, n_spk={self.n_spk}, "
            f"device={self.device!r}, half={self.is_half})"
        )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="rvc-infer",
        description="RVC inference CLI (supports both .pth and .safetensors).",
    )
    p.add_argument("--model", required=True,
                   help="Path to the RVC checkpoint (.pth or .safetensors).")
    p.add_argument("--device", default=None,
                   help="Torch device, e.g. 'cuda:0', 'cpu', 'mps'.")
    p.add_argument("--is_half", default=None, type=lambda v: v.lower() in ("1", "true", "yes"),
                   help="Force FP16 (True/False). Default: auto.")

    p.add_argument("--f0up_key", type=int, default=0,
                   help="Pitch shift in semitones (default: 0).")
    p.add_argument("--f0method", default=DEFAULT_F0_METHOD, choices=F0_METHODS,
                   help=f"Pitch-extraction algorithm (default: {DEFAULT_F0_METHOD}).")
    p.add_argument("--index_path", default="",
                   help="Path to a .index retrieval file (optional).")
    p.add_argument("--index_rate", type=float, default=0.75,
                   help="Index blend rate, 0.0-1.0 (default: 0.75).")
    p.add_argument("--filter_radius", type=int, default=3,
                   help="Median filter radius (default: 3).")
    p.add_argument("--resample_sr", type=int, default=0,
                   help="Target sample rate, 0 = native (default: 0).")
    p.add_argument("--rms_mix_rate", type=float, default=0.25,
                   help="RMS mix rate (default: 0.25).")
    p.add_argument("--protect", type=float, default=0.33,
                   help="Voiceless protection (default: 0.33).")
    p.add_argument("--sid", type=int, default=0,
                   help="Speaker ID for multi-speaker models (default: 0).")

    sub = p.add_subparsers(dest="command", required=True)

    p_one = sub.add_parser("single", help="Convert a single audio file.")
    p_one.add_argument("--input", required=True, help="Input audio file path.")
    p_one.add_argument("--output", required=True, help="Output WAV path.")

    p_batch = sub.add_parser("batch", help="Convert every audio file in a folder.")
    p_batch.add_argument("--input_dir", required=True, help="Folder with input audio.")
    p_batch.add_argument("--output_dir", required=True, help="Output folder.")
    p_batch.add_argument("--recursive", action="store_true",
                         help="Recurse into subfolders.")
    p_batch.add_argument("--overwrite", action="store_true",
                         help="Overwrite existing output files.")

    return p


def _main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = _build_parser().parse_args()

    inferer = RVCInferer(
        args.model,
        device=args.device,
        is_half=args.is_half,
        index_path=args.index_path,
    )
    print(f"[+] Loaded model: {inferer!r}")

    if args.command == "single":
        inferer.convert(
            args.input,
            args.output,
            f0up_key=args.f0up_key,
            f0_method=args.f0method,
            index_rate=args.index_rate,
            filter_radius=args.filter_radius,
            resample_sr=args.resample_sr,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
            sid=args.sid,
        )
        print(f"[✓] Wrote {args.output}")
        return 0

    if args.command == "batch":
        stats = inferer.convert_batch(
            args.input_dir,
            args.output_dir,
            recursive=args.recursive,
            overwrite=args.overwrite,
            f0up_key=args.f0up_key,
            f0_method=args.f0method,
            index_rate=args.index_rate,
            filter_radius=args.filter_radius,
            resample_sr=args.resample_sr,
            rms_mix_rate=args.rms_mix_rate,
            protect=args.protect,
            sid=args.sid,
        )
        print(f"[✓] {stats}")
        return 0 if stats["failed"] == 0 else 2

    return 1


if __name__ == "__main__":
    sys.exit(_main())
