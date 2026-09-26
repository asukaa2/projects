"""
infer/lib/infer_pack/safetensors_utils.py
=========================================

Safetensors <-> pth conversion and loader utilities for RVC checkpoints.

RVC checkpoints historically ship as a single ``.pth`` file produced by
``torch.save`` with the following structure::

    {
        "weight":  OrderedDict[str, torch.Tensor],   # model state_dict
        "config":  list,                              # synthesizer ctor args
        "info":    str,                              # free-form description
        "sr":      str,                               # "32k" | "40k" | "48k"
        "f0":      int,                               # 0 (no pitch) | 1 (pitch)
        "version": str,                               # "v1" | "v2"
    }

``torch.save`` uses Python's pickle protocol, which means **loading a
``.pth`` file is unsafe** — arbitrary code can execute during deserialization.
``safetensors`` avoids this by storing raw tensors with a JSON metadata
header, and is therefore the recommended format for distributing RVC
weights.

This module provides four operations:

* :func:`ckpt_to_safetensors`  — write an RVC checkpoint dict to ``.safetensors``
* :func:`safetensors_to_ckpt`  — read an RVC checkpoint back from ``.safetensors``
* :func:`convert_pth_to_safetensors` — one-shot file converter
* :func:`load_rvc_checkpoint` — auto-detect ``.pth`` vs ``.safetensors`` and
  return a unified checkpoint dict

The on-disk layout we use is intentionally simple and self-describing:

* Tensor keys are written verbatim from ``ckpt["weight"]``.
* Non-tensor fields (``config``, ``info``, ``sr``, ``f0``, ``version``,
  ``info``) are JSON-serialized and stored in the safetensors
  ``metadata`` block under the prefix ``rvc.`` (e.g. ``rvc.config``).

This keeps the file 100 % compatible with the safetensors spec and
loadable from any language binding, not just PyTorch.
"""

from __future__ import annotations

import json
import logging
import os
from collections import OrderedDict
from typing import Any, Dict, Optional, Union

import torch

try:
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    _HAS_SAFETENSORS = True
except ImportError:  # pragma: no cover - optional, surfaced at runtime
    _HAS_SAFETENSORS = False
    safe_open = None
    load_file = None
    save_file = None

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: File extensions we recognize as RVC checkpoints (case-insensitive).
PTH_EXTS = (".pth", ".pt")
SAFETENSORS_EXTS = (".safetensors", ".st")

#: Metadata key prefix used inside the safetensors file.
META_PREFIX = "rvc."

#: Required keys inside an RVC checkpoint dict.
REQUIRED_CKPT_KEYS = ("weight", "config", "sr", "f0", "version")

#: Keys that must be JSON-serializable (i.e. *not* tensors) and are stored in
#: the safetensors ``metadata`` block.
META_KEYS = ("config", "info", "sr", "f0", "version")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_safetensors() -> None:
    """Raise a helpful error if the optional ``safetensors`` package is missing."""
    if not _HAS_SAFETENSORS:
        raise ImportError(
            "The 'safetensors' package is required for this operation. "
            "Install it with:  pip install safetensors>=0.4.0"
        )


def _is_safetensors_path(path: str) -> bool:
    return path.lower().endswith(SAFETENSORS_EXTS)


def _is_pth_path(path: str) -> bool:
    return path.lower().endswith(PTH_EXTS)


def _default_output_path(pth_path: str) -> str:
    """``foo/bar.pth`` -> ``foo/bar.safetensors`` (handles ``.pt`` too)."""
    base, _ = os.path.splitext(pth_path)
    return base + ".safetensors"


def _normalize_weight_dict(weight: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
    """Return an OrderedDict of contiguous CPU tensors, ready for safetensors."""
    out: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for k, v in weight.items():
        if not isinstance(v, torch.Tensor):
            raise TypeError(
                f"weight[{k!r}] is {type(v).__name__}, expected torch.Tensor. "
                "Only tensors may be stored in the 'weight' block."
            )
        # safetensors requires contiguous CPU tensors; we never mutate the
        # caller's tensors in place.
        t = v.detach().cpu()
        if not t.is_contiguous():
            t = t.contiguous()
        out[k] = t
    return out


def _serialize_metadata(ckpt: Dict[str, Any]) -> Dict[str, str]:
    """Build the safetensors ``metadata`` dict from non-tensor checkpoint fields."""
    meta: Dict[str, str] = {}
    for key in META_KEYS:
        if key not in ckpt:
            continue
        try:
            meta[META_PREFIX + key] = json.dumps(ckpt[key], ensure_ascii=False)
        except TypeError as exc:  # pragma: no cover - defensive
            raise TypeError(
                f"Cannot JSON-serialize checkpoint field {key!r}: {exc}. "
                "Only JSON-compatible types (lists, dicts, str, int, float, bool, None) "
                "are allowed in checkpoint metadata."
            ) from exc
    return meta


def _deserialize_metadata(meta: Dict[str, str]) -> Dict[str, Any]:
    """Reverse of :func:`_serialize_metadata`."""
    out: Dict[str, Any] = {}
    prefix_len = len(META_PREFIX)
    for k, v in meta.items():
        if not k.startswith(META_PREFIX):
            # Forward-compatible: ignore unknown metadata keys silently.
            continue
        key = k[prefix_len:]
        try:
            out[key] = json.loads(v)
        except (json.JSONDecodeError, TypeError):
            # Keep raw string if JSON parse fails (older format or hand-edited).
            out[key] = v
    return out


# ---------------------------------------------------------------------------
# Core API: checkpoint dict <-> safetensors file
# ---------------------------------------------------------------------------

def ckpt_to_safetensors(
    ckpt: Dict[str, Any],
    output_path: str,
    *,
    metadata: Optional[Dict[str, str]] = None,
    dtype: Optional[torch.dtype] = None,
) -> str:
    """Serialize an RVC checkpoint dict to a ``.safetensors`` file.

    Parameters
    ----------
    ckpt:
        Checkpoint dict with at least the keys in :data:`REQUIRED_CKPT_KEYS`.
        ``ckpt["weight"]`` must be a mapping of ``str -> torch.Tensor``;
        everything else must be JSON-serializable.
    output_path:
        Destination file path. Parent directories are auto-created.
    metadata:
        Optional extra ``str -> str`` metadata merged into the file's
        metadata block (e.g. ``{"author": "alice", "license": "MIT"}``).
        Keys starting with ``rvc.`` are reserved.
    dtype:
        If given (e.g. ``torch.float16``), every tensor in ``ckpt["weight"]``
        is cast to this dtype before being written. Use this to produce a
        half-precision distribution file from a float32 training checkpoint.

    Returns
    -------
    str
        The absolute path of the written file.
    """
    _ensure_safetensors()

    missing = [k for k in REQUIRED_CKPT_KEYS if k not in ckpt]
    if missing:
        raise KeyError(
            f"Checkpoint is missing required keys: {missing}. "
            f"Got keys: {list(ckpt.keys())}"
        )

    weight = _normalize_weight_dict(ckpt["weight"])
    if dtype is not None:
        weight = OrderedDict((k, v.to(dtype)) for k, v in weight.items())

    # Build metadata: rvc.* fields first, then user-supplied extras.
    file_meta = _serialize_metadata(ckpt)
    if metadata:
        for k, v in metadata.items():
            if k.startswith(META_PREFIX):
                raise ValueError(
                    f"User metadata key {k!r} collides with reserved prefix "
                    f"{META_PREFIX!r}. Please rename it."
                )
            if not isinstance(v, str):
                raise TypeError(
                    f"metadata[{k!r}] must be a string, got {type(v).__name__}."
                )
            file_meta[k] = v

    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    save_file(weight, output_path, metadata=file_meta)
    logger.info(
        "Wrote safetensors checkpoint -> %s  (%d tensors, %s)",
        output_path, len(weight), dtype if dtype is not None else "kept dtype",
    )
    return os.path.abspath(output_path)


def safetensors_to_ckpt(path: str) -> Dict[str, Any]:
    """Load an RVC checkpoint from a ``.safetensors`` file.

    Returns a dict with the same shape as the one produced by
    ``torch.load`` on a ``.pth`` file, so existing code paths can use it
    as a drop-in replacement.
    """
    _ensure_safetensors()
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Safetensors file not found: {path}")

    meta: Dict[str, str] = {}
    weight: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    with safe_open(path, framework="pt", device="cpu") as f:
        meta = dict(f.metadata() or {})
        for k in f.keys():
            weight[k] = f.get_tensor(k)

    ckpt: Dict[str, Any] = {"weight": weight}
    ckpt.update(_deserialize_metadata(meta))

    # Defaults for forward-compatibility with files written before
    # some fields existed.
    ckpt.setdefault("info", "")
    ckpt.setdefault("f0", 1)
    ckpt.setdefault("version", "v1")
    ckpt.setdefault("sr", "48k")
    ckpt.setdefault("config", [])

    return ckpt


# ---------------------------------------------------------------------------
# File-level helpers
# ---------------------------------------------------------------------------

def convert_pth_to_safetensors(
    pth_path: str,
    output_path: Optional[str] = None,
    *,
    dtype: Optional[torch.dtype] = None,
    extra_metadata: Optional[Dict[str, str]] = None,
    overwrite: bool = False,
) -> str:
    """Convert a single ``.pth`` RVC checkpoint to a ``.safetensors`` file.

    Parameters
    ----------
    pth_path:
        Path to the source ``.pth`` (or ``.pt``) checkpoint.
    output_path:
        Destination ``.safetensors`` path. If ``None``, the source stem is
        reused with a ``.safetensors`` suffix.
    dtype:
        Optional cast for the weight tensors (e.g. ``torch.float16``).
    extra_metadata:
        Extra ``str -> str`` metadata appended to the file.
    overwrite:
        If ``False`` and the destination exists, raise ``FileExistsError``.

    Returns
    -------
    str
        Absolute path of the produced file.
    """
    _ensure_safetensors()
    if not os.path.isfile(pth_path):
        raise FileNotFoundError(f"Source .pth not found: {pth_path}")

    if output_path is None:
        output_path = _default_output_path(pth_path)
    output_path = os.path.abspath(output_path)

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    logger.info("Loading pth checkpoint: %s", pth_path)
    ckpt = torch.load(pth_path, map_location="cpu")

    # RVC training checkpoints sometimes store the model under "model"
    # instead of "weight" — normalize before writing.
    if "weight" not in ckpt and "model" in ckpt:
        ckpt["weight"] = ckpt.pop("model")

    return ckpt_to_safetensors(
        ckpt,
        output_path,
        metadata=extra_metadata,
        dtype=dtype,
    )


def convert_safetensors_to_pth(
    safetensors_path: str,
    output_path: Optional[str] = None,
    *,
    overwrite: bool = False,
) -> str:
    """Inverse of :func:`convert_pth_to_safetensors`.

    Useful for users who want to fall back to ``.pth`` tooling (e.g. older
    Gradio tabs that hard-code the ``.pth`` extension).
    """
    if not os.path.isfile(safetensors_path):
        raise FileNotFoundError(f"Source .safetensors not found: {safetensors_path}")

    if output_path is None:
        base, _ = os.path.splitext(safetensors_path)
        output_path = base + ".pth"
    output_path = os.path.abspath(output_path)

    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(
            f"Destination already exists: {output_path}. "
            "Pass overwrite=True to replace it."
        )

    ckpt = safetensors_to_ckpt(safetensors_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    torch.save(ckpt, output_path)
    logger.info("Wrote pth checkpoint -> %s", output_path)
    return output_path


def load_rvc_checkpoint(path: str) -> Dict[str, Any]:
    """Auto-detect ``.pth`` vs ``.safetensors`` and return a checkpoint dict.

    The returned dict is normalized so that callers do not need to branch
    on file format. Specifically:

    * ``ckpt["weight"]`` is always an ``OrderedDict[str, torch.Tensor]``.
    * If the source is a raw training checkpoint (``ckpt["model"]``),
      it is renamed to ``ckpt["weight"]`` and ``enc_q.*`` tensors are
      dropped (they are only used by the trainer's posterior and bloat
      the file by ~30 %).
    * ``config``, ``info``, ``sr``, ``f0``, ``version`` are always present
      (with sensible defaults if missing).
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if _is_safetensors_path(path):
        return safetensors_to_ckpt(path)

    if _is_pth_path(path):
        ckpt = torch.load(path, map_location="cpu")
        if "weight" not in ckpt and "model" in ckpt:
            # Raw training checkpoint: strip enc_q and rename.
            raw = ckpt.pop("model")
            weight = OrderedDict(
                (k, v) for k, v in raw.items() if "enc_q" not in k
            )
            ckpt["weight"] = weight
        ckpt.setdefault("info", "")
        ckpt.setdefault("f0", 1)
        ckpt.setdefault("version", "v1")
        ckpt.setdefault("sr", "48k")
        ckpt.setdefault("config", [])
        return ckpt

    raise ValueError(
        f"Unsupported checkpoint format: {path!r}. "
        f"Expected one of {PTH_EXTS + SAFETENSORS_EXTS}."
    )


def list_checkpoint_files(folder: str) -> Dict[str, list]:
    """Walk ``folder`` and bucket RVC checkpoint files by format.

    Returns ``{"pth": [...], "safetensors": [...]}`` with absolute paths.
    """
    out: Dict[str, list] = {"pth": [], "safetensors": []}
    for root, _dirs, files in os.walk(folder):
        for name in files:
            full = os.path.join(root, name)
            if _is_safetensors_path(name):
                out["safetensors"].append(os.path.abspath(full))
            elif _is_pth_path(name):
                out["pth"].append(os.path.abspath(full))
    return out


__all__ = [
    "PTH_EXTS",
    "SAFETENSORS_EXTS",
    "META_PREFIX",
    "REQUIRED_CKPT_KEYS",
    "ckpt_to_safetensors",
    "safetensors_to_ckpt",
    "convert_pth_to_safetensors",
    "convert_safetensors_to_pth",
    "load_rvc_checkpoint",
    "list_checkpoint_files",
]
