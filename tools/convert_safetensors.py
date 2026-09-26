#!/usr/bin/env python3
"""
tools/convert_safetensors.py
============================

CLI tool to convert RVC checkpoints between ``.pth`` and ``.safetensors``.

The safetensors format is safer than ``.pth`` (no arbitrary code execution
on load), smaller on disk in many cases, and loadable from any language
binding — making it the recommended distribution format for sharing RVC
models.

Supported conversions
---------------------
* ``.pth``      → ``.safetensors``  (recommended for distribution)
* ``.safetensors`` → ``.pth``        (back-compat with legacy tooling)

Examples
--------
Single file:

.. code-block:: bash

    # produces ./singer_v2.safetensors
    python tools/convert_safetensors.py assets/weights/singer_v2.pth

    # explicit destination + cast to FP16 to halve file size
    python tools/convert_safetensors.py singer.pth out/singer_fp16.safetensors --dtype fp16

Whole folder (batch mode):

.. code-block:: bash

    # convert every .pth under assets/weights to .safetensors next to it
    python tools/convert_safetensors.py --batch assets/weights

    # write all converted files into ./converted/, recurse subfolders,
    # skip files that already have a .safetensors sibling
    python tools/convert_safetensors.py --batch --recursive \\
        --output-dir converted --skip-existing assets/weights

Reverse direction (safetensors -> pth):

.. code-block:: bash

    python tools/convert_safetensors.py --to pth singer.safetensors
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Make the project root importable so ``infer.lib.infer_pack.*`` resolves.
_NOW_DIR = os.getcwd()
if _NOW_DIR not in sys.path:
    sys.path.append(_NOW_DIR)

import torch  # noqa: E402  (after sys.path tweak)

from infer.lib.infer_pack.safetensors_utils import (  # noqa: E402
    PTH_EXTS,
    SAFETENSORS_EXTS,
    convert_pth_to_safetensors,
    convert_safetensors_to_pth,
    list_checkpoint_files,
)

logger = logging.getLogger("convert_safetensors")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DTYPE_MAP = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "float16": torch.float16,
    "bf16": torch.bfloat16,
}


def _resolve_dtype(name: str) -> torch.dtype:
    if name not in _DTYPE_MAP:
        raise argparse.ArgumentTypeError(
            f"Unknown dtype {name!r}. Choices: {sorted(_DTYPE_MAP)}"
        )
    return _DTYPE_MAP[name]


def _extra_metadata_from_args(args) -> dict:
    meta = {}
    if args.author:
        meta["author"] = args.author
    if args.license:
        meta["license"] = args.license
    if args.note:
        meta["note"] = args.note
    return meta


def _convert_one(
    src: str,
    dst: str,
    *,
    to: str,
    dtype,
    overwrite: bool,
    extra_metadata: dict,
) -> str:
    """Dispatch a single-file conversion to the right helper."""
    if to == "safetensors":
        return convert_pth_to_safetensors(
            src,
            dst,
            dtype=dtype,
            extra_metadata=extra_metadata or None,
            overwrite=overwrite,
        )
    return convert_safetensors_to_pth(src, dst, overwrite=overwrite)


def _default_dst(src: str, to: str) -> str:
    base, _ = os.path.splitext(src)
    return base + (".safetensors" if to == "safetensors" else ".pth")


def _is_target_ext(path: str, to: str) -> bool:
    """True if ``path`` is already in the target format."""
    if to == "safetensors":
        return path.lower().endswith(SAFETENSORS_EXTS)
    return path.lower().endswith(PTH_EXTS)


def _is_source_ext(path: str, to: str) -> bool:
    """True if ``path`` has the source extension for the requested direction."""
    if to == "safetensors":
        return path.lower().endswith(PTH_EXTS)
    return path.lower().endswith(SAFETENSORS_EXTS)


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------

def cmd_single(args) -> int:
    src = os.path.abspath(args.path)
    dst = args.output or _default_dst(src, args.to)
    dst = os.path.abspath(dst)

    if os.path.exists(dst) and not args.overwrite:
        logger.error("Destination exists (use --overwrite): %s", dst)
        return 1

    extra = _extra_metadata_from_args(args)
    t0 = time.time()
    out = _convert_one(
        src, dst,
        to=args.to, dtype=args.dtype,
        overwrite=args.overwrite, extra_metadata=extra,
    )
    size_mb = os.path.getsize(out) / 1024 / 1024
    logger.info("Done in %.2fs -> %s  (%.2f MB)", time.time() - t0, out, size_mb)
    return 0


def cmd_batch(args) -> int:
    src_dir = os.path.abspath(args.path)
    if not os.path.isdir(src_dir):
        logger.error("Not a directory: %s", src_dir)
        return 1

    files = list_checkpoint_files(src_dir)
    # Pick the right bucket for the requested direction.
    source_bucket = "pth" if args.to == "safetensors" else "safetensors"
    candidates = files[source_bucket]
    if not candidates:
        logger.warning(
            "No %s files found under %s", source_bucket, src_dir,
        )
        return 0

    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    extra = _extra_metadata_from_args(args)
    total, ok, skipped, failed = len(candidates), 0, 0, 0
    for i, src in enumerate(candidates, 1):
        # Build the destination path. When --output-dir is set, mirror the
        # relative path under it; otherwise write next to the source.
        if args.output_dir:
            rel = os.path.relpath(src, src_dir)
            base, _ = os.path.splitext(rel)
            dst = os.path.join(args.output_dir, base + (
                ".safetensors" if args.to == "safetensors" else ".pth"
            ))
        else:
            dst = _default_dst(src, args.to)

        if not args.overwrite and os.path.exists(dst):
            logger.info("[%d/%d] skip (exists): %s", i, total, dst)
            skipped += 1
            continue

        if args.skip_existing:
            # When --skip-existing is set we already bail out above
            # because the destination file exists.
            pass

        t0 = time.time()
        try:
            out = _convert_one(
                src, dst,
                to=args.to, dtype=args.dtype,
                overwrite=args.overwrite, extra_metadata=extra,
            )
            ok += 1
            size_mb = os.path.getsize(out) / 1024 / 1024
            logger.info(
                "[%d/%d] ok in %.2fs (%.2f MB): %s",
                i, total, time.time() - t0, size_mb, out,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("[%d/%d] FAILED %s: %s", i, total, src, exc)
            failed += 1

    logger.info(
        "Batch done: %d ok, %d skipped, %d failed (of %d).",
        ok, skipped, failed, total,
    )
    return 0 if failed == 0 else 2


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="convert_safetensors",
        description=(
            "Convert RVC checkpoints between .pth and .safetensors. "
            "Safetensors is the recommended format for distribution: it "
            "loads safely (no pickle), is language-agnostic, and stores "
            "checkpoint metadata as JSON."
        ),
    )
    p.add_argument(
        "--to",
        choices=("safetensors", "pth"),
        default="safetensors",
        help="Target format (default: safetensors).",
    )
    p.add_argument(
        "--dtype",
        type=_resolve_dtype,
        default=None,
        help="Cast tensors to this dtype when writing safetensors "
             "(fp16/fp32/bf16). Default: keep source dtype. "
             "Ignored when --to pth.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the destination file if it already exists.",
    )
    p.add_argument(
        "--author", default=None,
        help="Optional 'author' metadata string written into the file.",
    )
    p.add_argument(
        "--license", default=None,
        help="Optional 'license' metadata string written into the file.",
    )
    p.add_argument(
        "--note", default=None,
        help="Optional free-form 'note' metadata string written into the file.",
    )
    p.add_argument(
        "--batch", action="store_true",
        help="Treat PATH as a directory and convert every checkpoint in it.",
    )
    p.add_argument(
        "--recursive", action="store_true",
        help="With --batch, recurse into subfolders.",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="With --batch, write converted files into this directory "
             "(relative structure preserved). Default: write next to each source.",
    )
    p.add_argument(
        "--skip-existing", action="store_true",
        help="With --batch, skip sources that already have a converted sibling.",
    )
    p.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging.",
    )
    p.add_argument(
        "path",
        help="Source file (single mode) or directory (--batch mode).",
    )
    p.add_argument(
        "output", nargs="?", default=None,
        help="Destination file (single mode). Ignored in --batch mode.",
    )
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.batch:
        return cmd_batch(args)
    return cmd_single(args)


if __name__ == "__main__":
    sys.exit(main())
