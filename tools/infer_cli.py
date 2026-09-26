#!/usr/bin/env python3
"""
RVC Inference CLI

Usage:
    rvc-cli infer --input_path song.wav --model_name my_model --opt_path out.wav
    rvc-cli batch --input_dir ./songs --model_name my_model --opt_dir ./output
"""

import argparse
import os
import sys
import glob
from pathlib import Path

now_dir = os.getcwd()
sys.path.append(now_dir)

from dotenv import load_dotenv
from scipy.io import wavfile

from configs.config import Config
from infer.modules.vc.modules import VC


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

AUDIO_EXTS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac")


def str2bool(value):
    """Robust bool parser for argparse."""
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if value.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got {value!r}")


def existing_file(path: str) -> str:
    if not os.path.isfile(path):
        raise argparse.ArgumentTypeError(f"File not found: {path}")
    return path


def existing_dir(path: str) -> str:
    if not os.path.isdir(path):
        raise argparse.ArgumentTypeError(f"Directory not found: {path}")
    return path


def ensure_parent(path: str) -> str:
    """Make sure the parent directory for an output path exists."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rvc-cli",
        description="RVC voice-conversion CLI (infer / batch).",
    )
    parser.add_argument(
        "-v", "--version", action="version", version="rvc-cli 1.0.0"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # ---- shared inference options (parent parser) -------------------------
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model_name", required=True,
                        help="Model name stored in assets/weight_root")
    common.add_argument("--index_path", default="",
                        help="Path to the .index file (optional)")
    common.add_argument("--f0up_key", type=int, default=0,
                        help="Pitch shift in semitones (default: 0)")
    common.add_argument("--f0method", default="harvest",
                        choices=["harvest", "pm", "crepe", "rmvpe"],
                        help="Pitch-extraction algorithm (default: harvest)")
    common.add_argument("--index_rate", type=float, default=0.66,
                        help="Index blend rate (default: 0.66)")
    common.add_argument("--filter_radius", type=int, default=3,
                        help="Median filter radius (default: 3)")
    common.add_argument("--resample_sr", type=int, default=0,
                        help="Target sample rate, 0 = keep (default: 0)")
    common.add_argument("--rms_mix_rate", type=float, default=1.0,
                        help="RMS mix rate (default: 1.0)")
    common.add_argument("--protect", type=float, default=0.33,
                        help="Voiceless protection (default: 0.33)")
    common.add_argument("--device", default=None,
                        help="cuda / cpu (default: from config)")
    common.add_argument("--is_half", type=str2bool, default=None,
                        help="Use FP16 (True/False, default: from config)")

    # ---- infer (single file) ---------------------------------------------
    p_infer = sub.add_parser("infer", parents=[common],
                             help="Convert a single audio file")
    p_infer.add_argument("--input_path", required=True, type=existing_file,
                         help="Input audio file")
    p_infer.add_argument("--opt_path", required=True, type=ensure_parent,
                         help="Output .wav file path")
    p_infer.add_argument("--overwrite", type=str2bool, default=True,
                         help="Overwrite output if it exists (default: True)")

    # ---- batch (folder) ---------------------------------------------------
    p_batch = sub.add_parser("batch", parents=[common],
                             help="Convert every audio file in a folder")
    p_batch.add_argument("--input_dir", required=True, type=existing_dir,
                         help="Folder with input audio files")
    p_batch.add_argument("--opt_dir", required=True,
                         help="Folder where converted files are written")
    p_batch.add_argument("--recursive", type=str2bool, default=False,
                         help="Recurse into subfolders (default: False)")
    p_batch.add_argument("--overwrite", type=str2bool, default=False,
                         help="Overwrite existing outputs (default: False)")

    return parser


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

def make_vc(args) -> VC:
    config = Config()
    if args.device:
        config.device = args.device
    if args.is_half is not None:
        config.is_half = args.is_half

    vc = VC(config)
    vc.get_vc(args.model_name)
    return vc


def infer_one(vc: VC, args, input_path: str, opt_path: str) -> None:
    """Run a single voice conversion."""
    print(f"[+] {input_path}  ->  {opt_path}")
    _, wav_opt = vc.vc_single(
        0,                       # sid
        input_path,
        args.f0up_key,
        None,                    # f0_file
        args.f0method,
        args.index_path or None,
        None,                    # file_index2
        args.index_rate,
        args.filter_radius,
        args.resample_sr,
        args.rms_mix_rate,
        args.protect,
    )
    wavfile.write(opt_path, wav_opt[0], wav_opt[1])


def collect_audio_files(folder: str, recursive: bool) -> list:
    pattern = "**/*" if recursive else "*"
    files = []
    for path in glob.iglob(os.path.join(folder, pattern), recursive=recursive):
        if os.path.isfile(path) and path.lower().endswith(AUDIO_EXTS):
            files.append(path)
    return sorted(files)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_infer(args) -> int:
    if os.path.exists(args.opt_path) and not args.overwrite:
        print(f"[!] Output exists, skipping (use --overwrite True): {args.opt_path}")
        return 1

    vc = make_vc(args)
    infer_one(vc, args, args.input_path, args.opt_path)
    print("[✓] Done.")
    return 0


def cmd_batch(args) -> int:
    files = collect_audio_files(args.input_dir, args.recursive)
    if not files:
        print(f"[!] No audio files found in {args.input_dir}")
        return 1

    os.makedirs(args.opt_dir, exist_ok=True)
    vc = make_vc(args)

    total, ok, skipped, failed = len(files), 0, 0, 0
    for i, in_path in enumerate(files, 1):
        rel = os.path.relpath(in_path, args.input_dir)
        stem = Path(rel).with_suffix("")
        out_path = os.path.join(args.opt_dir, f"{stem}.wav")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        if os.path.exists(out_path) and not args.overwrite:
            print(f"[{i}/{total}] skip (exists): {out_path}")
            skipped += 1
            continue

        try:
            infer_one(vc, args, in_path, out_path)
            ok += 1
        except Exception as exc:                      # noqa: BLE001
            print(f"[{i}/{total}] FAILED {in_path}: {exc}")
            failed += 1

    print(f"\n[✓] Batch complete: {ok} ok, {skipped} skipped, {failed} failed "
          f"(of {total}).")
    return 0 if failed == 0 else 2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args()

    # argparse already guarantees a subcommand thanks to required=True
    if args.command == "infer":
        return cmd_infer(args)
    if args.command == "batch":
        return cmd_batch(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
