# RVC EasyGUI

A simple, easy-to-use voice timbre conversion / voice changer framework based on
**Retrieval-based Voice Conversion (RVC)**. Comes with a Gradio WebUI, a
command-line batch inference tool, and now a **safe `.safetensors` checkpoint
format** for distributing trained models.

[![Open In Colab](https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&color=525252)](https://colab.research.google.com/drive/1r4IRL0UA7JEoZ0ZK8PKfMyTIBHKpyhcw)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3%2B-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](./LICENSE)

---

## Table of contents

1. [Highlights](#highlights)
2. [Quick start](#quick-start)
3. [Using the WebUI](#using-the-webui)
4. [Inference from the command line](#inference-from-the-command-line)
5. [Inference from Python](#inference-from-python)
6. [Converting checkpoints to `.safetensors`](#converting-checkpoints-to-safetensors)
7. [Project layout](#project-layout)
8. [Configuration files](#configuration-files)
9. [Training](#training)
10. [Credits](#credits)
11. [Contributors](#contributors)

---

## Highlights

- **WebUI** built with Gradio — drag a model in, drop an audio file, hit
  *Convert*. Pitch shift, retrieval-index blend, RMS envelope, and voiceless
  protection are all exposed.
- **Single-file and batch CLI** (`tools/infer_cli.py`) — convert one file or
  recurse a whole library, with sensible overwrite/skip behavior.
- **New high-level inference API** (`infer/lib/infer_pack/inference.py`) — one
  class, one method, both checkpoint formats supported transparently.
- **New safe checkpoint format** — convert legacy `.pth` models to
  `.safetensors` (no arbitrary code execution on load, language-agnostic,
  ~50% smaller when cast to FP16) and back. Works in the WebUI, the CLI,
  and any Python script that uses `RVCInferer` / `process_ckpt`.
- **Format-agnostic loader** — every checkpoint-consuming code path now
  auto-detects `.pth` vs `.safetensors`, so you can mix and match freely.
- **Single unified config file** — all model architectures and realtime
  defaults now live in one `configs/config.json` (was 6 separate files).
  Loader falls back to the old layout for back-compat.
- **Internationalized UI** — bundled locales for English, Chinese
  (simplified / traditional / HK / SG), Japanese, French, Italian, Russian,
  Spanish, and Turkish.

---

## Quick start

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

> The repo already pins everything you need, including the new
> `safetensors>=0.4.0` package used by the checkpoint converter.

### 2. (Optional) Download pretrained assets

Use the helper script to pull HuBERT, the f0 predictors, the RMVPE model,
and other required assets into `assets/`:

```bash
# Linux / macOS
bash tools/dlmodels.sh
# Windows
tools\dlmodels.bat
```

### 3. Launch the WebUI

```bash
python app.py
# Open http://localhost:7865
```

### 4. Or convert some audio right now

```bash
python tools/infer_cli.py infer \
    --model_name my_model \
    --input_path  in.wav \
    --opt_path   out.wav \
    --f0up_key   0
```

---

## Using the WebUI

Run `python app.py` and open the printed URL in your browser. The main tabs:

| Tab              | What it does                                                                  |
|------------------|-------------------------------------------------------------------------------|
| **Model Inference** | Pick a checkpoint from `assets/weights/`, drop an audio file, choose pitch method, and convert. Both `.pth` and `.safetensors` files show up in the dropdown. |
| **Training**     | Pre-process a dataset, extract features, train a new model, and (new!) save the result as either `.pth` or `.safetensors`. |
| **UVR5**        | Vocal / accompaniment separation.                                            |
| **Merge**       | Mix two checkpoints with a slider — format-agnostic, output format inferred from the target filename. |

---

## Inference from the command line

### `tools/infer_cli.py` — drop-in CLI for the WebUI model

This is a thin wrapper around the existing `infer.modules.vc.modules.VC`
class. It expects models to live under `assets/weights/<name>.pth` (or
`.safetensors` — just pass the name without extension).

```bash
# Single file
python tools/infer_cli.py infer \
    --model_name   my_model \
    --input_path   in.wav \
    --opt_path     out.wav \
    --f0up_key     12 \
    --f0method     rmvpe \
    --index_path   my_model.index \
    --index_rate   0.75

# Batch convert a folder (non-recursive, skip existing)
python tools/infer_cli.py batch \
    --model_name  my_model \
    --input_dir   ./songs \
    --opt_dir     ./converted
```

### `python -m infer.lib.infer_pack.inference` — new, format-agnostic

A self-contained CLI that loads the checkpoint by **absolute path** (so it
works from anywhere) and accepts either `.pth` or `.safetensors`. It uses
the new [`RVCInferer`](#inference-from-python) API described below.

```bash
# Single file
python -m infer.lib.infer_pack.inference \
    --model   /path/to/my_model.safetensors \
    --device  cuda:0 \
    single \
    --input   in.wav \
    --output  out.wav \
    --f0up_key 0

# Batch convert a folder, recursing into subfolders
python -m infer.lib.infer_pack.inference \
    --model  /path/to/my_model.pth \
    --f0up_key 12 \
    batch \
    --input_dir   ./songs \
    --output_dir  ./converted \
    --recursive \
    --overwrite
```

---

## Inference from Python

The new module `infer.lib.infer_pack.inference` exposes a small,
well-documented `RVCInferer` class. It accepts either `.pth` or
`.safetensors` and builds the correct synthesizer architecture
(`SynthesizerTrnMs256NSFsid`, `…768NSFsid`, etc.) automatically based on the
checkpoint's `version` and `f0` flags.

```python
from infer.lib.infer_pack.inference import RVCInferer

# Load the model — .pth or .safetensors both work
inferer = RVCInferer(
    "assets/weights/singer_v2.safetensors",
    device="cuda:0",      # 'cpu', 'mps', 'xpu', 'cuda:1', ...
    is_half=True,         # FP16 on CUDA, FP32 on CPU (auto by default)
    index_path="singer_v2.index",  # optional retrieval index
)

# Inspect what we just loaded
print(inferer)
# -> RVCInferer(model='singer_v2.safetensors', version='v2',
#               if_f0=1, tgt_sr=48000, n_spk=1, device='cuda:0', half=True)

# Convert a single file
sr, audio = inferer.convert(
    "in.wav",
    "out.wav",          # if None, nothing is written and (sr, audio) is returned
    f0up_key=12,        # semitones; +12 = one octave up
    f0_method="rmvpe",  # 'harvest', 'pm', 'crepe', 'rmvpe', 'fcpe'
    index_rate=0.75,    # 0.0 disables the index, 1.0 fully trusts it
    filter_radius=3,
    rms_mix_rate=0.25,
    protect=0.33,
    sid=0,              # speaker id for multi-speaker models
)

# Batch convert a folder (relative structure preserved)
stats = inferer.convert_batch(
    "songs/",
    "converted/",
    recursive=True,
    overwrite=False,
    f0up_key=12,
    f0_method="rmvpe",
)
# -> {'total': 20, 'ok': 20, 'skipped': 0, 'failed': 0}
```

The class lazily builds the audio pipeline (librosa / torchaudio / faiss
back-ends) on the first `.convert()` call, so simply constructing an
`RVCInferer` (e.g. to inspect its metadata via `inferer.info()`) only
requires `torch` and `safetensors` — useful for model-management scripts.

---

## Converting checkpoints to `.safetensors`

RVC checkpoints have historically been distributed as `.pth` files, which
use Python's pickle protocol. **Loading a pickle file from an untrusted
source can execute arbitrary code on your machine.**

The safer alternative is
[`safetensors`](https://github.com/huggingface/safetensors): a format that
stores raw tensors with a small JSON metadata header. Loading a
`.safetensors` file cannot run code, the format is language-agnostic, and
the file size is often smaller than the equivalent `.pth`.

This repo now ships a complete round-trip converter for RVC checkpoints.

### The on-disk format

Inside the `.safetensors` file:

* **Tensors** — every entry of `ckpt["weight"]` is stored verbatim under
  the same key (`emb_g.weight`, `enc_p.conv_pre.weight`, etc.).
* **Metadata** — non-tensor checkpoint fields are JSON-serialized and
  stored in the safetensors `metadata` block under the prefix `rvc.`:

  | Field          | Stored as         | Example value                                |
  |----------------|-------------------|----------------------------------------------|
  | `config`       | `rvc.config`      | `[1025, 32, 192, …, 48000]` (JSON list)      |
  | `info`         | `rvc.info`        | `"trained on dataset X, 100 epoch"`          |
  | `sr`           | `rvc.sr`          | `"48k"`                                       |
  | `f0`           | `rvc.f0`          | `1` or `0`                                    |
  | `version`      | `rvc.version`     | `"v1"` or `"v2"`                              |

You can additionally attach arbitrary user metadata (`author`, `license`,
`note`, …) when writing a file.

### `tools/convert_safetensors.py` — the CLI converter

```bash
# Convert a single file (output defaults to <stem>.safetensors next to it)
python tools/convert_safetensors.py assets/weights/singer_v2.pth

# Explicit destination, FP16 cast (~50% smaller), with attribution metadata
python tools/convert_safetensors.py \
    assets/weights/singer_v2.pth \
    out/singer_v2.safetensors \
    --dtype fp16 \
    --author "alice" \
    --license "CC-BY-4.0" \
    --note   "100 epoch, dataset: 30min clean vocals"

# Batch convert every .pth under assets/weights/
python tools/convert_safetensors.py --batch assets/weights

# ...and write them all to ./converted/, recursing subfolders
python tools/convert_safetensors.py \
    --batch --recursive \
    --output-dir converted \
    --skip-existing \
    assets/weights

# Reverse direction: .safetensors back to .pth
python tools/convert_safetensors.py --to pth singer_v2.safetensors
```

### Python API

```python
from infer.lib.infer_pack.safetensors_utils import (
    convert_pth_to_safetensors,        # .pth  -> .safetensors
    convert_safetensors_to_pth,        # .safetensors -> .pth
    ckpt_to_safetensors,                # in-memory ckpt dict -> .safetensors
    safetensors_to_ckpt,                # .safetensors -> in-memory ckpt dict
    load_rvc_checkpoint,                # auto-detect .pth vs .safetensors
    list_checkpoint_files,              # walk a folder, bucket by format
)
import torch

# Convert a .pth to a half-precision .safetensors
convert_pth_to_safetensors(
    "assets/weights/singer_v2.pth",
    "out/singer_v2.fp16.safetensors",
    dtype=torch.float16,
    extra_metadata={"author": "alice"},
    overwrite=True,
)

# Auto-detect the format and load it uniformly
ckpt = load_rvc_checkpoint("assets/weights/singer_v2.safetensors")
print(ckpt["version"], ckpt["sr"], ckpt["f0"], len(ckpt["weight"]))
```

### What changed in the code base

| Location                                          | Change                                                                                          |
|---------------------------------------------------|-------------------------------------------------------------------------------------------------|
| `requirements.txt`                                | Added `safetensors>=0.4.0`.                                                                     |
| `infer/lib/infer_pack/safetensors_utils.py`       | **New file** — full conversion + loader API.                                                   |
| `infer/lib/infer_pack/inference.py`               | **New file** — high-level `RVCInferer` class + CLI entry point.                                 |
| `tools/convert_safetensors.py`                   | **New file** — `pth <-> safetensors` conversion CLI.                                           |
| `infer/lib/train/process_ckpt.py`                | `savee`, `extract_small_model`, `merge`, `change_info`, `show_info` all understand both formats. |
| `infer/modules/vc/utils.py`                      | Fixed a latent bug in `load_hubert` (`self.device` -> `config.device`, dropped the undefined `hubert_model` shadowing). |

---

## Project layout

```
projects/
├── app.py                      # Gradio WebUI entry point
├── infer-web.py                # Alternate WebUI entry point (legacy)
├── core.py                      # shared helpers used by the WebUI
├── configs/                     # ★ Single config.json (model archs + realtime defaults) + Config class
├── assets/
│   ├── weights/                 # drop trained models here (.pth or .safetensors)
│   ├── pretrained/              # base pretrained checkpoints (v1)
│   ├── pretrained_v2/           # base pretrained checkpoints (v2)
│   ├── hubert/                  # HuBERT content encoder
│   ├── uvr5_weights/            # vocal-removal models
│   └── rmvpe/                   # RMVPE pitch extractor
├── audios/                      # bundled sample audios
├── i18n/                        # locale strings + i18n loader
├── infer/
│   ├── lib/
│   │   ├── infer_pack/          # ★ RVC inference package
│   │   │   ├── models.py        # synthesizer class definitions
│   │   │   ├── modules.py       # resblocks, upsampling blocks
│   │   │   ├── attentions.py    # attention + encoder
│   │   │   ├── commons.py       # init helpers, padding utils
│   │   │   ├── transforms.py    # FFT / piecewise-linear transforms
│   │   │   ├── models_onnx.py   # ONNX-exportable variant
│   │   │   ├── onnx_inference.py
│   │   │   ├── safetensors_utils.py  ★ NEW: pth <-> safetensors API
│   │   │   ├── inference.py     ★ NEW: high-level RVCInferer + CLI
│   │   │   └── modules/F0Predictor/...
│   │   ├── predictor/           # pitch extractors (crepe/rmvpe/fcpe/swipe)
│   │   ├── train/               # training-time utils (mel/losses/ckpt processing)
│   │   ├── audio.py
│   │   ├── rmvpe.py
│   │   ├── slicer2.py
│   │   └── uvr.py
│   └── modules/
│       ├── vc/                  # VC wrapper used by the WebUI
│       ├── onnx/                # ONNX export helpers
│       ├── train/               # preprocessing, feature extraction, train loop
│       └── ipex/                # Intel Extension for PyTorch (optional)
├── tools/
│   ├── infer_cli.py             # original CLI (single + batch)
│   ├── convert_safetensors.py   ★ NEW: pth <-> safetensors converter
│   ├── export_onnx.py
│   ├── onnx_inference_demo.py
│   ├── rvc_for_realtime.py      # realtime voice-changer inference
│   ├── dlmodels.sh / .bat       # download pretrained assets
│   └── infer/                   # extra training / index helpers
└── requirements.txt
```

★ = added or significantly changed in this update.

---

## Configuration files

All configuration lives in a single, human-readable file:
[`configs/config.json`](./configs/config.json). It has two top-level
sections:

```jsonc
{
  "schema_version": 1,
  "models": {
    "v1/32k": { "train": {...}, "data": {...}, "model": {...} },
    "v1/40k": { ... },
    "v1/48k": { ... },
    "v2/32k": { ... },
    "v2/48k": { ... }
  },
  "realtime": {
    "pth_path": "assets/weights/kikiV1.pth",
    "index_path": "logs/kikiV1.index",
    "pitch": 12.0,
    "f0method": "rmvpe",
    ...
  }
}
```

* **`models`** — per-architecture hyper-parameters, keyed by
  `<version>/<sample_rate>`. Each entry has the same shape as the old
  per-file configs (`train` / `data` / `model` blocks). When you load a
  checkpoint, the values stored in the checkpoint's `config` field
  (which mirrors one of these entries) are used to rebuild the
  synthesizer — you don't need to keep the JSON alongside the model.
* **`realtime`** — default values for the realtime voice-changer tab
  (`tools/rvc_for_realtime.py`). The WebUI passes runtime overrides
  via function arguments; these are only used as initial defaults.

This layout replaces the historical six-file structure:

```
configs/
├── v1/32k.json   v1/40k.json   v1/48k.json     ← removed
├── v2/32k.json   v2/48k.json                   ← removed
└── config.json                                ← repurposed (now the unified file above)
```

For backwards compatibility, `configs/config.py:load_config_json()`
falls back to the old per-file layout if `configs/config.json` is
missing or has no `models` section — so users on older checkouts still
load successfully.

The `Config` class in `configs/config.py` is responsible for:
* loading the unified config and exposing `cfg.json_config["v1/32k"]`,
  `cfg.json_config["v2/48k"]`, etc. (same shape as before);
* picking the device (CUDA / MPS / XPU / CPU);
* deciding whether FP16 is safe on the chosen device (and flipping
  `train.fp16_run` to `False` for every architecture when it isn't, via
  `use_fp32_config()`);
* exposing the `x_pad / x_query / x_center / x_max` chunking parameters
  used by the inference `Pipeline`.

A standalone module-level helper, `load_realtime_config()`, returns
the `realtime` section directly without constructing the full `Config`
singleton — useful for tools that only need the realtime defaults.

---

## Training

Training lives under `infer/modules/train/` and is wired up through the
WebUI's "Training" tab. The high-level flow is:

1. **Pre-process** the dataset (`infer/modules/train/preprocess.py`)
   → 16 kHz wav chunks.
2. **Extract pitch** with one of the F0 predictors in
   `infer/lib/predictor/` (`rmvpe.py`, `crepe.py`, `fcpe.py`, `swipe.py`,
   `pyworld.py`, `pm` via `parselmouth`).
3. **Extract HuBERT features** (`infer/modules/train/extract_feature_print.py`)
   → 256-dim content vectors.
4. **Train** (`infer/modules/train/train.py`) — the trainer periodically
   calls `infer/lib/train/process_ckpt.savee(...)`, which now accepts an
   optional `fmt="safetensors"` argument so you can save your finished
   model directly in the safer format.

The checkpoint produced at step 4 is ready to drop into `assets/weights/`
and use immediately — either from the WebUI or from any of the CLIs.

---

## Credits

This project builds on top of outstanding open-source work. Thank you to
all the original authors.

| Project                                       | Used for                                |
|-----------------------------------------------|-----------------------------------------|
| [ContentVec](https://github.com/auspicious3000/contentvec/)  | HuBERT feature extractor                |
| [VITS](https://github.com/jaywalnut310/vits)                  | Synthesizer backbone                    |
| [HiFi-GAN](https://github.com/jik876/hifi-gan)               | Neural vocoder                           |
| [Gradio](https://github.com/gradio-app/gradio)               | WebUI                                   |
| [FFmpeg](https://github.com/FFmpeg/FFmpeg)                   | Audio I/O                               |
| [Ultimate Vocal Remover](https://github.com/Anjok07/ultimatevocalremovergui) | UVR5 stem separation |
| [pymss-project/pymss](https://github.com/pymss-project/pymss) | RTP/voice streaming helpers             |
| [audio-slicer](https://github.com/openvpi/audio-slicer)     | Automatic audio chunking                |
| [RMVPE](https://github.com/Dream-High/RMVPE)                 | Pitch (f0) extraction                   |
| [safetensors](https://github.com/huggingface/safetensors)    | Safe checkpoint serialization           |

The RMVPE pretrained model was trained and tested by
[yxlllc](https://github.com/yxlllc/RMVPE) and
[RVC-Boss](https://github.com/RVC-Boss).

---

## Contributors

Thanks to everyone who has contributed to this project.

<a href="https://github.com/asukaa2/projects/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=Ezui0/projects" />
</a>

---

## License

This project is released under the MIT License — see [LICENSE](./LICENSE).
