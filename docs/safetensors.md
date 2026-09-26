# Safetensors model format

This document is a focused reference for the `.safetensors` checkpoint
support added to this repository. For the high-level overview, see the
[main README](../README.md).

## Why safetensors?

RVC checkpoints have traditionally been distributed as `.pth` files
produced by `torch.save`, which uses Python's pickle protocol. **Pickle
deserialization can run arbitrary code** — loading a `.pth` from an
untrusted source can compromise your machine.

`safetensors` is a modern alternative:

- **Safe by design.** Files contain only tensors plus a small JSON
  metadata header. No code ever runs when loading.
- **Language-agnostic.** The format is supported by PyTorch, TensorFlow,
  JAX, NumPy, Rust, Go, etc. You can load RVC weights from any tool.
- **Faster load.** Memory-mapped, no pickling overhead.
- **Often smaller on disk.** Especially when combined with a FP16 cast.

## The on-disk layout

```
┌────────────────────────────────────────────────────────────┐
│ 8-byte little-endian uint64:  size of the JSON header       │
├────────────────────────────────────────────────────────────┤
│ JSON header (UTF-8)                                          │
│ {                                                            │
│   "emb_g.weight":          {"dtype":"F32","shape":[1,256],  │
│                             "data_offsets":[0,1024]},        │
│   "enc_p.conv_pre.weight": {"dtype":"F32","shape":[192,256,7]│
│                             "data_offsets":[1024, ...]},      │
│   ...                                                        │
│   "__metadata__": {                                          │
│     "rvc.config":  "[1025, 32, 192, ...]",  ← JSON string     │
│     "rvc.info":    "\"trained on dataset X\"",               │
│     "rvc.sr":      "\"48k\"",                                │
│     "rvc.f0":      "1",                                      │
│     "rvc.version": "\"v2\"",                                 │
│     "author":      "alice",   ← optional user metadata       │
│     "license":     "CC-BY-4.0"                               │
│   }                                                          │
│ }                                                            │
├────────────────────────────────────────────────────────────┤
│ Raw tensor data (concatenated, in the order declared above)  │
└────────────────────────────────────────────────────────────┘
```

The `rvc.` prefix is reserved by this library. Any other metadata key
is treated as user-supplied and passed through unchanged.

## Quick reference

### CLI

```bash
# Convert a single .pth -> .safetensors (default output next to source)
python tools/convert_safetensors.py singer_v2.pth

# Convert + cast to FP16 + add metadata
python tools/convert_safetensors.py singer_v2.pth out/singer_v2.safetensors \
    --dtype fp16 --author alice --license CC-BY-4.0 --note "100 epoch"

# Batch convert a folder, write into a separate output dir, recurse
python tools/convert_safetensors.py --batch --recursive \
    --output-dir converted --skip-existing \
    assets/weights

# Reverse direction
python tools/convert_safetensors.py --to pth singer_v2.safetensors
```

### Python API

```python
import torch
from infer.lib.infer_pack.safetensors_utils import (
    convert_pth_to_safetensors,
    convert_safetensors_to_pth,
    ckpt_to_safetensors,
    safetensors_to_ckpt,
    load_rvc_checkpoint,
    list_checkpoint_files,
)

# 1) Convert one file
convert_pth_to_safetensors(
    "assets/weights/singer_v2.pth",
    dtype=torch.float16,                  # optional cast
    extra_metadata={"author": "alice"},   # optional
    overwrite=True,                        # optional
)

# 2) Read it back into a normalized ckpt dict
ckpt = safetensors_to_ckpt("assets/weights/singer_v2.safetensors")
print(ckpt["version"], ckpt["sr"], ckpt["f0"], len(ckpt["weight"]))

# 3) Auto-detect .pth vs .safetensors and load uniformly
ckpt = load_rvc_checkpoint("assets/weights/singer_v2.safetensors")
# (also works for .pth — same return shape)

# 4) Write an in-memory ckpt dict to .safetensors directly
ckpt_to_safetensors(
    {"weight": ..., "config": [...], "sr": "48k", "f0": 1, "version": "v2"},
    "out/singer_v2.safetensors",
    metadata={"note": "extracted via process_ckpt"},
)

# 5) Reverse direction: .safetensors -> .pth
convert_safetensors_to_pth(
    "assets/weights/singer_v2.safetensors",
    "out/singer_v2.pth",
)

# 6) Walk a folder and bucket checkpoints by format
files = list_checkpoint_files("assets/weights")
# -> {"pth": [...], "safetensors": [...]}
```

### Using `process_ckpt` (training / WebUI side)

The functions under `infer/lib/train/process_ckpt.py` now all understand
the `safetensors` format. The convention is: **if the destination file
name ends in `.safetensors`, the file is written as safetensors; otherwise
as `.pth`.**

```python
from infer.lib.train import process_ckpt

# Save a freshly trained model as .safetensors — just give it a
# `.safetensors` name.
process_ckpt.savee(
    ckpt["weight"], sr="48k", if_f0=1, name="singer_v2.safetensors",
    epoch=100, version="v2", hps=hps,
)
# writes assets/weights/singer_v2.safetensors

# Show info works on both formats automatically
print(process_ckpt.show_info("assets/weights/singer_v2.safetensors"))
# -> '模型信息:...\n采样率:48k\n模型是否输入音高引导:1\n版本:v2'

# Merge two checkpoints — output format inferred from the target name
process_ckpt.merge(
    path1="singer_a.safetensors",
    path2="singer_b.safetensors",   # input formats can be mixed
    alpha1=0.5, sr="48k", f0="是", info="blend",
    name="singer_blend.safetensors",
    version="v2",
)
```

## Running inference on a `.safetensors` model

The new `RVCInferer` class auto-detects the format on construction:

```python
from infer.lib.infer_pack.inference import RVCInferer

inferer = RVCInferer(
    "assets/weights/singer_v2.safetensors",   # or .pth, same code
    device="cuda:0",
    is_half=True,
    index_path="singer_v2.index",
)
sr, audio = inferer.convert(
    "in.wav", "out.wav",
    f0up_key=12,
    f0_method="rmvpe",
)
```

Or from the CLI:

```bash
python -m infer.lib.infer_pack.inference \
    --model assets/weights/singer_v2.safetensors \
    --device cuda:0 \
    single --input in.wav --output out.wav --f0up_key 12
```

## Format compatibility matrix

| Source format | Reader API                       | Writer API                       | WebUI | CLI |
|---------------|----------------------------------|----------------------------------|-------|-----|
| `.pth`        | `torch.load`                     | `torch.save`                     | yes   | yes |
| `.safetensors`| `safetensors_to_ckpt`            | `ckpt_to_safetensors`            | yes   | yes |
| Mixed         | `load_rvc_checkpoint` (auto)     | `*_save_checkpoint(fmt=...)`     | yes   | yes |

If you have existing `.pth` files, you don't need to do anything — they
will keep working in the WebUI and CLI. The new format is purely additive.
