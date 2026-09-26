import os
import sys
import traceback
from collections import OrderedDict

import torch

from i18n.i18n import I18nAuto
from infer.lib.infer_pack.safetensors_utils import (
    ckpt_to_safetensors,
    safetensors_to_ckpt,
)

i18n = I18nAuto()


def _save_checkpoint(opt, name, fmt="pth"):
    """Persist a checkpoint dict in either ``.pth`` or ``.safetensors``.

    Parameters
    ----------
    opt:
        Normalized checkpoint dict (must contain ``weight``, ``config``,
        ``sr``, ``f0``, ``version``).
    name:
        Output filename *without* extension.
    fmt:
        ``"pth"`` or ``"safetensors"``.
    """
    out_dir = "assets/weights"
    os.makedirs(out_dir, exist_ok=True)
    if fmt == "safetensors":
        return ckpt_to_safetensors(opt, os.path.join(out_dir, name + ".safetensors"))
    if fmt == "pth":
        return torch.save(opt, os.path.join(out_dir, name + ".pth"))
    raise ValueError(f"Unknown checkpoint format: {fmt!r}")


def savee(ckpt, sr, if_f0, name, epoch, version, hps, fmt="pth"):
    try:
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        opt["config"] = [
            hps.data.filter_length // 2 + 1,
            32,
            hps.model.inter_channels,
            hps.model.hidden_channels,
            hps.model.filter_channels,
            hps.model.n_heads,
            hps.model.n_layers,
            hps.model.kernel_size,
            hps.model.p_dropout,
            hps.model.resblock,
            hps.model.resblock_kernel_sizes,
            hps.model.resblock_dilation_sizes,
            hps.model.upsample_rates,
            hps.model.upsample_initial_channel,
            hps.model.upsample_kernel_sizes,
            hps.model.spk_embed_dim,
            hps.model.gin_channels,
            hps.data.sampling_rate,
        ]
        opt["info"] = "%sepoch" % epoch
        opt["sr"] = sr
        opt["f0"] = if_f0
        opt["version"] = version
        _save_checkpoint(opt, name, fmt=fmt)
        return "Success."
    except:
        return traceback.format_exc()


def show_info(path):
    """Print a one-line summary of any RVC checkpoint.

    Accepts both ``.pth`` and ``.safetensors`` files.
    """
    try:
        if str(path).lower().endswith((".safetensors", ".st")):
            a = safetensors_to_ckpt(path)
        else:
            a = torch.load(path, map_location="cpu")
        return "模型信息:%s\n采样率:%s\n模型是否输入音高引导:%s\n版本:%s" % (
            a.get("info", "None"),
            a.get("sr", "None"),
            a.get("f0", "None"),
            a.get("version", "None"),
        )
    except:
        return traceback.format_exc()


def extract_small_model(path, name, sr, if_f0, info, version):
    try:
        ckpt = torch.load(path, map_location="cpu")
        if "model" in ckpt:
            ckpt = ckpt["model"]
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt.keys():
            if "enc_q" in key:
                continue
            opt["weight"][key] = ckpt[key].half()
        if sr == "40k":
            opt["config"] = [
                1025,
                32,
                192,
                192,
                768,
                2,
                6,
                3,
                0,
                "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                109,
                256,
                40000,
            ]
        elif sr == "48k":
            if version == "v1":
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 6, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    48000,
                ]
            else:
                opt["config"] = [
                    1025,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [12, 10, 2, 2],
                    512,
                    [24, 20, 4, 4],
                    109,
                    256,
                    48000,
                ]
        elif sr == "32k":
            if version == "v1":
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 4, 2, 2, 2],
                    512,
                    [16, 16, 4, 4, 4],
                    109,
                    256,
                    32000,
                ]
            else:
                opt["config"] = [
                    513,
                    32,
                    192,
                    192,
                    768,
                    2,
                    6,
                    3,
                    0,
                    "1",
                    [3, 7, 11],
                    [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                    [10, 8, 2, 2],
                    512,
                    [20, 16, 4, 4],
                    109,
                    256,
                    32000,
                ]
        if info == "":
            info = "Extracted model."
        opt["info"] = info
        opt["version"] = version
        opt["sr"] = sr
        opt["f0"] = int(if_f0)
        fmt = "safetensors" if str(name).endswith((".safetensors", ".st")) else "pth"
        _save_checkpoint(opt, name if not name.endswith((".safetensors", ".st")) else name.rsplit(".", 1)[0], fmt=fmt)
        return "Success."
    except:
        return traceback.format_exc()


def change_info(path, info, name):
    try:
        # Load the source checkpoint with format auto-detection so that
        # users can relabel either .pth or .safetensors files.
        if str(path).lower().endswith((".safetensors", ".st")):
            ckpt = safetensors_to_ckpt(path)
        else:
            ckpt = torch.load(path, map_location="cpu")
        ckpt["info"] = info
        if name == "":
            name = os.path.basename(path)
        # If the target name explicitly ends in .safetensors we honor it;
        # otherwise the original file extension is preserved.
        if str(name).lower().endswith((".safetensors", ".st")):
            stem = name.rsplit(".", 1)[0]
            _save_checkpoint(ckpt, stem, fmt="safetensors")
        elif str(name).lower().endswith((".pth", ".pt")):
            stem = name.rsplit(".", 1)[0]
            _save_checkpoint(ckpt, stem, fmt="pth")
        else:
            # Preserve original format by writing to the matching extension.
            fmt = "safetensors" if str(path).lower().endswith((".safetensors", ".st")) else "pth"
            _save_checkpoint(ckpt, name, fmt=fmt)
        return "Success."
    except:
        return traceback.format_exc()


def merge(path1, path2, alpha1, sr, f0, info, name, version):
    try:
        from infer.lib.infer_pack.safetensors_utils import load_rvc_checkpoint

        def extract(ckpt):
            a = ckpt["model"]
            opt = OrderedDict()
            opt["weight"] = {}
            for key in a.keys():
                if "enc_q" in key:
                    continue
                opt["weight"][key] = a[key]
            return opt

        # Use the format-agnostic loader so users can merge .pth with .pth,
        # .safetensors with .safetensors, or mix the two freely.
        ckpt1 = load_rvc_checkpoint(path1)
        ckpt2 = load_rvc_checkpoint(path2)
        cfg = ckpt1["config"]
        if "model" in ckpt1:
            ckpt1 = extract(ckpt1)
        else:
            ckpt1 = ckpt1["weight"]
        if "model" in ckpt2:
            ckpt2 = extract(ckpt2)
        else:
            ckpt2 = ckpt2["weight"]
        if sorted(list(ckpt1.keys())) != sorted(list(ckpt2.keys())):
            return "Fail to merge the models. The model architectures are not the same."
        opt = OrderedDict()
        opt["weight"] = {}
        for key in ckpt1.keys():
            # try:
            if key == "emb_g.weight" and ckpt1[key].shape != ckpt2[key].shape:
                min_shape0 = min(ckpt1[key].shape[0], ckpt2[key].shape[0])
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key][:min_shape0].float())
                    + (1 - alpha1) * (ckpt2[key][:min_shape0].float())
                ).half()
            else:
                opt["weight"][key] = (
                    alpha1 * (ckpt1[key].float()) + (1 - alpha1) * (ckpt2[key].float())
                ).half()
        # except:
        #     pdb.set_trace()
        opt["config"] = cfg
        """
        if(sr=="40k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 10, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 40000]
        elif(sr=="48k"):opt["config"] = [1025, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10,6,2,2,2], 512, [16, 16, 4, 4], 109, 256, 48000]
        elif(sr=="32k"):opt["config"] = [513, 32, 192, 192, 768, 2, 6, 3, 0, "1", [3, 7, 11], [[1, 3, 5], [1, 3, 5], [1, 3, 5]], [10, 4, 2, 2, 2], 512, [16, 16, 4, 4,4], 109, 256, 32000]
        """
        opt["sr"] = sr
        opt["f0"] = 1 if f0 == i18n("是") else 0
        opt["version"] = version
        opt["info"] = info
        fmt = "safetensors" if str(name).endswith((".safetensors", ".st")) else "pth"
        _save_checkpoint(opt, name if not name.endswith((".safetensors", ".st")) else name.rsplit(".", 1)[0], fmt=fmt)
        return "Success."
    except:
        return traceback.format_exc()
