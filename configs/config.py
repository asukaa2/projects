import argparse
import os
import sys
import json
from multiprocessing import cpu_count

import torch
try:
    import intel_extension_for_pytorch as ipex # pylint: disable=import-error, unused-import
    if torch.xpu.is_available():
        from infer.modules.ipex import ipex_init
        ipex_init()
except Exception:
    pass
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unified config location
# ---------------------------------------------------------------------------
# All RVC configuration lives in a single file:  configs/config.json
# It has two top-level keys:
#   - "models":   per-architecture hyper-parameters, keyed by
#                 "<version>/<sample_rate>" (e.g. "v1/32k", "v2/48k").
#   - "realtime": default values for the realtime voice-changer tab.
#
# The previous layout had six files:
#     configs/v1/32k.json   configs/v1/40k.json   configs/v1/48k.json
#     configs/v2/32k.json   configs/v2/48k.json   configs/config.json
# Those are now consolidated into the single file above. As a transitional
# safety net, :func:`load_config_json` falls back to the old per-file
# layout if the unified file is missing — so users who pin an older
# checkout still load successfully.

UNIFIED_CONFIG_PATH = "configs/config.json"

# Architectures we know about. Order is preserved for deterministic iteration
# and matches the historical ``version_config_list``.
version_config_list = [
    "v1/32k",
    "v1/40k",
    "v1/48k",
    "v2/32k",
    "v2/48k",
]


def singleton_variable(func):
    def wrapper(*args, **kwargs):
        if not wrapper.instance:
            wrapper.instance = func(*args, **kwargs)
        return wrapper.instance

    wrapper.instance = None
    return wrapper


def _load_unified_config() -> dict:
    """Load the unified configs/config.json and return its raw contents."""
    with open(UNIFIED_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_legacy_per_file() -> dict:
    """Back-compat fallback: read the old configs/v1/*.json + v2/*.json files.

    Returns a dict keyed by "<version>/<sr>" (without the trailing ``.json``)
    so that downstream code that does ``config.json_config["v1/32k"]`` still
    works without modification.
    """
    d = {}
    legacy_paths = [
        ("v1/32k", "configs/v1/32k.json"),
        ("v1/40k", "configs/v1/40k.json"),
        ("v1/48k", "configs/v1/48k.json"),
        ("v2/32k", "configs/v2/32k.json"),
        ("v2/48k", "configs/v2/48k.json"),
    ]
    for key, path in legacy_paths:
        with open(path, "r", encoding="utf-8") as f:
            d[key] = json.load(f)
    return d


def load_realtime_config() -> dict:
    """Return the realtime voice-changer defaults from the unified file.

    Returns an empty dict if the unified file or its ``realtime`` section
    is missing. The realtime tab in the WebUI passes these values via
    function arguments, so an empty dict is safe — the caller's own
    defaults take over.
    """
    if not os.path.isfile(UNIFIED_CONFIG_PATH):
        return {}
    try:
        raw = _load_unified_config()
    except (json.JSONDecodeError, OSError):
        return {}
    return raw.get("realtime", {})


@singleton_variable
class Config:
    def __init__(self):
        self.device = "cuda:0"
        self.is_half = True
        self.n_cpu = 0
        self.gpu_name = None
        self.json_config = self.load_config_json()
        self.gpu_mem = None
        (
            self.python_cmd,
            self.listen_port,
            self.iscolab,
            self.noparallel,
            self.noautoopen,
            self.dml,
        ) = self.arg_parse()
        self.instead = ""
        self.x_pad, self.x_query, self.x_center, self.x_max = self.device_config()

    @staticmethod
    def load_config_json() -> dict:
        """Load the architecture configs and return them as a flat dict.

        Returns
        -------
        dict
            ``{"v1/32k": {...}, "v1/40k": {...}, "v1/48k": {...},
               "v2/32k": {...}, "v2/48k": {...}}`` — the same shape that the
            old per-file loader produced, so call sites in
            ``infer-web.py`` and ``core.py`` keep working unchanged.
        """
        # Try the unified file first.
        if os.path.isfile(UNIFIED_CONFIG_PATH):
            try:
                raw = _load_unified_config()
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "Failed to read %s: %s. Falling back to per-file layout.",
                    UNIFIED_CONFIG_PATH, exc,
                )
                return _load_legacy_per_file()
            models = raw.get("models")
            if not isinstance(models, dict) or not models:
                logger.warning(
                    "%s has no 'models' section. Falling back to per-file layout.",
                    UNIFIED_CONFIG_PATH,
                )
                return _load_legacy_per_file()
            return models

        # No unified file — fall back to the legacy per-file layout.
        logger.info(
            "%s not found; loading legacy per-file configs (v1/*.json, v2/*.json).",
            UNIFIED_CONFIG_PATH,
        )
        return _load_legacy_per_file()

    @staticmethod
    def arg_parse() -> tuple:
        exe = sys.executable or "python"
        parser = argparse.ArgumentParser()
        parser.add_argument("--port", type=int, default=7865, help="Listen port")
        parser.add_argument("--pycmd", type=str, default=exe, help="Python command")
        parser.add_argument("--colab", action="store_true", help="Launch in colab")
        parser.add_argument(
            "--noparallel", action="store_true", help="Disable parallel processing"
        )
        parser.add_argument(
            "--noautoopen",
            action="store_true",
            help="Do not open in browser automatically",
        )
        parser.add_argument(
            "--dml",
            action="store_true",
            help="torch_dml",
        )
        cmd_opts = parser.parse_args()

        cmd_opts.port = cmd_opts.port if 0 <= cmd_opts.port <= 65535 else 7865

        return (
            cmd_opts.pycmd,
            cmd_opts.port,
            cmd_opts.colab,
            cmd_opts.noparallel,
            cmd_opts.noautoopen,
            cmd_opts.dml,
        )

    # has_mps is only available in nightly pytorch (for now) and MasOS 12.3+.
    # check `getattr` and try it for compatibility
    @staticmethod
    def has_mps() -> bool:
        if not torch.backends.mps.is_available():
            return False
        try:
            torch.zeros(1).to(torch.device("mps"))
            return True
        except Exception:
            return False

    @staticmethod
    def has_xpu() -> bool:
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True
        else:
            return False

    def use_fp32_config(self):
        for config_file in version_config_list:
            self.json_config[config_file]["train"]["fp16_run"] = False

    def device_config(self) -> tuple:
        if torch.cuda.is_available():
            if self.has_xpu():
                self.device = self.instead = "xpu:0"
                self.is_half = True
            i_device = int(self.device.split(":")[-1])
            self.gpu_name = torch.cuda.get_device_name(i_device)
            if (
                ("16" in self.gpu_name and "V100" not in self.gpu_name.upper())
                or "P40" in self.gpu_name.upper()
                or "P10" in self.gpu_name.upper()
                or "1060" in self.gpu_name
                or "1070" in self.gpu_name
                or "1080" in self.gpu_name
            ):
                logger.info("Found GPU %s, force to fp32", self.gpu_name)
                self.is_half = False
                self.use_fp32_config()
            else:
                logger.info("Found GPU %s", self.gpu_name)
            self.gpu_mem = int(
                torch.cuda.get_device_properties(i_device).total_memory
                / 1024
                / 1024
                / 1024
                + 0.4
            )
            if self.gpu_mem <= 4:
                with open("infer/modules/train/preprocess.py", "r") as f:
                    strr = f.read().replace("3.7", "3.0")
                with open("infer/modules/train/preprocess.py", "w") as f:
                    f.write(strr)
        elif self.has_mps():
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "mps"
            self.is_half = False
            self.use_fp32_config()
        else:
            logger.info("No supported Nvidia GPU found")
            self.device = self.instead = "cpu"
            self.is_half = False
            self.use_fp32_config()

        if self.n_cpu == 0:
            self.n_cpu = cpu_count()

        if self.is_half:
            # 6G显存配置
            x_pad = 3
            x_query = 10
            x_center = 60
            x_max = 65
        else:
            # 5G显存配置
            x_pad = 1
            x_query = 6
            x_center = 38
            x_max = 41

        if self.gpu_mem is not None and self.gpu_mem <= 4:
            x_pad = 1
            x_query = 5
            x_center = 30
            x_max = 32
        if self.dml:
            logger.info("Use DirectML instead")
            if (
                os.path.exists(
                    r"runtime\Lib\site-packages\onnxruntime\capi\DirectML.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime",
                        r"runtime\Lib\site-packages\onnxruntime-cuda",
                    )
                except:
                    pass
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime-dml",
                        r"runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
            # if self.device != "cpu":
            import torch_directml

            self.device = torch_directml.device(torch_directml.default_device())
            self.is_half = False
        else:
            if self.instead:
                logger.info(f"Use {self.instead} instead")
            if (
                os.path.exists(
                    r"runtime\Lib\site-packages\onnxruntime\capi\onnxruntime_providers_cuda.dll"
                )
                == False
            ):
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime",
                        r"runtime\Lib\site-packages\onnxruntime-dml",
                    )
                except:
                    pass
                try:
                    os.rename(
                        r"runtime\Lib\site-packages\onnxruntime-cuda",
                        r"runtime\Lib\site-packages\onnxruntime",
                    )
                except:
                    pass
        return x_pad, x_query, x_center, x_max
