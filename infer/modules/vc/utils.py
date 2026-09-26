import os
import gc
import sys
import torch
import codecs
import librosa
import requests
import numpy as np
import soundfile as sf
import torch.nn.functional as F
from infer.modules.vc import fairseq


def HF_download_file(url, output_path=None):
    url = url.replace("/blob/", "/resolve/").replace("?download=true", "").strip()
    output_path = os.path.basename(url) if output_path is None else (os.path.join(output_path, os.path.basename(url)) if os.path.isdir(output_path) else output_path)
    response = requests.get(url, stream=True, timeout=300)

    if response.status_code == 200:
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=10 * 1024 * 1024):
                f.write(chunk)

        return output_path
    else: raise ValueError(response.status_code)

def check_predictors(method):
    def download(predictors):
        if not os.path.exists(os.path.join("models", predictors)): 
           HF_download_file(codecs.decode("uggcf://uhttvatsnpr.pb/NauC/Ivrganzrfr-EIP-Cebwrpg/erfbyir/znva/cerqvpgbef/", "rot13") + predictors, os.path.join("models", predictors))

    model_dict = {
        **dict.fromkeys(["rmvpe", "rmvpe-legacy"], "rmvpe.pt"), 
        **dict.fromkeys(["fcpe"], "fcpe.pt"), 
        **dict.fromkeys(["fcpe-legacy"], "fcpe_legacy.pt"), 
        **dict.fromkeys(["crepe-full", "mangio-crepe-full"], "crepe_full.pth"), 
        **dict.fromkeys(["crepe-large", "mangio-crepe-large"], "crepe_large.pth"), 
        **dict.fromkeys(["crepe-medium", "mangio-crepe-medium"], "crepe_medium.pth"), 
        **dict.fromkeys(["crepe-small", "mangio-crepe-small"], "crepe_small.pth"), 
        **dict.fromkeys(["crepe-tiny", "mangio-crepe-tiny"], "crepe_tiny.pth"), 
    }

    if method in model_dict: download(model_dict[method])



def get_index_path_from_model(sid):
    return next(
        (
            f
            for f in [
                os.path.join(root, name)
                for root, _, files in os.walk(os.getenv("index_root"), topdown=False)
                for name in files
                if name.endswith(".index") and "trained" not in name
            ]
            if sid.split(".")[0] in f
        ),
        "",
    )


def load_hubert(config):
    models = fairseq.load_model("assets/hubert/hubert_base.pt").to(self.device).eval()
                
    if config.is_half:
        hubert_model = hubert_model.half()
    else:
        hubert_model = hubert_model.float()
    
    return models.eval()
