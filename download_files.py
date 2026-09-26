import os
import urllib.request
from urllib.error import URLError, HTTPError

assets_folder = "./assets/"
if not os.path.exists(assets_folder):
    os.makedirs(assets_folder)

files = {
    "rmvpe/rmvpe.pt": "https://huggingface.co/Rejekts/project/resolve/main/rmvpe.pt",
    "hubert/hubert_base.pt": "https://huggingface.co/Rejekts/project/resolve/main/hubert_base.pt",
    "pretrained_v2/D40k.pth": "https://huggingface.co/Rejekts/project/resolve/main/D40k.pth",
    "pretrained_v2/G40k.pth": "https://huggingface.co/Rejekts/project/resolve/main/G40k.pth",
    "pretrained_v2/f0D40k.pth": "https://huggingface.co/Rejekts/project/resolve/main/f0D40k.pth",
    "pretrained_v2/f0G40k.pth": "https://huggingface.co/Rejekts/project/resolve/main/f0G40k.pth",
}

for file, link in files.items():
    file_path = os.path.join(assets_folder, file)
    
    # Create subdirectories if they don't exist
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    if os.path.exists(file_path):
        print(f"Already exists: {file_path}")
        continue
    
    print(f"Downloading {link} -> {file_path}")
    try:
        urllib.request.urlretrieve(link, file_path)
    except (URLError, HTTPError) as e:
        print(f"Failed to download {link}: {e}")
