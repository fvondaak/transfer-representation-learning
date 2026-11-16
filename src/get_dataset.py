import os
from pathlib import Path

import kagglehub

ROOT = Path(__file__).resolve().parent.parent
DATAPATH = ROOT / "data"
os.makedirs(DATAPATH, exist_ok=True)

path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia",
    output_dir=str(DATAPATH),
    force_download=True,
)
