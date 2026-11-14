"""Download original ImageNet-1k archives using authorized download URLs.

Obtain access at https://image-net.org/challenges/LSVRC/2012/2012-downloads.php
and supply the download links through the IMAGENET_*_URL environment variables.
Archives remain compressed and unmodified in the project's data directory.
"""

import argparse
import os
from pathlib import Path

from torchvision.datasets.utils import check_integrity, download_url

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ARCHIVES = {
    "train": ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    "val": ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    "devkit": ("ILSVRC2012_devkit_t12.tar.gz", "fa75699e90414af021442c21a62c3abf"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    args = parser.parse_args()
    splits = ["train", "val"] if args.split == "all" else [args.split]
    pending = []
    for split in ["devkit", *splits]:
        filename, checksum = ARCHIVES[split]
        if check_integrity(DATA_DIR / filename, checksum):
            print(f"Already verified: {filename}")
            continue
        variable = f"IMAGENET_{split.upper()}_URL"
        url = os.environ.get(variable)
        if not url:
            parser.error(
                f"Set {variable} to your authorized ImageNet download URL, "
                f"or place {filename} in {DATA_DIR}. "
                "torchvision.datasets.ImageNet cannot download ImageNet automatically."
            )
        pending.append((url, filename, checksum))

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for url, filename, checksum in pending:
        # Publish only complete, checksum-verified downloads under the final name.
        download_url(url, str(DATA_DIR), filename=filename + ".part", md5=checksum)
        (DATA_DIR / (filename + ".part")).replace(DATA_DIR / filename)
    print(f"Verified raw ImageNet archives are stored in {DATA_DIR}")


if __name__ == "__main__":
    main()
