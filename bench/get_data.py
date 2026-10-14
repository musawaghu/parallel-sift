"""Download Oxford 102 Flowers and extract the 50 images used in the 2023 study.

The 2023 notebook read `dataset/1.jpg … dataset/50.jpg`. These are
`image_00001.jpg … image_00050.jpg` from the Oxford 102 Category Flower Dataset
(Nilsback & Zisserman, 2008). This was verified by reproducing the 2023 keypoint
counts exactly (see README). Outputs:

    data/images/{i}.jpg   original JPEGs
    data/gray/{i}.npy     decoded grayscale arrays (the no-GIL interpreter has no OpenCV to decode JPEGs)

Usage:  python bench/get_data.py [--n 50]
"""
from __future__ import annotations

import argparse
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

URL = "https://thor.robots.ox.ac.uk/flowers/102/102flowers.tgz"
ROOT = Path(__file__).resolve().parents[1] / "data"


def download(dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 340_000_000:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL} (~345 MB) ...")
    urllib.request.urlretrieve(URL, dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50)
    args = ap.parse_args()

    archive = ROOT / "raw" / "102flowers.tgz"
    download(archive)
    wanted = {f"jpg/image_{i:05d}.jpg": i for i in range(1, args.n + 1)}
    (ROOT / "images").mkdir(parents=True, exist_ok=True)
    (ROOT / "gray").mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive, "r:gz") as tar:
        for member in tar:
            i = wanted.pop(member.name, None)
            if i is None:
                continue
            jpg, npy = ROOT / "images" / f"{i}.jpg", ROOT / "gray" / f"{i}.npy"
            if npy.exists():
                if not wanted:
                    break
                continue
            jpg.write_bytes(tar.extractfile(member).read())
            np.save(npy, cv2.imread(str(jpg), cv2.IMREAD_GRAYSCALE))
            if not wanted:
                break
    if wanted:
        raise SystemExit(f"missing from archive: {sorted(wanted)}")
    print(f"extracted {args.n} images to {ROOT / 'images'}")


if __name__ == "__main__":
    main()
