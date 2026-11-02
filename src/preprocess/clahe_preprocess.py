"""Build a copy of the dataset with CLAHE (Contrast Limited Adaptive Histogram Equalization) applied.

CLAHE is applied only to the lightness channel (L) of the CIELAB colour space, so contrast is
enhanced without shifting the hue of fire and smoke. The a/b colour channels are left untouched.

The output keeps the same layout as the original dataset, so it can be used as a drop-in
replacement:

    data/clahe_dataset/
        train/images/*.jpg   (CLAHE-processed)
        train/labels/*.txt   (copied unchanged)
        val/...
        test/...

All parameters are read from a JSON config file (default: src/preprocess/clahe_config.json):

    data_root, output_root   paths, relative to the project root (or absolute)
    splits                   splits to process, e.g. ["train", "val", "test"]
    clahe.clip_limit         contrast limit; higher values give stronger enhancement
    clahe.tile_grid          the image is split into a tile_grid x tile_grid grid
    jpeg_quality             quality of the saved JPEG images (0-100)
    workers                  number of processes (null = all CPUs)
    overwrite                reprocess images that already exist

Usage:
    python src/preprocess/clahe_preprocess.py
    python src/preprocess/clahe_preprocess.py --config path/to/other_config.json
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
from tqdm import tqdm

from clahe import apply_clahe  # shared with the inference API

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "clahe_config.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_KEYS = ["data_root", "output_root", "splits", "clahe", "jpeg_quality", "workers", "overwrite"]


def load_config(path=CONFIG_PATH):
    """Read the JSON config and resolve its paths against the project root."""
    path = Path(path)
    cfg = json.loads(path.read_text())

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    missing += [f"clahe.{k}" for k in ("clip_limit", "tile_grid") if k not in cfg.get("clahe", {})]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")

    for key in ("data_root", "output_root"):
        cfg[key] = PROJECT_ROOT / cfg[key]  # an absolute path in the config is kept as-is
    return cfg



def process_image(src, dst, clip_limit, tile_grid, jpeg_quality, overwrite):
    """Return 'done', 'skipped' or an error message for one image."""
    if dst.exists() and not overwrite:
        return "skipped"
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return f"could not read {src}"
    out = apply_clahe(img, clip_limit, tile_grid)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if dst.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(dst), out, params):
        return f"could not write {dst}"
    return "done"


def _process_pair(pair, **kwargs):
    return process_image(*pair, **kwargs)


def process_split(split, cfg):
    src_img_dir = cfg["data_root"] / split / "images"
    src_lbl_dir = cfg["data_root"] / split / "labels"
    dst_img_dir = cfg["output_root"] / split / "images"
    dst_lbl_dir = cfg["output_root"] / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    pairs = [(p, dst_img_dir / p.name) for p in images]

    worker = partial(_process_pair, clip_limit=cfg["clahe"]["clip_limit"], tile_grid=cfg["clahe"]["tile_grid"],
                     jpeg_quality=cfg["jpeg_quality"], overwrite=cfg["overwrite"])
    counts = {"done": 0, "skipped": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        for result in tqdm(pool.map(worker, pairs, chunksize=32), total=len(pairs), desc=split, unit="img"):
            if result in counts:
                counts[result] += 1
            else:
                errors.append(result)

    # Labels are not affected by CLAHE (geometry is unchanged), so they are copied as-is
    if src_lbl_dir.is_dir():
        shutil.copytree(src_lbl_dir, dst_lbl_dir, dirs_exist_ok=True)

    return len(images), counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="JSON config file (default: %(default)s)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    clahe = cfg["clahe"]
    print(f"config {args.config}")
    print(f"CLAHE  clip_limit={clahe['clip_limit']}  tile_grid={clahe['tile_grid']}x{clahe['tile_grid']}  "
          f"jpeg_quality={cfg['jpeg_quality']}  overwrite={cfg['overwrite']}")
    print(f"{cfg['data_root']}  ->  {cfg['output_root']}")

    for split in cfg["splits"]:
        if not (cfg["data_root"] / split / "images").is_dir():
            print(f"{split:<5} no images folder — skipped")
            continue
        total, counts, errors = process_split(split, cfg)
        print(f"{split:<5} {total:>6,} images  |  processed={counts['done']:,}  "
              f"skipped={counts['skipped']:,}  errors={len(errors)}")
        for err in errors[:10]:
            print(f"   ! {err}")


if __name__ == "__main__":
    main()
