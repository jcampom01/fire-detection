"""Build copies of the datasets with the images already resized to the training input size.

Decoding the full-resolution images (mostly 1280x720, some 1920x1080) on every epoch makes the
CPU the training bottleneck and uses several GB of RAM in the tf.data pipeline. Resizing them
once to disk removes both problems.

Images are resized to size x size without keeping the aspect ratio, exactly as
tf.image.resize does in src/train/data.py, so training on these copies is equivalent to training
on the full-resolution images (the resize in data.py becomes a no-op). INTER_AREA is used, which
antialiases when downscaling.

Each (dataset, size) pair gets its own folder with the same layout as the original:

    data/clahe_dataset_224/
        train/images/*.jpg   (resized)
        train/labels/*.txt   (copied unchanged: YOLO coordinates are normalised)
        val/...
        test/...

All parameters are read from a JSON config file (default: src/preprocess/resize_config.json):

    data_roots     datasets to resize, relative to the project root (or absolute)
    sizes          output sizes; each dataset is written to <data_root>_<size>
    splits         splits to process, e.g. ["train", "val", "test"]
    jpeg_quality   quality of the saved JPEG images (0-100)
    workers        number of processes (null = all CPUs)
    overwrite      reprocess images that already exist

Usage:
    python src/preprocess/resize_dataset.py
    python src/preprocess/resize_dataset.py --config path/to/other_config.json
"""

import argparse
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import cv2
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(__file__).resolve().parent / "resize_config.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
REQUIRED_KEYS = ["data_roots", "sizes", "splits", "jpeg_quality", "workers", "overwrite"]


def load_config(path=CONFIG_PATH):
    """Read the JSON config and resolve its paths against the project root."""
    path = Path(path)
    cfg = json.loads(path.read_text())

    missing = [k for k in REQUIRED_KEYS if k not in cfg]
    if missing:
        raise KeyError(f"{path}: missing keys {missing}")

    cfg["data_roots"] = [PROJECT_ROOT / p for p in cfg["data_roots"]]  # an absolute path is kept as-is
    return cfg


def process_image(src, dst, size, jpeg_quality, overwrite):
    """Return 'done', 'skipped' or an error message for one image."""
    if dst.exists() and not overwrite:
        return "skipped"
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        return f"could not read {src}"
    out = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality] if dst.suffix.lower() in {".jpg", ".jpeg"} else []
    if not cv2.imwrite(str(dst), out, params):
        return f"could not write {dst}"
    return "done"


def _process_pair(pair, **kwargs):
    return process_image(*pair, **kwargs)


def process_split(data_root, output_root, split, size, cfg):
    src_img_dir = data_root / split / "images"
    src_lbl_dir = data_root / split / "labels"
    dst_img_dir = output_root / split / "images"
    dst_lbl_dir = output_root / split / "labels"
    dst_img_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in src_img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)
    pairs = [(p, dst_img_dir / p.name) for p in images]

    worker = partial(_process_pair, size=size, jpeg_quality=cfg["jpeg_quality"], overwrite=cfg["overwrite"])
    counts = {"done": 0, "skipped": 0}
    errors = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        for result in tqdm(pool.map(worker, pairs, chunksize=32), total=len(pairs),
                           desc=f"{output_root.name}/{split}", unit="img"):
            if result in counts:
                counts[result] += 1
            else:
                errors.append(result)

    # YOLO labels use normalised coordinates, so they stay valid after resizing
    if src_lbl_dir.is_dir():
        shutil.copytree(src_lbl_dir, dst_lbl_dir, dirs_exist_ok=True)

    return len(images), counts, errors


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=CONFIG_PATH, help="JSON config file (default: %(default)s)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    print(f"config {args.config}")
    print(f"sizes {cfg['sizes']}  jpeg_quality={cfg['jpeg_quality']}  overwrite={cfg['overwrite']}")

    for data_root in cfg["data_roots"]:
        for size in cfg["sizes"]:
            output_root = data_root.with_name(f"{data_root.name}_{size}")
            print(f"\n{data_root}  ->  {output_root}")
            for split in cfg["splits"]:
                if not (data_root / split / "images").is_dir():
                    print(f"{split:<5} no images folder — skipped")
                    continue
                total, counts, errors = process_split(data_root, output_root, split, size, cfg)
                print(f"{split:<5} {total:>6,} images  |  processed={counts['done']:,}  "
                      f"skipped={counts['skipped']:,}  errors={len(errors)}")
                for err in errors[:10]:
                    print(f"   ! {err}")


if __name__ == "__main__":
    main()
