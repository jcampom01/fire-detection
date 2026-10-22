"""Build image-level classification labels from the YOLO detection labels.

For every split (train / val / test) this writes one CSV with one row per image.

YOLO class ids in the original dataset:
    0 -> smoke
    1 -> fire

Image-level classes produced here:
    0 -> no_fire  (no label file, or an empty label file)
    1 -> smoke    (only smoke boxes)
    2 -> fire     (at least one fire box; images with fire and smoke are also class 2)

Usage:
    python src/preprocess/prepare_labels.py
"""

import argparse
import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "original_dataset"
OUTPUT_DIR = PROJECT_ROOT / "data" / "classification_labels"

SPLITS = ["train", "val", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

YOLO_SMOKE, YOLO_FIRE = 0, 1
CLASS_NAMES = {0: "no_fire", 1: "smoke", 2: "fire"}

CSV_COLUMNS = ["image", "image_path", "label", "label_name", "n_smoke", "n_fire"]


def count_boxes(label_path):
    """Return (n_smoke, n_fire) for a YOLO label file. A missing file counts as no boxes."""
    n_smoke = n_fire = 0
    if label_path is None or not label_path.is_file():
        return n_smoke, n_fire

    for line_no, line in enumerate(label_path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            cls = int(float(line.split()[0]))
        except ValueError:
            raise ValueError(f"{label_path}:{line_no}: malformed line {line!r}")
        if cls == YOLO_SMOKE:
            n_smoke += 1
        elif cls == YOLO_FIRE:
            n_fire += 1
        else:
            raise ValueError(f"{label_path}:{line_no}: unknown class id {cls}")
    return n_smoke, n_fire


def image_label(n_smoke, n_fire):
    """Fire takes priority over smoke: fire + smoke -> fire."""
    if n_fire:
        return 2
    if n_smoke:
        return 1
    return 0


def build_split(split, data_root):
    img_dir = data_root / split / "images"
    lbl_dir = data_root / split / "labels"
    images = sorted(p for p in img_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS)

    rows = []
    for img in images:
        n_smoke, n_fire = count_boxes(lbl_dir / f"{img.stem}.txt")
        label = image_label(n_smoke, n_fire)
        rows.append({
            "image": img.name,
            "image_path": img.relative_to(PROJECT_ROOT).as_posix(),
            "label": label,
            "label_name": CLASS_NAMES[label],
            "n_smoke": n_smoke,
            "n_fire": n_fire,
        })
    return rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    for split in SPLITS:
        rows = build_split(split, args.data_root)
        out_path = args.output_dir / f"{split}.csv"
        write_csv(rows, out_path)

        counts = {name: sum(r["label"] == c for r in rows) for c, name in CLASS_NAMES.items()}
        summary = "  ".join(f"{name}={n:,}" for name, n in counts.items())
        print(f"{split:<5} {len(rows):>6,} images  |  {summary}  ->  {out_path}")


if __name__ == "__main__":
    main()
