"""Dataset: read the split CSVs, build binary labels and stream the images with tf.data."""

from pathlib import Path

import pandas as pd

from tf_setup import tf

SUBTYPES = ["background", "smoke_only", "fire_only", "fire_and_smoke"]


def subtype(n_smoke, n_fire):
    if n_smoke and n_fire:
        return "fire_and_smoke"
    if n_fire:
        return "fire_only"
    if n_smoke:
        return "smoke_only"
    return "background"


def load_split(split, cfg):
    """One row per image with its path (in the chosen dataset version), binary label and subtype."""
    df = pd.read_csv(cfg["data"]["labels_dir"] / f"{split}.csv")
    df["path"] = [str(cfg["data"]["image_root"] / split / "images" / name) for name in df["image"]]
    df["alarm"] = (df["label"] > 0).astype(int)
    df["subtype"] = [subtype(s, f) for s, f in zip(df["n_smoke"], df["n_fire"])]

    max_images = cfg.get("debug_max_images")
    if max_images:
        df = df.sample(min(max_images, len(df)), random_state=cfg["seed"]).reset_index(drop=True)

    missing = [p for p in df["path"].head(20) if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"images not found, e.g. {missing[0]} (has this dataset version been built?)")
    return df


def make_dataset(df, cfg, training):
    """Stream images from disk (the dataset does not fit in RAM): decode, resize, batch."""
    size = cfg["image_size"]

    def load(path, label):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size), antialias=True)
        return tf.clip_by_value(img, 0, 255), tf.cast(label, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((df["path"].values, df["alarm"].values))
    if training:
        ds = ds.shuffle(len(df), seed=cfg["seed"], reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE).batch(cfg["batch_size"])
    return ds.prefetch(tf.data.AUTOTUNE)
