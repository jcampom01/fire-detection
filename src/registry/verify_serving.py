"""End-to-end check of the serving stack: send test images to the API and compare with the training predictions.

The reference is predictions/predictions_test.csv, logged in MLflow by the run of the version being served, so the
check works for any version. Images are the original files (as a camera would send them): the API does the
CLAHE/resize/JPEG preparation, TF Serving the model.

Pass criteria (CPU vs GPU and float16 make tiny differences unavoidable, so the max alone is not a criterion):
    mean |prob_api - prob_train|   <= --max-mean
    p95  |prob_api - prob_train|   <= --max-p95
    decision flips                 only for images whose training prob is within --near of the threshold
    every response                 from the same model version (a deploy during the check makes it meaningless)
Exits 1 if any criterion fails, so it can gate a deploy.

Usage (stacks fire-mlops and fire-serving up):
    python src/registry/verify_serving.py                  # whole test split (~25 min with B4 on CPU)
    python src/registry/verify_serving.py --limit 200      # evenly spaced sample, quick check after a deploy
    python src/registry/verify_serving.py --out /tmp/check.csv
"""

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")  # presigned URLs do not resolve from the host

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REGISTERED_MODEL = "fire-detector"
LABELS = ROOT / "data/classification_labels/test.csv"
REFERENCE = "predictions/predictions_test.csv"
DEFAULT_TRACKING_URI = "http://localhost:5000"
DEFAULT_SERVING_API_URL = "http://127.0.0.1:8080"


# --- Inputs ---

def served_model(api):
    r = httpx.get(f"{api}/model", timeout=10)
    r.raise_for_status()
    info = r.json()
    return int(info["version"]), info["meta"]["threshold"]


def reference(client, version):
    """Training predictions of the run behind this registered version, joined with the image paths."""
    run_id = client.get_model_version(REGISTERED_MODEL, str(version)).run_id
    with tempfile.TemporaryDirectory() as tmp:
        ref = pd.read_csv(client.download_artifacts(run_id, REFERENCE, tmp))
    labels = pd.read_csv(LABELS)[["image", "image_path", "label"]]
    df = ref.merge(labels, on="image", validate="one_to_one")
    if len(df) != len(ref):
        sys.exit(f"{len(ref) - len(df)} reference images are missing from {LABELS}")
    return run_id, df


def sample(df, limit):
    if not limit or limit >= len(df):
        return df
    idx = np.linspace(0, len(df) - 1, limit).round().astype(int)
    return df.iloc[idx]


# --- Requests ---

def ask(http, api, path):
    with open(ROOT / path, "rb") as f:
        r = http.post(f"{api}/predict", files={"file": (Path(path).name, f, "image/jpeg")})
    r.raise_for_status()
    body = r.json()
    lat = body["latency_ms"]
    return body["prob"], body["alarm"], body["model_version"], lat["preprocess"], lat["model"]


def predict_all(api, paths, workers):
    done, results = 0, []
    with httpx.Client(timeout=60) as http, ThreadPoolExecutor(workers) as pool:
        for res in pool.map(lambda p: ask(http, api, p), paths):
            results.append(res)
            done += 1
            if done % 100 == 0 or done == len(paths):
                print(f"   {done}/{len(paths)}", end="\r", flush=True)
    print()
    return results


# --- Check ---

def check(df, threshold, version, args):
    diff = df["diff"]
    flips = df[df["alarm_api"] != (df["prob"] >= threshold)]  # the CSV "alarm" column is the label, not the decision
    far_flips = flips[(flips["prob"] - threshold).abs() >= args.near]
    versions = set(df["model_version"])
    checks = [
        ("mean |diff|", f"{diff.mean():.5f}", f"<= {args.max_mean}", diff.mean() <= args.max_mean),
        ("p95 |diff|", f"{diff.quantile(0.95):.5f}", f"<= {args.max_p95}", diff.quantile(0.95) <= args.max_p95),
        ("decision flips", f"{len(flips)} ({len(far_flips)} far from threshold)", f"only within {args.near}",
         far_flips.empty),
        ("model version", ", ".join(f"v{v}" for v in sorted(versions)), f"only v{version}", versions == {version}),
    ]
    print(f"\n{'check':<16} {'value':<28} {'rule':<20} result")
    for name, value, rule, ok in checks:
        print(f"{name:<16} {value:<28} {rule:<20} {'ok' if ok else 'FAIL'}")
    print(f"(max |diff| {diff.max():.5f}, informative only)")
    for _, row in flips.iterrows():
        print(f"   flip: {row['image']}  train {row['prob']:.4f}  api {row['prob_api']:.4f}  threshold {threshold:.4f}")
    return all(ok for *_, ok in checks)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--limit", type=int, help="evenly spaced sample of N test images (default: all)")
    p.add_argument("--workers", type=int, default=4, help="concurrent requests")
    p.add_argument("--max-mean", type=float, default=0.001)
    p.add_argument("--max-p95", type=float, default=0.005)
    p.add_argument("--near", type=float, default=0.01, help="flips allowed only this close to the threshold")
    p.add_argument("--out", type=Path, help="write per-image results to this CSV")
    args = p.parse_args()

    api = os.environ.get("SERVING_API_URL", DEFAULT_SERVING_API_URL)
    client = MlflowClient(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))

    version, threshold = served_model(api)
    run_id, df = reference(client, version)
    df = sample(df, args.limit)
    print(f"served v{version} (run {run_id[:8]}, threshold {threshold:.4f}): {len(df)} images, {args.workers} workers")

    start = time.time()
    results = predict_all(api, df["image_path"].tolist(), args.workers)
    elapsed = time.time() - start
    columns = ["prob_api", "alarm_api", "model_version", "preprocess_ms", "model_ms"]
    df = df.assign(**dict(zip(columns, zip(*results))))
    df["diff"] = (df["prob_api"] - df["prob"]).abs()
    print(f"{len(df) / elapsed:.1f} img/s with {args.workers} workers · latency inside the API (p50 / p95):")
    for name in ["preprocess_ms", "model_ms"]:
        print(f"   {name:<14} {df[name].median():5.0f} / {df[name].quantile(0.95):5.0f}")

    ok = check(df, threshold, version, args)
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"per-image results: {args.out}")
    print("\nPASSED" if ok else "\nFAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
