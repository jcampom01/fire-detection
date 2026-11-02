"""Deploy a registered model version to TF Serving: export it (if needed) and make it the served version.

Export: SavedModel with a JPEG-bytes signature. The training images were written as JPEG twice (after CLAHE and
after resizing, both quality 95) and decoded by TensorFlow, whose default JPEG decoder uses a fast, approximate
DCT. Other decoders (OpenCV) give slightly different pixels, and the model is sensitive to them. So the API
rebuilds those JPEGs and the model decodes them inside the graph, exactly as src/train/data.py did:
    serving_default(image_bytes: string[N]) -> {"alarm": float32[N, 1]}
The export also carries warmup requests (assets.extra/tf_serving_warmup_requests): TF Serving runs them when it
loads the version, before serving it, so the first real request is not slow.

Output: deploy/serving/models/fire_detector/<version>/ (SavedModel + meta.json), written to a temporary folder and
renamed when complete, so TF Serving never sees a half-written version. Exports are kept: going back to an
already exported version only rewrites the config.

Activation: deploy/serving/models/models.config pins the served version (TF Serving re-reads it every few seconds).
Pinning, instead of TF Serving's default "highest version", is what makes a rollback to a lower version work.

Called by `promote.py deploy` and `promote.py rollback`. Manual usage:
    python src/registry/export.py                       # the version in @production
    python src/registry/export.py --version 1
    python src/registry/export.py --version 3 --force   # re-export an existing version
    python src/registry/export.py --version 2 --no-activate
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")  # presigned URLs do not resolve from the host
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # exporting needs no GPU (and skips its slow start-up)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/train"))

from tf_setup import keras, tf  # noqa: E402  (preloads the CUDA libraries before TensorFlow)

import mlflow  # noqa: E402
from tensorflow_serving.apis import predict_pb2, prediction_log_pb2  # noqa: E402

REGISTERED_MODEL = "fire-detector"
SERVING_NAME = "fire_detector"
MODELS_DIR = ROOT / "deploy/serving/models"
CONFIG_PATH = MODELS_DIR / "models.config"
RESIZE_CONFIG = ROOT / "src/preprocess/resize_config.json"
DEFAULT_TRACKING_URI = "http://localhost:5000"
WARMUP_REQUESTS = 5

# Path inside the TF Serving container (deploy/serving/models is mounted at /models)
CONFIG_TEMPLATE = """model_config_list {{
  config {{
    name: "{name}"
    base_path: "/models/{name}"
    model_platform: "tensorflow"
    model_version_policy {{ specific {{ versions: {version} }} }}
  }}
}}
"""


def serving_meta(meta):
    """Inference metadata + how the API must build the JPEG it sends (same quality as the dataset files)."""
    meta = json.loads(json.dumps(meta))  # deep copy
    meta["preprocessing"]["jpeg_quality"] = json.loads(RESIZE_CONFIG.read_text())["jpeg_quality"]
    meta["preprocessing"]["input"] = ("JPEG bytes, decoded inside the model with TensorFlow (as in training). "
                                      "The API writes a JPEG after CLAHE (if any) and after resizing.")
    return meta


def export(model, size, out_dir):
    """SavedModel whose serving_default takes JPEG bytes and decodes them like src/train/data.py."""

    def decode(image_bytes):
        img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, (size, size), antialias=True)
        return tf.clip_by_value(img, 0, 255)

    def serve(image_bytes):
        images = tf.map_fn(decode, image_bytes, fn_output_signature=tf.TensorSpec((size, size, 3), tf.float32))
        return {"alarm": model(images, training=False)}

    archive = keras.export.ExportArchive()
    archive.track(model)
    archive.add_endpoint(name="serving_default", fn=serve,
                         input_signature=[tf.TensorSpec((None,), tf.string, name="image_bytes")])
    archive.write_out(str(out_dir))


def write_warmup(out_dir, size):
    """Predict requests TF Serving replays when loading the version (format: TFRecord of PredictionLog)."""
    image = tf.io.encode_jpeg(tf.fill((size, size, 3), tf.constant(128, tf.uint8))).numpy()  # a grey JPEG
    request = predict_pb2.PredictRequest()
    request.model_spec.name = SERVING_NAME
    request.model_spec.signature_name = "serving_default"
    request.inputs["image_bytes"].CopyFrom(tf.make_tensor_proto([image], dtype=tf.string))
    record = prediction_log_pb2.PredictionLog(predict_log=prediction_log_pb2.PredictLog(request=request))

    path = out_dir / "assets.extra" / "tf_serving_warmup_requests"
    path.parent.mkdir(parents=True, exist_ok=True)
    with tf.io.TFRecordWriter(str(path)) as writer:
        for _ in range(WARMUP_REQUESTS):
            writer.write(record.SerializeToString())


def activate(version):
    """Pin the served version in the TF Serving config (written to a temporary file, then renamed)."""
    tmp = CONFIG_PATH.with_suffix(".tmp")
    tmp.write_text(CONFIG_TEMPLATE.format(name=SERVING_NAME, version=version))
    tmp.rename(CONFIG_PATH)
    print(f"v{version} activated in {CONFIG_PATH.name}: TF Serving switches within a few seconds")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", type=int, help="registered version (default: the one in @production)")
    parser.add_argument("--force", action="store_true", help="re-export even if the version is already exported")
    parser.add_argument("--no-activate", action="store_true", help="export only, do not change the served version")
    args = parser.parse_args()

    uri = os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    try:
        urllib.request.urlopen(f"{uri}/health", timeout=5)
    except OSError as e:
        raise SystemExit(f"MLflow not reachable at {uri} ({e}).\n"
                         f"Start it with: cd deploy/mlops && docker compose up -d") from None
    mlflow.set_tracking_uri(uri)
    client = mlflow.MlflowClient()
    version = args.version or int(client.get_model_version_by_alias(REGISTERED_MODEL, "production").version)

    target = MODELS_DIR / SERVING_NAME / str(version)
    if target.exists() and not args.force:
        print(f"v{version} already exported at {target}")
    else:
        with tempfile.TemporaryDirectory() as download:
            print(f"downloading v{version} from MLflow...")
            path = mlflow.artifacts.download_artifacts(f"models:/{REGISTERED_MODEL}/{version}", dst_path=download)
            meta = serving_meta(mlflow.models.Model.load(path).metadata)
            model = keras.models.load_model(f"{path}/data/model.keras")

            # Write next to the target (same filesystem, so the rename is atomic), then swap it in
            tmp = target.parent / f".tmp-{version}"
            shutil.rmtree(tmp, ignore_errors=True)
            print("exporting SavedModel...")
            export(model, meta["image_size"], tmp)
            write_warmup(tmp, meta["image_size"])
            (tmp / "meta.json").write_text(json.dumps(meta, indent=2))
            if target.exists():
                shutil.rmtree(target)
            tmp.rename(target)
        print(f"v{version} exported to {target}")

    if not args.no_activate:
        activate(version)


if __name__ == "__main__":
    main()
