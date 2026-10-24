"""Model: pretrained backbone + binary head, freezing/unfreezing and compilation."""

from tf_setup import keras

# Backbones from keras.applications. All of them include their own input preprocessing,
# so they all take raw RGB images in the 0-255 range.
BACKBONES = {
    "mobilenetv3large": lambda shape: keras.applications.MobileNetV3Large(
        input_shape=shape, include_top=False, weights="imagenet", include_preprocessing=True),
    "efficientnetb0": lambda shape: keras.applications.EfficientNetB0(
        input_shape=shape, include_top=False, weights="imagenet"),
    "efficientnetb4": lambda shape: keras.applications.EfficientNetB4(
        input_shape=shape, include_top=False, weights="imagenet"),
    "convnexttiny": lambda shape: keras.applications.ConvNeXtTiny(
        input_shape=shape, include_top=False, weights="imagenet", include_preprocessing=True),
}


def build_model(cfg):
    """Return (model, backbone). The backbone starts frozen."""
    if cfg["model"] not in BACKBONES:
        raise ValueError(f"unknown model {cfg['model']!r}, expected one of {list(BACKBONES)}")
    shape = (cfg["image_size"], cfg["image_size"], 3)
    backbone = BACKBONES[cfg["model"]](shape)
    backbone.trainable = False

    inputs = keras.Input(shape)
    # training=False keeps BatchNorm in inference mode, also once the backbone is unfrozen
    x = backbone(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(cfg["head"]["dropout"])(x)
    outputs = keras.layers.Dense(1, activation="sigmoid", dtype="float32", name="alarm")(x)
    return keras.Model(inputs, outputs, name=cfg["experiment"]), backbone


def unfreeze_top(backbone, fraction):
    """Unfreeze the last `fraction` of the backbone layers, keeping BatchNorm frozen."""
    backbone.trainable = True
    cut = int(len(backbone.layers) * (1 - fraction))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cut and not isinstance(layer, keras.layers.BatchNormalization)
    return sum(layer.trainable for layer in backbone.layers)


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc"),
                 keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
