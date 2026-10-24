"""Two-phase training: frozen backbone (phase 1), then fine-tuning of its last layers (phase 2).

The best weights over both phases, by the monitored validation metric, are saved to
<output_dir>/best.weights.h5 and loaded back into the model at the end.
"""

import numpy as np
import pandas as pd

from model import compile_model, unfreeze_top
from tf_setup import keras


def run_phase(name, model, train_ds, val_ds, phase_cfg, cfg, best_value, extra_callbacks=()):
    """Fit one phase. Only saves weights that beat `best_value` (the best of the previous phases)."""
    monitor = cfg["monitor"]
    callbacks = [
        keras.callbacks.ModelCheckpoint(cfg["output_dir"] / "best.weights.h5", monitor=monitor, mode="max",
                                        save_best_only=True, save_weights_only=True,
                                        initial_value_threshold=best_value),
        keras.callbacks.EarlyStopping(monitor=monitor, mode="max", patience=phase_cfg["patience"]),
        *extra_callbacks,
    ]
    print(f"\n=== {name}: up to {phase_cfg['epochs']} epochs, lr={phase_cfg['lr']} ===")
    hist = model.fit(train_ds, validation_data=val_ds, epochs=phase_cfg["epochs"], callbacks=callbacks, verbose=2)
    df = pd.DataFrame(hist.history)
    df.insert(0, "phase", name)
    best = df[monitor].max()
    return df, best if best_value is None else max(best, best_value)


def train(cfg, model, backbone, train_ds, val_ds):
    """Run both phases and return the per-epoch history (columns: phase, epoch, metrics...)."""
    best_weights = cfg["output_dir"] / "best.weights.h5"

    compile_model(model, cfg["phase1"]["lr"])
    h1, best = run_phase("phase1", model, train_ds, val_ds, cfg["phase1"], cfg, None)

    p2 = cfg["phase2"]
    model.load_weights(best_weights)
    n = unfreeze_top(backbone, p2["unfreeze_fraction"])
    print(f"\nphase2: {n}/{len(backbone.layers)} backbone layers trainable")
    compile_model(model, p2["lr"])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor=cfg["monitor"], mode="max",
                                                  factor=p2["reduce_lr_factor"],
                                                  patience=p2["reduce_lr_patience"], min_lr=1e-7)
    h2, _ = run_phase("phase2", model, train_ds, val_ds, p2, cfg, best, [reduce_lr])

    model.load_weights(best_weights)
    history = pd.concat([h1, h2], ignore_index=True)
    history.insert(1, "epoch", np.arange(1, len(history) + 1))
    return history
