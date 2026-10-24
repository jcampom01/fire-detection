"""TensorFlow import and GPU setup.

Every module that needs TensorFlow or Keras imports them from here (`from tf_setup import keras, tf`),
so the CUDA libraries are always preloaded before TensorFlow is imported for the first time.
"""

import ctypes
import glob
import os
import site

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def preload_nvidia_libs():
    """Load the pip-installed CUDA libraries before TensorFlow looks for them.

    On WSL TensorFlow does not always find some of them (e.g. libcusolver) and silently falls back
    to the CPU. Loading them first with RTLD_GLOBAL makes them visible. Two passes because some
    libraries depend on others that come later in alphabetical order.
    """
    libs = sorted(so for d in site.getsitepackages() for so in glob.glob(f"{d}/nvidia/*/lib/lib*.so*"))
    for _ in range(2):
        for so in libs:
            try:
                ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


preload_nvidia_libs()
import keras  # noqa: E402
import tensorflow as tf  # noqa: E402


def setup_runtime(seed, mixed_precision):
    """Fix the random seeds, set the precision policy and let GPU memory grow on demand.

    Returns the list of visible GPUs.
    """
    keras.utils.set_random_seed(seed)
    if mixed_precision:
        keras.mixed_precision.set_global_policy("mixed_float16")
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    return gpus
