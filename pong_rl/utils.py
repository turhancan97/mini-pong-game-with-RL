"""Utility helpers for reproducibility and shared concerns."""
from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf


def set_random_seeds(seed: int, enable_tf_determinism: bool = True) -> None:
    """Seed all relevant RNGs to promote reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    if enable_tf_determinism:
        try:
            tf.config.experimental.enable_op_determinism()
        except AttributeError:
            pass
