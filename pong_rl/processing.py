"""Utilities for processing game frames."""
from __future__ import annotations

import numpy as np
import skimage.color
import skimage.exposure
import skimage.transform

IMG_HEIGHT = 40
IMG_WIDTH = 40
IMG_HISTORY = 4


def process_frame(raw_frame: np.ndarray) -> np.ndarray:
    """Convert a raw RGB frame into a normalized grayscale image."""
    grey_image = skimage.color.rgb2gray(raw_frame)
    cropped_image = grey_image[0:400, 0:400]
    reduced_image = skimage.transform.resize(
        cropped_image, (IMG_HEIGHT, IMG_WIDTH), anti_aliasing=True
    )
    reduced_image = skimage.exposure.rescale_intensity(reduced_image, out_range=(0, 255))
    reduced_image = reduced_image / 128.0
    return reduced_image
