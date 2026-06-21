from __future__ import annotations

from typing import Literal

import cv2
import numpy as np


PreprocessMode = Literal["auto", "clahe", "denoise", "none"]

LOW_LIGHT_THRESHOLD = 80.0


def _validate_frame(frame: np.ndarray) -> None:
    if frame is None:
        raise ValueError("frame must not be None")
    if not isinstance(frame, np.ndarray):
        raise TypeError("frame must be a numpy.ndarray")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("frame must be a BGR image with shape (height, width, 3)")
    if frame.size == 0:
        raise ValueError("frame must not be empty")


def enhance_low_light_clahe(frame: np.ndarray) -> np.ndarray:
    """
    Enhance low-light BGR frames using CLAHE on the LAB luminance channel.
    """
    _validate_frame(frame)

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab_frame)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l_channel = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def reduce_blur_and_noise(frame: np.ndarray) -> np.ndarray:
    """
    Apply edge-preserving bilateral denoising to reduce rain streaks and mild blur.
    """
    _validate_frame(frame)
    return cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)


def preprocess_frame(frame: np.ndarray, mode: PreprocessMode = "auto") -> np.ndarray:
    """
    Preprocess a BGR frame for traffic violation detection.

    In auto mode, low-light enhancement is applied when average brightness is below
    LOW_LIGHT_THRESHOLD. Edge-preserving denoising is always applied in auto mode.
    """
    _validate_frame(frame)

    if mode not in {"auto", "clahe", "denoise", "none"}:
        raise ValueError("mode must be one of: 'auto', 'clahe', 'denoise', 'none'")

    processed_frame = frame.copy()

    if mode == "none":
        return processed_frame

    if mode == "clahe":
        return enhance_low_light_clahe(processed_frame)

    if mode == "denoise":
        return reduce_blur_and_noise(processed_frame)

    average_brightness = float(np.mean(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)))
    if average_brightness < LOW_LIGHT_THRESHOLD:
        processed_frame = enhance_low_light_clahe(processed_frame)

    return reduce_blur_and_noise(processed_frame)
