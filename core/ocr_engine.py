from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO


LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PLATE_MODEL_PATH = PROJECT_ROOT / "lp_best.pt"
UNKNOWN_PLATE_TEXT = "MANUAL_REVIEW"
OCR_CONFIDENCE_THRESHOLD = 0.20
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Standard Indian vehicle registration number pattern:
# Format: <State-Code:2 letters> <District-Code:2 digits> <Series:1-2 letters> <Number:4 digits>
# Examples: MH12AB1234, DL3CAA1111, KA01MX9999
# Spaces and hyphens between groups are optional.
_INDIAN_PLATE_REGEX = re.compile(
    r"^[A-Z]{2}[ -]?[0-9]{2}[ -]?[A-Z]{1,2}[ -]?[0-9]{4}$"
)

_plate_models: dict[Path, YOLO] = {}
_ocr_reader: easyocr.Reader | None = None


@dataclass(frozen=True)
class OCRResult:
    text: str
    confidence: float
    plate_detected: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": round(float(self.confidence), 4),
            "plate_detected": self.plate_detected,
        }


def _validate_image(image: np.ndarray, image_name: str) -> None:
    if image is None:
        raise ValueError(f"{image_name} must not be None")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{image_name} must be a numpy.ndarray")
    if image.ndim not in {2, 3}:
        raise ValueError(f"{image_name} must be a grayscale or BGR image")
    if image.size == 0:
        raise ValueError(f"{image_name} must not be empty")


def _resolve_model_path(model_path: str | Path) -> Path:
    weights = Path(model_path)
    if weights.exists():
        return weights

    root_fallback = PROJECT_ROOT / weights.name
    if root_fallback.exists():
        return root_fallback

    return weights


def _load_plate_model(model_path: str | Path = DEFAULT_PLATE_MODEL_PATH) -> YOLO:
    weights = _resolve_model_path(model_path)
    if not weights.exists():
        raise FileNotFoundError(
            f"License plate detector weights not found: {weights}. "
            "Train the plate detector and save it as lp_best.pt, or pass a valid model path."
        )

    resolved_weights = weights.resolve()
    if resolved_weights not in _plate_models:
        _plate_models[resolved_weights] = YOLO(str(resolved_weights))
    return _plate_models[resolved_weights]


def _load_ocr_reader() -> easyocr.Reader:
    global _ocr_reader

    if _ocr_reader is None:
        _ocr_reader = easyocr.Reader(["en"], gpu=True)
    return _ocr_reader


def _clip_bbox(bbox: np.ndarray, image_shape: tuple[int, ...], padding_ratio: float = 0.03) -> tuple[int, int, int, int]:
    height, width = image_shape[:2]
    x1, y1, x2, y2 = bbox.astype(float)

    box_width = max(0.0, x2 - x1)
    box_height = max(0.0, y2 - y1)
    pad_x = box_width * padding_ratio
    pad_y = box_height * padding_ratio

    x1 = max(0, min(int(round(x1 - pad_x)), width - 1))
    y1 = max(0, min(int(round(y1 - pad_y)), height - 1))
    x2 = max(0, min(int(round(x2 + pad_x)), width - 1))
    y2 = max(0, min(int(round(y2 + pad_y)), height - 1))

    return x1, y1, x2, y2


def extract_license_plate(
    vehicle_crop: np.ndarray,
    model_path: str | Path = DEFAULT_PLATE_MODEL_PATH,
    confidence_threshold: float = 0.25,
    image_size: int = 640,
) -> np.ndarray | None:
    """
    Localize and crop the license plate from a violating vehicle crop.
    """
    _validate_image(vehicle_crop, "vehicle_crop")

    model = _load_plate_model(model_path)
    results = model.predict(vehicle_crop, conf=confidence_threshold, imgsz=image_size, verbose=False)

    best_box: np.ndarray | None = None
    best_confidence = -1.0
    best_area = -1.0

    for result in results:
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        confidences = boxes.conf.cpu().numpy()

        for bbox, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = bbox
            area = max(0.0, float(x2 - x1)) * max(0.0, float(y2 - y1))

            if float(confidence) > best_confidence or (
                float(confidence) == best_confidence and area > best_area
            ):
                best_box = bbox
                best_confidence = float(confidence)
                best_area = area

    if best_box is None:
        LOGGER.info("No license plate detected in vehicle crop.")
        return None

    x1, y1, x2, y2 = _clip_bbox(best_box, vehicle_crop.shape)
    if x2 <= x1 or y2 <= y1:
        LOGGER.warning("License plate detector returned an invalid bounding box.")
        return None

    plate_crop = vehicle_crop[y1:y2, x1:x2]
    return plate_crop if plate_crop.size > 0 else None


def preprocess_plate_for_ocr(plate_crop: np.ndarray) -> np.ndarray:
    """
    Convert a plate crop to a high-contrast binary image for OCR.
    """
    _validate_image(plate_crop, "plate_crop")

    upscaled = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    if upscaled.ndim == 3:
        grayscale = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    else:
        grayscale = upscaled.copy()

    blurred = cv2.GaussianBlur(grayscale, (5, 5), 0)
    min_dimension = min(blurred.shape[:2])
    block_size = max(3, min(31, min_dimension // 2))
    if block_size % 2 == 0:
        block_size += 1

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        7,
    )

    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return binary


def _ocr_variants(plate_crop: np.ndarray) -> list[np.ndarray]:
    _validate_image(plate_crop, "plate_crop")

    upscaled = cv2.resize(plate_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    grayscale = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY) if upscaled.ndim == 3 else upscaled.copy()
    denoised = cv2.bilateralFilter(grayscale, d=5, sigmaColor=35, sigmaSpace=35)
    adaptive = preprocess_plate_for_ocr(plate_crop)
    inverted_adaptive = cv2.bitwise_not(adaptive)

    return [
        upscaled,
        grayscale,
        denoised,
        adaptive,
        inverted_adaptive,
    ]


def _clean_ocr_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def _sort_ocr_results(results: list[tuple[Any, str, float]]) -> list[tuple[Any, str, float]]:
    def sort_key(item: tuple[Any, str, float]) -> tuple[float, float]:
        bbox = np.asarray(item[0], dtype=np.float32)
        return float(np.mean(bbox[:, 1])), float(np.mean(bbox[:, 0]))

    return sorted(results, key=sort_key)


def recognize_plate_text(
    plate_crop: np.ndarray,
    confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
) -> tuple[str, float]:
    """
    Read alphanumeric registration text from a localized plate crop.
    """
    _validate_image(plate_crop, "plate_crop")

    try:
        reader = _load_ocr_reader()
    except Exception as exc:
        LOGGER.warning("EasyOCR reader initialization failed: %s", exc)
        return UNKNOWN_PLATE_TEXT, 0.0

    best_text = ""
    best_confidence = 0.0

    for candidate in _ocr_variants(plate_crop):
        try:
            ocr_results = reader.readtext(
                candidate,
                detail=1,
                paragraph=False,
                allowlist=OCR_ALLOWLIST,
                min_size=5,
                text_threshold=0.2,
                low_text=0.2,
                link_threshold=0.2,
                mag_ratio=1.5,
            )
        except Exception as exc:
            LOGGER.warning("EasyOCR failed during license plate recognition: %s", exc)
            continue

        cleaned_parts: list[tuple[str, float]] = []
        for _, raw_text, confidence in _sort_ocr_results(ocr_results):
            cleaned_text = _clean_ocr_text(raw_text)
            if cleaned_text:
                cleaned_parts.append((cleaned_text, float(confidence)))

        if not cleaned_parts:
            continue

        registration_text = "".join(text for text, _ in cleaned_parts)
        total_characters = sum(len(text) for text, _ in cleaned_parts)
        weighted_confidence = sum(len(text) * confidence for text, confidence in cleaned_parts) / max(total_characters, 1)

        # Prefer results that match the Indian plate format; otherwise fall back to best length/confidence.
        is_valid_plate = bool(_INDIAN_PLATE_REGEX.match(registration_text))
        current_is_valid = bool(_INDIAN_PLATE_REGEX.match(best_text))

        if is_valid_plate and not current_is_valid:
            # Upgrade: any valid-format result beats a non-format result regardless of confidence.
            best_text = registration_text
            best_confidence = float(weighted_confidence)
        elif is_valid_plate == current_is_valid:
            # Both on the same tier — pick the higher confidence (or longer on tie).
            if weighted_confidence > best_confidence or (
                weighted_confidence == best_confidence and len(registration_text) > len(best_text)
            ):
                best_text = registration_text
                best_confidence = float(weighted_confidence)
        # else: current is valid and candidate is not — keep current.

    if not best_text or len(best_text) < 4:
        return UNKNOWN_PLATE_TEXT, best_confidence

    if best_confidence < confidence_threshold:
        return UNKNOWN_PLATE_TEXT, best_confidence

    # Final format validation: if the winning text doesn't match the Indian plate pattern,
    # still return it (it may be a foreign/commercial plate), but log a warning.
    if not _INDIAN_PLATE_REGEX.match(best_text):
        LOGGER.debug(
            "OCR result '%s' does not match Indian plate format — returning as-is.",
            best_text,
        )

    return best_text, best_confidence


def read_vehicle_license_plate(
    vehicle_crop: np.ndarray,
    model_path: str | Path = DEFAULT_PLATE_MODEL_PATH,
    plate_confidence_threshold: float = 0.25,
    ocr_confidence_threshold: float = OCR_CONFIDENCE_THRESHOLD,
) -> OCRResult:
    """
    End-to-end helper: localize a plate from a vehicle crop and OCR the registration.
    """
    plate_crop = extract_license_plate(
        vehicle_crop=vehicle_crop,
        model_path=model_path,
        confidence_threshold=plate_confidence_threshold,
    )

    if plate_crop is None:
        text, confidence = recognize_plate_text(vehicle_crop, confidence_threshold=ocr_confidence_threshold)
        return OCRResult(text=text, confidence=confidence, plate_detected=False)

    text, confidence = recognize_plate_text(plate_crop, confidence_threshold=ocr_confidence_threshold)
    if text == UNKNOWN_PLATE_TEXT:
        fallback_text, fallback_confidence = recognize_plate_text(
            vehicle_crop,
            confidence_threshold=ocr_confidence_threshold,
        )
        if fallback_text != UNKNOWN_PLATE_TEXT and fallback_confidence >= confidence:
            return OCRResult(text=fallback_text, confidence=fallback_confidence, plate_detected=False)

    return OCRResult(text=text, confidence=confidence, plate_detected=True)
